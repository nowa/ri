use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    File,
    Directory,
    Symlink,
    Other,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FileInfo {
    pub name: String,
    pub path: String,
    pub kind: FileKind,
    pub size: u64,
    pub mtime_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileErrorCode {
    NotFound,
    NotDirectory,
    Io,
    ShellUnavailable,
    Spawn,
    Timeout,
    CallbackError,
    Aborted,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
#[error("{message}")]
pub struct FileError {
    pub code: FileErrorCode,
    pub message: String,
    pub path: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreateDirOptions {
    pub recursive: bool,
}

impl Default for CreateDirOptions {
    fn default() -> Self {
        Self { recursive: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoveOptions {
    pub recursive: bool,
    pub force: bool,
}

impl Default for RemoveOptions {
    fn default() -> Self {
        Self {
            recursive: false,
            force: true,
        }
    }
}

pub type ExecCallback = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static>;

#[derive(Clone, Default)]
pub struct ExecOptions {
    pub env: Vec<(String, String)>,
    pub timeout_ms: Option<u64>,
    pub abort_flag: Option<Arc<AtomicBool>>,
    pub on_stdout: Option<ExecCallback>,
    pub on_stderr: Option<ExecCallback>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

#[derive(Debug, Clone)]
pub struct LocalExecutionEnv {
    cwd: PathBuf,
    shell_path: PathBuf,
}

impl LocalExecutionEnv {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            shell_path: PathBuf::from(
                std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_owned()),
            ),
        }
    }

    pub fn with_shell(mut self, shell_path: impl Into<PathBuf>) -> Self {
        self.shell_path = shell_path.into();
        self
    }

    pub fn absolute_path(&self, path: impl AsRef<Path>) -> String {
        display_path(&self.resolve(path))
    }

    pub fn join_path<I, P>(&self, parts: I) -> String
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut path = PathBuf::new();
        for part in parts {
            path.push(part);
        }
        display_path(&path)
    }

    pub fn create_dir(
        &self,
        path: impl AsRef<Path>,
        options: CreateDirOptions,
    ) -> Result<(), FileError> {
        let path = self.resolve(path);
        let result = if options.recursive {
            fs::create_dir_all(&path)
        } else {
            fs::create_dir(&path)
        };
        result.map_err(|error| file_error_from_io(error, &path))
    }

    pub fn write_file(&self, path: impl AsRef<Path>, content: &[u8]) -> Result<(), FileError> {
        let path = self.resolve(path);
        self.write_file_resolved(&path, content, None)
    }

    pub fn write_file_with_abort(
        &self,
        path: impl AsRef<Path>,
        content: &[u8],
        abort_flag: Option<&AtomicBool>,
    ) -> Result<(), FileError> {
        let path = self.resolve(path);
        self.write_file_resolved(&path, content, abort_flag)
    }

    fn write_file_resolved(
        &self,
        path: &Path,
        content: &[u8],
        abort_flag: Option<&AtomicBool>,
    ) -> Result<(), FileError> {
        check_file_abort(abort_flag, path)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| file_error_from_io(error, parent))?;
        }
        fs::write(path, content).map_err(|error| file_error_from_io(error, path))
    }

    pub fn append_file(&self, path: impl AsRef<Path>, content: &[u8]) -> Result<(), FileError> {
        let path = self.resolve(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| file_error_from_io(error, parent))?;
        }
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|error| file_error_from_io(error, &path))?;
        file.write_all(content)
            .map_err(|error| file_error_from_io(error, &path))
    }

    pub fn read_text_file(&self, path: impl AsRef<Path>) -> Result<String, FileError> {
        let path = self.resolve(path);
        self.read_text_file_resolved(&path, None)
    }

    pub fn read_text_file_with_abort(
        &self,
        path: impl AsRef<Path>,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<String, FileError> {
        let path = self.resolve(path);
        self.read_text_file_resolved(&path, abort_flag)
    }

    fn read_text_file_resolved(
        &self,
        path: &Path,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<String, FileError> {
        check_file_abort(abort_flag, path)?;
        fs::read_to_string(path).map_err(|error| file_error_from_io(error, path))
    }

    pub fn read_text_lines(
        &self,
        path: impl AsRef<Path>,
        max_lines: usize,
    ) -> Result<Vec<String>, FileError> {
        self.read_text_lines_with_abort(path, max_lines, None)
    }

    pub fn read_text_lines_with_abort(
        &self,
        path: impl AsRef<Path>,
        max_lines: usize,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<Vec<String>, FileError> {
        Ok(self
            .read_text_file_with_abort(path, abort_flag)?
            .lines()
            .take(max_lines)
            .map(ToOwned::to_owned)
            .collect())
    }

    pub fn read_binary_file(&self, path: impl AsRef<Path>) -> Result<Vec<u8>, FileError> {
        let path = self.resolve(path);
        self.read_binary_file_resolved(&path, None)
    }

    pub fn read_binary_file_with_abort(
        &self,
        path: impl AsRef<Path>,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<Vec<u8>, FileError> {
        let path = self.resolve(path);
        self.read_binary_file_resolved(&path, abort_flag)
    }

    fn read_binary_file_resolved(
        &self,
        path: &Path,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<Vec<u8>, FileError> {
        check_file_abort(abort_flag, path)?;
        fs::read(path).map_err(|error| file_error_from_io(error, path))
    }

    pub fn list_dir(&self, path: impl AsRef<Path>) -> Result<Vec<FileInfo>, FileError> {
        let path = self.resolve(path);
        self.list_dir_resolved(&path, None)
    }

    pub fn list_dir_with_abort(
        &self,
        path: impl AsRef<Path>,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<Vec<FileInfo>, FileError> {
        let path = self.resolve(path);
        self.list_dir_resolved(&path, abort_flag)
    }

    fn list_dir_resolved(
        &self,
        path: &Path,
        abort_flag: Option<&AtomicBool>,
    ) -> Result<Vec<FileInfo>, FileError> {
        check_file_abort(abort_flag, path)?;
        if !path.is_dir() {
            return Err(FileError {
                code: if path.exists() {
                    FileErrorCode::NotDirectory
                } else {
                    FileErrorCode::NotFound
                },
                message: format!("Not a directory: {}", display_path(&path)),
                path: display_path(&path),
            });
        }
        let mut entries = fs::read_dir(&path)
            .map_err(|error| file_error_from_io(error, &path))?
            .map(|entry| {
                let entry = entry.map_err(|error| file_error_from_io(error, &path))?;
                self.file_info(entry.path())
            })
            .collect::<Result<Vec<_>, _>>()?;
        entries.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(entries)
    }

    pub fn file_info(&self, path: impl AsRef<Path>) -> Result<FileInfo, FileError> {
        let path = self.resolve(path);
        let metadata =
            fs::symlink_metadata(&path).map_err(|error| file_error_from_io(error, &path))?;
        Ok(FileInfo {
            name: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned(),
            path: display_path(&path),
            kind: file_kind(&metadata),
            size: metadata.len(),
            mtime_ms: metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis())
                .unwrap_or_default(),
        })
    }

    pub fn canonical_path(&self, path: impl AsRef<Path>) -> Result<String, FileError> {
        let path = self.resolve(path);
        fs::canonicalize(&path)
            .map(|path| display_path(&path))
            .map_err(|error| file_error_from_io(error, &path))
    }

    pub fn exists(&self, path: impl AsRef<Path>) -> bool {
        self.resolve(path).exists()
    }

    pub fn remove(&self, path: impl AsRef<Path>, options: RemoveOptions) -> Result<(), FileError> {
        let path = self.resolve(path);
        if !path.exists() {
            return if options.force {
                Ok(())
            } else {
                Err(file_error(FileErrorCode::NotFound, "Path not found", &path))
            };
        }
        if path.is_dir() && !path.is_symlink() {
            if options.recursive {
                fs::remove_dir_all(&path)
            } else {
                fs::remove_dir(&path)
            }
        } else {
            fs::remove_file(&path)
        }
        .map_err(|error| file_error_from_io(error, &path))
    }

    pub fn create_temp_dir(&self, prefix: &str) -> Result<String, FileError> {
        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!("{prefix}{}", unique_suffix()));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(display_path(&path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(file_error_from_io(error, &path)),
            }
        }
        Err(file_error(
            FileErrorCode::Io,
            "Unable to create a unique temp directory",
            &std::env::temp_dir(),
        ))
    }

    pub fn create_temp_file(&self, prefix: &str, suffix: &str) -> Result<String, FileError> {
        for _ in 0..100 {
            let path = std::env::temp_dir().join(format!("{prefix}{}{suffix}", unique_suffix()));
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(_) => return Ok(display_path(&path)),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(file_error_from_io(error, &path)),
            }
        }
        Err(file_error(
            FileErrorCode::Io,
            "Unable to create a unique temp file",
            &std::env::temp_dir(),
        ))
    }

    pub fn exec(&self, command: &str, options: ExecOptions) -> Result<ExecOutput, FileError> {
        let ExecOptions {
            env,
            timeout_ms,
            abort_flag,
            on_stdout,
            on_stderr,
        } = options;
        if abort_flag
            .as_ref()
            .is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst))
        {
            return Err(file_error(
                FileErrorCode::Aborted,
                "Command aborted",
                &self.shell_path,
            ));
        }
        if !self.shell_path.exists() {
            return Err(file_error(
                FileErrorCode::ShellUnavailable,
                "Shell is unavailable",
                &self.shell_path,
            ));
        }
        let mut child = Command::new(&self.shell_path)
            .arg("-lc")
            .arg(command)
            .current_dir(&self.cwd)
            .envs(env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|error| FileError {
                code: FileErrorCode::Spawn,
                message: error.to_string(),
                path: display_path(&self.shell_path),
            })?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_shell_path = self.shell_path.clone();
        let stderr_shell_path = self.shell_path.clone();
        let stdout_reader =
            std::thread::spawn(move || read_pipe(stdout, on_stdout, stdout_shell_path));
        let stderr_reader =
            std::thread::spawn(move || read_pipe(stderr, on_stderr, stderr_shell_path));

        let started = Instant::now();
        let timeout = timeout_ms.map(Duration::from_millis);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if abort_flag
                        .as_ref()
                        .is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst))
                    {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = stdout_reader.join();
                        let _ = stderr_reader.join();
                        return Err(file_error(
                            FileErrorCode::Aborted,
                            "Command aborted",
                            &self.shell_path,
                        ));
                    }
                    if let Some(timeout) = timeout
                        && started.elapsed() >= timeout
                    {
                        let _ = child.kill();
                        let _ = child.wait();
                        let _ = stdout_reader.join();
                        let _ = stderr_reader.join();
                        return Err(file_error(
                            FileErrorCode::Timeout,
                            "Command timed out",
                            &self.shell_path,
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = stdout_reader.join();
                    let _ = stderr_reader.join();
                    return Err(FileError {
                        code: FileErrorCode::Spawn,
                        message: error.to_string(),
                        path: display_path(&self.shell_path),
                    });
                }
            }
        };

        let stdout = stdout_reader.join().map_err(|_| {
            file_error(
                FileErrorCode::Io,
                "Failed to join stdout reader",
                &self.shell_path,
            )
        })??;
        let stderr = stderr_reader.join().map_err(|_| {
            file_error(
                FileErrorCode::Io,
                "Failed to join stderr reader",
                &self.shell_path,
            )
        })??;

        Ok(ExecOutput {
            stdout: String::from_utf8_lossy(&stdout).to_string(),
            stderr: String::from_utf8_lossy(&stderr).to_string(),
            exit_code: status.code().unwrap_or(-1),
        })
    }

    pub fn cleanup(&self) {}

    fn resolve(&self, path: impl AsRef<Path>) -> PathBuf {
        let path = path.as_ref();
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }
}

fn read_pipe(
    pipe: Option<impl Read>,
    callback: Option<ExecCallback>,
    shell_path: PathBuf,
) -> Result<Vec<u8>, FileError> {
    let mut output = Vec::new();
    if let Some(mut pipe) = pipe {
        let mut buffer = [0_u8; 8192];
        loop {
            let bytes_read = pipe
                .read(&mut buffer)
                .map_err(|error| file_error_from_io(error, &shell_path))?;
            if bytes_read == 0 {
                break;
            }
            output.extend_from_slice(&buffer[..bytes_read]);
            if let Some(callback) = &callback {
                let chunk = String::from_utf8_lossy(&buffer[..bytes_read]).to_string();
                callback(&chunk).map_err(|message| {
                    file_error(FileErrorCode::CallbackError, &message, &shell_path)
                })?;
            }
        }
    }
    Ok(output)
}

fn check_file_abort(abort_flag: Option<&AtomicBool>, path: &Path) -> Result<(), FileError> {
    if abort_flag.is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst)) {
        Err(file_error(
            FileErrorCode::Aborted,
            "Operation aborted",
            path,
        ))
    } else {
        Ok(())
    }
}

fn file_kind(metadata: &fs::Metadata) -> FileKind {
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        FileKind::Symlink
    } else if file_type.is_file() {
        FileKind::File
    } else if file_type.is_dir() {
        FileKind::Directory
    } else {
        FileKind::Other
    }
}

fn file_error_from_io(error: std::io::Error, path: &Path) -> FileError {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => FileErrorCode::NotFound,
        _ => FileErrorCode::Io,
    };
    FileError {
        code,
        message: error.to_string(),
        path: display_path(path),
    }
}

fn file_error(code: FileErrorCode, message: &str, path: &Path) -> FileError {
    FileError {
        code,
        message: message.to_owned(),
        path: display_path(path),
    }
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{nanos}-{}", std::process::id())
}
