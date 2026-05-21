use std::{
    fs,
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

#[cfg(unix)]
unsafe extern "C" {
    fn kill(pid: i32, sig: i32) -> i32;
}

#[cfg(unix)]
const SIGKILL: i32 = 9;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    File,
    Directory,
    Symlink,
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
    PermissionDenied,
    NotDirectory,
    IsDirectory,
    Invalid,
    NotSupported,
    Io,
    Unknown,
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
            force: false,
        }
    }
}

pub type ExecCallback = Arc<dyn Fn(&str) -> Result<(), String> + Send + Sync + 'static>;

#[derive(Clone, Default)]
pub struct ExecOptions {
    pub cwd: Option<PathBuf>,
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
    shell_env: Vec<(String, String)>,
}

impl LocalExecutionEnv {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            shell_path: default_shell_path(),
            shell_env: Vec::new(),
        }
    }

    pub fn with_shell(mut self, shell_path: impl Into<PathBuf>) -> Self {
        self.shell_path = shell_path.into();
        self
    }

    pub fn with_shell_env(mut self, shell_env: Vec<(String, String)>) -> Self {
        self.shell_env = shell_env;
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
        check_file_abort(abort_flag, path)?;
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
        let path = self.resolve(path);
        check_file_abort(abort_flag, &path)?;
        if max_lines == 0 {
            return Ok(Vec::new());
        }
        let file = fs::File::open(&path).map_err(|error| file_error_from_io(error, &path))?;
        let mut lines = Vec::new();
        for line in BufReader::new(file).lines() {
            check_file_abort(abort_flag, &path)?;
            lines.push(line.map_err(|error| file_error_from_io(error, &path))?);
            if lines.len() >= max_lines {
                break;
            }
        }
        check_file_abort(abort_flag, &path)?;
        Ok(lines)
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
                check_file_abort(abort_flag, &path)?;
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
        let kind = file_kind(&metadata).ok_or_else(|| {
            file_error(FileErrorCode::NotSupported, "Unsupported file type", &path)
        })?;
        Ok(FileInfo {
            name: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .to_owned(),
            path: display_path(&path),
            kind,
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
            cwd,
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
        if !shell_path_is_available(&self.shell_path) {
            return Err(file_error(
                FileErrorCode::ShellUnavailable,
                "Shell is unavailable",
                &self.shell_path,
            ));
        }
        let cwd = cwd
            .map(|path| self.resolve(path))
            .unwrap_or_else(|| self.cwd.clone());
        let mut command_builder = Command::new(&self.shell_path);
        command_builder
            .arg("-lc")
            .arg(command)
            .current_dir(cwd)
            .envs(self.shell_env.iter().cloned())
            .envs(env)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command_builder.process_group(0);
        let mut child = command_builder.spawn().map_err(|error| FileError {
            code: FileErrorCode::Spawn,
            message: error.to_string(),
            path: display_path(&self.shell_path),
        })?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let stdout_shell_path = self.shell_path.clone();
        let stderr_shell_path = self.shell_path.clone();
        let callback_error = Arc::new(Mutex::new(None));
        let stdout_callback_error = callback_error.clone();
        let stderr_callback_error = callback_error.clone();
        let stdout_reader = std::thread::spawn(move || {
            read_pipe(stdout, on_stdout, stdout_shell_path, stdout_callback_error)
        });
        let stderr_reader = std::thread::spawn(move || {
            read_pipe(stderr, on_stderr, stderr_shell_path, stderr_callback_error)
        });

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
                        kill_process_tree(&mut child);
                        let _ = child.wait();
                        let _ = stdout_reader.join();
                        let _ = stderr_reader.join();
                        return Err(file_error(
                            FileErrorCode::Aborted,
                            "Command aborted",
                            &self.shell_path,
                        ));
                    }
                    if let Some(message) = callback_error.lock().expect("callback lock").clone() {
                        kill_process_tree(&mut child);
                        let _ = child.wait();
                        let _ = stdout_reader.join();
                        let _ = stderr_reader.join();
                        return Err(file_error(
                            FileErrorCode::CallbackError,
                            &message,
                            &self.shell_path,
                        ));
                    }
                    if let Some(timeout) = timeout
                        && started.elapsed() >= timeout
                    {
                        kill_process_tree(&mut child);
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
                    kill_process_tree(&mut child);
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
    callback_error: Arc<Mutex<Option<String>>>,
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
                    *callback_error.lock().expect("callback lock") = Some(message.clone());
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

fn file_kind(metadata: &fs::Metadata) -> Option<FileKind> {
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        Some(FileKind::Symlink)
    } else if file_type.is_file() {
        Some(FileKind::File)
    } else if file_type.is_dir() {
        Some(FileKind::Directory)
    } else {
        None
    }
}

fn file_error_from_io(error: std::io::Error, path: &Path) -> FileError {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => FileErrorCode::NotFound,
        std::io::ErrorKind::PermissionDenied => FileErrorCode::PermissionDenied,
        std::io::ErrorKind::NotADirectory => FileErrorCode::NotDirectory,
        std::io::ErrorKind::IsADirectory => FileErrorCode::IsDirectory,
        std::io::ErrorKind::InvalidInput | std::io::ErrorKind::InvalidData => {
            FileErrorCode::Invalid
        }
        std::io::ErrorKind::Unsupported => FileErrorCode::NotSupported,
        _ => FileErrorCode::Unknown,
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

fn default_shell_path() -> PathBuf {
    if Path::new("/bin/bash").exists() {
        return PathBuf::from("/bin/bash");
    }
    if let Some(path) = std::env::var_os("PATH").and_then(find_bash_on_path) {
        return path;
    }
    PathBuf::from("sh")
}

fn find_bash_on_path(path_var: std::ffi::OsString) -> Option<PathBuf> {
    std::env::split_paths(&path_var)
        .map(|path| path.join(if cfg!(windows) { "bash.exe" } else { "bash" }))
        .find(|path| path.exists())
}

fn shell_path_is_available(path: &Path) -> bool {
    path.exists() || (!path.is_absolute() && path.components().count() == 1)
}

fn kill_process_tree(child: &mut Child) {
    #[cfg(unix)]
    {
        let pid = child.id() as i32;
        // The child is started in a fresh process group so background grandchildren
        // follow the shell timeout/abort behavior instead of leaking after the shell dies.
        unsafe {
            let _ = kill(-pid, SIGKILL);
        }
    }
    let _ = child.kill();
}

fn unique_suffix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("{nanos}-{}", std::process::id())
}
