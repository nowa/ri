use super::{
    DEFAULT_MAX_BYTES, ExecCallback, ExecOptions, FileError, FileErrorCode, LocalExecutionEnv,
    TruncationOptions, truncate_tail,
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellCaptureResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub cancelled: bool,
    pub truncated: bool,
    pub full_output_path: Option<String>,
}

pub fn sanitize_binary_output(output: &str) -> String {
    output
        .chars()
        .filter(|ch| {
            let code = *ch as u32;
            code == 0x09
                || code == 0x0a
                || code == 0x0d
                || (code > 0x1f && !(0xfff9..=0xfffb).contains(&code))
        })
        .collect()
}

pub fn execute_shell_with_capture(
    env: &LocalExecutionEnv,
    command: &str,
    mut options: ExecOptions,
) -> Result<ShellCaptureResult, FileError> {
    let abort_flag = options.abort_flag.clone();
    let stdout_callback = options.on_stdout.take();
    let stderr_callback = options.on_stderr.take();
    let state = Arc::new(Mutex::new(ShellCaptureState::default()));
    options.on_stdout = Some(capture_callback(env, state.clone(), stdout_callback));
    options.on_stderr = Some(capture_callback(env, state.clone(), stderr_callback));

    let output = match env.exec(command, options) {
        Ok(output) => output,
        Err(error) if error.code == FileErrorCode::Timeout => {
            return Err(error);
        }
        Err(error) if error.code == FileErrorCode::Aborted => {
            return finish_capture(env, state, None, true);
        }
        Err(error) => return Err(error),
    };

    let cancelled = abort_flag
        .as_ref()
        .is_some_and(|abort_flag| abort_flag.load(std::sync::atomic::Ordering::SeqCst));
    finish_capture(
        env,
        state,
        (!cancelled).then_some(output.exit_code),
        cancelled,
    )
}

#[derive(Debug, Default)]
struct ShellCaptureState {
    chunks: VecDeque<String>,
    output_bytes: usize,
    total_bytes: usize,
    full_output_path: Option<String>,
}

fn capture_callback(
    env: &LocalExecutionEnv,
    state: Arc<Mutex<ShellCaptureState>>,
    inner: Option<ExecCallback>,
) -> ExecCallback {
    let env = env.clone();
    Arc::new(move |chunk| {
        capture_chunk(&env, &state, chunk)?;
        if let Some(inner) = &inner {
            inner(chunk)?;
        }
        Ok(())
    })
}

fn capture_chunk(
    env: &LocalExecutionEnv,
    state: &Arc<Mutex<ShellCaptureState>>,
    chunk: &str,
) -> Result<(), String> {
    let text = sanitize_binary_output(chunk).replace('\r', "");
    let mut state = state
        .lock()
        .map_err(|_| "capture lock poisoned".to_owned())?;
    state.total_bytes += chunk.len();
    if state.total_bytes > DEFAULT_MAX_BYTES && state.full_output_path.is_none() {
        let path = env
            .create_temp_file("bash-", ".log")
            .map_err(|error| error.message)?;
        let initial_content = state.chunks.iter().cloned().collect::<String>() + &text;
        env.append_file(&path, initial_content.as_bytes())
            .map_err(|error| error.message)?;
        state.full_output_path = Some(path);
    } else if let Some(path) = &state.full_output_path {
        env.append_file(path, text.as_bytes())
            .map_err(|error| error.message)?;
    }
    state.output_bytes += text.len();
    state.chunks.push_back(text);
    while state.output_bytes > DEFAULT_MAX_BYTES * 2 && state.chunks.len() > 1 {
        if let Some(removed) = state.chunks.pop_front() {
            state.output_bytes = state.output_bytes.saturating_sub(removed.len());
        }
    }
    Ok(())
}

fn finish_capture(
    env: &LocalExecutionEnv,
    state: Arc<Mutex<ShellCaptureState>>,
    exit_code: Option<i32>,
    cancelled: bool,
) -> Result<ShellCaptureResult, FileError> {
    let (tail_output, mut full_output_path) = {
        let state = state.lock().map_err(|_| FileError {
            code: FileErrorCode::Unknown,
            message: "capture lock poisoned".to_owned(),
            path: String::new(),
        })?;
        (
            state.chunks.iter().cloned().collect::<String>(),
            state.full_output_path.clone(),
        )
    };
    let truncation = truncate_tail(&tail_output, TruncationOptions::default());
    if truncation.truncated && full_output_path.is_none() {
        let path = env.create_temp_file("bash-", ".log")?;
        env.append_file(&path, tail_output.as_bytes())?;
        full_output_path = Some(path);
    }
    Ok(ShellCaptureResult {
        output: if truncation.truncated {
            truncation.content
        } else {
            tail_output
        },
        exit_code,
        cancelled,
        truncated: truncation.truncated,
        full_output_path,
    })
}
