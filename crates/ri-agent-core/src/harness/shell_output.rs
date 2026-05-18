use super::{
    DEFAULT_MAX_BYTES, ExecOptions, FileError, FileErrorCode, LocalExecutionEnv, TruncationOptions,
    truncate_tail,
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
    options: ExecOptions,
) -> Result<ShellCaptureResult, FileError> {
    let output = match env.exec(command, options) {
        Ok(output) => output,
        Err(error) if error.code == FileErrorCode::Timeout => {
            return Err(error);
        }
        Err(error) => return Err(error),
    };

    let combined = sanitize_binary_output(&(output.stdout + &output.stderr)).replace('\r', "");
    let mut full_output_path = None;
    if combined.len() > DEFAULT_MAX_BYTES {
        let path = env.create_temp_file("bash-", ".log")?;
        env.append_file(&path, combined.as_bytes())?;
        full_output_path = Some(path);
    }

    let truncation = truncate_tail(&combined, TruncationOptions::default());
    Ok(ShellCaptureResult {
        output: truncation.content,
        exit_code: Some(output.exit_code),
        cancelled: false,
        truncated: truncation.truncated,
        full_output_path,
    })
}
