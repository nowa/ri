pub const DEFAULT_MAX_LINES: usize = 2000;
pub const DEFAULT_MAX_BYTES: usize = 50 * 1024;
pub const GREP_MAX_LINE_LENGTH: usize = 500;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    pub text: String,
    pub original_bytes: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncatedBy {
    Lines,
    Bytes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TruncationOptions {
    pub max_lines: usize,
    pub max_bytes: usize,
}

impl Default for TruncationOptions {
    fn default() -> Self {
        Self {
            max_lines: DEFAULT_MAX_LINES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationResult {
    pub content: String,
    pub truncated: bool,
    pub truncated_by: Option<TruncatedBy>,
    pub total_lines: usize,
    pub total_bytes: usize,
    pub output_lines: usize,
    pub output_bytes: usize,
    pub last_line_partial: bool,
    pub first_line_exceeds_limit: bool,
    pub max_lines: usize,
    pub max_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineTruncation {
    pub text: String,
    pub was_truncated: bool,
}

pub fn utf8_byte_len(text: &str) -> usize {
    text.len()
}

pub fn format_size(bytes: usize) -> String {
    if bytes < 1024 {
        format!("{bytes}B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1}KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

pub fn truncate_head(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines;
    let max_bytes = options.max_bytes;
    let total_bytes = utf8_byte_len(content);
    let lines = content.split('\n').collect::<Vec<_>>();
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return truncation_result(
            content.to_owned(),
            false,
            None,
            total_lines,
            total_bytes,
            total_lines,
            total_bytes,
            false,
            false,
            max_lines,
            max_bytes,
        );
    }

    if lines.first().map(|line| line.len()).unwrap_or_default() > max_bytes {
        return truncation_result(
            String::new(),
            true,
            Some(TruncatedBy::Bytes),
            total_lines,
            total_bytes,
            0,
            0,
            false,
            true,
            max_lines,
            max_bytes,
        );
    }

    let mut output = Vec::new();
    let mut output_bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    for (index, line) in lines.iter().take(max_lines).enumerate() {
        let line_bytes = line.len() + usize::from(index > 0);
        if output_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            break;
        }
        output.push(*line);
        output_bytes += line_bytes;
    }

    if output.len() >= max_lines && output_bytes <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output.join("\n");
    let final_output_bytes = utf8_byte_len(&output_content);
    truncation_result(
        output_content,
        true,
        Some(truncated_by),
        total_lines,
        total_bytes,
        output.len(),
        final_output_bytes,
        false,
        false,
        max_lines,
        max_bytes,
    )
}

pub fn truncate_tail(content: &str, options: TruncationOptions) -> TruncationResult {
    let max_lines = options.max_lines;
    let max_bytes = options.max_bytes;
    let total_bytes = utf8_byte_len(content);
    let lines = content.split('\n').collect::<Vec<_>>();
    let total_lines = lines.len();

    if total_lines <= max_lines && total_bytes <= max_bytes {
        return truncation_result(
            content.to_owned(),
            false,
            None,
            total_lines,
            total_bytes,
            total_lines,
            total_bytes,
            false,
            false,
            max_lines,
            max_bytes,
        );
    }

    let mut output = Vec::new();
    let mut output_bytes = 0;
    let mut truncated_by = TruncatedBy::Lines;
    let mut last_line_partial = false;

    for line in lines.iter().rev().take(max_lines) {
        let line_bytes = line.len() + usize::from(!output.is_empty());
        if output_bytes + line_bytes > max_bytes {
            truncated_by = TruncatedBy::Bytes;
            if output.is_empty() {
                let truncated_line = truncate_string_to_bytes_from_end(line, max_bytes);
                output.insert(0, truncated_line);
                last_line_partial = true;
            }
            break;
        }
        output.insert(0, (*line).to_owned());
        output_bytes += line_bytes;
    }

    if output.len() >= max_lines && output_bytes <= max_bytes {
        truncated_by = TruncatedBy::Lines;
    }

    let output_content = output.join("\n");
    let final_output_bytes = utf8_byte_len(&output_content);
    truncation_result(
        output_content,
        true,
        Some(truncated_by),
        total_lines,
        total_bytes,
        output.len(),
        final_output_bytes,
        last_line_partial,
        false,
        max_lines,
        max_bytes,
    )
}

pub fn truncate_head_utf8(text: &str, max_bytes: usize) -> Truncation {
    truncate_utf8(text, max_bytes, false)
}

pub fn truncate_tail_utf8(text: &str, max_bytes: usize) -> Truncation {
    truncate_utf8(text, max_bytes, true)
}

fn truncate_utf8(text: &str, max_bytes: usize, keep_tail: bool) -> Truncation {
    let original_bytes = text.len();
    if original_bytes <= max_bytes {
        return Truncation {
            text: text.to_owned(),
            original_bytes,
            truncated: false,
        };
    }

    let selected = if keep_tail {
        let mut start = text.len();
        for (index, _) in text.char_indices().rev() {
            if text.len() - index > max_bytes {
                break;
            }
            start = index;
        }
        text[start..].to_owned()
    } else {
        let mut end = 0;
        for (index, ch) in text.char_indices() {
            let next = index + ch.len_utf8();
            if next > max_bytes {
                break;
            }
            end = next;
        }
        text[..end].to_owned()
    };

    Truncation {
        text: selected,
        original_bytes,
        truncated: true,
    }
}

pub fn truncate_line(line: &str, max_chars: usize) -> LineTruncation {
    if line.chars().count() <= max_chars {
        return LineTruncation {
            text: line.to_owned(),
            was_truncated: false,
        };
    }

    LineTruncation {
        text: format!(
            "{}... [truncated]",
            line.chars().take(max_chars).collect::<String>()
        ),
        was_truncated: true,
    }
}

fn truncate_string_to_bytes_from_end(text: &str, max_bytes: usize) -> String {
    if max_bytes == 0 {
        return String::new();
    }

    let mut start = text.len();
    let mut output_bytes = 0;
    for (index, ch) in text.char_indices().rev() {
        let char_bytes = ch.len_utf8();
        if output_bytes + char_bytes > max_bytes {
            break;
        }
        output_bytes += char_bytes;
        start = index;
    }
    text[start..].to_owned()
}

#[allow(clippy::too_many_arguments)]
fn truncation_result(
    content: String,
    truncated: bool,
    truncated_by: Option<TruncatedBy>,
    total_lines: usize,
    total_bytes: usize,
    output_lines: usize,
    output_bytes: usize,
    last_line_partial: bool,
    first_line_exceeds_limit: bool,
    max_lines: usize,
    max_bytes: usize,
) -> TruncationResult {
    TruncationResult {
        content,
        truncated,
        truncated_by,
        total_lines,
        total_bytes,
        output_lines,
        output_bytes,
        last_line_partial,
        first_line_exceeds_limit,
        max_lines,
        max_bytes,
    }
}
