use crate::types::{AssistantMessage, StopReason};
use regex::Regex;

static OVERFLOW_PATTERNS: std::sync::LazyLock<Vec<Regex>> = std::sync::LazyLock::new(|| {
    [
        r"prompt is too long",
        r"request_too_large",
        r"input is too long for requested model",
        r"exceeds the context window",
        r"exceeds (?:the )?(?:model'?s )?maximum context length of [\d,]+ tokens?",
        r"input token count.*exceeds the maximum",
        r"maximum prompt length is \d+",
        r"reduce the length of the messages",
        r"maximum context length is \d+ tokens",
        r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)",
        r"exceeds the limit of \d+",
        r"exceeds the available context size",
        r"greater than the context length",
        r"context window exceeds limit",
        r"exceeded model token limit",
        r"too large for model with \d+ maximum context length",
        r"model_context_window_exceeded",
        r"prompt too long; exceeded (?:max )?context length",
        r"context[_ ]length[_ ]exceeded",
        r"too many tokens",
        r"token limit exceeded",
        r"^4(?:00|13|29)\s*(?:status code)?\s*\(no body\)",
    ]
    .into_iter()
    .map(|pattern| Regex::new(&format!("(?i){pattern}")).expect("valid overflow regex"))
    .collect()
});

static NON_OVERFLOW_PATTERNS: std::sync::LazyLock<Vec<Regex>> = std::sync::LazyLock::new(|| {
    [
        r"^(Throttling error|Service unavailable):",
        r"rate limit",
        r"too many requests",
    ]
    .into_iter()
    .map(|pattern| Regex::new(&format!("(?i){pattern}")).expect("valid non-overflow regex"))
    .collect()
});

pub fn is_context_overflow(message: &AssistantMessage, context_window: Option<u64>) -> bool {
    if message.stop_reason == StopReason::Error {
        if let Some(error) = &message.error_message {
            let non_overflow = NON_OVERFLOW_PATTERNS
                .iter()
                .any(|pattern| pattern.is_match(error));
            if !non_overflow
                && OVERFLOW_PATTERNS
                    .iter()
                    .any(|pattern| pattern.is_match(error))
            {
                return true;
            }
        }
    }

    if let Some(context_window) = context_window {
        let input_tokens = message.usage.input + message.usage.cache_read;
        if message.stop_reason == StopReason::Stop && input_tokens > context_window {
            return true;
        }

        if message.stop_reason == StopReason::Length
            && message.usage.output == 0
            && input_tokens as f64 >= context_window as f64 * 0.99
        {
            return true;
        }
    }

    false
}

pub fn overflow_pattern_count() -> usize {
    OVERFLOW_PATTERNS.len()
}
