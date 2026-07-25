//! Bounded retry for assistant-producing calls, mirroring pi `utils/retry.ts`.

use crate::{AssistantMessage, StopReason};
use regex::Regex;
use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicBool, Ordering},
};

fn build_provider_error_pattern(patterns: &[&str]) -> Regex {
    Regex::new(&format!("(?i){}", patterns.join("|"))).expect("valid provider error pattern")
}

static NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    build_provider_error_pattern(&[
        // OpenCode Go/free-tier limits returned as 429 JSON error types by
        // OpenCode's Zen API. These are subscription/account limits, not
        // transient throttles.
        "GoUsageLimitError",
        "FreeUsageLimitError",
        // OpenCode Go subscription-limit text asks users to enable
        // available-balance usage after rolling/weekly/monthly limits are
        // reached.
        "Monthly usage limit reached",
        "available balance",
        // Generic quota/budget/billing exhaustion. `insufficient_quota` is
        // OpenAI's quota/billing error code; the other strings cover common
        // gateway wording.
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing",
    ])
});

static RETRYABLE_PROVIDER_ERROR_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    build_provider_error_pattern(&[
        // Generic provider load, HTTP status, and server-side transient
        // failures.
        "overloaded",
        "rate.?limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "524",
        "service.?unavailable",
        "server.?error",
        "internal.?error",
        // Wrapper/provider text for transient upstream failures, including
        // OpenRouter "Provider returned error" responses.
        "provider.?returned.?error",
        // Network, proxy, and fetch transport failures, including OpenAI
        // Codex raw-fetch failures and connection drops.
        "network.?error",
        "connection.?error",
        "connection.?refused",
        "connection.?lost",
        "other side closed",
        "fetch failed",
        "upstream.?connect",
        "reset before headers",
        "socket hang up",
        "socket connection was closed",
        "timed? out",
        "timeout",
        "terminated",
        // WebSocket transports can report close/error text instead of
        // HTTP/fetch text.
        "websocket.?closed",
        "websocket.?error",
        // Premature stream endings from SDKs and transports.
        "ended without",
        "stream ended before message_stop",
        "stream ended before a terminal response event",
        "http2 request did not get a response",
        // Provider-requested retry delay cap failures should flow through the
        // outer retry policy so callers can surface/abort the backoff.
        "retry delay",
        // Explicit retry guidance emitted mid-stream by OpenAI Responses and
        // Bedrock stream exceptions.
        "you can retry your request",
        "try your request again",
        "please retry your request",
        // gRPC based providers (e.g. NVIDIA NIM).
        "ResourceExhausted",
    ])
});

/// True when the error text matches a terminal account/subscription limit
/// (pi `NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN`). Shared with the OpenAI
/// Codex inner retry loop, whose terminal rate-limit gate uses the same
/// pattern (pi openai-codex-responses.ts `isTerminalRateLimitError`).
/// Provider-SDK retry semantics (openai-node / @anthropic-ai/sdk share one
/// implementation). pi forwards `options.maxRetries` to these SDKs, so the
/// backoff shape below is what pi actually performs on the anthropic,
/// openai-completions, openai-responses, azure, and openrouter-images paths.
pub const SDK_INITIAL_RETRY_DELAY_MS: f64 = 500.0;
pub const SDK_MAX_RETRY_DELAY_MS: f64 = 8_000.0;
/// The SDKs shave up to 25% off each delay as jitter.
pub const SDK_RETRY_JITTER_FRACTION: f64 = 0.25;

/// SDK `shouldRetry`: an explicit `x-should-retry` header wins, then request
/// timeouts (408), lock timeouts (409), rate limits (429), and server errors.
pub fn sdk_should_retry_status(status: u16, x_should_retry: Option<&str>) -> bool {
    match x_should_retry {
        Some("true") => return true,
        Some("false") => return false,
        _ => {}
    }
    status == 408 || status == 409 || status == 429 || status >= 500
}

/// JS `parseFloat`: leading numeric prefix, ignoring trailing garbage.
fn js_parse_float(text: &str) -> Option<f64> {
    let trimmed = text.trim_start();
    let bytes = trimmed.as_bytes();
    let mut end = 0;
    let mut seen_digit = false;
    let mut seen_dot = false;
    let mut seen_exponent = false;
    while end < bytes.len() {
        let byte = bytes[end];
        match byte {
            b'+' | b'-' if end == 0 => {}
            b'+' | b'-' if seen_exponent && matches!(bytes[end - 1], b'e' | b'E') => {}
            b'0'..=b'9' => seen_digit = true,
            b'.' if !seen_dot && !seen_exponent => seen_dot = true,
            b'e' | b'E' if seen_digit && !seen_exponent => seen_exponent = true,
            _ => break,
        }
        end += 1;
    }
    if !seen_digit {
        return None;
    }
    trimmed[..end]
        .parse::<f64>()
        .ok()
        .filter(|value| !value.is_nan())
}

/// JS `Date.parse` for retry-after headers: IMF-fixdate (RFC 2822) and
/// ISO-8601 forms. Timezone-less datetimes are read as UTC (JS uses local
/// time, but real retry-after dates always carry a zone).
pub(crate) fn parse_http_retry_after_date_ms(text: &str) -> Option<i64> {
    use chrono::{DateTime, Utc};
    if let Ok(date) = DateTime::parse_from_rfc2822(text) {
        return Some(date.with_timezone(&Utc).timestamp_millis());
    }
    if let Ok(date) = DateTime::parse_from_rfc3339(text) {
        return Some(date.with_timezone(&Utc).timestamp_millis());
    }
    if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(datetime.and_utc().timestamp_millis());
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return Some(date.and_hms_opt(0, 0, 0)?.and_utc().timestamp_millis());
    }
    None
}

/// SDK `retryRequest` header handling: `retry-after-ms` wins when it parses,
/// then `retry-after` as seconds, then as an HTTP date. The SDKs apply no cap
/// or floor to header-provided delays.
pub fn sdk_retry_after_delay_ms(
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    now_ms: i64,
) -> Option<f64> {
    if let Some(value) = retry_after_ms.filter(|value| !value.is_empty())
        && let Some(millis) = js_parse_float(value)
    {
        return Some(millis);
    }
    let retry_after = retry_after.filter(|value| !value.is_empty())?;
    if let Some(seconds) = js_parse_float(retry_after) {
        return Some(seconds * 1000.0);
    }
    let date_ms = parse_http_retry_after_date_ms(retry_after)?;
    Some((date_ms - now_ms) as f64)
}

/// SDK `calculateDefaultRetryTimeoutMillis`: `min(0.5s * 2^attempt, 8s)` with
/// up to 25% jitter shaved off. `attempt` is 0-based (the first retry uses
/// 500ms before jitter); `jitter_fraction` is the SDK's
/// `Math.random() * 0.25` sample.
pub fn sdk_default_retry_delay_ms(attempt: u32, jitter_fraction: f64) -> u64 {
    let base = (SDK_INITIAL_RETRY_DELAY_MS * 2f64.powi(attempt as i32)).min(SDK_MAX_RETRY_DELAY_MS);
    let jitter = jitter_fraction.clamp(0.0, SDK_RETRY_JITTER_FRACTION);
    (base * (1.0 - jitter)).max(0.0) as u64
}

/// Full SDK delay for one retry: header-provided delay when present,
/// otherwise the jittered exponential default. Negative header delays (a
/// retry-after date in the past) collapse to no wait.
pub fn sdk_retry_delay_ms(
    attempt: u32,
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    now_ms: i64,
    jitter_fraction: f64,
) -> u64 {
    match sdk_retry_after_delay_ms(retry_after_ms, retry_after, now_ms) {
        Some(delay_ms) => delay_ms.max(0.0) as u64,
        None => sdk_default_retry_delay_ms(attempt, jitter_fraction),
    }
}

/// A jitter sample in `[0, 0.25)` standing in for the SDKs'
/// `Math.random() * 0.25`.
pub fn sdk_retry_jitter_sample() -> f64 {
    use std::hash::{BuildHasher, Hasher};
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(0);
    let sample = (hasher.finish() >> 11) as f64 / (1u64 << 53) as f64;
    sample * SDK_RETRY_JITTER_FRACTION
}

pub fn is_non_retryable_provider_limit_error(error_message: &str) -> bool {
    NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.is_match(error_message)
}

/// Retry policy: bounded attempts with exponential backoff
/// (`base_delay_ms * 2^(attempt-1)`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
    pub enabled: bool,
    /// Max retry attempts (0 = no retries). The initial call never counts as a
    /// retry.
    pub max_retries: u32,
    /// Base delay in ms. Per-attempt delay is `base_delay_ms * 2^(attempt-1)`.
    pub base_delay_ms: u64,
}

/// Optional callbacks emitted by [`retry_assistant_call`] around each retry.
#[derive(Clone, Default)]
pub struct RetryCallbacks {
    /// Emitted before the backoff sleep of each retry attempt (1-indexed).
    pub on_retry_scheduled: Option<Arc<dyn Fn(u32, u32, u64, &str) + Send + Sync>>,
    /// Emitted after the backoff sleep, immediately before the retried call
    /// starts.
    pub on_retry_attempt_start: Option<Arc<dyn Fn() + Send + Sync>>,
    /// Emitted once when the loop ends: success if a later call completed
    /// normally.
    pub on_retry_finished: Option<Arc<dyn Fn(bool, u32, Option<&str>) + Send + Sync>>,
}

impl std::fmt::Debug for RetryCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RetryCallbacks")
    }
}

const RETRY_SLEEP_POLL_MS: u64 = 10;

/// Sleep for `delay_ms`, returning `false` when the abort flag is raised
/// before the sleep completes.
async fn retry_sleep(delay_ms: u64, abort_flag: Option<&Arc<AtomicBool>>) -> bool {
    let mut remaining = delay_ms;
    loop {
        if abort_flag.is_some_and(|flag| flag.load(Ordering::SeqCst)) {
            return false;
        }
        if remaining == 0 {
            return true;
        }
        let step = remaining.min(RETRY_SLEEP_POLL_MS);
        tokio::time::sleep(std::time::Duration::from_millis(step)).await;
        remaining -= step;
    }
}

/// Run a single assistant-producing call with bounded retry on transient
/// errors, mirroring pi `retryAssistantCall`.
///
/// - A successful response is returned immediately. Aborts are terminal and
///   never retried, but reported as unsuccessful if they happen after a retry
///   was scheduled. Aborts during the backoff sleep are normalized to an
///   aborted `AssistantMessage` too.
/// - A non-retryable error (per [`is_retryable_assistant_error`]) is returned
///   immediately so deterministic errors fail fast.
/// - Otherwise retries up to `max_retries` times with exponential backoff.
/// - `produce` errors (as opposed to error-stop-reason messages) propagate
///   without retrying, matching pi's promise-rejection behavior.
pub async fn retry_assistant_call<F, Fut, E>(
    mut produce: F,
    policy: Option<&RetryPolicy>,
    abort_flag: Option<&Arc<AtomicBool>>,
    callbacks: Option<&RetryCallbacks>,
) -> Result<AssistantMessage, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<AssistantMessage, E>>,
{
    let max_attempts = policy
        .filter(|policy| policy.enabled)
        .map(|policy| policy.max_retries)
        .unwrap_or(0);

    let mut attempt = 0u32;
    let mut last_retry: Option<(u32, String)> = None;
    loop {
        let response = produce().await?;

        // Abort: terminal but not successful. Never retry an aborted message.
        if response.stop_reason == StopReason::Aborted {
            if let (Some((attempt, _)), Some(callbacks)) = (&last_retry, callbacks)
                && let Some(on_retry_finished) = &callbacks.on_retry_finished
            {
                on_retry_finished(false, *attempt, None);
            }
            return Ok(response);
        }

        // Success: non-error, non-abort responses return as-is.
        if response.stop_reason != StopReason::Error {
            if let (Some((attempt, _)), Some(callbacks)) = (&last_retry, callbacks)
                && let Some(on_retry_finished) = &callbacks.on_retry_finished
            {
                on_retry_finished(true, *attempt, None);
            }
            return Ok(response);
        }

        // Non-retryable, or budget exhausted: return the final error message.
        if attempt >= max_attempts || !is_retryable_assistant_error(&response) {
            if let (Some((attempt, _)), Some(callbacks)) = (&last_retry, callbacks)
                && let Some(on_retry_finished) = &callbacks.on_retry_finished
            {
                on_retry_finished(false, *attempt, response.error_message.as_deref());
            }
            return Ok(response);
        }

        attempt += 1;
        let error_message = response
            .error_message
            .clone()
            .unwrap_or_else(|| "Unknown error".to_owned());
        last_retry = Some((attempt, error_message.clone()));
        let delay_ms = policy
            .map(|policy| policy.base_delay_ms)
            .unwrap_or(0)
            .saturating_mul(1u64 << (attempt - 1).min(62));
        if let Some(callbacks) = callbacks
            && let Some(on_retry_scheduled) = &callbacks.on_retry_scheduled
        {
            on_retry_scheduled(attempt, max_attempts, delay_ms, &error_message);
        }

        // Normalize aborts during retry backoff to the same AssistantMessage
        // shape as provider stream aborts, so callers do not need to care when
        // cancellation happened.
        if !retry_sleep(delay_ms, abort_flag).await {
            if let Some(callbacks) = callbacks
                && let Some(on_retry_finished) = &callbacks.on_retry_finished
            {
                on_retry_finished(false, attempt, Some(&error_message));
            }
            let mut aborted = response;
            aborted.stop_reason = StopReason::Aborted;
            aborted.error_message = None;
            return Ok(aborted);
        }
        if let Some(callbacks) = callbacks
            && let Some(on_retry_attempt_start) = &callbacks.on_retry_attempt_start
        {
            on_retry_attempt_start();
        }
    }
}

/// Classifies whether a failed assistant message looks like a transient
/// provider or transport error, so callers can decide if the last assistant
/// turn should be restarted.
pub fn is_retryable_assistant_error(message: &AssistantMessage) -> bool {
    if message.stop_reason != StopReason::Error {
        return false;
    }
    let Some(error_message) = message.error_message.as_deref() else {
        return false;
    };
    if error_message.is_empty() {
        return false;
    }
    if NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN.is_match(error_message) {
        return false;
    }
    RETRYABLE_PROVIDER_ERROR_PATTERN.is_match(error_message)
}
