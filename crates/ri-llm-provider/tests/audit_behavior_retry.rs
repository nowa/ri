//! Regression tests for the retry/error-body parity audit (pi v0.81.1):
//! codex inner-retry semantics, retry-after parsing, terminal rate-limit
//! gating, SDK-shaped provider error strings, bedrock formatting/payload
//! parity, abortable retry waits, null provider headers, and SSE BOM
//! stripping.

use futures::StreamExt;
use ri_llm_provider::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Duration,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        ..Default::default()
    }
}

fn text_of(message: &AssistantMessage) -> Option<&str> {
    match message.content.first()? {
        AssistantContent::Text(text) => Some(&text.text),
        _ => None,
    }
}

fn error_assistant(error_message: &str) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        model: "test".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Error,
        error_message: Some(error_message.to_owned()),
        timestamp: 0,
    }
}

const CODEX_TEST_JWT_PAYLOAD: &str =
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjX3Rlc3QifX0=";

fn codex_test_token() -> String {
    format!("aaa.{CODEX_TEST_JWT_PAYLOAD}.bbb")
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|index| index + 4)
    else {
        return false;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
    let content_length = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    request.len() >= header_end + content_length
}

/// One-shot mock server answering every accepted connection with the same
/// canned HTTP response; counts requests and records the first request text.
async fn mock_response_server(
    response: String,
    request_count: Arc<AtomicUsize>,
    first_request: Arc<Mutex<Option<String>>>,
) -> String {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let response = response.clone();
            let request_count = request_count.clone();
            let first_request = first_request.clone();
            tokio::spawn(async move {
                let mut request = Vec::new();
                let mut buf = [0u8; 1024];
                loop {
                    let n = socket.read(&mut buf).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    request.extend_from_slice(&buf[..n]);
                    if request_is_complete(&request) {
                        break;
                    }
                }
                if request.is_empty() {
                    return;
                }
                request_count.fetch_add(1, Ordering::SeqCst);
                first_request
                    .lock()
                    .expect("first request lock")
                    .get_or_insert_with(|| String::from_utf8_lossy(&request).into_owned());
                let _ = socket.write_all(response.as_bytes()).await;
                let _ = socket.flush().await;
            });
        }
    });
    format!("http://{addr}")
}

fn http_response(status_line: &str, headers: &[(&str, &str)], body: &str) -> String {
    let mut response = format!("HTTP/1.1 {status_line}\r\n");
    for (name, value) in headers {
        response.push_str(&format!("{name}: {value}\r\n"));
    }
    response.push_str(&format!(
        "content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    ));
    response
}

// ============================================================================
// Fix 1/2/3/4: codex inner retry defaults, cap semantics, retry-after parsing,
// terminal rate-limit gate (pi openai-codex-responses.ts).
// ============================================================================

#[test]
fn openai_codex_default_max_retries_is_zero() {
    assert_eq!(OPENAI_CODEX_DEFAULT_MAX_RETRIES, 0);
    // Without an explicit max_retries even a plain 429 never retries.
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", None, None, 0, 0),
        None
    );
}

#[test]
fn openai_codex_retry_delay_cap_applies_only_to_429_retry_after() {
    // (a) 429 retry-after 600s with no option: default cap 60s.
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "", None, Some("600"), 0, 0, 3, None),
        Some(60_000)
    );
    // (b) non-429 retry-after is never capped, even with an explicit cap.
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(503, "", None, Some("120"), 0, 0, 3, Some(10_000)),
        Some(120_000)
    );
    // (c) an explicit cap of 0 disables capping entirely.
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "", None, Some("5"), 0, 0, 3, Some(0)),
        Some(5_000)
    );
    // Exponential backoff is never capped.
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "", None, None, 3, 0, 5, Some(500)),
        Some(8_000)
    );
    // Explicit caps still clamp 429 retry-after delays.
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "", None, Some("60"), 0, 0, 3, Some(10_000)),
        Some(10_000)
    );
}

#[test]
fn openai_codex_retry_after_parsing_matches_js_semantics() {
    // Non-finite retry-after-ms falls through to retry-after.
    assert_eq!(
        openai_codex_retry_after_delay_ms(Some("abc"), Some("5"), 0),
        Some(5_000)
    );
    // Number("") is 0 in JS.
    assert_eq!(
        openai_codex_retry_after_delay_ms(Some(""), None, 0),
        Some(0)
    );
    // Whitespace and hex literals parse like JS Number().
    assert_eq!(
        openai_codex_retry_after_delay_ms(Some("  5  "), None, 0),
        Some(5)
    );
    assert_eq!(
        openai_codex_retry_after_delay_ms(Some("0x10"), None, 0),
        Some(16)
    );
    // Negative values clamp to 0.
    assert_eq!(
        openai_codex_retry_after_delay_ms(Some("-100"), None, 0),
        Some(0)
    );
    // retry-after seconds, fractional, negative clamps.
    assert_eq!(
        openai_codex_retry_after_delay_ms(None, Some("1.5"), 0),
        Some(1_500)
    );
    assert_eq!(
        openai_codex_retry_after_delay_ms(None, Some("-3"), 0),
        Some(0)
    );
    // Empty retry-after is falsy in pi: no delay.
    assert_eq!(openai_codex_retry_after_delay_ms(None, Some(""), 0), None);
    // IMF-fixdate and ISO-8601 http-dates are both accepted.
    assert_eq!(
        openai_codex_retry_after_delay_ms(None, Some("Thu, 01 Jan 1970 00:00:45 GMT"), 0),
        Some(45_000)
    );
    assert_eq!(
        openai_codex_retry_after_delay_ms(None, Some("1970-01-01T00:00:45Z"), 0),
        Some(45_000)
    );
    // Past dates clamp to 0.
    assert_eq!(
        openai_codex_retry_after_delay_ms(None, Some("1970-01-01T00:00:45Z"), 100_000),
        Some(0)
    );
    // Unparseable retry-after yields no delay (falls back to backoff).
    assert_eq!(
        openai_codex_retry_after_delay_ms(None, Some("soon"), 0),
        None
    );
    // Unparseable retry-after-ms with no retry-after: backoff applies.
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "", Some("abc"), None, 1, 0, 3, None),
        Some(2_000)
    );
}

#[test]
fn openai_codex_terminal_rate_limit_gate_uses_raw_body() {
    for terminal in [
        "GoUsageLimitError",
        "FreeUsageLimitError",
        "Monthly usage limit reached",
        "enable available balance usage",
        "insufficient_quota",
        "out of budget",
        "quota exceeded",
        "billing hard limit",
    ] {
        assert!(
            !is_openai_codex_retryable_error(429, terminal),
            "429 {terminal} must be terminal"
        );
        assert_eq!(
            openai_codex_retry_delay_ms_with_limits(429, terminal, None, None, 0, 0, 3, None),
            None,
            "429 {terminal} must not schedule a retry"
        );
    }
    // Terminal text on non-429 statuses does not disable status retryability.
    assert!(is_openai_codex_retryable_error(500, "billing subsystem"));
    assert!(is_openai_codex_retryable_error(429, "slow down"));

    // pi matches the RAW body with a regex (`rate.?limit` etc.), not a
    // whitespace-normalized substring.
    assert!(is_openai_codex_retryable_error(
        400,
        r#"{"error":{"code":"rate_limit_exceeded","message":"Slow down"}}"#
    ));
    assert!(is_openai_codex_retryable_error(400, "rateXlimit"));
    assert!(!is_openai_codex_retryable_error(400, "rate  limit"));
    assert!(is_openai_codex_retryable_error(
        400,
        "upstream connect error"
    ));
    assert!(!is_openai_codex_retryable_error(400, "plain bad request"));
}

#[tokio::test]
async fn openai_codex_http_500_fails_after_single_request_by_default() {
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let base_url = mock_response_server(
        http_response("500 Internal Server Error", &[], "server exploded"),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.transport = Some(Transport::Sse);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("server exploded"));
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn openai_codex_terminal_429_fails_fast_even_with_retry_budget() {
    let body = r#"{"error":{"code":"GoUsageLimitError","message":"Monthly usage limit reached"}}"#;
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let base_url = mock_response_server(
        http_response("429 Too Many Requests", &[("retry-after", "1")], body),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.transport = Some(Transport::Sse);
    options.stream.max_retries = Some(3);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");

    assert_eq!(message.stop_reason, StopReason::Error);
    // pi surfaces the friendly usage-limit message on 429 bodies.
    assert_eq!(
        message.error_message.as_deref(),
        Some("You have hit your ChatGPT usage limit.")
    );
    assert_eq!(request_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn openai_codex_opted_in_retry_budget_is_honored() {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    // First request: retryable 500 with a 1ms retry-after-ms; second: success.
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let request_count = Arc::new(AtomicUsize::new(0));
    let server_count = request_count.clone();
    tokio::spawn(async move {
        let sse = concat!(
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_retry\"}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg\",\"content\":[{\"type\":\"output_text\",\"text\":\"Code\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_retry\",\"status\":\"completed\"}}\n\n",
        );
        for attempt in 0..2 {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = socket.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request_is_complete(&request) {
                    break;
                }
            }
            server_count.fetch_add(1, Ordering::SeqCst);
            let response = if attempt == 0 {
                http_response(
                    "500 Internal Server Error",
                    &[("retry-after-ms", "1")],
                    "overloaded",
                )
            } else {
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    sse.len(),
                    sse
                )
            };
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        }
    });

    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = format!("http://{addr}");
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.transport = Some(Transport::Sse);
    options.stream.max_retries = Some(2);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");

    assert_eq!(text_of(&message), Some("Code"));
    assert_eq!(request_count.load(Ordering::SeqCst), 2);
}

// ============================================================================
// Fix 7: retry sleeps observe the abort flag.
// ============================================================================

#[tokio::test]
async fn openai_codex_retry_after_sleep_observes_abort() {
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let base_url = mock_response_server(
        http_response(
            "429 Too Many Requests",
            &[("retry-after", "60")],
            "rate limited",
        ),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.transport = Some(Transport::Sse);
    options.stream.max_retries = Some(2);
    options.stream.abort_flag = Some(abort_flag.clone());

    let abort_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        abort_flag.store(true, Ordering::SeqCst);
    });

    let started = std::time::Instant::now();
    let message = tokio::time::timeout(
        Duration::from_secs(5),
        complete_simple(&model, user_context("hello"), options),
    )
    .await
    .expect("abort during a 60s retry-after sleep must return promptly")
    .expect("complete");
    abort_task.await.expect("abort task");

    assert_eq!(message.stop_reason, StopReason::Aborted);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "aborted after {:?}",
        started.elapsed()
    );
}

// ============================================================================
// Fix 8: WebSocket close code 1009 reason text.
// ============================================================================

#[test]
fn openai_codex_websocket_close_error_appends_message_too_big() {
    assert_eq!(
        openai_codex_websocket_close_error_message(Some(1009), ""),
        "WebSocket closed 1009 message too big"
    );
    assert_eq!(
        openai_codex_websocket_close_error_message(Some(1009), "explicit reason"),
        "WebSocket closed 1009 explicit reason"
    );
    assert_eq!(
        openai_codex_websocket_close_error_message(Some(1000), "done"),
        "WebSocket closed 1000 done"
    );
    assert_eq!(
        openai_codex_websocket_close_error_message(Some(1000), ""),
        "WebSocket closed 1000"
    );
    assert_eq!(
        openai_codex_websocket_close_error_message(None, ""),
        "WebSocket closed"
    );
}

// ============================================================================
// Fix 9: cached WebSocket idle TTL.
// ============================================================================

#[test]
fn openai_codex_websocket_cache_idle_ttl_is_five_minutes() {
    assert_eq!(OPENAI_CODEX_SESSION_WEBSOCKET_CACHE_TTL_MS, 5 * 60 * 1000);
    assert!(!openai_codex_websocket_session_idle_expired(
        0,
        5 * 60 * 1000 - 1
    ));
    assert!(openai_codex_websocket_session_idle_expired(
        0,
        5 * 60 * 1000
    ));
    // The max-age rotation still applies independently.
    assert!(openai_codex_websocket_session_expired(0, 55 * 60 * 1000));
}

// ============================================================================
// Fix 5: SDK-shaped provider error strings.
// ============================================================================

#[test]
fn anthropic_error_messages_mirror_the_sdk_shapes() {
    // The Anthropic SDK folds the WHOLE parsed body into the message.
    assert_eq!(
        anthropic_provider_error_from_body(
            429,
            r#"{"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#
        ),
        r#"429 {"type":"error","error":{"type":"rate_limit_error","message":"Rate limited"}}"#
    );
    // Top-level message wins when present.
    assert_eq!(
        anthropic_provider_error_from_body(400, r#"{"message":"boom"}"#),
        "400 boom"
    );
    // Empty body: "<status> status code (no body)" exactly like the SDK.
    assert_eq!(
        anthropic_provider_error_from_body(429, ""),
        "429 status code (no body)"
    );
    // Non-JSON bodies pass through raw.
    assert_eq!(
        anthropic_provider_error_from_body(502, "<html>gateway</html>"),
        "502 <html>gateway</html>"
    );
    // An empty JSON object stringifies like the SDK does.
    assert_eq!(anthropic_provider_error_from_body(400, "{}"), "400 {}");
}

#[test]
fn openai_error_messages_mirror_the_sdk_and_format_provider_error() {
    // The parsed `error` field surfaces as the body next to the status.
    assert_eq!(
        openai_provider_error_from_body(
            403,
            r#"{"error":{"message":"denied","type":"forbidden"}}"#,
            Some("OpenAI API error")
        ),
        r#"OpenAI API error (403): {"message":"denied","type":"forbidden"}"#
    );
    assert_eq!(
        openai_provider_error_from_body(429, r#"{"error":{"message":"Rate limited"}}"#, None),
        r#"429: {"message":"Rate limited"}"#
    );
    // Body "{}" behaves like an empty body (pi isNonEmptyObject).
    assert_eq!(
        openai_provider_error_from_body(400, "{}", Some("OpenAI API error")),
        "OpenAI API error (400): 400 status code (no body)"
    );
    assert_eq!(
        openai_provider_error_from_body(400, "{}", None),
        "400 status code (no body)"
    );
    // Truly empty body.
    assert_eq!(
        openai_provider_error_from_body(401, "", None),
        "401 status code (no body)"
    );
    // Non-JSON body folds into the message like the SDK errText fallback.
    assert_eq!(
        openai_provider_error_from_body(502, "<html>bad gateway</html>", None),
        "502 <html>bad gateway</html>"
    );
    // A string `error` field is stringified by makeMessage.
    assert_eq!(
        openai_provider_error_from_body(403, r#"{"error":"blocked by gateway WAF"}"#, None),
        r#"403 "blocked by gateway WAF""#
    );
}

#[test]
fn openai_completions_error_appends_openrouter_metadata_raw_once() {
    let body = r#"{"error":{"message":"Provider returned error","code":403,"metadata":{"raw":"upstream WAF blocked policy XYZ"}}}"#;
    let message = openai_completions_provider_error_from_body(403, body);
    assert_eq!(
        message.matches("upstream WAF blocked policy XYZ").count(),
        1,
        "{message}"
    );
    assert!(message.contains("403"));
}

#[test]
fn google_error_messages_mirror_the_genai_sdk() {
    // JSON error bodies re-serialize whole (status digits included), so the
    // outer retry classifier sees the 429.
    let body = r#"{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}"#;
    let message = google_provider_error_from_body(429, "Too Many Requests", true, body);
    assert_eq!(message, body);
    assert!(is_retryable_assistant_error(&error_assistant(&message)));

    // Non-JSON bodies wrap in the SDK's synthetic error envelope.
    assert_eq!(
        google_provider_error_from_body(502, "Bad Gateway", false, "<html>oops</html>"),
        r#"{"error":{"message":"<html>oops</html>","code":502,"status":"Bad Gateway"}}"#
    );
}

#[test]
fn provider_error_bodies_truncate_by_utf16_code_units() {
    // 3 emoji at 2 UTF-16 units each; cap of 4 keeps 2 emoji and reports the
    // JS `text.length - maxChars` arithmetic.
    let text = "😀😀😀";
    assert_eq!(
        truncate_provider_error_text(text, 4),
        "😀😀... [truncated 2 chars]"
    );
    // ASCII within the cap is untouched.
    assert_eq!(truncate_provider_error_text("abc", 4000), "abc");
    let long = "x".repeat(4050);
    assert_eq!(
        truncate_provider_error_text(&long, 4000),
        format!("{}... [truncated 50 chars]", "x".repeat(4000))
    );
}

#[tokio::test]
async fn anthropic_http_error_posts_v1_messages_and_surfaces_sdk_message() {
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let base_url = mock_response_server(
        http_response("400 Bad Request", &[], ""),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = get_model("anthropic", "claude-sonnet-4-5").expect("anthropic model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("anthropic-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");

    assert_eq!(message.stop_reason, StopReason::Error);
    // Empty-body composition matches the Anthropic SDK, which keeps ri's
    // overflow auto-detection (`^4(00|13) ... (no body)`) aligned with pi.
    assert_eq!(
        message.error_message.as_deref(),
        Some("400 status code (no body)")
    );
    let request = first_request
        .lock()
        .expect("request lock")
        .clone()
        .expect("request captured");
    // pi's Anthropic SDK posts to {baseUrl}/v1/messages.
    assert!(
        request.starts_with("POST /v1/messages HTTP/1.1"),
        "{request}"
    );
}

// ============================================================================
// Fix 6 / item F: bedrock HTTP error prefixes and claude detection.
// ============================================================================

#[test]
fn bedrock_http_errors_carry_exception_prefixes() {
    let throttle = format_bedrock_http_error(
        429,
        Some("ThrottlingException:http://internal.amazon.com/coral/"),
        r#"{"message":"Too many tokens, please wait before trying again."}"#,
    );
    assert_eq!(
        throttle,
        r#"Throttling error: 429: {"message":"Too many tokens, please wait before trying again."}"#
    );
    // The prefix anchors ri's NON_OVERFLOW pattern: throttling is not
    // misclassified as context overflow despite the "too many tokens" text.
    assert!(!is_context_overflow(
        &error_assistant(&throttle),
        Some(200_000)
    ));
    // ...but it remains retryable via the 429 digits.
    assert!(is_retryable_assistant_error(&error_assistant(&throttle)));

    // Error name from the body __type when no header exists.
    assert_eq!(
        format_bedrock_http_error(
            503,
            None,
            r#"{"__type":"com.amazon.bedrock#ServiceUnavailableException","message":"try later"}"#
        ),
        r#"Service unavailable: 503: {"__type":"com.amazon.bedrock#ServiceUnavailableException","message":"try later"}"#
    );

    // Gateway bodies surface with status digits instead of Unknown: UnknownError.
    assert_eq!(
        format_bedrock_http_error(403, None, "<html>blocked</html>"),
        "Unknown: 403: <html>blocked</html>"
    );

    // Empty body keeps the SDK's Unknown/UnknownError shape.
    assert_eq!(
        format_bedrock_http_error(403, None, ""),
        "Unknown: UnknownError"
    );

    // Data-retention rejections gain the docs hint.
    let retention = format_bedrock_http_error(
        400,
        Some("ValidationException"),
        r#"{"message":"data retention mode 'default' is not available for this model"}"#,
    );
    assert!(retention.starts_with("Validation error: 400: "));
    assert!(retention.ends_with(
        "See https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html for supported data retention modes."
    ));
}

#[test]
fn bedrock_claude_detection_requires_prefixed_id_or_named_claude() {
    let mut model = Model::faux("bedrock-converse-stream", "amazon-bedrock", "claude-3-opus");
    model.name = "Some Model".to_owned();
    // Bare "claude" in the id does not count (pi matches the name only).
    assert!(!is_bedrock_anthropic_claude_model(&model));

    model.id = "us.anthropic.claude-opus-4-5-20251101-v1:0".to_owned();
    assert!(is_bedrock_anthropic_claude_model(&model));

    model.id = "arn:aws:bedrock:us-east-1:123:application-inference-profile/abc".to_owned();
    model.name = "Claude Opus 4.5".to_owned();
    assert!(is_bedrock_anthropic_claude_model(&model));
}

// ============================================================================
// Items B/C: bedrock cache TTL wire value and Claude 5 prompt caching.
// ============================================================================

#[test]
fn bedrock_long_cache_ttl_uses_the_aws_wire_value() {
    let model = get_model("amazon-bedrock", "anthropic.claude-fable-5").expect("model");
    let system = build_bedrock_system_prompt(Some("system"), &model, CacheRetention::Long)
        .expect("system prompt");
    assert_eq!(system[1]["cachePoint"]["ttl"], json!("1h"));
    // Short retention omits the ttl.
    let system = build_bedrock_system_prompt(Some("system"), &model, CacheRetention::Short)
        .expect("system prompt");
    assert_eq!(system[1]["cachePoint"].get("ttl"), None);
}

#[test]
fn bedrock_prompt_caching_includes_claude_5_family() {
    for model_id in [
        "anthropic.claude-fable-5",
        "anthropic.claude-sonnet-5",
        "us.anthropic.claude-opus-4-5-20251101-v1:0",
    ] {
        let model = get_model("amazon-bedrock", model_id)
            .unwrap_or_else(|| Model::faux("bedrock-converse-stream", "amazon-bedrock", model_id));
        assert!(
            supports_bedrock_prompt_caching(&model),
            "{model_id} should support prompt caching"
        );
    }
    let nova = get_model("amazon-bedrock", "amazon.nova-pro-v1:0").expect("nova");
    assert!(!supports_bedrock_prompt_caching(&nova));
}

// ============================================================================
// Item G: bedrock usage totals.
// ============================================================================

#[tokio::test]
async fn bedrock_usage_total_tokens_fall_back_to_input_plus_output() {
    let model = get_model("amazon-bedrock", "anthropic.claude-fable-5").expect("model");
    let (sender, stream) = assistant_message_event_stream();
    let mut output = error_assistant("");
    output.stop_reason = StopReason::Stop;
    output.error_message = None;

    process_bedrock_converse_stream_events(
        vec![
            json!({ "messageStart": { "role": "assistant" } }),
            json!({ "messageStop": { "stopReason": "end_turn" } }),
            json!({ "metadata": { "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cacheReadInputTokens": 100,
                "cacheWriteInputTokens": 50,
            } } }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process events");
    drop(sender);
    drop(stream);

    // pi: totalTokens || input + output — cache tokens are not added.
    assert_eq!(output.usage.total_tokens, 15);
    assert_eq!(output.usage.cache_read, 100);
    assert_eq!(output.usage.cache_write, 50);
}

// ============================================================================
// Item D: bedrock simple-stream defaults + thinking budget adjustment.
// ============================================================================

struct CapturePayload(Arc<Mutex<Option<Value>>>);

impl ProviderPayloadHook for CapturePayload {
    fn on_payload(&self, _model: &Model, payload: Value) -> Result<Value, String> {
        *self.0.lock().expect("payload lock") = Some(payload.clone());
        Ok(payload)
    }
}

#[tokio::test]
async fn bedrock_simple_stream_defaults_max_tokens_and_thinking_budget() {
    let mut model =
        get_model("amazon-bedrock", "anthropic.claude-opus-4-5-20251101-v1:0").expect("model");
    // Keep the spawned request away from real AWS and skip SigV4 signing.
    model.base_url = "http://127.0.0.1:1".to_owned();
    model
        .headers
        .insert("authorization".to_owned(), "Bearer test".to_owned());
    assert!(!supports_bedrock_adaptive_thinking(&model));

    let captured = Arc::new(Mutex::new(None));
    let context = user_context("hello");
    let mut options = SimpleStreamOptions::default();
    options.reasoning = Some(ThinkingLevel::High);
    options
        .payload_hooks
        .push(Arc::new(CapturePayload(captured.clone())));

    let stream = stream_simple(&model, context.clone(), options).expect("stream");
    drop(stream);

    let payload = captured
        .lock()
        .expect("payload lock")
        .clone()
        .expect("captured payload");

    // pi: unset maxTokens falls back to the clamped model cap, then the
    // thinking budget grows/clamps it (adjustMaxTokensForThinking).
    let base = simple_options::clamp_max_tokens_to_context(&model, &context, model.max_tokens);
    let adjusted = simple_options::adjust_max_tokens_for_thinking(
        base,
        model.max_tokens,
        ThinkingLevel::High,
        None,
    );
    let expected_max =
        simple_options::clamp_max_tokens_to_context(&model, &context, adjusted.max_tokens);
    let expected_budget = adjusted
        .thinking_budget
        .min(expected_max.saturating_sub(1024));

    assert_eq!(payload["inferenceConfig"]["maxTokens"], json!(expected_max));
    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        json!(expected_budget)
    );
}

// ============================================================================
// Item E: bedrock reserved headers include authorization.
// ============================================================================

#[tokio::test]
async fn bedrock_blocks_caller_supplied_authorization_header() {
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let base_url = mock_response_server(
        http_response("200 OK", &[], ""),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = get_model("amazon-bedrock", "anthropic.claude-fable-5").expect("model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    // Bearer-token auth owns the authorization header.
    options.stream.api_key = Some("bearer-token".to_owned());
    options
        .stream
        .headers
        .insert("authorization".to_owned(), "Bearer evil".to_owned());
    options
        .stream
        .headers
        .insert("x-amz-target".to_owned(), "evil".to_owned());
    options
        .stream
        .headers
        .insert("x-custom".to_owned(), "kept".to_owned());

    let _ = complete_simple(&model, user_context("hello"), options).await;

    let request = first_request
        .lock()
        .expect("request lock")
        .clone()
        .expect("request captured");
    let lower = request.to_ascii_lowercase();
    assert!(
        lower.contains("authorization: bearer bearer-token"),
        "{lower}"
    );
    assert!(!lower.contains("bearer evil"), "{lower}");
    assert!(!lower.contains("x-amz-target"), "{lower}");
    assert!(lower.contains("x-custom: kept"), "{lower}");
}

// ============================================================================
// Item I: anthropic fail-fast auth assertion.
// ============================================================================

#[tokio::test]
async fn anthropic_fails_fast_without_credentials() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let saved = std::env::var("ANTHROPIC_API_KEY").ok();
    // Tests hold ENV_LOCK while mutating the process environment.
    unsafe {
        std::env::remove_var("ANTHROPIC_API_KEY");
    }

    // Exercise the http ApiProvider directly: the runtime provider layer
    // wraps errors into the event stream and may consult credential storage.
    ensure_builtin_api_providers();
    let provider = get_api_provider("anthropic-messages").expect("anthropic provider");
    let model = get_model("anthropic", "claude-sonnet-4-5").expect("anthropic model");
    let error = provider
        .stream_simple(
            &model,
            user_context("hello"),
            SimpleStreamOptions::default(),
        )
        .err()
        .expect("missing auth must fail fast");
    assert!(
        error
            .to_string()
            .contains("No API key for provider: anthropic"),
        "{error:?}"
    );

    // An auth header satisfies the assertion (request may fail later; the
    // provider must not reject it up front).
    let mut options = SimpleStreamOptions::default();
    options
        .stream
        .headers
        .insert("x-api-key".to_owned(), "key".to_owned());
    let mut model = model.clone();
    model.base_url = "http://127.0.0.1:1".to_owned();
    assert!(
        provider
            .stream_simple(&model, user_context("hello"), options)
            .is_ok()
    );

    if let Some(saved) = saved {
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", saved);
        }
    }
}

// ============================================================================
// Fix 10: null provider headers unset model defaults.
// ============================================================================

#[test]
fn stream_options_from_json_tolerates_null_headers() {
    let options = stream_options_from_json(json!({
        "headers": { "x-removed": null, "x-kept": "yes" },
        "maxRetries": 2,
    }))
    .expect("options with null header");

    assert_eq!(
        options.headers.get("x-kept").map(String::as_str),
        Some("yes")
    );
    assert!(!options.headers.contains_key("x-removed"));
    assert_eq!(options.max_retries, Some(2));

    let mut merged = BTreeMap::from([
        ("X-Removed".to_owned(), "default".to_owned()),
        ("x-other".to_owned(), "default".to_owned()),
    ]);
    merged.extend(options.headers.clone());
    remove_null_provider_headers(&mut merged, &options);
    assert!(!merged.contains_key("X-Removed"));
    assert_eq!(merged.get("x-other").map(String::as_str), Some("default"));
    assert_eq!(merged.get("x-kept").map(String::as_str), Some("yes"));
}

#[tokio::test]
async fn null_option_header_unsets_model_default_header_on_the_wire() {
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let sse = concat!(
        "data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let base_url = mock_response_server(
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            sse.len(),
            sse
        ),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = Model::faux("openai-completions", "openai", "mock-model");
    model.base_url = base_url;
    model
        .headers
        .insert("x-default".to_owned(), "model-default".to_owned());

    let stream_options = stream_options_from_json(json!({
        "apiKey": "test-key",
        "headers": { "x-default": null, "x-option": "set" },
    }))
    .expect("options");
    let message = complete_simple(
        &model,
        user_context("hello"),
        SimpleStreamOptions::from_stream_options(stream_options),
    )
    .await
    .expect("complete");

    assert_eq!(text_of(&message), Some("Hi"));
    let request = first_request
        .lock()
        .expect("request lock")
        .clone()
        .expect("request captured");
    let lower = request.to_ascii_lowercase();
    assert!(!lower.contains("x-default"), "{lower}");
    assert!(lower.contains("x-option: set"), "{lower}");
}

// ============================================================================
// Fix 11: SSE parser strips a leading UTF-8 BOM.
// ============================================================================

#[tokio::test]
async fn sse_parser_strips_a_leading_utf8_bom() {
    let request_count = Arc::new(AtomicUsize::new(0));
    let first_request = Arc::new(Mutex::new(None));
    let sse = concat!(
        "\u{feff}",
        "data: {\"id\":\"chatcmpl_bom\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let base_url = mock_response_server(
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            sse.len(),
            sse
        ),
        request_count.clone(),
        first_request.clone(),
    )
    .await;

    let mut model = Model::faux("openai-completions", "openai", "mock-model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(text_of(&message), Some("Hi"));
    assert_eq!(message.response_id.as_deref(), Some("chatcmpl_bom"));
}
