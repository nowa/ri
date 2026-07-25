// Parity-audit wave for the OpenAI Responses surfaces (pi v0.81.1):
// Azure base-URL/payload rules, prompt_cache_key clamping, openai-nosession
// session affinity, Codex transport timeouts and header clamps, and provider
// error-body passthrough.

use ri_llm_provider::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Duration};

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

const CODEX_TEST_JWT_PAYLOAD: &str =
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjX3Rlc3QifX0=";

fn codex_test_token() -> String {
    format!("aaa.{CODEX_TEST_JWT_PAYLOAD}.bbb")
}

const CODEX_FALLBACK_SSE: &str = concat!(
    "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_audit\"}}\n\n",
    "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_audit\"}}\n\n",
    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Fallback\"}\n\n",
    "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_audit\",\"content\":[{\"type\":\"output_text\",\"text\":\"Fallback\"}]}}\n\n",
    "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_audit\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":6}}}\n\n",
);

// ============================================================================
// Azure OpenAI Responses
// ============================================================================

#[test]
fn azure_base_url_normalizes_microsoft_foundry_endpoints() {
    assert_eq!(
        normalize_azure_openai_base_url("https://marc-quicktests-resource.ai.azure.com").as_deref(),
        Ok("https://marc-quicktests-resource.ai.azure.com/openai/v1")
    );
    assert_eq!(
        normalize_azure_openai_base_url("https://my-resource.services.ai.azure.com").as_deref(),
        Ok("https://my-resource.services.ai.azure.com/openai/v1")
    );
    assert_eq!(
        normalize_azure_openai_base_url(
            "https://my-resource.services.ai.azure.com/openai/v1/responses"
        )
        .as_deref(),
        Ok("https://my-resource.services.ai.azure.com/openai/v1")
    );
    assert_eq!(
        normalize_azure_openai_base_url("https://my-resource.openai.azure.com/openai/v1/responses")
            .as_deref(),
        Ok("https://my-resource.openai.azure.com/openai/v1")
    );
    // /openai/v1 endpoints stay untouched.
    assert_eq!(
        normalize_azure_openai_base_url("https://my-resource.services.ai.azure.com/openai/v1")
            .as_deref(),
        Ok("https://my-resource.services.ai.azure.com/openai/v1")
    );
    // Non-Azure hosts keep explicit proxy paths.
    assert_eq!(
        normalize_azure_openai_base_url("https://my-proxy.example.com/openai/v1/responses")
            .as_deref(),
        Ok("https://my-proxy.example.com/openai/v1/responses")
    );
}

#[test]
fn azure_payload_disables_storage_and_clamps_prompt_cache_key() {
    let model = get_model("azure-openai-responses", "gpt-4o-mini").expect("azure model");
    let payload = build_azure_openai_responses_payload(
        &model,
        &user_context("hello"),
        AzureOpenAIResponsesPayloadOptions {
            session_id: Some("x".repeat(67)),
            ..Default::default()
        },
    );

    assert_eq!(payload["store"], false);
    assert_eq!(payload["prompt_cache_key"], "x".repeat(64));
}

// ============================================================================
// OpenAI Responses payload + headers
// ============================================================================

#[test]
fn openai_responses_payload_clamps_prompt_cache_key_to_openai_limit() {
    let model = get_model("openai", "gpt-5.4").expect("model");
    let payload = build_openai_responses_payload(
        &model,
        &user_context("hi"),
        OpenAIResponsesPayloadOptions {
            session_id: Some("x".repeat(67)),
            ..Default::default()
        },
    );

    assert_eq!(payload["prompt_cache_key"], "x".repeat(64));
}

#[test]
fn openai_responses_payload_sends_none_reasoning_for_gpt_5_6_models() {
    for model_id in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
        let model = get_model("openai", model_id).expect("model");
        assert_eq!(
            model.thinking_level_map.get(&ThinkingLevel::Off),
            Some(&Some("none".to_owned())),
            "{model_id} off mapping"
        );

        let payload = build_openai_responses_payload(
            &model,
            &user_context("hi"),
            OpenAIResponsesPayloadOptions::default(),
        );
        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "none" }),
            "{model_id}"
        );
    }
}

#[test]
fn openai_responses_headers_support_openai_nosession_format() {
    let mut model = get_model("openai", "gpt-5.4").expect("model");
    model.provider = "proxy".to_owned();
    model.base_url = "https://proxy.example.com/v1".to_owned();
    model.compat = Some(json!({ "sessionAffinityFormat": "openai-nosession" }));

    let headers = build_openai_responses_default_headers(
        &model,
        Some("session-proxy"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );

    assert!(headers.get("session_id").is_none());
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-proxy")
    );
    assert!(headers.get("x-session-id").is_none());
}

#[test]
fn openai_responses_headers_use_openai_nosession_for_opencode_models() {
    let model = get_model("opencode", "gpt-5.4").expect("opencode model");
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("sessionAffinityFormat"))
            .and_then(Value::as_str),
        Some("openai-nosession")
    );

    let headers = build_openai_responses_default_headers(
        &model,
        Some("session-opencode"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-opencode")
    );
    assert!(headers.get("x-session-id").is_none());

    // The body-side cache key is unaffected by the no-session header format.
    let payload = build_openai_responses_payload(
        &model,
        &user_context("hi"),
        OpenAIResponsesPayloadOptions {
            session_id: Some("session-opencode".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["prompt_cache_key"], "session-opencode");
}

// ============================================================================
// Codex session-id header clamp
// ============================================================================

#[test]
fn codex_transport_headers_clamp_session_id_to_64_chars() {
    let token = codex_test_token();
    let session_id = "x".repeat(67);

    let sse = build_openai_codex_sse_headers(
        &BTreeMap::new(),
        &BTreeMap::new(),
        "acc_test",
        &token,
        Some(&session_id),
    );
    assert_eq!(sse.get("session-id"), Some(&"x".repeat(64)));
    assert_eq!(sse.get("x-client-request-id"), Some(&"x".repeat(64)));

    let websocket = build_openai_codex_websocket_headers(
        &BTreeMap::new(),
        &BTreeMap::new(),
        "acc_test",
        &token,
        &session_id,
    );
    assert_eq!(websocket.get("session-id"), Some(&"x".repeat(64)));
    assert_eq!(websocket.get("x-client-request-id"), Some(&"x".repeat(64)));
}

// ============================================================================
// Codex SSE headers-arrival timeout
// ============================================================================

#[tokio::test]
async fn codex_sse_headers_arrival_timeout_errors_without_response() {
    let (base_url, request_task) = mock_hanging_headers_server().await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.transport = Some(Transport::Sse);
    options.stream.timeout_ms = Some(100);
    options.stream.max_retries = Some(0);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("Codex SSE response headers timed out after 100ms")
    );
    assert!(request.starts_with("POST /codex/responses HTTP/1.1"));
}

#[tokio::test]
async fn codex_sse_stream_timeout_does_not_kill_streaming_body_after_headers() {
    // The chunk gap exceeds the stream timeout; only header arrival is guarded.
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            concat!(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_slow\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_slow\"}}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Slow\"}\n\n",
            ),
            concat!(
                "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_slow\",\"content\":[{\"type\":\"output_text\",\"text\":\"Slow\"}]}}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_slow\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":6}}}\n\n",
            ),
        ],
        Duration::from_millis(300),
    )
    .await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.transport = Some(Transport::Sse);
    options.stream.timeout_ms = Some(150);
    options.stream.max_retries = Some(0);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let _request = request_task.await.expect("request task");

    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(text_of(&message), Some("Slow"));
    assert_eq!(message.response_id.as_deref(), Some("resp_slow"));
}

// ============================================================================
// Codex websocket timeouts
// ============================================================================

#[tokio::test]
async fn codex_websocket_connect_timeout_falls_back_to_sse() {
    reset_openai_codex_websocket_debug_stats(Some("audit-ws-connect-timeout"));
    let (base_url, request_task) = mock_websocket_connect_hang_then_sse_server().await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("audit-ws-connect-timeout".to_owned());
    options.stream.transport = Some(Transport::Auto);
    options
        .stream
        .extra
        .insert("websocketConnectTimeoutMs".to_owned(), json!(50));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Fallback"));
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(
        request
            .websocket_handshake
            .starts_with("GET /codex/responses HTTP/1.1")
    );
    assert!(
        request
            .sse_request
            .starts_with("POST /codex/responses HTTP/1.1")
    );
    let stats = get_openai_codex_websocket_debug_stats("audit-ws-connect-timeout").expect("stats");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    assert_eq!(
        stats.last_websocket_error.as_deref(),
        Some("WebSocket connect timeout after 50ms")
    );
}

#[tokio::test]
async fn codex_websocket_idle_before_first_event_falls_back_to_sse() {
    reset_openai_codex_websocket_debug_stats(Some("audit-ws-idle-before"));
    let (base_url, request_task) = mock_websocket_idle_then_sse_server().await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("audit-ws-idle-before".to_owned());
    options.stream.transport = Some(Transport::Auto);
    options.stream.timeout_ms = Some(100);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Fallback"));
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert!(
        request.message.contains("\"type\":\"response.create\""),
        "websocket request frame should have been sent; frame={:?}",
        request.message
    );
    assert!(
        request
            .sse_request
            .starts_with("POST /codex/responses HTTP/1.1")
    );
    let stats = get_openai_codex_websocket_debug_stats("audit-ws-idle-before").expect("stats");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
    assert_eq!(
        stats.last_websocket_error.as_deref(),
        Some("WebSocket idle timeout after 100ms")
    );
}

#[tokio::test]
async fn codex_websocket_idle_after_stream_start_errors() {
    reset_openai_codex_websocket_debug_stats(Some("audit-ws-idle-after"));
    let (base_url, request_task) = mock_websocket_idle_after_event_server(json!({
        "type": "response.created",
        "response": { "id": "resp_idle" },
    }))
    .await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("audit-ws-idle-after".to_owned());
    options.stream.transport = Some(Transport::Auto);
    options.stream.timeout_ms = Some(100);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("WebSocket idle timeout after 100ms")
    );
    assert!(request.contains("\"type\":\"response.create\""));
    let stats = get_openai_codex_websocket_debug_stats("audit-ws-idle-after").expect("stats");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(stats.sse_fallbacks, 0);
}

// ============================================================================
// Provider error bodies (pi provider-error-body-regression)
// ============================================================================

#[tokio::test]
async fn openai_completions_error_surfaces_status_and_body() {
    let (base_url, request_task) =
        mock_json_status_server(403, "Forbidden", r#"{"error":"blocked by gateway WAF"}"#).await;
    let mut model = Model::faux("openai-completions", "openrouter", "test-model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test".to_owned());

    let message = complete_simple(&model, user_context("hi"), options)
        .await
        .expect("complete");
    let _request = request_task.await.expect("request task");

    assert_eq!(message.stop_reason, StopReason::Error);
    // pi: the openai SDK stringifies the body's string `error` field into the
    // message ("403 \"...\""); a string field is not a separate body.
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"403 "blocked by gateway WAF""#)
    );
}

#[tokio::test]
async fn openai_completions_error_does_not_duplicate_openrouter_metadata_raw() {
    let body = r#"{"error":{"message":"Provider returned error","code":403,"metadata":{"raw":"upstream WAF blocked policy XYZ"}}}"#;
    let (base_url, request_task) = mock_json_status_server(403, "Forbidden", body).await;
    let mut model = Model::faux("openai-completions", "openrouter", "test-model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test".to_owned());

    let message = complete_simple(&model, user_context("hi"), options)
        .await
        .expect("complete");
    let _request = request_task.await.expect("request task");

    assert_eq!(message.stop_reason, StopReason::Error);
    let error = message.error_message.as_deref().unwrap_or_default();
    assert!(error.contains("403"), "{error}");
    assert_eq!(
        error.matches("upstream WAF blocked policy XYZ").count(),
        1,
        "{error}"
    );
}

#[tokio::test]
async fn openai_responses_error_keeps_prefix_and_surfaces_body() {
    let (base_url, request_task) =
        mock_json_status_server(403, "Forbidden", r#"{"error":"blocked by gateway WAF"}"#).await;
    let mut model = get_model("openai", "gpt-5.4").expect("model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test".to_owned());

    let message = complete_simple(&model, user_context("hi"), options)
        .await
        .expect("complete");
    let _request = request_task.await.expect("request task");

    assert_eq!(message.stop_reason, StopReason::Error);
    // pi: string `error` field folds into the SDK message, so the prefix
    // carries the SDK-shaped text rather than a separate body.
    assert_eq!(
        message.error_message.as_deref(),
        Some(r#"OpenAI API error (403): 403 "blocked by gateway WAF""#)
    );
}

// ============================================================================
// Mock servers (mirrors of the provider_core helpers)
// ============================================================================

struct MockCodexFallbackRequest {
    websocket_handshake: String,
    message: String,
    sse_request: String,
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|value| value.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    request.len() >= header_end + 4 + content_length
}

fn request_bytes_to_display_string(request: &[u8]) -> String {
    let Some(split) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return String::from_utf8_lossy(request).into_owned();
    };
    let (head, body) = request.split_at(split + 4);
    let head_text = String::from_utf8_lossy(head).into_owned();
    if head_text
        .to_ascii_lowercase()
        .contains("content-encoding: zstd")
    {
        if let Ok(decoded) = zstd::decode_all(body) {
            return format!("{head_text}{}", String::from_utf8_lossy(&decoded));
        }
    }
    format!("{head_text}{}", String::from_utf8_lossy(body))
}

async fn read_request(socket: &mut tokio::net::TcpStream) -> Vec<u8> {
    use tokio::io::AsyncReadExt;

    let mut request = Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        let n = socket.read(&mut buf).await.expect("read request");
        if n == 0 {
            break;
        }
        request.extend_from_slice(&buf[..n]);
        if request_is_complete(&request) {
            break;
        }
    }
    request
}

async fn read_client_websocket_text_frame(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;

    let mut header = [0u8; 2];
    socket.read_exact(&mut header).await.expect("frame header");
    assert_eq!(header[0] & 0x0f, 0x1);
    assert_ne!(header[1] & 0x80, 0, "client frames must be masked");
    let mut len = usize::from(header[1] & 0x7f);
    if len == 126 {
        let mut extended = [0u8; 2];
        socket
            .read_exact(&mut extended)
            .await
            .expect("extended len");
        len = usize::from(u16::from_be_bytes(extended));
    } else if len == 127 {
        let mut extended = [0u8; 8];
        socket
            .read_exact(&mut extended)
            .await
            .expect("extended len");
        len = usize::try_from(u64::from_be_bytes(extended)).expect("frame len");
    }
    let mut mask = [0u8; 4];
    socket.read_exact(&mut mask).await.expect("mask");
    let mut payload = vec![0u8; len];
    socket.read_exact(&mut payload).await.expect("payload");
    for (index, byte) in payload.iter_mut().enumerate() {
        *byte ^= mask[index % 4];
    }
    String::from_utf8(payload).expect("text frame")
}

fn server_websocket_text_frame(text: &str) -> Vec<u8> {
    let payload = text.as_bytes();
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x81);
    if payload.len() <= 125 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

async fn write_sse_response(socket: &mut tokio::net::TcpStream, body: &str) {
    use tokio::io::AsyncWriteExt;

    let response = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body,
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("write sse response");
}

async fn mock_json_status_server(
    status: u16,
    reason: &'static str,
    body: impl Into<String>,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{io::AsyncWriteExt, net::TcpListener};

    let body = body.into();
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let request = read_request(&mut socket).await;
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        request_bytes_to_display_string(&request)
    });
    (format!("http://{addr}"), task)
}

/// Reads the request and never writes response headers; the client's
/// headers-arrival timeout must fire.
async fn mock_hanging_headers_server() -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{io::AsyncReadExt, net::TcpListener};

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let request = read_request(&mut socket).await;
        // Hold the connection until the client gives up and closes it.
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
        }
        request_bytes_to_display_string(&request)
    });
    (format!("http://{addr}"), task)
}

async fn mock_delayed_sse_server(
    chunks: Vec<&'static str>,
    delay: Duration,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{io::AsyncWriteExt, net::TcpListener, time::sleep};

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let request = read_request(&mut socket).await;
        let headers =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        if socket.write_all(headers.as_bytes()).await.is_err() {
            return request_bytes_to_display_string(&request);
        }
        for (index, chunk) in chunks.iter().enumerate() {
            if socket.write_all(chunk.as_bytes()).await.is_err() {
                break;
            }
            if socket.flush().await.is_err() {
                break;
            }
            if index + 1 < chunks.len() {
                sleep(delay).await;
            }
        }
        request_bytes_to_display_string(&request)
    });
    (format!("http://{addr}"), task)
}

/// First connection: the websocket handshake is read but never answered, so
/// the connect timeout fires. Second connection: serves the SSE fallback.
async fn mock_websocket_connect_hang_then_sse_server()
-> (String, tokio::task::JoinHandle<MockCodexFallbackRequest>) {
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut ws_socket, _) = listener.accept().await.expect("accept websocket");
        let handshake = read_request(&mut ws_socket).await;
        // Keep ws_socket open (no handshake response) while the client falls
        // back to SSE on a fresh connection.
        let (mut sse_socket, _) = listener.accept().await.expect("accept sse request");
        let request = read_request(&mut sse_socket).await;
        write_sse_response(&mut sse_socket, CODEX_FALLBACK_SSE).await;
        drop(ws_socket);
        MockCodexFallbackRequest {
            websocket_handshake: String::from_utf8_lossy(&handshake).into_owned(),
            message: String::new(),
            sse_request: request_bytes_to_display_string(&request),
        }
    });
    (format!("http://{addr}"), task)
}

/// First connection: completes the websocket handshake, reads the request
/// frame, then stays silent so the idle timeout fires before any event.
/// Second connection: serves the SSE fallback.
async fn mock_websocket_idle_then_sse_server()
-> (String, tokio::task::JoinHandle<MockCodexFallbackRequest>) {
    use tokio::{io::AsyncWriteExt, net::TcpListener};

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut ws_socket, _) = listener.accept().await.expect("accept websocket");
        let handshake = read_request(&mut ws_socket).await;
        ws_socket
            .write_all(
                b"HTTP/1.1 101 Switching Protocols\r\n\
                  Upgrade: websocket\r\n\
                  Connection: Upgrade\r\n\
                  Sec-WebSocket-Accept: test\r\n\r\n",
            )
            .await
            .expect("write handshake response");
        let message = read_client_websocket_text_frame(&mut ws_socket).await;
        // Send no events; hold the socket open until the client falls back.
        let (mut sse_socket, _) = listener.accept().await.expect("accept sse request");
        let request = read_request(&mut sse_socket).await;
        write_sse_response(&mut sse_socket, CODEX_FALLBACK_SSE).await;
        drop(ws_socket);
        MockCodexFallbackRequest {
            websocket_handshake: String::from_utf8_lossy(&handshake).into_owned(),
            message,
            sse_request: request_bytes_to_display_string(&request),
        }
    });
    (format!("http://{addr}"), task)
}

/// Completes the websocket handshake, answers the request with a single event
/// and then stays silent so the post-start idle timeout fires.
async fn mock_websocket_idle_after_event_server(
    event: Value,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept websocket");
        let _handshake = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 101 Switching Protocols\r\n\
                  Upgrade: websocket\r\n\
                  Connection: Upgrade\r\n\
                  Sec-WebSocket-Accept: test\r\n\r\n",
            )
            .await
            .expect("write handshake response");
        let message = read_client_websocket_text_frame(&mut socket).await;
        socket
            .write_all(&server_websocket_text_frame(&event.to_string()))
            .await
            .expect("write websocket frame");
        // Stay open until the client times out and closes the connection.
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
        }
        message
    });
    (format!("http://{addr}"), task)
}
