//! Parity-audit wave tests for the Bedrock provider surface, porting the pi
//! GAP rows from docs/test-parity-audit-2026-07-23.tsv for
//! bedrock-convert-messages.test.ts, bedrock-custom-headers.test.ts, and
//! bedrock-endpoint-resolution.test.ts.

use ri_llm_provider::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Mutex};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    values: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn clearing(keys: &[&'static str]) -> Self {
        let values = keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in keys {
            remove_env(key);
        }
        Self { values }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            match value {
                Some(value) => set_env(key, value),
                None => remove_env(key),
            }
        }
    }
}

fn set_env(key: &str, value: &str) {
    // These tests hold ENV_LOCK while mutating the process environment.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    // These tests hold ENV_LOCK while mutating the process environment.
    unsafe {
        std::env::remove_var(key);
    }
}

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

fn bedrock_test_model() -> Model {
    get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("bedrock model")
}

// ---------------------------------------------------------------------------
// bedrock-convert-messages.test.ts parity
// ---------------------------------------------------------------------------

/// pi: "replaces user messages with only unknown content blocks with a
/// placeholder" — the raw-path counterpart lives in provider_core.rs
/// (bedrock_raw_message_conversion_replaces_user_messages_with_only_unknown_blocks);
/// this covers a whole raw conversation so the turn ordering stays intact.
#[test]
fn bedrock_raw_conversion_keeps_placeholder_user_turn_between_assistant_turns() {
    let model = bedrock_test_model();
    let messages = convert_bedrock_raw_messages(
        &[
            json!({ "role": "user", "content": "hello" }),
            json!({ "role": "assistant", "content": [{ "type": "text", "text": "hi" }] }),
            json!({
                "role": "user",
                "content": [{ "type": "unknown", "data": "foo" }],
            }),
        ],
        &model,
        CacheRetention::None,
    );

    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["role"], json!("user"));
    assert_eq!(messages[2]["content"], json!([{ "text": "<empty>" }]));
}

/// pi: "filters blank user text blocks when other content remains" on the
/// typed conversion path.
#[test]
fn bedrock_user_blank_text_blocks_filtered_when_other_content_remains() {
    let model = bedrock_test_model();
    let context = Context {
        messages: vec![Message::User(UserMessage {
            content: UserContentValue::Blocks(vec![
                UserContent::Text(TextContent::new("")),
                UserContent::Text(TextContent::new("hello")),
            ]),
            timestamp: 1,
        })],
        ..Default::default()
    };

    let messages = convert_bedrock_messages(&context, &model, CacheRetention::None);

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], json!("user"));
    assert_eq!(messages[0]["content"], json!([{ "text": "hello" }]));
}

/// Raw-path counterpart of the blank-filter rule: blank text blocks drop when
/// other representable content remains, and an all-blank user message becomes
/// the placeholder instead of a blank block or a dropped turn.
#[test]
fn bedrock_raw_user_blank_text_blocks_filtered_when_other_content_remains() {
    let model = bedrock_test_model();
    let messages = convert_bedrock_raw_messages(
        &[
            json!({
                "role": "user",
                "content": [
                    { "type": "text", "text": "" },
                    { "type": "text", "text": "hello" },
                ],
            }),
            json!({ "role": "user", "content": "   " }),
        ],
        &model,
        CacheRetention::None,
    );

    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["content"], json!([{ "text": "hello" }]));
    assert_eq!(messages[1]["content"], json!([{ "text": "<empty>" }]));
}

// ---------------------------------------------------------------------------
// bedrock-endpoint-resolution.test.ts parity
// ---------------------------------------------------------------------------

/// pi: "handles missing regions for explicit, scoped, and ambient profiles" —
/// explicit `profile` options and request-scoped AWS_PROFILE overrides keep
/// the catalog endpoint pinned (deriving the region from it); only an ambient
/// process-env AWS_PROFILE unpins the endpoint and defers the region to the
/// AWS config chain.
#[test]
fn bedrock_endpoint_resolution_handles_missing_regions_for_profile_matrix() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"]);

    let eu_model = get_model(
        "amazon-bedrock",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("eu model");

    // Explicit profile option: endpoint stays pinned, region derives from it.
    let config = resolve_bedrock_client_config(
        &eu_model,
        BedrockClientOptions {
            profile: Some("bedrock-profile".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(config.profile.as_deref(), Some("bedrock-profile"));
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
    );
    assert_eq!(config.region.as_deref(), Some("eu-central-1"));

    // Request-scoped AWS_PROFILE override: same pinning as an explicit option.
    let config = resolve_bedrock_client_config(
        &eu_model,
        BedrockClientOptions {
            env: BTreeMap::from([(
                "AWS_PROFILE".to_owned(),
                "scoped-bedrock-profile".to_owned(),
            )]),
            ..Default::default()
        },
    );
    assert_eq!(config.profile.as_deref(), Some("scoped-bedrock-profile"));
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
    );
    assert_eq!(config.region.as_deref(), Some("eu-central-1"));

    // Ambient process-env AWS_PROFILE: unpin the endpoint and leave the
    // region to the AWS config chain.
    set_env("AWS_PROFILE", "ambient-bedrock-profile");
    let config = resolve_bedrock_client_config(&eu_model, BedrockClientOptions::default());
    assert_eq!(config.profile.as_deref(), Some("ambient-bedrock-profile"));
    assert_eq!(config.endpoint, None);
    assert_eq!(config.region, None);
}

// ---------------------------------------------------------------------------
// bedrock-custom-headers.test.ts parity (VC1, VC2, VC4)
// ---------------------------------------------------------------------------

fn bedrock_hi_eventstream_body() -> Vec<u8> {
    aws_eventstream_body(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "Hi" }
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
        json!({
            "metadata": {
                "usage": {
                    "inputTokens": 3,
                    "outputTokens": 1,
                    "totalTokens": 4
                }
            }
        }),
    ])
}

fn faux_bedrock_model(base_url: String) -> Model {
    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = base_url;
    model
}

/// VC1: caller headers on the full-options stream path are injected into the
/// outgoing Bedrock request (happy path).
#[tokio::test]
async fn bedrock_stream_forwards_caller_header_happy_path() {
    let (base_url, request_task) = mock_binary_server(
        bedrock_hi_eventstream_body(),
        "application/vnd.amazon.eventstream",
    )
    .await;
    let model = faux_bedrock_model(base_url);
    let mut options = StreamOptions::default();
    options.api_key = Some("bedrock-token".to_owned());
    options.cache_retention = Some(CacheRetention::None);
    options
        .headers
        .insert("x-custom".to_owned(), "v".to_owned());

    let message = complete(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hi"));
    let lower = request.to_ascii_lowercase();
    assert!(lower.contains("x-custom: v"), "{request}");
    assert!(
        lower.contains("authorization: bearer bedrock-token"),
        "{request}"
    );
}

/// VC2: reserved headers (`x-amz-*`, `host`) are skipped case-insensitively
/// while allowed caller headers still apply. Note: unlike pi, ri deliberately
/// lets a caller `authorization` header through as BYO bearer auth, so it is
/// not part of the reserved set here.
#[tokio::test]
async fn bedrock_reserved_headers_skipped_case_insensitively_while_allowed_apply() {
    let (base_url, request_task) = mock_binary_server(
        bedrock_hi_eventstream_body(),
        "application/vnd.amazon.eventstream",
    )
    .await;
    let expected_host = base_url
        .strip_prefix("http://")
        .expect("http base url")
        .to_owned();
    let model = faux_bedrock_model(base_url);
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("bedrock-token".to_owned());
    options.stream.cache_retention = Some(CacheRetention::None);
    options.stream.headers = BTreeMap::from([
        ("x-amz-date".to_owned(), "evil-date".to_owned()),
        ("X-Amz-Security-Token".to_owned(), "evil-token".to_owned()),
        ("HOST".to_owned(), "evil-host".to_owned()),
        ("Host".to_owned(), "evil-host-2".to_owned()),
        ("x-allowed".to_owned(), "ok".to_owned()),
        ("X-Custom-Mixed".to_owned(), "mixed-ok".to_owned()),
    ]);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hi"));
    let lower = request.to_ascii_lowercase();
    // Allowed headers apply regardless of the caller's key casing.
    assert!(lower.contains("x-allowed: ok"), "{request}");
    assert!(lower.contains("x-custom-mixed: mixed-ok"), "{request}");
    // Reserved headers are skipped case-insensitively: no caller value leaks
    // and no capitalised duplicate key is added back.
    assert!(!lower.contains("evil"), "{request}");
    assert!(!lower.contains("x-amz-"), "{request}");
    // The real host header for the target endpoint stays intact.
    assert!(
        lower.contains(&format!("host: {expected_host}")),
        "{request}"
    );
    // Reserved filtering must not disturb the auth header.
    assert!(
        lower.contains("authorization: bearer bedrock-token"),
        "{request}"
    );
}

/// VC4: the simple-stream entry point forwards caller headers end-to-end
/// through a mock Bedrock runtime server (regression guard).
#[tokio::test]
async fn bedrock_complete_simple_forwards_headers_end_to_end() {
    let (base_url, request_task) = mock_binary_server(
        bedrock_hi_eventstream_body(),
        "application/vnd.amazon.eventstream",
    )
    .await;
    let model = faux_bedrock_model(base_url);
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("bedrock-token".to_owned());
    options.stream.cache_retention = Some(CacheRetention::None);
    options
        .stream
        .headers
        .insert("x-custom".to_owned(), "v".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hi"));
    assert_eq!(message.usage.input, 3);
    assert_eq!(message.usage.output, 1);
    let lower = request.to_ascii_lowercase();
    assert!(lower.contains("x-custom: v"), "{request}");
    assert!(
        request.starts_with("POST /model/anthropic.claude-test-v1%3A0/converse-stream HTTP/1.1"),
        "{request}"
    );
}

// ---------------------------------------------------------------------------
// Mock-server helpers (copied from tests/provider_core.rs)
// ---------------------------------------------------------------------------

async fn mock_binary_server(
    body: Vec<u8>,
    content_type: &'static str,
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
        let (mut socket, _) = listener.accept().await.expect("accept request");
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
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len(),
        );
        socket
            .write_all(headers.as_bytes())
            .await
            .expect("write headers");
        socket.write_all(&body).await.expect("write response");
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
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

fn aws_eventstream_body(events: Vec<Value>) -> Vec<u8> {
    events
        .into_iter()
        .flat_map(|event| aws_eventstream_frame(event.to_string().as_bytes()))
        .collect()
}

fn aws_eventstream_frame(payload: &[u8]) -> Vec<u8> {
    let total_len = 16 + payload.len();
    let mut frame = Vec::with_capacity(total_len);
    frame.extend_from_slice(&(total_len as u32).to_be_bytes());
    frame.extend_from_slice(&0_u32.to_be_bytes());
    frame.extend_from_slice(&0_u32.to_be_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&0_u32.to_be_bytes());
    frame
}
