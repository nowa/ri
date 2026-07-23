//! Test-parity audit wave: deferred-tools gaps (pi `test/deferred-tools.test.ts`),
//! pi-messages error paths (pi `test/pi-messages.test.ts`), and positive
//! max_completion_tokens assertions (pi `test/openai-completions-empty-tools.test.ts`).

use ri_llm_provider::*;
use serde_json::{Map, Value, json};

fn deferred_tool(name: &str) -> Tool {
    Tool {
        name: name.to_owned(),
        description: format!("The {name} tool"),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        }),
    }
}

fn assistant_tool_call_message(tool_name: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "call_1".to_owned(),
            name: tool_name.to_owned(),
            arguments: Map::new(),
            thought_signature: None,
        })],
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        model: "claude-opus-4-6".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 2,
    }
}

fn tool_result_message(
    tool_call_id: &str,
    added_tool_names: Vec<String>,
    timestamp: i64,
) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: "base_tool".to_owned(),
        content: vec![ToolResultContent::text("done")],
        details: None,
        usage: None,
        is_error: false,
        added_tool_names: Some(added_tool_names),
        timestamp,
    }
}

fn deferred_context(tools: Vec<Tool>, added_tool_names: Vec<String>) -> Context {
    let mut user = UserMessage::text("Hello");
    user.timestamp = 1;
    let mut tail_user = UserMessage::text("Hello");
    tail_user.timestamp = 4;
    Context {
        system_prompt: None,
        messages: vec![
            Message::User(user),
            Message::Assistant(assistant_tool_call_message("base_tool")),
            Message::ToolResult(tool_result_message("call_1", added_tool_names, 3)),
            Message::User(tail_user),
        ],
        tools,
    }
}

fn anthropic_tool_result_content(payload: &Value) -> Vec<Value> {
    payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter_map(|message| message["content"].as_array())
        .find(|content| content.iter().any(|block| block["type"] == "tool_result"))
        .expect("tool result content")
        .clone()
}

fn anthropic_tool_names(payload: &Value) -> Vec<String> {
    payload["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| tool["name"].as_str().expect("tool name").to_owned())
        .collect()
}

fn tool_result_has_tool_reference(payload: &Value) -> bool {
    anthropic_tool_result_content(payload).iter().any(|block| {
        block["type"] == "tool_reference"
            || block["content"].as_array().is_some_and(|content| {
                content
                    .iter()
                    .any(|inner| inner["type"] == "tool_reference")
            })
    })
}

fn openai_responses_tool_names(payload: &Value) -> Vec<String> {
    payload["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .map(|tool| tool["name"].as_str().expect("tool name").to_owned())
        .collect()
}

fn openai_responses_has_tool_search(payload: &Value) -> bool {
    payload["input"].as_array().is_some_and(|input| {
        input
            .iter()
            .any(|item| item["type"] == "tool_search_call" || item["type"] == "tool_search_output")
    })
}

// --- deferred tools -------------------------------------------------------

#[test]
fn anthropic_loads_tool_introduced_by_openai_history_after_switching_providers() {
    // The deferral marker was recorded while an OpenAI model ran the tool
    // batch; the Anthropic target must still honor it.
    let mut context = deferred_context(
        vec![deferred_tool("base_tool"), deferred_tool("late_tool")],
        vec!["late_tool".to_owned()],
    );
    let Message::Assistant(assistant) = &mut context.messages[1] else {
        panic!("expected assistant");
    };
    assistant.api = "openai-responses".to_owned();
    assistant.provider = "openai".to_owned();
    assistant.model = "gpt-5.4".to_owned();

    let model = get_model("anthropic", "claude-opus-4-8").expect("model");
    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());

    let tools = payload["tools"].as_array().expect("tools");
    assert_eq!(
        anthropic_tool_names(&payload),
        vec!["base_tool", "late_tool"]
    );
    assert!(tools[0].get("defer_loading").is_none());
    assert_eq!(tools[1]["defer_loading"], json!(true));

    let content = anthropic_tool_result_content(&payload);
    let tool_result = content
        .iter()
        .find(|block| block["type"] == "tool_result")
        .expect("tool result");
    assert_eq!(
        tool_result["content"],
        json!([{ "type": "tool_reference", "tool_name": "late_tool" }])
    );
}

#[test]
fn anthropic_oauth_normalizes_names_before_checking_prior_tool_usage() {
    // The assistant used "Read"; the marker says "read". Under OAuth
    // canonicalization both normalize to "Read", so the tool was already used
    // before its marker and stays immediate.
    let mut context = deferred_context(
        vec![deferred_tool("base_tool"), deferred_tool("read")],
        vec!["read".to_owned()],
    );
    let Message::Assistant(assistant) = &mut context.messages[1] else {
        panic!("expected assistant");
    };
    assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "call_1".to_owned(),
        name: "Read".to_owned(),
        arguments: Map::new(),
        thought_signature: None,
    })];

    let model = get_model("anthropic", "claude-opus-4-6").expect("model");
    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            use_claude_code_tool_names: true,
            ..Default::default()
        },
    );

    assert_eq!(anthropic_tool_names(&payload), vec!["base_tool", "Read"]);
    assert!(
        payload["tools"]
            .as_array()
            .expect("tools")
            .iter()
            .all(|tool| tool.get("defer_loading").is_none())
    );
    assert!(!tool_result_has_tool_reference(&payload));
}

#[test]
fn anthropic_oauth_canonicalized_markers_match_active_tools() {
    // The marker was recorded as "Read" but the active tool is "read"; the
    // OAuth normalizer must match them and defer the tool.
    let context = deferred_context(
        vec![deferred_tool("base_tool"), deferred_tool("read")],
        vec!["Read".to_owned()],
    );
    let model = get_model("anthropic", "claude-opus-4-6").expect("model");
    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            use_claude_code_tool_names: true,
            ..Default::default()
        },
    );

    let tools = payload["tools"].as_array().expect("tools");
    assert_eq!(anthropic_tool_names(&payload), vec!["base_tool", "Read"]);
    assert!(tools[0].get("defer_loading").is_none());
    assert_eq!(tools[1]["defer_loading"], json!(true));
    let content = anthropic_tool_result_content(&payload);
    assert!(content.iter().any(|block| {
        block["type"] == "tool_result"
            && block["content"].as_array().is_some_and(|inner| {
                inner.iter().any(|reference| {
                    reference["type"] == "tool_reference" && reference["tool_name"] == "Read"
                })
            })
    }));
}

#[test]
fn anthropic_oauth_canonicalization_deduplicates_active_tools() {
    // "read" and "Read" normalize to the same name; the last definition wins.
    let mut user = UserMessage::text("Hello");
    user.timestamp = 1;
    let mut canonical = deferred_tool("Read");
    canonical.description = "Canonical definition".to_owned();
    let context = Context {
        system_prompt: None,
        messages: vec![Message::User(user)],
        tools: vec![deferred_tool("read"), canonical],
    };
    let model = get_model("anthropic", "claude-opus-4-6").expect("model");
    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            use_claude_code_tool_names: true,
            ..Default::default()
        },
    );

    let tools = payload["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["name"], json!("Read"));
    assert_eq!(tools[0]["description"], json!("Canonical definition"));
}

#[test]
fn anthropic_supports_explicit_tool_reference_compat_override_on_other_providers() {
    // supportsToolReferences=true enables deferral even though the provider
    // check would default to false for a non-anthropic provider.
    let mut model = get_model("anthropic", "claude-opus-4-6").expect("model");
    model.provider = "anthropic-proxy".to_owned();
    model.compat = Some(json!({ "supportsToolReferences": true }));
    let context = deferred_context(
        vec![deferred_tool("base_tool"), deferred_tool("late_tool")],
        vec!["late_tool".to_owned()],
    );
    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());

    let late_tool = payload["tools"]
        .as_array()
        .expect("tools")
        .iter()
        .find(|tool| tool["name"] == "late_tool")
        .expect("late tool");
    assert_eq!(late_tool["defer_loading"], json!(true));
}

#[test]
fn kimi_emits_deferred_schemas_after_all_tool_results_in_a_batch() {
    let mut model = Model::faux("openai-completions", "moonshotai", "deferred-tools-model");
    model.compat = Some(json!({ "deferredToolsMode": "kimi" }));
    let mut context = deferred_context(
        vec![
            deferred_tool("base_tool"),
            deferred_tool("late_tool"),
            deferred_tool("later_tool"),
        ],
        vec!["late_tool".to_owned()],
    );
    context.messages.insert(
        3,
        Message::ToolResult(tool_result_message(
            "call_2",
            vec!["later_tool".to_owned()],
            3,
        )),
    );

    let messages = convert_openai_completions_messages(&model, &context);

    let roles = messages
        .iter()
        .map(|message| message["role"].as_str().expect("role").to_owned())
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        vec!["user", "assistant", "tool", "tool", "system", "user"]
    );
    // The system tools message follows ALL results and carries both late tools.
    let system_tools = messages[4]["tools"]
        .as_array()
        .expect("system tools")
        .iter()
        .map(|tool| tool["function"]["name"].as_str().expect("name").to_owned())
        .collect::<Vec<_>>();
    assert_eq!(system_tools, vec!["late_tool", "later_tool"]);
}

#[test]
fn openai_unsupported_catalog_models_use_the_normal_tool_list() {
    for model_id in ["gpt-5.2", "gpt-5.4-nano", "gpt-5.5-pro"] {
        let model = get_model("openai", model_id).expect("catalog model");
        let context = deferred_context(
            vec![deferred_tool("base_tool"), deferred_tool("late_tool")],
            vec!["late_tool".to_owned()],
        );
        let payload = build_openai_responses_payload(
            &model,
            &context,
            OpenAIResponsesPayloadOptions::default(),
        );

        assert_eq!(
            openai_responses_tool_names(&payload),
            vec!["base_tool", "late_tool"],
            "{model_id}"
        );
        assert!(!openai_responses_has_tool_search(&payload), "{model_id}");
    }
}

#[test]
fn openai_explicitly_disabled_tool_search_uses_the_normal_tool_list() {
    // supportsToolSearch=false overrides a supporting model.
    let mut model = get_model("openai", "gpt-5.4").expect("model");
    model.provider = "openai-proxy".to_owned();
    model.compat = Some(json!({ "supportsToolSearch": false }));
    let context = deferred_context(
        vec![deferred_tool("base_tool"), deferred_tool("late_tool")],
        vec!["late_tool".to_owned()],
    );
    let payload =
        build_openai_responses_payload(&model, &context, OpenAIResponsesPayloadOptions::default());

    assert_eq!(
        openai_responses_tool_names(&payload),
        vec!["base_tool", "late_tool"]
    );
    assert!(!openai_responses_has_tool_search(&payload));
}

// --- pi-messages error paths ----------------------------------------------

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

async fn mock_sse_server(body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
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
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

#[tokio::test(flavor = "current_thread")]
async fn pi_messages_propagates_server_sent_error_events() {
    let sse = concat!(
        "data: {\"type\":\"start\"}\n\n",
        "data: {\"type\":\"error\",\"reason\":\"error\",\"usage\":{\"input\":10,\"output\":5,\"cacheRead\":0,\"cacheWrite\":0,\"totalTokens\":15,\"cost\":{\"input\":0.1,\"output\":0.2,\"cacheRead\":0,\"cacheWrite\":0,\"total\":0.3}},\"errorMessage\":\"Upstream failed\"}\n\n",
    );
    let (url, _request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("pi-messages", "radius", "radius-model");
    model.base_url = url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("stream result");

    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("Upstream failed"));
    assert_eq!(message.usage.input, 10);
    assert_eq!(message.usage.output, 5);
    assert_eq!(message.usage.total_tokens, 15);
    assert!((message.usage.cost.total - 0.3).abs() < 1e-12);
}

#[tokio::test(flavor = "current_thread")]
async fn pi_messages_errors_when_no_api_key_is_provided() {
    let mut model = Model::faux("pi-messages", "radius", "radius-model");
    model.base_url = "http://127.0.0.1:9".to_owned();

    let error = complete_simple(
        &model,
        user_context("hello"),
        SimpleStreamOptions::default(),
    )
    .await
    .expect_err("must fail without an api key");

    assert!(error.to_string().contains("No API key provided"), "{error}");
}

#[tokio::test(flavor = "current_thread")]
async fn pi_messages_errors_when_the_stream_ends_without_a_terminal_event() {
    let sse = concat!(
        "data: {\"type\":\"start\"}\n\n",
        "data: {\"type\":\"text_start\",\"contentIndex\":0}\n\n",
        "data: {\"type\":\"text_delta\",\"contentIndex\":0,\"delta\":\"partial\"}\n\n",
    );
    let (url, _request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("pi-messages", "radius", "radius-model");
    model.base_url = url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("stream result");

    assert_eq!(message.stop_reason, StopReason::Error);
    assert!(
        message
            .error_message
            .as_deref()
            .is_some_and(|error| error.contains("stream ended without a terminal event")),
        "{:?}",
        message.error_message
    );
}

// --- openai-completions max tokens ----------------------------------------

const COMPLETIONS_SSE: &str = concat!(
    "data: {\"id\":\"chatcmpl_audit\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}}\n\n",
    "data: [DONE]\n\n",
);

#[tokio::test(flavor = "current_thread")]
async fn openai_completions_sends_default_max_tokens_as_max_completion_tokens() {
    let (url, request_task) = mock_sse_server(COMPLETIONS_SSE).await;
    let mut model = Model::faux("openai-completions", "audit-openai", "audit-model");
    model.base_url = url;
    assert_eq!(model.max_tokens, 16_384);
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = complete_simple(&model, user_context("hi"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("ok"));
    // model.maxTokens lands in max_completion_tokens, never max_tokens.
    assert!(
        request.contains("\"max_completion_tokens\":16384"),
        "{request}"
    );
    assert!(!request.contains("\"max_tokens\""), "{request}");
}

#[tokio::test(flavor = "current_thread")]
async fn openai_completions_sends_explicit_max_tokens_as_max_completion_tokens() {
    let (url, request_task) = mock_sse_server(COMPLETIONS_SSE).await;
    let mut model = Model::faux("openai-completions", "audit-openai", "audit-model");
    model.base_url = url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.max_tokens = Some(1234);

    let message = complete_simple(&model, user_context("hi"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("ok"));
    assert!(
        request.contains("\"max_completion_tokens\":1234"),
        "{request}"
    );
    assert!(!request.contains("\"max_tokens\""), "{request}");
}

#[tokio::test(flavor = "current_thread")]
async fn openai_completions_clamps_explicit_max_tokens_to_remaining_context() {
    let (url, request_task) = mock_sse_server(COMPLETIONS_SSE).await;
    let mut model = Model::faux("openai-completions", "audit-openai", "audit-model");
    model.base_url = url;
    model.context_window = 10_000;
    model.max_tokens = 8_000;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.max_tokens = Some(7_000);

    // 8000 chars estimate to 2000 tokens; the explicit 7000 exceeds the
    // remaining budget 10000 - 2000 - 4096 = 3904 and is clamped.
    let message = complete_simple(&model, user_context(&"x".repeat(8_000)), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("ok"));
    assert!(
        request.contains("\"max_completion_tokens\":3904"),
        "{request}"
    );
    assert!(!request.contains("\"max_tokens\""), "{request}");
}
