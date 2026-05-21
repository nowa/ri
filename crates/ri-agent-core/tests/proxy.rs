use futures::StreamExt;
use ri_agent_core::*;
use ri_llm_provider::{
    AssistantContent, AssistantMessageEvent, CacheRetention, Context, Model, SimpleStreamOptions,
    StopReason, ThinkingLevel, Transport, UserMessage,
};
use serde_json::{Value, json};
use std::sync::Arc;

#[tokio::test]
async fn stream_proxy_posts_serializable_options_and_rebuilds_partial_events() {
    let events = vec![
        json!({ "type": "start" }),
        json!({ "type": "text_start", "contentIndex": 0 }),
        json!({ "type": "text_delta", "contentIndex": 0, "delta": "Hel" }),
        json!({ "type": "text_delta", "contentIndex": 0, "delta": "lo" }),
        json!({ "type": "text_end", "contentIndex": 0, "contentSignature": "sig_text" }),
        json!({ "type": "thinking_start", "contentIndex": 1 }),
        json!({ "type": "thinking_delta", "contentIndex": 1, "delta": "plan" }),
        json!({ "type": "thinking_end", "contentIndex": 1, "contentSignature": "sig_think" }),
        json!({ "type": "toolcall_start", "contentIndex": 2, "id": "call_1", "toolName": "calc" }),
        json!({ "type": "toolcall_delta", "contentIndex": 2, "delta": "{\"x\":" }),
        json!({ "type": "toolcall_delta", "contentIndex": 2, "delta": "2}" }),
        json!({ "type": "toolcall_end", "contentIndex": 2 }),
        json!({ "type": "toolcall_end", "contentIndex": 9 }),
        json!({ "type": "done", "reason": "toolUse", "usage": usage_json(5, 3) }),
    ];
    let (proxy_url, request_task) = mock_proxy_sse_server(events, 200, None).await;
    let model = Model::faux("proxy-test-api", "proxy-provider", "proxy-model");
    let context = Context {
        messages: vec![ri_llm_provider::Message::User(UserMessage::text("hello"))],
        ..Default::default()
    };
    let mut options = ProxyStreamOptions::new(proxy_url, "proxy-token");
    options.stream_options = proxy_stream_options();

    let events = collect_events(stream_proxy(model, context, options)).await;
    let request = request_task.await.expect("request task");

    assert!(request.starts_with("POST /api/stream HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer proxy-token")
    );
    let body = request_body_json(&request);
    assert_eq!(body["model"]["id"], "proxy-model");
    assert_eq!(body["context"]["messages"][0]["content"], "hello");
    assert_eq!(body["options"]["temperature"], 0.25);
    assert_eq!(body["options"]["maxTokens"], 128);
    assert_eq!(body["options"]["reasoning"], "high");
    assert_eq!(body["options"]["cacheRetention"], "long");
    assert_eq!(body["options"]["sessionId"], "proxy-session");
    assert_eq!(body["options"]["transport"], "websocket-cached");
    assert_eq!(body["options"]["thinkingBudgets"]["high"], 4096);
    assert_eq!(body["options"]["maxRetryDelayMs"], 50);
    assert_eq!(body["options"]["headers"]["x-proxy"], "yes");
    assert_eq!(body["options"]["metadata"]["trace"], "abc");
    assert!(body["options"].get("apiKey").is_none());
    assert!(body["options"].get("maxRetries").is_none());

    assert!(matches!(events[0], AssistantMessageEvent::Start { .. }));
    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel")
    ));
    let tool_call = events.iter().find_map(|event| match event {
        AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call),
        _ => None,
    });
    assert_eq!(tool_call.expect("tool call").arguments["x"], 2);
    let done = events.iter().find_map(|event| match event {
        AssistantMessageEvent::Done { reason, message } => Some((reason, message)),
        _ => None,
    });
    let (reason, message) = done.expect("done event");
    assert_eq!(*reason, StopReason::ToolUse);
    assert_eq!(message.usage.input, 5);
    match &message.content[0] {
        AssistantContent::Text(text) => {
            assert_eq!(text.text, "Hello");
            assert_eq!(text.text_signature.as_deref(), Some("sig_text"));
        }
        other => panic!("expected text, got {other:?}"),
    }
    match &message.content[1] {
        AssistantContent::Thinking(thinking) => {
            assert_eq!(thinking.thinking, "plan");
            assert_eq!(thinking.thinking_signature.as_deref(), Some("sig_think"));
        }
        other => panic!("expected thinking, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_proxy_maps_proxy_event_reconstruction_errors_to_error_events() {
    let events = vec![json!({ "type": "text_delta", "contentIndex": 0, "delta": "orphan" })];
    let (proxy_url, request_task) = mock_proxy_sse_server(events, 200, None).await;
    let model = Model::faux("proxy-test-api", "proxy-provider", "proxy-model");
    let options = ProxyStreamOptions::new(proxy_url, "proxy-token");

    let events = collect_events(stream_proxy(model, Context::default(), options)).await;
    request_task.await.expect("request task");

    assert_eq!(events.len(), 1);
    match &events[0] {
        AssistantMessageEvent::Error { reason, error } => {
            assert_eq!(*reason, StopReason::Error);
            assert_eq!(
                error.error_message.as_deref(),
                Some("Received text_delta for non-text content")
            );
        }
        event => panic!("expected proxy reconstruction error, got {event:?}"),
    }
}

#[tokio::test]
async fn agent_uses_custom_proxy_stream_provider() {
    let events = vec![
        json!({ "type": "start" }),
        json!({ "type": "text_start", "contentIndex": 0 }),
        json!({ "type": "text_delta", "contentIndex": 0, "delta": "Proxy agent" }),
        json!({ "type": "text_end", "contentIndex": 0 }),
        json!({ "type": "done", "reason": "stop", "usage": usage_json(2, 2) }),
    ];
    let (proxy_url, request_task) = mock_proxy_sse_server(events, 200, None).await;
    let model = Model::faux("proxy-agent-api", "proxy-provider", "proxy-agent-model");
    let mut options = AgentOptions::new(model);
    options.stream_provider = Some(Arc::new(ProxyStreamProvider::new(proxy_url, "agent-token")));
    options.stream_options.stream.session_id = Some("agent-proxy-session".to_owned());

    let agent = Agent::new(options);
    agent.prompt("hello").await.expect("prompt");
    let request = request_task.await.expect("request task");
    let state = agent.state();

    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer agent-token")
    );
    assert_eq!(state.messages.len(), 2);
    let assistant = match &state.messages[1] {
        AgentMessage::Assistant(message) => message,
        other => panic!("expected assistant, got {other:?}"),
    };
    match &assistant.content[0] {
        AssistantContent::Text(text) => assert_eq!(text.text, "Proxy agent"),
        other => panic!("expected text, got {other:?}"),
    }
}

async fn collect_events(
    mut stream: ri_llm_provider::AssistantMessageEventStream,
) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

fn proxy_stream_options() -> SimpleStreamOptions {
    let mut options = SimpleStreamOptions::default();
    options.stream.temperature = Some(0.25);
    options.stream.max_tokens = Some(128);
    options.reasoning = Some(ThinkingLevel::High);
    options.stream.cache_retention = Some(CacheRetention::Long);
    options.stream.session_id = Some("proxy-session".to_owned());
    options.stream.transport = Some(Transport::WebsocketCached);
    options.stream.max_retry_delay_ms = Some(50);
    options
        .stream
        .headers
        .insert("x-proxy".to_owned(), "yes".to_owned());
    options
        .stream
        .metadata
        .insert("trace".to_owned(), json!("abc"));
    options.thinking_budgets = Some(ri_llm_provider::ThinkingBudgets {
        high: Some(4096),
        ..Default::default()
    });
    options.stream.api_key = Some("must-not-cross-proxy".to_owned());
    options.stream.max_retries = Some(9);
    options
}

fn usage_json(input: u64, output: u64) -> Value {
    json!({
        "input": input,
        "output": output,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": input + output,
        "cost": { "input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0 }
    })
}

async fn mock_proxy_sse_server(
    events: Vec<Value>,
    status: u16,
    error_body: Option<Value>,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::net::TcpListener;

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept proxy request");
        let request = read_http_request(&mut socket).await;
        if status == 200 {
            let body = events
                .into_iter()
                .map(|event| format!("data: {}\n\n", event))
                .collect::<String>();
            write_http_response(&mut socket, status, "OK", "text/event-stream", &body).await;
        } else {
            let body = error_body
                .unwrap_or_else(|| json!({ "error": "failed" }))
                .to_string();
            write_http_response(
                &mut socket,
                status,
                "Bad Gateway",
                "application/json",
                &body,
            )
            .await;
        }
        request
    });
    (format!("http://{addr}"), task)
}

async fn read_http_request(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;

    let mut request = Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        let n = socket.read(&mut buf).await.expect("read request");
        assert_ne!(n, 0, "request ended before headers");
        request.extend_from_slice(&buf[..n]);
        if request.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
    }
    let header_end = request
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .expect("header end")
        + 4;
    let headers = String::from_utf8_lossy(&request[..header_end]).into_owned();
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|value| value.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    let already_read = request.len() - header_end;
    if already_read < content_length {
        let mut body = vec![0u8; content_length - already_read];
        socket.read_exact(&mut body).await.expect("read body");
        request.extend_from_slice(&body);
    }
    String::from_utf8(request).expect("request utf8")
}

async fn write_http_response(
    socket: &mut tokio::net::TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &str,
) {
    use tokio::io::AsyncWriteExt;

    let response = format!(
        "HTTP/1.1 {status} {reason}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    );
    socket
        .write_all(response.as_bytes())
        .await
        .expect("write response");
}

fn request_body_json(request: &str) -> Value {
    let body = request
        .split("\r\n\r\n")
        .nth(1)
        .expect("request body")
        .trim_matches(char::from(0));
    serde_json::from_str(body).expect("request json")
}
