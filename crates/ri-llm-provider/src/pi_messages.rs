//! pi-messages API implementation (pi `api/pi-messages.ts`).
//!
//! Streams pi's own message protocol directly to a backend: the request is a
//! single POST of `{ model, context, options }` to `<base_url>/messages`, the
//! response is an SSE stream of serialized assistant-message events plus a
//! terminal `done`/`error` event. This is the wire protocol spoken by the
//! Radius gateway, but any backend implementing it can be used.

use crate::types::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions,
    StopReason, TextContent, ThinkingContent, ToolCall, Usage, now_millis,
};
use crate::{AssistantMessageEventSender, parse_json_with_repair};
use serde_json::{Map, Value, json};

/// Build the pi-messages request payload.
pub fn build_pi_messages_payload(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> Value {
    let mut request_options = Map::new();
    if let Some(temperature) = options.stream.temperature {
        request_options.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(max_tokens) = options.stream.max_tokens {
        request_options.insert("maxTokens".to_owned(), json!(max_tokens));
    }
    if let Some(reasoning) = options.reasoning {
        request_options.insert(
            "reasoning".to_owned(),
            serde_json::to_value(reasoning).unwrap_or(Value::Null),
        );
    }
    let cache_retention = options.stream.cache_retention.or_else(|| {
        // Backend defaults apply when unset; only the legacy env opt-in maps.
        (crate::get_provider_env_value("PI_CACHE_RETENTION", &options.stream.env).as_deref()
            == Some("long"))
        .then_some(crate::types::CacheRetention::Long)
    });
    if let Some(cache_retention) = cache_retention {
        request_options.insert(
            "cacheRetention".to_owned(),
            serde_json::to_value(cache_retention).unwrap_or(Value::Null),
        );
    }
    if let Some(session_id) = &options.stream.session_id {
        request_options.insert("sessionId".to_owned(), json!(session_id));
    }
    if let Some(tool_choice) = options.stream.extra.get("toolChoice") {
        request_options.insert("toolChoice".to_owned(), tool_choice.clone());
    }
    json!({
        "model": model.id,
        "context": context,
        "options": Value::Object(request_options),
    })
}

/// Request URL for a pi-messages backend.
pub fn pi_messages_url(model: &Model, debug: bool) -> String {
    let base = model.base_url.trim_end_matches('/');
    if debug {
        format!("{base}/messages?debug=1")
    } else {
        format!("{base}/messages")
    }
}

/// Streaming converter from wire events to assistant-message events.
#[derive(Debug, Default)]
pub struct PiMessagesStreamProcessor {
    tool_json: std::collections::BTreeMap<usize, String>,
    terminal: bool,
}

fn set_content(partial: &mut AssistantMessage, index: usize, content: AssistantContent) {
    if partial.content.len() <= index {
        partial
            .content
            .resize(index + 1, AssistantContent::Text(TextContent::new("")));
    }
    partial.content[index] = content;
}

impl PiMessagesStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// Apply one wire event. Terminal events push `Done`/`Error` themselves.
    pub fn process_event(
        &mut self,
        event: Value,
        partial: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let index = event
            .get("contentIndex")
            .and_then(Value::as_u64)
            .unwrap_or_default() as usize;
        let text_of = |field: &str| {
            event
                .get(field)
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        match event_type {
            "start" => sender.push(AssistantMessageEvent::Start {
                partial: partial.clone(),
            }),
            "text_start" => {
                set_content(partial, index, AssistantContent::Text(TextContent::new("")));
                sender.push(AssistantMessageEvent::TextStart {
                    content_index: index,
                    partial: partial.clone(),
                });
            }
            "text_delta" => {
                if let Some(AssistantContent::Text(text)) = partial.content.get_mut(index) {
                    text.text.push_str(&text_of("delta"));
                }
                sender.push(AssistantMessageEvent::TextDelta {
                    content_index: index,
                    delta: text_of("delta"),
                    partial: partial.clone(),
                });
            }
            "text_end" => {
                let content = text_of("content");
                set_content(
                    partial,
                    index,
                    AssistantContent::Text(TextContent {
                        text: content.clone(),
                        text_signature: event
                            .get("contentSignature")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                    }),
                );
                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content,
                    partial: partial.clone(),
                });
            }
            "thinking_start" => {
                set_content(
                    partial,
                    index,
                    AssistantContent::Thinking(ThinkingContent::new("")),
                );
                sender.push(AssistantMessageEvent::ThinkingStart {
                    content_index: index,
                    partial: partial.clone(),
                });
            }
            "thinking_delta" => {
                if let Some(AssistantContent::Thinking(thinking)) = partial.content.get_mut(index) {
                    thinking.thinking.push_str(&text_of("delta"));
                }
                sender.push(AssistantMessageEvent::ThinkingDelta {
                    content_index: index,
                    delta: text_of("delta"),
                    partial: partial.clone(),
                });
            }
            "thinking_end" => {
                let content = text_of("content");
                set_content(
                    partial,
                    index,
                    AssistantContent::Thinking(ThinkingContent {
                        thinking: content.clone(),
                        thinking_signature: event
                            .get("contentSignature")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned),
                        redacted: event
                            .get("redacted")
                            .and_then(Value::as_bool)
                            .unwrap_or(false),
                    }),
                );
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content,
                    partial: partial.clone(),
                });
            }
            "toolcall_start" => {
                set_content(
                    partial,
                    index,
                    AssistantContent::ToolCall(ToolCall {
                        id: text_of("id"),
                        name: text_of("toolName"),
                        arguments: Map::new(),
                        thought_signature: None,
                    }),
                );
                self.tool_json.insert(index, String::new());
                sender.push(AssistantMessageEvent::ToolcallStart {
                    content_index: index,
                    partial: partial.clone(),
                });
            }
            "toolcall_delta" => {
                let buffer = self.tool_json.entry(index).or_default();
                buffer.push_str(&text_of("delta"));
                let parsed = parse_json_with_repair::<Value>(buffer)
                    .ok()
                    .and_then(|value| value.as_object().cloned())
                    .unwrap_or_default();
                if let Some(AssistantContent::ToolCall(tool_call)) = partial.content.get_mut(index)
                {
                    tool_call.arguments = parsed;
                }
                sender.push(AssistantMessageEvent::ToolcallDelta {
                    content_index: index,
                    delta: text_of("delta"),
                    partial: partial.clone(),
                });
            }
            "toolcall_end" => {
                if let Ok(tool_call) =
                    serde_json::from_value::<ToolCall>(event.get("toolCall").cloned().into())
                {
                    set_content(
                        partial,
                        index,
                        AssistantContent::ToolCall(tool_call.clone()),
                    );
                    sender.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: index,
                        tool_call,
                        partial: partial.clone(),
                    });
                }
            }
            "done" | "error" => {
                self.terminal = true;
                if let Some(usage) = event.get("usage")
                    && let Ok(usage) = serde_json::from_value::<Usage>(usage.clone())
                {
                    partial.usage = usage;
                }
                partial.response_id = event
                    .get("responseId")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned);
                if let Some(rewrite) = event.get("rewrite") {
                    partial.diagnostics.push(json!({
                        "type": "pi_messages_rewrite",
                        "timestamp": now_millis(),
                        "details": rewrite,
                    }));
                }
                if event_type == "done" {
                    partial.stop_reason = match event.get("reason").and_then(Value::as_str) {
                        Some("length") => StopReason::Length,
                        Some("toolUse") => StopReason::ToolUse,
                        _ => StopReason::Stop,
                    };
                    sender.push(AssistantMessageEvent::Done {
                        reason: partial.stop_reason,
                        message: partial.clone(),
                    });
                } else {
                    partial.stop_reason = match event.get("reason").and_then(Value::as_str) {
                        Some("aborted") => StopReason::Aborted,
                        _ => StopReason::Error,
                    };
                    partial.error_message = event
                        .get("errorMessage")
                        .and_then(Value::as_str)
                        .map(ToOwned::to_owned);
                    sender.push(AssistantMessageEvent::Error {
                        reason: partial.stop_reason,
                        error: partial.clone(),
                    });
                }
            }
            _ => {}
        }
    }
}

/// Maximum characters kept when embedding a raw response body in a
/// diagnostic (pi `pi-messages.ts` `truncateDiagnosticString`).
const MAX_PI_MESSAGES_DIAGNOSTIC_CHARS: usize = 8192;

/// Parse a response body like pi's `parsePiMessagesErrorBody`: the body must
/// be JSON whose `error` field is a non-null, non-array object.
fn parse_pi_messages_error_body(body: &str) -> Option<Value> {
    let parsed = serde_json::from_str::<Value>(body).ok()?;
    parsed
        .get("error")
        .is_some_and(|error| error.is_object())
        .then_some(parsed)
}

fn truncate_pi_messages_diagnostic_string(value: &str) -> String {
    if value.chars().count() <= MAX_PI_MESSAGES_DIAGNOSTIC_CHARS {
        return value.to_owned();
    }
    let mut truncated = value
        .chars()
        .take(MAX_PI_MESSAGES_DIAGNOSTIC_CHARS)
        .collect::<String>();
    truncated.push('\u{2026}');
    truncated
}

fn format_pi_messages_response_error(
    status: u16,
    status_text: &str,
    body: &str,
    error_body: Option<&Value>,
) -> String {
    let error = error_body.and_then(|value| value.get("error"));
    let message = error
        .and_then(|error| error.get("message"))
        .and_then(Value::as_str);
    let code = error
        .and_then(|error| error.get("code"))
        .and_then(Value::as_str);
    let suffix = message.unwrap_or(body);
    match code {
        Some(code) => format!("{status} {status_text}: {suffix} ({code})"),
        None => format!("{status} {status_text}: {suffix}"),
    }
}

/// Format an HTTP error body like pi's `PiMessagesResponseError`.
pub fn pi_messages_response_error_message(status: u16, status_text: &str, body: &str) -> String {
    let error_body = parse_pi_messages_error_body(body);
    format_pi_messages_response_error(status, status_text, body, error_body.as_ref())
}

/// Response-failure diagnostic matching pi's `createPiMessagesResponseError`
/// details (`version: 1`) recorded under the `pi_messages_response_failure`
/// marker by `createErrorEvent`.
pub fn pi_messages_response_failure_diagnostic(
    model: &Model,
    url: &str,
    status: u16,
    status_text: &str,
    body: &str,
) -> Value {
    let error_body = parse_pi_messages_error_body(body);
    let message = format_pi_messages_response_error(status, status_text, body, error_body.as_ref());
    let code = error_body
        .as_ref()
        .and_then(|value| value.pointer("/error/code"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);

    let mut error_info = Map::new();
    error_info.insert(
        "name".to_owned(),
        Value::String("PiMessagesResponseError".to_owned()),
    );
    error_info.insert("message".to_owned(), Value::String(message));
    if let Some(code) = code {
        error_info.insert("code".to_owned(), Value::String(code));
    }

    let mut details = Map::new();
    details.insert("version".to_owned(), json!(1));
    details.insert("provider".to_owned(), json!(model.provider));
    details.insert("model".to_owned(), json!(model.id));
    details.insert("url".to_owned(), json!(url));
    details.insert("status".to_owned(), json!(status));
    details.insert("statusText".to_owned(), json!(status_text));
    match &error_body {
        Some(error_body) => {
            details.insert(
                "error".to_owned(),
                error_body.get("error").cloned().unwrap_or(Value::Null),
            );
        }
        None => {
            details.insert(
                "body".to_owned(),
                Value::String(truncate_pi_messages_diagnostic_string(body)),
            );
        }
    }
    details.insert("timestampMs".to_owned(), json!(now_millis()));

    json!({
        "type": "pi_messages_response_failure",
        "timestamp": now_millis(),
        "error": Value::Object(error_info),
        "details": Value::Object(details),
    })
}

/// Record the response-failure diagnostic on the assistant message and return
/// the formatted error message (pi surfaces both via `createErrorEvent`).
pub fn append_pi_messages_response_failure(
    output: &mut AssistantMessage,
    model: &Model,
    url: &str,
    status: u16,
    status_text: &str,
    body: &str,
) -> String {
    output
        .diagnostics
        .push(pi_messages_response_failure_diagnostic(
            model,
            url,
            status,
            status_text,
            body,
        ));
    pi_messages_response_error_message(status, status_text, body)
}
