use crate::types::AgentStreamProvider;
use futures::StreamExt;
use ri_llm_provider::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    CacheRetention, Context, Model, SimpleStreamOptions, StopReason, TextContent, ThinkingBudgets,
    ThinkingLevel, ToolCall, Transport, Usage, assistant_message_event_stream,
    json_repair::parse_streaming_json, now_millis,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

#[derive(Debug, Clone)]
pub struct ProxyStreamOptions {
    pub auth_token: String,
    pub proxy_url: String,
    pub stream_options: SimpleStreamOptions,
}

impl ProxyStreamOptions {
    pub fn new(proxy_url: impl Into<String>, auth_token: impl Into<String>) -> Self {
        Self {
            proxy_url: proxy_url.into(),
            auth_token: auth_token.into(),
            stream_options: SimpleStreamOptions::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProxyStreamProvider {
    pub proxy_url: String,
    pub auth_token: String,
}

impl ProxyStreamProvider {
    pub fn new(proxy_url: impl Into<String>, auth_token: impl Into<String>) -> Self {
        Self {
            proxy_url: proxy_url.into(),
            auth_token: auth_token.into(),
        }
    }
}

impl AgentStreamProvider for ProxyStreamProvider {
    fn stream(
        &self,
        model: &Model,
        context: Context,
        mut options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, String> {
        let proxy_options = ProxyStreamOptions {
            auth_token: self.auth_token.clone(),
            proxy_url: self.proxy_url.clone(),
            stream_options: std::mem::take(&mut options),
        };
        Ok(stream_proxy(model.clone(), context, proxy_options))
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxyRequest<'a> {
    model: &'a Model,
    context: &'a Context,
    options: ProxySerializableStreamOptions,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProxySerializableStreamOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<ThinkingLevel>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_retention: Option<CacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    headers: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    metadata: Map<String, Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    transport: Option<Transport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budgets: Option<ThinkingBudgets>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_retry_delay_ms: Option<u64>,
}

impl From<&SimpleStreamOptions> for ProxySerializableStreamOptions {
    fn from(options: &SimpleStreamOptions) -> Self {
        Self {
            temperature: options.stream.temperature,
            max_tokens: options.stream.max_tokens,
            reasoning: options.reasoning,
            cache_retention: options.stream.cache_retention,
            session_id: options.stream.session_id.clone(),
            headers: options.stream.headers.clone(),
            metadata: options.stream.metadata.clone(),
            transport: options.stream.transport,
            thinking_budgets: options.thinking_budgets.clone(),
            max_retry_delay_ms: options.stream.max_retry_delay_ms,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ProxyAssistantMessageEvent {
    Start,
    TextStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    TextDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    TextEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "contentSignature")]
        content_signature: Option<String>,
    },
    ThinkingStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    ThinkingDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    ThinkingEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        #[serde(rename = "contentSignature")]
        content_signature: Option<String>,
    },
    ToolcallStart {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        id: String,
        #[serde(rename = "toolName")]
        tool_name: String,
    },
    ToolcallDelta {
        #[serde(rename = "contentIndex")]
        content_index: usize,
        delta: String,
    },
    ToolcallEnd {
        #[serde(rename = "contentIndex")]
        content_index: usize,
    },
    Done {
        reason: StopReason,
        usage: Usage,
    },
    Error {
        reason: StopReason,
        #[serde(rename = "errorMessage")]
        error_message: Option<String>,
        usage: Usage,
    },
}

pub fn stream_proxy(
    model: Model,
    context: Context,
    options: ProxyStreamOptions,
) -> AssistantMessageEventStream {
    let (sender, stream) = assistant_message_event_stream();
    tokio::spawn(async move {
        let mut partial = empty_proxy_assistant_message(&model);
        let mut tool_partial_json = BTreeMap::<usize, String>::new();
        let result = run_proxy_request(
            &model,
            &context,
            &options,
            &mut partial,
            &mut tool_partial_json,
            &sender,
        )
        .await;
        if let Err(error) = result {
            let reason = if is_aborted(options.stream_options.stream.abort_flag.as_ref()) {
                StopReason::Aborted
            } else {
                StopReason::Error
            };
            partial.stop_reason = reason;
            partial.error_message = Some(error);
            partial.timestamp = now_millis();
            sender.push(AssistantMessageEvent::Error {
                reason,
                error: partial,
            });
        }
    });
    stream
}

async fn run_proxy_request(
    model: &Model,
    context: &Context,
    options: &ProxyStreamOptions,
    partial: &mut AssistantMessage,
    tool_partial_json: &mut BTreeMap<usize, String>,
    sender: &ri_llm_provider::AssistantMessageEventSender,
) -> Result<(), String> {
    check_abort(options.stream_options.stream.abort_flag.as_ref())?;
    let url = format!("{}/api/stream", options.proxy_url.trim_end_matches('/'));
    let request_body = ProxyRequest {
        model,
        context,
        options: ProxySerializableStreamOptions::from(&options.stream_options),
    };
    let client = reqwest::Client::new();
    let mut request = client
        .post(url)
        .bearer_auth(&options.auth_token)
        .header("content-type", "application/json")
        .json(&request_body);
    if let Some(timeout_ms) = options.stream_options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms));
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        let error = serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|value| {
                value
                    .get("error")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .map(|message| format!("Proxy error: {message}"))
            .unwrap_or_else(|| {
                status
                    .canonical_reason()
                    .map(|reason| format!("Proxy error: {} {reason}", status.as_u16()))
                    .unwrap_or_else(|| format!("Proxy error: {}", status.as_u16()))
            });
        return Err(error);
    }

    let mut byte_stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut terminal = false;
    while let Some(chunk) = byte_stream.next().await {
        check_abort(options.stream_options.stream.abort_flag.as_ref())?;
        let chunk = chunk.map_err(|error| error.to_string())?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim_end_matches('\r').to_owned();
            buffer.drain(..=line_end);
            if let Some(data) = line.strip_prefix("data: ") {
                let data = data.trim();
                if !data.is_empty() {
                    let proxy_event = serde_json::from_str::<ProxyAssistantMessageEvent>(data)
                        .map_err(|error| error.to_string())?;
                    terminal |=
                        process_proxy_event(proxy_event, partial, tool_partial_json, sender)?;
                }
            }
        }
    }
    check_abort(options.stream_options.stream.abort_flag.as_ref())?;
    if !terminal {
        sender.end(partial.clone());
    }
    Ok(())
}

fn process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: &mut AssistantMessage,
    tool_partial_json: &mut BTreeMap<usize, String>,
    sender: &ri_llm_provider::AssistantMessageEventSender,
) -> Result<bool, String> {
    match proxy_event {
        ProxyAssistantMessageEvent::Start => {
            sender.push(AssistantMessageEvent::Start {
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::TextStart { content_index } => {
            ensure_content_slot(partial, content_index, AssistantContent::text(""));
            sender.push(AssistantMessageEvent::TextStart {
                content_index,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::TextDelta {
            content_index,
            delta,
        } => {
            let content = content_mut(partial, content_index)?;
            let AssistantContent::Text(text) = content else {
                return Err("Received text_delta for non-text content".to_owned());
            };
            text.text.push_str(&delta);
            sender.push(AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::TextEnd {
            content_index,
            content_signature,
        } => {
            let content = content_mut(partial, content_index)?;
            let AssistantContent::Text(text) = content else {
                return Err("Received text_end for non-text content".to_owned());
            };
            text.text_signature = content_signature.map(Value::String);
            let content = text.text.clone();
            sender.push(AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::ThinkingStart { content_index } => {
            ensure_content_slot(partial, content_index, AssistantContent::thinking(""));
            sender.push(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
        } => {
            let content = content_mut(partial, content_index)?;
            let AssistantContent::Thinking(thinking) = content else {
                return Err("Received thinking_delta for non-thinking content".to_owned());
            };
            thinking.thinking.push_str(&delta);
            sender.push(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::ThinkingEnd {
            content_index,
            content_signature,
        } => {
            let content = content_mut(partial, content_index)?;
            let AssistantContent::Thinking(thinking) = content else {
                return Err("Received thinking_end for non-thinking content".to_owned());
            };
            thinking.thinking_signature = content_signature;
            let content = thinking.thinking.clone();
            sender.push(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::ToolcallStart {
            content_index,
            id,
            tool_name,
        } => {
            ensure_content_slot(
                partial,
                content_index,
                AssistantContent::ToolCall(ToolCall {
                    id,
                    name: tool_name,
                    arguments: Map::new(),
                    thought_signature: None,
                }),
            );
            tool_partial_json.insert(content_index, String::new());
            sender.push(AssistantMessageEvent::ToolcallStart {
                content_index,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::ToolcallDelta {
            content_index,
            delta,
        } => {
            let scratch = tool_partial_json.entry(content_index).or_default();
            scratch.push_str(&delta);
            let content = content_mut(partial, content_index)?;
            let AssistantContent::ToolCall(tool_call) = content else {
                return Err("Received toolcall_delta for non-toolCall content".to_owned());
            };
            tool_call.arguments = parse_streaming_json(Some(scratch))
                .as_object()
                .cloned()
                .unwrap_or_default();
            sender.push(AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::ToolcallEnd { content_index } => {
            tool_partial_json.remove(&content_index);
            let content = content_mut(partial, content_index)?;
            let AssistantContent::ToolCall(tool_call) = content else {
                return Ok(false);
            };
            sender.push(AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call: tool_call.clone(),
                partial: partial.clone(),
            });
            Ok(false)
        }
        ProxyAssistantMessageEvent::Done { reason, usage } => {
            partial.stop_reason = reason;
            partial.usage = usage;
            partial.timestamp = now_millis();
            sender.push(AssistantMessageEvent::Done {
                reason,
                message: partial.clone(),
            });
            Ok(true)
        }
        ProxyAssistantMessageEvent::Error {
            reason,
            error_message,
            usage,
        } => {
            partial.stop_reason = reason;
            partial.error_message = error_message;
            partial.usage = usage;
            partial.timestamp = now_millis();
            sender.push(AssistantMessageEvent::Error {
                reason,
                error: partial.clone(),
            });
            Ok(true)
        }
    }
}

fn empty_proxy_assistant_message(model: &Model) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: now_millis(),
    }
}

fn ensure_content_slot(
    partial: &mut AssistantMessage,
    content_index: usize,
    content: AssistantContent,
) {
    while partial.content.len() <= content_index {
        partial.content.push(AssistantContent::Text(TextContent {
            text: String::new(),
            text_signature: None,
        }));
    }
    partial.content[content_index] = content;
}

fn content_mut(
    partial: &mut AssistantMessage,
    content_index: usize,
) -> Result<&mut AssistantContent, String> {
    partial
        .content
        .get_mut(content_index)
        .ok_or_else(|| format!("Received proxy event for missing content index {content_index}"))
}

fn check_abort(abort_flag: Option<&Arc<AtomicBool>>) -> Result<(), String> {
    if is_aborted(abort_flag) {
        Err("Request aborted by user".to_owned())
    } else {
        Ok(())
    }
}

fn is_aborted(abort_flag: Option<&Arc<AtomicBool>>) -> bool {
    abort_flag.is_some_and(|flag| flag.load(Ordering::SeqCst))
}
