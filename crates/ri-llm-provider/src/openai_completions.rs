use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    CacheRetention, Context, Message, Model, StopReason, TextContent, ThinkingContent,
    ThinkingLevel, Tool, ToolCall, Usage, UserContent, UserContentValue,
    github_copilot_headers::build_copilot_dynamic_headers,
    json_repair::parse_json_with_repair,
    message_transform::{normalize_openai_completions_tool_call_id, transform_messages},
    models::calculate_cost,
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAICompletionsPayloadOptions {
    pub tool_choice: Option<String>,
    pub reasoning: Option<ThinkingLevel>,
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub headers: BTreeMap<String, String>,
}

pub fn build_openai_completions_payload(
    model: &Model,
    context: &Context,
    options: OpenAICompletionsPayloadOptions,
) -> Value {
    let cache_retention = resolve_openai_completions_cache_retention(options.cache_retention);
    let anthropic_cache_markers =
        uses_anthropic_cache_control(model) && cache_retention != CacheRetention::None;
    let messages =
        convert_openai_completions_messages_with_cache(model, context, anthropic_cache_markers);

    let mut payload = json!({
        "model": model.id,
        "messages": messages,
        "stream": true,
    });

    if supports_usage_in_streaming(model) {
        payload["stream_options"] = json!({ "include_usage": true });
    }
    if supports_store(model) {
        payload["store"] = Value::Bool(false);
    }

    if !context.tools.is_empty() {
        payload["tools"] = Value::Array(
            context
                .tools
                .iter()
                .map(|tool| format_tool(tool, model, anthropic_cache_markers))
                .collect(),
        );
    } else if has_tool_history(&context.messages) {
        payload["tools"] = Value::Array(Vec::new());
    }

    if let Some(tool_choice) = options.tool_choice.filter(|value| !value.is_empty()) {
        payload["tool_choice"] = Value::String(tool_choice);
    }

    if should_set_prompt_cache_key(model, cache_retention) {
        if let Some(session_id) = options.session_id {
            payload["prompt_cache_key"] = Value::String(session_id);
        }
    }
    if cache_retention == CacheRetention::Long && supports_long_cache_retention(model) {
        payload["prompt_cache_retention"] = Value::String("24h".to_owned());
    }

    if let Some(max_tokens) = options.max_tokens.filter(|value| *value > 0) {
        if max_tokens_field(model) == "max_tokens" {
            payload["max_tokens"] = json!(max_tokens);
        } else {
            payload["max_completion_tokens"] = json!(max_tokens);
        }
    }

    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }

    apply_reasoning_options(&mut payload, model, options.reasoning);
    if should_enable_zai_tool_stream(model) && !context.tools.is_empty() {
        payload["tool_stream"] = Value::Bool(true);
    }
    apply_provider_routing_options(&mut payload, model);

    payload
}

pub fn convert_openai_completions_messages(model: &Model, context: &Context) -> Vec<Value> {
    convert_openai_completions_messages_with_cache(model, context, false)
}

fn convert_openai_completions_messages_with_cache(
    model: &Model,
    context: &Context,
    anthropic_cache_markers: bool,
) -> Vec<Value> {
    let mut messages = Vec::new();
    if let Some(system_prompt) = &context.system_prompt {
        let role = if model.reasoning && supports_developer_role(model) {
            "developer"
        } else {
            "system"
        };
        messages.push(json!({
            "role": role,
            "content": format_text_content(system_prompt, anthropic_cache_markers),
        }));
    }

    let transformed_messages = transform_messages(
        &context.messages,
        model,
        Some(&|id, model, _source| normalize_openai_completions_tool_call_id(id, model)),
    );
    let last_user_index = transformed_messages
        .iter()
        .rposition(|message| matches!(message, Message::User(_)));
    let mut last_role: Option<&str> = None;
    let mut index = 0;
    while index < transformed_messages.len() {
        let message = &transformed_messages[index];
        if requires_assistant_after_tool_result(model)
            && last_role == Some("toolResult")
            && matches!(message, Message::User(_))
        {
            messages.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results.",
            }));
        }

        match message {
            Message::User(user) => {
                let cache = anthropic_cache_markers && Some(index) == last_user_index;
                if let Some(content) = format_user_content(&user.content, cache) {
                    messages.push(json!({
                        "role": "user",
                        "content": content,
                    }));
                    last_role = Some("user");
                }
            }
            Message::Assistant(assistant) => {
                let assistant_text_parts = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(block) if !block.text.trim().is_empty() => {
                            Some(text_part(&block.text, false))
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let assistant_text = assistant_text_parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<String>();
                let thinking_blocks = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Thinking(block) if !block.thinking.trim().is_empty() => {
                            Some(block)
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let mut tool_calls = Vec::new();
                let mut reasoning_details = Vec::new();
                for content in &assistant.content {
                    match content {
                        AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(json!({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_owned()),
                                },
                            }));
                            if let Some(signature) = &tool_call.thought_signature
                                && let Ok(detail) = serde_json::from_str::<Value>(signature)
                            {
                                reasoning_details.push(detail);
                            }
                        }
                        _ => {}
                    }
                }
                let mut message = json!({
                    "role": "assistant",
                    "content": if requires_thinking_as_text(model) && !thinking_blocks.is_empty() {
                        let thinking_text = thinking_blocks
                            .iter()
                            .map(|block| block.thinking.as_str())
                            .collect::<Vec<_>>()
                            .join("\n\n");
                        let mut parts = vec![text_part(&thinking_text, false)];
                        parts.extend(assistant_text_parts);
                        Value::Array(parts)
                    } else {
                        Value::String(assistant_text)
                    },
                });
                if !requires_thinking_as_text(model) && !thinking_blocks.is_empty() {
                    if let Some(signature) = thinking_blocks
                        .first()
                        .and_then(|block| block.thinking_signature.as_deref())
                        .filter(|signature| !signature.is_empty())
                    {
                        let field = if model.provider == "opencode-go" && signature == "reasoning" {
                            "reasoning_content"
                        } else {
                            signature
                        };
                        message[field] = Value::String(
                            thinking_blocks
                                .iter()
                                .map(|block| block.thinking.as_str())
                                .collect::<Vec<_>>()
                                .join("\n"),
                        );
                    }
                }
                if requires_assistant_after_tool_result(model) && message["content"] == "" {
                    message["content"] = Value::String(String::new());
                }
                if !tool_calls.is_empty() {
                    message["tool_calls"] = Value::Array(tool_calls);
                }
                if !reasoning_details.is_empty() {
                    message["reasoning_details"] = Value::Array(reasoning_details);
                }
                if requires_reasoning_content_on_assistant_messages(model)
                    && model.reasoning
                    && message.get("reasoning_content").is_none()
                {
                    message["reasoning_content"] = Value::String(String::new());
                }

                let has_content = match &message["content"] {
                    Value::String(text) => !text.is_empty(),
                    Value::Array(parts) => !parts.is_empty(),
                    Value::Null => false,
                    _ => true,
                };
                if has_content || message.get("tool_calls").is_some() {
                    messages.push(message);
                    last_role = Some("assistant");
                }
            }
            Message::ToolResult(_) => {
                let mut image_blocks = Vec::new();
                let mut next = index;
                while next < transformed_messages.len() {
                    let Message::ToolResult(tool_result) = &transformed_messages[next] else {
                        break;
                    };
                    let text = tool_result
                        .content
                        .iter()
                        .filter_map(|content| match content {
                            crate::ToolResultContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    let has_images = tool_result
                        .content
                        .iter()
                        .any(|content| matches!(content, crate::ToolResultContent::Image(_)));
                    let mut tool_message = json!({
                        "role": "tool",
                        "tool_call_id": tool_result.tool_call_id,
                        "content": if text.is_empty() && has_images {
                            "(see attached image)"
                        } else {
                            text.as_str()
                        },
                    });
                    if requires_tool_result_name(model) {
                        tool_message["name"] = Value::String(tool_result.tool_name.clone());
                    }
                    messages.push(tool_message);

                    if has_images && model.input.contains(&crate::InputKind::Image) {
                        for content in &tool_result.content {
                            if let crate::ToolResultContent::Image(image) = content {
                                image_blocks.push(json!({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": format!("data:{};base64,{}", image.mime_type, image.data),
                                    },
                                }));
                            }
                        }
                    }
                    next += 1;
                }

                index = next - 1;
                if !image_blocks.is_empty() {
                    if requires_assistant_after_tool_result(model) {
                        messages.push(json!({
                            "role": "assistant",
                            "content": "I have processed the tool results.",
                        }));
                    }
                    let mut content = vec![json!({
                        "type": "text",
                        "text": "Attached image(s) from tool result:",
                    })];
                    content.extend(image_blocks);
                    messages.push(json!({
                        "role": "user",
                        "content": content,
                    }));
                    last_role = Some("user");
                } else {
                    last_role = Some("toolResult");
                }
            }
        }
        index += 1;
    }

    messages
}

pub fn resolve_openai_completions_cache_retention(
    cache_retention: Option<CacheRetention>,
) -> CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention;
    }
    if std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long") {
        CacheRetention::Long
    } else {
        CacheRetention::Short
    }
}

pub fn build_openai_completions_default_headers(
    model: &Model,
    session_id: Option<&str>,
    cache_retention: CacheRetention,
    option_headers: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    build_openai_completions_default_headers_with_context(
        model,
        None,
        session_id,
        cache_retention,
        option_headers,
    )
}

pub fn build_openai_completions_default_headers_with_context(
    model: &Model,
    context: Option<&Context>,
    session_id: Option<&str>,
    cache_retention: CacheRetention,
    option_headers: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut headers = model.headers.clone();
    if model.provider == "github-copilot"
        && let Some(context) = context
    {
        headers.extend(build_copilot_dynamic_headers(context));
    }
    if let Some(session_id) = session_id.filter(|value| !value.is_empty())
        && cache_retention != CacheRetention::None
        && send_session_affinity_headers(model)
    {
        headers.insert("session_id".to_owned(), session_id.to_owned());
        headers.insert("x-client-request-id".to_owned(), session_id.to_owned());
        headers.insert("x-session-affinity".to_owned(), session_id.to_owned());
    }
    headers.extend(option_headers.clone());
    headers
}

pub fn apply_openai_completions_chunk_metadata(
    output: &mut AssistantMessage,
    model: &Model,
    chunk: &Value,
) {
    if output.response_id.is_none() {
        if let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            output.response_id = Some(id.to_owned());
        }
    }
    if output.response_model.is_none() {
        if let Some(chunk_model) = chunk
            .get("model")
            .and_then(Value::as_str)
            .filter(|chunk_model| !chunk_model.is_empty() && *chunk_model != model.id)
        {
            output.response_model = Some(chunk_model.to_owned());
        }
    }
    if let Some(usage) = chunk.get("usage") {
        output.usage = parse_openai_completions_chunk_usage(usage, model);
    } else if let Some(choice_usage) = chunk.pointer("/choices/0/usage") {
        output.usage = parse_openai_completions_chunk_usage(choice_usage, model);
    }
}

pub fn process_openai_completions_chunks<I>(
    chunks: I,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    model: &Model,
) -> Result<(), String>
where
    I: IntoIterator<Item = Value>,
{
    let mut processor = OpenAICompletionsStreamProcessor::new();
    for chunk in chunks {
        processor.process_chunk(chunk, output, sender, model)?;
    }
    processor.finish(output, sender)
}

#[derive(Debug, Default)]
pub struct OpenAICompletionsStreamProcessor {
    started: bool,
    text_index: Option<usize>,
    thinking_index: Option<usize>,
    has_finish_reason: bool,
    tool_calls_by_index: BTreeMap<i64, usize>,
    tool_calls_by_id: BTreeMap<String, usize>,
    tool_call_partial_args: BTreeMap<usize, String>,
}

impl OpenAICompletionsStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_chunk(
        &mut self,
        chunk: Value,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
        model: &Model,
    ) -> Result<(), String> {
        if !self.started {
            self.started = true;
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        if !chunk.is_object() {
            return Ok(());
        }
        apply_openai_completions_chunk_metadata(output, model, &chunk);

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return Ok(());
        };

        if let Some(finish_reason) = choice.get("finish_reason").and_then(Value::as_str) {
            let (stop_reason, error_message) = map_openai_completions_finish_reason(finish_reason);
            output.stop_reason = stop_reason;
            output.error_message = error_message;
            self.has_finish_reason = true;
        }

        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            return Ok(());
        };

        if let Some(content) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|content| !content.is_empty())
        {
            push_openai_completions_text_delta(output, sender, &mut self.text_index, content);
        }

        if let Some((signature, reasoning_delta)) = openai_completions_reasoning_delta(model, delta)
        {
            push_openai_completions_thinking_delta(
                output,
                sender,
                &mut self.thinking_index,
                signature,
                reasoning_delta,
            );
        }

        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let Some(tool_call) = tool_call.as_object() else {
                    continue;
                };
                let content_index = ensure_openai_completions_tool_call_block(
                    output,
                    sender,
                    tool_call,
                    &mut self.tool_calls_by_index,
                    &mut self.tool_calls_by_id,
                    &mut self.tool_call_partial_args,
                );
                let argument_delta = tool_call
                    .get("function")
                    .and_then(Value::as_object)
                    .and_then(|function| function.get("arguments"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if !argument_delta.is_empty() {
                    let partial_args = self
                        .tool_call_partial_args
                        .entry(content_index)
                        .or_default();
                    partial_args.push_str(argument_delta);
                    if let Some(AssistantContent::ToolCall(block)) =
                        output.content.get_mut(content_index)
                    {
                        block.arguments = parse_openai_completions_arguments(partial_args);
                    }
                }
                sender.push(AssistantMessageEvent::ToolcallDelta {
                    content_index,
                    delta: argument_delta.to_owned(),
                    partial: output.clone(),
                });
            }
        }

        if let Some(details) = delta.get("reasoning_details").and_then(Value::as_array) {
            apply_openai_completions_reasoning_details(output, details);
        }
        Ok(())
    }

    pub fn finish(
        self,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) -> Result<(), String> {
        finish_openai_completions_blocks(output, sender, &self.tool_call_partial_args);

        if !self.has_finish_reason {
            return Err("Stream ended without finish_reason".to_owned());
        }
        if output.stop_reason == StopReason::Error {
            let message = output
                .error_message
                .clone()
                .unwrap_or_else(|| "Provider returned an error stop reason".to_owned());
            sender.push(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                error: output.clone(),
            });
            return Err(message);
        }

        sender.push(AssistantMessageEvent::Done {
            reason: output.stop_reason,
            message: output.clone(),
        });
        Ok(())
    }
}

fn push_openai_completions_text_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    text_index: &mut Option<usize>,
    delta: &str,
) {
    let index = *text_index.get_or_insert_with(|| {
        output
            .content
            .push(AssistantContent::Text(TextContent::new("")));
        let index = output.content.len() - 1;
        sender.push(AssistantMessageEvent::TextStart {
            content_index: index,
            partial: output.clone(),
        });
        index
    });

    if let Some(AssistantContent::Text(block)) = output.content.get_mut(index) {
        block.text.push_str(delta);
    }
    sender.push(AssistantMessageEvent::TextDelta {
        content_index: index,
        delta: delta.to_owned(),
        partial: output.clone(),
    });
}

fn push_openai_completions_thinking_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    thinking_index: &mut Option<usize>,
    signature: &str,
    delta: &str,
) {
    let index = *thinking_index.get_or_insert_with(|| {
        output
            .content
            .push(AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(signature.to_owned()),
                redacted: false,
            }));
        let index = output.content.len() - 1;
        sender.push(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            partial: output.clone(),
        });
        index
    });

    if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(index) {
        block.thinking.push_str(delta);
    }
    sender.push(AssistantMessageEvent::ThinkingDelta {
        content_index: index,
        delta: delta.to_owned(),
        partial: output.clone(),
    });
}

fn ensure_openai_completions_tool_call_block(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    tool_call: &Map<String, Value>,
    tool_calls_by_index: &mut BTreeMap<i64, usize>,
    tool_calls_by_id: &mut BTreeMap<String, usize>,
    tool_call_partial_args: &mut BTreeMap<usize, String>,
) -> usize {
    let stream_index = tool_call.get("index").and_then(|index| {
        index
            .as_i64()
            .or_else(|| index.as_u64().map(|index| index as i64))
    });
    let id = tool_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    let name = tool_call
        .get("function")
        .and_then(Value::as_object)
        .and_then(|function| function.get("name"))
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty());

    let content_index = stream_index
        .and_then(|stream_index| tool_calls_by_index.get(&stream_index).copied())
        .or_else(|| id.and_then(|id| tool_calls_by_id.get(id).copied()))
        .unwrap_or_else(|| {
            let block = ToolCall {
                id: id.unwrap_or_default().to_owned(),
                name: name.unwrap_or_default().to_owned(),
                arguments: Map::new(),
                thought_signature: None,
            };
            output.content.push(AssistantContent::ToolCall(block));
            let content_index = output.content.len() - 1;
            tool_call_partial_args.insert(content_index, String::new());
            sender.push(AssistantMessageEvent::ToolcallStart {
                content_index,
                partial: output.clone(),
            });
            content_index
        });

    if let Some(stream_index) = stream_index {
        tool_calls_by_index.insert(stream_index, content_index);
    }
    if let Some(id) = id {
        tool_calls_by_id.insert(id.to_owned(), content_index);
    }
    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(content_index) {
        if block.id.is_empty()
            && let Some(id) = id
        {
            block.id = id.to_owned();
        }
        if block.name.is_empty()
            && let Some(name) = name
        {
            block.name = name.to_owned();
        }
    }

    content_index
}

fn finish_openai_completions_blocks(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    tool_call_partial_args: &BTreeMap<usize, String>,
) {
    for index in 0..output.content.len() {
        match &mut output.content[index] {
            AssistantContent::Text(block) => {
                let content = block.text.clone();
                sender.push(AssistantMessageEvent::TextEnd {
                    content_index: index,
                    content,
                    partial: output.clone(),
                });
            }
            AssistantContent::Thinking(block) => {
                let content = block.thinking.clone();
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: index,
                    content,
                    partial: output.clone(),
                });
            }
            AssistantContent::ToolCall(block) => {
                if let Some(partial_args) = tool_call_partial_args.get(&index) {
                    block.arguments = parse_openai_completions_arguments(partial_args);
                }
                let tool_call = block.clone();
                sender.push(AssistantMessageEvent::ToolcallEnd {
                    content_index: index,
                    tool_call,
                    partial: output.clone(),
                });
            }
        }
    }
}

fn openai_completions_reasoning_delta<'a>(
    model: &Model,
    delta: &'a Map<String, Value>,
) -> Option<(&'static str, &'a str)> {
    for field in ["reasoning_content", "reasoning", "reasoning_text"] {
        let value = delta
            .get(field)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        if let Some(value) = value {
            let signature = if model.provider == "opencode-go" && field == "reasoning" {
                "reasoning_content"
            } else {
                field
            };
            return Some((signature, value));
        }
    }
    None
}

fn apply_openai_completions_reasoning_details(output: &mut AssistantMessage, details: &[Value]) {
    for detail in details {
        if detail.get("type").and_then(Value::as_str) != Some("reasoning.encrypted") {
            continue;
        }
        let Some(id) = detail.get("id").and_then(Value::as_str) else {
            continue;
        };
        if detail.get("data").is_none() {
            continue;
        }
        if let Some(AssistantContent::ToolCall(tool_call)) =
            output.content.iter_mut().find(|content| match content {
                AssistantContent::ToolCall(tool_call) => tool_call.id == id,
                _ => false,
            })
        {
            tool_call.thought_signature = Some(detail.to_string());
        }
    }
}

fn parse_openai_completions_arguments(arguments: &str) -> Map<String, Value> {
    if arguments.trim().is_empty() {
        return Map::new();
    }
    parse_json_with_repair::<Value>(arguments)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn map_openai_completions_finish_reason(reason: &str) -> (StopReason, Option<String>) {
    match reason {
        "stop" | "end" => (StopReason::Stop, None),
        "length" => (StopReason::Length, None),
        "function_call" | "tool_calls" => (StopReason::ToolUse, None),
        "content_filter" => (
            StopReason::Error,
            Some("Provider finish_reason: content_filter".to_owned()),
        ),
        "network_error" => (
            StopReason::Error,
            Some("Provider finish_reason: network_error".to_owned()),
        ),
        _ => (
            StopReason::Error,
            Some(format!("Provider finish_reason: {reason}")),
        ),
    }
}

pub fn parse_openai_completions_chunk_usage(raw_usage: &Value, model: &Model) -> Usage {
    let prompt_tokens = raw_usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = raw_usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            raw_usage
                .get("prompt_cache_hit_tokens")
                .and_then(Value::as_u64)
        })
        .unwrap_or(0);
    let cache_write = raw_usage
        .pointer("/prompt_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output_tokens = raw_usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let input = prompt_tokens.saturating_sub(cache_read + cache_write);
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens: input + output_tokens + cache_read + cache_write,
        cost: Default::default(),
    };
    calculate_cost(model, &mut usage);
    usage
}

fn format_user_content(content: &UserContentValue, cache_control: bool) -> Option<Value> {
    match content {
        UserContentValue::Plain(text) => Some(format_text_content(text, cache_control)),
        UserContentValue::Blocks(blocks) => {
            let parts = blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => text_part(&text.text, cache_control),
                    UserContent::Image(image) => json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{};base64,{}", image.mime_type, image.data),
                        },
                    }),
                })
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| Value::Array(parts))
        }
    }
}

fn format_text_content(text: &str, cache_control: bool) -> Value {
    if cache_control {
        Value::Array(vec![text_part(text, true)])
    } else {
        Value::String(text.to_owned())
    }
}

fn text_part(text: &str, cache_control: bool) -> Value {
    let mut part = json!({ "type": "text", "text": text });
    if cache_control {
        part["cache_control"] = json!({ "type": "ephemeral" });
    }
    part
}

fn format_tool(tool: &Tool, model: &Model, cache_control: bool) -> Value {
    let mut function = json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    });
    if supports_strict_mode(model) {
        function["strict"] = Value::Bool(true);
    }
    let mut formatted = json!({
        "type": "function",
        "function": function,
    });
    if cache_control {
        formatted["cache_control"] = json!({ "type": "ephemeral" });
    }
    formatted
}

fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|message| match message {
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_))),
        Message::ToolResult(_) => true,
        _ => false,
    })
}

fn supports_strict_mode(model: &Model) -> bool {
    !model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsStrictMode"))
        .and_then(Value::as_bool)
        .map(|enabled| !enabled)
        .unwrap_or(false)
}

fn supports_usage_in_streaming(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsUsageInStreaming"))
        .and_then(Value::as_bool)
        != Some(false)
}

fn supports_store(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsStore"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| !is_nonstandard_openai_completions_model(model))
}

fn supports_developer_role(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsDeveloperRole"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| !is_nonstandard_openai_completions_model(model))
}

fn is_nonstandard_openai_completions_model(model: &Model) -> bool {
    let provider = model.provider.as_str();
    let base_url = model.base_url.as_str();

    provider == "cerebras"
        || base_url.contains("cerebras.ai")
        || provider == "xai"
        || base_url.contains("api.x.ai")
        || is_together_openai_completions_model(model)
        || base_url.contains("chutes.ai")
        || base_url.contains("deepseek.com")
        || is_zai_openai_completions_model(model)
        || is_moonshot_openai_completions_model(model)
        || provider == "opencode"
        || base_url.contains("opencode.ai")
        || is_cloudflare_workers_ai_model(model)
        || is_cloudflare_ai_gateway_model(model)
}

fn is_together_openai_completions_model(model: &Model) -> bool {
    model.provider == "together"
        || model.base_url.contains("api.together.ai")
        || model.base_url.contains("api.together.xyz")
}

fn is_zai_openai_completions_model(model: &Model) -> bool {
    model.provider == "zai" || model.base_url.contains("api.z.ai")
}

fn is_moonshot_openai_completions_model(model: &Model) -> bool {
    model.provider == "moonshotai"
        || model.provider == "moonshotai-cn"
        || model.base_url.contains("api.moonshot.")
}

fn is_cloudflare_workers_ai_model(model: &Model) -> bool {
    model.provider == "cloudflare-workers-ai" || model.base_url.contains("api.cloudflare.com")
}

fn is_cloudflare_ai_gateway_model(model: &Model) -> bool {
    model.provider == "cloudflare-ai-gateway"
        || model.base_url.contains("gateway.ai.cloudflare.com")
}

fn is_grok_openai_completions_model(model: &Model) -> bool {
    model.provider == "xai" || model.base_url.contains("api.x.ai")
}

fn is_deepseek_openai_completions_model(model: &Model) -> bool {
    model.provider == "deepseek" || model.base_url.contains("deepseek.com")
}

fn uses_max_tokens_field(model: &Model) -> bool {
    model.base_url.contains("chutes.ai")
        || is_moonshot_openai_completions_model(model)
        || is_cloudflare_ai_gateway_model(model)
        || is_together_openai_completions_model(model)
}

fn uses_anthropic_cache_control(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("cacheControlFormat"))
        .and_then(Value::as_str)
        == Some("anthropic")
        || (model.provider == "openrouter" && model.id.contains("anthropic/"))
}

fn should_enable_zai_tool_stream(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("zaiToolStream"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn requires_thinking_as_text(model: &Model) -> bool {
    compat_bool(model, "requiresThinkingAsText")
}

fn requires_tool_result_name(model: &Model) -> bool {
    compat_bool(model, "requiresToolResultName")
}

fn requires_assistant_after_tool_result(model: &Model) -> bool {
    compat_bool(model, "requiresAssistantAfterToolResult")
}

fn requires_reasoning_content_on_assistant_messages(model: &Model) -> bool {
    compat_bool(model, "requiresReasoningContentOnAssistantMessages")
}

fn send_session_affinity_headers(model: &Model) -> bool {
    compat_bool(model, "sendSessionAffinityHeaders")
}

fn max_tokens_field(model: &Model) -> &'static str {
    match model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("maxTokensField"))
        .and_then(Value::as_str)
    {
        Some("max_tokens") => "max_tokens",
        Some("max_completion_tokens") => "max_completion_tokens",
        _ if uses_max_tokens_field(model) => "max_tokens",
        _ => "max_completion_tokens",
    }
}

fn should_set_prompt_cache_key(model: &Model, cache_retention: CacheRetention) -> bool {
    (model.base_url.contains("api.openai.com") && cache_retention != CacheRetention::None)
        || (cache_retention == CacheRetention::Long && supports_long_cache_retention(model))
}

fn supports_long_cache_retention(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsLongCacheRetention"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            !(is_together_openai_completions_model(model)
                || is_cloudflare_workers_ai_model(model)
                || is_cloudflare_ai_gateway_model(model))
        })
}

fn compat_bool(model: &Model, key: &str) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get(key))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

fn apply_reasoning_options(payload: &mut Value, model: &Model, reasoning: Option<ThinkingLevel>) {
    if !model.reasoning {
        return;
    }

    if model.provider == "groq" && model.id.starts_with("qwen/qwen3") {
        if reasoning.is_some() {
            payload["reasoning_effort"] = Value::String("default".to_owned());
        }
    } else {
        match thinking_format(model) {
            "zai" | "qwen" => {
                payload["enable_thinking"] = Value::Bool(reasoning.is_some());
            }
            "qwen-chat-template" => {
                payload["chat_template_kwargs"] = json!({
                    "enable_thinking": reasoning.is_some(),
                    "preserve_thinking": true,
                });
            }
            "deepseek" => {
                payload["thinking"] = json!({
                    "type": if reasoning.is_some() { "enabled" } else { "disabled" },
                });
                if let Some(reasoning) = reasoning {
                    payload["reasoning_effort"] =
                        Value::String(thinking_level_wire(model, reasoning));
                }
            }
            "openrouter" => {
                if let Some(reasoning) = reasoning {
                    payload["reasoning"] =
                        json!({ "effort": thinking_level_wire(model, reasoning) });
                } else if model.thinking_level_map.get(&ThinkingLevel::Off) != Some(&None) {
                    let effort = model
                        .thinking_level_map
                        .get(&ThinkingLevel::Off)
                        .and_then(|value| value.clone())
                        .unwrap_or_else(|| "none".to_owned());
                    payload["reasoning"] = json!({ "effort": effort });
                }
            }
            "together" => {
                payload["reasoning"] = json!({ "enabled": reasoning.is_some() });
                if let Some(reasoning) = reasoning
                    && supports_reasoning_effort(model)
                {
                    payload["reasoning_effort"] =
                        Value::String(thinking_level_wire(model, reasoning));
                }
            }
            _ => {
                if let Some(reasoning) = reasoning {
                    if supports_reasoning_effort(model) {
                        payload["reasoning_effort"] =
                            Value::String(thinking_level_wire(model, reasoning));
                    }
                } else if supports_reasoning_effort(model)
                    && let Some(Some(off_value)) = model.thinking_level_map.get(&ThinkingLevel::Off)
                {
                    payload["reasoning_effort"] = Value::String(off_value.clone());
                }
            }
        }
    }
}

fn supports_reasoning_effort(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsReasoningEffort"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            !(is_grok_openai_completions_model(model)
                || is_zai_openai_completions_model(model)
                || is_moonshot_openai_completions_model(model)
                || is_together_openai_completions_model(model)
                || is_cloudflare_ai_gateway_model(model))
        })
}

fn thinking_format(model: &Model) -> &str {
    if let Some(format) = model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("thinkingFormat"))
        .and_then(Value::as_str)
    {
        return format;
    }

    if is_deepseek_openai_completions_model(model) {
        "deepseek"
    } else if is_zai_openai_completions_model(model) {
        "zai"
    } else if is_together_openai_completions_model(model) {
        "together"
    } else if model.provider == "openrouter" || model.base_url.contains("openrouter.ai") {
        "openrouter"
    } else {
        "openai"
    }
}

fn apply_provider_routing_options(payload: &mut Value, model: &Model) {
    let Some(compat) = model.compat.as_ref() else {
        return;
    };

    if model.base_url.contains("openrouter.ai")
        && let Some(routing) = compat.get("openRouterRouting")
        && routing.is_object()
    {
        payload["provider"] = routing.clone();
    }

    if model.base_url.contains("ai-gateway.vercel.sh")
        && let Some(routing) = compat
            .get("vercelGatewayRouting")
            .and_then(Value::as_object)
    {
        let mut gateway_options = Map::new();
        if let Some(only) = routing.get("only").filter(|value| !value.is_null()) {
            gateway_options.insert("only".to_owned(), only.clone());
        }
        if let Some(order) = routing.get("order").filter(|value| !value.is_null()) {
            gateway_options.insert("order".to_owned(), order.clone());
        }
        if !gateway_options.is_empty() {
            payload["providerOptions"] = json!({ "gateway": gateway_options });
        }
    }
}

fn thinking_level_wire(model: &Model, level: ThinkingLevel) -> String {
    if let Some(Some(mapped)) = model.thinking_level_map.get(&level) {
        return mapped.clone();
    }
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
    }
    .to_owned()
}
