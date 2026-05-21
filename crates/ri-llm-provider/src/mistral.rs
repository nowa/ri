use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    Context, InputKind, Message, Model, SimpleStreamOptions, StopReason, TextContent,
    ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolResultContent, Usage, UserContent,
    UserContentValue,
    json_repair::{parse_streaming_json, sanitize_surrogates, short_hash},
    message_transform::transform_messages,
    models::{calculate_cost, clamp_thinking_level},
    simple_options::apply_simple_stream_defaults,
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
};

const MISTRAL_TOOL_CALL_ID_LENGTH: usize = 9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MistralToolChoice {
    Auto,
    None,
    Any,
    Required,
    Function { name: String },
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct MistralPayloadOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tool_choice: Option<MistralToolChoice>,
    pub prompt_mode: Option<String>,
    pub reasoning_effort: Option<String>,
}

pub fn build_mistral_simple_payload(
    model: &Model,
    context: &Context,
    options: SimpleStreamOptions,
) -> Value {
    let options = apply_simple_stream_defaults(model, options);
    let reasoning = options
        .reasoning
        .map(|level| clamp_thinking_level(model, level))
        .filter(|level| *level != ThinkingLevel::Off);
    let should_use_reasoning = model.reasoning && reasoning.is_some();
    let mut payload_options = MistralPayloadOptions {
        temperature: options.stream.temperature,
        max_tokens: options.stream.max_tokens,
        tool_choice: options
            .stream
            .extra
            .get("toolChoice")
            .and_then(parse_mistral_tool_choice),
        prompt_mode: options
            .stream
            .extra
            .get("promptMode")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        reasoning_effort: options
            .stream
            .extra
            .get("reasoningEffort")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned),
        ..Default::default()
    };

    let has_explicit_reasoning_control =
        payload_options.prompt_mode.is_some() || payload_options.reasoning_effort.is_some();
    if !has_explicit_reasoning_control && should_use_reasoning && uses_prompt_mode_reasoning(model)
    {
        payload_options.prompt_mode = Some("reasoning".to_owned());
    } else if !has_explicit_reasoning_control
        && let Some(reasoning) = reasoning.filter(|_| should_use_reasoning)
    {
        if uses_reasoning_effort(model) {
            payload_options.reasoning_effort = Some(map_reasoning_effort(model, reasoning));
        }
    }

    build_mistral_chat_payload(model, context, payload_options)
}

pub fn build_mistral_chat_payload(
    model: &Model,
    context: &Context,
    options: MistralPayloadOptions,
) -> Value {
    let supports_images = model.input.contains(&InputKind::Image);
    let tool_call_id_normalizer = MistralToolCallIdNormalizer::default();
    let transformed_messages = transform_messages(
        &context.messages,
        model,
        Some(&|id, _model, _source| tool_call_id_normalizer.normalize(id)),
    );
    let mut messages = to_mistral_messages(&transformed_messages, supports_images);
    if let Some(system_prompt) = &context.system_prompt {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": sanitize_surrogates(system_prompt),
            }),
        );
    }

    let mut payload = json!({
        "model": model.id,
        "stream": true,
        "messages": messages,
    });

    if !context.tools.is_empty() {
        payload["tools"] = Value::Array(context.tools.iter().map(format_tool).collect());
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = options.max_tokens {
        payload["maxTokens"] = json!(max_tokens);
    }
    if let Some(tool_choice) = options.tool_choice {
        payload["toolChoice"] = format_tool_choice(tool_choice);
    }
    if let Some(prompt_mode) = options.prompt_mode {
        payload["promptMode"] = Value::String(prompt_mode);
    }
    if let Some(reasoning_effort) = options.reasoning_effort {
        payload["reasoningEffort"] = Value::String(reasoning_effort);
    }

    payload
}

pub fn build_mistral_request_headers(
    model: &Model,
    session_id: Option<&str>,
    option_headers: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut headers = model.headers.clone();
    headers.extend(option_headers.clone());
    if let Some(session_id) = session_id
        && headers
            .get("x-affinity")
            .is_none_or(|value| value.is_empty())
    {
        headers.insert("x-affinity".to_owned(), session_id.to_owned());
    }
    headers
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MistralActiveBlock {
    Text(usize),
    Thinking(usize),
}

pub fn process_mistral_chat_chunks<I>(
    chunks: I,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    model: &Model,
) -> Result<(), String>
where
    I: IntoIterator<Item = Value>,
{
    let mut processor = MistralChatStreamProcessor::new();
    for chunk in chunks {
        processor.process_chunk(chunk, output, sender, model)?;
    }
    processor.finish(output, sender)
}

#[derive(Debug, Default)]
pub struct MistralChatStreamProcessor {
    started: bool,
    active_block: Option<MistralActiveBlock>,
    tool_blocks_by_key: BTreeMap<String, usize>,
    tool_call_partial_args: BTreeMap<usize, String>,
}

impl MistralChatStreamProcessor {
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
        apply_mistral_chunk_metadata(output, model, &chunk);

        let Some(choice) = chunk
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first())
        else {
            return Ok(());
        };

        if let Some(finish_reason) = choice
            .get("finishReason")
            .or_else(|| choice.get("finish_reason"))
            .and_then(Value::as_str)
        {
            output.stop_reason = map_mistral_finish_reason(finish_reason);
        }

        let Some(delta) = choice.get("delta").and_then(Value::as_object) else {
            return Ok(());
        };

        if let Some(content) = delta.get("content") {
            process_mistral_content_delta(output, sender, &mut self.active_block, content);
        }

        if let Some(tool_calls) = delta
            .get("toolCalls")
            .or_else(|| delta.get("tool_calls"))
            .and_then(Value::as_array)
        {
            for tool_call in tool_calls {
                finish_mistral_active_block(output, sender, &mut self.active_block);
                process_mistral_tool_call_delta(
                    output,
                    sender,
                    tool_call,
                    &mut self.tool_blocks_by_key,
                    &mut self.tool_call_partial_args,
                );
            }
        }

        Ok(())
    }

    pub fn finish(
        mut self,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) -> Result<(), String> {
        if !self.started {
            self.started = true;
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        finish_mistral_active_block(output, sender, &mut self.active_block);
        finish_mistral_tool_call_blocks(output, sender, &self.tool_call_partial_args);

        if output.stop_reason == StopReason::Error {
            let message = "Provider returned an error stop reason".to_owned();
            output.error_message = Some(message.clone());
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

fn apply_mistral_chunk_metadata(output: &mut AssistantMessage, model: &Model, chunk: &Value) {
    if output.response_id.is_none() {
        if let Some(id) = chunk
            .get("id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
        {
            output.response_id = Some(id.to_owned());
        }
    }

    if let Some(usage) = chunk.get("usage") {
        output.usage = parse_mistral_usage(model, usage);
    }
}

fn parse_mistral_usage(model: &Model, usage: &Value) -> Usage {
    let input = usage_u64(usage, "promptTokens", "prompt_tokens").unwrap_or(0);
    let output_tokens = usage_u64(usage, "completionTokens", "completion_tokens").unwrap_or(0);
    let total_tokens = usage_u64(usage, "totalTokens", "total_tokens")
        .unwrap_or_else(|| input.saturating_add(output_tokens));
    let mut usage = Usage {
        input,
        output: output_tokens,
        cache_read: 0,
        cache_write: 0,
        total_tokens,
        cost: Default::default(),
    };
    calculate_cost(model, &mut usage);
    usage
}

fn usage_u64(usage: &Value, camel: &str, snake: &str) -> Option<u64> {
    usage
        .get(camel)
        .or_else(|| usage.get(snake))
        .and_then(Value::as_u64)
}

fn process_mistral_content_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    active_block: &mut Option<MistralActiveBlock>,
    content: &Value,
) {
    if let Some(text) = content.as_str() {
        if !text.is_empty() {
            push_mistral_text_delta(output, sender, active_block, text);
        }
        return;
    }

    let Some(items) = content.as_array() else {
        return;
    };
    for item in items {
        if let Some(text) = item.as_str() {
            if !text.is_empty() {
                push_mistral_text_delta(output, sender, active_block, text);
            }
            continue;
        }
        let Some(kind) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "text" => {
                if let Some(text) = item.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    push_mistral_text_delta(output, sender, active_block, text);
                }
            }
            "thinking" => {
                let thinking_delta = item
                    .get("thinking")
                    .and_then(Value::as_array)
                    .map(|parts| {
                        parts
                            .iter()
                            .filter_map(|part| part.get("text").and_then(Value::as_str))
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                if !thinking_delta.is_empty() {
                    push_mistral_thinking_delta(output, sender, active_block, &thinking_delta);
                }
            }
            _ => {}
        }
    }
}

fn push_mistral_text_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    active_block: &mut Option<MistralActiveBlock>,
    delta: &str,
) {
    let delta = sanitize_surrogates(delta);
    if !matches!(active_block, Some(MistralActiveBlock::Text(_))) {
        finish_mistral_active_block(output, sender, active_block);
        let index = output.content.len();
        output
            .content
            .push(AssistantContent::Text(TextContent::new("")));
        *active_block = Some(MistralActiveBlock::Text(index));
        sender.push(AssistantMessageEvent::TextStart {
            content_index: index,
            partial: output.clone(),
        });
    }
    let Some(MistralActiveBlock::Text(index)) = *active_block else {
        return;
    };
    if let Some(AssistantContent::Text(text)) = output.content.get_mut(index) {
        text.text.push_str(&delta);
    }
    sender.push(AssistantMessageEvent::TextDelta {
        content_index: index,
        delta,
        partial: output.clone(),
    });
}

fn push_mistral_thinking_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    active_block: &mut Option<MistralActiveBlock>,
    delta: &str,
) {
    let delta = sanitize_surrogates(delta);
    if !matches!(active_block, Some(MistralActiveBlock::Thinking(_))) {
        finish_mistral_active_block(output, sender, active_block);
        let index = output.content.len();
        output
            .content
            .push(AssistantContent::Thinking(ThinkingContent::new("")));
        *active_block = Some(MistralActiveBlock::Thinking(index));
        sender.push(AssistantMessageEvent::ThinkingStart {
            content_index: index,
            partial: output.clone(),
        });
    }
    let Some(MistralActiveBlock::Thinking(index)) = *active_block else {
        return;
    };
    if let Some(AssistantContent::Thinking(thinking)) = output.content.get_mut(index) {
        thinking.thinking.push_str(&delta);
    }
    sender.push(AssistantMessageEvent::ThinkingDelta {
        content_index: index,
        delta,
        partial: output.clone(),
    });
}

fn finish_mistral_active_block(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    active_block: &mut Option<MistralActiveBlock>,
) {
    let Some(block) = active_block.take() else {
        return;
    };
    match block {
        MistralActiveBlock::Text(index) => {
            let content = output
                .content
                .get(index)
                .and_then(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            sender.push(AssistantMessageEvent::TextEnd {
                content_index: index,
                content,
                partial: output.clone(),
            });
        }
        MistralActiveBlock::Thinking(index) => {
            let content = output
                .content
                .get(index)
                .and_then(|content| match content {
                    AssistantContent::Thinking(thinking) => Some(thinking.thinking.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            sender.push(AssistantMessageEvent::ThinkingEnd {
                content_index: index,
                content,
                partial: output.clone(),
            });
        }
    }
}

fn process_mistral_tool_call_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    tool_call: &Value,
    tool_blocks_by_key: &mut BTreeMap<String, usize>,
    tool_call_partial_args: &mut BTreeMap<usize, String>,
) {
    let function = tool_call.get("function").unwrap_or(&Value::Null);
    let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
    let call_id = tool_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty() && *id != "null")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| derive_mistral_tool_call_id(&format!("toolcall:{index}"), 0));
    let key = format!("{call_id}:{index}");
    let content_index = if let Some(content_index) = tool_blocks_by_key.get(&key).copied() {
        content_index
    } else {
        let content_index = output.content.len();
        output.content.push(AssistantContent::ToolCall(ToolCall {
            id: call_id.clone(),
            name: function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            arguments: Default::default(),
            thought_signature: None,
        }));
        tool_blocks_by_key.insert(key, content_index);
        sender.push(AssistantMessageEvent::ToolcallStart {
            content_index,
            partial: output.clone(),
        });
        content_index
    };

    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(content_index) {
        if block.name.is_empty()
            && let Some(name) = function.get("name").and_then(Value::as_str)
        {
            block.name = name.to_owned();
        }
    }

    let args_delta = match function.get("arguments") {
        Some(Value::String(arguments)) => arguments.clone(),
        Some(arguments) if !arguments.is_null() => {
            serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_owned())
        }
        _ => "{}".to_owned(),
    };
    let partial_args = tool_call_partial_args.entry(content_index).or_default();
    partial_args.push_str(&args_delta);
    if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(content_index) {
        block.arguments = parse_mistral_arguments(partial_args);
    }
    sender.push(AssistantMessageEvent::ToolcallDelta {
        content_index,
        delta: args_delta,
        partial: output.clone(),
    });
}

fn finish_mistral_tool_call_blocks(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    tool_call_partial_args: &BTreeMap<usize, String>,
) {
    for (index, partial_args) in tool_call_partial_args {
        let tool_call =
            if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(*index) {
                block.arguments = parse_mistral_arguments(partial_args);
                Some(block.clone())
            } else {
                None
            };
        if let Some(tool_call) = tool_call {
            sender.push(AssistantMessageEvent::ToolcallEnd {
                content_index: *index,
                tool_call,
                partial: output.clone(),
            });
        }
    }
}

fn parse_mistral_arguments(arguments: &str) -> serde_json::Map<String, Value> {
    parse_streaming_json(Some(arguments))
        .as_object()
        .cloned()
        .unwrap_or_default()
}

#[derive(Default)]
struct MistralToolCallIdNormalizer {
    id_map: RefCell<HashMap<String, String>>,
    reverse_map: RefCell<HashMap<String, String>>,
}

impl MistralToolCallIdNormalizer {
    fn normalize(&self, id: &str) -> String {
        if let Some(existing) = self.id_map.borrow().get(id) {
            return existing.clone();
        }

        let mut attempt = 0;
        loop {
            let candidate = derive_mistral_tool_call_id(id, attempt);
            let owner = self.reverse_map.borrow().get(&candidate).cloned();
            if owner.as_deref().is_none_or(|owner| owner == id) {
                self.id_map
                    .borrow_mut()
                    .insert(id.to_owned(), candidate.clone());
                self.reverse_map
                    .borrow_mut()
                    .insert(candidate.clone(), id.to_owned());
                return candidate;
            }
            attempt += 1;
        }
    }
}

fn derive_mistral_tool_call_id(id: &str, attempt: usize) -> String {
    let normalized = id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    if attempt == 0 && normalized.len() == MISTRAL_TOOL_CALL_ID_LENGTH {
        return normalized;
    }
    let seed_base = if normalized.is_empty() {
        id.to_owned()
    } else {
        normalized
    };
    let seed = if attempt == 0 {
        seed_base
    } else {
        format!("{seed_base}:{attempt}")
    };
    short_hash(&seed)
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(MISTRAL_TOOL_CALL_ID_LENGTH)
        .collect()
}

fn to_mistral_messages(messages: &[Message], supports_images: bool) -> Vec<Value> {
    let mut result = Vec::new();

    for message in messages {
        match message {
            Message::User(user) => {
                if let Some(message) = format_user_message(&user.content, supports_images) {
                    result.push(message);
                }
            }
            Message::Assistant(assistant) => {
                let mut content_parts = Vec::new();
                let mut tool_calls = Vec::new();
                for content in &assistant.content {
                    match content {
                        AssistantContent::Text(block) => {
                            if !block.text.trim().is_empty() {
                                content_parts.push(json!({
                                    "type": "text",
                                    "text": sanitize_surrogates(&block.text),
                                }));
                            }
                        }
                        AssistantContent::Thinking(block) => {
                            if !block.thinking.trim().is_empty() {
                                content_parts.push(json!({
                                    "type": "thinking",
                                    "thinking": [{
                                        "type": "text",
                                        "text": sanitize_surrogates(&block.thinking),
                                    }],
                                }));
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            tool_calls.push(json!({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": serde_json::to_string(&tool_call.arguments)
                                        .unwrap_or_else(|_| "{}".to_owned()),
                                },
                            }));
                        }
                    }
                }
                if !content_parts.is_empty() || !tool_calls.is_empty() {
                    let mut assistant_message = json!({ "role": "assistant" });
                    if !content_parts.is_empty() {
                        assistant_message["content"] = Value::Array(content_parts);
                    }
                    if !tool_calls.is_empty() {
                        assistant_message["toolCalls"] = Value::Array(tool_calls);
                    }
                    result.push(assistant_message);
                }
            }
            Message::ToolResult(tool_result) => {
                let text = tool_result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ToolResultContent::Text(text) => Some(sanitize_surrogates(&text.text)),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = tool_result
                    .content
                    .iter()
                    .any(|part| matches!(part, ToolResultContent::Image(_)));
                let mut content = vec![json!({
                    "type": "text",
                    "text": build_tool_result_text(
                        &text,
                        has_images,
                        supports_images,
                        tool_result.is_error,
                    ),
                })];
                if supports_images {
                    for part in &tool_result.content {
                        if let ToolResultContent::Image(image) = part {
                            content.push(json!({
                                "type": "image_url",
                                "imageUrl": format!("data:{};base64,{}", image.mime_type, image.data),
                            }));
                        }
                    }
                }
                result.push(json!({
                    "role": "tool",
                    "toolCallId": tool_result.tool_call_id,
                    "name": tool_result.tool_name,
                    "content": content,
                }));
            }
        }
    }

    result
}

fn format_user_message(content: &UserContentValue, supports_images: bool) -> Option<Value> {
    match content {
        UserContentValue::Plain(text) => Some(json!({
            "role": "user",
            "content": sanitize_surrogates(text),
        })),
        UserContentValue::Blocks(blocks) => {
            let had_images = blocks
                .iter()
                .any(|block| matches!(block, UserContent::Image(_)));
            let content = blocks
                .iter()
                .filter_map(|block| match block {
                    UserContent::Text(text) => Some(json!({
                        "type": "text",
                        "text": sanitize_surrogates(&text.text),
                    })),
                    UserContent::Image(image) if supports_images => Some(json!({
                        "type": "image_url",
                        "imageUrl": format!("data:{};base64,{}", image.mime_type, image.data),
                    })),
                    UserContent::Image(_) => None,
                })
                .collect::<Vec<_>>();
            if !content.is_empty() {
                Some(json!({ "role": "user", "content": content }))
            } else if had_images {
                Some(json!({
                    "role": "user",
                    "content": "(image omitted: model does not support images)",
                }))
            } else {
                None
            }
        }
    }
}

fn format_tool(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": strip_symbol_keys(&tool.parameters),
            "strict": false,
        },
    })
}

fn strip_symbol_keys(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(strip_symbol_keys).collect()),
        Value::Object(object) => object
            .iter()
            .map(|(key, value)| (key.clone(), strip_symbol_keys(value)))
            .collect(),
        _ => value.clone(),
    }
}

fn format_tool_choice(choice: MistralToolChoice) -> Value {
    match choice {
        MistralToolChoice::Auto => Value::String("auto".to_owned()),
        MistralToolChoice::None => Value::String("none".to_owned()),
        MistralToolChoice::Any => Value::String("any".to_owned()),
        MistralToolChoice::Required => Value::String("required".to_owned()),
        MistralToolChoice::Function { name } => json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

fn parse_mistral_tool_choice(value: &Value) -> Option<MistralToolChoice> {
    match value.as_str() {
        Some("auto") => Some(MistralToolChoice::Auto),
        Some("none") => Some(MistralToolChoice::None),
        Some("any") => Some(MistralToolChoice::Any),
        Some("required") => Some(MistralToolChoice::Required),
        Some(_) => None,
        None => {
            let object = value.as_object()?;
            if object.get("type").and_then(Value::as_str) != Some("function") {
                return None;
            }
            let name = object
                .get("function")?
                .get("name")?
                .as_str()
                .map(str::to_owned)?;
            Some(MistralToolChoice::Function { name })
        }
    }
}

fn build_tool_result_text(
    text: &str,
    has_images: bool,
    supports_images: bool,
    is_error: bool,
) -> String {
    let trimmed = text.trim();
    let error_prefix = if is_error { "[tool error] " } else { "" };

    if !trimmed.is_empty() {
        let image_suffix = if has_images && !supports_images {
            "\n[tool image omitted: model does not support images]"
        } else {
            ""
        };
        return format!("{error_prefix}{trimmed}{image_suffix}");
    }

    if has_images {
        if supports_images {
            return format!("{error_prefix}(see attached image)");
        }
        return format!("{error_prefix}(image omitted: model does not support images)");
    }

    format!("{error_prefix}(no tool output)")
}

fn uses_reasoning_effort(model: &Model) -> bool {
    matches!(
        model.id.as_str(),
        "mistral-small-2603" | "mistral-small-latest" | "mistral-medium-3.5"
    )
}

fn uses_prompt_mode_reasoning(model: &Model) -> bool {
    model.reasoning && !uses_reasoning_effort(model)
}

fn map_reasoning_effort(model: &Model, level: ThinkingLevel) -> String {
    model
        .thinking_level_map
        .get(&level)
        .and_then(Clone::clone)
        .unwrap_or_else(|| "high".to_owned())
}

fn map_mistral_finish_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::Stop,
        "length" | "model_length" => StopReason::Length,
        "tool_calls" => StopReason::ToolUse,
        "error" => StopReason::Error,
        _ => StopReason::Stop,
    }
}
