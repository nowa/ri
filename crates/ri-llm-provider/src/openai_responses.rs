use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    CacheRetention, Context, InputKind, Model, StopReason, TextContent, TextSignatureV1,
    ThinkingLevel, Tool, ToolCall, ToolResultContent, Usage, UsageCost, UserContent,
    UserContentValue, github_copilot_headers::build_copilot_dynamic_headers,
    parse_json_with_repair, short_hash, transform_messages,
};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAIResponsesPayloadOptions {
    pub cache_retention: Option<CacheRetention>,
    pub session_id: Option<String>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub service_tier: Option<String>,
    pub reasoning_effort: Option<ThinkingLevel>,
    pub reasoning_summary: Option<String>,
}

pub fn build_openai_responses_payload(
    model: &Model,
    context: &Context,
    options: OpenAIResponsesPayloadOptions,
) -> Value {
    let messages = convert_openai_responses_messages(
        model,
        context,
        &["openai", "openai-codex", "opencode"],
        true,
    );
    let cache_retention = resolve_openai_responses_cache_retention(options.cache_retention);
    let mut payload = json!({
        "model": model.id,
        "input": messages,
        "stream": true,
        "store": false,
    });

    if cache_retention != CacheRetention::None {
        if let Some(session_id) = options.session_id {
            payload["prompt_cache_key"] = Value::String(session_id);
        }
    }
    if cache_retention == CacheRetention::Long
        && supports_openai_responses_long_cache_retention(model)
    {
        payload["prompt_cache_retention"] = Value::String("24h".to_owned());
    }
    if let Some(max_tokens) = options.max_tokens.filter(|value| *value > 0) {
        payload["max_output_tokens"] = Value::Number(max_tokens.into());
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if !context.tools.is_empty() {
        payload["tools"] = Value::Array(convert_openai_responses_tools(&context.tools, None));
    }
    if let Some(service_tier) = options.service_tier {
        payload["service_tier"] = Value::String(service_tier);
    }
    if model.reasoning {
        let reasoning_summary = options
            .reasoning_summary
            .filter(|summary| !summary.is_empty());
        if options.reasoning_effort.is_some() || reasoning_summary.is_some() {
            let effort = options
                .reasoning_effort
                .map(|level| openai_responses_reasoning_effort(model, level))
                .unwrap_or_else(|| "medium".to_owned());
            payload["reasoning"] = json!({
                "effort": effort,
                "summary": reasoning_summary.unwrap_or_else(|| "auto".to_owned()),
            });
            payload["include"] = json!(["reasoning.encrypted_content"]);
        } else if model.provider != "github-copilot" {
            match model.thinking_level_map.get(&ThinkingLevel::Off) {
                Some(None) => {}
                Some(Some(effort)) => {
                    payload["reasoning"] = json!({ "effort": effort });
                }
                None => {
                    payload["reasoning"] = json!({ "effort": "none" });
                }
            }
        }
    }

    payload
}

pub fn convert_openai_responses_tools(tools: &[Tool], strict: Option<bool>) -> Vec<Value> {
    let strict = strict.unwrap_or(false);
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": strict,
            })
        })
        .collect()
}

pub fn resolve_openai_responses_cache_retention(
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

pub fn build_openai_responses_default_headers(
    model: &Model,
    session_id: Option<&str>,
    cache_retention: CacheRetention,
    option_headers: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    build_openai_responses_default_headers_with_context(
        model,
        None,
        session_id,
        cache_retention,
        option_headers,
    )
}

pub fn build_openai_responses_default_headers_with_context(
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
    {
        if send_openai_responses_session_id_header(model) {
            headers.insert("session_id".to_owned(), session_id.to_owned());
        }
        headers.insert("x-client-request-id".to_owned(), session_id.to_owned());
    }
    headers.extend(option_headers.clone());
    headers
}

fn supports_openai_responses_long_cache_retention(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsLongCacheRetention"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

fn send_openai_responses_session_id_header(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("sendSessionIdHeader"))
        .and_then(Value::as_bool)
        .unwrap_or(true)
}

fn openai_responses_reasoning_effort(model: &Model, level: ThinkingLevel) -> String {
    if let Some(Some(mapped)) = model.thinking_level_map.get(&level) {
        return mapped.clone();
    }
    match level {
        ThinkingLevel::Off => "none",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
    }
    .to_owned()
}

pub fn convert_openai_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &[&str],
    include_system_prompt: bool,
) -> Vec<Value> {
    let allowed_tool_call_providers = allowed_tool_call_providers
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let transformed_messages = transform_messages(
        &context.messages,
        model,
        Some(&|id, target_model, source| {
            normalize_openai_responses_tool_call_id(
                id,
                target_model,
                source,
                &allowed_tool_call_providers,
            )
        }),
    );
    let mut messages = Vec::new();

    if include_system_prompt {
        if let Some(system_prompt) = &context.system_prompt {
            messages.push(json!({
                "role": if model.reasoning { "developer" } else { "system" },
                "content": system_prompt,
            }));
        }
    }

    let mut message_index = 0usize;
    for message in transformed_messages {
        match message {
            crate::Message::User(user) => match user.content {
                UserContentValue::Plain(text) => messages.push(json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }],
                })),
                UserContentValue::Blocks(blocks) => {
                    let content = blocks
                        .into_iter()
                        .map(|block| match block {
                            UserContent::Text(text) => {
                                json!({ "type": "input_text", "text": text.text })
                            }
                            UserContent::Image(image) => json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
                            }),
                        })
                        .collect::<Vec<_>>();
                    if !content.is_empty() {
                        messages.push(json!({ "role": "user", "content": content }));
                    }
                }
            },
            crate::Message::Assistant(assistant) => {
                let is_different_model = assistant.model != model.id
                    && assistant.provider == model.provider
                    && assistant.api == model.api;
                let mut output = Vec::new();
                for block in assistant.content {
                    match block {
                        AssistantContent::Thinking(thinking) => {
                            if let Some(signature) = thinking.thinking_signature {
                                if let Ok(value) = serde_json::from_str::<Value>(&signature) {
                                    output.push(value);
                                }
                            }
                        }
                        AssistantContent::Text(text) => {
                            if text.text.trim().is_empty() {
                                continue;
                            }
                            let signature = openai_responses_text_signature_parts(&text);
                            let mut message_id = signature
                                .as_ref()
                                .map(|signature| signature.id.clone())
                                .unwrap_or_else(|| format!("msg_{message_index}"));
                            if message_id.len() > 64 {
                                message_id = format!("msg_{}", short_hash(&message_id));
                            }
                            let mut message = json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text.text,
                                    "annotations": [],
                                }],
                                "status": "completed",
                                "id": message_id,
                            });
                            if let Some(phase) = signature.and_then(|signature| signature.phase) {
                                message["phase"] = Value::String(phase);
                            }
                            output.push(message);
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let (call_id, item_id_raw) =
                                split_responses_tool_call_id(&tool_call.id);
                            let mut item_id = item_id_raw.map(ToOwned::to_owned);
                            if is_different_model
                                && item_id
                                    .as_deref()
                                    .map(|id| id.starts_with("fc_"))
                                    .unwrap_or(false)
                            {
                                item_id = None;
                            }
                            output.push(json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": tool_call.name,
                                "arguments": serde_json::to_string(&tool_call.arguments).unwrap_or_else(|_| "{}".to_owned()),
                            }));
                        }
                    }
                }
                messages.extend(output);
            }
            crate::Message::ToolResult(tool_result) => {
                let text_result = tool_result
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let has_images = tool_result
                    .content
                    .iter()
                    .any(|content| matches!(content, ToolResultContent::Image(_)));
                let has_text = !text_result.is_empty();
                let (call_id, _) = split_responses_tool_call_id(&tool_result.tool_call_id);

                let output = if has_images && model.input.contains(&InputKind::Image) {
                    let mut parts = Vec::new();
                    if has_text {
                        parts.push(json!({ "type": "input_text", "text": text_result }));
                    }
                    for block in &tool_result.content {
                        if let ToolResultContent::Image(image) = block {
                            parts.push(json!({
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": format!("data:{};base64,{}", image.mime_type, image.data),
                            }));
                        }
                    }
                    Value::Array(parts)
                } else {
                    Value::String(if has_text {
                        text_result
                    } else {
                        "(see attached image)".to_owned()
                    })
                };

                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));
            }
        }
        message_index += 1;
    }

    messages
}

pub fn normalize_openai_responses_tool_call_id(
    id: &str,
    model: &Model,
    source: &AssistantMessage,
    allowed_tool_call_providers: &BTreeSet<&str>,
) -> String {
    if !allowed_tool_call_providers.contains(model.provider.as_str()) {
        return normalize_responses_id_part(id);
    }
    if !id.contains('|') {
        return normalize_responses_id_part(id);
    }

    let (call_id, item_id) = split_responses_tool_call_id(id);
    let normalized_call_id = normalize_responses_id_part(call_id);
    let is_foreign_tool_call = source.provider != model.provider || source.api != model.api;
    let mut normalized_item_id = if is_foreign_tool_call {
        build_foreign_responses_item_id(item_id.unwrap_or_default())
    } else {
        normalize_responses_id_part(item_id.unwrap_or_default())
    };
    if !normalized_item_id.starts_with("fc_") {
        normalized_item_id = normalize_responses_id_part(&format!("fc_{normalized_item_id}"));
    }
    format!("{normalized_call_id}|{normalized_item_id}")
}

pub fn build_foreign_responses_item_id(item_id: &str) -> String {
    let normalized = format!("fc_{}", short_hash(item_id));
    if normalized.len() > 64 {
        normalized.chars().take(64).collect()
    } else {
        normalized
    }
}

fn normalize_responses_id_part(part: &str) -> String {
    let sanitized = part
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let normalized = if sanitized.len() > 64 {
        sanitized.chars().take(64).collect::<String>()
    } else {
        sanitized
    };
    normalized.trim_end_matches('_').to_owned()
}

fn split_responses_tool_call_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once('|') {
        Some((call_id, item_id)) => (call_id, Some(item_id)),
        None => (id, None),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct OpenAIResponsesTextSignature {
    id: String,
    phase: Option<String>,
}

fn openai_responses_text_signature_parts(
    text: &TextContent,
) -> Option<OpenAIResponsesTextSignature> {
    let signature = text.text_signature.as_deref()?;
    if signature.starts_with('{')
        && let Ok(parsed) = serde_json::from_str::<TextSignatureV1>(signature)
        && parsed.v == 1
    {
        let phase = parsed
            .phase
            .filter(|phase| phase == "commentary" || phase == "final_answer");
        return Some(OpenAIResponsesTextSignature {
            id: parsed.id,
            phase,
        });
    }
    Some(OpenAIResponsesTextSignature {
        id: signature.to_owned(),
        phase: None,
    })
}

fn openai_responses_text_signature(id: Option<&str>, phase: Option<&str>) -> Option<String> {
    id.map(|id| {
        serde_json::to_string(&TextSignatureV1 {
            v: 1,
            id: id.to_owned(),
            phase: phase
                .filter(|phase| *phase == "commentary" || *phase == "final_answer")
                .map(ToOwned::to_owned),
        })
        .expect("serialize text signature")
    })
}

pub fn process_openai_responses_events<I>(
    events: I,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    model: &Model,
) -> Result<(), String>
where
    I: IntoIterator<Item = Value>,
{
    let mut processor = OpenAIResponsesStreamProcessor::new();
    for event in events {
        processor.process_event(event, output, sender, model)?;
    }

    Ok(())
}

#[derive(Debug, Default)]
pub struct OpenAIResponsesStreamProcessor {
    current_block_index: Option<usize>,
    current_item_type: Option<String>,
    current_item: Option<Value>,
    current_partial_json: String,
    terminal: bool,
}

impl OpenAIResponsesStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub fn process_event(
        &mut self,
        event: Value,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
        model: &Model,
    ) -> Result<(), String> {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match event_type {
            "response.created" => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_owned());
                }
            }
            "response.output_item.added" => {
                let Some(item) = event.get("item") else {
                    return Ok(());
                };
                let Some(item_type) = item.get("type").and_then(Value::as_str) else {
                    return Ok(());
                };
                self.current_item_type = Some(item_type.to_owned());
                self.current_item = Some(item.clone());

                if item_type == "function_call" {
                    self.current_partial_json = item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    let tool_call = ToolCall {
                        id: format!(
                            "{}|{}",
                            item.get("call_id")
                                .and_then(Value::as_str)
                                .unwrap_or_default(),
                            item.get("id").and_then(Value::as_str).unwrap_or_default()
                        ),
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned(),
                        arguments: parse_arguments(&self.current_partial_json),
                        thought_signature: None,
                    };
                    output.content.push(AssistantContent::ToolCall(tool_call));
                    self.current_block_index = Some(output.content.len() - 1);
                    sender.push(AssistantMessageEvent::ToolcallStart {
                        content_index: self.current_block_index.unwrap(),
                        partial: output.clone(),
                    });
                } else if item_type == "message" {
                    let text = TextContent {
                        text: String::new(),
                        text_signature: None,
                    };
                    output.content.push(AssistantContent::Text(text));
                    self.current_block_index = Some(output.content.len() - 1);
                    sender.push(AssistantMessageEvent::TextStart {
                        content_index: self.current_block_index.unwrap(),
                        partial: output.clone(),
                    });
                } else if item_type == "reasoning" {
                    output
                        .content
                        .push(AssistantContent::Thinking(crate::ThinkingContent::new("")));
                    self.current_block_index = Some(output.content.len() - 1);
                    sender.push(AssistantMessageEvent::ThinkingStart {
                        content_index: self.current_block_index.unwrap(),
                        partial: output.clone(),
                    });
                }
            }
            "response.reasoning_summary_part.added" => {
                if self.current_item_type.as_deref() == Some("reasoning")
                    && let Some(current_item) = self.current_item.as_mut()
                    && let Some(part) = event.get("part")
                {
                    let summary = current_item.as_object_mut().and_then(|item| {
                        item.entry("summary")
                            .or_insert_with(|| Value::Array(Vec::new()))
                            .as_array_mut()
                    });
                    if let Some(summary) = summary {
                        summary.push(part.clone());
                    }
                }
            }
            "response.reasoning_summary_text.delta" => {
                if self.current_item_type.as_deref() == Some("reasoning") {
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if let Some(current_item) = self.current_item.as_mut()
                        && let Some(last_part) = current_item
                            .get_mut("summary")
                            .and_then(Value::as_array_mut)
                            .and_then(|summary| summary.last_mut())
                    {
                        if let Some(text) = last_part.get("text").and_then(Value::as_str) {
                            let updated = format!("{text}{delta}");
                            last_part["text"] = Value::String(updated);
                        }
                        self.append_openai_responses_thinking_delta(output, sender, delta);
                    }
                }
            }
            "response.reasoning_summary_part.done" => {
                if self.current_item_type.as_deref() == Some("reasoning") {
                    if let Some(current_item) = self.current_item.as_mut()
                        && let Some(last_part) = current_item
                            .get_mut("summary")
                            .and_then(Value::as_array_mut)
                            .and_then(|summary| summary.last_mut())
                    {
                        if let Some(text) = last_part.get("text").and_then(Value::as_str) {
                            let updated = format!("{text}\n\n");
                            last_part["text"] = Value::String(updated);
                        }
                        self.append_openai_responses_thinking_delta(output, sender, "\n\n");
                    }
                }
            }
            "response.reasoning_text.delta" => {
                if self.current_item_type.as_deref() == Some("reasoning") {
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    self.append_openai_responses_thinking_delta(output, sender, delta);
                }
            }
            "response.content_part.added" => {
                if self.current_item_type.as_deref() == Some("message")
                    && let Some(part) = event.get("part")
                    && matches!(
                        part.get("type").and_then(Value::as_str),
                        Some("output_text" | "refusal")
                    )
                    && let Some(current_item) = self.current_item.as_mut()
                {
                    let content = current_item.as_object_mut().and_then(|item| {
                        item.entry("content")
                            .or_insert_with(|| Value::Array(Vec::new()))
                            .as_array_mut()
                    });
                    if let Some(content) = content {
                        content.push(part.clone());
                    }
                }
            }
            "response.output_text.delta" => {
                if self.current_item_type.as_deref() == Some("message") {
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if self.append_openai_responses_message_part_delta("output_text", "text", delta)
                    {
                        self.append_openai_responses_text_delta(output, sender, delta);
                    }
                }
            }
            "response.refusal.delta" => {
                if self.current_item_type.as_deref() == Some("message") {
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if self.append_openai_responses_message_part_delta("refusal", "refusal", delta)
                    {
                        self.append_openai_responses_text_delta(output, sender, delta);
                    }
                }
            }
            "response.output_text.done" => {
                if self.current_item_type.as_deref() == Some("message") {
                    let final_text = event
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if let Some(index) = self.current_block_index
                        && let Some(AssistantContent::Text(text)) = output.content.get_mut(index)
                        && !final_text.is_empty()
                    {
                        text.text = final_text.to_owned();
                    }
                }
            }
            "response.function_call_arguments.delta" => {
                if self.current_item_type.as_deref() == Some("function_call") {
                    let delta = event
                        .get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    self.current_partial_json.push_str(delta);
                    if let Some(index) = self.current_block_index {
                        if let Some(AssistantContent::ToolCall(tool_call)) =
                            output.content.get_mut(index)
                        {
                            tool_call.arguments = parse_arguments(&self.current_partial_json);
                        }
                        sender.push(AssistantMessageEvent::ToolcallDelta {
                            content_index: index,
                            delta: delta.to_owned(),
                            partial: output.clone(),
                        });
                    }
                }
            }
            "response.function_call_arguments.done" => {
                if self.current_item_type.as_deref() == Some("function_call") {
                    let arguments = event
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let previous_partial =
                        std::mem::replace(&mut self.current_partial_json, arguments.to_owned());
                    if let Some(index) = self.current_block_index {
                        if let Some(AssistantContent::ToolCall(tool_call)) =
                            output.content.get_mut(index)
                        {
                            tool_call.arguments = parse_arguments(&self.current_partial_json);
                        }
                        if let Some(delta) = arguments.strip_prefix(&previous_partial) {
                            if !delta.is_empty() {
                                sender.push(AssistantMessageEvent::ToolcallDelta {
                                    content_index: index,
                                    delta: delta.to_owned(),
                                    partial: output.clone(),
                                });
                            }
                        }
                    }
                }
            }
            "response.output_item.done" => {
                let Some(item) = event.get("item") else {
                    return Ok(());
                };
                if item.get("type").and_then(Value::as_str) == Some("function_call") {
                    let args = if self.current_partial_json.is_empty() {
                        parse_arguments(
                            item.get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}"),
                        )
                    } else {
                        parse_arguments(&self.current_partial_json)
                    };

                    let (index, tool_call) = if let Some(index) = self.current_block_index {
                        let Some(AssistantContent::ToolCall(tool_call)) =
                            output.content.get_mut(index)
                        else {
                            return Ok(());
                        };
                        tool_call.arguments = args;
                        (index, tool_call.clone())
                    } else {
                        let tool_call = ToolCall {
                            id: format!(
                                "{}|{}",
                                item.get("call_id")
                                    .and_then(Value::as_str)
                                    .unwrap_or_default(),
                                item.get("id").and_then(Value::as_str).unwrap_or_default()
                            ),
                            name: item
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            arguments: args,
                            thought_signature: None,
                        };
                        output
                            .content
                            .push(AssistantContent::ToolCall(tool_call.clone()));
                        (output.content.len() - 1, tool_call)
                    };

                    self.current_block_index = None;
                    self.current_item_type = None;
                    self.current_item = None;
                    self.current_partial_json.clear();
                    sender.push(AssistantMessageEvent::ToolcallEnd {
                        content_index: index,
                        tool_call,
                        partial: output.clone(),
                    });
                } else if item.get("type").and_then(Value::as_str) == Some("message") {
                    let item_id = item.get("id").and_then(Value::as_str);
                    let item_phase = item.get("phase").and_then(Value::as_str);
                    let final_text = item
                        .get("content")
                        .and_then(Value::as_array)
                        .map(|content| {
                            content
                                .iter()
                                .filter_map(|part| match part.get("type").and_then(Value::as_str) {
                                    Some("output_text") => part.get("text").and_then(Value::as_str),
                                    Some("refusal") => part.get("refusal").and_then(Value::as_str),
                                    _ => None,
                                })
                                .collect::<String>()
                        })
                        .filter(|text| !text.is_empty());
                    let index = if let Some(index) = self.current_block_index {
                        if let Some(AssistantContent::Text(text)) = output.content.get_mut(index) {
                            if let Some(final_text) = &final_text {
                                text.text = final_text.clone();
                            }
                            text.text_signature =
                                openai_responses_text_signature(item_id, item_phase);
                        }
                        index
                    } else {
                        output.content.push(AssistantContent::Text(TextContent {
                            text: final_text.unwrap_or_default().to_owned(),
                            text_signature: openai_responses_text_signature(item_id, item_phase),
                        }));
                        output.content.len() - 1
                    };

                    self.current_block_index = None;
                    self.current_item_type = None;
                    self.current_item = None;
                    sender.push(AssistantMessageEvent::TextEnd {
                        content_index: index,
                        content: match &output.content[index] {
                            AssistantContent::Text(text) => text.text.clone(),
                            _ => String::new(),
                        },
                        partial: output.clone(),
                    });
                } else if item.get("type").and_then(Value::as_str) == Some("reasoning") {
                    let summary_text = openai_responses_join_reasoning_text(item, "summary");
                    let content_text = openai_responses_join_reasoning_text(item, "content");
                    let index = if let Some(index) = self.current_block_index {
                        if let Some(AssistantContent::Thinking(thinking)) =
                            output.content.get_mut(index)
                        {
                            if !summary_text.is_empty() {
                                thinking.thinking = summary_text;
                            } else if !content_text.is_empty() {
                                thinking.thinking = content_text;
                            }
                            thinking.thinking_signature = Some(item.to_string());
                        }
                        index
                    } else {
                        let thinking = if !summary_text.is_empty() {
                            summary_text
                        } else {
                            content_text
                        };
                        output
                            .content
                            .push(AssistantContent::Thinking(crate::ThinkingContent {
                                thinking,
                                thinking_signature: Some(item.to_string()),
                                redacted: false,
                            }));
                        output.content.len() - 1
                    };

                    self.current_block_index = None;
                    self.current_item_type = None;
                    self.current_item = None;
                    let content = match &output.content[index] {
                        AssistantContent::Thinking(thinking) => thinking.thinking.clone(),
                        _ => String::new(),
                    };
                    sender.push(AssistantMessageEvent::ThinkingEnd {
                        content_index: index,
                        content,
                        partial: output.clone(),
                    });
                }
            }
            "response.completed" | "response.incomplete" | "response.done" => {
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_owned());
                }
                if let Some(usage) = event.pointer("/response/usage") {
                    output.usage = parse_openai_responses_usage(
                        usage,
                        model,
                        event
                            .pointer("/response/service_tier")
                            .and_then(Value::as_str),
                    );
                }
                output.stop_reason = match event.pointer("/response/status").and_then(Value::as_str)
                {
                    Some("incomplete") => StopReason::Length,
                    Some("failed" | "cancelled") => StopReason::Error,
                    _ => StopReason::Stop,
                };
                if output
                    .content
                    .iter()
                    .any(|content| matches!(content, AssistantContent::ToolCall(_)))
                    && output.stop_reason == StopReason::Stop
                {
                    output.stop_reason = StopReason::ToolUse;
                }
                self.terminal = true;
            }
            "error" => {
                let code = event
                    .get("code")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown error");
                return Err(format!("Error Code {code}: {message}"));
            }
            "response.failed" => {
                let message = if let Some(error) = event.pointer("/response/error") {
                    let code = error
                        .get("code")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    let message = error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("no message");
                    format!("{code}: {message}")
                } else if let Some(reason) = event
                    .pointer("/response/incomplete_details/reason")
                    .and_then(Value::as_str)
                {
                    format!("incomplete: {reason}")
                } else {
                    "Unknown error (no error details in response)".to_owned()
                };
                return Err(message.to_owned());
            }
            _ => {}
        }

        Ok(())
    }

    fn append_openai_responses_thinking_delta(
        &self,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
        delta: &str,
    ) {
        if let Some(index) = self.current_block_index {
            if let Some(AssistantContent::Thinking(thinking)) = output.content.get_mut(index) {
                thinking.thinking.push_str(delta);
            }
            sender.push(AssistantMessageEvent::ThinkingDelta {
                content_index: index,
                delta: delta.to_owned(),
                partial: output.clone(),
            });
        }
    }

    fn append_openai_responses_message_part_delta(
        &mut self,
        part_type: &str,
        text_field: &str,
        delta: &str,
    ) -> bool {
        let Some(current_item) = self.current_item.as_mut() else {
            return false;
        };
        let Some(last_part) = current_item
            .get_mut("content")
            .and_then(Value::as_array_mut)
            .and_then(|content| content.last_mut())
        else {
            return false;
        };
        if last_part.get("type").and_then(Value::as_str) != Some(part_type) {
            return false;
        }
        let updated = format!(
            "{}{}",
            last_part
                .get(text_field)
                .and_then(Value::as_str)
                .unwrap_or_default(),
            delta
        );
        last_part[text_field] = Value::String(updated);
        true
    }

    fn append_openai_responses_text_delta(
        &self,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
        delta: &str,
    ) {
        if let Some(index) = self.current_block_index {
            if let Some(AssistantContent::Text(text)) = output.content.get_mut(index) {
                text.text.push_str(delta);
            }
            sender.push(AssistantMessageEvent::TextDelta {
                content_index: index,
                delta: delta.to_owned(),
                partial: output.clone(),
            });
        }
    }

    pub fn finish(self, output: &mut AssistantMessage, sender: &AssistantMessageEventSender) {
        sender.push(AssistantMessageEvent::Done {
            reason: output.stop_reason,
            message: output.clone(),
        });
    }
}

fn openai_responses_join_reasoning_text(item: &Value, field: &str) -> String {
    item.get(field)
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default()
}

fn parse_arguments(json_text: &str) -> Map<String, Value> {
    if json_text.trim().is_empty() {
        return Map::new();
    }
    parse_json_with_repair::<Value>(json_text)
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

pub fn parse_openai_responses_usage(
    value: &Value,
    model: &Model,
    service_tier: Option<&str>,
) -> Usage {
    let input_tokens = value
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let cached_tokens = value
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let output = value
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let mut usage = Usage {
        input: input_tokens.saturating_sub(cached_tokens),
        output,
        cache_read: cached_tokens,
        cache_write: 0,
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(input_tokens + output),
        cost: UsageCost::default(),
    };
    usage.cost.input = (model.cost.input / 1_000_000.0) * usage.input as f64;
    usage.cost.output = (model.cost.output / 1_000_000.0) * usage.output as f64;
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
    let multiplier = openai_responses_service_tier_cost_multiplier(model, service_tier);
    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage
}

pub fn openai_responses_service_tier_cost_multiplier(
    model: &Model,
    service_tier: Option<&str>,
) -> f64 {
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") if model.id == "gpt-5.5" => 2.5,
        Some("priority") => 2.0,
        _ => 1.0,
    }
}
