use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    CacheRetention, Context, InputKind, Model, StopReason, TextContent, TextSignatureV1,
    ThinkingLevel, Tool, ToolCall, ToolResultContent, Usage, UsageCost, UserContent,
    UserContentValue, calculate_cost, github_copilot_headers::build_copilot_dynamic_headers,
    json_repair::parse_streaming_json, short_hash, transform_messages,
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
    /// Forwarded `tool_choice` value ("auto"/"none"/"required" or a
    /// function-object choice).
    pub tool_choice: Option<Value>,
}

fn supports_openai_responses_tool_search(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsToolSearch"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

// OpenAI Responses rejects max_output_tokens below 16:
// https://github.com/earendil-works/pi/issues/6265
const OPENAI_RESPONSES_MIN_OUTPUT_TOKENS: u64 = 16;

pub fn build_openai_responses_payload(
    model: &Model,
    context: &Context,
    options: OpenAIResponsesPayloadOptions,
) -> Value {
    let placement = crate::deferred_tools::split_deferred_tools(
        context,
        supports_openai_responses_tool_search(model),
        |name| name.to_owned(),
    );
    let messages = convert_openai_responses_messages_with_deferred(
        model,
        context,
        &["openai", "openai-codex", "opencode"],
        true,
        &placement.deferred,
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
            payload["prompt_cache_key"] = Value::String(
                crate::openai_codex_responses::clamp_openai_prompt_cache_key(&session_id),
            );
        }
    }
    if cache_retention == CacheRetention::Long
        && supports_openai_responses_long_cache_retention(model)
    {
        payload["prompt_cache_retention"] = Value::String("24h".to_owned());
    }
    if let Some(max_tokens) = options.max_tokens.filter(|value| *value > 0) {
        // OpenAI Responses rejects max_output_tokens below 16.
        payload["max_output_tokens"] =
            Value::Number(max_tokens.max(OPENAI_RESPONSES_MIN_OUTPUT_TOKENS).into());
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if !placement.immediate.is_empty() {
        payload["tools"] = Value::Array(convert_openai_responses_tools(&placement.immediate, None));
    }
    if let Some(service_tier) = options.service_tier {
        payload["service_tier"] = Value::String(service_tier);
    }
    if let Some(tool_choice) = options.tool_choice {
        payload["tool_choice"] = tool_choice;
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
        // pi forces encrypted-reasoning replay for xAI whenever the model
        // reasons, regardless of the effort/summary branch above.
        if model.provider == "xai" {
            payload["include"] = json!(["reasoning.encrypted_content"]);
        }
    }

    payload
}

pub fn convert_openai_responses_tools(tools: &[Tool], strict: Option<bool>) -> Vec<Value> {
    convert_openai_responses_tools_with_defer(tools, strict, false)
}

pub fn convert_openai_responses_tools_with_defer(
    tools: &[Tool],
    strict: Option<bool>,
    defer_loading: bool,
) -> Vec<Value> {
    let strict = strict.unwrap_or(false);
    tools
        .iter()
        .map(|tool| {
            let mut converted = json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": strict,
            });
            if defer_loading {
                converted["defer_loading"] = Value::Bool(true);
            }
            converted
        })
        .collect()
}

pub fn resolve_openai_responses_cache_retention(
    cache_retention: Option<CacheRetention>,
) -> CacheRetention {
    resolve_openai_responses_cache_retention_scoped(
        cache_retention,
        &std::collections::BTreeMap::new(),
    )
}

/// pi resolves PI_CACHE_RETENTION through the request-scoped provider env
/// before the process environment.
pub fn resolve_openai_responses_cache_retention_scoped(
    cache_retention: Option<CacheRetention>,
    env: &std::collections::BTreeMap<String, String>,
) -> CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention;
    }
    if crate::get_provider_env_value("PI_CACHE_RETENTION", env).as_deref() == Some("long") {
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
        let affinity_format = openai_responses_session_affinity_format(model);
        if affinity_format == "openrouter" {
            headers.insert("x-session-id".to_owned(), session_id.to_owned());
        } else {
            // "openai-nosession" keeps x-client-request-id but drops session_id.
            if affinity_format == "openai" && send_openai_responses_session_id_header(model) {
                headers.insert("session_id".to_owned(), session_id.to_owned());
            }
            headers.insert("x-client-request-id".to_owned(), session_id.to_owned());
        }
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

fn openai_responses_session_affinity_format(model: &Model) -> String {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("sessionAffinityFormat"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            if model.provider == "openrouter" || model.base_url.contains("openrouter.ai") {
                "openrouter".to_owned()
            } else {
                "openai".to_owned()
            }
        })
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
        ThinkingLevel::Max => "max",
    }
    .to_owned()
}

pub fn convert_openai_responses_messages(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &[&str],
    include_system_prompt: bool,
) -> Vec<Value> {
    convert_openai_responses_messages_with_deferred(
        model,
        context,
        allowed_tool_call_providers,
        include_system_prompt,
        &[],
    )
}

pub fn convert_openai_responses_messages_with_deferred(
    model: &Model,
    context: &Context,
    allowed_tool_call_providers: &[&str],
    include_system_prompt: bool,
    deferred_tools: &[(String, Tool)],
) -> Vec<Value> {
    let mut loaded_tool_names = std::collections::BTreeSet::new();
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
                "role": if model.reasoning
                    && model
                        .compat
                        .as_ref()
                        .and_then(|compat| compat.get("supportsDeveloperRole"))
                        .and_then(Value::as_bool)
                        != Some(false)
                {
                    "developer"
                } else {
                    "system"
                },
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
                let mut text_block_index = 0usize;
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
                            // Fallback ids must be valid Responses message ids
                            // (`msg_` prefix) and unique per text block within
                            // one assistant message (pi #5148).
                            let fallback_message_id = if text_block_index == 0 {
                                format!("msg_pi_{message_index}")
                            } else {
                                format!("msg_pi_{message_index}_{text_block_index}")
                            };
                            text_block_index += 1;
                            let signature = openai_responses_text_signature_parts(&text);
                            let mut message_id = signature
                                .as_ref()
                                .map(|signature| signature.id.clone())
                                .unwrap_or(fallback_message_id);
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
                            let arguments = serde_json::to_string(&tool_call.arguments)
                                .unwrap_or_else(|_| "{}".to_owned());
                            // pi only sets `id` when an item id is present; an
                            // explicit null is a wire difference.
                            output.push(match item_id {
                                Some(item_id) => json!({
                                    "type": "function_call",
                                    "id": item_id,
                                    "call_id": call_id,
                                    "name": tool_call.name,
                                    "arguments": arguments,
                                }),
                                None => json!({
                                    "type": "function_call",
                                    "call_id": call_id,
                                    "name": tool_call.name,
                                    "arguments": arguments,
                                }),
                            });
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
                    } else if has_images {
                        "(see attached image)".to_owned()
                    } else {
                        "(no tool output)".to_owned()
                    })
                };

                messages.push(json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output,
                }));

                let mut loaded_tools = Vec::new();
                for name in tool_result.added_tool_names.as_deref().unwrap_or_default() {
                    if loaded_tool_names.contains(name) {
                        continue;
                    }
                    let Some(tool) = deferred_tools
                        .iter()
                        .find(|(deferred_name, _)| deferred_name == name)
                        .map(|(_, tool)| tool)
                    else {
                        continue;
                    };
                    loaded_tool_names.insert(name.clone());
                    loaded_tools.push(tool.clone());
                }
                if !loaded_tools.is_empty() {
                    let names = loaded_tools
                        .iter()
                        .map(|tool| tool.name.clone())
                        .collect::<Vec<_>>();
                    let search_call_id = format!(
                        "pi_tool_load_{}",
                        short_hash(&format!("{}:{}", tool_result.tool_call_id, names.join(",")))
                    );
                    messages.push(json!({
                        "type": "tool_search_call",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": { "query": names.join(" "), "limit": names.len() },
                    }));
                    messages.push(json!({
                        "type": "tool_search_output",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": convert_openai_responses_tools_with_defer(
                            &loaded_tools,
                            None,
                            true,
                        ),
                    }));
                }
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
        // pi destructures `id.split("|")` into [callId, itemId]: only the
        // second segment survives; a third pipe-separated part is dropped.
        Some((call_id, rest)) => (call_id, Some(rest.split('|').next().unwrap_or(rest))),
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

/// One in-flight output item, keyed by the event `output_index` so items
/// whose deltas interleave (pi #6009) each keep their own block.
#[derive(Debug)]
enum ResponsesOutputSlot {
    Thinking {
        content_index: usize,
    },
    Text {
        content_index: usize,
    },
    ToolCall {
        content_index: usize,
        partial_json: String,
    },
}

#[derive(Debug, Default)]
pub struct OpenAIResponsesStreamProcessor {
    /// `None` keys events that omit `output_index` (legacy fixtures), matching
    /// the upstream JS map keyed by a possibly-undefined index.
    output_slots: BTreeMap<Option<u64>, ResponsesOutputSlot>,
    /// Reasoning item id -> content index, for the `response.completed`
    /// encrypted_content backfill (pi #6608).
    reasoning_block_indexes: BTreeMap<String, usize>,
    request_service_tier: Option<String>,
    terminal: bool,
}

fn event_output_index(event: &Value) -> Option<u64> {
    event.get("output_index").and_then(Value::as_u64)
}

impl OpenAIResponsesStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_request_service_tier(request_service_tier: Option<String>) -> Self {
        Self {
            request_service_tier,
            ..Default::default()
        }
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
                if let Some(item) = event.get("item") {
                    self.create_slot(event_output_index(&event), item, output, sender);
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(content_index) = self.thinking_index(event_output_index(&event)) {
                    append_openai_responses_thinking_delta(output, sender, content_index, delta);
                }
            }
            "response.reasoning_summary_part.done" => {
                if let Some(content_index) = self.thinking_index(event_output_index(&event)) {
                    append_openai_responses_thinking_delta(output, sender, content_index, "\n\n");
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(content_index) = self.text_index(event_output_index(&event)) {
                    append_openai_responses_text_delta(output, sender, content_index, delta);
                }
            }
            "response.function_call_arguments.delta" => {
                let delta = event
                    .get("delta")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(ResponsesOutputSlot::ToolCall {
                    content_index,
                    partial_json,
                }) = self.output_slots.get_mut(&event_output_index(&event))
                {
                    partial_json.push_str(delta);
                    let content_index = *content_index;
                    let arguments = parse_arguments(partial_json);
                    if let Some(AssistantContent::ToolCall(tool_call)) =
                        output.content.get_mut(content_index)
                    {
                        tool_call.arguments = arguments;
                    }
                    sender.push(AssistantMessageEvent::ToolcallDelta {
                        content_index,
                        delta: delta.to_owned(),
                        partial: output.clone(),
                    });
                }
            }
            "response.function_call_arguments.done" => {
                let arguments = event
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                if let Some(ResponsesOutputSlot::ToolCall {
                    content_index,
                    partial_json,
                }) = self.output_slots.get_mut(&event_output_index(&event))
                {
                    let previous_partial = std::mem::replace(partial_json, arguments.to_owned());
                    let content_index = *content_index;
                    let parsed = parse_arguments(arguments);
                    if let Some(AssistantContent::ToolCall(tool_call)) =
                        output.content.get_mut(content_index)
                    {
                        tool_call.arguments = parsed;
                    }
                    if let Some(delta) = arguments.strip_prefix(previous_partial.as_str()) {
                        if !delta.is_empty() {
                            sender.push(AssistantMessageEvent::ToolcallDelta {
                                content_index,
                                delta: delta.to_owned(),
                                partial: output.clone(),
                            });
                        }
                    }
                }
            }
            "response.output_item.done" => {
                let Some(item) = event.get("item") else {
                    return Ok(());
                };
                let output_index = event_output_index(&event);
                if !self.output_slots.contains_key(&output_index) {
                    self.create_slot(output_index, item, output, sender);
                }
                match item.get("type").and_then(Value::as_str) {
                    Some("reasoning") => {
                        let Some(content_index) = self.thinking_index(output_index) else {
                            return Ok(());
                        };
                        let summary_text = openai_responses_join_reasoning_text(item, "summary");
                        let content_text = openai_responses_join_reasoning_text(item, "content");
                        let content = if let Some(AssistantContent::Thinking(thinking)) =
                            output.content.get_mut(content_index)
                        {
                            if !summary_text.is_empty() {
                                thinking.thinking = summary_text;
                            } else if !content_text.is_empty() {
                                thinking.thinking = content_text;
                            }
                            thinking.thinking_signature = Some(item.to_string());
                            thinking.thinking.clone()
                        } else {
                            String::new()
                        };
                        if let Some(id) = item.get("id").and_then(Value::as_str) {
                            self.reasoning_block_indexes
                                .insert(id.to_owned(), content_index);
                        }
                        self.output_slots.remove(&output_index);
                        sender.push(AssistantMessageEvent::ThinkingEnd {
                            content_index,
                            content,
                            partial: output.clone(),
                        });
                    }
                    Some("message") => {
                        let Some(content_index) = self.text_index(output_index) else {
                            return Ok(());
                        };
                        let item_id = item.get("id").and_then(Value::as_str);
                        let item_phase = item.get("phase").and_then(Value::as_str);
                        let final_text = item
                            .get("content")
                            .and_then(Value::as_array)
                            .map(|content| {
                                content
                                    .iter()
                                    .filter_map(|part| {
                                        match part.get("type").and_then(Value::as_str) {
                                            Some("output_text") => {
                                                part.get("text").and_then(Value::as_str)
                                            }
                                            Some("refusal") => {
                                                part.get("refusal").and_then(Value::as_str)
                                            }
                                            _ => None,
                                        }
                                    })
                                    .collect::<String>()
                            })
                            .unwrap_or_default();
                        if let Some(AssistantContent::Text(text)) =
                            output.content.get_mut(content_index)
                        {
                            text.text = final_text.clone();
                            text.text_signature =
                                openai_responses_text_signature(item_id, item_phase);
                        }
                        self.output_slots.remove(&output_index);
                        sender.push(AssistantMessageEvent::TextEnd {
                            content_index,
                            content: final_text,
                            partial: output.clone(),
                        });
                    }
                    Some("function_call") => {
                        let Some(ResponsesOutputSlot::ToolCall {
                            content_index,
                            partial_json,
                        }) = self.output_slots.get(&output_index)
                        else {
                            return Ok(());
                        };
                        let content_index = *content_index;
                        let arguments_json = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .filter(|arguments| !arguments.is_empty())
                            .map(ToOwned::to_owned)
                            .unwrap_or_else(|| {
                                if partial_json.is_empty() {
                                    "{}".to_owned()
                                } else {
                                    partial_json.clone()
                                }
                            });
                        let tool_call = if let Some(AssistantContent::ToolCall(tool_call)) =
                            output.content.get_mut(content_index)
                        {
                            tool_call.arguments = parse_arguments(&arguments_json);
                            tool_call.clone()
                        } else {
                            return Ok(());
                        };
                        self.output_slots.remove(&output_index);
                        sender.push(AssistantMessageEvent::ToolcallEnd {
                            content_index,
                            tool_call,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "response.done" if model.api != "openai-codex-responses" => {}
            "response.completed" | "response.incomplete" | "response.done" => {
                if let Some(response_output) =
                    event.pointer("/response/output").and_then(Value::as_array)
                {
                    self.backfill_reasoning_signatures(response_output, output);
                }
                if let Some(id) = event.pointer("/response/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_owned());
                }
                if let Some(usage) = event.pointer("/response/usage") {
                    let service_tier = effective_openai_responses_service_tier(
                        model,
                        event
                            .pointer("/response/service_tier")
                            .and_then(Value::as_str),
                        self.request_service_tier.as_deref(),
                    );
                    output.usage = parse_openai_responses_usage(usage, model, service_tier);
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
                // Codex nests the error payload under `error`; prefer the
                // top-level fields and fall back to the nested object.
                let code = event
                    .get("code")
                    .and_then(Value::as_str)
                    .or_else(|| event.pointer("/error/code").and_then(Value::as_str))
                    .unwrap_or("unknown");
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .or_else(|| event.pointer("/error/message").and_then(Value::as_str))
                    .unwrap_or("Unknown error");
                return Err(format!("Error Code {code}: {message}"));
            }
            "response.failed" => {
                self.terminal = true;
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

    fn create_slot(
        &mut self,
        output_index: Option<u64>,
        item: &Value,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => {
                output
                    .content
                    .push(AssistantContent::Thinking(crate::ThinkingContent::new("")));
                let content_index = output.content.len() - 1;
                self.output_slots.insert(
                    output_index,
                    ResponsesOutputSlot::Thinking { content_index },
                );
                sender.push(AssistantMessageEvent::ThinkingStart {
                    content_index,
                    partial: output.clone(),
                });
            }
            Some("message") => {
                output.content.push(AssistantContent::Text(TextContent {
                    text: String::new(),
                    text_signature: None,
                }));
                let content_index = output.content.len() - 1;
                self.output_slots
                    .insert(output_index, ResponsesOutputSlot::Text { content_index });
                sender.push(AssistantMessageEvent::TextStart {
                    content_index,
                    partial: output.clone(),
                });
            }
            Some("function_call") => {
                let partial_json = item
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
                    arguments: Map::new(),
                    thought_signature: None,
                };
                output.content.push(AssistantContent::ToolCall(tool_call));
                let content_index = output.content.len() - 1;
                self.output_slots.insert(
                    output_index,
                    ResponsesOutputSlot::ToolCall {
                        content_index,
                        partial_json,
                    },
                );
                sender.push(AssistantMessageEvent::ToolcallStart {
                    content_index,
                    partial: output.clone(),
                });
            }
            _ => {}
        }
    }

    fn thinking_index(&self, output_index: Option<u64>) -> Option<usize> {
        match self.output_slots.get(&output_index) {
            Some(ResponsesOutputSlot::Thinking { content_index }) => Some(*content_index),
            _ => None,
        }
    }

    fn text_index(&self, output_index: Option<u64>) -> Option<usize> {
        match self.output_slots.get(&output_index) {
            Some(ResponsesOutputSlot::Text { content_index }) => Some(*content_index),
            _ => None,
        }
    }

    /// Azure OpenAI can omit `reasoning.encrypted_content` from
    /// `response.output_item.done` and provide it only in
    /// `response.completed.response.output`. Backfill the persisted reasoning
    /// signature from the terminal response so `store: false` multi-turn
    /// replay stays stateless (pi #6608).
    fn backfill_reasoning_signatures(
        &self,
        response_output: &[Value],
        output: &mut AssistantMessage,
    ) {
        for item in response_output {
            if item.get("type").and_then(Value::as_str) != Some("reasoning") {
                continue;
            }
            let Some(encrypted_content) = item
                .get("encrypted_content")
                .filter(|value| value.as_str().is_some_and(|content| !content.is_empty()))
            else {
                continue;
            };
            let Some(content_index) = item
                .get("id")
                .and_then(Value::as_str)
                .and_then(|id| self.reasoning_block_indexes.get(id).copied())
            else {
                continue;
            };
            let Some(AssistantContent::Thinking(thinking)) = output.content.get_mut(content_index)
            else {
                continue;
            };
            let Some(mut stored) = thinking
                .thinking_signature
                .as_deref()
                .and_then(|signature| serde_json::from_str::<Value>(signature).ok())
            else {
                continue;
            };
            if stored
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_some_and(|content| !content.is_empty())
            {
                continue;
            }
            stored["encrypted_content"] = encrypted_content.clone();
            thinking.thinking_signature = Some(stored.to_string());
        }
    }

    pub fn finish(self, output: &mut AssistantMessage, sender: &AssistantMessageEventSender) {
        sender.push(AssistantMessageEvent::Done {
            reason: output.stop_reason,
            message: output.clone(),
        });
    }
}

fn append_openai_responses_thinking_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_index: usize,
    delta: &str,
) {
    if let Some(AssistantContent::Thinking(thinking)) = output.content.get_mut(content_index) {
        thinking.thinking.push_str(delta);
    }
    sender.push(AssistantMessageEvent::ThinkingDelta {
        content_index,
        delta: delta.to_owned(),
        partial: output.clone(),
    });
}

fn append_openai_responses_text_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_index: usize,
    delta: &str,
) {
    if let Some(AssistantContent::Text(text)) = output.content.get_mut(content_index) {
        text.text.push_str(delta);
    }
    sender.push(AssistantMessageEvent::TextDelta {
        content_index,
        delta: delta.to_owned(),
        partial: output.clone(),
    });
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
    // pi finalizes tool-call arguments with parseStreamingJson: a stream cut
    // off mid-arguments (e.g. by the output token limit) must still recover
    // the parseable prefix instead of collapsing to empty arguments.
    parse_streaming_json(Some(json_text))
        .as_object()
        .cloned()
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
    let cache_write_tokens = value
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let output = value
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let mut usage = Usage {
        // OpenAI includes cached and cache-write tokens in input_tokens, so
        // subtract both.
        input: input_tokens
            .saturating_sub(cached_tokens)
            .saturating_sub(cache_write_tokens),
        output,
        cache_read: cached_tokens,
        cache_write: cache_write_tokens,
        cache_write_1h: None,
        reasoning: Some(
            value
                .pointer("/output_tokens_details/reasoning_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        ),
        // pi: usage.total_tokens || 0 — no computed fallback.
        total_tokens: value
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0),
        cost: UsageCost::default(),
    };
    calculate_cost(model, &mut usage);
    let multiplier = openai_responses_service_tier_cost_multiplier(model, service_tier);
    usage.cost.input *= multiplier;
    usage.cost.output *= multiplier;
    usage.cost.cache_read *= multiplier;
    usage.cost.cache_write *= multiplier;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage
}

fn effective_openai_responses_service_tier<'a>(
    model: &Model,
    response_service_tier: Option<&'a str>,
    request_service_tier: Option<&'a str>,
) -> Option<&'a str> {
    if model.api == "openai-codex-responses"
        && response_service_tier == Some("default")
        && matches!(request_service_tier, Some("flex" | "priority"))
    {
        return request_service_tier;
    }
    response_service_tier.or(request_service_tier)
}

pub fn openai_responses_service_tier_cost_multiplier(
    model: &Model,
    service_tier: Option<&str>,
) -> f64 {
    // pi wires the service-tier pricing hook only for the OpenAI and Codex
    // paths; Azure usage is never multiplied.
    if model.api == "azure-openai-responses" {
        return 1.0;
    }
    match service_tier {
        Some("flex") => 0.5,
        Some("priority") if model.id == "gpt-5.5" => 2.5,
        Some("priority") => 2.0,
        _ => 1.0,
    }
}
