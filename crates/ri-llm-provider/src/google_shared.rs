use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    Context, ImageContent, InputKind, Message, Model, SimpleStreamOptions, StopReason, TextContent,
    ThinkingBudgets, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolResultContent, Usage,
    UserContent, UserContentValue, json_repair::sanitize_surrogates,
    message_transform::transform_messages, models::calculate_cost,
    simple_options::apply_simple_stream_defaults,
};
use serde_json::{Map, Value, json};

const JSON_SCHEMA_META_DECLARATIONS: &[&str] = &[
    "$schema",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$vocabulary",
    "$comment",
    "$defs",
    "definitions",
];

pub fn sanitize_google_schema_for_openapi(schema: &Value) -> Value {
    match schema {
        Value::Object(object) => object
            .iter()
            .filter_map(|(key, value)| {
                if JSON_SCHEMA_META_DECLARATIONS.contains(&key.as_str()) {
                    None
                } else {
                    Some((key.clone(), sanitize_google_schema_for_openapi(value)))
                }
            })
            .collect(),
        _ => schema.clone(),
    }
}

pub fn convert_google_tools(tools: &[Tool], use_parameters: bool) -> Option<Vec<Value>> {
    if tools.is_empty() {
        return None;
    }

    let function_declarations = tools
        .iter()
        .map(|tool| {
            let mut declaration = json!({
                "name": tool.name,
                "description": tool.description,
            });
            if use_parameters {
                declaration["parameters"] = sanitize_google_schema_for_openapi(&tool.parameters);
            } else {
                declaration["parametersJsonSchema"] = tool.parameters.clone();
            }
            declaration
        })
        .collect::<Vec<_>>();

    Some(vec![json!({
        "functionDeclarations": function_declarations,
    })])
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GooglePayloadOptions {
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub tool_choice: Option<String>,
    pub thinking: Option<GoogleThinkingOptions>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleThinkingOptions {
    pub enabled: bool,
    pub budget_tokens: Option<i64>,
    pub level: Option<String>,
}

pub fn build_google_simple_payload(
    model: &Model,
    context: &Context,
    options: SimpleStreamOptions,
) -> Value {
    let options = apply_simple_stream_defaults(model, options);
    let thinking = if let Some(reasoning) = options.reasoning {
        let clamped = crate::clamp_thinking_level(model, reasoning);
        let effort = if clamped == ThinkingLevel::Off {
            ThinkingLevel::High
        } else {
            clamped
        };
        if is_gemini3_pro_model(&model.id)
            || is_gemini3_flash_model(&model.id)
            || is_gemma4_model(&model.id)
        {
            Some(GoogleThinkingOptions {
                enabled: true,
                budget_tokens: None,
                level: Some(google_thinking_level(&model.id, effort).to_owned()),
            })
        } else {
            Some(GoogleThinkingOptions {
                enabled: true,
                budget_tokens: Some(google_thinking_budget(
                    &model.id,
                    effort,
                    options.thinking_budgets.as_ref(),
                )),
                level: None,
            })
        }
    } else if model.reasoning {
        Some(GoogleThinkingOptions {
            enabled: false,
            budget_tokens: None,
            level: None,
        })
    } else {
        None
    };

    build_google_payload(
        model,
        context,
        GooglePayloadOptions {
            temperature: options.stream.temperature,
            max_tokens: options.stream.max_tokens,
            thinking,
            ..Default::default()
        },
    )
}

pub fn build_google_payload(
    model: &Model,
    context: &Context,
    options: GooglePayloadOptions,
) -> Value {
    let mut config = Map::new();
    if let Some(temperature) = options.temperature {
        config.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(max_tokens) = options.max_tokens {
        config.insert("maxOutputTokens".to_owned(), json!(max_tokens));
    }
    if let Some(system_prompt) = &context.system_prompt {
        config.insert(
            "systemInstruction".to_owned(),
            json!(sanitize_surrogates(system_prompt)),
        );
    }
    if let Some(tools) = convert_google_tools(&context.tools, false) {
        config.insert("tools".to_owned(), Value::Array(tools));
    }
    if !context.tools.is_empty()
        && let Some(tool_choice) = options.tool_choice
    {
        config.insert(
            "toolConfig".to_owned(),
            json!({
                "functionCallingConfig": {
                    "mode": map_google_tool_choice(&tool_choice),
                },
            }),
        );
    }
    if let Some(thinking) = options.thinking
        && model.reasoning
    {
        config.insert(
            "thinkingConfig".to_owned(),
            if thinking.enabled {
                let mut thinking_config =
                    Map::from_iter([("includeThoughts".to_owned(), json!(true))]);
                if let Some(level) = thinking.level {
                    thinking_config.insert("thinkingLevel".to_owned(), Value::String(level));
                } else if let Some(budget_tokens) = thinking.budget_tokens {
                    thinking_config.insert("thinkingBudget".to_owned(), json!(budget_tokens));
                }
                Value::Object(thinking_config)
            } else {
                google_disabled_thinking_config(&model.id)
            },
        );
    }

    json!({
        "model": model.id,
        "contents": convert_google_messages(model, context),
        "config": Value::Object(config),
    })
}

pub fn google_disabled_thinking_config(model_id: &str) -> Value {
    if is_gemini3_pro_model(model_id) {
        json!({ "thinkingLevel": "LOW" })
    } else if is_gemini3_flash_model(model_id) || is_gemma4_model(model_id) {
        json!({ "thinkingLevel": "MINIMAL" })
    } else {
        json!({ "thinkingBudget": 0 })
    }
}

pub fn google_thinking_level(model_id: &str, effort: ThinkingLevel) -> &'static str {
    if is_gemini3_pro_model(model_id) {
        return match effort {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "LOW",
            ThinkingLevel::Medium
            | ThinkingLevel::High
            | ThinkingLevel::XHigh
            | ThinkingLevel::Off => "HIGH",
        };
    }
    if is_gemma4_model(model_id) {
        return match effort {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "MINIMAL",
            ThinkingLevel::Medium
            | ThinkingLevel::High
            | ThinkingLevel::XHigh
            | ThinkingLevel::Off => "HIGH",
        };
    }
    match effort {
        ThinkingLevel::Minimal => "MINIMAL",
        ThinkingLevel::Low => "LOW",
        ThinkingLevel::Medium => "MEDIUM",
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => "HIGH",
    }
}

pub fn google_thinking_budget(
    model_id: &str,
    effort: ThinkingLevel,
    custom_budgets: Option<&ThinkingBudgets>,
) -> i64 {
    let budget = match effort {
        ThinkingLevel::Minimal => custom_budgets.and_then(|budgets| budgets.minimal),
        ThinkingLevel::Low => custom_budgets.and_then(|budgets| budgets.low),
        ThinkingLevel::Medium => custom_budgets.and_then(|budgets| budgets.medium),
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => {
            custom_budgets.and_then(|budgets| budgets.high)
        }
    };
    if let Some(budget) = budget {
        return budget as i64;
    }

    if model_id.contains("2.5-pro") {
        match effort {
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 2_048,
            ThinkingLevel::Medium => 8_192,
            ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => 32_768,
        }
    } else if model_id.contains("2.5-flash-lite") {
        match effort {
            ThinkingLevel::Minimal => 512,
            ThinkingLevel::Low => 2_048,
            ThinkingLevel::Medium => 8_192,
            ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => 24_576,
        }
    } else if model_id.contains("2.5-flash") {
        match effort {
            ThinkingLevel::Minimal => 128,
            ThinkingLevel::Low => 2_048,
            ThinkingLevel::Medium => 8_192,
            ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => 24_576,
        }
    } else {
        -1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GoogleActiveBlock {
    Text(usize),
    Thinking(usize),
}

pub fn process_google_stream_chunks<I>(
    chunks: I,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    model: &Model,
) -> Result<(), String>
where
    I: IntoIterator<Item = Value>,
{
    let mut processor = GoogleStreamProcessor::new();
    for chunk in chunks {
        processor.process_chunk(chunk, output, sender, model);
    }
    processor.finish(output, sender)
}

#[derive(Debug, Default)]
pub struct GoogleStreamProcessor {
    started: bool,
    active_block: Option<GoogleActiveBlock>,
    tool_call_counter: u64,
}

impl GoogleStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_chunk(
        &mut self,
        chunk: Value,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
        model: &Model,
    ) {
        if !self.started {
            self.started = true;
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        if !chunk.is_object() {
            return;
        }
        apply_google_chunk_metadata(output, model, &chunk);

        let Some(candidate) = chunk
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|candidates| candidates.first())
        else {
            return;
        };

        if let Some(parts) = candidate
            .pointer("/content/parts")
            .and_then(Value::as_array)
        {
            for part in parts {
                if let Some(text_value) = part.get("text") {
                    let text = text_value.as_str().unwrap_or_default();
                    process_google_text_part(output, sender, &mut self.active_block, part, text);
                }

                if let Some(function_call) = part.get("functionCall") {
                    finish_google_active_block(output, sender, &mut self.active_block);
                    process_google_function_call_part(
                        output,
                        sender,
                        function_call,
                        part.get("thoughtSignature").and_then(Value::as_str),
                        &mut self.tool_call_counter,
                    );
                }
            }
        }

        if let Some(finish_reason) = candidate.get("finishReason").and_then(Value::as_str) {
            output.stop_reason = map_google_finish_reason(finish_reason);
            if output
                .content
                .iter()
                .any(|block| matches!(block, AssistantContent::ToolCall(_)))
            {
                output.stop_reason = StopReason::ToolUse;
            }
        }
    }

    pub fn finish(
        mut self,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) -> Result<(), String> {
        if !self.started {
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        finish_google_active_block(output, sender, &mut self.active_block);

        if output.stop_reason == StopReason::Error {
            let message = "An unknown error occurred".to_owned();
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

fn process_google_text_part(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    active_block: &mut Option<GoogleActiveBlock>,
    part: &Value,
    text: &str,
) {
    let is_thinking = is_google_thinking_part(
        part.get("thought").and_then(Value::as_bool),
        part.get("thoughtSignature").and_then(Value::as_str),
    );
    let needs_new_block = !matches!(
        (is_thinking, *active_block),
        (true, Some(GoogleActiveBlock::Thinking(_))) | (false, Some(GoogleActiveBlock::Text(_)))
    );
    if needs_new_block {
        finish_google_active_block(output, sender, active_block);
        let content_index = output.content.len();
        if is_thinking {
            output
                .content
                .push(AssistantContent::Thinking(ThinkingContent::new("")));
            *active_block = Some(GoogleActiveBlock::Thinking(content_index));
            sender.push(AssistantMessageEvent::ThinkingStart {
                content_index,
                partial: output.clone(),
            });
        } else {
            output
                .content
                .push(AssistantContent::Text(TextContent::new("")));
            *active_block = Some(GoogleActiveBlock::Text(content_index));
            sender.push(AssistantMessageEvent::TextStart {
                content_index,
                partial: output.clone(),
            });
        }
    }

    let incoming_signature = part.get("thoughtSignature").and_then(Value::as_str);
    match *active_block {
        Some(GoogleActiveBlock::Thinking(content_index)) => {
            if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(content_index) {
                block.thinking.push_str(text);
                block.thinking_signature = retain_google_thought_signature(
                    block.thinking_signature.as_deref(),
                    incoming_signature,
                );
            }
            sender.push(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta: text.to_owned(),
                partial: output.clone(),
            });
        }
        Some(GoogleActiveBlock::Text(content_index)) => {
            if let Some(AssistantContent::Text(block)) = output.content.get_mut(content_index) {
                let existing = block.text_signature.as_ref().and_then(Value::as_str);
                block.text.push_str(text);
                block.text_signature =
                    retain_google_thought_signature(existing, incoming_signature)
                        .map(Value::String);
            }
            sender.push(AssistantMessageEvent::TextDelta {
                content_index,
                delta: text.to_owned(),
                partial: output.clone(),
            });
        }
        None => {}
    }
}

fn finish_google_active_block(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    active_block: &mut Option<GoogleActiveBlock>,
) {
    let Some(block) = active_block.take() else {
        return;
    };
    match block {
        GoogleActiveBlock::Text(content_index) => {
            let content = output
                .content
                .get(content_index)
                .and_then(|block| match block {
                    AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            sender.push(AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial: output.clone(),
            });
        }
        GoogleActiveBlock::Thinking(content_index) => {
            let content = output
                .content
                .get(content_index)
                .and_then(|block| match block {
                    AssistantContent::Thinking(thinking) => Some(thinking.thinking.clone()),
                    _ => None,
                })
                .unwrap_or_default();
            sender.push(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                partial: output.clone(),
            });
        }
    }
}

fn process_google_function_call_part(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    function_call: &Value,
    thought_signature: Option<&str>,
    tool_call_counter: &mut u64,
) {
    let name = function_call
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let provided_id = function_call
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    let duplicate = provided_id.is_some_and(|id| {
        output.content.iter().any(|block| {
            matches!(
                block,
                AssistantContent::ToolCall(tool_call) if tool_call.id == id
            )
        })
    });
    let id = if let Some(id) = provided_id.filter(|_| !duplicate) {
        id.to_owned()
    } else {
        *tool_call_counter += 1;
        format!("{name}_generated_{tool_call_counter}")
    };
    let arguments = function_call
        .get("args")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let tool_call = ToolCall {
        id,
        name,
        arguments,
        thought_signature: thought_signature
            .filter(|signature| !signature.is_empty())
            .map(ToOwned::to_owned),
    };
    let content_index = output.content.len();
    output
        .content
        .push(AssistantContent::ToolCall(tool_call.clone()));
    sender.push(AssistantMessageEvent::ToolcallStart {
        content_index,
        partial: output.clone(),
    });
    let delta = serde_json::to_string(&Value::Object(tool_call.arguments.clone()))
        .unwrap_or_else(|_| "{}".to_owned());
    sender.push(AssistantMessageEvent::ToolcallDelta {
        content_index,
        delta,
        partial: output.clone(),
    });
    sender.push(AssistantMessageEvent::ToolcallEnd {
        content_index,
        tool_call,
        partial: output.clone(),
    });
}

fn apply_google_chunk_metadata(output: &mut AssistantMessage, model: &Model, chunk: &Value) {
    if output.response_id.is_none() {
        if let Some(response_id) = chunk
            .get("responseId")
            .and_then(Value::as_str)
            .filter(|response_id| !response_id.is_empty())
        {
            output.response_id = Some(response_id.to_owned());
        }
    }

    let Some(usage) = chunk.get("usageMetadata") else {
        return;
    };
    let prompt = usage
        .get("promptTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let cache_read = usage
        .get("cachedContentTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let input = prompt.saturating_sub(cache_read);
    let output_tokens = usage
        .get("candidatesTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or_default()
        .saturating_add(
            usage
                .get("thoughtsTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or_default(),
        );
    let total_tokens = usage
        .get("totalTokenCount")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            input
                .saturating_add(output_tokens)
                .saturating_add(cache_read)
        });
    output.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write: 0,
        total_tokens,
        cost: Default::default(),
    };
    calculate_cost(model, &mut output.usage);
}

fn map_google_finish_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::Stop,
        "MAX_TOKENS" => StopReason::Length,
        _ => StopReason::Error,
    }
}

fn map_google_tool_choice(choice: &str) -> &'static str {
    match choice {
        "none" => "NONE",
        "any" => "ANY",
        "auto" => "AUTO",
        _ => "AUTO",
    }
}

fn is_gemini3_pro_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("gemini-3") && lower.contains("-pro")
}

fn is_gemini3_flash_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("gemini-3") && lower.contains("-flash")
}

fn is_gemma4_model(model_id: &str) -> bool {
    let lower = model_id.to_ascii_lowercase();
    lower.contains("gemma4") || lower.contains("gemma-4")
}

pub fn is_google_thinking_part(thought: Option<bool>, _thought_signature: Option<&str>) -> bool {
    thought == Some(true)
}

pub fn retain_google_thought_signature(
    existing: Option<&str>,
    incoming: Option<&str>,
) -> Option<String> {
    if let Some(incoming) = incoming.filter(|value| !value.is_empty()) {
        Some(incoming.to_owned())
    } else {
        existing.map(ToOwned::to_owned)
    }
}

pub fn requires_google_tool_call_id(model_id: &str) -> bool {
    model_id.starts_with("claude-") || model_id.starts_with("gpt-oss-")
}

pub fn convert_google_messages(model: &Model, context: &Context) -> Vec<Value> {
    let normalizer = |id: &str, target: &Model, _source: &AssistantMessage| {
        normalize_google_tool_call_id(id, target)
    };
    let transformed_messages = transform_messages(&context.messages, model, Some(&normalizer));
    let mut contents = Vec::new();

    for message in transformed_messages {
        match message {
            Message::User(user) => {
                if let Some(content) = convert_google_user_message(&user.content) {
                    contents.push(content);
                }
            }
            Message::Assistant(assistant) => {
                let is_same_provider_and_model =
                    assistant.provider == model.provider && assistant.model == model.id;
                let parts =
                    convert_google_assistant_parts(&assistant, model, is_same_provider_and_model);
                if !parts.is_empty() {
                    contents.push(json!({
                        "role": "model",
                        "parts": parts,
                    }));
                }
            }
            Message::ToolResult(tool_result) => {
                let text_result = tool_result
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                let image_content = if model.input.contains(&InputKind::Image) {
                    tool_result
                        .content
                        .iter()
                        .filter_map(|content| match content {
                            ToolResultContent::Image(image) => Some(image),
                            ToolResultContent::Text(_) => None,
                        })
                        .collect::<Vec<_>>()
                } else {
                    Vec::new()
                };
                let has_text = !text_result.is_empty();
                let has_images = !image_content.is_empty();
                let supports_multimodal_response =
                    supports_google_multimodal_function_response(&model.id);
                let response_value = if has_text {
                    sanitize_surrogates(&text_result)
                } else if has_images {
                    "(see attached image)".to_owned()
                } else {
                    String::new()
                };
                let image_parts = image_content
                    .iter()
                    .map(|image| google_inline_data_part(image))
                    .collect::<Vec<_>>();
                let response_key = if tool_result.is_error {
                    "error"
                } else {
                    "output"
                };
                let mut function_response = json!({
                    "name": tool_result.tool_name,
                    "response": {
                        response_key: response_value,
                    },
                });
                if has_images && supports_multimodal_response {
                    function_response["parts"] = Value::Array(image_parts.clone());
                }
                if requires_google_tool_call_id(&model.id) {
                    function_response["id"] = Value::String(tool_result.tool_call_id.clone());
                }
                let function_response_part = json!({
                    "functionResponse": function_response,
                });

                if let Some(last) = contents.last_mut() {
                    let has_function_response = last["role"] == "user"
                        && last["parts"]
                            .as_array()
                            .map(|parts| {
                                parts
                                    .iter()
                                    .any(|part| part.get("functionResponse").is_some())
                            })
                            .unwrap_or(false);
                    if has_function_response {
                        last["parts"]
                            .as_array_mut()
                            .expect("parts")
                            .push(function_response_part);
                    } else {
                        contents.push(json!({
                            "role": "user",
                            "parts": [function_response_part],
                        }));
                    }
                } else {
                    contents.push(json!({
                        "role": "user",
                        "parts": [function_response_part],
                    }));
                }

                if has_images && !supports_multimodal_response {
                    let mut parts = vec![json!({ "text": "Tool result image:" })];
                    parts.extend(image_parts);
                    contents.push(json!({
                        "role": "user",
                        "parts": parts,
                    }));
                }
            }
        }
    }

    contents
}

fn convert_google_user_message(content: &UserContentValue) -> Option<Value> {
    match content {
        UserContentValue::Plain(text) => Some(json!({
            "role": "user",
            "parts": [{ "text": sanitize_surrogates(text) }],
        })),
        UserContentValue::Blocks(blocks) => {
            let parts = blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => json!({ "text": sanitize_surrogates(&text.text) }),
                    UserContent::Image(image) => google_inline_data_part(image),
                })
                .collect::<Vec<_>>();
            (!parts.is_empty()).then(|| {
                json!({
                    "role": "user",
                    "parts": parts,
                })
            })
        }
    }
}

fn convert_google_assistant_parts(
    assistant: &AssistantMessage,
    model: &Model,
    is_same_provider_and_model: bool,
) -> Vec<Value> {
    let mut parts = Vec::new();
    for block in &assistant.content {
        match block {
            AssistantContent::Text(text) => {
                if text.text.trim().is_empty() {
                    continue;
                }
                let mut part = json!({ "text": sanitize_surrogates(&text.text) });
                if let Some(signature) =
                    resolve_google_text_signature(is_same_provider_and_model, text)
                {
                    part["thoughtSignature"] = Value::String(signature);
                }
                parts.push(part);
            }
            AssistantContent::Thinking(thinking) => {
                if thinking.thinking.trim().is_empty() {
                    continue;
                }
                if is_same_provider_and_model {
                    let mut part = json!({
                        "thought": true,
                        "text": sanitize_surrogates(&thinking.thinking),
                    });
                    if let Some(signature) = resolve_google_thought_signature(
                        is_same_provider_and_model,
                        thinking.thinking_signature.as_deref(),
                    ) {
                        part["thoughtSignature"] = Value::String(signature);
                    }
                    parts.push(part);
                } else {
                    parts.push(json!({ "text": sanitize_surrogates(&thinking.thinking) }));
                }
            }
            AssistantContent::ToolCall(tool_call) => {
                let mut function_call = json!({
                    "name": tool_call.name,
                    "args": tool_call.arguments,
                });
                if requires_google_tool_call_id(&model.id) {
                    function_call["id"] = Value::String(tool_call.id.clone());
                }
                let mut part = json!({
                    "functionCall": function_call,
                });
                if let Some(signature) = resolve_google_thought_signature(
                    is_same_provider_and_model,
                    tool_call.thought_signature.as_deref(),
                ) {
                    part["thoughtSignature"] = Value::String(signature);
                }
                parts.push(part);
            }
        }
    }
    parts
}

fn resolve_google_text_signature(
    is_same_provider_and_model: bool,
    text: &TextContent,
) -> Option<String> {
    let signature = text
        .text_signature
        .as_ref()
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    resolve_google_thought_signature(is_same_provider_and_model, signature.as_deref())
}

fn resolve_google_thought_signature(
    is_same_provider_and_model: bool,
    signature: Option<&str>,
) -> Option<String> {
    signature
        .filter(|signature| {
            is_same_provider_and_model && is_valid_google_thought_signature(signature)
        })
        .map(ToOwned::to_owned)
}

fn is_valid_google_thought_signature(signature: &str) -> bool {
    if signature.is_empty() || signature.len() % 4 != 0 {
        return false;
    }

    let unpadded = signature.trim_end_matches('=');
    let padding_len = signature.len() - unpadded.len();
    padding_len <= 2
        && !unpadded.is_empty()
        && unpadded
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '+' || ch == '/')
}

fn google_inline_data_part(image: &ImageContent) -> Value {
    json!({
        "inlineData": {
            "mimeType": image.mime_type,
            "data": image.data,
        },
    })
}

fn normalize_google_tool_call_id(id: &str, model: &Model) -> String {
    if !requires_google_tool_call_id(&model.id) {
        return id.to_owned();
    }
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn supports_google_multimodal_function_response(model_id: &str) -> bool {
    get_gemini_major_version(model_id)
        .map(|version| version >= 3)
        .unwrap_or(true)
}

fn get_gemini_major_version(model_id: &str) -> Option<u64> {
    let model_id = model_id.to_ascii_lowercase();
    let rest = model_id
        .strip_prefix("gemini-live-")
        .or_else(|| model_id.strip_prefix("gemini-"))?;
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty())
        .then(|| digits.parse::<u64>().ok())
        .flatten()
}
