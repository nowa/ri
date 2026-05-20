use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    CacheRetention, Context, Message, Model, SimpleStreamOptions, StopReason, TextContent,
    ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolResultContent, Usage, UserContent,
    UserContentValue,
    anthropic_compat::{from_claude_code_tool_name, to_claude_code_tool_name},
    json_repair::{parse_json_with_repair, parse_streaming_json, sanitize_surrogates},
    message_transform::transform_messages,
};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

const ANTHROPIC_MESSAGE_EVENTS: &[&str] = &[
    "message_start",
    "message_delta",
    "message_stop",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicSseEvent {
    pub event: Option<String>,
    pub data: String,
    pub raw: Vec<String>,
}

pub fn parse_anthropic_sse_messages(body: &str) -> Vec<AnthropicSseEvent> {
    let mut events = Vec::new();
    let mut event: Option<String> = None;
    let mut data = Vec::new();
    let mut raw = Vec::new();

    let flush = |events: &mut Vec<AnthropicSseEvent>,
                 event: &mut Option<String>,
                 data: &mut Vec<String>,
                 raw: &mut Vec<String>| {
        if event.is_none() && data.is_empty() && raw.is_empty() {
            return;
        }
        events.push(AnthropicSseEvent {
            event: event.take(),
            data: data.join("\n"),
            raw: std::mem::take(raw),
        });
        data.clear();
    };

    for line in body.lines() {
        if line.is_empty() {
            flush(&mut events, &mut event, &mut data, &mut raw);
            continue;
        }
        raw.push(line.to_owned());
        if let Some(value) = line.strip_prefix("event:") {
            event = Some(value.trim_start().to_owned());
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start().to_owned());
        }
    }
    flush(&mut events, &mut event, &mut data, &mut raw);

    events
}

pub fn process_anthropic_sse_body(model: &Model, body: &str) -> Result<AssistantMessage, String> {
    let mut output = AssistantMessage {
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
        timestamp: 0,
    };
    let mut content_indexes: BTreeMap<u64, usize> = BTreeMap::new();
    let mut partial_tool_json: BTreeMap<u64, String> = BTreeMap::new();
    let mut stopped = false;

    for sse in parse_anthropic_sse_messages(body) {
        let event_name = sse.event.as_deref().unwrap_or_default();
        if !ANTHROPIC_MESSAGE_EVENTS.contains(&event_name) {
            continue;
        }
        if stopped && event_name != "message_stop" {
            continue;
        }
        let event: Value = parse_json_with_repair(&sse.data).map_err(|error| {
            format!(
                "Could not parse Anthropic SSE event {event_name}: {error}; data={}; raw={}",
                sse.data,
                sse.raw.join("\\n")
            )
        })?;
        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(id) = event.pointer("/message/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_owned());
                }
                if let Some(usage) = event.pointer("/message/usage") {
                    output.usage = parse_anthropic_usage(usage, &output.usage);
                }
            }
            Some("content_block_start") => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let block = event.get("content_block").unwrap_or(&Value::Null);
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        output
                            .content
                            .push(AssistantContent::Text(TextContent::new(text)));
                        content_indexes.insert(index, output.content.len() - 1);
                    }
                    Some("tool_use") => {
                        let input = block
                            .get("input")
                            .and_then(Value::as_object)
                            .cloned()
                            .unwrap_or_default();
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            name: block
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            arguments: input,
                            thought_signature: None,
                        }));
                        content_indexes.insert(index, output.content.len() - 1);
                    }
                    _ => {}
                }
            }
            Some("content_block_delta") => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let Some(content_index) = content_indexes.get(&index).copied() else {
                    continue;
                };
                let delta = event.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str)
                            && let Some(AssistantContent::Text(block)) =
                                output.content.get_mut(content_index)
                        {
                            block.text.push_str(text);
                        }
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let scratch = partial_tool_json.entry(index).or_default();
                        scratch.push_str(partial);
                        if let Some(AssistantContent::ToolCall(tool_call)) =
                            output.content.get_mut(content_index)
                        {
                            tool_call.arguments = parse_streaming_json(Some(scratch))
                                .as_object()
                                .cloned()
                                .unwrap_or_else(Map::new);
                        }
                    }
                    _ => {}
                }
            }
            Some("message_delta") => {
                if let Some(stop_reason) =
                    event.pointer("/delta/stop_reason").and_then(Value::as_str)
                {
                    output.stop_reason = map_anthropic_stop_reason(stop_reason);
                }
                if let Some(usage) = event.get("usage") {
                    output.usage = parse_anthropic_usage(usage, &output.usage);
                }
            }
            Some("message_stop") => {
                stopped = true;
            }
            _ => {}
        }
    }

    Ok(output)
}

#[derive(Debug, Default)]
pub struct AnthropicStreamProcessor {
    started: bool,
    content_indexes: BTreeMap<u64, usize>,
    partial_tool_json: BTreeMap<u64, String>,
    stopped: bool,
    tools: Vec<Tool>,
    use_claude_code_tool_names: bool,
}

impl AnthropicStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tool_name_options(tools: Vec<Tool>, use_claude_code_tool_names: bool) -> Self {
        Self {
            tools,
            use_claude_code_tool_names,
            ..Default::default()
        }
    }

    pub fn with_claude_code_tool_names(tools: Vec<Tool>) -> Self {
        Self::with_tool_name_options(tools, true)
    }

    fn inbound_tool_name(&self, name: &str) -> String {
        anthropic_inbound_tool_name(name, &self.tools, self.use_claude_code_tool_names)
    }

    pub fn process_event(
        &mut self,
        event: Value,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) -> Result<(), String> {
        if !self.started {
            self.started = true;
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !ANTHROPIC_MESSAGE_EVENTS.contains(&event_type) {
            return Ok(());
        }
        if self.stopped && event_type != "message_stop" {
            return Ok(());
        }

        match event_type {
            "message_start" => {
                if let Some(id) = event.pointer("/message/id").and_then(Value::as_str) {
                    output.response_id = Some(id.to_owned());
                }
                if let Some(usage) = event.pointer("/message/usage") {
                    output.usage = parse_anthropic_usage(usage, &output.usage);
                }
            }
            "content_block_start" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let block = event.get("content_block").unwrap_or(&Value::Null);
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        output
                            .content
                            .push(AssistantContent::Text(TextContent::new("")));
                        let content_index = output.content.len() - 1;
                        self.content_indexes.insert(index, content_index);
                        sender.push(AssistantMessageEvent::TextStart {
                            content_index,
                            partial: output.clone(),
                        });
                        if let Some(text) = block
                            .get("text")
                            .and_then(Value::as_str)
                            .filter(|text| !text.is_empty())
                        {
                            push_anthropic_text_delta(output, sender, content_index, text);
                        }
                    }
                    Some("thinking") => {
                        output
                            .content
                            .push(AssistantContent::Thinking(ThinkingContent {
                                thinking: String::new(),
                                thinking_signature: Some(String::new()),
                                redacted: false,
                            }));
                        let content_index = output.content.len() - 1;
                        self.content_indexes.insert(index, content_index);
                        sender.push(AssistantMessageEvent::ThinkingStart {
                            content_index,
                            partial: output.clone(),
                        });
                    }
                    Some("redacted_thinking") => {
                        output
                            .content
                            .push(AssistantContent::Thinking(ThinkingContent {
                                thinking: "[Reasoning redacted]".to_owned(),
                                thinking_signature: block
                                    .get("data")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned),
                                redacted: true,
                            }));
                        let content_index = output.content.len() - 1;
                        self.content_indexes.insert(index, content_index);
                        sender.push(AssistantMessageEvent::ThinkingStart {
                            content_index,
                            partial: output.clone(),
                        });
                    }
                    Some("tool_use") => {
                        let name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let input = block
                            .get("input")
                            .and_then(Value::as_object)
                            .cloned()
                            .unwrap_or_default();
                        output.content.push(AssistantContent::ToolCall(ToolCall {
                            id: block
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            name: self.inbound_tool_name(name),
                            arguments: input,
                            thought_signature: None,
                        }));
                        let content_index = output.content.len() - 1;
                        self.content_indexes.insert(index, content_index);
                        self.partial_tool_json.insert(index, String::new());
                        sender.push(AssistantMessageEvent::ToolcallStart {
                            content_index,
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let Some(content_index) = self.content_indexes.get(&index).copied() else {
                    return Ok(());
                };
                let delta = event.get("delta").unwrap_or(&Value::Null);
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        if let Some(text) = delta.get("text").and_then(Value::as_str) {
                            push_anthropic_text_delta(output, sender, content_index, text);
                        }
                    }
                    Some("thinking_delta") => {
                        if let Some(thinking) = delta.get("thinking").and_then(Value::as_str) {
                            push_anthropic_thinking_delta(output, sender, content_index, thinking);
                        }
                    }
                    Some("signature_delta") => {
                        if let Some(signature) = delta.get("signature").and_then(Value::as_str)
                            && let Some(AssistantContent::Thinking(block)) =
                                output.content.get_mut(content_index)
                        {
                            block
                                .thinking_signature
                                .get_or_insert_with(String::new)
                                .push_str(signature);
                        }
                    }
                    Some("input_json_delta") => {
                        let partial = delta
                            .get("partial_json")
                            .and_then(Value::as_str)
                            .unwrap_or_default();
                        let scratch = self.partial_tool_json.entry(index).or_default();
                        scratch.push_str(partial);
                        if let Some(AssistantContent::ToolCall(tool_call)) =
                            output.content.get_mut(content_index)
                        {
                            tool_call.arguments = parse_streaming_json(Some(scratch))
                                .as_object()
                                .cloned()
                                .unwrap_or_else(Map::new);
                        }
                        sender.push(AssistantMessageEvent::ToolcallDelta {
                            content_index,
                            delta: partial.to_owned(),
                            partial: output.clone(),
                        });
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let Some(content_index) = self.content_indexes.remove(&index) else {
                    return Ok(());
                };
                if let Some(partial) = self.partial_tool_json.remove(&index)
                    && let Some(AssistantContent::ToolCall(block)) =
                        output.content.get_mut(content_index)
                {
                    block.arguments = parse_streaming_json(Some(&partial))
                        .as_object()
                        .cloned()
                        .unwrap_or_else(Map::new);
                }
                match output.content.get(content_index) {
                    Some(AssistantContent::Text(block)) => {
                        sender.push(AssistantMessageEvent::TextEnd {
                            content_index,
                            content: block.text.clone(),
                            partial: output.clone(),
                        });
                    }
                    Some(AssistantContent::Thinking(block)) => {
                        sender.push(AssistantMessageEvent::ThinkingEnd {
                            content_index,
                            content: block.thinking.clone(),
                            partial: output.clone(),
                        });
                    }
                    Some(AssistantContent::ToolCall(block)) => {
                        sender.push(AssistantMessageEvent::ToolcallEnd {
                            content_index,
                            tool_call: block.clone(),
                            partial: output.clone(),
                        });
                    }
                    None => {}
                }
            }
            "message_delta" => {
                if let Some(stop_reason) =
                    event.pointer("/delta/stop_reason").and_then(Value::as_str)
                {
                    output.stop_reason = map_anthropic_stop_reason(stop_reason);
                }
                if let Some(usage) = event.get("usage") {
                    output.usage = parse_anthropic_usage(usage, &output.usage);
                }
            }
            "message_stop" => {
                self.stopped = true;
            }
            _ => {}
        }

        Ok(())
    }

    pub fn finish(self, output: &mut AssistantMessage, sender: &AssistantMessageEventSender) {
        sender.push(AssistantMessageEvent::Done {
            reason: output.stop_reason,
            message: output.clone(),
        });
    }
}

fn push_anthropic_text_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_index: usize,
    delta: &str,
) {
    if let Some(AssistantContent::Text(block)) = output.content.get_mut(content_index) {
        block.text.push_str(delta);
    }
    sender.push(AssistantMessageEvent::TextDelta {
        content_index,
        delta: delta.to_owned(),
        partial: output.clone(),
    });
}

fn push_anthropic_thinking_delta(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_index: usize,
    delta: &str,
) {
    if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(content_index) {
        block.thinking.push_str(delta);
    }
    sender.push(AssistantMessageEvent::ThinkingDelta {
        content_index,
        delta: delta.to_owned(),
        partial: output.clone(),
    });
}

fn parse_anthropic_usage(usage: &Value, previous: &Usage) -> Usage {
    let input = usage
        .get("input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(previous.input);
    let output = usage
        .get("output_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(previous.output);
    let cache_read = usage
        .get("cache_read_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(previous.cache_read);
    let cache_write = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(previous.cache_write);
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        cost: Default::default(),
    }
}

fn map_anthropic_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Stop,
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicPayloadOptions {
    pub cache_retention: Option<CacheRetention>,
    pub thinking_enabled: Option<bool>,
    pub thinking_budget_tokens: Option<u64>,
    pub effort: Option<String>,
    pub thinking_display: Option<String>,
    pub use_claude_code_tool_names: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicClientOptions {
    pub api_key: String,
    pub interleaved_thinking: bool,
    pub use_fine_grained_tool_streaming_beta: bool,
    pub headers: BTreeMap<String, String>,
    pub session_id: Option<String>,
    pub cache_retention: Option<CacheRetention>,
}

impl Default for AnthropicClientOptions {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            interleaved_thinking: true,
            use_fine_grained_tool_streaming_beta: false,
            headers: BTreeMap::new(),
            session_id: None,
            cache_retention: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicClientConfig {
    pub api_key: Option<String>,
    pub auth_token: Option<String>,
    pub base_url: String,
    pub default_headers: BTreeMap<String, String>,
    pub is_oauth_token: bool,
}

pub fn build_anthropic_simple_payload(
    model: &Model,
    context: &Context,
    options: SimpleStreamOptions,
) -> Value {
    build_anthropic_simple_payload_for_client(model, context, options, false)
}

pub fn build_anthropic_simple_payload_for_client(
    model: &Model,
    context: &Context,
    options: SimpleStreamOptions,
    use_claude_code_tool_names: bool,
) -> Value {
    let mut payload_options = AnthropicPayloadOptions::default();
    if let Some(reasoning) = options.reasoning {
        payload_options.thinking_enabled = Some(true);
        if supports_anthropic_adaptive_thinking(&model.id) {
            payload_options.effort =
                Some(map_anthropic_thinking_level_to_effort(model, reasoning).to_owned());
        } else {
            payload_options.thinking_budget_tokens = Some(anthropic_thinking_budget(
                reasoning,
                options.thinking_budgets.as_ref(),
            ));
        }
    } else {
        payload_options.thinking_enabled = Some(false);
    }
    payload_options.use_claude_code_tool_names = use_claude_code_tool_names;
    build_anthropic_payload(model, context, payload_options)
}

pub fn build_anthropic_payload(
    model: &Model,
    context: &Context,
    options: AnthropicPayloadOptions,
) -> Value {
    let cache_retention = resolve_anthropic_cache_retention(options.cache_retention);
    let cache_control = anthropic_cache_control(model, cache_retention);
    let use_claude_code_tool_names = options.use_claude_code_tool_names;
    let mut payload = json!({
        "model": model.id,
        "messages": convert_anthropic_messages(
            context,
            model,
            cache_control.as_ref(),
            use_claude_code_tool_names,
        ),
        "max_tokens": model.max_tokens / 3,
        "stream": true,
    });

    if let Some(system_prompt) = &context.system_prompt {
        let mut block = json!({
            "type": "text",
            "text": system_prompt,
        });
        if let Some(cache_control) = &cache_control {
            block["cache_control"] = cache_control.clone();
        }
        payload["system"] = Value::Array(vec![block]);
    }

    if !context.tools.is_empty() {
        let last_tool_index = context.tools.len().saturating_sub(1);
        payload["tools"] = Value::Array(
            context
                .tools
                .iter()
                .enumerate()
                .map(|(index, tool)| {
                    format_anthropic_tool(
                        tool,
                        supports_anthropic_eager_tool_input_streaming(model),
                        (index == last_tool_index
                            && supports_anthropic_cache_control_on_tools(model))
                        .then_some(cache_control.as_ref())
                        .flatten(),
                        use_claude_code_tool_names,
                    )
                })
                .collect(),
        );
    }

    if model.reasoning {
        match options.thinking_enabled {
            Some(true) => {
                let display = options
                    .thinking_display
                    .unwrap_or_else(|| "summarized".to_owned());
                if supports_anthropic_adaptive_thinking(&model.id) {
                    payload["thinking"] = json!({ "type": "adaptive", "display": display });
                    if let Some(effort) = options.effort {
                        payload["output_config"] = json!({ "effort": effort });
                    }
                } else {
                    payload["thinking"] = json!({
                        "type": "enabled",
                        "budget_tokens": options.thinking_budget_tokens.unwrap_or(1_024),
                        "display": display,
                    });
                }
            }
            Some(false) => {
                payload["thinking"] = json!({ "type": "disabled" });
            }
            None => {}
        }
    }

    payload
}

pub fn build_anthropic_client_config(
    model: &Model,
    context: &Context,
    options: AnthropicClientOptions,
) -> AnthropicClientConfig {
    let beta_features = anthropic_beta_features(
        model,
        options.interleaved_thinking,
        options.use_fine_grained_tool_streaming_beta,
    );

    if model.provider == "github-copilot" {
        let mut headers = anthropic_browser_headers(&beta_features);
        merge_headers(&mut headers, &model.headers);
        merge_headers(&mut headers, &build_copilot_dynamic_headers(context));
        merge_headers(&mut headers, &options.headers);
        return AnthropicClientConfig {
            api_key: None,
            auth_token: Some(options.api_key),
            base_url: model.base_url.clone(),
            default_headers: headers,
            is_oauth_token: false,
        };
    }

    if model.provider == "cloudflare-ai-gateway" {
        let cache_retention = resolve_anthropic_cache_retention(options.cache_retention);
        let mut headers = anthropic_browser_headers(&beta_features);
        if !options.api_key.is_empty() {
            headers.insert(
                "cf-aig-authorization".to_owned(),
                format!("Bearer {}", options.api_key),
            );
        }
        if let Some(session_id) = options.session_id
            && cache_retention != CacheRetention::None
            && supports_anthropic_session_affinity(model)
        {
            headers.insert("x-session-affinity".to_owned(), session_id);
        }
        merge_headers(&mut headers, &model.headers);
        merge_headers(&mut headers, &options.headers);
        return AnthropicClientConfig {
            api_key: None,
            auth_token: None,
            base_url: model.base_url.clone(),
            default_headers: headers,
            is_oauth_token: false,
        };
    }

    if is_anthropic_oauth_token(&options.api_key) {
        let mut headers = anthropic_browser_headers(&[]);
        headers.insert(
            "anthropic-beta".to_owned(),
            ["claude-code-20250219", "oauth-2025-04-20"]
                .into_iter()
                .chain(beta_features.iter().copied())
                .collect::<Vec<_>>()
                .join(","),
        );
        headers.insert("user-agent".to_owned(), "claude-cli/2.1.75".to_owned());
        headers.insert("x-app".to_owned(), "cli".to_owned());
        merge_headers(&mut headers, &model.headers);
        merge_headers(&mut headers, &options.headers);
        return AnthropicClientConfig {
            api_key: None,
            auth_token: Some(options.api_key),
            base_url: model.base_url.clone(),
            default_headers: headers,
            is_oauth_token: true,
        };
    }

    let cache_retention = resolve_anthropic_cache_retention(options.cache_retention);
    let mut headers = anthropic_browser_headers(&beta_features);
    if let Some(session_id) = options.session_id
        && cache_retention != CacheRetention::None
        && supports_anthropic_session_affinity(model)
    {
        headers.insert("x-session-affinity".to_owned(), session_id);
    }
    merge_headers(&mut headers, &model.headers);
    merge_headers(&mut headers, &options.headers);

    AnthropicClientConfig {
        api_key: Some(options.api_key),
        auth_token: None,
        base_url: model.base_url.clone(),
        default_headers: headers,
        is_oauth_token: false,
    }
}

pub fn build_anthropic_default_headers(
    model: &Model,
    context: &Context,
) -> BTreeMap<String, String> {
    let mut headers = model.headers.clone();
    if should_use_anthropic_fine_grained_tool_streaming_beta(model, context) {
        headers.insert(
            "anthropic-beta".to_owned(),
            "fine-grained-tool-streaming-2025-05-14".to_owned(),
        );
    }
    headers
}

fn anthropic_beta_features(
    model: &Model,
    interleaved_thinking: bool,
    use_fine_grained_tool_streaming_beta: bool,
) -> Vec<&'static str> {
    let mut beta_features = Vec::new();
    if use_fine_grained_tool_streaming_beta {
        beta_features.push("fine-grained-tool-streaming-2025-05-14");
    }
    if interleaved_thinking && !supports_anthropic_adaptive_thinking(&model.id) {
        beta_features.push("interleaved-thinking-2025-05-14");
    }
    beta_features
}

fn anthropic_browser_headers(beta_features: &[&str]) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::from([
        ("accept".to_owned(), "application/json".to_owned()),
        (
            "anthropic-dangerous-direct-browser-access".to_owned(),
            "true".to_owned(),
        ),
    ]);
    if !beta_features.is_empty() {
        headers.insert("anthropic-beta".to_owned(), beta_features.join(","));
    }
    headers
}

fn merge_headers(target: &mut BTreeMap<String, String>, source: &BTreeMap<String, String>) {
    for (key, value) in source {
        target.insert(key.clone(), value.clone());
    }
}

fn is_anthropic_oauth_token(api_key: &str) -> bool {
    api_key.contains("sk-ant-oat")
}

fn supports_anthropic_session_affinity(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("sendSessionAffinityHeaders"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            model.provider == "fireworks"
                || (model.provider == "cloudflare-ai-gateway"
                    && model.base_url.contains("anthropic"))
        })
}

fn build_copilot_dynamic_headers(context: &Context) -> BTreeMap<String, String> {
    let initiator = match context.messages.last() {
        Some(Message::User(_)) | None => "user",
        Some(_) => "agent",
    };
    let mut headers = BTreeMap::from([
        ("X-Initiator".to_owned(), initiator.to_owned()),
        ("Openai-Intent".to_owned(), "conversation-edits".to_owned()),
    ]);
    if has_copilot_vision_input(context) {
        headers.insert("Copilot-Vision-Request".to_owned(), "true".to_owned());
    }
    headers
}

fn has_copilot_vision_input(context: &Context) -> bool {
    context.messages.iter().any(|message| match message {
        Message::User(user) => match &user.content {
            crate::UserContentValue::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, crate::UserContent::Image(_))),
            crate::UserContentValue::Plain(_) => false,
        },
        Message::ToolResult(result) => result
            .content
            .iter()
            .any(|block| matches!(block, crate::ToolResultContent::Image(_))),
        Message::Assistant(_) => false,
    })
}

fn convert_anthropic_messages(
    context: &Context,
    model: &Model,
    cache_control: Option<&Value>,
    use_claude_code_tool_names: bool,
) -> Vec<Value> {
    let transformed_messages = transform_messages(
        &context.messages,
        model,
        Some(&|id, _model, _source| normalize_anthropic_tool_call_id(id)),
    );
    let mut messages = Vec::new();
    let mut index = 0;
    while index < transformed_messages.len() {
        match &transformed_messages[index] {
            Message::User(user) => match &user.content {
                UserContentValue::Plain(text) => {
                    if !text.trim().is_empty() {
                        messages.push(json!({
                            "role": "user",
                            "content": sanitize_surrogates(text),
                        }));
                    }
                }
                UserContentValue::Blocks(blocks) => {
                    let content = blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserContent::Text(text) => {
                                let text = sanitize_surrogates(&text.text);
                                (!text.trim().is_empty())
                                    .then(|| json!({ "type": "text", "text": text }))
                            }
                            UserContent::Image(image) => Some(json!({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image.mime_type,
                                    "data": image.data,
                                },
                            })),
                        })
                        .collect::<Vec<_>>();
                    if !content.is_empty() {
                        messages.push(json!({ "role": "user", "content": content }));
                    }
                }
            },
            Message::Assistant(assistant) => {
                let content = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) => {
                            let text = sanitize_surrogates(&text.text);
                            (!text.trim().is_empty()).then(|| {
                                json!({
                                    "type": "text",
                                    "text": text,
                                })
                            })
                        }
                        AssistantContent::Thinking(thinking) if thinking.redacted => thinking
                            .thinking_signature
                            .as_deref()
                            .filter(|signature| !signature.trim().is_empty())
                            .map(|signature| {
                                json!({
                                    "type": "redacted_thinking",
                                    "data": signature,
                                })
                            }),
                        AssistantContent::Thinking(thinking) => {
                            let thinking_text = sanitize_surrogates(&thinking.thinking);
                            if thinking_text.trim().is_empty() {
                                None
                            } else if let Some(signature) = thinking
                                .thinking_signature
                                .as_deref()
                                .filter(|signature| !signature.trim().is_empty())
                            {
                                Some(json!({
                                    "type": "thinking",
                                    "thinking": thinking_text,
                                    "signature": signature,
                                }))
                            } else {
                                Some(json!({
                                    "type": "text",
                                    "text": thinking_text,
                                }))
                            }
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let name = anthropic_outbound_tool_name(
                                &tool_call.name,
                                use_claude_code_tool_names,
                            );
                            Some(json!({
                                "type": "tool_use",
                                "id": tool_call.id,
                                "name": name,
                                "input": tool_call.arguments,
                            }))
                        }
                    })
                    .collect::<Vec<_>>();
                if !content.is_empty() {
                    messages.push(json!({ "role": "assistant", "content": content }));
                }
            }
            Message::ToolResult(_) => {
                let mut tool_results = Vec::new();
                let mut next = index;
                while next < transformed_messages.len() {
                    let Message::ToolResult(tool_result) = &transformed_messages[next] else {
                        break;
                    };
                    tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": tool_result.tool_call_id,
                        "content": convert_anthropic_tool_result_content(&tool_result.content),
                        "is_error": tool_result.is_error,
                    }));
                    next += 1;
                }
                index = next - 1;
                messages.push(json!({ "role": "user", "content": tool_results }));
            }
        }
        index += 1;
    }
    append_anthropic_cache_control_to_last_user(&mut messages, cache_control);
    messages
}

fn normalize_anthropic_tool_call_id(id: &str) -> String {
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

fn convert_anthropic_tool_result_content(content: &[ToolResultContent]) -> Value {
    let has_images = content
        .iter()
        .any(|part| matches!(part, ToolResultContent::Image(_)));
    if !has_images {
        return Value::String(
            content
                .iter()
                .filter_map(|part| match part {
                    ToolResultContent::Text(text) => Some(sanitize_surrogates(&text.text)),
                    ToolResultContent::Image(_) => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        );
    }

    let mut blocks = content
        .iter()
        .map(|part| match part {
            ToolResultContent::Text(text) => json!({
                "type": "text",
                "text": sanitize_surrogates(&text.text),
            }),
            ToolResultContent::Image(image) => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image.mime_type,
                    "data": image.data,
                },
            }),
        })
        .collect::<Vec<_>>();

    let has_text = blocks
        .iter()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("text"));
    if !has_text {
        blocks.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
    }

    Value::Array(blocks)
}

fn append_anthropic_cache_control_to_last_user(
    messages: &mut [Value],
    cache_control: Option<&Value>,
) {
    let Some(cache_control) = cache_control else {
        return;
    };
    let Some(last_message) = messages
        .iter_mut()
        .rev()
        .find(|message| message.get("role").and_then(Value::as_str) == Some("user"))
    else {
        return;
    };

    match last_message.get_mut("content") {
        Some(Value::String(text)) => {
            let text = std::mem::take(text);
            last_message["content"] = json!([{
                "type": "text",
                "text": text,
                "cache_control": cache_control,
            }]);
        }
        Some(Value::Array(blocks)) => {
            if let Some(last_block) = blocks.last_mut()
                && last_block
                    .get("type")
                    .and_then(Value::as_str)
                    .map(|kind| matches!(kind, "text" | "image" | "tool_result"))
                    .unwrap_or(false)
            {
                last_block["cache_control"] = cache_control.clone();
            }
        }
        _ => {}
    }
}

fn format_anthropic_tool(
    tool: &Tool,
    supports_eager_input_streaming: bool,
    cache_control: Option<&Value>,
    use_claude_code_tool_names: bool,
) -> Value {
    let properties = tool
        .parameters
        .get("properties")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let required = tool
        .parameters
        .get("required")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let name = anthropic_outbound_tool_name(&tool.name, use_claude_code_tool_names);
    let mut formatted = json!({
        "name": name,
        "description": tool.description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    });
    if supports_eager_input_streaming {
        formatted["eager_input_streaming"] = Value::Bool(true);
    }
    if let Some(cache_control) = cache_control {
        formatted["cache_control"] = cache_control.clone();
    }
    formatted
}

fn anthropic_outbound_tool_name(name: &str, use_claude_code_tool_names: bool) -> String {
    if use_claude_code_tool_names {
        to_claude_code_tool_name(name)
    } else {
        name.to_owned()
    }
}

fn anthropic_inbound_tool_name(
    name: &str,
    tools: &[Tool],
    use_claude_code_tool_names: bool,
) -> String {
    if use_claude_code_tool_names {
        from_claude_code_tool_name(name, tools)
    } else {
        name.to_owned()
    }
}

pub fn resolve_anthropic_cache_retention(
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

fn anthropic_cache_control(model: &Model, cache_retention: CacheRetention) -> Option<Value> {
    if cache_retention == CacheRetention::None {
        return None;
    }
    let mut cache_control = json!({ "type": "ephemeral" });
    if cache_retention == CacheRetention::Long && supports_anthropic_long_cache_retention(model) {
        cache_control["ttl"] = Value::String("1h".to_owned());
    }
    Some(cache_control)
}

fn supports_anthropic_long_cache_retention(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsLongCacheRetention"))
        .and_then(Value::as_bool)
        .unwrap_or(model.provider != "fireworks")
}

fn supports_anthropic_eager_tool_input_streaming(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsEagerToolInputStreaming"))
        .and_then(Value::as_bool)
        .unwrap_or(model.provider != "fireworks")
}

fn supports_anthropic_cache_control_on_tools(model: &Model) -> bool {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get("supportsCacheControlOnTools"))
        .and_then(Value::as_bool)
        .unwrap_or(model.provider != "fireworks")
}

fn should_use_anthropic_fine_grained_tool_streaming_beta(model: &Model, context: &Context) -> bool {
    !context.tools.is_empty() && !supports_anthropic_eager_tool_input_streaming(model)
}

fn supports_anthropic_adaptive_thinking(model_id: &str) -> bool {
    model_id.contains("opus-4-6")
        || model_id.contains("opus-4.6")
        || model_id.contains("opus-4-7")
        || model_id.contains("opus-4.7")
        || model_id.contains("sonnet-4-6")
        || model_id.contains("sonnet-4.6")
}

fn map_anthropic_thinking_level_to_effort(model: &Model, level: ThinkingLevel) -> &'static str {
    if let Some(Some(mapped)) = model.thinking_level_map.get(&level) {
        return match mapped.as_str() {
            "low" => "low",
            "medium" => "medium",
            "high" => "high",
            "xhigh" => "xhigh",
            "max" => "max",
            _ => "high",
        };
    }
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => "high",
    }
}

fn anthropic_thinking_budget(
    level: ThinkingLevel,
    budgets: Option<&crate::ThinkingBudgets>,
) -> u64 {
    match level {
        ThinkingLevel::Minimal => budgets.and_then(|budget| budget.minimal).unwrap_or(1_024),
        ThinkingLevel::Low => budgets.and_then(|budget| budget.low).unwrap_or(2_048),
        ThinkingLevel::Medium => budgets.and_then(|budget| budget.medium).unwrap_or(8_192),
        ThinkingLevel::High => budgets.and_then(|budget| budget.high).unwrap_or(16_384),
        ThinkingLevel::XHigh => budgets.and_then(|budget| budget.high).unwrap_or(16_384),
        ThinkingLevel::Off => 1_024,
    }
}
