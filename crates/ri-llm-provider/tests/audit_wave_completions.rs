//! Parity-audit wave: gaps from pi's `openai-completions-tool-choice.test.ts`
//! and `openai-completions-prompt-cache.test.ts` (pi v0.81.1).

use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        ..Default::default()
    }
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().cloned().expect("object")
}

fn compat_value<'a>(model: &'a Model, key: &str) -> Option<&'a Value> {
    model.compat.as_ref().and_then(|compat| compat.get(key))
}

fn build_payload(model: &Model, options: OpenAICompletionsPayloadOptions) -> Value {
    build_openai_completions_payload(model, &user_context("Hi"), options)
}

fn reasoning_options(reasoning: Option<ThinkingLevel>) -> OpenAICompletionsPayloadOptions {
    OpenAICompletionsPayloadOptions {
        reasoning,
        ..Default::default()
    }
}

fn assistant_message(model: &Model, content: Vec<AssistantContent>) -> AssistantMessage {
    AssistantMessage {
        content,
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 2,
    }
}

fn tool_result(
    tool_call_id: &str,
    content: Vec<ToolResultContent>,
    added_tool_names: Option<Vec<String>>,
) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: tool_call_id.to_owned(),
        tool_name: "read".to_owned(),
        content,
        details: None,
        usage: None,
        is_error: false,
        added_tool_names,
        timestamp: 3,
    }
}

fn deferred_tool(name: &str) -> Tool {
    Tool {
        name: name.to_owned(),
        description: format!("The {name} tool"),
        parameters: json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
        }),
    }
}

// --- openai-completions-tool-choice.test.ts ---------------------------------

/// pi: "stores z.ai GLM-5.2 effort metadata".
#[test]
fn zai_glm52_catalog_stores_effort_metadata() {
    for provider in ["zai", "zai-coding-cn"] {
        let model = get_model(provider, "glm-5.2").expect("glm-5.2");
        assert_eq!(
            compat_value(&model, "supportsReasoningEffort"),
            Some(&json!(true)),
            "{provider}"
        );
        assert_eq!(
            model.thinking_level_map,
            BTreeMap::from([
                (ThinkingLevel::Minimal, None),
                (ThinkingLevel::Low, Some("high".to_owned())),
                (ThinkingLevel::Medium, Some("high".to_owned())),
                (ThinkingLevel::High, Some("high".to_owned())),
                (ThinkingLevel::Max, Some("max".to_owned())),
            ]),
            "{provider}"
        );
    }
}

/// pi: "maps z.ai GLM-5.2 thinking levels to reasoning_effort".
#[test]
fn zai_glm52_maps_thinking_levels_to_reasoning_effort() {
    let model = get_model("zai", "glm-5.2").expect("glm-5.2");
    for (reasoning, effort) in [
        (ThinkingLevel::Low, "high"),
        (ThinkingLevel::Medium, "high"),
        (ThinkingLevel::High, "high"),
        (ThinkingLevel::Max, "max"),
    ] {
        let payload = build_payload(&model, reasoning_options(Some(reasoning)));
        assert_eq!(
            payload["thinking"],
            json!({ "type": "enabled", "clear_thinking": false }),
            "{reasoning:?}"
        );
        assert_eq!(payload["reasoning_effort"], effort, "{reasoning:?}");
        assert!(
            payload.get("enable_thinking").is_none(),
            "{reasoning:?} legacy enable_thinking"
        );
    }
}

/// pi: "preserves z.ai thinking when replaying reasoning_content".
#[test]
fn zai_glm52_preserves_thinking_when_replaying_reasoning_content() {
    let model = get_model("zai", "glm-5.2").expect("glm-5.2");
    let assistant = assistant_message(
        &model,
        vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "prior reasoning".to_owned(),
                thinking_signature: Some("reasoning_content".to_owned()),
                redacted: false,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_1".to_owned(),
                name: "read".to_owned(),
                arguments: object(json!({ "path": "README.md" })),
                thought_signature: None,
            }),
        ],
    );
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Read README.md")),
            Message::Assistant(assistant),
            Message::ToolResult(tool_result(
                "call_1",
                vec![ToolResultContent::text("contents")],
                None,
            )),
            Message::User(UserMessage::text("Continue")),
        ],
        ..Default::default()
    };

    let payload = build_openai_completions_payload(
        &model,
        &context,
        reasoning_options(Some(ThinkingLevel::High)),
    );

    let replayed_assistant = payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == "assistant")
        .expect("replayed assistant");
    assert_eq!(replayed_assistant["reasoning_content"], "prior reasoning");
    assert_eq!(
        payload["thinking"],
        json!({ "type": "enabled", "clear_thinking": false })
    );
}

/// pi: "omits z.ai GLM-5.2 reasoning_effort when thinking is off".
#[test]
fn zai_glm52_omits_reasoning_effort_when_thinking_off() {
    let model = get_model("zai", "glm-5.2").expect("glm-5.2");
    let payload = build_payload(&model, reasoning_options(None));

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("reasoning_effort").is_none());
    assert!(payload.get("enable_thinking").is_none());
}

/// pi: "uses Ant Ling compatibility metadata".
#[test]
fn ant_ling_uses_compatibility_metadata() {
    let model = get_model("ant-ling", "Ring-2.6-1T").expect("Ring-2.6-1T");
    assert_eq!(compat_value(&model, "supportsStore"), Some(&json!(false)));
    assert_eq!(
        compat_value(&model, "supportsDeveloperRole"),
        Some(&json!(false))
    );
    assert_eq!(
        compat_value(&model, "supportsReasoningEffort"),
        Some(&json!(false))
    );
    assert_eq!(
        compat_value(&model, "maxTokensField"),
        Some(&json!("max_tokens"))
    );
    assert_eq!(
        compat_value(&model, "thinkingFormat"),
        Some(&json!("ant-ling"))
    );
    assert_eq!(
        compat_value(&model, "supportsLongCacheRetention"),
        Some(&json!(false))
    );
    assert!(compat_value(&model, "supportsStrictMode").is_none());
    assert!(compat_value(&model, "requiresReasoningContentOnAssistantMessages").is_none());

    let payload = build_openai_completions_payload(
        &model,
        &Context {
            system_prompt: Some("Follow instructions.".to_owned()),
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            max_tokens: Some(123),
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("ant-ling-session".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(payload["max_tokens"], 123);
    assert!(payload.get("max_completion_tokens").is_none());
    assert_eq!(payload["messages"][0]["role"], "system");
    assert_eq!(payload["reasoning"], json!({ "effort": "high" }));
    assert!(payload.get("reasoning_effort").is_none());
    assert!(payload.get("store").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

/// pi: "omits Ant Ling reasoning for unmapped direct reasoning efforts" —
/// medium maps to null in Ring-2.6-1T's thinkingLevelMap.
#[test]
fn ant_ling_omits_reasoning_for_unmapped_direct_efforts() {
    let model = get_model("ant-ling", "Ring-2.6-1T").expect("Ring-2.6-1T");
    let payload = build_payload(&model, reasoning_options(Some(ThinkingLevel::Medium)));

    assert!(payload.get("reasoning").is_none());
    assert!(payload.get("reasoning_effort").is_none());
}

/// pi: "omits Ant Ling reasoning for ... non-reasoning models".
#[test]
fn ant_ling_omits_reasoning_for_non_reasoning_models() {
    let model = get_model("ant-ling", "Ling-2.6-flash").expect("Ling-2.6-flash");
    assert!(!model.reasoning);
    let payload = build_payload(&model, reasoning_options(Some(ThinkingLevel::High)));

    assert!(payload.get("reasoning").is_none());
    assert!(payload.get("reasoning_effort").is_none());
}

fn local_chat_template_model(id: &str, compat: Value) -> Model {
    let mut model = Model::faux("openai-completions", "local-vllm", id);
    model.base_url = "http://localhost:8000/v1".to_owned();
    model.reasoning = true;
    model.input = vec![InputKind::Text];
    model.compat = Some(compat);
    model
}

/// pi: "uses configurable chat template boolean thinking kwargs".
#[test]
fn chat_template_boolean_thinking_kwargs() {
    let model = local_chat_template_model(
        "deepseek-ai/DeepSeek-V3.1",
        json!({
            "thinkingFormat": "chat-template",
            "supportsReasoningEffort": false,
            "chatTemplateKwargs": { "thinking": { "$var": "thinking.enabled" } },
        }),
    );

    for (reasoning, expected) in [(Some(ThinkingLevel::High), true), (None, false)] {
        let payload = build_payload(&model, reasoning_options(reasoning));
        assert_eq!(
            payload["chat_template_kwargs"],
            json!({ "thinking": expected }),
            "{reasoning:?}"
        );
        assert!(payload.get("thinking").is_none(), "{reasoning:?}");
        assert!(payload.get("reasoning_effort").is_none(), "{reasoning:?}");
    }
}

/// pi: "uses configurable chat template effort kwargs with static kwargs".
#[test]
fn chat_template_effort_kwargs_with_static_kwargs() {
    let mut model = local_chat_template_model(
        "unsloth/gpt-oss-120b-GGUF",
        json!({
            "thinkingFormat": "chat-template",
            "supportsReasoningEffort": false,
            "chatTemplateKwargs": {
                "preserve_thinking": true,
                "reasoning_effort": { "$var": "thinking.effort", "omitWhenOff": true },
            },
        }),
    );
    model
        .thinking_level_map
        .insert(ThinkingLevel::XHigh, Some("max".to_owned()));

    let payload = build_payload(&model, reasoning_options(Some(ThinkingLevel::XHigh)));
    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "preserve_thinking": true, "reasoning_effort": "max" })
    );
    assert!(payload.get("reasoning_effort").is_none());

    // omitWhenOff drops the effort kwarg while static kwargs survive.
    let payload = build_payload(&model, reasoning_options(None));
    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "preserve_thinking": true })
    );
}

/// pi: "sends thinking disabled for OpenCode Go Kimi K2.6 when thinking is off".
#[test]
fn opencode_go_kimi_k26_sends_disabled_thinking_when_off() {
    let model = get_model("opencode-go", "kimi-k2.6").expect("opencode-go kimi");
    let payload = build_payload(&model, reasoning_options(None));

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

/// pi: "sends thinking enabled for OpenCode Go Kimi K2.6 when thinking is
/// enabled" — supportsReasoningEffort:false suppresses reasoning_effort.
#[test]
fn opencode_go_kimi_k26_sends_enabled_thinking_without_reasoning_effort() {
    let model = get_model("opencode-go", "kimi-k2.6").expect("opencode-go kimi");
    let payload = build_payload(&model, reasoning_options(Some(ThinkingLevel::High)));

    assert_eq!(payload["thinking"], json!({ "type": "enabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

/// pi: "omits disabled thinking for Moonshot Kimi K2.7 Code models" —
/// thinkingLevelMap.off === null suppresses the disabled marker entirely.
#[test]
fn moonshot_kimi_k27_code_omits_disabled_thinking_when_off() {
    for provider in ["moonshotai", "moonshotai-cn"] {
        let model = get_model(provider, "kimi-k2.7-code").expect("kimi-k2.7-code");
        let payload = build_payload(&model, reasoning_options(None));

        assert!(payload.get("thinking").is_none(), "{provider}");
        assert!(payload.get("reasoning_effort").is_none(), "{provider}");
    }
}

/// pi: "keeps disabled thinking for Moonshot Kimi K2.6 when thinking is off".
#[test]
fn moonshot_kimi_k26_keeps_disabled_thinking_when_off() {
    let model = get_model("moonshotai-cn", "kimi-k2.6").expect("moonshot kimi");
    let payload = build_payload(&model, reasoning_options(None));

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("reasoning_effort").is_none());
}

/// pi: "sends max_tokens for OpenCode completions models".
#[test]
fn opencode_completions_models_send_max_tokens() {
    for (provider, model_id) in [("opencode-go", "kimi-k2.6"), ("opencode", "grok-build-0.1")] {
        let model = get_model(provider, model_id).expect("opencode model");
        assert_eq!(
            compat_value(&model, "maxTokensField"),
            Some(&json!("max_tokens")),
            "{provider}/{model_id}"
        );

        let payload = build_payload(
            &model,
            OpenAICompletionsPayloadOptions {
                max_tokens: Some(123),
                ..Default::default()
            },
        );
        assert_eq!(payload["max_tokens"], 123, "{provider}/{model_id}");
        assert!(
            payload.get("max_completion_tokens").is_none(),
            "{provider}/{model_id}"
        );
    }
}

/// pi: "omits reasoning effort for OpenCode Grok Build".
#[test]
fn opencode_grok_build_omits_reasoning_effort() {
    let model = get_model("opencode", "grok-build-0.1").expect("grok build");
    let payload = build_payload(&model, reasoning_options(Some(ThinkingLevel::High)));

    assert!(payload.get("reasoning_effort").is_none());
}

// --- convert-messages divergences found last wave ----------------------------

/// pi resets the deferred-name accumulator per tool-result batch
/// (openai-completions.ts ~1072); a later batch must not re-ship earlier
/// batches' tool schemas.
#[test]
fn kimi_deferred_tool_names_reset_between_tool_result_batches() {
    let mut model = Model::faux("openai-completions", "moonshotai", "deferred-tools-model");
    model.compat = Some(json!({ "deferredToolsMode": "kimi" }));
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Hello")),
            Message::Assistant(assistant_message(
                &model,
                vec![AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_owned(),
                    name: "base_tool".to_owned(),
                    arguments: Map::new(),
                    thought_signature: None,
                })],
            )),
            Message::ToolResult(tool_result(
                "call_1",
                vec![ToolResultContent::text("done")],
                Some(vec!["late_tool".to_owned()]),
            )),
            Message::Assistant(assistant_message(
                &model,
                vec![AssistantContent::ToolCall(ToolCall {
                    id: "call_2".to_owned(),
                    name: "late_tool".to_owned(),
                    arguments: Map::new(),
                    thought_signature: None,
                })],
            )),
            Message::ToolResult(tool_result(
                "call_2",
                vec![ToolResultContent::text("done again")],
                Some(vec!["later_tool".to_owned()]),
            )),
            Message::User(UserMessage::text("Continue")),
        ],
        tools: vec![
            deferred_tool("base_tool"),
            deferred_tool("late_tool"),
            deferred_tool("later_tool"),
        ],
        ..Default::default()
    };

    let messages = convert_openai_completions_messages(&model, &context);

    let system_tool_batches = messages
        .iter()
        .filter(|message| message["role"] == "system" && message.get("tools").is_some())
        .map(|message| {
            message["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .map(|tool| {
                    tool["function"]["name"]
                        .as_str()
                        .expect("tool name")
                        .to_owned()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        system_tool_batches,
        vec![vec!["late_tool".to_owned()], vec!["later_tool".to_owned()]]
    );
}

/// pi emits the Kimi system-tools message AFTER the user message carrying
/// tool-result images.
#[test]
fn kimi_system_tools_message_follows_image_tool_result_user_message() {
    let mut model = Model::faux("openai-completions", "moonshotai", "deferred-tools-model");
    model.compat = Some(json!({ "deferredToolsMode": "kimi" }));
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Hello")),
            Message::Assistant(assistant_message(
                &model,
                vec![AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_owned(),
                    name: "base_tool".to_owned(),
                    arguments: Map::new(),
                    thought_signature: None,
                })],
            )),
            Message::ToolResult(tool_result(
                "call_1",
                vec![
                    ToolResultContent::text("done"),
                    ToolResultContent::Image(ImageContent {
                        data: "aGk=".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                Some(vec!["late_tool".to_owned()]),
            )),
            Message::User(UserMessage::text("Continue")),
        ],
        tools: vec![deferred_tool("base_tool"), deferred_tool("late_tool")],
        ..Default::default()
    };

    let messages = convert_openai_completions_messages(&model, &context);

    let roles = messages
        .iter()
        .map(|message| {
            let role = message["role"].as_str().expect("role");
            if role == "system" && message.get("tools").is_some() {
                "system-tools"
            } else {
                role
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        vec!["user", "assistant", "tool", "user", "system-tools", "user"]
    );
    // The image user message directly precedes the deferred-tools message.
    assert_eq!(
        messages[3]["content"][0]["text"],
        "Attached image(s) from tool result:"
    );
    assert_eq!(
        messages[4]["tools"][0]["function"]["name"],
        json!("late_tool")
    );
}

// --- openai-completions-prompt-cache.test.ts ---------------------------------

fn openai_completions_gpt4o_mini() -> Model {
    let mut model = get_model("openai", "gpt-4o-mini").expect("gpt-4o-mini");
    model.api = "openai-completions".to_owned();
    model
}

/// pi: "clamps prompt_cache_key to OpenAI's 64-character limit".
#[test]
fn completions_prompt_cache_key_clamped_to_64_chars() {
    let model = openai_completions_gpt4o_mini();
    let payload = build_payload(
        &model,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Short),
            session_id: Some("x".repeat(67)),
            ..Default::default()
        },
    );

    assert_eq!(payload["prompt_cache_key"], "x".repeat(64));
}

/// pi: "uses OpenAI no-session format when configured" — the affinity pair is
/// kept but the session_id header is dropped.
#[test]
fn completions_openai_nosession_affinity_format() {
    let mut model = openai_completions_gpt4o_mini();
    model.compat = Some(json!({
        "sendSessionAffinityHeaders": true,
        "sessionAffinityFormat": "openai-nosession",
    }));

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-nosession"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-nosession")
    );
    assert_eq!(
        headers.get("x-session-affinity").map(String::as_str),
        Some("session-nosession")
    );
    assert!(headers.get("x-session-id").is_none());

    let payload = build_payload(
        &model,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Short),
            session_id: Some("session-nosession".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["prompt_cache_key"], "session-nosession");
    assert!(payload.get("session_id").is_none());
}

/// pi: "uses OpenRouter session-affinity header when configured" — the
/// explicit compat override applies on a non-OpenRouter proxy.
#[test]
fn completions_explicit_openrouter_affinity_override() {
    let mut model = openai_completions_gpt4o_mini();
    model.base_url = "https://proxy.example.com/v1".to_owned();
    model.compat = Some(json!({
        "sendSessionAffinityHeaders": true,
        "sessionAffinityFormat": "openrouter",
    }));

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-proxy"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert_eq!(
        headers.get("x-session-id").map(String::as_str),
        Some("session-proxy")
    );
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());
    assert!(headers.get("x-session-affinity").is_none());

    let payload = build_payload(
        &model,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Short),
            session_id: Some("session-proxy".to_owned()),
            ..Default::default()
        },
    );
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("session_id").is_none());
}

/// pi: "omits OpenRouter session-affinity data when disabled" —
/// sendSessionAffinityHeaders defaults to false.
#[test]
fn completions_omits_session_affinity_when_disabled() {
    let mut model = openai_completions_gpt4o_mini();
    model.provider = "openrouter".to_owned();
    model.base_url = "https://openrouter.ai/api/v1".to_owned();
    model.compat = None;

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-openrouter"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert!(headers.get("x-session-id").is_none());
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());
    assert!(headers.get("x-session-affinity").is_none());

    let payload = build_payload(
        &model,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Short),
            session_id: Some("session-openrouter".to_owned()),
            ..Default::default()
        },
    );
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("session_id").is_none());
}
