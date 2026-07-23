//! Parity-audit wave tests for the Anthropic provider surface, porting the pi
//! GAP rows from docs/test-parity-audit-2026-07-23.tsv for
//! anthropic-empty-thinking-signature-compat.test.ts,
//! anthropic-force-adaptive-thinking.test.ts, and the Anthropic
//! PI_CACHE_RETENTION env rows of cache-retention.test.ts.

use ri_llm_provider::*;
use serde_json::{Value, json};
use std::{collections::BTreeMap, sync::Mutex};

static ENV_LOCK: Mutex<()> = Mutex::new(());

struct EnvGuard {
    values: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn clearing(keys: &[&'static str]) -> Self {
        let values = keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in keys {
            remove_env(key);
        }
        Self { values }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            match value {
                Some(value) => set_env(key, value),
                None => remove_env(key),
            }
        }
    }
}

fn set_env(key: &str, value: &str) {
    // These tests hold ENV_LOCK while mutating the process environment.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    // These tests hold ENV_LOCK while mutating the process environment.
    unsafe {
        std::env::remove_var(key);
    }
}

fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        ..Default::default()
    }
}

fn empty_assistant_for_model(model: &Model) -> AssistantMessage {
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
        timestamp: 0,
    }
}

/// Mirrors the pi test fixture `makeModel` from
/// anthropic-empty-thinking-signature-compat.test.ts: an Anthropic-compatible
/// third-party model whose compat is controlled per test.
fn mimo_model(compat: Option<Value>) -> Model {
    Model {
        id: "mimo-v2.5-pro".to_owned(),
        name: "MiMo-V2.5-Pro".to_owned(),
        api: "anthropic-messages".to_owned(),
        provider: "xiaomi-token-plan-ams".to_owned(),
        base_url: "http://127.0.0.1:9/anthropic".to_owned(),
        reasoning: true,
        thinking_level_map: BTreeMap::new(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 1_048_576,
        max_tokens: 1_024,
        headers: BTreeMap::new(),
        compat,
    }
}

/// Mirrors the pi test fixture `makeContext`: user turn, assistant thinking
/// replay, user turn.
fn thinking_replay_context(model: &Model, thinking: &str, thinking_signature: &str) -> Context {
    let mut assistant = empty_assistant_for_model(model);
    assistant.content = vec![AssistantContent::Thinking(ThinkingContent {
        thinking: thinking.to_owned(),
        thinking_signature: Some(thinking_signature.to_owned()),
        redacted: false,
    })];
    Context {
        system_prompt: None,
        messages: vec![
            Message::User(UserMessage::text("first")),
            Message::Assistant(assistant),
            Message::User(UserMessage::text("second")),
        ],
        tools: Vec::new(),
    }
}

fn assistant_content(payload: &Value) -> &Value {
    let message = payload["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|message| message["role"] == json!("assistant"))
        .expect("assistant message");
    &message["content"]
}

// pi: anthropic-empty-thinking-signature-compat.test.ts
// "converts empty-signature thinking to text by default"
#[test]
fn anthropic_converts_empty_signature_thinking_to_text_by_default() {
    let model = mimo_model(None);
    let context = thinking_replay_context(&model, "internal reasoning", "");

    let payload = build_anthropic_simple_payload(&model, &context, SimpleStreamOptions::default());

    assert_eq!(
        assistant_content(&payload),
        &json!([{ "type": "text", "text": "internal reasoning" }])
    );
}

// pi: anthropic-empty-thinking-signature-compat.test.ts
// "preserves empty-signature thinking when allowEmptySignature is enabled"
#[test]
fn anthropic_preserves_empty_signature_thinking_when_allow_empty_signature_enabled() {
    let model = mimo_model(Some(json!({ "allowEmptySignature": true })));
    // A whitespace-only signature counts as empty, matching pi.
    let context = thinking_replay_context(&model, "internal reasoning", " ");

    let payload = build_anthropic_simple_payload(&model, &context, SimpleStreamOptions::default());

    assert_eq!(
        assistant_content(&payload),
        &json!([{
            "type": "thinking",
            "thinking": "internal reasoning",
            "signature": "",
        }])
    );
}

// pi: anthropic-empty-thinking-signature-compat.test.ts
// "allows empty signatures for Kimi Coding k3"
#[test]
fn anthropic_allows_empty_signatures_for_kimi_coding_k3() {
    let model = get_model("kimi-coding", "k3").expect("kimi-coding k3");
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("allowEmptySignature"))
            .and_then(Value::as_bool),
        Some(true),
        "catalog must mark kimi-coding k3 with allowEmptySignature"
    );

    let context = thinking_replay_context(&model, "internal reasoning", " ");
    let payload = build_anthropic_simple_payload(&model, &context, SimpleStreamOptions::default());

    assert_eq!(
        assistant_content(&payload),
        &json!([{
            "type": "thinking",
            "thinking": "internal reasoning",
            "signature": "",
        }])
    );
}

fn kimi_adaptive_payload(model_id: &str, reasoning: ThinkingLevel) -> Value {
    let model = get_model("kimi-coding", model_id).expect(model_id);
    assert_eq!(
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("forceAdaptiveThinking"))
            .and_then(Value::as_bool),
        Some(true),
        "catalog must mark kimi-coding {model_id} with forceAdaptiveThinking"
    );
    build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(reasoning),
            ..Default::default()
        },
    )
}

// pi: anthropic-force-adaptive-thinking.test.ts
// "uses adaptive thinking effort without a token budget for Kimi Coding
// kimi-for-coding"
#[test]
fn anthropic_simple_payload_uses_adaptive_effort_without_budget_for_kimi_for_coding() {
    let payload = kimi_adaptive_payload("kimi-for-coding", ThinkingLevel::Medium);

    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert!(payload["thinking"].get("budget_tokens").is_none());
    assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
}

// pi: anthropic-force-adaptive-thinking.test.ts
// "uses adaptive thinking effort without a token budget for Kimi Coding k3"
#[test]
fn anthropic_simple_payload_uses_adaptive_effort_without_budget_for_kimi_k3() {
    let payload = kimi_adaptive_payload("k3", ThinkingLevel::Max);

    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert!(payload["thinking"].get("budget_tokens").is_none());
    assert_eq!(payload["output_config"], json!({ "effort": "max" }));
}

// pi: anthropic-force-adaptive-thinking.test.ts
// "uses adaptive thinking effort without a token budget for Kimi Coding
// kimi-for-coding-highspeed"
#[test]
fn anthropic_simple_payload_uses_adaptive_effort_without_budget_for_kimi_highspeed() {
    let payload = kimi_adaptive_payload("kimi-for-coding-highspeed", ThinkingLevel::Medium);

    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert!(payload["thinking"].get("budget_tokens").is_none());
    assert_eq!(payload["output_config"], json!({ "effort": "medium" }));
}

// pi: anthropic-force-adaptive-thinking.test.ts
// "allows built-in adaptive models to opt out with compat.forceAdaptiveThinking
// false"
#[test]
fn anthropic_simple_payload_lets_adaptive_models_opt_out_with_compat_false() {
    let mut model = get_model("anthropic", "claude-opus-4-8").expect("claude-opus-4-8");
    let options = SimpleStreamOptions {
        reasoning: Some(ThinkingLevel::Medium),
        ..Default::default()
    };

    // Sanity: without the override the built-in id is adaptive.
    let adaptive_payload =
        build_anthropic_simple_payload(&model, &user_context("Hello"), options.clone());
    assert_eq!(adaptive_payload["thinking"]["type"], json!("adaptive"));

    model.compat = Some(json!({ "forceAdaptiveThinking": false }));
    let payload = build_anthropic_simple_payload(&model, &user_context("Hello"), options);

    assert_eq!(payload["thinking"]["type"], json!("enabled"));
    assert!(payload["thinking"].get("budget_tokens").is_some());
    assert!(payload.get("output_config").is_none());
}

// pi: cache-retention.test.ts
// "should use 1h cache TTL when PI_CACHE_RETENTION=long" (Anthropic provider)
#[test]
fn anthropic_payload_uses_one_hour_cache_ttl_when_env_retention_is_long() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["PI_CACHE_RETENTION"]);
    set_env("PI_CACHE_RETENTION", "long");

    let model = get_model("anthropic", "claude-haiku-4-5").expect("claude-haiku-4-5");
    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };

    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());

    assert_eq!(
        payload["system"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    assert_eq!(
        payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}
