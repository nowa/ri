//! Catalog assertion tests ported from pi's test suite during the 2026-07
//! test-parity audit wave. Sources:
//! `anthropic-adaptive-thinking-models.test.ts`, `supports-xhigh.test.ts`,
//! `xai-responses.test.ts`, `providers.test.ts`, `fireworks-models.test.ts`,
//! `xiaomi-models.test.ts`, `qwen-token-plan-models.test.ts`,
//! `env-api-keys.test.ts`, and `cache-retention.test.ts` (pi v0.81.1 HEAD).

use ri_llm_provider::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn supported_levels(provider: &str, model_id: &str) -> Vec<ThinkingLevel> {
    let model =
        get_model(provider, model_id).unwrap_or_else(|| panic!("missing {provider}/{model_id}"));
    get_supported_thinking_levels(&model)
}

fn compat_bool(model: &Model, key: &str) -> Option<bool> {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get(key))
        .and_then(Value::as_bool)
}

// --- anthropic-adaptive-thinking-models.test.ts -----------------------------

/// pi: "marks built-in Anthropic Messages models that use adaptive thinking".
#[test]
fn adaptive_thinking_flags_sweep_all_catalog_providers() {
    let mut flagged = get_providers()
        .into_iter()
        .flat_map(|provider| get_models(&provider))
        .filter(|model| model.api == "anthropic-messages")
        .filter(|model| compat_bool(model, "forceAdaptiveThinking") == Some(true))
        .map(|model| format!("{}/{}", model.provider, model.id))
        .collect::<Vec<_>>();
    flagged.sort();

    for expected in [
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-4-8",
        "anthropic/claude-sonnet-5",
        "cloudflare-ai-gateway/claude-fable-5",
        "kimi-coding/kimi-for-coding",
        "kimi-coding/k3",
        "kimi-coding/kimi-for-coding-highspeed",
        "opencode/claude-opus-4-8",
        "vercel-ai-gateway/anthropic/claude-opus-4.8",
        "vercel-ai-gateway/anthropic/claude-sonnet-5",
    ] {
        assert!(
            flagged.contains(&expected.to_owned()),
            "adaptive-thinking sweep should flag {expected}: {flagged:?}"
        );
    }

    // pi restricts the flag to current adaptive Claude generations
    // (opus 4.6-4.8, sonnet 4.6/5, fable 5) and Kimi Coding models:
    // /(opus[-.]4[-.][678]|sonnet[-.]4[-.]6|sonnet[-.]5|fable[-.]5|kimi-coding\/)/
    let matches_expected_generation = |entry: &str| {
        [
            "opus-4-6",
            "opus-4.6",
            "opus-4-7",
            "opus-4.7",
            "opus-4-8",
            "opus-4.8",
            "sonnet-4-6",
            "sonnet-4.6",
            "sonnet-5",
            "sonnet.5",
            "fable-5",
            "fable.5",
        ]
        .iter()
        .any(|needle| entry.contains(needle))
            || entry.starts_with("kimi-coding/")
    };
    for entry in &flagged {
        assert!(
            matches_expected_generation(entry),
            "unexpected adaptive-thinking flag on {entry}"
        );
    }
}

// --- supports-xhigh.test.ts --------------------------------------------------

/// pi: "includes xhigh and max for OpenAI gpt-5.6-* models".
#[test]
fn openai_gpt_5_6_models_expose_full_thinking_ladder() {
    for model_id in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
        assert_eq!(
            supported_levels("openai", model_id),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh,
                ThinkingLevel::Max,
            ],
            "{model_id}"
        );
    }
}

/// pi: "includes only medium/high/xhigh for OpenAI GPT-5.5 Pro" and
/// "includes only medium/high/xhigh for OpenRouter GPT-5.5 Pro".
#[test]
fn gpt_5_5_pro_restricts_to_medium_high_xhigh() {
    for (provider, model_id) in [
        ("openai", "gpt-5.5-pro"),
        ("openrouter", "openai/gpt-5.5-pro"),
    ] {
        assert_eq!(
            supported_levels(provider, model_id),
            vec![
                ThinkingLevel::Medium,
                ThinkingLevel::High,
                ThinkingLevel::XHigh
            ],
            "{provider}/{model_id}"
        );
    }
}

/// pi: "includes only high/max plus off for DeepSeek V4 Flash" on the
/// DeepSeek provider and on opencode-go.
#[test]
fn deepseek_v4_flash_exposes_off_high_max() {
    for provider in ["deepseek", "opencode-go"] {
        assert_eq!(
            supported_levels(provider, "deepseek-v4-flash"),
            vec![ThinkingLevel::Off, ThinkingLevel::High, ThinkingLevel::Max],
            "{provider}"
        );
    }
}

/// pi: "includes only high plus off for OpenCode Go Kimi K2.6".
#[test]
fn opencode_go_kimi_k2_6_exposes_off_high_only() {
    assert_eq!(
        supported_levels("opencode-go", "kimi-k2.6"),
        vec![ThinkingLevel::Off, ThinkingLevel::High]
    );
}

/// pi: "excludes thinking off for Moonshot Kimi K2.7 Code models".
#[test]
fn moonshot_kimi_k2_7_code_excludes_thinking_off() {
    for provider in ["moonshotai", "moonshotai-cn"] {
        assert_eq!(
            supported_levels(provider, "kimi-k2.7-code"),
            vec![
                ThinkingLevel::Minimal,
                ThinkingLevel::Low,
                ThinkingLevel::Medium,
                ThinkingLevel::High,
            ],
            "{provider}"
        );
    }
}

/// pi: "includes only low, high, max for Kimi Coding K3".
#[test]
fn kimi_coding_k3_exposes_low_high_max() {
    assert_eq!(
        supported_levels("kimi-coding", "k3"),
        vec![ThinkingLevel::Low, ThinkingLevel::High, ThinkingLevel::Max]
    );
}

/// pi: "includes only high for OpenCode Grok Build".
#[test]
fn opencode_grok_build_exposes_high_only() {
    assert_eq!(
        supported_levels("opencode", "grok-build-0.1"),
        vec![ThinkingLevel::High]
    );
}

/// pi: "includes max but not xhigh for OpenRouter Opus 4.6
/// (openai-completions API)".
#[test]
fn openrouter_opus_4_6_exposes_max_but_not_xhigh() {
    let levels = supported_levels("openrouter", "anthropic/claude-opus-4.6");
    assert!(levels.contains(&ThinkingLevel::Max), "{levels:?}");
    assert!(!levels.contains(&ThinkingLevel::XHigh), "{levels:?}");
}

/// pi: "includes xhigh and max but not off for Bedrock Claude Fable 5".
#[test]
fn bedrock_claude_fable_5_supports_xhigh_and_max_but_not_off() {
    let levels = supported_levels("amazon-bedrock", "global.anthropic.claude-fable-5");
    assert!(levels.contains(&ThinkingLevel::XHigh), "{levels:?}");
    assert!(levels.contains(&ThinkingLevel::Max), "{levels:?}");
    assert!(!levels.contains(&ThinkingLevel::Off), "{levels:?}");
}

// --- xai-responses.test.ts ---------------------------------------------------

/// pi: "excludes retired and redundant models from the built-in catalog".
#[test]
fn xai_catalog_excludes_retired_models() {
    let ids = get_models("xai")
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    for retired in [
        "grok-3",
        "grok-3-fast",
        "grok-4.20-0309-non-reasoning",
        "grok-4.20-0309-reasoning",
        "grok-code-fast-1",
    ] {
        assert!(
            !ids.contains(&retired.to_owned()),
            "xai catalog should retire {retired}: {ids:?}"
        );
    }
}

/// pi: "uses Responses with low/medium/high efforts only for Grok 4.5".
#[test]
fn xai_grok_4_5_uses_responses_with_low_medium_high_efforts() {
    let grok_4_5 = get_model("xai", "grok-4.5").expect("grok-4.5");
    assert_eq!(grok_4_5.api, "openai-responses");
    assert_eq!(
        get_supported_thinking_levels(&grok_4_5),
        vec![
            ThinkingLevel::Low,
            ThinkingLevel::Medium,
            ThinkingLevel::High
        ]
    );

    let grok_4_3 = get_model("xai", "grok-4.3").expect("grok-4.3");
    assert_eq!(grok_4_3.api, "openai-completions");
}

/// pi: "uses /responses with bearer auth and xAI-compatible request fields".
/// ri builds the request from the catalog model, so this asserts the payload
/// and default headers the Responses dispatch would send to
/// https://api.x.ai/v1/responses.
#[test]
fn xai_grok_4_5_responses_request_fields_match_pi() {
    let model = get_model("xai", "grok-4.5").expect("grok-4.5");
    assert_eq!(model.base_url, "https://api.x.ai/v1");
    assert_eq!(
        compat_bool(&model, "supportsLongCacheRetention"),
        Some(false)
    );

    let context = Context {
        system_prompt: Some("You are a careful coding assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text("hello"))],
        ..Default::default()
    };
    let payload = build_openai_responses_payload(
        &model,
        &context,
        OpenAIResponsesPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("pi-session-123".to_owned()),
            reasoning_effort: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );

    assert_eq!(payload["model"], "grok-4.5");
    assert_eq!(payload["store"], false);
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["prompt_cache_key"], "pi-session-123");
    assert_eq!(payload["reasoning"]["effort"], "medium");
    assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
    // supportsLongCacheRetention:false drops prompt_cache_retention even for
    // long cache retention requests.
    assert!(payload.get("prompt_cache_retention").is_none());
    let input = payload["input"].as_array().expect("input messages");
    assert!(
        input.iter().any(|message| message["role"] == "developer"
            && message["content"] == "You are a careful coding assistant."),
        "system prompt should ride the developer role: {input:?}"
    );

    let headers = build_openai_responses_default_headers(
        &model,
        Some("pi-session-123"),
        CacheRetention::Long,
        &BTreeMap::new(),
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("pi-session-123")
    );
}

// --- providers.test.ts -------------------------------------------------------

/// pi: "uses official Kimi K3 pricing for Moonshot providers".
#[test]
fn moonshot_kimi_k3_uses_official_pricing() {
    for provider in ["moonshotai", "moonshotai-cn"] {
        let model = get_model(provider, "kimi-k3").expect(provider);
        assert_eq!(
            model.cost,
            ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 0.0,
                tiers: Vec::new(),
            },
            "{provider}"
        );
    }
}

/// pi: "uses API-equivalent implied pricing for Kimi Coding subscription
/// models".
#[test]
fn kimi_coding_subscription_models_use_implied_pricing() {
    for (model_id, input, output, cache_read) in [
        ("k3", 3.0, 15.0, 0.3),
        ("kimi-for-coding", 0.95, 4.0, 0.19),
        ("kimi-for-coding-highspeed", 1.9, 8.0, 0.38),
    ] {
        let model = get_model("kimi-coding", model_id).expect(model_id);
        assert_eq!(
            model.cost,
            ModelCost {
                input,
                output,
                cache_read,
                cache_write: 0.0,
                tiers: Vec::new(),
            },
            "{model_id}"
        );
    }
}

// --- fireworks-models.test.ts ------------------------------------------------

/// pi: "aligns GLM 5.2 Fast with GLM 5.2's OpenAI-compatible config".
#[test]
fn fireworks_glm_5_2_fast_aligns_with_base_model() {
    let base = get_model("fireworks", "accounts/fireworks/models/glm-5p2").expect("glm-5p2");
    let fast =
        get_model("fireworks", "accounts/fireworks/routers/glm-5p2-fast").expect("glm-5p2-fast");

    assert_eq!(base.api, "openai-completions");
    assert_eq!(fast.api, base.api);
    assert_eq!(fast.base_url, base.base_url);
    assert_eq!(fast.compat, base.compat);
    assert_eq!(fast.thinking_level_map, base.thinking_level_map);
}

// --- xiaomi-models.test.ts ---------------------------------------------------

/// pi: "keeps mimo-v2-flash|mimo-v2-omni on the API billing provider" and
/// "omits API-billing-only models from xiaomi-token-plan-cn|ams|sgp".
#[test]
fn xiaomi_api_billing_models_stay_off_token_plans() {
    let api_billing_ids = get_models("xiaomi")
        .into_iter()
        .map(|model| model.id)
        .collect::<Vec<_>>();
    for model_id in ["mimo-v2-flash", "mimo-v2-omni"] {
        assert!(
            api_billing_ids.contains(&model_id.to_owned()),
            "xiaomi should keep {model_id}: {api_billing_ids:?}"
        );
    }

    for provider in [
        "xiaomi-token-plan-cn",
        "xiaomi-token-plan-ams",
        "xiaomi-token-plan-sgp",
    ] {
        let token_plan_ids = get_models(provider)
            .into_iter()
            .map(|model| model.id)
            .collect::<Vec<_>>();
        assert!(!token_plan_ids.is_empty(), "{provider} catalog");
        for model_id in ["mimo-v2-flash", "mimo-v2-omni"] {
            assert!(
                !token_plan_ids.contains(&model_id.to_owned()),
                "{provider} should omit {model_id}: {token_plan_ids:?}"
            );
        }
    }
}

// --- qwen-token-plan-models.test.ts -------------------------------------------

const QWEN_TOKEN_PLAN_TEXT_MODELS: [&str; 15] = [
    "MiniMax-M2.5",
    "deepseek-v3.2",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "glm-5",
    "glm-5.1",
    "glm-5.2",
    "kimi-k2.5",
    "kimi-k2.6",
    "kimi-k2.7-code",
    "qwen3.6-flash",
    "qwen3.6-plus",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.8-max-preview",
];

const QWEN_TOKEN_PLAN_IMAGE_MODELS: [&str; 4] = [
    "qwen-image-2.0",
    "qwen-image-2.0-pro",
    "wan2.7-image",
    "wan2.7-image-pro",
];

/// pi: "exposes all text models on qwen-token-plan|qwen-token-plan-cn" and
/// "omits image models from qwen-token-plan|qwen-token-plan-cn".
#[test]
fn qwen_token_plan_exposes_text_models_and_omits_image_models() {
    for provider in ["qwen-token-plan", "qwen-token-plan-cn"] {
        let ids = get_models(provider)
            .into_iter()
            .map(|model| model.id)
            .collect::<Vec<_>>();
        for expected in QWEN_TOKEN_PLAN_TEXT_MODELS {
            assert!(
                ids.contains(&expected.to_owned()),
                "{provider} should include {expected}: {ids:?}"
            );
        }
        for excluded in QWEN_TOKEN_PLAN_IMAGE_MODELS {
            assert!(
                !ids.contains(&excluded.to_owned()),
                "{provider} should not include {excluded}: {ids:?}"
            );
        }
    }
}

// --- env-api-keys.test.ts ------------------------------------------------------

/// pi: "resolves ZAI China Coding Plan credentials from
/// ZAI_CODING_CN_API_KEY". This is the only test in this binary that mutates
/// the process environment.
#[test]
fn zai_coding_cn_env_key_resolves_credentials() {
    const KEY: &str = "ZAI_CODING_CN_API_KEY";
    let previous = std::env::var(KEY).ok();
    // Safety: no other test in this binary reads or writes this variable.
    unsafe {
        std::env::set_var(KEY, "zai-coding-cn-token");
    }

    let keys = find_env_keys("zai-coding-cn");
    let api_key = get_env_api_key("zai-coding-cn");

    // Safety: see above.
    unsafe {
        match previous {
            Some(value) => std::env::set_var(KEY, value),
            None => std::env::remove_var(KEY),
        }
    }

    assert_eq!(keys, Some(vec![KEY.to_owned()]));
    assert_eq!(api_key, Some("zai-coding-cn-token".to_owned()));
}

// --- cache-retention.test.ts ----------------------------------------------------

/// pi: "should omit long cache retention for opencode/... and
/// opencode-go/kimi-k2.6". Asserts the per-model
/// supportsLongCacheRetention:false catalog flag and that long-retention
/// requests drop both prompt cache fields.
#[test]
fn opencode_models_disable_long_cache_retention() {
    for (provider, model_id) in [
        ("opencode", "deepseek-v4-flash"),
        ("opencode", "deepseek-v4-pro"),
        ("opencode", "kimi-k2.5"),
        ("opencode", "kimi-k2.6"),
        ("opencode", "minimax-m2.7"),
        ("opencode-go", "kimi-k2.6"),
    ] {
        let model = get_model(provider, model_id)
            .unwrap_or_else(|| panic!("missing {provider}/{model_id}"));
        assert_eq!(model.api, "openai-completions", "{provider}/{model_id}");
        assert_eq!(
            compat_bool(&model, "supportsLongCacheRetention"),
            Some(false),
            "{provider}/{model_id} supportsLongCacheRetention"
        );

        let payload = build_openai_completions_payload(
            &model,
            &Context {
                messages: vec![Message::User(UserMessage::text("hi"))],
                ..Default::default()
            },
            OpenAICompletionsPayloadOptions {
                cache_retention: Some(CacheRetention::Long),
                session_id: Some("session-opencode-long-cache-unsupported".to_owned()),
                ..Default::default()
            },
        );
        assert!(
            payload.get("prompt_cache_key").is_none(),
            "{provider}/{model_id} prompt_cache_key"
        );
        assert!(
            payload.get("prompt_cache_retention").is_none(),
            "{provider}/{model_id} prompt_cache_retention"
        );
    }
}
