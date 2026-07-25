//! Regression tests for behavior-parity fixes ported from pi v0.81.1 during
//! the 2026-07 provider behavior audit. Sources:
//! `api/anthropic-messages.ts` (compat defaults, max_tokens fallback,
//! forceAdaptiveThinking), `api/mistral-conversations.ts` (cached prompt
//! tokens, SDK URL building), `api/google-generative-ai.ts` /
//! `api/google-vertex.ts` / `api/google-shared.ts` (flash detection, gemma-4
//! split, generated tool-call ids, usage totals), and `api/pi-messages.ts`
//! (response-failure diagnostics).

use ri_llm_provider::*;
use serde_json::{Value, json};

fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        ..Default::default()
    }
}

fn lookup_tool() -> Tool {
    Tool {
        name: "lookup".to_owned(),
        description: "Look up a value".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "value": { "type": "string" },
            },
            "required": ["value"],
        }),
    }
}

fn tool_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        tools: vec![lookup_tool()],
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

// --- anthropic-messages.ts:952 max_tokens fallback ---------------------------

// pi sends `options?.maxTokens ?? model.maxTokens`; the raw-options path must
// fall back to the full model cap, not a fraction of it.
#[test]
fn anthropic_payload_max_tokens_falls_back_to_model_max_tokens() {
    let model = Model::faux("anthropic-messages", "anthropic", "claude-test");
    assert_eq!(model.max_tokens, 16_384);

    let payload = build_anthropic_payload(
        &model,
        &user_context("Hello"),
        AnthropicPayloadOptions::default(),
    );
    assert_eq!(payload["max_tokens"], json!(16_384));

    let payload = build_anthropic_payload(
        &model,
        &user_context("Hello"),
        AnthropicPayloadOptions {
            max_tokens: Some(777),
            ..Default::default()
        },
    );
    assert_eq!(payload["max_tokens"], json!(777));
}

// --- anthropic-messages.ts:175-179 compat defaults ---------------------------

// pi's compat defaults are constants (eager streaming, long cache retention,
// and tool cache_control all default on) regardless of provider; custom
// fireworks-provider models without catalog compat use those defaults.
#[test]
fn anthropic_compat_defaults_are_constants_for_custom_models() {
    let model = Model::faux("anthropic-messages", "fireworks", "custom-claude");
    let context = tool_context("Use the tool");

    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        },
    );
    // supportsEagerToolInputStreaming defaults to true.
    assert_eq!(payload["tools"][0]["eager_input_streaming"], json!(true));
    // supportsCacheControlOnTools defaults to true and
    // supportsLongCacheRetention defaults to true (1h TTL).
    assert_eq!(
        payload["tools"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    // Eager streaming on means the fine-grained beta is not needed.
    let headers = build_anthropic_default_headers(&model, &context);
    assert!(headers.get("anthropic-beta").is_none());
}

// Catalog fireworks models carry explicit compat flags, so their behavior is
// unchanged by the constant defaults.
#[test]
fn anthropic_catalog_fireworks_compat_flags_still_apply() {
    let model =
        get_model("fireworks", "accounts/fireworks/models/kimi-k2p6").expect("fireworks model");
    let context = tool_context("Use the tool");

    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        },
    );
    assert!(payload["tools"][0].get("eager_input_streaming").is_none());
    assert!(payload["tools"][0].get("cache_control").is_none());
    let headers = build_anthropic_default_headers(&model, &context);
    assert_eq!(
        headers.get("anthropic-beta").map(String::as_str),
        Some("fine-grained-tool-streaming-2025-05-14")
    );
}

// pi's sendSessionAffinityHeaders default is false: custom models without the
// compat flag never send x-session-affinity, whatever the provider.
#[test]
fn anthropic_session_affinity_defaults_off_without_compat_flag() {
    for (provider, base_url) in [
        ("fireworks", "https://api.fireworks.ai/inference"),
        (
            "cloudflare-ai-gateway",
            "https://gateway.ai.cloudflare.com/v1/acct/gw/anthropic",
        ),
    ] {
        let mut model = Model::faux("anthropic-messages", provider, "custom-claude");
        model.base_url = base_url.to_owned();
        let config = build_anthropic_client_config(
            &model,
            &user_context("Hello"),
            AnthropicClientOptions {
                api_key: "test-key".to_owned(),
                session_id: Some("session-1".to_owned()),
                ..Default::default()
            },
        );
        assert!(
            config.default_headers.get("x-session-affinity").is_none(),
            "{provider} custom model must not send session affinity"
        );
    }

    // Catalog models with the explicit compat flag still send it.
    let catalog =
        get_model("fireworks", "accounts/fireworks/models/kimi-k2p6").expect("fireworks model");
    let config = build_anthropic_client_config(
        &catalog,
        &user_context("Hello"),
        AnthropicClientOptions {
            api_key: "test-key".to_owned(),
            session_id: Some("session-2".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(
        config
            .default_headers
            .get("x-session-affinity")
            .map(String::as_str),
        Some("session-2")
    );
}

// pi's supportsTemperature default is true; only the catalog compat flag
// (Opus 4.7/4.8) drops the parameter.
#[test]
fn anthropic_temperature_defaults_on_for_custom_opus_47_aliases() {
    let model = Model::faux("anthropic-messages", "custom", "claude-opus-4-7-alias");
    let payload = build_anthropic_payload(
        &model,
        &user_context("Hello"),
        AnthropicPayloadOptions {
            temperature: Some(0.4),
            ..Default::default()
        },
    );
    assert_eq!(payload["temperature"], json!(0.4));

    let catalog = get_model("anthropic", "claude-opus-4-7").expect("claude-opus-4-7");
    let payload = build_anthropic_payload(
        &catalog,
        &user_context("Hello"),
        AnthropicPayloadOptions {
            temperature: Some(0.4),
            ..Default::default()
        },
    );
    assert!(payload.get("temperature").is_none());
}

// --- anthropic-messages.ts:800,842 forceAdaptiveThinking ---------------------

// pi requires compat.forceAdaptiveThinking === true; model-id substrings alone
// must not trigger adaptive thinking.
#[test]
fn anthropic_adaptive_thinking_requires_explicit_compat_flag() {
    let mut model = Model::faux("anthropic-messages", "custom", "claude-sonnet-5-alias");
    model.reasoning = true;
    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(payload["thinking"]["type"], json!("enabled"));
    assert_eq!(payload["thinking"]["budget_tokens"], json!(8_192));
    assert!(payload.get("output_config").is_none());

    // Catalog models with the compat flag stay adaptive.
    let catalog = get_model("anthropic", "claude-sonnet-5").expect("claude-sonnet-5");
    let payload = build_anthropic_simple_payload(
        &catalog,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(payload["thinking"]["type"], json!("adaptive"));
}

// Without the compat flag, budget-thinking models keep the interleaved
// thinking beta; flagged catalog models drop it.
#[test]
fn anthropic_interleaved_beta_follows_compat_flag_not_model_id() {
    let mut model = Model::faux("anthropic-messages", "custom", "claude-sonnet-5-alias");
    model.reasoning = true;
    let config = build_anthropic_client_config(
        &model,
        &user_context("Hello"),
        AnthropicClientOptions {
            api_key: "test-key".to_owned(),
            ..Default::default()
        },
    );
    assert_eq!(
        config
            .default_headers
            .get("anthropic-beta")
            .map(String::as_str),
        Some("interleaved-thinking-2025-05-14")
    );

    let catalog = get_model("anthropic", "claude-sonnet-5").expect("claude-sonnet-5");
    let config = build_anthropic_client_config(
        &catalog,
        &user_context("Hello"),
        AnthropicClientOptions {
            api_key: "test-key".to_owned(),
            ..Default::default()
        },
    );
    assert!(config.default_headers.get("anthropic-beta").is_none());
}

// --- mistral-conversations.ts:274-293 cached prompt tokens -------------------

fn mistral_usage_output(usage: Value) -> Usage {
    let model = Model::faux("mistral-conversations", "mistral", "devstral-test");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();
    process_mistral_chat_chunks(
        [json!({ "usage": usage, "choices": [] })],
        &mut output,
        &sender,
        &model,
    )
    .expect("process mistral chunks");
    drop(sender);
    drop(stream);
    output.usage
}

// pi reads six cached-token field aliases and subtracts cached tokens from
// the prompt tokens (input excludes the cached prefix).
#[test]
fn mistral_usage_reads_all_six_cached_token_aliases() {
    for usage in [
        json!({ "promptTokens": 10, "completionTokens": 5, "totalTokens": 15,
                "promptTokensDetails": { "cachedTokens": 4 } }),
        json!({ "promptTokens": 10, "completionTokens": 5, "totalTokens": 15,
                "prompt_tokens_details": { "cached_tokens": 4 } }),
        json!({ "promptTokens": 10, "completionTokens": 5, "totalTokens": 15,
                "promptTokenDetails": { "cachedTokens": 4 } }),
        json!({ "promptTokens": 10, "completionTokens": 5, "totalTokens": 15,
                "prompt_token_details": { "cached_tokens": 4 } }),
        json!({ "promptTokens": 10, "completionTokens": 5, "totalTokens": 15,
                "numCachedTokens": 4 }),
        json!({ "promptTokens": 10, "completionTokens": 5, "totalTokens": 15,
                "num_cached_tokens": 4 }),
    ] {
        let parsed = mistral_usage_output(usage.clone());
        assert_eq!(parsed.input, 6, "{usage}");
        assert_eq!(parsed.output, 5, "{usage}");
        assert_eq!(parsed.cache_read, 4, "{usage}");
        assert_eq!(parsed.cache_write, 0, "{usage}");
        assert_eq!(parsed.total_tokens, 15, "{usage}");
    }
}

// pi clamps cached tokens to [0, promptTokens] and treats the first
// non-nullish alias as authoritative (non-numeric values become 0), while
// null aliases fall through to the next one.
#[test]
fn mistral_usage_clamps_and_validates_cached_tokens() {
    // Cached above the prompt count clamps to the prompt count.
    let parsed = mistral_usage_output(json!({
        "promptTokens": 10, "completionTokens": 5, "numCachedTokens": 20,
    }));
    assert_eq!(parsed.input, 0);
    assert_eq!(parsed.cache_read, 10);
    assert_eq!(parsed.total_tokens, 15);

    // Negative cached counts clamp to zero.
    let parsed = mistral_usage_output(json!({
        "promptTokens": 10, "completionTokens": 5, "numCachedTokens": -3,
    }));
    assert_eq!(parsed.input, 10);
    assert_eq!(parsed.cache_read, 0);

    // A non-numeric first alias short-circuits the chain to zero (pi's ??
    // chain stops at the first non-nullish value).
    let parsed = mistral_usage_output(json!({
        "promptTokens": 10, "completionTokens": 5,
        "promptTokensDetails": { "cachedTokens": "4" },
        "num_cached_tokens": 3,
    }));
    assert_eq!(parsed.cache_read, 0);
    assert_eq!(parsed.input, 10);

    // Null aliases fall through to later aliases.
    let parsed = mistral_usage_output(json!({
        "promptTokens": 10, "completionTokens": 5,
        "promptTokensDetails": { "cachedTokens": null },
        "num_cached_tokens": 3,
    }));
    assert_eq!(parsed.cache_read, 3);
    assert_eq!(parsed.input, 7);
}

// pi: totalTokens falls back to input + output + cacheRead + cacheWrite,
// which equals promptTokens + completionTokens after the cache subtraction.
#[test]
fn mistral_usage_total_fallback_includes_cached_tokens() {
    let parsed = mistral_usage_output(json!({
        "promptTokens": 10, "completionTokens": 5, "numCachedTokens": 4,
    }));
    assert_eq!(parsed.input, 6);
    assert_eq!(parsed.cache_read, 4);
    assert_eq!(parsed.total_tokens, 15);
}

// --- mistral SDK URL building -------------------------------------------------

// pi's Mistral SDK strips leading slashes from "/v1/chat/completions" and
// resolves it against the serverURL path, so a custom base ending in /v1 gets
// the segment doubled.
#[test]
fn mistral_chat_completions_url_always_appends_v1() {
    assert_eq!(
        mistral_chat_completions_url("https://api.mistral.ai"),
        "https://api.mistral.ai/v1/chat/completions"
    );
    assert_eq!(
        mistral_chat_completions_url("https://api.mistral.ai/"),
        "https://api.mistral.ai/v1/chat/completions"
    );
    assert_eq!(
        mistral_chat_completions_url("https://proxy.example.com/v1"),
        "https://proxy.example.com/v1/v1/chat/completions"
    );
    assert_eq!(
        mistral_chat_completions_url("https://proxy.example.com/v1/"),
        "https://proxy.example.com/v1/v1/chat/completions"
    );
}

// --- google-generative-ai.ts:405-434 flash detection --------------------------

// pi's Gemini 3 Flash detection also matches the exact ids
// gemini-flash-latest and gemini-flash-lite-latest, which therefore use
// level-based thinking and MINIMAL when disabled.
#[test]
fn google_flash_latest_ids_use_level_based_thinking() {
    for id in ["gemini-flash-latest", "gemini-flash-lite-latest"] {
        let mut model = Model::faux("google-generative-ai", "google", id);
        model.reasoning = true;

        let payload = build_google_simple_payload(
            &model,
            &user_context("Hello"),
            SimpleStreamOptions {
                reasoning: Some(ThinkingLevel::Medium),
                ..Default::default()
            },
        );
        assert_eq!(
            payload["config"]["thinkingConfig"],
            json!({ "includeThoughts": true, "thinkingLevel": "MEDIUM" }),
            "{id}"
        );

        let payload = build_google_simple_payload(
            &model,
            &user_context("Hello"),
            SimpleStreamOptions::default(),
        );
        assert_eq!(
            payload["config"]["thinkingConfig"],
            json!({ "thinkingLevel": "MINIMAL" }),
            "{id}"
        );
    }
}

// --- google-vertex.ts:512-552 no gemma-4 branch --------------------------------

// pi's Vertex implementation has no gemma-4 branch: gemma-4 on Vertex uses
// the generic budget path (-1 dynamic budget, thinkingBudget 0 to disable),
// while the Generative AI API keeps the level-based gemma-4 behavior.
#[test]
fn google_vertex_gemma4_uses_budget_path_unlike_genai() {
    let mut vertex = Model::faux("google-vertex", "google-vertex", "gemma-4-31b-it");
    vertex.reasoning = true;

    let payload = build_google_simple_payload(
        &vertex,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Low),
            ..Default::default()
        },
    );
    assert_eq!(
        payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": -1 })
    );

    let payload = build_google_simple_payload(
        &vertex,
        &user_context("Hello"),
        SimpleStreamOptions::default(),
    );
    assert_eq!(
        payload["config"]["thinkingConfig"],
        json!({ "thinkingBudget": 0 })
    );

    let mut genai = Model::faux("google-generative-ai", "google", "gemma-4-31b-it");
    genai.reasoning = true;

    let payload = build_google_simple_payload(
        &genai,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Low),
            ..Default::default()
        },
    );
    assert_eq!(
        payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingLevel": "MINIMAL" })
    );

    let payload = build_google_simple_payload(
        &genai,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );
    assert_eq!(
        payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingLevel": "HIGH" })
    );

    let payload = build_google_simple_payload(
        &genai,
        &user_context("Hello"),
        SimpleStreamOptions::default(),
    );
    assert_eq!(
        payload["config"]["thinkingConfig"],
        json!({ "thinkingLevel": "MINIMAL" })
    );
}

// --- google-generative-ai.ts:186 generated tool-call ids ----------------------

// pi generates `${name}_${Date.now()}_${++counter}` for missing or duplicate
// function-call ids.
#[test]
fn google_generated_tool_call_ids_use_name_millis_counter() {
    let model = Model::faux("google-generative-ai", "google", "gemini-2.5-flash");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    let function_call_chunk = |call: Value| {
        json!({
            "candidates": [{ "content": { "parts": [{ "functionCall": call }] } }],
        })
    };
    process_google_stream_chunks(
        [
            function_call_chunk(json!({ "name": "lookup", "args": { "q": "a" } })),
            function_call_chunk(json!({ "id": "call-1", "name": "lookup", "args": { "q": "b" } })),
            function_call_chunk(json!({ "id": "call-1", "name": "lookup", "args": { "q": "c" } })),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process google chunks");
    drop(sender);
    drop(stream);

    let ids = output
        .content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.id.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 3);
    assert_eq!(ids[1], "call-1");

    let assert_generated = |id: &str, counter: &str| {
        let parts = id.split('_').collect::<Vec<_>>();
        assert_eq!(parts.len(), 3, "generated id shape: {id}");
        assert_eq!(parts[0], "lookup", "generated id name: {id}");
        let millis = parts[1].parse::<u64>().expect("millis component");
        assert!(millis > 1_700_000_000_000, "millis in id: {id}");
        assert_eq!(parts[2], counter, "counter in id: {id}");
    };
    assert_generated(&ids[0], "1");
    assert_generated(&ids[2], "2");
}

// --- google-generative-ai.ts:218-226 usage totalTokens fallback ----------------

// pi reports `totalTokenCount || 0`; ri must not synthesize a sum.
#[test]
fn google_usage_total_tokens_defaults_to_zero_when_absent() {
    let model = Model::faux("google-generative-ai", "google", "gemini-2.5-flash");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_google_stream_chunks(
        [json!({
            "candidates": [{ "content": { "parts": [{ "text": "Hi" }] }, "finishReason": "STOP" }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 3,
                "thoughtsTokenCount": 2,
            },
        })],
        &mut output,
        &sender,
        &model,
    )
    .expect("process google chunks");
    drop(sender);
    drop(stream);

    assert_eq!(output.usage.input, 10);
    assert_eq!(output.usage.output, 5);
    assert_eq!(output.usage.total_tokens, 0);
}

// --- pi-messages.ts:117,142,330 response-failure diagnostics -------------------

#[test]
fn pi_messages_failure_diagnostic_records_structured_error_body() {
    let model = Model::faux("pi-messages", "radius", "pi-model");
    let body = r#"{"error":{"message":"quota exhausted","code":"payment_required"}}"#;
    let diagnostic = pi_messages_response_failure_diagnostic(
        &model,
        "https://api.example.com/messages",
        402,
        "Payment Required",
        body,
    );

    assert_eq!(diagnostic["type"], json!("pi_messages_response_failure"));
    assert!(diagnostic["timestamp"].as_i64().unwrap_or_default() > 0);
    assert_eq!(
        diagnostic["error"],
        json!({
            "name": "PiMessagesResponseError",
            "message": "402 Payment Required: quota exhausted (payment_required)",
            "code": "payment_required",
        })
    );
    let details = &diagnostic["details"];
    assert_eq!(details["version"], json!(1));
    assert_eq!(details["provider"], json!("radius"));
    assert_eq!(details["model"], json!("pi-model"));
    assert_eq!(details["url"], json!("https://api.example.com/messages"));
    assert_eq!(details["status"], json!(402));
    assert_eq!(details["statusText"], json!("Payment Required"));
    assert_eq!(
        details["error"],
        json!({ "message": "quota exhausted", "code": "payment_required" })
    );
    assert!(details.get("body").is_none());
    assert!(details["timestampMs"].as_i64().unwrap_or_default() > 0);
}

#[test]
fn pi_messages_failure_diagnostic_truncates_unstructured_bodies() {
    let model = Model::faux("pi-messages", "radius", "pi-model");
    let body = "x".repeat(9_000);
    let diagnostic = pi_messages_response_failure_diagnostic(
        &model,
        "https://api.example.com/messages",
        500,
        "Internal Server Error",
        &body,
    );

    let details = &diagnostic["details"];
    assert!(details.get("error").is_none());
    let recorded = details["body"].as_str().expect("body string");
    assert_eq!(recorded.chars().count(), 8_193);
    assert!(recorded.ends_with('\u{2026}'));
    assert!(recorded.starts_with("xxxx"));
    // The formatted message keeps the untruncated body (pi behavior).
    assert_eq!(
        diagnostic["error"]["message"],
        json!(format!("500 Internal Server Error: {body}"))
    );
    assert!(diagnostic["error"].get("code").is_none());
}

// pi's parsePiMessagesErrorBody requires `error` to be a plain object; other
// shapes fall back to the raw-body diagnostic.
#[test]
fn pi_messages_failure_diagnostic_rejects_non_object_error_bodies() {
    let model = Model::faux("pi-messages", "radius", "pi-model");
    for body in [r#"{"error":["boom"]}"#, r#"{"error":"boom"}"#, "plain text"] {
        let diagnostic = pi_messages_response_failure_diagnostic(
            &model,
            "https://api.example.com/messages",
            400,
            "Bad Request",
            body,
        );
        let details = &diagnostic["details"];
        assert!(details.get("error").is_none(), "{body}");
        assert_eq!(details["body"], json!(body), "{body}");
    }

    // Numeric codes are ignored (pi only accepts string codes).
    let diagnostic = pi_messages_response_failure_diagnostic(
        &model,
        "https://api.example.com/messages",
        400,
        "Bad Request",
        r#"{"error":{"message":"boom","code":42}}"#,
    );
    assert!(diagnostic["error"].get("code").is_none());
    assert_eq!(
        diagnostic["error"]["message"],
        json!("400 Bad Request: boom")
    );
}

#[test]
fn pi_messages_append_failure_attaches_diagnostic_and_returns_message() {
    let model = Model::faux("pi-messages", "radius", "pi-model");
    let mut output = empty_assistant_for_model(&model);
    let body = r#"{"error":{"message":"boom","code":"bad"}}"#;

    let message = append_pi_messages_response_failure(
        &mut output,
        &model,
        "https://api.example.com/messages",
        400,
        "Bad Request",
        body,
    );

    assert_eq!(message, "400 Bad Request: boom (bad)");
    assert_eq!(
        message,
        pi_messages_response_error_message(400, "Bad Request", body)
    );
    assert_eq!(output.diagnostics.len(), 1);
    assert_eq!(
        output.diagnostics[0]["type"],
        json!("pi_messages_response_failure")
    );
    assert_eq!(output.diagnostics[0]["details"]["version"], json!(1));
}
