use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use futures::{StreamExt, future::BoxFuture};
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());
const PROXY_ENV_KEYS: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "npm_config_http_proxy",
    "npm_config_https_proxy",
    "npm_config_proxy",
    "npm_config_no_proxy",
    "npm_config_all_proxy",
    "NPM_CONFIG_HTTP_PROXY",
    "NPM_CONFIG_HTTPS_PROXY",
    "NPM_CONFIG_PROXY",
    "NPM_CONFIG_NO_PROXY",
    "NPM_CONFIG_ALL_PROXY",
];

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

#[test]
fn builtin_api_provider_registry_exposes_main_provider_surfaces() {
    ensure_builtin_api_providers();
    let mut apis = get_api_providers()
        .into_iter()
        .map(|provider| provider.api().to_owned())
        .collect::<Vec<_>>();
    apis.sort();

    for expected in [
        "anthropic-messages",
        "azure-openai-responses",
        "bedrock-converse-stream",
        "google-generative-ai",
        "google-vertex",
        "mistral-conversations",
        "openai-codex-responses",
        "openai-completions",
        "openai-responses",
    ] {
        assert!(
            apis.contains(&expected.to_owned()),
            "missing builtin API provider surface {expected}; registered: {apis:?}"
        );
    }
}

#[test]
fn simple_stream_defaults_match_pi_base_options_max_tokens() {
    let mut model = Model::faux("openai-responses", "openai", "small-output");
    model.context_window = 128_000;
    model.max_tokens = 8_192;
    let options = apply_simple_stream_defaults(&model, SimpleStreamOptions::default());
    assert_eq!(options.stream.max_tokens, Some(8_192));

    let mut capped = Model::faux("openai-responses", "openai", "full-window-output");
    capped.context_window = 128_000;
    capped.max_tokens = 128_000;
    let options = apply_simple_stream_defaults(&capped, SimpleStreamOptions::default());
    assert_eq!(options.stream.max_tokens, Some(32_000));

    let mut explicit = SimpleStreamOptions::default();
    explicit.stream.max_tokens = Some(777);
    let options = apply_simple_stream_defaults(&capped, explicit);
    assert_eq!(options.stream.max_tokens, Some(777));

    let mut unknown = Model::faux("openai-responses", "openai", "unknown-output");
    unknown.max_tokens = 0;
    let options = apply_simple_stream_defaults(&unknown, SimpleStreamOptions::default());
    assert_eq!(options.stream.max_tokens, None);
}

#[test]
fn simple_stream_thinking_adjustment_matches_pi_budget_rules() {
    let adjusted = adjust_max_tokens_for_thinking(3_000, 3_000, ThinkingLevel::High, None);
    assert_eq!(
        adjusted,
        ThinkingTokenAdjustment {
            max_tokens: 3_000,
            thinking_budget: 1_976,
        }
    );

    let adjusted = adjust_max_tokens_for_thinking(
        4_096,
        32_000,
        ThinkingLevel::XHigh,
        Some(&ThinkingBudgets {
            high: Some(1_234),
            ..Default::default()
        }),
    );
    assert_eq!(
        adjusted,
        ThinkingTokenAdjustment {
            max_tokens: 5_330,
            thinking_budget: 1_234,
        }
    );
}

#[test]
fn registered_model_apis_have_builtin_provider_implementations() {
    ensure_builtin_api_providers();

    let mut checked = 0usize;
    for provider in get_providers() {
        for model in get_models(&provider) {
            checked += 1;
            assert!(
                get_api_provider(&model.api).is_some(),
                "model {}/{} points at unregistered API provider {}",
                model.provider,
                model.id,
                model.api
            );
        }
    }

    for (provider, model_id) in [
        ("minimax-cn", "MiniMax-M2.7"),
        ("moonshotai", "kimi-k2"),
        ("moonshotai-cn", "kimi-k2"),
    ] {
        let model = get_model(provider, model_id).expect("synthesized known provider model");
        checked += 1;
        assert!(
            get_api_provider(&model.api).is_some(),
            "synthesized model {}/{} points at unregistered API provider {}",
            model.provider,
            model.id,
            model.api
        );
    }

    assert!(checked > 0, "model registry should expose provider models");
}

struct RecordingApiProvider {
    api: String,
    called: Arc<AtomicBool>,
}

impl ApiProvider for RecordingApiProvider {
    fn api(&self) -> &str {
        &self.api
    }

    fn stream(
        &self,
        model: &Model,
        _context: Context,
        _options: StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.called.store(true, Ordering::SeqCst);
        let (sender, stream) = assistant_message_event_stream();
        sender.end(empty_assistant_for_model(model));
        Ok(stream)
    }
}

#[test]
fn api_registry_provider_rejects_mismatched_model_api_before_delegating() {
    let api = format!("record-api-{}", now_millis());
    let source_id = format!("source-{api}");
    let called = Arc::new(AtomicBool::new(false));
    register_api_provider(
        Arc::new(RecordingApiProvider {
            api: api.clone(),
            called: called.clone(),
        }),
        Some(source_id.clone()),
    );

    let provider = get_api_provider(&api).expect("registered provider");
    let model = Model::faux(format!("other-{api}"), "fake-provider", "fake-model");
    let err = match provider.stream(&model, user_context("hello"), StreamOptions::default()) {
        Ok(_) => panic!("mismatched model api should fail"),
        Err(err) => err,
    };

    assert!(matches!(
        err,
        ProviderError::MismatchedApi { actual, expected }
            if actual == model.api && expected == api
    ));
    assert!(!called.load(Ordering::SeqCst));

    unregister_api_providers(&source_id);
    assert!(get_api_provider(&api).is_none());
}

#[test]
fn api_provider_reset_restores_builtins_after_clear_and_removes_custom_providers() {
    let api = format!("reset-api-{}", now_millis());
    register_api_provider(
        Arc::new(RecordingApiProvider {
            api: api.clone(),
            called: Arc::new(AtomicBool::new(false)),
        }),
        Some(format!("source-{api}")),
    );
    assert!(get_api_provider(&api).is_some());

    clear_api_providers();
    ensure_builtin_api_providers();
    assert!(get_api_provider("openai-responses").is_some());
    assert!(get_api_provider("anthropic-messages").is_some());
    assert!(get_api_provider(&api).is_none());

    register_api_provider(
        Arc::new(RecordingApiProvider {
            api: api.clone(),
            called: Arc::new(AtomicBool::new(false)),
        }),
        Some(format!("source-{api}")),
    );
    reset_api_providers();
    assert!(get_api_provider("openai-responses").is_some());
    assert!(get_api_provider("bedrock-converse-stream").is_some());
    assert!(get_api_provider(&api).is_none());
}

fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        ..Default::default()
    }
}

fn text_of(message: &AssistantMessage) -> Option<&str> {
    match message.content.first()? {
        AssistantContent::Text(text) => Some(&text.text),
        _ => None,
    }
}

fn object(value: Value) -> Map<String, Value> {
    value.as_object().cloned().expect("object")
}

fn validate_single_value(schema: Value, input: Value) -> Result<Value, ValidationError> {
    let tool = Tool {
        name: "echo".to_owned(),
        description: "Echo tool".to_owned(),
        parameters: json!({
            "type": "object",
            "required": ["value"],
            "properties": {
                "value": schema
            }
        }),
    };
    let call = ToolCall {
        id: "tool-1".to_owned(),
        name: "echo".to_owned(),
        arguments: object(json!({ "value": input })),
        thought_signature: None,
    };
    validate_tool_arguments(&tool, &call).map(|args| args["value"].clone())
}

fn assert_usage_total_matches_components(label: &str, usage: &Usage) {
    assert_eq!(usage.total_tokens, usage.component_total(), "{label}");
    assert!(usage.total_tokens_match_components(), "{label}");
}

fn empty_assistant(stop_reason: StopReason) -> AssistantMessage {
    faux_assistant_message(
        Vec::<AssistantContent>::new(),
        FauxAssistantOptions {
            stop_reason: Some(stop_reason),
            ..Default::default()
        },
    )
}

fn error_assistant(error_message: &str) -> AssistantMessage {
    faux_assistant_message(
        Vec::<AssistantContent>::new(),
        FauxAssistantOptions {
            stop_reason: Some(StopReason::Error),
            error_message: Some(error_message.to_owned()),
            ..Default::default()
        },
    )
}

async fn collect_events(mut stream: AssistantMessageEventStream) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
    }
    events
}

#[tokio::test]
async fn assistant_event_stream_ignores_pushes_after_terminal_event() {
    let (sender, stream) = assistant_message_event_stream();
    let done = empty_assistant(StopReason::Stop);
    let late = error_assistant("late error");

    sender.push(AssistantMessageEvent::Done {
        reason: StopReason::Stop,
        message: done.clone(),
    });
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: late,
    });
    drop(sender);

    let events = tokio::time::timeout(Duration::from_millis(50), collect_events(stream))
        .await
        .expect("stream should close after terminal event");
    assert_eq!(events.len(), 1);
    assert!(matches!(
        &events[0],
        AssistantMessageEvent::Done { message, .. } if message == &done
    ));
}

#[tokio::test]
async fn assistant_event_stream_end_closes_without_terminal_event() {
    let (sender, mut stream) = assistant_message_event_stream();

    sender.end(empty_assistant(StopReason::Stop));

    let next = tokio::time::timeout(Duration::from_millis(50), stream.next())
        .await
        .expect("stream should close when sender is ended");
    assert!(next.is_none());
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

#[test]
fn supports_xhigh_model_metadata_port() {
    let opus_46 = get_model("anthropic", "claude-opus-4-6").expect("model");
    assert!(get_supported_thinking_levels(&opus_46).contains(&ThinkingLevel::XHigh));
    assert_eq!(
        opus_46.thinking_level_map.get(&ThinkingLevel::XHigh),
        Some(&Some("max".to_owned()))
    );

    let opus_47 = get_model("anthropic", "claude-opus-4-7").expect("model");
    assert!(get_supported_thinking_levels(&opus_47).contains(&ThinkingLevel::XHigh));
    assert_eq!(
        opus_47.thinking_level_map.get(&ThinkingLevel::XHigh),
        Some(&Some("xhigh".to_owned()))
    );

    let sonnet = get_model("anthropic", "claude-sonnet-4-5").expect("model");
    assert!(!get_supported_thinking_levels(&sonnet).contains(&ThinkingLevel::XHigh));

    for model_id in ["gpt-5.4", "gpt-5.5"] {
        let model = get_model("openai-codex", model_id).expect("model");
        assert!(get_supported_thinking_levels(&model).contains(&ThinkingLevel::XHigh));
    }

    for (provider, model_id) in [
        ("deepseek", "deepseek-v4-flash"),
        ("opencode-go", "deepseek-v4-flash"),
        ("openrouter", "deepseek/deepseek-v4-flash"),
    ] {
        let model = get_model(provider, model_id).expect("model");
        assert_eq!(
            get_supported_thinking_levels(&model),
            vec![
                ThinkingLevel::Off,
                ThinkingLevel::High,
                ThinkingLevel::XHigh
            ]
        );
        assert_eq!(
            model.thinking_level_map.get(&ThinkingLevel::XHigh),
            Some(&Some("max".to_owned()))
        );
    }

    let openrouter_opus =
        get_model("openrouter", "anthropic/claude-opus-4.6").expect("openrouter opus");
    assert!(get_supported_thinking_levels(&openrouter_opus).contains(&ThinkingLevel::XHigh));
    assert_eq!(
        openrouter_opus
            .thinking_level_map
            .get(&ThinkingLevel::XHigh),
        Some(&Some("max".to_owned()))
    );
}

#[tokio::test]
async fn unsupported_xhigh_reasoning_returns_error_message_without_network() {
    for api in ["openai-responses", "openai-completions"] {
        let mut model = get_model("openai", "gpt-5-mini").expect("gpt-5-mini");
        model.api = api.to_owned();
        model.base_url = "http://127.0.0.1:9".to_owned();
        let options = SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        };

        let message = complete_simple(&model, user_context("hello"), options)
            .await
            .expect("error stream");

        assert_eq!(message.stop_reason, StopReason::Error, "{api}");
        assert!(
            message
                .error_message
                .as_deref()
                .unwrap_or_default()
                .contains("xhigh"),
            "{api}: {:?}",
            message.error_message
        );
    }
}

#[test]
fn fireworks_and_together_model_metadata_match_provider_catalog() {
    let fireworks =
        get_model("fireworks", "accounts/fireworks/models/kimi-k2p6").expect("fireworks kimi");
    assert_eq!(fireworks.api, "anthropic-messages");
    assert_eq!(fireworks.provider, "fireworks");
    assert_eq!(fireworks.base_url, "https://api.fireworks.ai/inference");
    assert!(fireworks.reasoning);
    assert_eq!(fireworks.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(fireworks.context_window, 262_000);
    assert_eq!(fireworks.max_tokens, 262_000);
    assert_eq!(
        fireworks.cost,
        ModelCost {
            input: 0.95,
            output: 4.0,
            cache_read: 0.16,
            cache_write: 0.0,
        }
    );
    assert_eq!(
        fireworks.compat,
        Some(json!({
            "sendSessionAffinityHeaders": true,
            "supportsEagerToolInputStreaming": false,
            "supportsCacheControlOnTools": false,
            "supportsLongCacheRetention": false,
        }))
    );

    let fireworks_router = get_model("fireworks", "accounts/fireworks/routers/kimi-k2p5-turbo")
        .expect("fireworks router");
    assert_eq!(fireworks_router.api, "anthropic-messages");
    assert_eq!(
        fireworks_router.base_url,
        "https://api.fireworks.ai/inference"
    );
    assert_eq!(
        fireworks_router.input,
        vec![InputKind::Text, InputKind::Image]
    );

    let together = get_model("together", "moonshotai/Kimi-K2.6").expect("together kimi");
    assert_eq!(together.api, "openai-completions");
    assert_eq!(together.provider, "together");
    assert_eq!(together.base_url, "https://api.together.ai/v1");
    assert!(together.reasoning);
    assert_eq!(
        together.thinking_level_map,
        BTreeMap::from([
            (ThinkingLevel::Minimal, None),
            (ThinkingLevel::Low, None),
            (ThinkingLevel::Medium, None),
        ])
    );
    assert_eq!(together.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(together.context_window, 262_144);
    assert_eq!(together.max_tokens, 131_000);
    assert_eq!(
        together.cost,
        ModelCost {
            input: 1.2,
            output: 4.5,
            cache_read: 0.2,
            cache_write: 0.0,
        }
    );
    assert_eq!(
        together.compat,
        Some(json!({
            "supportsStore": false,
            "supportsDeveloperRole": false,
            "supportsReasoningEffort": false,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "together",
            "supportsStrictMode": false,
            "supportsLongCacheRetention": false,
        }))
    );

    let gpt_oss = get_model("together", "openai/gpt-oss-120b").expect("gpt oss");
    assert_eq!(
        gpt_oss.thinking_level_map,
        BTreeMap::from([(ThinkingLevel::Off, None), (ThinkingLevel::Minimal, None)])
    );
    assert_eq!(
        gpt_oss.compat,
        Some(json!({
            "supportsReasoningEffort": true,
            "thinkingFormat": "openai",
        }))
    );

    let deepseek = get_model("together", "deepseek-ai/DeepSeek-V4-Pro").expect("deepseek");
    assert_eq!(
        deepseek.thinking_level_map,
        BTreeMap::from([
            (ThinkingLevel::Minimal, None),
            (ThinkingLevel::Low, None),
            (ThinkingLevel::Medium, None),
            (ThinkingLevel::High, Some("high".to_owned())),
            (ThinkingLevel::XHigh, None),
        ])
    );
    assert_eq!(
        deepseek.compat,
        Some(json!({
            "supportsReasoningEffort": true,
            "thinkingFormat": "together",
        }))
    );

    let minimax = get_model("together", "MiniMaxAI/MiniMax-M2.7").expect("minimax");
    assert_eq!(
        minimax.thinking_level_map,
        BTreeMap::from([
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, None),
            (ThinkingLevel::Low, None),
            (ThinkingLevel::Medium, None),
        ])
    );
    assert_eq!(
        minimax
            .compat
            .as_ref()
            .and_then(|compat| compat.get("supportsReasoningEffort"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert!(
        minimax
            .compat
            .as_ref()
            .and_then(|compat| compat.get("thinkingFormat"))
            .is_none()
    );
}

#[test]
fn fireworks_and_together_env_keys_resolve_from_provider_specific_variables() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["FIREWORKS_API_KEY", "TOGETHER_API_KEY"]);

    set_env("FIREWORKS_API_KEY", "test-fireworks-key");
    assert_eq!(
        find_env_keys("fireworks"),
        Some(vec!["FIREWORKS_API_KEY".to_owned()])
    );
    assert_eq!(
        get_env_api_key("fireworks"),
        Some("test-fireworks-key".to_owned())
    );

    set_env("TOGETHER_API_KEY", "test-together-key");
    assert_eq!(
        find_env_keys("together"),
        Some(vec!["TOGETHER_API_KEY".to_owned()])
    );
    assert_eq!(
        get_env_api_key("together"),
        Some("test-together-key".to_owned())
    );
}

#[test]
fn cloudflare_model_metadata_and_base_url_resolution_match_provider_catalog() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"]);

    let gateway_workers = get_model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    )
    .expect("cloudflare gateway workers model");
    assert!(is_cloudflare_provider(&gateway_workers.provider));
    assert_eq!(gateway_workers.api, "openai-completions");
    assert_eq!(
        gateway_workers.base_url,
        CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL
    );
    assert!(gateway_workers.reasoning);
    assert_eq!(
        gateway_workers.input,
        vec![InputKind::Text, InputKind::Image]
    );
    assert_eq!(gateway_workers.context_window, 256_000);
    assert_eq!(gateway_workers.max_tokens, 256_000);
    assert_eq!(
        gateway_workers.compat,
        Some(json!({
            "sendSessionAffinityHeaders": true,
            "supportsReasoningEffort": false,
            "supportsStrictMode": false,
            "supportsLongCacheRetention": false,
            "maxTokensField": "max_tokens",
        }))
    );
    assert_eq!(
        resolve_cloudflare_base_url(&gateway_workers).expect_err("missing env"),
        "CLOUDFLARE_ACCOUNT_ID is required for provider cloudflare-ai-gateway but is not set."
    );

    set_env("CLOUDFLARE_ACCOUNT_ID", "account-id");
    set_env("CLOUDFLARE_GATEWAY_ID", "gateway-id");
    assert_eq!(
        resolve_cloudflare_base_url(&gateway_workers).expect("resolved gateway url"),
        "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/compat"
    );
    let mut custom_placeholders = gateway_workers.clone();
    custom_placeholders.base_url =
        "https://example.test/{lower}/{1BAD}/{}/{CLOUDFLARE_ACCOUNT_ID}/{BROKEN".to_owned();
    assert_eq!(
        resolve_cloudflare_base_url(&custom_placeholders).expect("resolved custom placeholders"),
        "https://example.test/{lower}/{1BAD}/{}/account-id/{BROKEN"
    );

    let workers = get_model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6")
        .expect("cloudflare workers model");
    assert_eq!(workers.api, "openai-completions");
    assert_eq!(workers.base_url, CLOUDFLARE_WORKERS_AI_BASE_URL);
    assert!(workers.reasoning);
    assert_eq!(workers.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(
        resolve_cloudflare_base_url(&workers).expect("resolved workers url"),
        "https://api.cloudflare.com/client/v4/accounts/account-id/ai/v1"
    );

    let gateway_openai =
        get_model("cloudflare-ai-gateway", "gpt-5.1").expect("cloudflare openai model");
    assert_eq!(gateway_openai.api, "openai-responses");
    assert_eq!(
        gateway_openai.base_url,
        CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL
    );
    assert_eq!(
        resolve_cloudflare_base_url(&gateway_openai).expect("resolved openai url"),
        "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/openai"
    );

    let gateway_anthropic = get_model("cloudflare-ai-gateway", "claude-sonnet-4-5")
        .expect("cloudflare anthropic model");
    assert_eq!(gateway_anthropic.api, "anthropic-messages");
    assert_eq!(
        gateway_anthropic.base_url,
        CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL
    );
    assert_eq!(
        resolve_cloudflare_base_url(&gateway_anthropic).expect("resolved anthropic url"),
        "https://gateway.ai.cloudflare.com/v1/account-id/gateway-id/anthropic"
    );
}

#[test]
fn openai_compatible_provider_base_urls_match_provider_catalog() {
    let expected = [
        (
            "xai",
            "grok-3-fast",
            "openai-completions",
            "https://api.x.ai/v1",
        ),
        (
            "cerebras",
            "gpt-oss-120b",
            "openai-completions",
            "https://api.cerebras.ai/v1",
        ),
        (
            "huggingface",
            "moonshotai/Kimi-K2.5",
            "openai-completions",
            "https://router.huggingface.co/v1",
        ),
        (
            "minimax",
            "MiniMax-M2.7",
            "anthropic-messages",
            "https://api.minimax.io/anthropic",
        ),
        (
            "kimi-coding",
            "kimi-k2-thinking",
            "anthropic-messages",
            "https://api.kimi.com/coding",
        ),
        (
            "vercel-ai-gateway",
            "google/gemini-2.5-flash",
            "anthropic-messages",
            "https://ai-gateway.vercel.sh",
        ),
        (
            "xiaomi",
            "mimo-v2.5-pro",
            "openai-completions",
            "https://api.xiaomimimo.com/v1",
        ),
        (
            "xiaomi-token-plan-cn",
            "mimo-v2.5-pro",
            "openai-completions",
            "https://token-plan-cn.xiaomimimo.com/v1",
        ),
        (
            "xiaomi-token-plan-ams",
            "mimo-v2.5-pro",
            "openai-completions",
            "https://token-plan-ams.xiaomimimo.com/v1",
        ),
        (
            "xiaomi-token-plan-sgp",
            "mimo-v2.5-pro",
            "openai-completions",
            "https://token-plan-sgp.xiaomimimo.com/v1",
        ),
    ];

    for (provider, model_id, api, base_url) in expected {
        let model = get_model(provider, model_id)
            .unwrap_or_else(|| panic!("missing model registry entry: {provider}/{model_id}"));
        assert_eq!(model.api, api, "{provider}/{model_id} api");
        assert_eq!(model.base_url, base_url, "{provider}/{model_id} base URL");
        assert_ne!(
            model.base_url, "https://example.invalid",
            "{provider}/{model_id} should have a live provider endpoint"
        );
    }

    let kimi = get_model("kimi-coding", "kimi-k2-thinking").expect("kimi model");
    assert_eq!(
        kimi.headers.get("User-Agent").map(String::as_str),
        Some("KimiCLI/1.5")
    );
}

#[test]
fn opencode_model_metadata_and_env_key_match_provider_catalog() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["OPENCODE_API_KEY"]);

    assert!(!get_models("opencode").is_empty());
    assert!(!get_models("opencode-go").is_empty());

    let zen_pickle = get_model("opencode", "big-pickle").expect("zen pickle");
    assert_eq!(zen_pickle.api, "openai-completions");
    assert_eq!(zen_pickle.provider, "opencode");
    assert_eq!(zen_pickle.base_url, "https://opencode.ai/zen/v1");
    assert!(zen_pickle.reasoning);
    assert_eq!(zen_pickle.input, vec![InputKind::Text]);
    assert_eq!(zen_pickle.context_window, 200_000);
    assert_eq!(zen_pickle.max_tokens, 128_000);

    let zen_claude = get_model("opencode", "claude-sonnet-4-5").expect("zen claude");
    assert_eq!(zen_claude.api, "anthropic-messages");
    assert_eq!(zen_claude.base_url, "https://opencode.ai/zen");
    assert_eq!(zen_claude.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(zen_claude.context_window, 200_000);
    assert_eq!(zen_claude.max_tokens, 64_000);
    assert_eq!(
        zen_claude.cost,
        ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        }
    );

    let zen_gemini = get_model("opencode", "gemini-3-flash").expect("zen gemini");
    assert_eq!(zen_gemini.api, "google-generative-ai");
    assert_eq!(zen_gemini.base_url, "https://opencode.ai/zen/v1");
    assert_eq!(zen_gemini.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(
        zen_gemini.thinking_level_map,
        BTreeMap::from([(ThinkingLevel::Off, None)])
    );

    let zen_codex = get_model("opencode", "gpt-5.2-codex").expect("zen codex");
    assert_eq!(zen_codex.api, "openai-responses");
    assert_eq!(zen_codex.base_url, "https://opencode.ai/zen/v1");
    assert!(get_supported_thinking_levels(&zen_codex).contains(&ThinkingLevel::XHigh));
    assert_eq!(zen_codex.context_window, 400_000);
    assert_eq!(zen_codex.max_tokens, 128_000);

    let go_deepseek = get_model("opencode-go", "deepseek-v4-flash").expect("go deepseek");
    assert_eq!(go_deepseek.api, "openai-completions");
    assert_eq!(go_deepseek.base_url, "https://opencode.ai/zen/go/v1");
    assert_eq!(
        go_deepseek.thinking_level_map,
        BTreeMap::from([
            (ThinkingLevel::Minimal, None),
            (ThinkingLevel::Low, None),
            (ThinkingLevel::Medium, None),
            (ThinkingLevel::High, Some("high".to_owned())),
            (ThinkingLevel::XHigh, Some("max".to_owned())),
        ])
    );
    assert_eq!(
        go_deepseek.compat,
        Some(json!({
            "requiresReasoningContentOnAssistantMessages": true,
            "thinkingFormat": "deepseek",
        }))
    );

    let go_kimi = get_model("opencode-go", "kimi-k2.5").expect("go kimi");
    assert_eq!(go_kimi.api, "openai-completions");
    assert_eq!(go_kimi.base_url, "https://opencode.ai/zen/go/v1");
    assert_eq!(go_kimi.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(go_kimi.context_window, 262_144);
    assert_eq!(go_kimi.max_tokens, 65_536);

    let go_minimax = get_model("opencode-go", "minimax-m2.5").expect("go minimax");
    assert_eq!(go_minimax.api, "anthropic-messages");
    assert_eq!(go_minimax.base_url, "https://opencode.ai/zen/go");
    assert_eq!(go_minimax.context_window, 204_800);
    assert_eq!(go_minimax.max_tokens, 65_536);

    set_env("OPENCODE_API_KEY", "opencode-token");
    assert_eq!(
        find_env_keys("opencode"),
        Some(vec!["OPENCODE_API_KEY".to_owned()])
    );
    assert_eq!(
        find_env_keys("opencode-go"),
        Some(vec!["OPENCODE_API_KEY".to_owned()])
    );
    assert_eq!(
        get_env_api_key("opencode-go"),
        Some("opencode-token".to_owned())
    );
}

#[test]
fn openrouter_image_model_registry_matches_generated_catalog() {
    let model = get_image_model("openrouter", "google/gemini-2.5-flash-image").expect("model");
    assert_eq!(model.id, "google/gemini-2.5-flash-image");
    assert_eq!(model.name, "Google: Nano Banana (Gemini 2.5 Flash Image)");
    assert_eq!(model.api, "openrouter-images");
    assert_eq!(model.provider, "openrouter");
    assert_eq!(model.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(model.input, vec![InputKind::Image, InputKind::Text]);
    assert_eq!(model.output, vec![OutputKind::Image, OutputKind::Text]);
    assert_eq!(
        model.cost,
        ModelCost {
            input: 0.3,
            output: 2.5,
            cache_read: 0.03,
            cache_write: 0.08333333333333334,
        }
    );
    assert_eq!(get_image_providers(), vec!["openrouter".to_owned()]);

    let models = get_image_models("openrouter");
    assert_eq!(models.len(), 28);
    assert_eq!(
        models
            .iter()
            .map(|candidate| candidate.id.as_str())
            .collect::<Vec<_>>(),
        vec![
            "black-forest-labs/flux.2-flex",
            "black-forest-labs/flux.2-klein-4b",
            "black-forest-labs/flux.2-max",
            "black-forest-labs/flux.2-pro",
            "bytedance-seed/seedream-4.5",
            "google/gemini-2.5-flash-image",
            "google/gemini-3-pro-image-preview",
            "google/gemini-3.1-flash-image-preview",
            "openai/gpt-5-image",
            "openai/gpt-5-image-mini",
            "openai/gpt-5.4-image-2",
            "openrouter/auto",
            "recraft/recraft-v3",
            "recraft/recraft-v4",
            "recraft/recraft-v4-pro",
            "recraft/recraft-v4-pro-vector",
            "recraft/recraft-v4-vector",
            "recraft/recraft-v4.1",
            "recraft/recraft-v4.1-pro",
            "recraft/recraft-v4.1-pro-vector",
            "recraft/recraft-v4.1-utility",
            "recraft/recraft-v4.1-utility-pro",
            "recraft/recraft-v4.1-vector",
            "sourceful/riverflow-v2-fast",
            "sourceful/riverflow-v2-fast-preview",
            "sourceful/riverflow-v2-max-preview",
            "sourceful/riverflow-v2-pro",
            "sourceful/riverflow-v2-standard-preview",
        ]
    );
    assert!(
        models
            .iter()
            .all(|candidate| candidate.api == "openrouter-images"
                && candidate.provider == "openrouter"
                && candidate.base_url == "https://openrouter.ai/api/v1")
    );
    assert!(
        models
            .iter()
            .any(|candidate| candidate.id == "black-forest-labs/flux.2-pro")
    );
    let sourceful = get_image_model("openrouter", "sourceful/riverflow-v2-standard-preview")
        .expect("sourceful model");
    assert_eq!(sourceful.name, "Sourceful: Riverflow V2 Standard Preview");
    assert_eq!(sourceful.input, vec![InputKind::Text, InputKind::Image]);
    assert_eq!(sourceful.output, vec![OutputKind::Image]);
}

#[test]
fn registered_image_model_apis_have_builtin_provider_implementations() {
    ensure_builtin_images_api_providers();
    let mut image_apis = get_images_api_providers()
        .into_iter()
        .map(|provider| provider.api().to_owned())
        .collect::<Vec<_>>();
    image_apis.sort();
    assert!(
        image_apis.contains(&"openrouter-images".to_owned()),
        "missing builtin OpenRouter Images provider; registered: {image_apis:?}"
    );

    let mut checked = 0usize;
    for provider in get_image_providers() {
        for model in get_image_models(&provider) {
            checked += 1;
            assert!(
                get_images_api_provider(&model.api).is_some(),
                "image model {}/{} points at unregistered Images API provider {}",
                model.provider,
                model.id,
                model.api
            );
        }
    }

    assert!(checked > 0, "image model registry should expose models");
}

#[test]
fn openrouter_images_payload_uses_chat_completions_image_modalities() {
    let model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    let context = ImagesContext {
        input: vec![ImagesContent::text("Generate a dog")],
    };

    let payload = build_openrouter_images_payload(&model, &context);

    assert_eq!(payload["model"], "google/gemini-3.1-flash-image-preview");
    assert_eq!(payload["stream"], false);
    assert_eq!(payload["modalities"], json!(["image", "text"]));
    assert_eq!(payload["messages"][0]["role"], "user");
    assert_eq!(
        payload["messages"][0]["content"][0],
        json!({ "type": "text", "text": "Generate a dog" })
    );
}

#[test]
fn openrouter_images_payload_formats_image_input_and_image_only_output() {
    let model = get_image_model("openrouter", "black-forest-labs/flux.2-pro").expect("model");
    let context = ImagesContext {
        input: vec![
            ImagesContent::text("Create a variation"),
            ImagesContent::Image(ImageContent {
                data: "ZmFrZQ==".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ],
    };

    let payload = build_openrouter_images_payload(&model, &context);

    assert_eq!(payload["modalities"], json!(["image"]));
    assert_eq!(
        payload["messages"][0]["content"][1],
        json!({
            "type": "image_url",
            "image_url": { "url": "data:image/png;base64,ZmFrZQ==" },
        })
    );
}

#[test]
fn openrouter_images_response_returns_text_images_response_id_and_usage() {
    let model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    let response = json!({
        "id": "img-1",
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "prompt_tokens_details": { "cached_tokens": 0 }
        },
        "choices": [
            {
                "message": {
                    "content": "Here is your image.",
                    "images": [
                        { "image_url": "data:image/png;base64,ZmFrZS1wbmc=" },
                        { "image_url": { "url": "https://example.com/not-inline.png" } },
                        { "image_url": "not-a-data-url" }
                    ]
                }
            }
        ]
    });

    let output = parse_openrouter_images_response(&model, &response);

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-1"));
    assert!(output.timestamp > 0);
    assert_eq!(output.output.len(), 2);
    assert!(matches!(
        &output.output[0],
        ImagesContent::Text(text) if text.text == "Here is your image."
    ));
    assert!(matches!(
        &output.output[1],
        ImagesContent::Image(image)
            if image.mime_type == "image/png" && image.data == "ZmFrZS1wbmc="
    ));
    let usage = output.usage.expect("usage");
    assert_eq!(usage.input, 12);
    assert_eq!(usage.output, 34);
    assert_eq!(usage.cache_read, 0);
    assert_eq!(usage.cache_write, 0);
    assert_eq!(usage.total_tokens, 46);
}

#[test]
fn openrouter_images_usage_and_error_mapping_match_provider() {
    let mut model = get_image_model("openrouter", "google/gemini-2.5-flash-image").expect("model");
    model
        .headers
        .insert("HTTP-Referer".to_owned(), "https://example.com".to_owned());

    let usage = parse_openrouter_images_usage(
        &json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {
                "cached_tokens": 40,
                "cache_write_tokens": 15
            }
        }),
        &model,
    );
    assert_eq!(usage.input, 60);
    assert_eq!(usage.output, 10);
    assert_eq!(usage.cache_read, 25);
    assert_eq!(usage.cache_write, 15);
    assert_eq!(usage.total_tokens, 110);
    assert_usage_total_matches_components("openrouter images", &usage);
    assert!(usage.cost.total > 0.0);

    let headers = build_openrouter_images_default_headers(
        &model,
        &BTreeMap::from([
            (
                "HTTP-Referer".to_owned(),
                "https://override.example".to_owned(),
            ),
            ("X-Test".to_owned(), "yes".to_owned()),
        ]),
    );
    assert_eq!(
        headers.get("HTTP-Referer").map(String::as_str),
        Some("https://override.example")
    );
    assert_eq!(headers.get("X-Test").map(String::as_str), Some("yes"));

    let aborted = openrouter_images_error(&model, "Request aborted", true);
    assert_eq!(aborted.stop_reason, ImagesStopReason::Aborted);
    assert_eq!(aborted.error_message.as_deref(), Some("Request aborted"));
}

#[test]
fn openrouter_images_retry_delay_respects_headers_and_backoff() {
    assert_eq!(
        openrouter_images_retry_delay_ms(429, "rate limit", Some("1500"), None, 0, 3, None, 0),
        Some(1500)
    );
    assert_eq!(
        openrouter_images_retry_delay_ms(503, "overloaded", None, Some("2.5"), 0, 3, None, 0),
        Some(2500)
    );
    assert_eq!(
        openrouter_images_retry_delay_ms(500, "server error", None, None, 1, 3, Some(100), 0),
        Some(100)
    );
    assert_eq!(
        openrouter_images_retry_delay_ms(
            400,
            "upstream connect refused",
            None,
            None,
            0,
            3,
            None,
            0,
        ),
        Some(1000)
    );
    assert_eq!(
        openrouter_images_retry_delay_ms(429, "rate limit", None, None, 3, 3, None, 0),
        None
    );
    assert_eq!(
        openrouter_images_retry_delay_ms(400, "bad request", None, None, 0, 3, None, 0),
        None
    );
}

#[tokio::test]
async fn builtin_openrouter_images_provider_posts_json_and_parses_response() {
    let body = concat!(
        "{\"id\":\"img-http\",\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2},",
        "\"choices\":[{\"message\":{\"content\":\"Done\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,ZmFrZQ==\"}]}}]}"
    );
    let (base_url, request_task) = mock_json_server(body).await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    model.headers.insert(
        "HTTP-Referer".to_owned(),
        "https://model-header.example".to_owned(),
    );
    model
        .headers
        .insert("X-Model-Header".to_owned(), "model".to_owned());
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate an icon")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            headers: BTreeMap::from([("HTTP-Referer".to_owned(), "https://ri.test".to_owned())]),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-http"));
    assert_eq!(output.usage.as_ref().map(|usage| usage.input), Some(5));
    assert_eq!(output.usage.as_ref().map(|usage| usage.output), Some(2));
    assert!(matches!(
        &output.output[0],
        ImagesContent::Text(text) if text.text == "Done"
    ));
    assert!(matches!(
        &output.output[1],
        ImagesContent::Image(image) if image.mime_type == "image/png" && image.data == "ZmFrZQ=="
    ));
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer openrouter-key")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("http-referer: https://ri.test")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-model-header: model")
    );
    assert!(
        !request
            .to_ascii_lowercase()
            .contains("http-referer: https://model-header.example")
    );
    assert!(request.contains("\"stream\":false"));
    assert!(request.contains("\"modalities\":[\"image\",\"text\"]"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_preserves_custom_authorization_header() {
    let body = concat!(
        "{\"id\":\"img-auth\",\"choices\":[{\"message\":{\"content\":\"Auth OK\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,YXV0aA==\"}]}}]}"
    );
    let (base_url, request_task) = mock_json_server(body).await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate an authenticated image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            headers: BTreeMap::from([(
                "Authorization".to_owned(),
                "Bearer upstream-image-token".to_owned(),
            )]),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-auth"));
    assert!(matches!(
        output.output.get(1),
        Some(ImagesContent::Image(image)) if image.data == "YXV0aA=="
    ));
    assert!(lower_request.contains("authorization: bearer upstream-image-token"));
    assert!(!lower_request.contains("authorization: bearer openrouter-key"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_retries_retryable_errors() {
    let success = concat!(
        r#"{"id":"img-retry","choices":[{"message":{"content":"Done","#,
        "\"images\":[{\"image_url\":\"data:image/png;base64,cmV0cnk=\"}]}}]}"
    );
    let (base_url, requests_task) = mock_json_status_sequence_server(vec![
        (
            429,
            "Too Many Requests",
            r#"{"error":{"message":"Rate limit exceeded"}}"#,
        ),
        (200, "OK", success),
    ])
    .await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    let seen_responses = Arc::new(Mutex::new(Vec::new()));

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate retry image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            max_retries: Some(1),
            max_retry_delay_ms: Some(0),
            response_hooks: vec![Arc::new(RecordingImagesResponseHook {
                seen: seen_responses.clone(),
            })],
            ..Default::default()
        },
    )
    .await
    .expect("generate images");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert!(matches!(
        output.output.get(1),
        Some(ImagesContent::Image(image)) if image.data == "cmV0cnk="
    ));
    let requests = requests_task.await.expect("request task");
    assert_eq!(requests.len(), 2);
    assert!(requests[0].starts_with("POST /chat/completions HTTP/1.1"));
    assert!(requests[1].starts_with("POST /chat/completions HTTP/1.1"));
    assert_eq!(
        seen_responses
            .lock()
            .expect("responses")
            .iter()
            .map(|response| response.status)
            .collect::<Vec<_>>(),
        vec![429, 200]
    );
}

#[tokio::test]
async fn builtin_openrouter_images_provider_respects_abort_flag_during_retry_backoff() {
    let (base_url, requests_task) = mock_http_sequence_server(vec![MockHttpSequenceResponse {
        status: 429,
        reason: "Too Many Requests",
        content_type: "application/json",
        headers: vec![("retry-after-ms", "5000")],
        body: r#"{"error":{"message":"Rate limit exceeded"}}"#,
    }])
    .await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_for_request = abort_flag.clone();

    let request_handle = tokio::spawn(async move {
        generate_images(
            &model,
            ImagesContext {
                input: vec![ImagesContent::text("Generate retry abort image")],
            },
            ImagesOptions {
                api_key: Some("openrouter-key".to_owned()),
                max_retries: Some(1),
                abort_flag: Some(abort_flag_for_request),
                ..Default::default()
            },
        )
        .await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    abort_flag.store(true, Ordering::SeqCst);
    let output = tokio::time::timeout(Duration::from_secs(1), request_handle)
        .await
        .expect("image retry should observe abort")
        .expect("join image request")
        .expect("generate images");
    let requests = requests_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
    assert_eq!(output.error_message.as_deref(), Some("Request was aborted"));
    assert!(output.output.is_empty());
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_respects_abort_flag_while_waiting_for_response_headers()
{
    let (base_url, request_rx, server_task) = mock_hanging_response_server().await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_for_request = abort_flag.clone();

    let request_handle = tokio::spawn(async move {
        generate_images(
            &model,
            ImagesContext {
                input: vec![ImagesContent::text("Generate abortable request image")],
            },
            ImagesOptions {
                api_key: Some("openrouter-key".to_owned()),
                abort_flag: Some(abort_flag_for_request),
                ..Default::default()
            },
        )
        .await
    });
    let request = tokio::time::timeout(Duration::from_secs(1), request_rx)
        .await
        .expect("hanging server should receive request")
        .expect("request from hanging server");
    abort_flag.store(true, Ordering::SeqCst);
    let output = tokio::time::timeout(Duration::from_secs(1), request_handle)
        .await
        .expect("image request should observe abort while waiting for headers")
        .expect("join image request")
        .expect("generate images");
    server_task.abort();
    let _ = server_task.await;

    assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
    assert_eq!(output.error_message.as_deref(), Some("Request was aborted"));
    assert!(output.output.is_empty());
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_respects_request_timeout() {
    let delayed_body = concat!(
        "{\"id\":\"img-timeout\",\"choices\":[{\"message\":{\"content\":\"Too slow\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,c2xvdw==\"}]}}]}"
    );
    let (base_url, request_task) = mock_delayed_binary_server(
        vec![b"{".to_vec(), delayed_body.as_bytes().to_vec()],
        "application/json",
        Duration::from_millis(100),
    )
    .await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate timeout image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            timeout_ms: Some(20),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    assert!(output.output.is_empty());
    assert!(
        output.error_message.as_deref().is_some_and(|message| {
            let message = message.to_ascii_lowercase();
            message.contains("timed out") || message.contains("error decoding response body")
        }),
        "{output:?}"
    );
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_respects_abort_flag_while_reading_response_body() {
    let delayed_body = concat!(
        "\"id\":\"img-abort\",\"choices\":[{\"message\":{\"content\":\"Too slow\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,YWJvcnQ=\"}]}}]}"
    );
    let (base_url, request_task) = mock_delayed_binary_server(
        vec![b"{".to_vec(), delayed_body.as_bytes().to_vec()],
        "application/json",
        Duration::from_millis(200),
    )
    .await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_for_request = abort_flag.clone();

    let request_handle = tokio::spawn(async move {
        generate_images(
            &model,
            ImagesContext {
                input: vec![ImagesContent::text("Generate abortable image")],
            },
            ImagesOptions {
                api_key: Some("openrouter-key".to_owned()),
                abort_flag: Some(abort_flag_for_request),
                ..Default::default()
            },
        )
        .await
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    abort_flag.store(true, Ordering::SeqCst);
    let output = tokio::time::timeout(Duration::from_secs(1), request_handle)
        .await
        .expect("image request should observe abort")
        .expect("join image request")
        .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
    assert_eq!(output.error_message.as_deref(), Some("Request was aborted"));
    assert!(output.output.is_empty());
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
}

struct ReplaceImagesPayloadHook {
    replacement: Value,
    seen: Arc<Mutex<Vec<Value>>>,
}

impl ImagesPayloadHook for ReplaceImagesPayloadHook {
    fn on_payload(&self, _model: &ImagesModel, payload: Value) -> Result<Value, String> {
        self.seen.lock().expect("seen payloads").push(payload);
        Ok(self.replacement.clone())
    }
}

struct ErrorImagesPayloadHook {
    message: &'static str,
}

impl ImagesPayloadHook for ErrorImagesPayloadHook {
    fn on_payload(&self, _model: &ImagesModel, _payload: Value) -> Result<Value, String> {
        Err(self.message.to_owned())
    }
}

struct RecordingProviderResponseHook {
    seen: Arc<Mutex<Vec<ProviderResponse>>>,
}

impl ProviderResponseHook for RecordingProviderResponseHook {
    fn on_response(
        &self,
        _model: Model,
        response: ProviderResponse,
    ) -> BoxFuture<'static, Result<(), String>> {
        let seen = self.seen.clone();
        Box::pin(async move {
            seen.lock().expect("seen responses").push(response);
            Ok(())
        })
    }
}

struct RecordingImagesResponseHook {
    seen: Arc<Mutex<Vec<ProviderResponse>>>,
}

impl ImagesResponseHook for RecordingImagesResponseHook {
    fn on_response(
        &self,
        _model: ImagesModel,
        response: ProviderResponse,
    ) -> BoxFuture<'static, Result<(), String>> {
        let seen = self.seen.clone();
        Box::pin(async move {
            seen.lock().expect("seen responses").push(response);
            Ok(())
        })
    }
}

struct ErrorImagesResponseHook {
    message: &'static str,
}

impl ImagesResponseHook for ErrorImagesResponseHook {
    fn on_response(
        &self,
        _model: ImagesModel,
        _response: ProviderResponse,
    ) -> BoxFuture<'static, Result<(), String>> {
        let message = self.message.to_owned();
        Box::pin(async move { Err(message) })
    }
}

struct DelayedImagesResponseHook {
    seen: Arc<Mutex<Vec<String>>>,
}

impl ImagesResponseHook for DelayedImagesResponseHook {
    fn on_response(
        &self,
        model: ImagesModel,
        response: ProviderResponse,
    ) -> BoxFuture<'static, Result<(), String>> {
        let seen = self.seen.clone();
        Box::pin(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            seen.lock()
                .expect("seen delayed responses")
                .push(format!("{}:{}", model.id, response.status));
            Ok(())
        })
    }
}

#[tokio::test]
async fn builtin_openrouter_images_provider_applies_payload_and_response_hooks() {
    let body = concat!(
        "{\"id\":\"img-hooks\",\"choices\":[{\"message\":{\"content\":\"Hooked\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,aG9vaw==\"}]}}]}"
    );
    let (base_url, request_task) = mock_json_server(body).await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    let seen_payloads = Arc::new(Mutex::new(Vec::<Value>::new()));
    let seen_responses = Arc::new(Mutex::new(Vec::<ProviderResponse>::new()));
    let replacement = json!({
        "model": model.id,
        "messages": [
            {
                "role": "user",
                "content": [{ "type": "text", "text": "Hooked prompt" }]
            }
        ],
        "stream": false,
        "modalities": ["image", "text"]
    });

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Original prompt")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            timeout_ms: Some(5_000),
            payload_hooks: vec![Arc::new(ReplaceImagesPayloadHook {
                replacement,
                seen: seen_payloads.clone(),
            })],
            response_hooks: vec![Arc::new(RecordingImagesResponseHook {
                seen: seen_responses.clone(),
            })],
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-hooks"));
    assert!(request.contains("Hooked prompt"));
    assert!(!request.contains("Original prompt"));
    let seen_payloads = seen_payloads.lock().expect("seen payloads");
    assert_eq!(seen_payloads.len(), 1);
    assert_eq!(
        seen_payloads[0]["messages"][0]["content"][0]["text"],
        "Original prompt"
    );
    let seen_responses = seen_responses.lock().expect("seen responses");
    assert_eq!(seen_responses.len(), 1);
    assert_eq!(seen_responses[0].status, 200);
    assert!(
        seen_responses[0]
            .headers
            .get("content-type")
            .is_some_and(|value| value.contains("application/json"))
    );
}

#[tokio::test]
async fn builtin_openrouter_images_provider_maps_payload_hook_errors_before_request() {
    let model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate hook error image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            payload_hooks: vec![Arc::new(ErrorImagesPayloadHook {
                message: "payload hook exploded",
            })],
            ..Default::default()
        },
    )
    .await
    .expect("generate images");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    assert!(output.output.is_empty());
    assert_eq!(
        output.error_message.as_deref(),
        Some("payload hook exploded")
    );
}

#[tokio::test]
async fn builtin_openrouter_images_provider_maps_response_hook_errors_after_response() {
    let body = concat!(
        "{\"id\":\"img-response-hook\",\"choices\":[{\"message\":{\"content\":\"Hooked\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,aG9vaw==\"}]}}]}"
    );
    let (base_url, request_task) = mock_json_server(body).await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate response hook error image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            response_hooks: vec![Arc::new(ErrorImagesResponseHook {
                message: "response hook exploded",
            })],
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    assert!(output.output.is_empty());
    assert_eq!(
        output.error_message.as_deref(),
        Some("response hook exploded")
    );
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_maps_nonretryable_http_errors() {
    let (base_url, request_task) = mock_json_status_server(
        401,
        "Unauthorized",
        r#"{"error":{"message":"Invalid OpenRouter key"}}"#,
    )
    .await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate unauthorized image")],
        },
        ImagesOptions {
            api_key: Some("bad-key".to_owned()),
            max_retries: Some(2),
            max_retry_delay_ms: Some(0),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    assert!(output.output.is_empty());
    assert_eq!(
        output.error_message.as_deref(),
        Some("Invalid OpenRouter key")
    );
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_waits_for_async_response_hooks() {
    let body = concat!(
        "{\"id\":\"img-async-hook\",\"choices\":[{\"message\":{\"content\":\"Async hook\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,YXN5bmM=\"}]}}]}"
    );
    let (base_url, request_task) = mock_json_server(body).await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;
    let seen = Arc::new(Mutex::new(Vec::new()));

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate async hook image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            response_hooks: vec![Arc::new(DelayedImagesResponseHook { seen: seen.clone() })],
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-async-hook"));
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert_eq!(
        *seen.lock().expect("seen delayed responses"),
        vec![format!("{}:200", model.id)]
    );
}

#[tokio::test]
async fn builtin_openrouter_images_provider_maps_invalid_json_response() {
    let (base_url, request_task) = mock_json_server("{not json").await;
    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = base_url;

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate invalid json image")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let request = request_task.await.expect("request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Error);
    assert!(output.output.is_empty());
    assert!(
        output
            .error_message
            .as_deref()
            .is_some_and(|message| message.starts_with("Could not parse OpenRouter image response")),
        "{output:?}"
    );
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
}

#[tokio::test]
async fn builtin_openrouter_images_provider_respects_pre_aborted_flag() {
    let model = get_image_model("openrouter", "black-forest-labs/flux.2-pro").expect("model");
    let abort_flag = Arc::new(AtomicBool::new(true));

    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate a dog")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            abort_flag: Some(abort_flag),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");

    assert_eq!(output.stop_reason, ImagesStopReason::Aborted);
    assert_eq!(output.error_message.as_deref(), Some("Request was aborted"));
    assert!(output.output.is_empty());
}

#[derive(Clone)]
struct RecordingImagesProvider {
    api: String,
    seen: Arc<Mutex<Vec<(String, Vec<String>, ImagesOptions)>>>,
}

#[async_trait]
impl ImagesApiProvider for RecordingImagesProvider {
    fn api(&self) -> &str {
        &self.api
    }

    async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Result<AssistantImages, ProviderError> {
        let input_texts = context
            .input
            .iter()
            .filter_map(|content| match content {
                ImagesContent::Text(text) => Some(text.text.clone()),
                ImagesContent::Image(_) => None,
            })
            .collect::<Vec<_>>();
        self.seen
            .lock()
            .expect("seen")
            .push((model.id.clone(), input_texts, options));
        Ok(AssistantImages {
            api: model.api.clone(),
            provider: model.provider.clone(),
            model: model.id.clone(),
            output: vec![
                ImagesContent::text("Here is your image."),
                ImagesContent::Image(ImageContent {
                    data: "ZmFrZS1wbmc=".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ],
            response_id: Some("img-1".to_owned()),
            usage: None,
            stop_reason: ImagesStopReason::Stop,
            error_message: None,
            timestamp: now_millis(),
        })
    }
}

#[tokio::test]
async fn images_api_registry_dispatches_generate_images_and_reports_missing_provider() {
    let api = format!("record-images-{}", now_millis());
    let source_id = format!("source-{api}");
    let seen = Arc::new(Mutex::new(Vec::new()));
    register_images_api_provider(
        Arc::new(RecordingImagesProvider {
            api: api.clone(),
            seen: seen.clone(),
        }),
        Some(source_id.clone()),
    );

    let model = ImagesModel {
        id: "fake-image-model".to_owned(),
        name: "Fake Image Model".to_owned(),
        api: api.clone(),
        provider: "fake-provider".to_owned(),
        base_url: "https://example.invalid/v1".to_owned(),
        input: vec![InputKind::Text, InputKind::Image],
        output: vec![OutputKind::Text, OutputKind::Image],
        cost: ModelCost::default(),
        headers: BTreeMap::new(),
    };
    let abort_flag = Arc::new(AtomicBool::new(false));
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate a dog")],
        },
        ImagesOptions {
            api_key: Some("test-api-key".to_owned()),
            headers: BTreeMap::from([("X-Test".to_owned(), "yes".to_owned())]),
            abort_flag: Some(abort_flag),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-1"));
    assert!(matches!(
        &output.output[0],
        ImagesContent::Text(text) if text.text == "Here is your image."
    ));
    assert!(matches!(
        &output.output[1],
        ImagesContent::Image(image)
            if image.mime_type == "image/png" && image.data == "ZmFrZS1wbmc="
    ));
    let seen_guard = seen.lock().expect("seen");
    assert_eq!(seen_guard.len(), 1);
    assert_eq!(seen_guard[0].0, "fake-image-model");
    assert_eq!(seen_guard[0].1, vec!["Generate a dog"]);
    assert_eq!(seen_guard[0].2.api_key.as_deref(), Some("test-api-key"));
    assert_eq!(
        seen_guard[0].2.headers.get("X-Test").map(String::as_str),
        Some("yes")
    );
    assert!(seen_guard[0].2.abort_flag.is_some());
    drop(seen_guard);

    let mut missing = model.clone();
    missing.api = format!("missing-{api}");
    let err = generate_images(&missing, ImagesContext::default(), ImagesOptions::default())
        .await
        .expect_err("missing provider");
    assert!(matches!(err, ProviderError::MissingApi(api) if api == missing.api));

    let direct_provider = get_images_api_provider(&api).expect("registered image provider");
    let mut mismatched = model.clone();
    mismatched.api = format!("other-{api}");
    let err = match direct_provider
        .generate_images(
            &mismatched,
            ImagesContext::default(),
            ImagesOptions::default(),
        )
        .await
    {
        Ok(_) => panic!("mismatched image model api should fail"),
        Err(err) => err,
    };
    assert!(matches!(
        err,
        ProviderError::MismatchedApi { actual, expected }
            if actual == mismatched.api && expected == api
    ));
    assert_eq!(seen.lock().expect("seen").len(), 1);

    unregister_images_api_providers(&source_id);
    assert!(get_images_api_provider(&api).is_none());
}

#[test]
fn images_api_provider_reset_restores_builtins_after_clear_and_removes_custom_providers() {
    let api = format!("reset-images-{}", now_millis());
    register_images_api_provider(
        Arc::new(RecordingImagesProvider {
            api: api.clone(),
            seen: Arc::new(Mutex::new(Vec::new())),
        }),
        Some(format!("source-{api}")),
    );
    assert!(get_images_api_provider(&api).is_some());

    clear_images_api_providers();
    ensure_builtin_images_api_providers();
    assert!(get_images_api_provider("openrouter-images").is_some());
    assert!(get_images_api_provider(&api).is_none());

    register_images_api_provider(
        Arc::new(RecordingImagesProvider {
            api: api.clone(),
            seen: Arc::new(Mutex::new(Vec::new())),
        }),
        Some(format!("source-{api}")),
    );
    reset_images_api_providers();
    assert!(get_images_api_provider("openrouter-images").is_some());
    assert!(get_images_api_provider(&api).is_none());
}

#[test]
fn anthropic_oauth_authorize_url_uses_localhost_callback() {
    let url = build_anthropic_authorize_url("challenge-value", "state-value");

    assert!(url.starts_with("https://claude.ai/oauth/authorize?"));
    assert!(url.contains("client_id=9d1c250a-e61b-44d9-88ed-5944d1962f5e"));
    assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A53692%2Fcallback"));
    assert!(url.contains("scope=org%3Acreate_api_key+user%3Aprofile"));
    assert!(url.contains("code_challenge=challenge-value"));
    assert!(url.contains("state=state-value"));

    let parsed = parse_anthropic_authorization_input(
        "http://localhost:53692/callback?code=manual-code&state=state-value",
    );
    assert_eq!(parsed.code.as_deref(), Some("manual-code"));
    assert_eq!(parsed.state.as_deref(), Some("state-value"));
}

#[test]
fn oauth_pkce_generation_matches_source_shape_and_challenge_derivation() {
    let pkce = generate_pkce().expect("pkce");
    assert_eq!(pkce.verifier.len(), 43);
    assert_eq!(pkce.challenge.len(), 43);
    assert!(pkce.verifier.chars().all(is_base64url_no_pad_char));
    assert!(pkce.challenge.chars().all(is_base64url_no_pad_char));

    let expected_challenge = URL_SAFE_NO_PAD.encode(ring::digest::digest(
        &ring::digest::SHA256,
        pkce.verifier.as_bytes(),
    ));
    assert_eq!(pkce.challenge, expected_challenge);
}

fn is_base64url_no_pad_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'
}

#[test]
fn anthropic_oauth_authorization_code_request_keeps_localhost_redirect_uri() {
    let request = build_anthropic_authorization_code_token_request(
        "manual-code",
        "state-value",
        "verifier-value",
        ANTHROPIC_OAUTH_REDIRECT_URI,
    );
    let body = request.json_body().expect("json body");

    assert_eq!(request.url, "https://platform.claude.com/v1/oauth/token");
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.headers.get("Content-Type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(body["grant_type"], "authorization_code");
    assert_eq!(body["client_id"], ANTHROPIC_OAUTH_CLIENT_ID);
    assert_eq!(body["code"], "manual-code");
    assert_eq!(body["state"], "state-value");
    assert_eq!(body["redirect_uri"], "http://localhost:53692/callback");
    assert_eq!(body["code_verifier"], "verifier-value");
}

#[tokio::test]
async fn anthropic_oauth_callback_server_accepts_matching_state() {
    let server = start_oauth_callback_server(anthropic_oauth_callback_server_options_with_port(
        "state-value",
        0,
    ))
    .await
    .expect("callback server");
    let port = server.local_addr.port();

    let not_found = oauth_callback_get(port, "/wrong").await;
    assert!(not_found.starts_with("HTTP/1.1 404 Not Found"));
    assert!(
        server
            .redirect_uri
            .ends_with(&format!(":{port}{ANTHROPIC_OAUTH_CALLBACK_PATH}"))
    );

    let response = oauth_callback_get(port, "/callback?code=manual-code&state=state-value").await;
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains("Anthropic authentication completed"));
    let callback = server
        .wait_for_code()
        .await
        .expect("wait for code")
        .expect("callback");
    assert_eq!(callback.code, "manual-code");
    assert_eq!(callback.state, "state-value");
}

#[tokio::test]
async fn anthropic_oauth_callback_server_rejects_state_mismatch_then_accepts_code() {
    let server = start_oauth_callback_server(anthropic_oauth_callback_server_options_with_port(
        "state-value",
        0,
    ))
    .await
    .expect("callback server");
    let port = server.local_addr.port();

    let mismatch = oauth_callback_get(port, "/callback?code=manual-code&state=wrong-state").await;
    assert!(mismatch.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(mismatch.contains("<title>Authentication failed</title>"));
    assert!(mismatch.contains("<h1>Authentication failed</h1>"));
    assert!(mismatch.contains("State mismatch"));

    let provider_error = oauth_callback_get(
        port,
        "/callback?error=%3Cscript%3Ebad%26quote%3D%22yes%22%3C%2Fscript%3E",
    )
    .await;
    assert!(provider_error.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(provider_error.contains("Authentication did not complete."));
    assert!(
        provider_error
            .contains("Error: &lt;script&gt;bad&amp;quote=&quot;yes&quot;&lt;/script&gt;")
    );
    assert!(!provider_error.contains("<script>"));

    let response = oauth_callback_get(port, "/callback?code=manual-code&state=state-value").await;
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains("<title>Authentication successful</title>"));
    assert!(response.contains("<h1>Authentication successful</h1>"));
    assert!(response.contains("Anthropic authentication completed"));
    let callback = server
        .wait_for_code()
        .await
        .expect("wait for code")
        .expect("callback");
    assert_eq!(callback.code, "manual-code");
    assert_eq!(callback.state, "state-value");
}

#[tokio::test]
async fn anthropic_oauth_login_flow_manual_input_exchanges_code() {
    let (token_url, request_task) = mock_json_server(
        r#"{"access_token":"manual-access","refresh_token":"manual-refresh","expires_in":3600}"#,
    )
    .await;
    let flow = start_anthropic_oauth_login_flow_with_pkce("verifier-value", "challenge-value", 0)
        .await
        .expect("login flow");
    let port = flow.local_addr.port();

    assert!(
        flow.auth_url
            .starts_with("https://claude.ai/oauth/authorize?")
    );
    assert!(flow.auth_url.contains("code_challenge=challenge-value"));
    assert!(flow.auth_url.contains("state=verifier-value"));
    assert!(flow.auth_url.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fcallback"
    )));
    assert!(
        flow.instructions
            .as_deref()
            .unwrap_or_default()
            .contains("browser")
    );

    let credentials = finish_anthropic_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/callback?code=manual-code&state=verifier-value",
        &token_url,
        1_000_000,
    )
    .await
    .expect("credentials");
    let request = request_task.await.expect("token request");

    assert_eq!(credentials.access, "manual-access");
    assert_eq!(credentials.refresh, "manual-refresh");
    assert_eq!(credentials.expires, 1_000_000 + 3600 * 1000 - 5 * 60 * 1000);
    assert!(request.contains("\"grant_type\":\"authorization_code\""));
    assert!(request.contains("\"code\":\"manual-code\""));
    assert!(request.contains("\"state\":\"verifier-value\""));
    assert!(request.contains("\"code_verifier\":\"verifier-value\""));
    assert!(request.contains(&format!(
        "\"redirect_uri\":\"http://localhost:{port}/callback\""
    )));
}

#[tokio::test]
async fn anthropic_oauth_login_flow_manual_input_rejects_state_mismatch() {
    let flow = start_anthropic_oauth_login_flow_with_pkce("verifier-value", "challenge-value", 0)
        .await
        .expect("login flow");

    let error = finish_anthropic_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/callback?code=manual-code&state=wrong-state",
        "http://127.0.0.1:9/token",
        1_000_000,
    )
    .await
    .expect_err("state mismatch should fail before token exchange");

    assert_eq!(error, "OAuth state mismatch");
}

#[tokio::test(flavor = "current_thread")]
async fn anthropic_oauth_login_flow_manual_input_routes_token_exchange_through_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let (proxy_url, proxy_request_task) = mock_json_server(
        r#"{"access_token":"proxy-manual-access","refresh_token":"proxy-manual-refresh","expires_in":3600}"#,
    )
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let flow = start_anthropic_oauth_login_flow_with_pkce("verifier-value", "challenge-value", 0)
        .await
        .expect("login flow");
    let port = flow.local_addr.port();
    let credentials = finish_anthropic_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/callback?code=manual-code&state=verifier-value",
        "http://oauth.example/token",
        1_000_000,
    )
    .await
    .expect("credentials through proxy");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(credentials.access, "proxy-manual-access");
    assert_eq!(credentials.refresh, "proxy-manual-refresh");
    assert_eq!(credentials.expires, 1_000_000 + 3600 * 1000 - 5 * 60 * 1000);
    assert!(
        proxy_request.starts_with("POST http://oauth.example/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(proxy_request.contains("\"grant_type\":\"authorization_code\""));
    assert!(proxy_request.contains("\"code\":\"manual-code\""));
    assert!(proxy_request.contains("\"state\":\"verifier-value\""));
    assert!(proxy_request.contains("\"code_verifier\":\"verifier-value\""));
    assert!(proxy_request.contains(&format!(
        "\"redirect_uri\":\"http://localhost:{port}/callback\""
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn anthropic_oauth_login_flow_callback_routes_token_exchange_through_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let (proxy_url, proxy_request_task) = mock_json_server(
        r#"{"access_token":"proxy-callback-access","refresh_token":"proxy-callback-refresh","expires_in":3600}"#,
    )
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let flow = start_anthropic_oauth_login_flow_with_pkce("verifier-value", "challenge-value", 0)
        .await
        .expect("login flow");
    let port = flow.local_addr.port();
    let target = "/callback?code=callback-code&state=verifier-value".to_owned();
    let callback_task = tokio::spawn(async move { oauth_callback_get(port, &target).await });
    let credentials = finish_anthropic_oauth_login_from_callback_at(
        flow,
        "http://oauth.example/token",
        1_000_000,
    )
    .await
    .expect("credentials through proxy");
    let callback_response = callback_task.await.expect("callback task");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert!(callback_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(credentials.access, "proxy-callback-access");
    assert_eq!(credentials.refresh, "proxy-callback-refresh");
    assert_eq!(credentials.expires, 1_000_000 + 3600 * 1000 - 5 * 60 * 1000);
    assert!(
        proxy_request.starts_with("POST http://oauth.example/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(proxy_request.contains("\"grant_type\":\"authorization_code\""));
    assert!(proxy_request.contains("\"code\":\"callback-code\""));
    assert!(proxy_request.contains("\"state\":\"verifier-value\""));
    assert!(proxy_request.contains("\"code_verifier\":\"verifier-value\""));
    assert!(proxy_request.contains(&format!(
        "\"redirect_uri\":\"http://localhost:{port}/callback\""
    )));
}

#[test]
fn anthropic_oauth_refresh_request_omits_scope() {
    let request = build_anthropic_refresh_token_request("refresh-token");
    let body = request.json_body().expect("json body");

    assert_eq!(request.url, "https://platform.claude.com/v1/oauth/token");
    assert_eq!(request.method, "POST");
    assert_eq!(body["grant_type"], "refresh_token");
    assert_eq!(body["client_id"], ANTHROPIC_OAUTH_CLIENT_ID);
    assert_eq!(body["refresh_token"], "refresh-token");
    assert!(body.get("scope").is_none());
}

#[test]
fn anthropic_oauth_token_response_maps_credentials_and_expiry() {
    let credentials = parse_anthropic_oauth_token_response(
        r#"{
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "scope": "ignored"
        }"#,
        1_000_000,
    )
    .expect("credentials");

    assert_eq!(credentials.access, "new-access-token");
    assert_eq!(credentials.refresh, "new-refresh-token");
    assert_eq!(credentials.expires, 1_000_000 + 3600 * 1000 - 5 * 60 * 1000);

    let err = parse_anthropic_oauth_token_response("not json", 0).expect_err("invalid json");
    assert!(err.contains("invalid JSON"));
}

#[tokio::test]
async fn anthropic_oauth_refresh_token_posts_json_and_parses_credentials() {
    let (token_url, request_task) = mock_json_server(
        r#"{"access_token":"live-access","refresh_token":"live-refresh","expires_in":3600}"#,
    )
    .await;

    let credentials = refresh_anthropic_token_with_url_at("refresh-token", &token_url, 1_000_000)
        .await
        .expect("refresh token");
    let request = request_task.await.expect("request task");

    assert_eq!(credentials.access, "live-access");
    assert_eq!(credentials.refresh, "live-refresh");
    assert_eq!(credentials.expires, 1_000_000 + 3600 * 1000 - 5 * 60 * 1000);
    assert!(request.starts_with("POST / HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("content-type: application/json")
    );
    assert!(request.contains("\"grant_type\":\"refresh_token\""));
    assert!(request.contains("\"refresh_token\":\"refresh-token\""));
    assert!(!request.contains("scope"));
}

#[tokio::test(flavor = "current_thread")]
async fn anthropic_oauth_refresh_routes_token_request_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let (proxy_url, proxy_request_task) = mock_json_server(
        r#"{"access_token":"proxy-access","refresh_token":"proxy-refresh","expires_in":3600}"#,
    )
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let credentials = refresh_anthropic_token_with_url_at(
        "refresh-token",
        "http://oauth.example/token",
        1_000_000,
    )
    .await
    .expect("refresh token through proxy");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(credentials.access, "proxy-access");
    assert_eq!(credentials.refresh, "proxy-refresh");
    assert_eq!(credentials.expires, 1_000_000 + 3600 * 1000 - 5 * 60 * 1000);
    assert!(
        proxy_request.starts_with("POST http://oauth.example/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("content-type: application/json")
    );
    assert!(proxy_request.contains("\"grant_type\":\"refresh_token\""));
    assert!(proxy_request.contains("\"refresh_token\":\"refresh-token\""));
}

#[test]
fn openai_codex_oauth_authorize_url_matches_cli_flow_parameters() {
    let url = build_openai_codex_authorize_url("challenge-value", "state-value", None);

    assert!(url.starts_with("https://auth.openai.com/oauth/authorize?"));
    assert!(url.contains("response_type=code"));
    assert!(url.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
    assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback"));
    assert!(url.contains("scope=openid+profile+email+offline_access"));
    assert!(url.contains("code_challenge=challenge-value"));
    assert!(url.contains("id_token_add_organizations=true"));
    assert!(url.contains("codex_cli_simplified_flow=true"));
    assert!(url.contains("originator=pi"));
}

#[tokio::test]
async fn openai_codex_oauth_callback_server_rejects_state_mismatch_then_accepts_code() {
    let server = start_oauth_callback_server(openai_codex_oauth_callback_server_options_with_port(
        "state-value",
        0,
    ))
    .await
    .expect("callback server");
    let port = server.local_addr.port();

    let mismatch =
        oauth_callback_get(port, "/auth/callback?code=manual-code&state=wrong-state").await;
    assert!(mismatch.starts_with("HTTP/1.1 400 Bad Request"));
    assert!(mismatch.contains("State mismatch"));
    assert!(
        server
            .redirect_uri
            .ends_with(&format!(":{port}{OPENAI_CODEX_OAUTH_CALLBACK_PATH}"))
    );

    let response =
        oauth_callback_get(port, "/auth/callback?code=manual-code&state=state-value").await;
    assert!(response.starts_with("HTTP/1.1 200 OK"));
    assert!(response.contains("OpenAI authentication completed"));
    let callback = server
        .wait_for_code()
        .await
        .expect("wait for code")
        .expect("callback");
    assert_eq!(callback.code, "manual-code");
    assert_eq!(callback.state, "state-value");
}

#[tokio::test]
async fn openai_codex_oauth_login_flow_callback_exchanges_code() {
    let codex_access = codex_test_token();
    let (token_url, request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "codex-refresh",
        7200,
    ))
    .await;
    let flow = start_openai_codex_oauth_login_flow_with_pkce(
        "verifier-value",
        "challenge-value",
        "state-value",
        Some("codex-test"),
        0,
    )
    .await
    .expect("login flow");
    let port = flow.local_addr.port();

    assert!(
        flow.auth_url
            .starts_with("https://auth.openai.com/oauth/authorize?")
    );
    assert!(flow.auth_url.contains("code_challenge=challenge-value"));
    assert!(flow.auth_url.contains("state=state-value"));
    assert!(flow.auth_url.contains("originator=codex-test"));
    assert!(flow.auth_url.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fauth%2Fcallback"
    )));

    let target = "/auth/callback?code=callback-code&state=state-value".to_owned();
    let callback_task = tokio::spawn(async move { oauth_callback_get(port, &target).await });
    let credentials = finish_openai_codex_oauth_login_from_callback_at(flow, &token_url, 2_000_000)
        .await
        .expect("credentials");
    let callback_response = callback_task.await.expect("callback task");
    let request = request_task.await.expect("token request");

    assert!(callback_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(credentials.access, codex_access);
    assert_eq!(credentials.refresh, "codex-refresh");
    assert_eq!(credentials.expires, 2_000_000 + 7200 * 1000 - 5 * 60 * 1000);
    assert_eq!(credentials.extra.get("accountId"), Some(&json!("acc_test")));
    assert!(request.contains("grant_type=authorization_code"));
    assert!(request.contains("code=callback-code"));
    assert!(request.contains("code_verifier=verifier-value"));
    assert!(request.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fauth%2Fcallback"
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn openai_codex_oauth_login_flow_callback_routes_token_exchange_through_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let codex_access = codex_test_token();
    let (proxy_url, proxy_request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "proxy-codex-refresh",
        7200,
    ))
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let flow = start_openai_codex_oauth_login_flow_with_pkce(
        "verifier-value",
        "challenge-value",
        "state-value",
        Some("codex-test"),
        0,
    )
    .await
    .expect("login flow");
    let port = flow.local_addr.port();
    let target = "/auth/callback?code=callback-code&state=state-value".to_owned();
    let callback_task = tokio::spawn(async move { oauth_callback_get(port, &target).await });
    let credentials = finish_openai_codex_oauth_login_from_callback_at(
        flow,
        "http://auth.example/oauth/token",
        2_000_000,
    )
    .await
    .expect("credentials through proxy");
    let callback_response = callback_task.await.expect("callback task");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert!(callback_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(credentials.access, codex_access);
    assert_eq!(credentials.refresh, "proxy-codex-refresh");
    assert_eq!(credentials.expires, 2_000_000 + 7200 * 1000 - 5 * 60 * 1000);
    assert_eq!(credentials.extra.get("accountId"), Some(&json!("acc_test")));
    assert!(
        proxy_request.starts_with("POST http://auth.example/oauth/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(proxy_request.contains("grant_type=authorization_code"));
    assert!(proxy_request.contains("code=callback-code"));
    assert!(proxy_request.contains("code_verifier=verifier-value"));
    assert!(proxy_request.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fauth%2Fcallback"
    )));
}

#[tokio::test(flavor = "current_thread")]
async fn openai_codex_oauth_login_flow_manual_input_routes_token_exchange_through_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let codex_access = codex_test_token();
    let (proxy_url, proxy_request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "proxy-codex-manual-refresh",
        7200,
    ))
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let flow = start_openai_codex_oauth_login_flow_with_pkce(
        "verifier-value",
        "challenge-value",
        "state-value",
        Some("codex-test"),
        0,
    )
    .await
    .expect("login flow");
    let port = flow.local_addr.port();
    let credentials = finish_openai_codex_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/auth/callback?code=manual-code&state=state-value",
        "http://auth.example/oauth/token",
        2_000_000,
    )
    .await
    .expect("credentials through proxy");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(credentials.access, codex_access);
    assert_eq!(credentials.refresh, "proxy-codex-manual-refresh");
    assert_eq!(credentials.expires, 2_000_000 + 7200 * 1000 - 5 * 60 * 1000);
    assert_eq!(credentials.extra.get("accountId"), Some(&json!("acc_test")));
    assert!(
        proxy_request.starts_with("POST http://auth.example/oauth/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(proxy_request.contains("grant_type=authorization_code"));
    assert!(proxy_request.contains("code=manual-code"));
    assert!(proxy_request.contains("code_verifier=verifier-value"));
    assert!(proxy_request.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fauth%2Fcallback"
    )));
}

#[tokio::test]
async fn openai_codex_oauth_login_flow_manual_input_rejects_state_mismatch() {
    let flow = start_openai_codex_oauth_login_flow_with_pkce(
        "verifier-value",
        "challenge-value",
        "state-value",
        Some("codex-test"),
        0,
    )
    .await
    .expect("login flow");

    let error = finish_openai_codex_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/auth/callback?code=manual-code&state=wrong-state",
        "http://127.0.0.1:9/oauth/token",
        2_000_000,
    )
    .await
    .expect_err("state mismatch should fail before token exchange");

    assert_eq!(error, "State mismatch");
}

#[test]
fn openai_codex_oauth_refresh_request_uses_form_encoded_body() {
    let request = build_openai_codex_refresh_token_request("invalid refresh/token");

    assert_eq!(request.url, "https://auth.openai.com/oauth/token");
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.headers.get("Content-Type").map(String::as_str),
        Some("application/x-www-form-urlencoded")
    );
    assert!(request.body.contains("grant_type=refresh_token"));
    assert!(
        request
            .body
            .contains("refresh_token=invalid+refresh%2Ftoken")
    );
    assert!(
        request
            .body
            .contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann")
    );
}

#[test]
fn openai_codex_oauth_refresh_failure_message_includes_status_and_body() {
    let message = openai_codex_token_failure_message(
        "refresh",
        401,
        "Unauthorized",
        Some(
            r#"{"error":{"message":"Could not validate your token. Please try signing in again.","type":"invalid_request_error"}}"#,
        ),
    );

    assert!(message.starts_with("OpenAI Codex token refresh failed (401): "));
    assert!(message.contains("Could not validate your token"));
    assert!(!message.contains("stderr"));
}

#[test]
fn openai_codex_oauth_token_response_requires_account_id_and_preserves_extra() {
    let codex_access = codex_test_token();
    let credentials = parse_openai_codex_oauth_token_response(
        &codex_oauth_token_response(&codex_access, "codex-refresh", 7200),
        2_000_000,
    )
    .expect("token response with account id");

    assert_eq!(credentials.access, codex_access);
    assert_eq!(credentials.refresh, "codex-refresh");
    assert_eq!(credentials.extra.get("accountId"), Some(&json!("acc_test")));

    let error = parse_openai_codex_oauth_token_response(
        r#"{"access_token":"not-a-jwt","refresh_token":"codex-refresh","expires_in":7200}"#,
        2_000_000,
    )
    .expect_err("missing account id should fail like source OAuth login/refresh");
    assert_eq!(error, "Failed to extract accountId from token");
}

struct FakeOAuthRefresher {
    result: StoredOAuthCredentials,
    calls: Mutex<Vec<(String, StoredOAuthCredentials, i64)>>,
}

#[async_trait]
impl OAuthTokenRefresher for FakeOAuthRefresher {
    async fn refresh_token(
        &self,
        provider_id: &str,
        credentials: &StoredOAuthCredentials,
        now_millis: i64,
    ) -> Result<StoredOAuthCredentials, String> {
        self.calls.lock().expect("calls lock").push((
            provider_id.to_owned(),
            credentials.clone(),
            now_millis,
        ));
        Ok(self.result.clone())
    }
}

struct FailingOAuthRefresher {
    message: &'static str,
    calls: Mutex<Vec<(String, StoredOAuthCredentials, i64)>>,
}

#[async_trait]
impl OAuthTokenRefresher for FailingOAuthRefresher {
    async fn refresh_token(
        &self,
        provider_id: &str,
        credentials: &StoredOAuthCredentials,
        now_millis: i64,
    ) -> Result<StoredOAuthCredentials, String> {
        self.calls.lock().expect("calls lock").push((
            provider_id.to_owned(),
            credentials.clone(),
            now_millis,
        ));
        Err(self.message.to_owned())
    }
}

fn auth_storage_test_path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir()
        .join(format!(
            "ri-auth-storage-{label}-{}-{}",
            std::process::id(),
            now_millis()
        ))
        .join("auth.json")
}

#[test]
fn oauth_provider_registry_matches_built_in_source_metadata() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    reset_oauth_providers();

    let providers = get_oauth_providers();
    let provider_ids = providers
        .iter()
        .map(|provider| provider.id.as_str())
        .collect::<Vec<_>>();

    assert_eq!(
        provider_ids,
        vec!["anthropic", "github-copilot", "openai-codex"]
    );
    assert_eq!(
        get_oauth_provider("anthropic"),
        Some(OAuthProviderInfo {
            id: "anthropic".to_owned(),
            name: "Anthropic (Claude Pro/Max)".to_owned(),
            uses_callback_server: true,
        })
    );
    assert_eq!(
        get_oauth_provider("github-copilot"),
        Some(OAuthProviderInfo {
            id: "github-copilot".to_owned(),
            name: "GitHub Copilot".to_owned(),
            uses_callback_server: false,
        })
    );
    assert_eq!(
        get_oauth_provider("openai-codex"),
        Some(OAuthProviderInfo {
            id: "openai-codex".to_owned(),
            name: "ChatGPT Plus/Pro (Codex Subscription)".to_owned(),
            uses_callback_server: true,
        })
    );
    assert_eq!(get_oauth_provider("missing-provider"), None);
    assert_eq!(get_oauth_provider_info_list(), providers);
}

#[test]
fn pi_ai_cli_help_list_and_provider_selection_match_source_surface() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    reset_oauth_providers();
    let providers = get_oauth_providers();

    assert_eq!(
        render_cli_provider_list(&providers),
        "Available OAuth providers:\n\n  anthropic            Anthropic (Claude Pro/Max)\n  github-copilot       GitHub Copilot\n  openai-codex         ChatGPT Plus/Pro (Codex Subscription)\n"
    );

    let help = render_cli_help("npx @earendil-works/pi-ai", &providers);
    assert!(help.contains("Usage: npx @earendil-works/pi-ai <command> [provider]"));
    assert!(help.contains("  login [provider]  Login to an OAuth provider"));
    assert!(help.contains("  list              List available providers"));
    assert!(help.contains("  npx @earendil-works/pi-ai login anthropic"));

    assert_eq!(
        parse_cli_provider_selection("1", &providers).expect("anthropic selection"),
        "anthropic"
    );
    assert_eq!(
        parse_cli_provider_selection(" 2\n", &providers).expect("github selection"),
        "github-copilot"
    );
    assert_eq!(
        parse_cli_provider_selection("3", &providers).expect("openai selection"),
        "openai-codex"
    );
    assert_eq!(
        parse_cli_provider_selection("4", &providers).expect_err("out of range"),
        "Invalid selection"
    );
    assert_eq!(
        parse_cli_provider_selection("abc", &providers).expect_err("not a number"),
        "Invalid selection"
    );
}

#[test]
fn pi_ai_cli_auth_save_preserves_existing_entries_and_source_json_shape() {
    let path = auth_storage_test_path("pi-ai-cli-save");
    let mut existing = AuthStorage::new();
    existing.insert(
        "openai".to_owned(),
        AuthCredential::ApiKey {
            key: "plain-key".to_owned(),
        },
    );
    save_auth_storage_to_path(&path, &existing).expect("seed auth");

    let credentials = StoredOAuthCredentials {
        refresh: "refresh-token".to_owned(),
        access: "access-token".to_owned(),
        expires: 123_456,
        extra: BTreeMap::from([("accountId".to_owned(), json!("acct_123"))]),
    };
    save_cli_oauth_credentials(&path, "openai-codex", credentials).expect("save cli auth");

    let raw = std::fs::read_to_string(&path).expect("read auth file");
    let saved_json: Value = serde_json::from_str(&raw).expect("auth json");
    assert_eq!(saved_json["openai"]["type"], "api_key");
    assert_eq!(saved_json["openai"]["key"], "plain-key");
    assert_eq!(saved_json["openai-codex"]["type"], "oauth");
    assert_eq!(saved_json["openai-codex"]["refresh"], "refresh-token");
    assert_eq!(saved_json["openai-codex"]["access"], "access-token");
    assert_eq!(saved_json["openai-codex"]["expires"], 123_456);
    assert_eq!(saved_json["openai-codex"]["accountId"], "acct_123");

    let loaded = load_cli_auth(&path).expect("load cli auth");
    assert!(matches!(
        loaded.get("openai"),
        Some(AuthCredential::ApiKey { key }) if key == "plain-key"
    ));
    assert!(matches!(
        loaded.get("openai-codex"),
        Some(AuthCredential::OAuth { credentials })
            if credentials.access == "access-token"
                && credentials.extra.get("accountId") == Some(&json!("acct_123"))
    ));
}

#[test]
fn oauth_provider_registry_registers_custom_and_restores_built_ins() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    reset_oauth_providers();
    register_oauth_provider(OAuthProviderInfo {
        id: "anthropic".to_owned(),
        name: "Custom Anthropic".to_owned(),
        uses_callback_server: false,
    });
    register_oauth_provider(OAuthProviderInfo {
        id: "custom-oauth".to_owned(),
        name: "Custom OAuth".to_owned(),
        uses_callback_server: false,
    });

    assert_eq!(
        get_oauth_provider("anthropic")
            .expect("custom anthropic")
            .name,
        "Custom Anthropic"
    );
    assert_eq!(
        get_oauth_provider("custom-oauth")
            .expect("custom provider")
            .name,
        "Custom OAuth"
    );

    unregister_oauth_provider("anthropic");
    unregister_oauth_provider("custom-oauth");

    assert_eq!(
        get_oauth_provider("anthropic")
            .expect("restored anthropic")
            .name,
        "Anthropic (Claude Pro/Max)"
    );
    assert_eq!(get_oauth_provider("custom-oauth"), None);
    reset_oauth_providers();
}

#[tokio::test]
async fn oauth_auth_storage_resolves_api_key_and_current_oauth_without_refresh() {
    let path = auth_storage_test_path("current");
    let parent = path.parent().expect("auth storage parent");
    std::fs::create_dir_all(parent).expect("create auth storage dir");
    std::fs::write(
        &path,
        json!({
            "openai": { "type": "api_key", "key": "openai-key" },
            "anthropic": {
                "type": "oauth",
                "refresh": "anthropic-refresh",
                "access": "anthropic-access",
                "expires": 2_000_000
            }
        })
        .to_string(),
    )
    .expect("write auth storage");
    let refresher = FakeOAuthRefresher {
        result: StoredOAuthCredentials {
            refresh: "unused-refresh".to_owned(),
            access: "unused-access".to_owned(),
            expires: 3_000_000,
            extra: BTreeMap::new(),
        },
        calls: Mutex::new(Vec::new()),
    };

    let api_key = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "openai", &path, 1_000_000, &refresher,
    )
    .await
    .expect("api key resolution")
    .expect("api key result");
    let oauth = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "anthropic",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");

    assert_eq!(api_key.api_key, "openai-key");
    assert_eq!(api_key.credentials, None);
    assert!(!api_key.refreshed);
    assert_eq!(oauth.api_key, "anthropic-access");
    assert_eq!(
        oauth
            .credentials
            .as_ref()
            .map(|credentials| credentials.refresh.as_str()),
        Some("anthropic-refresh")
    );
    assert!(!oauth.refreshed);
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn anthropic_oauth_callback_login_persists_auth_storage_and_resolves_access_key() {
    let (token_url, request_task) = mock_json_server(
        r#"{"access_token":"callback-access","refresh_token":"callback-refresh","expires_in":3600}"#,
    )
    .await;
    let path = auth_storage_test_path("anthropic-login-roundtrip");
    let parent = path.parent().expect("auth storage parent");
    let flow = start_anthropic_oauth_login_flow_with_pkce("verifier-value", "challenge-value", 0)
        .await
        .expect("login flow");
    let port = flow.local_addr.port();
    let target = "/callback?code=callback-code&state=verifier-value".to_owned();
    let callback_task = tokio::spawn(async move { oauth_callback_get(port, &target).await });

    let credentials = finish_anthropic_oauth_login_from_callback_at(flow, &token_url, 1_000_000)
        .await
        .expect("callback credentials");
    let callback_response = callback_task.await.expect("callback task");
    let request = request_task.await.expect("token request");
    let stored_credentials = StoredOAuthCredentials::from(credentials);
    let mut storage = AuthStorage::new();
    storage.insert(
        "anthropic".to_owned(),
        AuthCredential::OAuth {
            credentials: stored_credentials.clone(),
        },
    );
    save_auth_storage_to_path(&path, &storage).expect("save auth storage");
    let saved_raw: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read auth storage"))
            .expect("saved auth json");
    let refresher = FailingOAuthRefresher {
        message: "should not refresh",
        calls: Mutex::new(Vec::new()),
    };

    let resolution = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "anthropic",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");

    assert!(callback_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(stored_credentials.access, "callback-access");
    assert_eq!(stored_credentials.refresh, "callback-refresh");
    assert_eq!(
        stored_credentials.expires,
        1_000_000 + 3600 * 1000 - 5 * 60 * 1000
    );
    assert!(request.contains("\"grant_type\":\"authorization_code\""));
    assert!(request.contains("\"code\":\"callback-code\""));
    assert!(request.contains("\"state\":\"verifier-value\""));
    assert!(request.contains("\"code_verifier\":\"verifier-value\""));
    assert!(request.contains(&format!(
        "\"redirect_uri\":\"http://localhost:{port}/callback\""
    )));
    assert_eq!(saved_raw["anthropic"]["type"], json!("oauth"));
    assert_eq!(saved_raw["anthropic"]["access"], json!("callback-access"));
    assert_eq!(saved_raw["anthropic"]["refresh"], json!("callback-refresh"));
    assert_eq!(resolution.api_key, "callback-access");
    assert_eq!(resolution.credentials, Some(stored_credentials));
    assert!(!resolution.refreshed);
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn anthropic_oauth_manual_input_login_persists_auth_storage_and_resolves_access_key() {
    let (token_url, request_task) = mock_json_server(
        r#"{"access_token":"manual-storage-access","refresh_token":"manual-storage-refresh","expires_in":3600}"#,
    )
    .await;
    let path = auth_storage_test_path("anthropic-manual-login-roundtrip");
    let parent = path.parent().expect("auth storage parent");
    let flow = start_anthropic_oauth_login_flow_with_pkce("verifier-value", "challenge-value", 0)
        .await
        .expect("login flow");
    let port = flow.local_addr.port();

    let credentials = finish_anthropic_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/callback?code=manual-code&state=verifier-value",
        &token_url,
        1_000_000,
    )
    .await
    .expect("manual credentials");
    let request = request_task.await.expect("token request");
    let stored_credentials = StoredOAuthCredentials::from(credentials);
    let mut storage = AuthStorage::new();
    storage.insert(
        "anthropic".to_owned(),
        AuthCredential::OAuth {
            credentials: stored_credentials.clone(),
        },
    );
    save_auth_storage_to_path(&path, &storage).expect("save auth storage");
    let saved_raw: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read auth storage"))
            .expect("saved auth json");
    let refresher = FailingOAuthRefresher {
        message: "should not refresh",
        calls: Mutex::new(Vec::new()),
    };

    let resolution = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "anthropic",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");

    assert_eq!(stored_credentials.access, "manual-storage-access");
    assert_eq!(stored_credentials.refresh, "manual-storage-refresh");
    assert_eq!(
        stored_credentials.expires,
        1_000_000 + 3600 * 1000 - 5 * 60 * 1000
    );
    assert!(request.contains("\"grant_type\":\"authorization_code\""));
    assert!(request.contains("\"code\":\"manual-code\""));
    assert!(request.contains("\"state\":\"verifier-value\""));
    assert!(request.contains("\"code_verifier\":\"verifier-value\""));
    assert!(request.contains(&format!(
        "\"redirect_uri\":\"http://localhost:{port}/callback\""
    )));
    assert_eq!(saved_raw["anthropic"]["type"], json!("oauth"));
    assert_eq!(
        saved_raw["anthropic"]["access"],
        json!("manual-storage-access")
    );
    assert_eq!(
        saved_raw["anthropic"]["refresh"],
        json!("manual-storage-refresh")
    );
    assert_eq!(resolution.api_key, "manual-storage-access");
    assert_eq!(resolution.credentials, Some(stored_credentials));
    assert!(!resolution.refreshed);
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn openai_codex_oauth_callback_login_persists_auth_storage_and_resolves_access_key() {
    let codex_access = codex_test_token();
    let (token_url, request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "codex-callback-refresh",
        7200,
    ))
    .await;
    let path = auth_storage_test_path("openai-codex-login-roundtrip");
    let parent = path.parent().expect("auth storage parent");
    let flow = start_openai_codex_oauth_login_flow_with_pkce(
        "verifier-value",
        "challenge-value",
        "state-value",
        Some("codex-test"),
        0,
    )
    .await
    .expect("login flow");
    let port = flow.local_addr.port();
    let target = "/auth/callback?code=callback-code&state=state-value".to_owned();
    let callback_task = tokio::spawn(async move { oauth_callback_get(port, &target).await });

    let credentials = finish_openai_codex_oauth_login_from_callback_at(flow, &token_url, 2_000_000)
        .await
        .expect("callback credentials");
    let callback_response = callback_task.await.expect("callback task");
    let request = request_task.await.expect("token request");
    let stored_credentials = StoredOAuthCredentials::from(credentials);
    let mut storage = AuthStorage::new();
    storage.insert(
        "openai-codex".to_owned(),
        AuthCredential::OAuth {
            credentials: stored_credentials.clone(),
        },
    );
    save_auth_storage_to_path(&path, &storage).expect("save auth storage");
    let saved_raw: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read auth storage"))
            .expect("saved auth json");
    let refresher = FailingOAuthRefresher {
        message: "should not refresh",
        calls: Mutex::new(Vec::new()),
    };

    let resolution = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "openai-codex",
        &path,
        2_000_000,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");

    assert!(callback_response.starts_with("HTTP/1.1 200 OK"));
    assert_eq!(stored_credentials.access, codex_access);
    assert_eq!(stored_credentials.refresh, "codex-callback-refresh");
    assert_eq!(
        stored_credentials.extra.get("accountId"),
        Some(&json!("acc_test"))
    );
    assert_eq!(
        stored_credentials.expires,
        2_000_000 + 7200 * 1000 - 5 * 60 * 1000
    );
    assert!(request.contains("grant_type=authorization_code"));
    assert!(request.contains("code=callback-code"));
    assert!(request.contains("code_verifier=verifier-value"));
    assert!(request.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fauth%2Fcallback"
    )));
    assert_eq!(saved_raw["openai-codex"]["type"], json!("oauth"));
    assert_eq!(
        saved_raw["openai-codex"]["access"],
        json!(stored_credentials.access.as_str())
    );
    assert_eq!(
        saved_raw["openai-codex"]["refresh"],
        json!("codex-callback-refresh")
    );
    assert_eq!(saved_raw["openai-codex"]["accountId"], json!("acc_test"));
    assert_eq!(resolution.api_key, stored_credentials.access);
    assert_eq!(resolution.credentials, Some(stored_credentials));
    assert!(!resolution.refreshed);
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn openai_codex_oauth_manual_input_login_persists_auth_storage_and_resolves_access_key() {
    let codex_access = codex_test_token();
    let (token_url, request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "codex-manual-storage-refresh",
        7200,
    ))
    .await;
    let path = auth_storage_test_path("openai-codex-manual-login-roundtrip");
    let parent = path.parent().expect("auth storage parent");
    let flow = start_openai_codex_oauth_login_flow_with_pkce(
        "verifier-value",
        "challenge-value",
        "state-value",
        Some("codex-test"),
        0,
    )
    .await
    .expect("login flow");
    let port = flow.local_addr.port();

    let credentials = finish_openai_codex_oauth_login_from_manual_input_at(
        flow,
        "http://localhost/auth/callback?code=manual-code&state=state-value",
        &token_url,
        2_000_000,
    )
    .await
    .expect("manual credentials");
    let request = request_task.await.expect("token request");
    let stored_credentials = StoredOAuthCredentials::from(credentials);
    let mut storage = AuthStorage::new();
    storage.insert(
        "openai-codex".to_owned(),
        AuthCredential::OAuth {
            credentials: stored_credentials.clone(),
        },
    );
    save_auth_storage_to_path(&path, &storage).expect("save auth storage");
    let saved_raw: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read auth storage"))
            .expect("saved auth json");
    let refresher = FailingOAuthRefresher {
        message: "should not refresh",
        calls: Mutex::new(Vec::new()),
    };

    let resolution = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "openai-codex",
        &path,
        2_000_000,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");

    assert_eq!(stored_credentials.access, codex_access);
    assert_eq!(stored_credentials.refresh, "codex-manual-storage-refresh");
    assert_eq!(
        stored_credentials.extra.get("accountId"),
        Some(&json!("acc_test"))
    );
    assert_eq!(
        stored_credentials.expires,
        2_000_000 + 7200 * 1000 - 5 * 60 * 1000
    );
    assert!(request.contains("grant_type=authorization_code"));
    assert!(request.contains("code=manual-code"));
    assert!(request.contains("code_verifier=verifier-value"));
    assert!(request.contains(&format!(
        "redirect_uri=http%3A%2F%2Flocalhost%3A{port}%2Fauth%2Fcallback"
    )));
    assert_eq!(saved_raw["openai-codex"]["type"], json!("oauth"));
    assert_eq!(
        saved_raw["openai-codex"]["access"],
        json!(stored_credentials.access.as_str())
    );
    assert_eq!(
        saved_raw["openai-codex"]["refresh"],
        json!("codex-manual-storage-refresh")
    );
    assert_eq!(saved_raw["openai-codex"]["accountId"], json!("acc_test"));
    assert_eq!(resolution.api_key, stored_credentials.access);
    assert_eq!(resolution.credentials, Some(stored_credentials));
    assert!(!resolution.refreshed);
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn oauth_auth_storage_refreshes_expired_oauth_and_persists_source_format() {
    let path = auth_storage_test_path("expired");
    let parent = path.parent().expect("auth storage parent");
    std::fs::create_dir_all(parent).expect("create auth storage dir");
    std::fs::write(
        &path,
        json!({
            "github-copilot": {
                "type": "oauth",
                "refresh": "ghu-refresh",
                "access": "expired-access",
                "expires": 999,
                "enterpriseUrl": "ghe.example.com"
            }
        })
        .to_string(),
    )
    .expect("write auth storage");
    let refresher = FakeOAuthRefresher {
        result: StoredOAuthCredentials {
            refresh: "ghu-refresh".to_owned(),
            access: "refreshed-access".to_owned(),
            expires: 2_000_000,
            extra: BTreeMap::from([("enterpriseUrl".to_owned(), json!("ghe.example.com"))]),
        },
        calls: Mutex::new(Vec::new()),
    };

    let result = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "github-copilot",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");
    let calls = refresher.calls.lock().expect("calls lock");
    let saved = load_auth_storage_from_path(&path).expect("load saved storage");
    let saved_raw: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read auth storage"))
            .expect("saved auth json");

    assert_eq!(result.api_key, "refreshed-access");
    assert!(result.refreshed);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "github-copilot");
    assert_eq!(calls[0].1.refresh, "ghu-refresh");
    assert_eq!(calls[0].1.enterprise_domain(), Some("ghe.example.com"));
    assert_eq!(calls[0].2, 1_000_000);
    assert_eq!(
        saved.get("github-copilot"),
        Some(&AuthCredential::OAuth {
            credentials: StoredOAuthCredentials {
                refresh: "ghu-refresh".to_owned(),
                access: "refreshed-access".to_owned(),
                expires: 2_000_000,
                extra: BTreeMap::from([("enterpriseUrl".to_owned(), json!("ghe.example.com"))]),
            }
        })
    );
    assert_eq!(
        saved_raw["github-copilot"]["enterpriseUrl"],
        json!("ghe.example.com")
    );
    assert!(saved_raw["github-copilot"].get("enterprise_url").is_none());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        assert_eq!(
            std::fs::metadata(&path)
                .expect("metadata")
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn oauth_auth_storage_refresh_failure_preserves_existing_file() {
    let path = auth_storage_test_path("refresh-failure");
    let parent = path.parent().expect("auth storage parent");
    std::fs::create_dir_all(parent).expect("create auth storage dir");
    let original = json!({
        "anthropic": {
            "type": "oauth",
            "refresh": "anthropic-refresh",
            "access": "expired-access",
            "expires": 999
        }
    })
    .to_string();
    std::fs::write(&path, &original).expect("write auth storage");
    let refresher = FailingOAuthRefresher {
        message: "network unavailable",
        calls: Mutex::new(Vec::new()),
    };

    let error = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "anthropic",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect_err("refresh failure");
    let calls = refresher.calls.lock().expect("calls lock");
    let saved_raw = std::fs::read_to_string(&path).expect("read auth storage");

    assert_eq!(
        error,
        "Failed to refresh OAuth token for anthropic: network unavailable"
    );
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].0, "anthropic");
    assert_eq!(calls[0].1.refresh, "anthropic-refresh");
    assert_eq!(calls[0].1.access, "expired-access");
    assert_eq!(calls[0].2, 1_000_000);
    assert_eq!(saved_raw, original);
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn oauth_auth_storage_missing_and_invalid_files_resolve_to_none() {
    let missing_path = auth_storage_test_path("missing");
    let invalid_path = auth_storage_test_path("invalid");
    let invalid_parent = invalid_path.parent().expect("invalid parent");
    std::fs::create_dir_all(invalid_parent).expect("create invalid parent");
    std::fs::write(&invalid_path, "not json").expect("write invalid auth storage");
    let refresher = FakeOAuthRefresher {
        result: StoredOAuthCredentials {
            refresh: "unused-refresh".to_owned(),
            access: "unused-access".to_owned(),
            expires: 3_000_000,
            extra: BTreeMap::new(),
        },
        calls: Mutex::new(Vec::new()),
    };

    assert!(
        resolve_auth_storage_api_key_from_path_with_refresher_at(
            "anthropic",
            &missing_path,
            1_000_000,
            &refresher,
        )
        .await
        .expect("missing storage")
        .is_none()
    );
    assert!(
        resolve_auth_storage_api_key_from_path_with_refresher_at(
            "anthropic",
            &invalid_path,
            1_000_000,
            &refresher,
        )
        .await
        .expect("invalid storage")
        .is_none()
    );
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(invalid_parent);
}

#[tokio::test]
async fn oauth_auth_storage_rejects_unknown_oauth_provider_before_refresh() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    reset_oauth_providers();
    let path = auth_storage_test_path("unknown-oauth-provider");
    let parent = path.parent().expect("auth storage parent");
    std::fs::create_dir_all(parent).expect("create auth storage dir");
    std::fs::write(
        &path,
        json!({
            "unknown-provider": {
                "type": "oauth",
                "refresh": "unknown-refresh",
                "access": "unknown-access",
                "expires": 2_000_000
            },
            "plain-provider": {
                "type": "api_key",
                "key": "plain-key"
            }
        })
        .to_string(),
    )
    .expect("write auth storage");
    let refresher = FakeOAuthRefresher {
        result: StoredOAuthCredentials {
            refresh: "unused-refresh".to_owned(),
            access: "unused-access".to_owned(),
            expires: 3_000_000,
            extra: BTreeMap::new(),
        },
        calls: Mutex::new(Vec::new()),
    };

    let error = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "unknown-provider",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect_err("unknown provider");
    let plain = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "plain-provider",
        &path,
        1_000_000,
        &refresher,
    )
    .await
    .expect("plain api key")
    .expect("plain result");

    assert_eq!(error, "Unknown OAuth provider: unknown-provider");
    assert_eq!(plain.api_key, "plain-key");
    assert!(plain.credentials.is_none());
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test]
async fn openai_codex_oauth_refresh_posts_form_and_maps_responses() {
    let codex_access = codex_test_token();
    let (token_url, request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "codex-refresh",
        7200,
    ))
    .await;

    let credentials =
        refresh_openai_codex_token_with_url_at("old refresh/token", &token_url, 2_000_000)
            .await
            .expect("refresh token");
    let request = request_task.await.expect("request task");

    assert_eq!(credentials.access, codex_access);
    assert_eq!(credentials.refresh, "codex-refresh");
    assert_eq!(credentials.expires, 2_000_000 + 7200 * 1000 - 5 * 60 * 1000);
    assert_eq!(credentials.extra.get("accountId"), Some(&json!("acc_test")));
    assert!(request.starts_with("POST / HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("content-type: application/x-www-form-urlencoded")
    );
    assert!(request.contains("grant_type=refresh_token"));
    assert!(request.contains("refresh_token=old+refresh%2Ftoken"));

    let (error_url, error_request_task) = mock_json_status_server(
        401,
        "Unauthorized",
        r#"{"error":{"message":"Could not validate your token.","type":"invalid_request_error"}}"#,
    )
    .await;
    let err = refresh_openai_codex_token_with_url_at("bad-token", &error_url, 0)
        .await
        .expect_err("refresh failure");
    let _ = error_request_task.await.expect("error request task");

    assert!(err.starts_with("OpenAI Codex token refresh failed (401): "));
    assert!(err.contains("Could not validate your token"));
    assert!(!err.contains("stderr"));
}

#[tokio::test(flavor = "current_thread")]
async fn openai_codex_oauth_refresh_routes_token_request_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let codex_access = codex_test_token();
    let (proxy_url, proxy_request_task) = mock_json_server(codex_oauth_token_response(
        &codex_access,
        "proxy-codex-refresh",
        7200,
    ))
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let credentials = refresh_openai_codex_token_with_url_at(
        "old refresh/token",
        "http://auth.example/oauth/token",
        2_000_000,
    )
    .await
    .expect("refresh token through proxy");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(credentials.access, codex_access);
    assert_eq!(credentials.refresh, "proxy-codex-refresh");
    assert_eq!(credentials.expires, 2_000_000 + 7200 * 1000 - 5 * 60 * 1000);
    assert_eq!(credentials.extra.get("accountId"), Some(&json!("acc_test")));
    assert!(
        proxy_request.starts_with("POST http://auth.example/oauth/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("content-type: application/x-www-form-urlencoded")
    );
    assert!(proxy_request.contains("grant_type=refresh_token"));
    assert!(proxy_request.contains("refresh_token=old+refresh%2Ftoken"));
    assert!(proxy_request.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
}

#[test]
fn github_copilot_oauth_device_flow_requests_match_provider() {
    let device = build_github_copilot_device_code_request("github.com");
    assert_eq!(device.url, "https://github.com/login/device/code");
    assert_eq!(device.method, "POST");
    assert_eq!(
        device.headers.get("Accept").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        device.headers.get("Content-Type").map(String::as_str),
        Some("application/x-www-form-urlencoded")
    );
    assert!(device.body.contains("client_id=Iv1.b507a08c87ecfe98"));
    assert!(device.body.contains("scope=read%3Auser"));

    let poll = build_github_copilot_access_token_poll_request("github.com", "device-code");
    assert_eq!(poll.url, "https://github.com/login/oauth/access_token");
    assert_eq!(poll.method, "POST");
    assert!(poll.body.contains("device_code=device-code"));
    assert!(
        poll.body
            .contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
    );
}

#[test]
fn github_copilot_oauth_poll_waits_before_first_poll_and_slows_down() {
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let (poll_times, outcome) = simulate_github_device_poll_times(
        start,
        5,
        900,
        &[
            GitHubDeviceTokenResponse::AuthorizationPending,
            GitHubDeviceTokenResponse::SlowDown {
                interval_seconds: Some(10),
            },
            GitHubDeviceTokenResponse::Success {
                access_token: "ghu_refresh_token".to_owned(),
            },
        ],
    );

    assert_eq!(
        poll_times,
        vec![start + 6_000, start + 12_000, start + 26_000]
    );
    assert_eq!(
        outcome,
        GitHubDevicePollOutcome::Success {
            access_token: "ghu_refresh_token".to_owned()
        }
    );
}

#[test]
fn github_copilot_oauth_poll_uses_remaining_lifetime_before_slow_down_timeout() {
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let (poll_times, outcome) = simulate_github_device_poll_times(
        start,
        5,
        25,
        &[
            GitHubDeviceTokenResponse::SlowDown {
                interval_seconds: Some(10),
            },
            GitHubDeviceTokenResponse::SlowDown {
                interval_seconds: Some(15),
            },
            GitHubDeviceTokenResponse::AuthorizationPending,
        ],
    );

    assert_eq!(
        poll_times,
        vec![start + 6_000, start + 20_000, start + 25_000]
    );
    assert_eq!(
        outcome,
        GitHubDevicePollOutcome::TimedOut {
            slow_down_seen: true
        }
    );
}

#[test]
fn github_copilot_oauth_refresh_and_base_url_helpers_match_provider() {
    assert_eq!(
        normalize_github_domain(" https://ghe.example.com/org "),
        Some("ghe.example.com".to_owned())
    );
    assert_eq!(
        normalize_github_domain("github.com"),
        Some("github.com".to_owned())
    );
    assert_eq!(normalize_github_domain("not a domain"), None);

    let refresh = build_github_copilot_refresh_request("ghu_refresh_token", None);
    assert_eq!(
        refresh.url,
        "https://api.github.com/copilot_internal/v2/token"
    );
    assert_eq!(
        refresh.headers.get("Authorization").map(String::as_str),
        Some("Bearer ghu_refresh_token")
    );
    assert_eq!(
        refresh
            .headers
            .get("Copilot-Integration-Id")
            .map(String::as_str),
        Some("vscode-chat")
    );

    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    assert_eq!(
        github_copilot_base_url(Some(token), None),
        "https://api.individual.githubcopilot.com"
    );
    assert_eq!(
        github_copilot_base_url(None, Some("ghe.example.com")),
        "https://copilot-api.ghe.example.com"
    );
    let credentials =
        parse_github_copilot_token_response("ghu_refresh_token", token, 9_999_999_999, None);
    assert_eq!(credentials.refresh, "ghu_refresh_token");
    assert_eq!(credentials.access, token);
    assert_eq!(credentials.expires, 9_999_999_999_000 - 5 * 60 * 1000);
}

#[tokio::test]
async fn github_copilot_oauth_network_device_poll_and_refresh_flows() {
    let (device_url, device_request_task) = mock_json_server(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device","verification_uri_complete":"https://github.com/login/device?user_code=ABCD-EFGH","interval":5,"expires_in":900}"#,
    )
    .await;
    let (poll_url, poll_request_task) =
        mock_json_server(r#"{"error":"slow_down","error_description":"wait","interval":10}"#).await;
    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    let (refresh_url, refresh_request_task) = mock_json_server(
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    )
    .await;
    let urls = GitHubCopilotUrls {
        device_code_url: device_url,
        access_token_url: poll_url,
        copilot_token_url: refresh_url,
    };

    let device = request_github_copilot_device_code_for_urls(&urls)
        .await
        .expect("device code");
    let device_request = device_request_task.await.expect("device request task");
    assert_eq!(device.device_code, "device-code");
    assert_eq!(device.user_code, "ABCD-EFGH");
    assert_eq!(device.interval_seconds, 5);
    assert_eq!(device.expires_in_seconds, 900);
    assert!(device_request.starts_with("POST / HTTP/1.1"));
    assert!(device_request.contains("scope=read%3Auser"));

    let poll = poll_github_copilot_access_token_for_urls(&urls, "device-code")
        .await
        .expect("poll");
    let poll_request = poll_request_task.await.expect("poll request task");
    assert_eq!(
        poll,
        GitHubDeviceTokenResponse::SlowDown {
            interval_seconds: Some(10)
        }
    );
    assert!(poll_request.contains("device_code=device-code"));

    let credentials = refresh_github_copilot_token_for_urls_at(
        "ghu_refresh_token",
        &urls,
        Some("ghe.example.com"),
        0,
    )
    .await
    .expect("refresh");
    let refresh_request = refresh_request_task.await.expect("refresh request task");
    assert_eq!(credentials.refresh, "ghu_refresh_token");
    assert_eq!(credentials.access, token);
    assert_eq!(
        credentials.enterprise_url.as_deref(),
        Some("ghe.example.com")
    );
    assert_eq!(credentials.expires, 9_999_999_999_000 - 5 * 60 * 1000);
    assert!(refresh_request.starts_with("GET / HTTP/1.1"));
    assert!(
        refresh_request
            .to_ascii_lowercase()
            .contains("authorization: bearer ghu_refresh_token")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn github_copilot_oauth_refresh_routes_token_request_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    let (proxy_url, proxy_request_task) = mock_json_server(
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    )
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");
    let urls = GitHubCopilotUrls {
        device_code_url: "http://github.example/login/device/code".to_owned(),
        access_token_url: "http://github.example/login/oauth/access_token".to_owned(),
        copilot_token_url: "http://api.github.example/copilot_internal/v2/token".to_owned(),
    };

    let credentials = refresh_github_copilot_token_for_urls_at(
        "ghu_refresh_token",
        &urls,
        Some("ghe.example.com"),
        0,
    )
    .await
    .expect("refresh token through proxy");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(credentials.refresh, "ghu_refresh_token");
    assert_eq!(credentials.access, token);
    assert_eq!(
        credentials.enterprise_url.as_deref(),
        Some("ghe.example.com")
    );
    assert_eq!(credentials.expires, 9_999_999_999_000 - 5 * 60 * 1000);
    assert!(
        proxy_request
            .starts_with("GET http://api.github.example/copilot_internal/v2/token HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: bearer ghu_refresh_token")
    );
}

#[tokio::test]
async fn github_copilot_oauth_complete_device_flow_waits_and_refreshes() {
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let (device_url, device_request_task) = mock_json_server(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device","interval":5,"expires_in":900}"#,
    )
    .await;
    let (poll_url, poll_requests_task) = mock_json_sequence_server(vec![
        r#"{"error":"authorization_pending","error_description":"pending"}"#,
        r#"{"error":"slow_down","error_description":"wait","interval":10}"#,
        r#"{"access_token":"ghu_refresh_token"}"#,
    ])
    .await;
    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    let (refresh_url, refresh_request_task) = mock_json_server(
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    )
    .await;
    let urls = GitHubCopilotUrls {
        device_code_url: device_url,
        access_token_url: poll_url,
        copilot_token_url: refresh_url,
    };
    let seen_device = Arc::new(Mutex::new(None::<GitHubDeviceCode>));
    let seen_device_ref = seen_device.clone();
    let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));
    let sleeps_ref = sleeps.clone();

    let login = login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        &urls,
        None,
        move |device| {
            *seen_device_ref.lock().expect("seen device") = Some(device.clone());
        },
        start,
        move |delay_ms| {
            sleeps_ref.lock().expect("sleeps").push(delay_ms);
            std::future::ready(())
        },
        &GitHubCopilotModelPolicyOptions::disabled(),
    )
    .await
    .expect("login");

    let device_request = device_request_task.await.expect("device request task");
    let poll_requests = poll_requests_task.await.expect("poll requests task");
    let refresh_request = refresh_request_task.await.expect("refresh request task");

    assert_eq!(
        sleeps.lock().expect("sleeps").as_slice(),
        &[6_000, 6_000, 14_000]
    );
    assert_eq!(
        seen_device
            .lock()
            .expect("seen device")
            .as_ref()
            .map(|device| device.user_code.as_str()),
        Some("ABCD-EFGH")
    );
    assert_eq!(login.device.device_code, "device-code");
    assert_eq!(login.credentials.refresh, "ghu_refresh_token");
    assert_eq!(login.credentials.access, token);
    assert!(device_request.contains("scope=read%3Auser"));
    assert_eq!(poll_requests.len(), 3);
    assert!(poll_requests[0].contains("device_code=device-code"));
    assert!(
        refresh_request
            .to_ascii_lowercase()
            .contains("authorization: bearer ghu_refresh_token")
    );
}

#[tokio::test]
async fn github_copilot_oauth_device_flow_persists_auth_storage_and_resolves_access_key() {
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let path = auth_storage_test_path("github-copilot-login-roundtrip");
    let parent = path.parent().expect("auth storage parent");
    let (device_url, device_request_task) = mock_json_server(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device","verification_uri_complete":"https://github.com/login/device?user_code=ABCD-EFGH","interval":1,"expires_in":60}"#,
    )
    .await;
    let (poll_url, poll_request_task) =
        mock_json_server(r#"{"access_token":"ghu_refresh_token"}"#).await;
    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    let (refresh_url, refresh_request_task) = mock_json_server(
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    )
    .await;
    let urls = GitHubCopilotUrls {
        device_code_url: device_url,
        access_token_url: poll_url,
        copilot_token_url: refresh_url,
    };
    let seen_device = Arc::new(Mutex::new(None::<GitHubDeviceCode>));
    let seen_device_ref = seen_device.clone();

    let login = login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        &urls,
        Some("ghe.example.com"),
        move |device| {
            *seen_device_ref.lock().expect("seen device") = Some(device.clone());
        },
        start,
        |_| std::future::ready(()),
        &GitHubCopilotModelPolicyOptions::disabled(),
    )
    .await
    .expect("login");
    let device_request = device_request_task.await.expect("device request task");
    let poll_request = poll_request_task.await.expect("poll request task");
    let refresh_request = refresh_request_task.await.expect("refresh request task");
    let stored_credentials = StoredOAuthCredentials::from(login.credentials);
    let mut storage = AuthStorage::new();
    storage.insert(
        "github-copilot".to_owned(),
        AuthCredential::OAuth {
            credentials: stored_credentials.clone(),
        },
    );
    save_auth_storage_to_path(&path, &storage).expect("save auth storage");
    let saved_raw: Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("read auth storage"))
            .expect("saved auth json");
    let refresher = FailingOAuthRefresher {
        message: "should not refresh",
        calls: Mutex::new(Vec::new()),
    };

    let resolution = resolve_auth_storage_api_key_from_path_with_refresher_at(
        "github-copilot",
        &path,
        start,
        &refresher,
    )
    .await
    .expect("oauth resolution")
    .expect("oauth result");

    assert_eq!(
        seen_device
            .lock()
            .expect("seen device")
            .as_ref()
            .map(|device| device.verification_uri_complete.as_deref()),
        Some(Some("https://github.com/login/device?user_code=ABCD-EFGH"))
    );
    assert!(device_request.contains("scope=read%3Auser"));
    assert!(poll_request.contains("device_code=device-code"));
    assert!(
        refresh_request
            .to_ascii_lowercase()
            .contains("authorization: bearer ghu_refresh_token")
    );
    assert_eq!(stored_credentials.access, token);
    assert_eq!(stored_credentials.refresh, "ghu_refresh_token");
    assert_eq!(
        stored_credentials.enterprise_domain(),
        Some("ghe.example.com")
    );
    assert_eq!(
        stored_credentials.expires,
        9_999_999_999_000 - 5 * 60 * 1000
    );
    assert_eq!(saved_raw["github-copilot"]["type"], json!("oauth"));
    assert_eq!(saved_raw["github-copilot"]["access"], json!(token));
    assert_eq!(
        saved_raw["github-copilot"]["refresh"],
        json!("ghu_refresh_token")
    );
    assert_eq!(
        saved_raw["github-copilot"]["enterpriseUrl"],
        json!("ghe.example.com")
    );
    assert!(saved_raw["github-copilot"].get("enterprise_url").is_none());
    assert_eq!(resolution.api_key, token);
    assert_eq!(resolution.credentials, Some(stored_credentials));
    assert!(!resolution.refreshed);
    assert!(refresher.calls.lock().expect("calls lock").is_empty());
    let _ = std::fs::remove_dir_all(parent);
}

#[tokio::test(flavor = "current_thread")]
async fn github_copilot_oauth_complete_device_flow_routes_requests_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    let (proxy_url, proxy_requests_task) = mock_json_sequence_server(vec![
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device","interval":1,"expires_in":60}"#,
        r#"{"access_token":"ghu_refresh_token"}"#,
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    ])
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");
    let urls = GitHubCopilotUrls {
        device_code_url: "http://github.example/login/device/code".to_owned(),
        access_token_url: "http://github.example/login/oauth/access_token".to_owned(),
        copilot_token_url: "http://api.github.example/copilot_internal/v2/token".to_owned(),
    };
    let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));
    let sleeps_ref = sleeps.clone();

    let login = login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        &urls,
        None,
        |_| {},
        start,
        move |delay_ms| {
            sleeps_ref.lock().expect("sleeps").push(delay_ms);
            std::future::ready(())
        },
        &GitHubCopilotModelPolicyOptions::disabled(),
    )
    .await
    .expect("login through proxy");
    let proxy_requests = proxy_requests_task.await.expect("proxy requests task");

    assert_eq!(login.device.device_code, "device-code");
    assert_eq!(login.credentials.refresh, "ghu_refresh_token");
    assert_eq!(login.credentials.access, token);
    assert_eq!(sleeps.lock().expect("sleeps").len(), 1);
    assert_eq!(proxy_requests.len(), 3);
    assert!(
        proxy_requests[0].starts_with("POST http://github.example/login/device/code HTTP/1.1"),
        "{}",
        proxy_requests[0]
    );
    assert!(proxy_requests[0].contains("scope=read%3Auser"));
    assert!(
        proxy_requests[1]
            .starts_with("POST http://github.example/login/oauth/access_token HTTP/1.1"),
        "{}",
        proxy_requests[1]
    );
    assert!(proxy_requests[1].contains("device_code=device-code"));
    assert!(
        proxy_requests[2]
            .starts_with("GET http://api.github.example/copilot_internal/v2/token HTTP/1.1"),
        "{}",
        proxy_requests[2]
    );
    assert!(
        proxy_requests[2]
            .to_ascii_lowercase()
            .contains("authorization: bearer ghu_refresh_token")
    );
}

#[tokio::test]
async fn github_copilot_oauth_login_enables_model_policies_after_refresh() {
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let (device_url, device_request_task) = mock_json_server(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/device","interval":1,"expires_in":60}"#,
    )
    .await;
    let (poll_url, poll_request_task) =
        mock_json_server(r#"{"access_token":"ghu_refresh_token"}"#).await;
    let token = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;";
    let (refresh_url, refresh_request_task) = mock_json_server(
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    )
    .await;
    let (policy_url, policy_requests_task) = mock_json_sequence_server(vec!["{}", "{}"]).await;
    let urls = GitHubCopilotUrls {
        device_code_url: device_url,
        access_token_url: poll_url,
        copilot_token_url: refresh_url,
    };
    let policy_options = GitHubCopilotModelPolicyOptions {
        enabled: true,
        base_url: Some(policy_url),
        model_ids: Some(vec!["gpt-4o".to_owned(), "claude-sonnet-4.6".to_owned()]),
    };

    let login = login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        &urls,
        None,
        |_| {},
        start,
        |_| std::future::ready(()),
        &policy_options,
    )
    .await
    .expect("login with model policy enable");
    let device_request = device_request_task.await.expect("device request task");
    let poll_request = poll_request_task.await.expect("poll request task");
    let refresh_request = refresh_request_task.await.expect("refresh request task");
    let policy_requests = policy_requests_task.await.expect("policy requests task");

    assert_eq!(login.credentials.access, token);
    assert!(device_request.contains("scope=read%3Auser"));
    assert!(poll_request.contains("device_code=device-code"));
    assert!(
        refresh_request
            .to_ascii_lowercase()
            .contains("authorization: bearer ghu_refresh_token")
    );
    assert_eq!(policy_requests.len(), 2);
    assert!(
        policy_requests[0].starts_with("POST /models/gpt-4o/policy HTTP/1.1"),
        "{}",
        policy_requests[0]
    );
    assert!(
        policy_requests[1].starts_with("POST /models/claude-sonnet-4.6/policy HTTP/1.1"),
        "{}",
        policy_requests[1]
    );
    for request in &policy_requests {
        let lower = request.to_ascii_lowercase();
        assert!(lower.contains("authorization: bearer tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;"));
        assert!(lower.contains("copilot-integration-id: vscode-chat"));
        assert!(lower.contains("openai-intent: chat-policy"));
        assert!(lower.contains("x-interaction-type: chat-policy"));
        assert!(request.contains(r#"{"state":"enabled"}"#));
    }
}

#[tokio::test]
async fn github_copilot_oauth_complete_device_flow_times_out_after_slow_down() {
    let start = chrono::DateTime::parse_from_rfc3339("2026-03-09T00:00:00Z")
        .expect("date")
        .timestamp_millis();
    let (poll_url, poll_requests_task) = mock_json_sequence_server(vec![
        r#"{"error":"slow_down","error_description":"wait","interval":10}"#,
        r#"{"error":"slow_down","error_description":"still waiting","interval":15}"#,
        r#"{"error":"authorization_pending","error_description":"pending"}"#,
    ])
    .await;
    let urls = GitHubCopilotUrls {
        device_code_url: "http://unused.invalid/device".to_owned(),
        access_token_url: poll_url,
        copilot_token_url: "http://unused.invalid/token".to_owned(),
    };
    let device = GitHubDeviceCode {
        device_code: "device-code".to_owned(),
        user_code: "ABCD-EFGH".to_owned(),
        verification_uri: "https://github.com/login/device".to_owned(),
        verification_uri_complete: None,
        interval_seconds: 5,
        expires_in_seconds: 25,
    };
    let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));
    let sleeps_ref = sleeps.clone();

    let err = complete_github_copilot_device_flow_for_urls_with_sleeper(
        &urls,
        &device,
        start,
        move |delay_ms| {
            sleeps_ref.lock().expect("sleeps").push(delay_ms);
            std::future::ready(())
        },
    )
    .await
    .expect_err("timeout");
    let poll_requests = poll_requests_task.await.expect("poll requests task");

    assert_eq!(
        sleeps.lock().expect("sleeps").as_slice(),
        &[6_000, 14_000, 5_000]
    );
    assert_eq!(poll_requests.len(), 3);
    assert!(err.contains("Device flow timed out after one or more slow_down responses"));
    assert!(err.contains("clock drift"));
}

#[test]
fn mistral_payload_serializes_tool_schema_as_plain_json() {
    let model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    let parameters = json!({
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            }
        }
    });
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Hi"))],
        tools: vec![Tool {
            name: "inspect_schema".to_owned(),
            description: "Inspect the schema".to_owned(),
            parameters: parameters.clone(),
        }],
        ..Default::default()
    };

    let payload = build_mistral_chat_payload(&model, &context, MistralPayloadOptions::default());
    assert_eq!(payload["tools"].as_array().map(Vec::len), Some(1));
    let function = &payload["tools"][0]["function"];
    assert_eq!(function["name"], "inspect_schema");
    assert_eq!(function["parameters"], parameters);
    assert_eq!(function["strict"], Value::Bool(false));
    assert!(function["parameters"].as_object().is_some());
    assert!(function["parameters"]["properties"].as_object().is_some());
    assert!(
        function["parameters"]["properties"]["nested"]
            .as_object()
            .is_some()
    );
    serde_json::to_string(&function["parameters"]).expect("plain JSON schema");
}

#[test]
fn mistral_simple_payload_selects_prompt_or_effort_reasoning_controls() {
    let context = user_context("Hello");

    let small = get_model("mistral", "mistral-small-2603").expect("mistral small");
    let payload = build_mistral_simple_payload(
        &small,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(
        payload.get("reasoningEffort").and_then(Value::as_str),
        Some("high")
    );
    assert!(payload.get("promptMode").is_none());

    let payload = build_mistral_simple_payload(&small, &context, SimpleStreamOptions::default());
    assert!(payload.get("reasoningEffort").is_none());
    assert!(payload.get("promptMode").is_none());

    let magistral = get_model("mistral", "magistral-medium-latest").expect("magistral");
    let payload = build_mistral_simple_payload(
        &magistral,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(
        payload.get("promptMode").and_then(Value::as_str),
        Some("reasoning")
    );
    assert!(payload.get("reasoningEffort").is_none());

    let medium = get_model("mistral", "mistral-medium-3.5").expect("mistral medium");
    let payload = build_mistral_simple_payload(
        &medium,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(
        payload.get("reasoningEffort").and_then(Value::as_str),
        Some("high")
    );
    assert!(payload.get("promptMode").is_none());

    let payload = build_mistral_simple_payload(&medium, &context, SimpleStreamOptions::default());
    assert!(payload.get("reasoningEffort").is_none());
    assert!(payload.get("promptMode").is_none());
}

#[test]
fn mistral_payload_preserves_image_tool_results_for_vision_models() {
    let mut model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    model.input = vec![InputKind::Text, InputKind::Image];
    let mut assistant = empty_assistant_for_model(&model);
    assistant.stop_reason = StopReason::ToolUse;
    assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "call_image".to_owned(),
        name: "read_image".to_owned(),
        arguments: object(json!({ "path": "circle.png" })),
        thought_signature: None,
    })];
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Read the image.")),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_image".to_owned(),
                tool_name: "read_image".to_owned(),
                content: vec![
                    ToolResultContent::text("A red circle with a diameter of 100 pixels."),
                    ToolResultContent::Image(ImageContent {
                        data: "base64-image".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
        ],
        ..Default::default()
    };

    let payload = build_mistral_chat_payload(&model, &context, MistralPayloadOptions::default());

    assert_eq!(payload["messages"][1]["role"], "assistant");
    assert_eq!(
        payload["messages"][1]["toolCalls"][0]["function"]["name"],
        "read_image"
    );
    assert_eq!(payload["messages"][2]["role"], "tool");
    assert_eq!(payload["messages"][2]["toolCallId"], "call_image");
    assert_eq!(
        payload["messages"][2]["content"],
        json!([
            {
                "type": "text",
                "text": "A red circle with a diameter of 100 pixels.",
            },
            {
                "type": "image_url",
                "imageUrl": "data:image/png;base64,base64-image",
            },
        ])
    );
}

#[test]
fn mistral_payload_synthesizes_missing_tool_results_and_normalizes_ids() {
    let model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    let mut assistant = empty_assistant_for_model(&model);
    assistant.provider = "openai".to_owned();
    assistant.api = "openai-responses".to_owned();
    assistant.model = "gpt-5-mini".to_owned();
    assistant.stop_reason = StopReason::ToolUse;
    assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "call_1|fc_1".to_owned(),
        name: "calculate".to_owned(),
        arguments: object(json!({ "expression": "25 * 18" })),
        thought_signature: Some("foreign-signature".to_owned()),
    })];
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Calculate 25 * 18.")),
            Message::Assistant(assistant),
            Message::User(UserMessage::text("Never mind. What is 2 + 2?")),
        ],
        ..Default::default()
    };

    let payload = build_mistral_chat_payload(&model, &context, MistralPayloadOptions::default());
    let tool_call_id = payload["messages"][1]["toolCalls"][0]["id"]
        .as_str()
        .expect("normalized tool call id");

    assert_ne!(tool_call_id, "call_1|fc_1");
    assert_eq!(tool_call_id.len(), 9);
    assert!(tool_call_id.chars().all(|ch| ch.is_ascii_alphanumeric()));
    assert_eq!(payload["messages"][2]["role"], "tool");
    assert_eq!(payload["messages"][2]["toolCallId"], tool_call_id);
    assert_eq!(
        payload["messages"][2]["content"],
        json!([{ "type": "text", "text": "[tool error] No result provided" }])
    );
    assert_eq!(payload["messages"][3]["role"], "user");
    assert_eq!(
        payload["messages"][3]["content"],
        "Never mind. What is 2 + 2?"
    );
}

#[test]
fn mistral_request_headers_apply_session_affinity_without_overriding_callers() {
    let mut model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    model
        .headers
        .insert("x-model-header".to_owned(), "model".to_owned());
    model
        .headers
        .insert("x-shared".to_owned(), "from-model".to_owned());

    let headers = build_mistral_request_headers(&model, Some("session-1"), &BTreeMap::from([]));
    assert_eq!(
        headers.get("x-model-header").map(String::as_str),
        Some("model")
    );
    assert_eq!(
        headers.get("x-affinity").map(String::as_str),
        Some("session-1")
    );

    let headers = build_mistral_request_headers(
        &model,
        Some("session-2"),
        &BTreeMap::from([
            ("x-shared".to_owned(), "from-options".to_owned()),
            ("x-affinity".to_owned(), "caller-affinity".to_owned()),
        ]),
    );
    assert_eq!(
        headers.get("x-shared").map(String::as_str),
        Some("from-options")
    );
    assert_eq!(
        headers.get("x-affinity").map(String::as_str),
        Some("caller-affinity")
    );

    let headers = build_mistral_request_headers(
        &model,
        Some("session-3"),
        &BTreeMap::from([("x-affinity".to_owned(), String::new())]),
    );
    assert_eq!(
        headers.get("x-affinity").map(String::as_str),
        Some("session-3")
    );
}

#[tokio::test]
async fn mistral_stream_chunks_preserve_response_id_usage_and_tool_calls() {
    let model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_mistral_chat_chunks(
        [
            json!({
                "id": "mistral-response-1",
                "usage": {
                    "promptTokens": 11,
                    "completionTokens": 7,
                    "totalTokens": 18
                },
                "choices": [{
                    "delta": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{ "type": "text", "text": "plan " }]
                            },
                            { "type": "text", "text": "Hello" }
                        ]
                    }
                }]
            }),
            json!({
                "id": "mistral-response-ignored",
                "choices": [{
                    "delta": {
                        "toolCalls": [{
                            "id": "null",
                            "index": 0,
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"q\""
                            }
                        }]
                    }
                }]
            }),
            json!({
                "choices": [{
                    "finishReason": "tool_calls",
                    "delta": {
                        "toolCalls": [{
                            "index": 0,
                            "function": { "arguments": ":\"rust\"}" }
                        }]
                    }
                }]
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process mistral chunks");
    drop(sender);

    assert_eq!(output.response_id.as_deref(), Some("mistral-response-1"));
    assert_eq!(output.stop_reason, StopReason::ToolUse);
    assert_eq!(output.usage.input, 11);
    assert_eq!(output.usage.output, 7);
    assert_eq!(output.usage.total_tokens, 18);
    assert_usage_total_matches_components("mistral stream usage", &output.usage);
    assert!(matches!(
        &output.content[0],
        AssistantContent::Thinking(thinking) if thinking.thinking == "plan "
    ));
    assert!(matches!(
        &output.content[1],
        AssistantContent::Text(text) if text.text == "Hello"
    ));
    let AssistantContent::ToolCall(tool_call) = &output.content[2] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.id.len(), 9);
    assert!(tool_call.id.chars().all(|ch| ch.is_ascii_alphanumeric()));
    assert_eq!(tool_call.name, "lookup");
    assert_eq!(tool_call.arguments["q"], "rust");

    let events = collect_events(stream).await;
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "plan "
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hello"
    )));
    let toolcall_end = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call),
            _ => None,
        })
        .expect("toolcall end");
    assert_eq!(toolcall_end.arguments["q"], "rust");
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done {
            reason: StopReason::ToolUse,
            ..
        })
    ));
}

#[tokio::test]
async fn mistral_stream_chunks_emit_start_before_done_for_empty_stream() {
    let model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_mistral_chat_chunks(std::iter::empty::<Value>(), &mut output, &sender, &model)
        .expect("empty mistral stream");
    drop(sender);

    let events = collect_events(stream).await;
    assert_eq!(events.len(), 2);
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            ..
        })
    ));
}

#[test]
fn bedrock_model_registry_exposes_available_models() {
    let models = get_models("amazon-bedrock");
    assert!(!models.is_empty());
    assert!(
        models
            .iter()
            .all(|model| model.provider == "amazon-bedrock")
    );
    assert!(
        models
            .iter()
            .any(|model| model.id == "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    );
}

#[test]
fn bedrock_endpoint_resolution_matches_region_and_profile_rules() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE"]);

    let eu_model = get_model(
        "amazon-bedrock",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("eu model");
    assert_eq!(
        eu_model.base_url,
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
    );

    let config = resolve_bedrock_client_config(&eu_model, BedrockClientOptions::default());
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-runtime.eu-central-1.amazonaws.com")
    );
    assert_eq!(config.region.as_deref(), Some("eu-central-1"));

    set_env("AWS_REGION", "us-east-2");
    let us_model = get_model("amazon-bedrock", "us.anthropic.claude-opus-4-7").expect("us model");
    let config = resolve_bedrock_client_config(&us_model, BedrockClientOptions::default());
    assert_eq!(config.region.as_deref(), Some("us-east-2"));
    assert_eq!(config.endpoint, None);

    let mut custom_endpoint = us_model.clone();
    custom_endpoint.base_url = "https://bedrock-vpc.example.com".to_owned();
    let config = resolve_bedrock_client_config(&custom_endpoint, BedrockClientOptions::default());
    assert_eq!(
        config.endpoint.as_deref(),
        Some("https://bedrock-vpc.example.com")
    );
    assert_eq!(config.region.as_deref(), Some("us-east-2"));
}

#[test]
fn bedrock_aws_profile_credentials_and_region_resolve_from_shared_files() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
    ]);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ri-aws-profile-{unique}"));
    std::fs::create_dir_all(&dir).expect("create temp profile dir");
    let credentials_path = dir.join("credentials");
    let config_path = dir.join("config");
    std::fs::write(
        &credentials_path,
        "[profile-test]\naws_access_key_id = AKIDPROFILE\naws_secret_access_key = profile-secret\naws_session_token = profile-session\n",
    )
    .expect("write credentials");
    std::fs::write(&config_path, "[profile profile-test]\nregion = eu-west-1\n")
        .expect("write config");
    set_env("AWS_PROFILE", "profile-test");
    set_env(
        "AWS_SHARED_CREDENTIALS_FILE",
        credentials_path.to_str().expect("credentials path"),
    );
    set_env(
        "AWS_CONFIG_FILE",
        config_path.to_str().expect("config path"),
    );

    let credentials = resolve_aws_credentials(None).expect("credentials");
    assert_eq!(credentials.access_key_id, "AKIDPROFILE");
    assert_eq!(credentials.secret_access_key, "profile-secret");
    assert_eq!(
        credentials.session_token.as_deref(),
        Some("profile-session")
    );
    assert_eq!(
        resolve_aws_profile_region(None).as_deref(),
        Some("eu-west-1")
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn bedrock_env_api_key_marker_matches_supported_http_auth_sources() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
    ]);

    set_env("AWS_PROFILE", "");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    remove_env("AWS_PROFILE");

    set_env("AWS_ACCESS_KEY_ID", "AKID");
    set_env("AWS_SECRET_ACCESS_KEY", "");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    remove_env("AWS_ACCESS_KEY_ID");
    remove_env("AWS_SECRET_ACCESS_KEY");

    set_env("AWS_BEARER_TOKEN_BEDROCK", "");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    remove_env("AWS_BEARER_TOKEN_BEDROCK");

    set_env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    remove_env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");

    set_env("AWS_WEB_IDENTITY_TOKEN_FILE", "");
    set_env("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/test");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    remove_env("AWS_WEB_IDENTITY_TOKEN_FILE");
    remove_env("AWS_ROLE_ARN");

    set_env("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/secrets/token");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    set_env("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/test");
    assert_eq!(
        get_env_api_key("amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );
    remove_env("AWS_WEB_IDENTITY_TOKEN_FILE");
    remove_env("AWS_ROLE_ARN");

    set_env(
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "/v2/credentials/task",
    );
    assert_eq!(
        get_env_api_key("amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );
    remove_env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI");

    set_env(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "http://169.254.170.2/v2/credentials/task",
    );
    assert_eq!(
        get_env_api_key("amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );

    remove_env("AWS_CONTAINER_CREDENTIALS_FULL_URI");
    set_env("AWS_PROFILE", "profile-test");
    assert_eq!(
        get_env_api_key("amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );

    remove_env("AWS_PROFILE");
    set_env("AWS_ACCESS_KEY_ID", "AKID");
    assert_eq!(get_env_api_key("amazon-bedrock"), None);
    set_env("AWS_SECRET_ACCESS_KEY", "secret");
    assert_eq!(
        get_env_api_key("amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );

    remove_env("AWS_ACCESS_KEY_ID");
    remove_env("AWS_SECRET_ACCESS_KEY");
    set_env("AWS_BEARER_TOKEN_BEDROCK", "bearer");
    assert_eq!(
        get_env_api_key("amazon-bedrock"),
        Some("<authenticated>".to_owned())
    );
}

#[test]
fn env_api_keys_ignore_empty_values_and_preserve_provider_precedence() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "OPENAI_API_KEY",
        "ANTHROPIC_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ]);

    set_env("OPENAI_API_KEY", "");
    assert_eq!(find_env_keys("openai"), None);
    assert_eq!(get_env_api_key("openai"), None);

    set_env("ANTHROPIC_OAUTH_TOKEN", "");
    set_env("ANTHROPIC_API_KEY", "anthropic-key");
    assert_eq!(
        find_env_keys("anthropic"),
        Some(vec!["ANTHROPIC_API_KEY".to_owned()])
    );
    assert_eq!(
        get_env_api_key("anthropic"),
        Some("anthropic-key".to_owned())
    );

    set_env("ANTHROPIC_OAUTH_TOKEN", "oauth-token");
    assert_eq!(
        find_env_keys("anthropic"),
        Some(vec![
            "ANTHROPIC_OAUTH_TOKEN".to_owned(),
            "ANTHROPIC_API_KEY".to_owned()
        ])
    );
    assert_eq!(get_env_api_key("anthropic"), Some("oauth-token".to_owned()));
}

#[test]
fn bedrock_raw_message_conversion_skips_unknown_user_content_blocks() {
    let model = get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("bedrock model");
    let messages = convert_bedrock_raw_messages(
        &[json!({
            "role": "user",
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "unknown", "data": "foo" },
            ],
        })],
        &model,
        CacheRetention::None,
    );

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["content"].as_array().map(Vec::len), Some(1));
    assert_eq!(messages[0]["content"][0], json!({ "text": "hello" }));
}

#[test]
fn bedrock_raw_message_conversion_skips_unknown_assistant_content_blocks() {
    let model = get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("bedrock model");
    let messages = convert_bedrock_raw_messages(
        &[json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "hello" },
                { "type": "unknown", "data": "foo" },
            ],
        })],
        &model,
        CacheRetention::None,
    );

    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["content"].as_array().map(Vec::len), Some(1));
    assert_eq!(messages[0]["content"][0], json!({ "text": "hello" }));
}

#[test]
fn bedrock_raw_message_conversion_skips_user_messages_with_only_unknown_blocks() {
    let model = get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("bedrock model");
    let messages = convert_bedrock_raw_messages(
        &[json!({
            "role": "user",
            "content": [{ "type": "unknown", "data": "foo" }],
        })],
        &model,
        CacheRetention::None,
    );

    assert!(messages.is_empty());
}

#[test]
fn bedrock_raw_message_conversion_skips_assistant_messages_with_only_unknown_blocks() {
    let model = get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("bedrock model");
    let messages = convert_bedrock_raw_messages(
        &[json!({
            "role": "assistant",
            "content": [{ "type": "unknown", "data": "foo" }],
        })],
        &model,
        CacheRetention::None,
    );

    assert!(messages.is_empty());
}

fn bedrock_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    }
}

#[test]
fn bedrock_payload_uses_adaptive_thinking_for_claude_opus_47() {
    let mut model =
        get_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1").expect("model");
    model.id = "global.anthropic.claude-opus-4-7-v1".to_owned();
    model.name = "Claude Opus 4.7 (Global)".to_owned();

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["output_config"],
        json!({ "effort": "high" })
    );
    assert!(
        payload["additionalModelRequestFields"]
            .get("anthropic_beta")
            .is_none()
    );
}

#[test]
fn bedrock_payload_maps_xhigh_to_max_for_opus_46() {
    let model = get_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1").expect("model");

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["output_config"],
        json!({ "effort": "max" })
    );
    assert!(
        payload["additionalModelRequestFields"]
            .get("anthropic_beta")
            .is_none()
    );
}

#[test]
fn bedrock_payload_maps_xhigh_to_native_opus_47_effort() {
    let mut model =
        get_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1").expect("model");
    model.id = "global.anthropic.claude-opus-4-7-v1".to_owned();
    model.name = "Claude Opus 4.7 (Global)".to_owned();

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["output_config"],
        json!({ "effort": "xhigh" })
    );
}

#[test]
fn bedrock_payload_omits_display_for_govcloud_nonadaptive_thinking() {
    let mut model = get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");
    model.id = "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0".to_owned();
    model.name = "Claude Sonnet 4.5 (GovCloud)".to_owned();

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "enabled", "budget_tokens": 16_384 })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["anthropic_beta"],
        json!(["interleaved-thinking-2025-05-14"])
    );
}

#[test]
fn bedrock_payload_omits_display_for_govcloud_adaptive_region() {
    let mut model =
        get_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1").expect("model");
    model.id = "global.anthropic.claude-opus-4-7-v1".to_owned();
    model.name = "Claude Opus 4.7 (Global)".to_owned();

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            region: Some("us-gov-west-1".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "adaptive" })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["output_config"],
        json!({ "effort": "high" })
    );
    assert!(
        payload["additionalModelRequestFields"]
            .get("anthropic_beta")
            .is_none()
    );
}

#[test]
fn bedrock_payload_uses_model_name_for_application_profile_adaptive_thinking() {
    let mut model =
        get_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1").expect("model");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile"
        .to_owned();
    model.name = "Claude Opus 4.6".to_owned();

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["output_config"],
        json!({ "effort": "high" })
    );
}

#[test]
fn bedrock_payload_injects_cache_points_when_application_profile_name_supports_claude_cache() {
    let mut model =
        get_model("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1").expect("model");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile"
        .to_owned();
    model.name = "Claude Sonnet 4.6".to_owned();
    let context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };

    let payload = build_bedrock_payload(&model, &context, BedrockPayloadOptions::default());

    assert_eq!(payload["system"].as_array().map(Vec::len), Some(2));
    assert!(payload["system"][1].get("cachePoint").is_some());
    let messages = payload["messages"].as_array().expect("messages");
    let last_content = messages
        .last()
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .expect("last content");
    assert!(last_content.get("cachePoint").is_some());
}

#[test]
fn bedrock_payload_forwards_request_metadata_only_when_provided() {
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            request_metadata: Some(json!({ "app": "pi-test", "env": "ci" })),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["requestMetadata"],
        json!({ "app": "pi-test", "env": "ci" })
    );

    let default_payload =
        build_bedrock_payload(&model, &bedrock_context(), BedrockPayloadOptions::default());
    assert!(default_payload.get("requestMetadata").is_none());
}

#[test]
fn bedrock_payload_uses_model_name_for_application_profile_fixed_budget_thinking() {
    let mut model = get_model(
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");
    model.id = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile"
        .to_owned();
    model.name = "Claude Sonnet 4.5".to_owned();

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["additionalModelRequestFields"]["thinking"],
        json!({
            "type": "enabled",
            "budget_tokens": 16_384,
            "display": "summarized",
        })
    );
    assert_eq!(
        payload["additionalModelRequestFields"]["anthropic_beta"],
        json!(["interleaved-thinking-2025-05-14"])
    );
}

#[test]
fn bedrock_payload_forwards_simple_inference_config() {
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");

    let payload = build_bedrock_payload(
        &model,
        &bedrock_context(),
        BedrockPayloadOptions {
            max_tokens: Some(12_345),
            temperature: Some(0.4),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["inferenceConfig"],
        json!({
            "maxTokens": 12_345,
            "temperature": 0.4,
        })
    );
}

#[test]
fn bedrock_payload_preserves_image_tool_results_in_converse_messages() {
    let mut model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");
    model.input = vec![InputKind::Text, InputKind::Image];
    let mut assistant = empty_assistant_for_model(&model);
    assistant.stop_reason = StopReason::ToolUse;
    assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "call_image".to_owned(),
        name: "read_image".to_owned(),
        arguments: object(json!({ "path": "circle.png" })),
        thought_signature: None,
    })];
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Read the image.")),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_image".to_owned(),
                tool_name: "read_image".to_owned(),
                content: vec![
                    ToolResultContent::text("A red circle with a diameter of 100 pixels."),
                    ToolResultContent::Image(ImageContent {
                        data: "base64-image".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
        ],
        ..Default::default()
    };

    let payload = build_bedrock_payload(
        &model,
        &context,
        BedrockPayloadOptions {
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["messages"][1]["content"][0]["toolUse"],
        json!({
            "toolUseId": "call_image",
            "name": "read_image",
            "input": { "path": "circle.png" },
        })
    );
    assert_eq!(
        payload["messages"][2]["content"][0]["toolResult"],
        json!({
            "toolUseId": "call_image",
            "content": [
                { "text": "A red circle with a diameter of 100 pixels." },
                {
                    "image": {
                        "format": "png",
                        "source": { "bytes": "base64-image" },
                    },
                },
            ],
            "status": "success",
        })
    );
}

#[tokio::test]
async fn bedrock_stream_events_preserve_blocks_usage_and_stop_reason() {
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_bedrock_converse_stream_events(
        [
            json!({ "messageStart": { "role": "assistant" } }),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "reasoningContent": { "text": "plan ", "signature": "sig-" } },
                },
            }),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "reasoningContent": { "text": "done", "signature": "2" } },
                },
            }),
            json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": { "text": "Answer" },
                },
            }),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 1,
                    "delta": { "text": "." },
                },
            }),
            json!({ "contentBlockStop": { "contentBlockIndex": 1 } }),
            json!({
                "contentBlockStart": {
                    "contentBlockIndex": 2,
                    "start": { "toolUse": { "toolUseId": "tool-1", "name": "lookup" } },
                },
            }),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 2,
                    "delta": { "toolUse": { "input": "{\"city\":\"Sin" } },
                },
            }),
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 2,
                    "delta": { "toolUse": { "input": "gapore\"}" } },
                },
            }),
            json!({ "contentBlockStop": { "contentBlockIndex": 2 } }),
            json!({
                "metadata": {
                    "usage": {
                        "inputTokens": 10,
                        "outputTokens": 5,
                        "cacheReadInputTokens": 4,
                        "cacheWriteInputTokens": 2,
                    },
                },
            }),
            json!({ "messageStop": { "stopReason": "tool_use" } }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process bedrock stream");
    drop(sender);

    assert_eq!(output.stop_reason, StopReason::ToolUse);
    assert_eq!(output.usage.input, 10);
    assert_eq!(output.usage.output, 5);
    assert_eq!(output.usage.cache_read, 4);
    assert_eq!(output.usage.cache_write, 2);
    assert_eq!(output.usage.total_tokens, 21);
    assert_usage_total_matches_components("bedrock stream usage", &output.usage);
    assert!(matches!(
        &output.content[0],
        AssistantContent::Thinking(thinking)
            if thinking.thinking == "plan done"
                && thinking.thinking_signature.as_deref() == Some("sig-2")
    ));
    assert!(matches!(
        &output.content[1],
        AssistantContent::Text(text) if text.text == "Answer."
    ));
    let AssistantContent::ToolCall(tool_call) = &output.content[2] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.id, "tool-1");
    assert_eq!(tool_call.name, "lookup");
    assert_eq!(tool_call.arguments["city"], "Singapore");

    let events = collect_events(stream).await;
    let event_names: Vec<&'static str> = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect();
    assert_eq!(
        event_names,
        vec![
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]
    );
}

#[tokio::test]
async fn bedrock_stream_events_format_exception_as_error_event() {
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    let error = process_bedrock_converse_stream_events(
        [
            json!({ "messageStart": { "role": "assistant" } }),
            json!({ "internalServerException": { "message": "temporary failure" } }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect_err("bedrock exception");
    drop(sender);

    assert_eq!(error, "Internal server error: temporary failure");
    assert_eq!(output.stop_reason, StopReason::Error);
    assert_eq!(
        output.error_message.as_deref(),
        Some("Internal server error: temporary failure")
    );
    let events = collect_events(stream).await;
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            ..
        })
    ));
}

#[test]
fn azure_openai_base_url_normalization_matches_provider_rules() {
    assert_eq!(
        normalize_azure_openai_base_url(
            "https://marc-quicktests-resource.cognitiveservices.azure.com"
        )
        .expect("cognitive services root"),
        "https://marc-quicktests-resource.cognitiveservices.azure.com/openai/v1"
    );
    assert_eq!(
        normalize_azure_openai_base_url("https://my-resource.openai.azure.com")
            .expect("azure openai root"),
        "https://my-resource.openai.azure.com/openai/v1"
    );
    assert_eq!(
        normalize_azure_openai_base_url("https://my-resource.cognitiveservices.azure.com/openai")
            .expect("openai path"),
        "https://my-resource.cognitiveservices.azure.com/openai/v1"
    );
    assert_eq!(
        normalize_azure_openai_base_url(
            "https://my-resource.cognitiveservices.azure.com/openai/v1"
        )
        .expect("openai v1 path"),
        "https://my-resource.cognitiveservices.azure.com/openai/v1"
    );
    assert_eq!(
        normalize_azure_openai_base_url("https://my-proxy.example.com/v1").expect("proxy path"),
        "https://my-proxy.example.com/v1"
    );
    assert_eq!(
        normalize_azure_openai_base_url(
            "https://my-resource.openai.azure.com/openai?api-version=2024-12-01"
        )
        .expect("azure query"),
        "https://my-resource.openai.azure.com/openai/v1"
    );
    assert_eq!(
        normalize_azure_openai_base_url("https://my-proxy.example.com/v1?custom=true")
            .expect("proxy query"),
        "https://my-proxy.example.com/v1?custom=true"
    );
    assert!(
        normalize_azure_openai_base_url("not-a-url")
            .expect_err("invalid")
            .contains("Invalid Azure OpenAI base URL")
    );
}

#[test]
fn azure_openai_config_builds_default_resource_url_from_env() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_RESOURCE_NAME",
        "AZURE_OPENAI_API_VERSION",
    ]);
    set_env("AZURE_OPENAI_RESOURCE_NAME", "my-resource");

    let model = get_model("azure-openai-responses", "gpt-4o-mini").expect("azure model");
    let config =
        resolve_azure_openai_config(&model, AzureOpenAIConfigOptions::default()).expect("config");
    assert_eq!(
        config.base_url,
        "https://my-resource.openai.azure.com/openai/v1"
    );
    assert_eq!(config.api_version, "v1");
}

#[test]
fn azure_openai_config_treats_empty_resource_and_api_version_as_unset() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_RESOURCE_NAME",
        "AZURE_OPENAI_API_VERSION",
    ]);
    set_env("AZURE_OPENAI_RESOURCE_NAME", "env-resource");
    set_env("AZURE_OPENAI_API_VERSION", "");

    let model = get_model("azure-openai-responses", "gpt-4o-mini").expect("azure model");
    let config = resolve_azure_openai_config(
        &model,
        AzureOpenAIConfigOptions {
            azure_resource_name: Some(String::new()),
            azure_api_version: Some(String::new()),
            ..Default::default()
        },
    )
    .expect("config");

    assert_eq!(
        config.base_url,
        "https://env-resource.openai.azure.com/openai/v1"
    );
    assert_eq!(config.api_version, "v1");
}

#[test]
fn azure_openai_deployment_name_prefers_option_env_map_then_model_id() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["AZURE_OPENAI_DEPLOYMENT_NAME_MAP"]);
    let model = get_model("azure-openai-responses", "gpt-4o-mini").expect("azure model");

    assert_eq!(
        resolve_azure_openai_deployment_name(&model, Some("option-deployment")),
        "option-deployment"
    );

    set_env(
        "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
        "ignored,gpt-4o-mini=prod-mini, other = other-prod, missing=",
    );
    assert_eq!(
        parse_azure_openai_deployment_name_map(Some(
            "ignored,gpt-4o-mini=prod-mini, other = other-prod, missing="
        ))
        .get("other")
        .map(String::as_str),
        Some("other-prod")
    );
    assert_eq!(
        parse_azure_openai_deployment_name_map(Some("gpt-4o-mini=prod-mini=ignored"))
            .get("gpt-4o-mini")
            .map(String::as_str),
        Some("prod-mini")
    );
    assert_eq!(
        resolve_azure_openai_deployment_name(&model, None),
        "prod-mini"
    );

    let mut unmapped = model.clone();
    unmapped.id = "unmapped-model".to_owned();
    assert_eq!(
        resolve_azure_openai_deployment_name(&unmapped, None),
        "unmapped-model"
    );
}

#[test]
fn azure_openai_responses_payload_uses_deployment_tools_session_and_reasoning() {
    let mut model = get_model("azure-openai-responses", "gpt-4o-mini").expect("azure model");
    model.reasoning = true;
    model
        .thinking_level_map
        .insert(ThinkingLevel::High, Some("high".to_owned()));
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Search docs."))],
        tools: vec![Tool {
            name: "lookup".to_owned(),
            description: "Lookup docs".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"],
            }),
        }],
        ..Default::default()
    };

    let payload = build_azure_openai_responses_payload(
        &model,
        &context,
        AzureOpenAIResponsesPayloadOptions {
            session_id: Some("session-1".to_owned()),
            max_tokens: Some(123),
            temperature: Some(0.2),
            reasoning_effort: Some(ThinkingLevel::High),
            reasoning_summary: Some("detailed".to_owned()),
            azure_deployment_name: Some("prod-mini".to_owned()),
        },
    );

    assert_eq!(payload["model"], "prod-mini");
    assert_eq!(payload["stream"], true);
    assert!(payload.get("store").is_none());
    assert_eq!(payload["prompt_cache_key"], "session-1");
    assert_eq!(payload["max_output_tokens"], 123);
    assert_eq!(payload["temperature"], 0.2);
    assert_eq!(
        payload["tools"][0],
        json!({
            "type": "function",
            "name": "lookup",
            "description": "Lookup docs",
            "parameters": {
                "type": "object",
                "properties": { "q": { "type": "string" } },
                "required": ["q"],
            },
            "strict": false,
        })
    );
    assert_eq!(
        payload["reasoning"],
        json!({ "effort": "high", "summary": "detailed" })
    );
    assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn azure_openai_responses_payload_omits_zero_max_tokens_and_defaults_empty_reasoning_summary() {
    let mut model = get_model("azure-openai-responses", "gpt-4o-mini").expect("azure model");
    model.reasoning = true;
    model
        .thinking_level_map
        .insert(ThinkingLevel::High, Some("high".to_owned()));
    let payload = build_azure_openai_responses_payload(
        &model,
        &user_context("hello"),
        AzureOpenAIResponsesPayloadOptions {
            max_tokens: Some(0),
            reasoning_effort: Some(ThinkingLevel::High),
            reasoning_summary: Some(String::new()),
            ..Default::default()
        },
    );

    assert!(payload.get("max_output_tokens").is_none());
    assert_eq!(
        payload["reasoning"],
        json!({ "effort": "high", "summary": "auto" })
    );
    assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn google_vertex_client_config_resolves_api_keys_adc_and_custom_base_urls() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]);

    let model = get_model("google-vertex", "gemini-3-flash-preview").expect("vertex model");
    let adc_config = resolve_google_vertex_client_config(
        &model,
        GoogleVertexOptions {
            api_key: Some("<authenticated>".to_owned()),
            project: Some("test-project".to_owned()),
            location: Some("us-central1".to_owned()),
            ..Default::default()
        },
    )
    .expect("adc config");
    assert_eq!(adc_config.api_key, None);
    assert_eq!(adc_config.project.as_deref(), Some("test-project"));
    assert_eq!(adc_config.location.as_deref(), Some("us-central1"));
    assert_eq!(adc_config.api_version, "v1");
    assert_eq!(adc_config.http_options, None);

    let marker_config = resolve_google_vertex_client_config(
        &model,
        GoogleVertexOptions {
            api_key: Some("gcp-vertex-credentials".to_owned()),
            project: Some("test-project".to_owned()),
            location: Some("us-central1".to_owned()),
            ..Default::default()
        },
    )
    .expect("marker config");
    assert_eq!(marker_config.api_key, None);

    set_env("GOOGLE_CLOUD_API_KEY", "<authenticated>");
    let env_placeholder_config = resolve_google_vertex_client_config(
        &model,
        GoogleVertexOptions {
            project: Some("test-project".to_owned()),
            location: Some("us-central1".to_owned()),
            ..Default::default()
        },
    )
    .expect("env placeholder config");
    assert_eq!(env_placeholder_config.api_key, None);
    remove_env("GOOGLE_CLOUD_API_KEY");

    set_env("GOOGLE_CLOUD_PROJECT", "");
    set_env("GCLOUD_PROJECT", "fallback-project");
    set_env("GOOGLE_CLOUD_LOCATION", "europe-west4");
    let env_fallback_config =
        resolve_google_vertex_client_config(&model, GoogleVertexOptions::default())
            .expect("env fallback config");
    assert_eq!(
        env_fallback_config.project.as_deref(),
        Some("fallback-project")
    );
    assert_eq!(
        env_fallback_config.location.as_deref(),
        Some("europe-west4")
    );
    remove_env("GOOGLE_CLOUD_PROJECT");
    remove_env("GCLOUD_PROJECT");
    remove_env("GOOGLE_CLOUD_LOCATION");

    let internal_gt_api_key_config = resolve_google_vertex_client_config(
        &model,
        GoogleVertexOptions {
            api_key: Some("<invalid>placeholder>".to_owned()),
            ..Default::default()
        },
    )
    .expect("internal > is not a placeholder");
    assert_eq!(
        internal_gt_api_key_config.api_key.as_deref(),
        Some("<invalid>placeholder>")
    );

    let api_key_config = resolve_google_vertex_client_config(
        &model,
        GoogleVertexOptions {
            api_key: Some("AIzaSyExampleRealisticLookingApiKey123456".to_owned()),
            ..Default::default()
        },
    )
    .expect("api key config");
    assert_eq!(
        api_key_config.api_key.as_deref(),
        Some("AIzaSyExampleRealisticLookingApiKey123456")
    );
    assert_eq!(api_key_config.project, None);
    assert_eq!(api_key_config.location, None);

    let mut custom_model = model.clone();
    custom_model.base_url = "https://proxy.example.com".to_owned();
    let custom_adc = resolve_google_vertex_client_config(
        &custom_model,
        GoogleVertexOptions {
            project: Some("test-project".to_owned()),
            location: Some("us-central1".to_owned()),
            ..Default::default()
        },
    )
    .expect("custom adc");
    let http_options = custom_adc.http_options.expect("http options");
    assert_eq!(http_options.base_url, "https://proxy.example.com");
    assert_eq!(http_options.base_url_resource_scope, "COLLECTION");
    assert_eq!(http_options.api_version, None);

    custom_model.base_url =
        "https://proxy.example.com/v1/projects/test-project/locations/global".to_owned();
    let custom_versioned = resolve_google_vertex_client_config(
        &custom_model,
        GoogleVertexOptions {
            project: Some("test-project".to_owned()),
            location: Some("us-central1".to_owned()),
            ..Default::default()
        },
    )
    .expect("custom versioned");
    assert_eq!(
        custom_versioned
            .http_options
            .expect("http options")
            .api_version
            .as_deref(),
        Some("")
    );
}

#[test]
fn google_vertex_client_config_forwards_custom_base_url_to_api_key_client() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["GOOGLE_CLOUD_API_KEY"]);

    let mut model = get_model("google-vertex", "gemini-3-flash-preview").expect("vertex model");
    model.base_url = "https://proxy.example.com".to_owned();
    let config = resolve_google_vertex_client_config(
        &model,
        GoogleVertexOptions {
            api_key: Some("AIzaSyExampleRealisticLookingApiKey123456".to_owned()),
            ..Default::default()
        },
    )
    .expect("api key config");

    assert_eq!(
        config.api_key.as_deref(),
        Some("AIzaSyExampleRealisticLookingApiKey123456")
    );
    assert_eq!(config.project, None);
    assert_eq!(config.location, None);
    let http_options = config.http_options.expect("http options");
    assert_eq!(http_options.base_url, "https://proxy.example.com");
    assert_eq!(http_options.base_url_resource_scope, "COLLECTION");
    assert_eq!(http_options.api_version, None);
}

#[test]
fn google_vertex_env_api_key_marker_requires_existing_adc_project_and_location() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "APPDATA",
        "HOME",
        "USERPROFILE",
    ]);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ri-google-vertex-env-{unique}"));
    std::fs::create_dir_all(&dir).expect("create vertex env temp dir");
    let explicit_adc_path = dir.join("adc.json");

    set_env(
        "GOOGLE_APPLICATION_CREDENTIALS",
        explicit_adc_path.to_str().expect("adc path"),
    );
    set_env("GOOGLE_CLOUD_PROJECT", "test-project");
    set_env("GOOGLE_CLOUD_LOCATION", "us-central1");
    assert_eq!(get_env_api_key("google-vertex"), None);

    std::fs::write(&explicit_adc_path, "{}").expect("write explicit adc");
    assert_eq!(
        get_env_api_key("google-vertex"),
        Some("<authenticated>".to_owned())
    );

    remove_env("GOOGLE_APPLICATION_CREDENTIALS");
    set_env("GOOGLE_CLOUD_PROJECT", "");
    assert_eq!(get_env_api_key("google-vertex"), None);

    set_env("GCLOUD_PROJECT", "fallback-project");
    let default_adc_path = dir
        .join(".config")
        .join("gcloud")
        .join("application_default_credentials.json");
    std::fs::create_dir_all(default_adc_path.parent().expect("adc parent"))
        .expect("create default adc parent");
    std::fs::write(&default_adc_path, "{}").expect("write default adc");
    set_env("HOME", dir.to_str().expect("home path"));
    assert_eq!(
        get_env_api_key("google-vertex"),
        Some("<authenticated>".to_owned())
    );

    set_env("GOOGLE_CLOUD_LOCATION", "");
    assert_eq!(get_env_api_key("google-vertex"), None);

    let _ = std::fs::remove_dir_all(dir);
}

fn google_tool(parameters: Value) -> Tool {
    Tool {
        name: "test_tool".to_owned(),
        description: "A test tool".to_owned(),
        parameters,
    }
}

#[test]
fn google_convert_tools_strips_schema_meta_keys_for_parameters() {
    let tools = vec![google_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "urn:bash-tool",
        "$comment": "A bash tool for demonstration",
        "$defs": {
            "commandDef": { "type": "string" },
        },
        "definitions": {
            "legacyDef": { "type": "number" },
        },
        "type": "object",
        "properties": {
            "command": { "type": "string" },
        },
        "required": ["command"],
    }))];

    let result = convert_google_tools(&tools, true).expect("tools");
    let parameters = &result[0]["functionDeclarations"][0]["parameters"];
    assert_eq!(
        *parameters,
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
            },
            "required": ["command"],
        })
    );
    let object = parameters.as_object().expect("parameters object");
    for key in ["$schema", "$id", "$comment", "$defs", "definitions"] {
        assert!(!object.contains_key(key));
    }
}

#[test]
fn google_convert_tools_recursively_strips_nested_schema_meta_keys() {
    let tools = vec![google_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "deep": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$id": "urn:nested",
                "type": "string",
            },
        },
    }))];

    let result = convert_google_tools(&tools, true).expect("tools");
    assert_eq!(
        result[0]["functionDeclarations"][0]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "deep": {
                    "type": "string",
                },
            },
        })
    );
}

#[test]
fn google_convert_tools_preserves_ref_when_stripping_meta_keys() {
    let tools = vec![google_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "refProp": {
                "$ref": "#/$defs/someDef",
                "type": "string",
            },
        },
    }))];

    let result = convert_google_tools(&tools, true).expect("tools");
    assert_eq!(
        result[0]["functionDeclarations"][0]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "refProp": {
                    "$ref": "#/$defs/someDef",
                    "type": "string",
                },
            },
        })
    );
}

#[test]
fn google_convert_tools_does_not_mutate_original_parameters() {
    let original = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "command": { "type": "string" },
        },
        "required": ["command"],
    });
    let tools = vec![google_tool(original.clone())];

    convert_google_tools(&tools, true).expect("tools");

    assert_eq!(tools[0].parameters, original);
}

#[test]
fn google_convert_tools_preserves_schema_for_parameters_json_schema() {
    let tools = vec![google_tool(json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "command": { "type": "string" },
        },
        "required": ["command"],
    }))];

    let result = convert_google_tools(&tools, false).expect("tools");
    assert_eq!(
        result[0]["functionDeclarations"][0]["parametersJsonSchema"],
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "command": { "type": "string" },
            },
            "required": ["command"],
        })
    );
}

#[test]
fn google_convert_tools_handles_tools_without_schema_meta() {
    let tools = vec![google_tool(json!({
        "type": "object",
        "properties": {
            "path": { "type": "string" },
        },
        "required": ["path"],
    }))];

    let result = convert_google_tools(&tools, true).expect("tools");
    assert_eq!(
        result[0]["functionDeclarations"][0]["parameters"],
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
            },
            "required": ["path"],
        })
    );
}

#[test]
fn google_convert_tools_returns_none_for_empty_tools() {
    assert_eq!(convert_google_tools(&[], false), None);
    assert_eq!(convert_google_tools(&[], true), None);
}

fn google_test_model(api: &str, provider: &str, id: &str, input: Vec<InputKind>) -> Model {
    Model {
        id: id.to_owned(),
        name: id.to_owned(),
        api: api.to_owned(),
        provider: provider.to_owned(),
        base_url: "https://example.com".to_owned(),
        reasoning: true,
        thinking_level_map: BTreeMap::new(),
        input,
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 8_192,
        headers: BTreeMap::new(),
        compat: None,
    }
}

fn google_assistant(model: &Model, content: Vec<AssistantContent>) -> AssistantMessage {
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
        timestamp: 1,
    }
}

fn google_image_tool_context(model: &Model) -> Context {
    Context {
        messages: vec![
            Message::User(UserMessage::text("read the files")),
            Message::Assistant(google_assistant(
                model,
                vec![
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_a".to_owned(),
                        name: "read".to_owned(),
                        arguments: object(json!({ "path": "a.txt" })),
                        thought_signature: None,
                    }),
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_img".to_owned(),
                        name: "read".to_owned(),
                        arguments: object(json!({ "path": "image.png" })),
                        thought_signature: None,
                    }),
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_b".to_owned(),
                        name: "read".to_owned(),
                        arguments: object(json!({ "path": "b.txt" })),
                        thought_signature: None,
                    }),
                ],
            )),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_a".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![ToolResultContent::text("alpha text")],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_img".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![ToolResultContent::Image(ImageContent {
                    data: "abc".to_owned(),
                    mime_type: "image/png".to_owned(),
                })],
                details: None,
                is_error: false,
                timestamp: 3,
            }),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_b".to_owned(),
                tool_name: "read".to_owned(),
                content: vec![ToolResultContent::text("beta text")],
                details: None,
                is_error: false,
                timestamp: 4,
            }),
        ],
        ..Default::default()
    }
}

fn google_unsigned_tool_context(model: &Model, thought_signature: Option<&str>) -> Context {
    Context {
        messages: vec![
            Message::User(UserMessage::text("Hi")),
            Message::Assistant(google_assistant(
                model,
                vec![
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_1".to_owned(),
                        name: "bash".to_owned(),
                        arguments: object(json!({ "command": "echo hi" })),
                        thought_signature: thought_signature.map(ToOwned::to_owned),
                    }),
                    AssistantContent::ToolCall(ToolCall {
                        id: "call_2".to_owned(),
                        name: "bash".to_owned(),
                        arguments: object(json!({ "command": "ls -la" })),
                        thought_signature: None,
                    }),
                ],
            )),
        ],
        ..Default::default()
    }
}

#[test]
fn google_thinking_detection_uses_explicit_thought_marker_only() {
    assert!(is_google_thinking_part(Some(true), None));
    assert!(is_google_thinking_part(
        Some(true),
        Some("opaque-signature")
    ));
    assert!(!is_google_thinking_part(None, Some("opaque-signature")));
    assert!(!is_google_thinking_part(
        Some(false),
        Some("opaque-signature")
    ));
    assert!(!is_google_thinking_part(None, None));
    assert!(!is_google_thinking_part(Some(false), Some("")));
}

#[test]
fn google_simple_payload_disables_thinking_for_gemini_reasoning_models() {
    let context = Context {
        system_prompt: Some("system prompt".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };

    for (model_id, expected) in [
        ("gemini-2.5-flash", json!({ "thinkingBudget": 0 })),
        (
            "gemini-3-flash-preview",
            json!({ "thinkingLevel": "MINIMAL" }),
        ),
        ("gemini-3.1-pro-preview", json!({ "thinkingLevel": "LOW" })),
    ] {
        let model = get_model("google", model_id).expect("google model");
        let payload = build_google_simple_payload(&model, &context, SimpleStreamOptions::default());
        assert_eq!(payload["model"], model_id);
        assert_eq!(payload["contents"][0]["parts"][0]["text"], "Hello");
        assert_eq!(payload["config"]["systemInstruction"], "system prompt");
        assert_eq!(payload["config"]["thinkingConfig"], expected);
    }
}

#[test]
fn google_simple_payload_maps_reasoning_to_budget_or_level() {
    let context = user_context("Think briefly.");
    let flash = get_model("google", "gemini-2.5-flash").expect("flash");
    let flash_payload = build_google_simple_payload(
        &flash,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );
    assert_eq!(
        flash_payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": 24576 })
    );

    let custom_payload = build_google_simple_payload(
        &flash,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::High),
            thinking_budgets: Some(ThinkingBudgets {
                high: Some(1234),
                ..Default::default()
            }),
            ..Default::default()
        },
    );
    assert_eq!(
        custom_payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": 1234 })
    );

    let flash_lite = get_model("google", "gemini-2.5-flash-lite").expect("flash lite");
    let flash_lite_payload = build_google_simple_payload(
        &flash_lite,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Minimal),
            ..Default::default()
        },
    );
    assert_eq!(
        flash_lite_payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": 512 })
    );

    let vertex_flash_lite =
        get_model("google-vertex", "gemini-2.5-flash-lite").expect("vertex flash lite");
    let vertex_flash_lite_payload = build_google_simple_payload(
        &vertex_flash_lite,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Minimal),
            ..Default::default()
        },
    );
    assert_eq!(
        vertex_flash_lite_payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingBudget": 128 })
    );

    let pro = get_model("google", "gemini-3.1-pro-preview").expect("gemini 3 pro");
    let pro_payload = build_google_simple_payload(
        &pro,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Low),
            ..Default::default()
        },
    );
    assert_eq!(
        pro_payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingLevel": "LOW" })
    );

    let flash3 = get_model("google", "gemini-3-flash-preview").expect("gemini 3 flash");
    let flash3_payload = build_google_simple_payload(
        &flash3,
        &context,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(
        flash3_payload["config"]["thinkingConfig"],
        json!({ "includeThoughts": true, "thinkingLevel": "MEDIUM" })
    );
}

#[test]
fn google_retain_thought_signature_preserves_and_updates_non_empty_values() {
    let first = retain_google_thought_signature(None, Some("sig-1"));
    assert_eq!(first.as_deref(), Some("sig-1"));

    let second = retain_google_thought_signature(first.as_deref(), None);
    assert_eq!(second.as_deref(), Some("sig-1"));

    let third = retain_google_thought_signature(second.as_deref(), Some(""));
    assert_eq!(third.as_deref(), Some("sig-1"));

    let updated = retain_google_thought_signature(third.as_deref(), Some("sig-2"));
    assert_eq!(updated.as_deref(), Some("sig-2"));
}

#[tokio::test]
async fn google_stream_chunks_preserve_response_id_signatures_usage_and_tool_calls() {
    let model = get_model("google", "gemini-2.5-flash").expect("google model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_google_stream_chunks(
        [
            json!({
                "responseId": "",
                "candidates": [{
                    "content": {
                        "parts": [
                            { "text": "plan ", "thought": true, "thoughtSignature": "think-sig" },
                            { "text": "next", "thought": true },
                        ],
                    },
                }],
            }),
            json!({
                "responseId": "google-response-1",
                "candidates": [{
                    "content": {
                        "parts": [
                            { "text": "Answer", "thoughtSignature": "text-sig" },
                            {
                                "functionCall": {
                                    "id": "call-1",
                                    "name": "lookup",
                                    "args": { "q": "rust" },
                                },
                                "thoughtSignature": "tool-sig",
                            },
                        ],
                    },
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "cachedContentTokenCount": 4,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 15,
                },
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process google chunks");
    drop(sender);

    assert_eq!(output.response_id.as_deref(), Some("google-response-1"));
    assert_eq!(output.stop_reason, StopReason::ToolUse);
    assert_eq!(output.usage.input, 6);
    assert_eq!(output.usage.output, 5);
    assert_eq!(output.usage.cache_read, 4);
    assert_eq!(output.usage.total_tokens, 15);
    assert_usage_total_matches_components("google stream usage", &output.usage);
    assert!(matches!(
        &output.content[0],
        AssistantContent::Thinking(thinking)
            if thinking.thinking == "plan next"
                && thinking.thinking_signature.as_deref() == Some("think-sig")
    ));
    assert!(matches!(
        &output.content[1],
        AssistantContent::Text(text)
            if text.text == "Answer"
                && text.text_signature.as_ref().and_then(Value::as_str) == Some("text-sig")
    ));
    let AssistantContent::ToolCall(tool_call) = &output.content[2] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.id, "call-1");
    assert_eq!(tool_call.name, "lookup");
    assert_eq!(tool_call.arguments["q"], "rust");
    assert_eq!(tool_call.thought_signature.as_deref(), Some("tool-sig"));

    let events = collect_events(stream).await;
    let event_names: Vec<&'static str> = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect();
    assert_eq!(
        event_names,
        vec![
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
            "done",
        ]
    );
}

#[tokio::test]
async fn google_stream_chunks_map_safety_finish_to_error_event() {
    let model = get_model("google", "gemini-2.5-flash").expect("google model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    let error = process_google_stream_chunks(
        [json!({
            "candidates": [{ "finishReason": "SAFETY" }],
        })],
        &mut output,
        &sender,
        &model,
    )
    .expect_err("safety finish");
    drop(sender);

    assert_eq!(error, "An unknown error occurred");
    assert_eq!(output.stop_reason, StopReason::Error);
    assert_eq!(
        output.error_message.as_deref(),
        Some("An unknown error occurred")
    );
    let events = collect_events(stream).await;
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            ..
        })
    ));
}

#[test]
fn google_convert_messages_keeps_separate_image_turn_for_gemini_2() {
    let model = google_test_model(
        "google-generative-ai",
        "google",
        "gemini-2.5-flash",
        vec![InputKind::Text, InputKind::Image],
    );
    let contents = convert_google_messages(&model, &google_image_tool_context(&model));

    assert_eq!(contents.len(), 5);
    assert!(
        contents[2]["parts"]
            .as_array()
            .expect("function responses")
            .iter()
            .all(|part| part.get("functionResponse").is_some())
    );
    assert_eq!(contents[3]["parts"][0]["text"], "Tool result image:");
    assert!(contents[3]["parts"][1].get("inlineData").is_some());
    assert!(contents[4]["parts"][0].get("functionResponse").is_some());
}

#[test]
fn google_convert_messages_nests_image_tool_results_for_gemini_3() {
    let model = google_test_model(
        "google-generative-ai",
        "google",
        "gemini-3-pro-preview",
        vec![InputKind::Text, InputKind::Image],
    );
    let contents = convert_google_messages(&model, &google_image_tool_context(&model));

    assert_eq!(contents.len(), 3);
    let tool_result_parts = contents[2]["parts"].as_array().expect("tool result parts");
    assert_eq!(tool_result_parts.len(), 3);
    let image_response = &tool_result_parts[1]["functionResponse"];
    assert!(image_response.is_object());
    assert_eq!(image_response["parts"].as_array().map(Vec::len), Some(1));
    assert!(image_response["parts"][0].get("inlineData").is_some());
}

#[test]
fn google_convert_messages_omits_validator_marker_for_unsigned_gemini_3_tool_calls() {
    let model = google_test_model(
        "google-generative-ai",
        "google",
        "gemini-3-pro-preview",
        vec![InputKind::Text],
    );
    let mut source_model = model.clone();
    source_model.id = "other-model".to_owned();
    let contents =
        convert_google_messages(&model, &google_unsigned_tool_context(&source_model, None));
    let model_turn = contents
        .iter()
        .find(|content| content["role"] == "model")
        .expect("model turn");
    let parts = model_turn["parts"].as_array().expect("parts");
    let function_call_parts = parts
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect::<Vec<_>>();

    assert_eq!(function_call_parts.len(), 2);
    assert!(function_call_parts[0].get("thoughtSignature").is_none());
    assert!(function_call_parts[1].get("thoughtSignature").is_none());
    assert!(
        !serde_json::to_string(model_turn)
            .expect("model JSON")
            .contains("skip_thought_signature_validator")
    );
    assert!(!parts.iter().any(|part| {
        part.get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains("Historical context"))
    }));
}

#[test]
fn google_convert_messages_omits_validator_marker_for_unsigned_vertex_tool_calls() {
    let model = google_test_model(
        "google-vertex",
        "google-vertex",
        "gemini-3-pro-preview",
        vec![InputKind::Text],
    );
    let contents = convert_google_messages(&model, &google_unsigned_tool_context(&model, None));
    let model_turn = contents
        .iter()
        .find(|content| content["role"] == "model")
        .expect("model turn");
    let function_call_parts = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect::<Vec<_>>();

    assert_eq!(function_call_parts.len(), 2);
    assert!(function_call_parts[0].get("thoughtSignature").is_none());
    assert!(function_call_parts[1].get("thoughtSignature").is_none());
    assert!(
        !serde_json::to_string(model_turn)
            .expect("model JSON")
            .contains("skip_thought_signature_validator")
    );
}

#[test]
fn google_convert_messages_preserves_valid_same_model_thought_signature() {
    let model = google_test_model(
        "google-generative-ai",
        "google",
        "gemini-3-pro-preview",
        vec![InputKind::Text],
    );
    let valid_signature = "AAAAAAAAAAAAAAAAAAAAAA==";
    let contents = convert_google_messages(
        &model,
        &google_unsigned_tool_context(&model, Some(valid_signature)),
    );
    let model_turn = contents
        .iter()
        .find(|content| content["role"] == "model")
        .expect("model turn");
    let function_call_parts = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .filter(|part| part.get("functionCall").is_some())
        .collect::<Vec<_>>();

    assert_eq!(function_call_parts.len(), 2);
    assert_eq!(
        function_call_parts[0]
            .get("thoughtSignature")
            .and_then(Value::as_str),
        Some(valid_signature)
    );
    assert!(function_call_parts[1].get("thoughtSignature").is_none());
}

#[test]
fn google_convert_messages_drops_invalid_same_model_thought_signatures() {
    let model = google_test_model(
        "google-generative-ai",
        "google",
        "gemini-3-pro-preview",
        vec![InputKind::Text],
    );
    for invalid_signature in ["AA=A", "AAAA====", "===="] {
        let contents = convert_google_messages(
            &model,
            &google_unsigned_tool_context(&model, Some(invalid_signature)),
        );
        let model_turn = contents
            .iter()
            .find(|content| content["role"] == "model")
            .expect("model turn");
        let function_call_part = model_turn["parts"]
            .as_array()
            .expect("parts")
            .iter()
            .find(|part| part.get("functionCall").is_some())
            .expect("function call");

        assert!(
            function_call_part.get("thoughtSignature").is_none(),
            "{invalid_signature} must be rejected like the source base64 validator"
        );
    }
}

#[test]
fn google_convert_messages_does_not_add_thought_signature_for_non_gemini_3_models() {
    let model = google_test_model(
        "google-generative-ai",
        "google",
        "gemini-2.5-flash",
        vec![InputKind::Text],
    );
    let mut source_model = model.clone();
    source_model.id = "other-model".to_owned();
    let contents =
        convert_google_messages(&model, &google_unsigned_tool_context(&source_model, None));
    let model_turn = contents
        .iter()
        .find(|content| content["role"] == "model")
        .expect("model turn");
    let function_call_part = model_turn["parts"]
        .as_array()
        .expect("parts")
        .iter()
        .find(|part| part.get("functionCall").is_some())
        .expect("function call");

    assert!(function_call_part.get("thoughtSignature").is_none());
}

#[test]
fn message_transform_normalizes_cross_provider_tool_call_ids() {
    let target_model = get_model("openrouter", "openai/gpt-5.2-codex").expect("openrouter target");
    let failing_id = "call_pAYbIr76hXIjncD9UE4eGfnS|t5nnb2qYMFWGSsr13fhCd1CaCu3t3qONEPuOudu4HSVEtA8YJSL6FAZUxvoOoD792VIJWl91g87EdqsCWp9krVsdBysQoDaf9lMCLb8BS4EYi4gQd5kBQBYLlgD71PYwvf+TbMD9J9/5OMD42oxSRj8H+vRf78/l2Xla33LWz4nOgsddBlbvabICRs8GHt5C9PK5keFtzyi3lsyVKNlfduK3iphsZqs4MLv4zyGJnvZo/+QzShyk5xnMSQX/f98+aEoNflEApCdEOXipipgeiNWnpFSHbcwmMkZoJhURNu+JEz3xCh1mrXeYoN5o+trLL3IXJacSsLYXDrYTipZZbJFRPAucgbnjYBC+/ZzJOfkwCs+Gkw7EoZR7ZQgJ8ma+9586n4tT4cI8DEhBSZsWMjrCt8dxKg==";
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: failing_id.to_owned(),
            name: "echo".to_owned(),
            arguments: object(json!({ "message": "hello" })),
            thought_signature: Some("source-only".to_owned()),
        })],
        api: "openai-responses".to_owned(),
        provider: "github-copilot".to_owned(),
        model: "gpt-5.2-codex".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let messages = vec![
        Message::User(UserMessage::text("Use echo")),
        Message::Assistant(assistant),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: failing_id.to_owned(),
            tool_name: "echo".to_owned(),
            content: vec![ToolResultContent::text("hello")],
            details: None,
            is_error: false,
            timestamp: 2,
        }),
    ];

    let transformed = transform_messages(
        &messages,
        &target_model,
        Some(&|id, model, _source| normalize_openai_completions_tool_call_id(id, model)),
    );
    let Message::Assistant(assistant) = &transformed[1] else {
        panic!("assistant");
    };
    let AssistantContent::ToolCall(tool_call) = &assistant.content[0] else {
        panic!("tool call");
    };

    assert_eq!(tool_call.id, "call_pAYbIr76hXIjncD9UE4eGfnS");
    assert!(tool_call.id.len() <= 40);
    assert_eq!(tool_call.thought_signature, None);

    let Message::ToolResult(tool_result) = &transformed[2] else {
        panic!("tool result");
    };
    assert_eq!(tool_result.tool_call_id, tool_call.id);
}

#[test]
fn message_transform_copilot_openai_to_anthropic_downgrades_thinking_and_signatures() {
    let target_model =
        get_model("github-copilot", "claude-sonnet-4.6").expect("copilot claude target");
    let anthropic_normalize = |id: &str, _model: &Model, _source: &AssistantMessage| {
        id.chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(64)
            .collect::<String>()
    };
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Let me think about this...".to_owned(),
                thinking_signature: Some("reasoning_content".to_owned()),
                redacted: false,
            }),
            AssistantContent::Text(TextContent::new("Hi there!")),
            AssistantContent::ToolCall(ToolCall {
                id: "call_123".to_owned(),
                name: "bash".to_owned(),
                arguments: object(json!({ "command": "ls" })),
                thought_signature: Some(
                    json!({
                        "type": "reasoning.encrypted",
                        "id": "call_123",
                        "data": "encrypted",
                    })
                    .to_string(),
                ),
            }),
        ],
        api: "openai-responses".to_owned(),
        provider: "github-copilot".to_owned(),
        model: "gpt-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let messages = vec![
        Message::User(UserMessage::text("run a command")),
        Message::Assistant(assistant),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_123".to_owned(),
            tool_name: "bash".to_owned(),
            content: vec![ToolResultContent::text("output")],
            details: None,
            is_error: false,
            timestamp: 2,
        }),
    ];

    let transformed = transform_messages(&messages, &target_model, Some(&anthropic_normalize));
    let Message::Assistant(assistant) = &transformed[1] else {
        panic!("assistant");
    };
    let text_blocks = assistant
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(text_blocks, vec!["Let me think about this...", "Hi there!"]);
    assert!(
        !assistant
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Thinking(_)))
    );
    let tool_call = assistant
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .expect("tool call");
    assert_eq!(tool_call.thought_signature, None);
}

#[test]
fn message_transform_same_model_keeps_only_truthy_empty_thinking_signatures() {
    let target_model = Model::faux("openai-responses", "openai", "gpt-5");
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some(String::new()),
                redacted: false,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: String::new(),
                thinking_signature: Some("reasoning".to_owned()),
                redacted: false,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "visible".to_owned(),
                thinking_signature: Some(String::new()),
                redacted: false,
            }),
        ],
        api: target_model.api.clone(),
        provider: target_model.provider.clone(),
        model: target_model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 1,
    };

    let transformed = transform_messages(&[Message::Assistant(assistant)], &target_model, None);
    let Message::Assistant(assistant) = &transformed[0] else {
        panic!("assistant");
    };
    let thinking = assistant
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Thinking(thinking) => Some((
                thinking.thinking.as_str(),
                thinking.thinking_signature.as_deref(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(
        thinking,
        vec![("", Some("reasoning")), ("visible", Some(""))]
    );
}

#[test]
fn message_transform_copilot_openai_to_anthropic_synthesizes_single_orphan_result() {
    let target_model =
        get_model("github-copilot", "claude-sonnet-4.6").expect("copilot claude target");
    let anthropic_normalize = |id: &str, _model: &Model, _source: &AssistantMessage| {
        id.chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(64)
            .collect::<String>()
    };
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "call_123|fc_123".to_owned(),
            name: "read".to_owned(),
            arguments: object(json!({ "path": "README.md" })),
            thought_signature: None,
        })],
        api: "openai-responses".to_owned(),
        provider: "github-copilot".to_owned(),
        model: "gpt-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let messages = vec![
        Message::User(UserMessage::text("read the file")),
        Message::Assistant(assistant),
    ];

    let transformed = transform_messages(&messages, &target_model, Some(&anthropic_normalize));
    let Message::Assistant(assistant) = &transformed[1] else {
        panic!("assistant");
    };
    assert!(matches!(
        assistant.content.first(),
        Some(AssistantContent::ToolCall(tool_call))
            if tool_call.id == "call_123_fc_123" && tool_call.name == "read"
    ));
    let Message::ToolResult(tool_result) = transformed.last().expect("synthetic result") else {
        panic!("synthetic result");
    };
    assert_eq!(tool_result.tool_call_id, "call_123_fc_123");
    assert_eq!(tool_result.tool_name, "read");
    assert!(tool_result.is_error);
    assert!(matches!(
        tool_result.content.first(),
        Some(ToolResultContent::Text(text)) if text.text == "No result provided"
    ));
}

#[test]
fn message_transform_synthesizes_only_missing_trailing_tool_results_after_normalization() {
    let target_model =
        get_model("github-copilot", "claude-sonnet-4.6").expect("copilot claude target");
    let anthropic_normalize = |id: &str, _model: &Model, _source: &AssistantMessage| {
        id.chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(64)
            .collect::<String>()
    };
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(ToolCall {
                id: "call_1|fc_1".to_owned(),
                name: "read".to_owned(),
                arguments: object(json!({ "path": "README.md" })),
                thought_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_2|fc_2".to_owned(),
                name: "bash".to_owned(),
                arguments: object(json!({ "command": "pwd" })),
                thought_signature: None,
            }),
        ],
        api: "openai-responses".to_owned(),
        provider: "github-copilot".to_owned(),
        model: "gpt-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let messages = vec![
        Message::User(UserMessage::text("run commands")),
        Message::Assistant(assistant),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call_1|fc_1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultContent::text("done")],
            details: None,
            is_error: false,
            timestamp: 2,
        }),
    ];

    let transformed = transform_messages(&messages, &target_model, Some(&anthropic_normalize));
    let Message::Assistant(assistant) = &transformed[1] else {
        panic!("assistant");
    };
    let tool_call_ids = assistant
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.id.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(tool_call_ids, vec!["call_1_fc_1", "call_2_fc_2"]);

    let existing_result = transformed
        .iter()
        .find_map(|message| match message {
            Message::ToolResult(tool_result) if !tool_result.is_error => Some(tool_result),
            _ => None,
        })
        .expect("existing tool result");
    assert_eq!(existing_result.tool_call_id, "call_1_fc_1");

    let synthetic_results = transformed
        .iter()
        .filter_map(|message| match message {
            Message::ToolResult(tool_result) if tool_result.is_error => Some(tool_result),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(synthetic_results.len(), 1);
    assert_eq!(synthetic_results[0].tool_call_id, "call_2_fc_2");
    assert_eq!(synthetic_results[0].tool_name, "bash");
    assert!(matches!(
        synthetic_results[0].content.first(),
        Some(ToolResultContent::Text(text)) if text.text == "No result provided"
    ));
}

#[test]
fn message_transform_downgrades_images_thinking_and_orphaned_tool_calls() {
    let mut text_only_model = Model::faux("openai-completions", "openai", "gpt-4o-mini");
    text_only_model.input = vec![InputKind::Text];

    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "visible thinking".to_owned(),
                thinking_signature: None,
                redacted: false,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "secret".to_owned(),
                thinking_signature: None,
                redacted: true,
            }),
            AssistantContent::Thinking(ThinkingContent {
                thinking: "   ".to_owned(),
                thinking_signature: None,
                redacted: false,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "tool-1".to_owned(),
                name: "read".to_owned(),
                arguments: Map::new(),
                thought_signature: None,
            }),
        ],
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        model: "claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };

    let messages = vec![
        Message::User(UserMessage {
            content: UserContentValue::Blocks(vec![
                UserContent::Text(TextContent::new("before")),
                UserContent::Image(ImageContent {
                    data: "image-a".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
                UserContent::Image(ImageContent {
                    data: "image-b".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
                UserContent::Text(TextContent::new("after")),
            ]),
            timestamp: 0,
        }),
        Message::Assistant(assistant),
        Message::User(UserMessage::text("continue")),
    ];

    let transformed = transform_messages(&messages, &text_only_model, None);
    let Message::User(user) = &transformed[0] else {
        panic!("user");
    };
    let UserContentValue::Blocks(blocks) = &user.content else {
        panic!("blocks");
    };
    assert_eq!(blocks.len(), 3);
    assert!(matches!(
        &blocks[1],
        UserContent::Text(text) if text.text == NON_VISION_USER_IMAGE_PLACEHOLDER
    ));

    let Message::Assistant(assistant) = &transformed[1] else {
        panic!("assistant");
    };
    assert_eq!(assistant.content.len(), 2);
    assert!(matches!(
        &assistant.content[0],
        AssistantContent::Text(text) if text.text == "visible thinking"
    ));

    let Message::ToolResult(synthetic) = &transformed[2] else {
        panic!("synthetic tool result");
    };
    assert_eq!(synthetic.tool_call_id, "tool-1");
    assert!(synthetic.is_error);
    assert!(matches!(
        synthetic.content.first(),
        Some(ToolResultContent::Text(text)) if text.text == "No result provided"
    ));
}

fn anthropic_sse_body(events: Vec<(&str, String)>) -> String {
    events
        .into_iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn minimal_anthropic_sse_events() -> Vec<(&'static str, String)> {
    vec![
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            })
            .to_string(),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" },
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Hello" },
            })
            .to_string(),
        ),
        (
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 0 }).to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            })
            .to_string(),
        ),
        (
            "message_stop",
            json!({ "type": "message_stop" }).to_string(),
        ),
    ]
}

#[test]
fn anthropic_sse_parser_repairs_malformed_event_and_streamed_tool_json() {
    let model = get_model("anthropic", "claude-haiku-4-5").expect("anthropic model");
    let malformed_tool_json_delta = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"A\H\",\"text\":\"col1\tcol2\"}"}}"#;
    let body = anthropic_sse_body(vec![
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                },
            })
            .to_string(),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_test",
                    "name": "edit",
                    "input": {},
                },
            })
            .to_string(),
        ),
        ("content_block_delta", malformed_tool_json_delta.to_owned()),
        (
            "content_block_stop",
            json!({ "type": "content_block_stop", "index": 0 }).to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "tool_use" },
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            })
            .to_string(),
        ),
        (
            "message_stop",
            json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    let result = process_anthropic_sse_body(&model, &body).expect("parsed");

    assert_eq!(result.stop_reason, StopReason::ToolUse);
    assert_eq!(result.error_message, None);
    let tool_call = result
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call),
            _ => None,
        })
        .expect("tool call");
    assert_eq!(tool_call.arguments["path"], "A\\H");
    assert_eq!(tool_call.arguments["text"], "col1\tcol2");
}

#[test]
fn anthropic_sse_parser_ignores_unknown_events_after_message_stop() {
    let model = get_model("anthropic", "claude-haiku-4-5").expect("anthropic model");
    let mut events = minimal_anthropic_sse_events();
    events.push(("done", "[DONE]".to_owned()));
    events.push(("proxy.stats", "not json".to_owned()));
    let body = anthropic_sse_body(events);

    let result = process_anthropic_sse_body(&model, &body).expect("parsed");

    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.error_message, None);
    assert_eq!(result.content, vec![AssistantContent::text("Hello")]);
}

#[test]
fn anthropic_sse_parser_preserves_response_id_and_initial_input_usage() {
    let model = get_model("anthropic", "claude-haiku-4-5").expect("anthropic model");
    let body = anthropic_sse_body(vec![
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_response_id",
                    "usage": {
                        "input_tokens": 21,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 3,
                        "cache_creation_input_tokens": 2,
                    },
                },
            })
            .to_string(),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" },
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Hello" },
            })
            .to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": {
                    "output_tokens": 5,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 2,
                },
            })
            .to_string(),
        ),
        (
            "message_stop",
            json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    let result = process_anthropic_sse_body(&model, &body).expect("parsed");

    assert_eq!(result.response_id.as_deref(), Some("msg_response_id"));
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.usage.input, 21);
    assert_eq!(result.usage.output, 5);
    assert_eq!(result.usage.cache_read, 3);
    assert_eq!(result.usage.cache_write, 2);
    assert_usage_total_matches_components("anthropic initial input usage", &result.usage);
}

#[test]
fn anthropic_sse_parser_preserves_start_usage_when_delta_omits_fields() {
    let model = get_model("anthropic", "claude-haiku-4-5").expect("anthropic model");
    let body = anthropic_sse_body(vec![
        (
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_partial_usage",
                    "usage": {
                        "input_tokens": 34,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 8,
                        "cache_creation_input_tokens": 5,
                    },
                },
            })
            .to_string(),
        ),
        (
            "content_block_start",
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" },
            })
            .to_string(),
        ),
        (
            "content_block_delta",
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": { "type": "text_delta", "text": "Preserved" },
            })
            .to_string(),
        ),
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 7 },
            })
            .to_string(),
        ),
        (
            "message_stop",
            json!({ "type": "message_stop" }).to_string(),
        ),
    ]);

    let result = process_anthropic_sse_body(&model, &body).expect("parsed");

    assert_eq!(result.response_id.as_deref(), Some("msg_partial_usage"));
    assert_eq!(result.stop_reason, StopReason::Stop);
    assert_eq!(result.content, vec![AssistantContent::text("Preserved")]);
    assert_eq!(result.usage.input, 34);
    assert_eq!(result.usage.output, 7);
    assert_eq!(result.usage.cache_read, 8);
    assert_eq!(result.usage.cache_write, 5);
    assert_usage_total_matches_components("anthropic partial usage", &result.usage);
}

#[test]
fn anthropic_sse_parser_maps_provider_stop_reason_errors() {
    let model = get_model("anthropic", "claude-haiku-4-5").expect("anthropic model");

    for reason in ["refusal", "sensitive"] {
        let body = anthropic_sse_body(vec![
            (
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": format!("msg_{reason}"),
                        "usage": { "input_tokens": 3, "output_tokens": 0 },
                    },
                })
                .to_string(),
            ),
            (
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": { "stop_reason": reason },
                    "usage": { "output_tokens": 1 },
                })
                .to_string(),
            ),
            (
                "message_stop",
                json!({ "type": "message_stop" }).to_string(),
            ),
        ]);

        let result = process_anthropic_sse_body(&model, &body).expect("parsed error reason");
        let expected_error = format!("Provider stop_reason: {reason}");
        assert_eq!(result.stop_reason, StopReason::Error, "{reason}");
        assert_eq!(
            result.error_message.as_deref(),
            Some(expected_error.as_str())
        );
        assert_eq!(result.usage.input, 3);
        assert_eq!(result.usage.output, 1);
    }

    let pause_body = anthropic_sse_body(vec![
        (
            "message_delta",
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "pause_turn" },
            })
            .to_string(),
        ),
        (
            "message_stop",
            json!({ "type": "message_stop" }).to_string(),
        ),
    ]);
    let pause = process_anthropic_sse_body(&model, &pause_body).expect("pause turn");
    assert_eq!(pause.stop_reason, StopReason::Stop);

    let unknown_body = anthropic_sse_body(vec![(
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": { "stop_reason": "provider_added_reason" },
        })
        .to_string(),
    )]);
    let error = process_anthropic_sse_body(&model, &unknown_body).expect_err("unknown reason");
    assert!(error.contains("Unhandled Anthropic stop_reason: provider_added_reason"));
}

fn anthropic_tool() -> Tool {
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

fn anthropic_test_model(compat: Option<Value>) -> Model {
    Model {
        id: "claude-opus-4-7".to_owned(),
        name: "Claude Opus 4.7".to_owned(),
        api: "anthropic-messages".to_owned(),
        provider: "test-anthropic".to_owned(),
        base_url: "https://example.com".to_owned(),
        reasoning: true,
        thinking_level_map: BTreeMap::from([(ThinkingLevel::XHigh, Some("xhigh".to_owned()))]),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 200_000,
        max_tokens: 32_000,
        headers: BTreeMap::new(),
        compat,
    }
}

fn anthropic_tool_context(tools: Vec<Tool>) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text("Use the tool"))],
        tools,
        ..Default::default()
    }
}

#[test]
fn anthropic_payload_sends_per_tool_eager_input_streaming_by_default() {
    let model = anthropic_test_model(None);
    let context = anthropic_tool_context(vec![anthropic_tool()]);

    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());
    let headers = build_anthropic_default_headers(&model, &context);

    assert_eq!(payload["tools"][0]["eager_input_streaming"], true);
    assert!(headers.get("anthropic-beta").is_none());
}

#[test]
fn anthropic_payload_uses_legacy_fine_grained_tool_streaming_beta_when_eager_disabled() {
    let model = anthropic_test_model(Some(json!({
        "supportsEagerToolInputStreaming": false,
    })));
    let context = anthropic_tool_context(vec![anthropic_tool()]);

    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());
    let headers = build_anthropic_default_headers(&model, &context);

    assert!(payload["tools"][0].get("eager_input_streaming").is_none());
    assert_eq!(
        headers.get("anthropic-beta").map(String::as_str),
        Some("fine-grained-tool-streaming-2025-05-14")
    );
}

#[test]
fn anthropic_payload_omits_fine_grained_beta_when_no_tools() {
    let model = anthropic_test_model(Some(json!({
        "supportsEagerToolInputStreaming": false,
    })));
    let context = anthropic_tool_context(Vec::new());

    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());
    let headers = build_anthropic_default_headers(&model, &context);

    assert!(payload.get("tools").is_none());
    assert!(headers.get("anthropic-beta").is_none());
}

#[test]
fn anthropic_client_config_adds_interleaved_thinking_beta_for_nonadaptive_models() {
    let opus_45 = get_model("anthropic", "claude-opus-4-5").expect("opus 4.5 model");
    let context = anthropic_tool_context(vec![anthropic_tool()]);

    let config = build_anthropic_client_config(
        &opus_45,
        &context,
        AnthropicClientOptions {
            api_key: "anthropic-key".to_owned(),
            ..Default::default()
        },
    );

    assert!(
        config
            .default_headers
            .get("anthropic-beta")
            .is_some_and(|beta| beta.contains("interleaved-thinking-2025-05-14"))
    );

    let without_interleaved = build_anthropic_client_config(
        &opus_45,
        &context,
        AnthropicClientOptions {
            api_key: "anthropic-key".to_owned(),
            interleaved_thinking: false,
            ..Default::default()
        },
    );
    assert!(
        without_interleaved
            .default_headers
            .get("anthropic-beta")
            .map_or(true, |beta| !beta.contains("interleaved-thinking"))
    );

    let opus_46 = get_model("anthropic", "claude-opus-4-6").expect("opus 4.6 model");
    let adaptive_config = build_anthropic_client_config(
        &opus_46,
        &context,
        AnthropicClientOptions {
            api_key: "anthropic-key".to_owned(),
            ..Default::default()
        },
    );
    assert!(
        adaptive_config
            .default_headers
            .get("anthropic-beta")
            .map_or(true, |beta| !beta.contains("interleaved-thinking"))
    );
}

#[test]
fn github_copilot_anthropic_client_config_matches_provider_headers() {
    let model = get_model("github-copilot", "claude-sonnet-4.6").expect("copilot claude model");
    assert_eq!(model.api, "anthropic-messages");
    assert_eq!(model.base_url, "https://api.individual.githubcopilot.com");
    assert_eq!(
        model
            .headers
            .get("Copilot-Integration-Id")
            .map(String::as_str),
        Some("vscode-chat")
    );

    let context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };
    let config = build_anthropic_client_config(
        &model,
        &context,
        AnthropicClientOptions {
            api_key: "tid_copilot_session_test_token".to_owned(),
            ..Default::default()
        },
    );

    assert_eq!(config.api_key, None);
    assert_eq!(
        config.auth_token.as_deref(),
        Some("tid_copilot_session_test_token")
    );
    assert_eq!(config.base_url, "https://api.individual.githubcopilot.com");
    assert!(!config.is_oauth_token);
    assert_eq!(
        config.default_headers.get("accept").map(String::as_str),
        Some("application/json")
    );
    assert!(
        config
            .default_headers
            .get("User-Agent")
            .is_some_and(|value| value.contains("GitHubCopilotChat"))
    );
    assert_eq!(
        config
            .default_headers
            .get("Copilot-Integration-Id")
            .map(String::as_str),
        Some("vscode-chat")
    );
    assert_eq!(
        config
            .default_headers
            .get("X-Initiator")
            .map(String::as_str),
        Some("user")
    );
    assert_eq!(
        config
            .default_headers
            .get("Openai-Intent")
            .map(String::as_str),
        Some("conversation-edits")
    );
    assert!(
        config
            .default_headers
            .get("anthropic-beta")
            .map_or(true, |beta| {
                !beta.contains("fine-grained-tool-streaming")
                    && !beta.contains("interleaved-thinking")
            })
    );

    let image_context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContentValue::Blocks(vec![UserContent::Image(ImageContent {
                    data: "red-circle".to_owned(),
                    mime_type: "image/png".to_owned(),
                })]),
                timestamp: 1,
            }),
            Message::Assistant(empty_assistant(StopReason::Stop)),
        ],
        ..Default::default()
    };
    let image_config = build_anthropic_client_config(
        &model,
        &image_context,
        AnthropicClientOptions {
            api_key: "tid_copilot_session_test_token".to_owned(),
            headers: BTreeMap::from([("X-Initiator".to_owned(), "override".to_owned())]),
            ..Default::default()
        },
    );
    assert_eq!(
        image_config
            .default_headers
            .get("Copilot-Vision-Request")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        image_config
            .default_headers
            .get("X-Initiator")
            .map(String::as_str),
        Some("override")
    );
}

#[test]
fn cloudflare_ai_gateway_anthropic_client_config_uses_cf_aig_auth_and_preserves_byok_headers() {
    let model = get_model("cloudflare-ai-gateway", "claude-sonnet-4-5")
        .expect("cloudflare gateway anthropic model");
    assert_eq!(model.api, "anthropic-messages");
    assert!(model.base_url.contains("gateway.ai.cloudflare.com"));
    assert!(model.base_url.contains("/anthropic"));

    let context = Context {
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };
    let config = build_anthropic_client_config(
        &model,
        &context,
        AnthropicClientOptions {
            api_key: "cf-token".to_owned(),
            headers: BTreeMap::from([(
                "Authorization".to_owned(),
                "Bearer upstream-token".to_owned(),
            )]),
            session_id: Some("cf-session".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(config.api_key, None);
    assert_eq!(config.auth_token, None);
    assert!(!config.is_oauth_token);
    assert_eq!(
        config
            .default_headers
            .get("cf-aig-authorization")
            .map(String::as_str),
        Some("Bearer cf-token")
    );
    assert_eq!(
        config
            .default_headers
            .get("Authorization")
            .map(String::as_str),
        Some("Bearer upstream-token")
    );
    assert!(!config.default_headers.contains_key("x-api-key"));
    assert_eq!(
        config
            .default_headers
            .get("x-session-affinity")
            .map(String::as_str),
        Some("cf-session")
    );
}

#[test]
fn fireworks_anthropic_client_config_applies_session_affinity_rules() {
    let fireworks =
        get_model("fireworks", "accounts/fireworks/models/kimi-k2p6").expect("fireworks model");
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Use the tool"))],
        tools: vec![anthropic_tool()],
        ..Default::default()
    };

    let config = build_anthropic_client_config(
        &fireworks,
        &context,
        AnthropicClientOptions {
            api_key: "test-key".to_owned(),
            session_id: Some("fireworks-session-1".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(
        config
            .default_headers
            .get("x-session-affinity")
            .map(String::as_str),
        Some("fireworks-session-1")
    );

    let none_config = build_anthropic_client_config(
        &fireworks,
        &context,
        AnthropicClientOptions {
            api_key: "test-key".to_owned(),
            session_id: Some("fireworks-session-2".to_owned()),
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        },
    );
    assert!(
        none_config
            .default_headers
            .get("x-session-affinity")
            .is_none()
    );

    let native = get_model("anthropic", "claude-opus-4-7").expect("anthropic model");
    let native_config = build_anthropic_client_config(
        &native,
        &context,
        AnthropicClientOptions {
            api_key: "test-key".to_owned(),
            session_id: Some("anthropic-session-1".to_owned()),
            ..Default::default()
        },
    );
    assert!(
        native_config
            .default_headers
            .get("x-session-affinity")
            .is_none()
    );
}

#[test]
fn fireworks_anthropic_payload_applies_tool_compat_rules() {
    let fireworks =
        get_model("fireworks", "accounts/fireworks/models/kimi-k2p6").expect("fireworks model");
    let native = get_model("anthropic", "claude-opus-4-7").expect("anthropic model");
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Use the tool"))],
        tools: vec![anthropic_tool()],
        ..Default::default()
    };

    let fireworks_payload =
        build_anthropic_payload(&fireworks, &context, AnthropicPayloadOptions::default());
    assert!(fireworks_payload["tools"][0].get("cache_control").is_none());
    assert!(
        fireworks_payload["tools"][0]
            .get("eager_input_streaming")
            .is_none()
    );
    assert_eq!(
        fireworks_payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let native_payload =
        build_anthropic_payload(&native, &context, AnthropicPayloadOptions::default());
    assert_eq!(
        native_payload["tools"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(native_payload["tools"][0]["eager_input_streaming"], true);
}

#[test]
fn anthropic_payload_adds_short_cache_control_to_system_last_user_and_last_tool() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["PI_CACHE_RETENTION"]);
    let model = anthropic_test_model(None);
    let context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        tools: vec![anthropic_tool()],
    };

    let payload = build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());

    assert_eq!(
        payload["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        payload["tools"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
}

#[test]
fn anthropic_payload_sets_one_hour_cache_ttl_for_long_retention() {
    let model = anthropic_test_model(None);
    let context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };

    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["system"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    assert_eq!(
        payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
}

#[test]
fn anthropic_payload_omits_cache_control_for_none_and_ttl_when_unsupported() {
    let model = anthropic_test_model(Some(json!({
        "supportsLongCacheRetention": false,
    })));
    let context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    };

    let long_payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        },
    );
    assert_eq!(
        long_payload["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let none_payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        },
    );
    assert!(none_payload["system"][0].get("cache_control").is_none());
    assert!(none_payload["messages"][0]["content"].as_str().is_some());
}

#[test]
fn anthropic_payload_preserves_assistant_tool_use_and_image_tool_results() {
    let mut model = anthropic_test_model(None);
    model.input = vec![InputKind::Text, InputKind::Image];
    let mut assistant = empty_assistant_for_model(&model);
    assistant.stop_reason = StopReason::ToolUse;
    assistant.content = vec![AssistantContent::ToolCall(ToolCall {
        id: "call_image".to_owned(),
        name: "inspect_image".to_owned(),
        arguments: object(json!({ "path": "circle.png" })),
        thought_signature: None,
    })];
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Inspect the attached output.")),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_image".to_owned(),
                tool_name: "inspect_image".to_owned(),
                content: vec![
                    ToolResultContent::text("A red circle with a diameter of 100 pixels."),
                    ToolResultContent::Image(ImageContent {
                        data: "base64-image".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_only_image".to_owned(),
                tool_name: "inspect_image".to_owned(),
                content: vec![ToolResultContent::Image(ImageContent {
                    data: "base64-only-image".to_owned(),
                    mime_type: "image/webp".to_owned(),
                })],
                details: None,
                is_error: true,
                timestamp: 3,
            }),
        ],
        ..Default::default()
    };

    let payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["messages"][1]["content"][0],
        json!({
            "type": "tool_use",
            "id": "call_image",
            "name": "inspect_image",
            "input": { "path": "circle.png" },
        })
    );
    assert_eq!(payload["messages"][2]["role"], "user");
    assert_eq!(
        payload["messages"][2]["content"][0],
        json!({
            "type": "tool_result",
            "tool_use_id": "call_image",
            "content": [
                {
                    "type": "text",
                    "text": "A red circle with a diameter of 100 pixels.",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "base64-image",
                    },
                },
            ],
            "is_error": false,
        })
    );
    assert_eq!(
        payload["messages"][2]["content"][1],
        json!({
            "type": "tool_result",
            "tool_use_id": "call_only_image",
            "content": [
                {
                    "type": "text",
                    "text": "(see attached image)",
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/webp",
                        "data": "base64-only-image",
                    },
                },
            ],
            "is_error": true,
        })
    );
}

#[test]
fn anthropic_simple_payload_disables_budget_reasoning_when_thinking_is_off() {
    let model = get_model("anthropic", "claude-sonnet-4-5").expect("model");
    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions::default(),
    );

    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
    assert!(payload.get("output_config").is_none());
}

#[test]
fn anthropic_simple_payload_disables_adaptive_reasoning_when_thinking_is_off() {
    for model_id in ["claude-opus-4-6", "claude-opus-4-7"] {
        let model = get_model("anthropic", model_id).expect("model");
        let payload = build_anthropic_simple_payload(
            &model,
            &user_context("Hello"),
            SimpleStreamOptions::default(),
        );

        assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
        assert!(payload.get("output_config").is_none());
    }
}

#[test]
fn anthropic_simple_payload_applies_base_options_and_budget_adjustment() {
    let mut model = get_model("anthropic", "claude-sonnet-4-5").expect("model");
    model.context_window = 10_000;
    model.max_tokens = 3_000;

    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );
    assert_eq!(payload["max_tokens"], 3_000);
    assert_eq!(
        payload["thinking"],
        json!({
            "type": "enabled",
            "budget_tokens": 1_976,
            "display": "summarized",
        })
    );

    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            stream: StreamOptions {
                temperature: Some(0.2),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    assert_eq!(payload["max_tokens"], 3_000);
    assert_eq!(payload["temperature"], 0.2);
    assert_eq!(payload["thinking"], json!({ "type": "disabled" }));
}

#[test]
fn anthropic_simple_payload_uses_adaptive_thinking_for_opus_47() {
    let model = get_model("anthropic", "claude-opus-4-7").expect("model");
    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "high" }));
}

#[test]
fn anthropic_simple_payload_maps_xhigh_to_max_for_opus_46() {
    let model = get_model("anthropic", "claude-opus-4-6").expect("model");
    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "max" }));
}

#[test]
fn anthropic_simple_payload_maps_xhigh_to_opus_47_effort() {
    let model = get_model("anthropic", "claude-opus-4-7").expect("model");
    let payload = build_anthropic_simple_payload(
        &model,
        &user_context("Hello"),
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["thinking"],
        json!({ "type": "adaptive", "display": "summarized" })
    );
    assert_eq!(payload["output_config"], json!({ "effort": "xhigh" }));
}

#[test]
fn anthropic_claude_code_tool_name_normalization_round_trips_known_tools() {
    let tools = vec![
        Tool {
            name: "todowrite".to_owned(),
            description: "Write a todo".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "read".to_owned(),
            description: "Read a file".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "write".to_owned(),
            description: "Write a file".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "edit".to_owned(),
            description: "Edit a file".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "bash".to_owned(),
            description: "Run a shell command".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "find".to_owned(),
            description: "Find files".to_owned(),
            parameters: json!({ "type": "object" }),
        },
    ];

    assert_eq!(to_claude_code_tool_name("todowrite"), "TodoWrite");
    assert_eq!(from_claude_code_tool_name("TodoWrite", &tools), "todowrite");
    assert_eq!(to_claude_code_tool_name("read"), "Read");
    assert_eq!(from_claude_code_tool_name("Read", &tools), "read");
    assert_eq!(to_claude_code_tool_name("write"), "Write");
    assert_eq!(from_claude_code_tool_name("Write", &tools), "write");
    assert_eq!(to_claude_code_tool_name("edit"), "Edit");
    assert_eq!(from_claude_code_tool_name("Edit", &tools), "edit");
    assert_eq!(to_claude_code_tool_name("bash"), "Bash");
    assert_eq!(from_claude_code_tool_name("Bash", &tools), "bash");
    assert_eq!(to_claude_code_tool_name("find"), "find");
    assert_eq!(from_claude_code_tool_name("Glob", &tools), "Glob");
    assert_eq!(to_claude_code_tool_name("my_custom_tool"), "my_custom_tool");
    assert_eq!(
        from_claude_code_tool_name("my_custom_tool", &tools),
        "my_custom_tool"
    );
}

#[test]
fn anthropic_oauth_payload_uses_claude_code_tool_names_for_tools_and_history() {
    let model = anthropic_test_model(None);
    let tools = vec![
        Tool {
            name: "todowrite".to_owned(),
            description: "Write todos".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "find".to_owned(),
            description: "Find files".to_owned(),
            parameters: json!({ "type": "object" }),
        },
        Tool {
            name: "my_custom_tool".to_owned(),
            description: "Custom tool".to_owned(),
            parameters: json!({ "type": "object" }),
        },
    ];
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(ToolCall {
                id: "call_todo".to_owned(),
                name: "todowrite".to_owned(),
                arguments: object(json!({ "todos": [] })),
                thought_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_find".to_owned(),
                name: "find".to_owned(),
                arguments: object(json!({ "pattern": "*.rs" })),
                thought_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_custom".to_owned(),
                name: "my_custom_tool".to_owned(),
                arguments: object(json!({ "value": true })),
                thought_signature: None,
            }),
        ],
        ..empty_assistant_for_model(&model)
    };
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Use tools")),
            Message::Assistant(assistant),
        ],
        tools,
        ..Default::default()
    };

    let default_payload =
        build_anthropic_payload(&model, &context, AnthropicPayloadOptions::default());
    assert_eq!(default_payload["tools"][0]["name"], "todowrite");
    assert_eq!(
        default_payload["messages"][1]["content"][0]["name"],
        "todowrite"
    );

    let oauth_payload = build_anthropic_payload(
        &model,
        &context,
        AnthropicPayloadOptions {
            use_claude_code_tool_names: true,
            ..Default::default()
        },
    );
    assert_eq!(oauth_payload["tools"][0]["name"], "TodoWrite");
    assert_eq!(oauth_payload["tools"][1]["name"], "find");
    assert_eq!(oauth_payload["tools"][2]["name"], "my_custom_tool");
    assert_eq!(
        oauth_payload["messages"][1]["content"][0]["name"],
        "TodoWrite"
    );
    assert_eq!(oauth_payload["messages"][1]["content"][1]["name"], "find");
    assert_eq!(
        oauth_payload["messages"][1]["content"][2]["name"],
        "my_custom_tool"
    );
}

#[tokio::test]
async fn anthropic_oauth_stream_processor_restores_source_tool_name() {
    let model = anthropic_test_model(None);
    let tools = vec![Tool {
        name: "todowrite".to_owned(),
        description: "Write todos".to_owned(),
        parameters: json!({ "type": "object" }),
    }];
    let mut processor = AnthropicStreamProcessor::with_claude_code_tool_names(tools);
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    processor
        .process_event(
            json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_todo",
                    "name": "TodoWrite",
                    "input": {},
                },
            }),
            &mut output,
            &sender,
        )
        .expect("content block start");
    processor
        .process_event(
            json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": "{\"todos\":[]}",
                },
            }),
            &mut output,
            &sender,
        )
        .expect("content block delta");
    processor
        .process_event(
            json!({ "type": "content_block_stop", "index": 0 }),
            &mut output,
            &sender,
        )
        .expect("content block stop");
    processor.finish(&mut output, &sender);
    drop(sender);

    let AssistantContent::ToolCall(tool_call) = &output.content[0] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.name, "todowrite");
    assert_eq!(tool_call.arguments, object(json!({ "todos": [] })));

    let events = collect_events(stream).await;
    let tool_call_end = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call),
            _ => None,
        })
        .expect("toolcall_end");
    assert_eq!(tool_call_end.name, "todowrite");
    assert_eq!(tool_call_end.arguments, object(json!({ "todos": [] })));
}

#[test]
fn anthropic_stream_processor_returns_error_for_provider_stop_reason_errors() {
    let model = anthropic_test_model(None);
    let mut processor = AnthropicStreamProcessor::new();
    let mut output = empty_assistant_for_model(&model);
    let (sender, _stream) = assistant_message_event_stream();

    let error = processor
        .process_event(
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "refusal" },
                "usage": { "input_tokens": 3, "output_tokens": 1 },
            }),
            &mut output,
            &sender,
        )
        .expect_err("refusal stop reason should fail the stream");

    assert_eq!(error, "Provider stop_reason: refusal");
    assert_eq!(output.stop_reason, StopReason::Error);
    assert_eq!(output.error_message.as_deref(), Some(error.as_str()));
    assert_eq!(output.usage.input, 3);
    assert_eq!(output.usage.output, 1);

    let unknown = processor
        .process_event(
            json!({
                "type": "message_delta",
                "delta": { "stop_reason": "provider_added_reason" },
            }),
            &mut output,
            &sender,
        )
        .expect_err("unknown stop reason should fail the stream");
    assert!(unknown.contains("Unhandled Anthropic stop_reason: provider_added_reason"));
}

#[tokio::test]
async fn openai_responses_stream_cleans_partial_json_from_tool_calls() {
    let model = get_model("openai", "gpt-5-mini").expect("model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();
    let arguments_json = r#"{"path":"README.md","content":"updated"}"#;

    process_openai_responses_events(
        [
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "edit",
                    "arguments": ""
                }
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": "{\"path\":\"README.md\""
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "delta": ",\"content\":\"updated\"}"
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "arguments": arguments_json
            }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_test",
                    "call_id": "call_test",
                    "name": "edit",
                    "arguments": arguments_json
                }
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process stream");
    drop(sender);

    assert_eq!(output.content.len(), 1);
    let AssistantContent::ToolCall(persisted_tool_call) = &output.content[0] else {
        panic!("tool call");
    };
    assert_eq!(persisted_tool_call.id, "call_test|fc_test");
    assert_eq!(persisted_tool_call.arguments["path"], "README.md");
    assert_eq!(persisted_tool_call.arguments["content"], "updated");

    let events = collect_events(stream).await;
    let tool_call_end = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call),
            _ => None,
        })
        .expect("toolcall_end");
    assert_eq!(tool_call_end, persisted_tool_call);
    assert_eq!(tool_call_end.arguments["path"], "README.md");
    assert_eq!(tool_call_end.arguments["content"], "updated");
}

#[test]
fn openai_responses_message_conversion_hashes_foreign_tool_item_ids() {
    let raw_tool_call_id = "call_4VnzVawQXPB9MgYib7CiQFEY|I9b95oN1wD/cHXKTw3PpRkL6KkCtzTJhUxMouMWYwHeTo2j3htzfSk7YPx2vifiIM4g3A8XXyOj8q4Bt6SLUG7gqY1E3ELkrkVQNHglRfUmWj84lqxJY+Puieb3VKyX0FB+83TUzn91cDMF/4gzt990IzqVrc+nIb9RRscRD070Du16q1glydVjWR0SBJsE6TbY/esOjFpqplogQqrajm1eI++f3eLi73R6q7hVusY0QbeFySVxABCjhN0lXB04caBe1rzHjYzul6MAXj7uq+0r17VLq+yrtyYhN12wkmFqHeqTyEei6EFPbMy24Nc+IbJlkP0OCg02W+gOnyBFcbi2ctvJFSOhSjt1CqBdqCnnhwUqXjbWiT0wh3DmLScRgTHmGkaI+oAcQQjfic65nxj+TnEkReA==";
    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: raw_tool_call_id.to_owned(),
            name: "edit".to_owned(),
            arguments: object(json!({ "path": "src/styles/app.css" })),
            thought_signature: None,
        })],
        api: "openai-responses".to_owned(),
        provider: "github-copilot".to_owned(),
        model: "gpt-5.5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let context = Context {
        system_prompt: Some("You are concise.".to_owned()),
        messages: vec![
            Message::User(UserMessage::text("Use the tool.")),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: raw_tool_call_id.to_owned(),
                tool_name: "edit".to_owned(),
                content: vec![ToolResultContent::text("ok")],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
        ],
        tools: Vec::new(),
    };

    let input = convert_openai_responses_messages(
        &model,
        &context,
        &["openai", "openai-codex", "opencode"],
        true,
    );
    let function_call = input
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .expect("function call");
    let item_id = function_call
        .get("id")
        .and_then(Value::as_str)
        .expect("function call id");
    let expected_item_id =
        build_foreign_responses_item_id(raw_tool_call_id.split_once('|').expect("pipe").1);
    assert_eq!(expected_item_id, "fc_ifd2c719fz6a9");
    assert_eq!(item_id, expected_item_id);
    assert!(item_id.len() <= 64);
    assert!(item_id.starts_with("fc_"));
    assert!(
        item_id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
    );
}

#[test]
fn openai_responses_message_conversion_keeps_tool_result_images_in_function_output() {
    let model = get_model("openai", "gpt-5-mini").expect("openai model");
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "call_1|fc_1".to_owned(),
            name: "get_circle_with_description".to_owned(),
            arguments: Map::new(),
            thought_signature: None,
        })],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let context = Context {
        system_prompt: None,
        messages: vec![
            Message::User(UserMessage::text("Use the tool.")),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "call_1|fc_1".to_owned(),
                tool_name: "get_circle_with_description".to_owned(),
                content: vec![
                    ToolResultContent::text("A red circle with a diameter of 100 pixels."),
                    ToolResultContent::Image(ImageContent {
                        data: "base64-image".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
        ],
        tools: Vec::new(),
    };

    let input =
        convert_openai_responses_messages(&model, &context, &["openai", "openai-codex"], true);
    let function_output_index = input
        .iter()
        .position(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .expect("function_call_output");
    let function_output = &input[function_output_index];
    assert_eq!(
        function_output.get("call_id").and_then(Value::as_str),
        Some("call_1")
    );
    let output = function_output
        .get("output")
        .and_then(Value::as_array)
        .expect("content array output");
    assert!(output.iter().any(|item| {
        item.get("type").and_then(Value::as_str) == Some("input_text")
            && item
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .contains("red circle")
    }));
    assert!(output.iter().any(|item| {
        item.get("type").and_then(Value::as_str) == Some("input_image")
            && item
                .get("image_url")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .starts_with("data:image/png;base64,")
    }));
    assert!(
        !input
            .iter()
            .skip(function_output_index + 1)
            .any(|item| item.get("role").and_then(Value::as_str) == Some("user"))
    );
}

#[test]
fn empty_message_conversion_skips_empty_user_blocks_and_empty_assistant_turns() {
    let empty_blocks_context = Context {
        messages: vec![Message::User(UserMessage {
            content: UserContentValue::Blocks(Vec::new()),
            timestamp: 1,
        })],
        ..Default::default()
    };

    let openai_model = get_model("openai", "gpt-5-mini").expect("openai model");
    assert!(
        convert_openai_responses_messages(
            &openai_model,
            &empty_blocks_context,
            &["openai", "openai-codex"],
            true,
        )
        .is_empty()
    );
    assert!(
        convert_openai_completions_messages(
            &{
                let mut model = get_model("openai", "gpt-4o-mini").expect("openai model");
                model.api = "openai-completions".to_owned();
                model
            },
            &empty_blocks_context
        )
        .is_empty()
    );

    let azure_model =
        get_model("azure-openai-responses", "gpt-4o-mini").expect("azure openai model");
    assert!(
        build_azure_openai_responses_payload(
            &azure_model,
            &empty_blocks_context,
            AzureOpenAIResponsesPayloadOptions::default(),
        )["input"]
            .as_array()
            .expect("azure input")
            .is_empty()
    );

    let codex_model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    assert!(
        build_openai_codex_responses_payload(
            &codex_model,
            &empty_blocks_context,
            OpenAICodexResponsesPayloadOptions::default(),
        )["input"]
            .as_array()
            .expect("codex input")
            .is_empty()
    );

    let anthropic_model = get_model("anthropic", "claude-sonnet-4-5").expect("anthropic model");
    assert!(
        build_anthropic_payload(
            &anthropic_model,
            &empty_blocks_context,
            AnthropicPayloadOptions::default(),
        )["messages"]
            .as_array()
            .expect("anthropic messages")
            .is_empty()
    );

    let bedrock_model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    assert!(
        build_bedrock_payload(
            &bedrock_model,
            &empty_blocks_context,
            BedrockPayloadOptions::default(),
        )["messages"]
            .as_array()
            .expect("bedrock messages")
            .is_empty()
    );

    let mistral_model = get_model("mistral", "devstral-medium-latest").expect("mistral model");
    assert!(
        build_mistral_chat_payload(
            &mistral_model,
            &empty_blocks_context,
            MistralPayloadOptions::default(),
        )["messages"]
            .as_array()
            .expect("mistral messages")
            .is_empty()
    );

    let empty_assistant = AssistantMessage {
        content: vec![
            AssistantContent::Text(TextContent::new("   \n\t")),
            AssistantContent::Thinking(ThinkingContent::new("")),
        ],
        api: openai_model.api.clone(),
        provider: openai_model.provider.clone(),
        model: openai_model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 2,
    };
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Hello")),
            Message::Assistant(empty_assistant),
            Message::User(UserMessage::text("Please respond this time.")),
        ],
        ..Default::default()
    };

    let responses_messages =
        convert_openai_responses_messages(&openai_model, &context, &["openai"], true);
    assert_two_user_no_assistant_messages("openai responses empty assistant", &responses_messages);
    assert!(!responses_messages.iter().any(|message| {
        message.get("type").and_then(Value::as_str) == Some("message")
            && message
                .pointer("/content/0/text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .trim()
                .is_empty()
    }));

    let google_model = get_model("google", "gemini-2.5-flash").expect("google model");
    let google_messages = convert_google_messages(&google_model, &context);
    assert_two_user_no_assistant_messages("google empty assistant", &google_messages);

    let openai_completions_messages = convert_openai_completions_messages(
        &{
            let mut model = get_model("openai", "gpt-4o-mini").expect("openai model");
            model.api = "openai-completions".to_owned();
            model
        },
        &context,
    );
    assert_two_user_no_assistant_messages(
        "openai completions empty assistant",
        &openai_completions_messages,
    );

    let azure_input = build_azure_openai_responses_payload(
        &azure_model,
        &context,
        AzureOpenAIResponsesPayloadOptions::default(),
    )["input"]
        .as_array()
        .expect("azure input")
        .clone();
    assert_two_user_no_assistant_messages("azure empty assistant", &azure_input);

    let codex_input = build_openai_codex_responses_payload(
        &codex_model,
        &context,
        OpenAICodexResponsesPayloadOptions::default(),
    )["input"]
        .as_array()
        .expect("codex input")
        .clone();
    assert_two_user_no_assistant_messages("codex empty assistant", &codex_input);

    let anthropic_messages = build_anthropic_payload(
        &anthropic_model,
        &context,
        AnthropicPayloadOptions::default(),
    )["messages"]
        .as_array()
        .expect("anthropic messages")
        .clone();
    assert_two_user_no_assistant_messages("anthropic empty assistant", &anthropic_messages);

    let bedrock_messages = build_bedrock_payload(
        &bedrock_model,
        &context,
        BedrockPayloadOptions::default(),
    )["messages"]
        .as_array()
        .expect("bedrock messages")
        .clone();
    assert_two_user_no_assistant_messages("bedrock empty assistant", &bedrock_messages);

    let mut aborted_assistant = empty_assistant_for_model(&bedrock_model);
    aborted_assistant.stop_reason = StopReason::Aborted;
    aborted_assistant.error_message = Some("Request was aborted".to_owned());
    let aborted_context = Context {
        messages: vec![
            Message::User(UserMessage::text("Hello, how are you?")),
            Message::Assistant(aborted_assistant),
            Message::User(UserMessage::text("What is 2 + 2?")),
        ],
        ..Default::default()
    };
    let bedrock_after_abort_messages = build_bedrock_payload(
        &bedrock_model,
        &aborted_context,
        BedrockPayloadOptions::default(),
    )["messages"]
        .as_array()
        .expect("bedrock messages")
        .clone();
    assert_two_user_no_assistant_messages(
        "bedrock aborted empty assistant",
        &bedrock_after_abort_messages,
    );

    let mistral_messages = build_mistral_chat_payload(
        &mistral_model,
        &context,
        MistralPayloadOptions::default(),
    )["messages"]
        .as_array()
        .expect("mistral messages")
        .clone();
    assert_two_user_no_assistant_messages("mistral empty assistant", &mistral_messages);

    for text in ["", "   \n\t  "] {
        let context = Context {
            messages: vec![Message::User(UserMessage::text(text))],
            ..Default::default()
        };
        let _ =
            build_openai_responses_payload(&openai_model, &context, Default::default())["input"]
                .as_array()
                .expect("openai input");
        let _ =
            build_azure_openai_responses_payload(&azure_model, &context, Default::default())
                ["input"]
                .as_array()
                .expect("azure input");
        let _ =
            build_openai_codex_responses_payload(&codex_model, &context, Default::default())
                ["input"]
                .as_array()
                .expect("codex input");
        let _ = build_openai_completions_payload(
            &{
                let mut model = get_model("openai", "gpt-4o-mini").expect("openai model");
                model.api = "openai-completions".to_owned();
                model
            },
            &context,
            Default::default(),
        )["messages"]
            .as_array()
            .expect("openai completions messages");
        let _ = convert_google_messages(&google_model, &context);
        let _ = build_anthropic_payload(&anthropic_model, &context, Default::default())["messages"]
            .as_array()
            .expect("anthropic messages");
        let _ = build_bedrock_payload(&bedrock_model, &context, Default::default())["messages"]
            .as_array()
            .expect("bedrock messages");
        let _ =
            build_mistral_chat_payload(&mistral_model, &context, Default::default())["messages"]
                .as_array()
                .expect("mistral messages");
    }
}

fn assert_two_user_no_assistant_messages(label: &str, messages: &[Value]) {
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.get("role").and_then(Value::as_str) == Some("user"))
            .count(),
        2,
        "{label}: expected two user messages"
    );
    assert!(
        !messages
            .iter()
            .any(|message| message.get("role").and_then(Value::as_str) == Some("assistant")),
        "{label}: empty assistant turn should be omitted"
    );
    assert!(
        !messages
            .iter()
            .any(|message| message.get("role").and_then(Value::as_str) == Some("model")),
        "{label}: empty model turn should be omitted"
    );
}

#[test]
fn openai_responses_payload_sets_prompt_cache_fields_for_long_retention() {
    let model = get_model("openai", "gpt-5-mini").expect("openai model");
    let context = user_context("Hello");

    let payload = build_openai_responses_payload(
        &model,
        &context,
        OpenAIResponsesPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-responses".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(payload["model"], model.id);
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["store"], false);
    assert_eq!(payload["prompt_cache_key"], "session-responses");
    assert_eq!(payload["prompt_cache_retention"], "24h");
    assert_eq!(payload["input"][0]["role"], "user");
}

#[test]
fn openai_responses_payload_includes_function_tools_with_default_strict_false() {
    let model = get_model("openai", "gpt-5-mini").expect("openai model");
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Inspect a file."))],
        tools: vec![Tool {
            name: "inspect".to_owned(),
            description: "Inspect a file".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            }),
        }],
        ..Default::default()
    };

    let payload =
        build_openai_responses_payload(&model, &context, OpenAIResponsesPayloadOptions::default());

    assert_eq!(
        payload["tools"][0],
        json!({
            "type": "function",
            "name": "inspect",
            "description": "Inspect a file",
            "parameters": {
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            },
            "strict": false,
        })
    );
    assert_eq!(
        convert_openai_responses_tools(&context.tools, Some(true))[0]["strict"],
        true
    );
}

#[test]
fn openai_responses_payload_sets_long_retention_for_proxy_when_supported() {
    let mut model = get_model("openai", "gpt-5-mini").expect("openai model");
    model.base_url = "https://my-proxy.example.com/v1".to_owned();

    let payload = build_openai_responses_payload(
        &model,
        &user_context("Hello"),
        OpenAIResponsesPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-proxy-long".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(payload["prompt_cache_key"], "session-proxy-long");
    assert_eq!(payload["prompt_cache_retention"], "24h");
}

#[test]
fn openai_responses_payload_omits_cache_fields_when_retention_is_none() {
    let model = get_model("openai", "gpt-5-mini").expect("openai model");
    let payload = build_openai_responses_payload(
        &model,
        &user_context("Hello"),
        OpenAIResponsesPayloadOptions {
            cache_retention: Some(CacheRetention::None),
            session_id: Some("session-responses-none".to_owned()),
            ..Default::default()
        },
    );

    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

#[test]
fn openai_responses_payload_omits_long_retention_when_compat_disables_it() {
    let mut model = get_model("openai", "gpt-5-mini").expect("openai model");
    model.compat = Some(json!({ "supportsLongCacheRetention": false }));

    let payload = build_openai_responses_payload(
        &model,
        &user_context("Hello"),
        OpenAIResponsesPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-responses-compat".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(payload["prompt_cache_key"], "session-responses-compat");
    assert!(payload.get("prompt_cache_retention").is_none());
}

#[test]
fn openai_responses_default_headers_apply_session_affinity_and_overrides() {
    let mut model = get_model("openai", "gpt-5-mini").expect("openai model");
    model
        .headers
        .insert("x-static".to_owned(), "model".to_owned());

    let headers = build_openai_responses_default_headers(
        &model,
        Some("session-responses"),
        CacheRetention::Long,
        &BTreeMap::from([("x-static".to_owned(), "override".to_owned())]),
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("session-responses")
    );
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-responses")
    );
    assert_eq!(
        headers.get("x-static").map(String::as_str),
        Some("override")
    );

    let mut proxy_model = model.clone();
    proxy_model.base_url = "https://proxy.example.com/v1".to_owned();
    let headers = build_openai_responses_default_headers(
        &proxy_model,
        Some("session-proxy"),
        CacheRetention::Long,
        &BTreeMap::new(),
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("session-proxy")
    );
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-proxy")
    );

    model.compat = Some(json!({ "sendSessionIdHeader": false }));
    let headers = build_openai_responses_default_headers(
        &model,
        Some("session-responses"),
        CacheRetention::Long,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-responses")
    );

    let headers = build_openai_responses_default_headers(
        &model,
        Some("session-responses"),
        CacheRetention::None,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());

    let headers = build_openai_responses_default_headers(
        &model,
        Some(""),
        CacheRetention::Long,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());

    let headers = build_openai_responses_default_headers(
        &model,
        Some("session-responses"),
        CacheRetention::Long,
        &BTreeMap::from([
            ("session_id".to_owned(), "override-session".to_owned()),
            (
                "x-client-request-id".to_owned(),
                "override-request".to_owned(),
            ),
        ]),
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("override-session")
    );
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("override-request")
    );
}

#[test]
fn openai_responses_and_completions_apply_copilot_dynamic_headers() {
    let mut responses_model = get_model("github-copilot", "gpt-5-mini").expect("copilot model");
    responses_model.api = "openai-responses".to_owned();
    let responses_headers = build_openai_responses_default_headers_with_context(
        &responses_model,
        Some(&user_context("hello")),
        Some("session-responses"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert_eq!(
        responses_headers.get("X-Initiator").map(String::as_str),
        Some("user")
    );
    assert_eq!(
        responses_headers.get("Openai-Intent").map(String::as_str),
        Some("conversation-edits")
    );
    assert!(responses_headers.get("Copilot-Vision-Request").is_none());

    let completions_model = get_model("github-copilot", "gpt-5.3-codex").expect("copilot model");
    let image_context = Context {
        messages: vec![
            Message::User(UserMessage {
                content: UserContentValue::Blocks(vec![UserContent::Image(ImageContent {
                    data: "red-circle".to_owned(),
                    mime_type: "image/png".to_owned(),
                })]),
                timestamp: 1,
            }),
            Message::Assistant(empty_assistant(StopReason::Stop)),
        ],
        ..Default::default()
    };
    let completions_headers = build_openai_completions_default_headers_with_context(
        &completions_model,
        Some(&image_context),
        Some("session-completions"),
        CacheRetention::Short,
        &BTreeMap::from([("X-Initiator".to_owned(), "override".to_owned())]),
    );
    assert_eq!(
        completions_headers
            .get("Copilot-Vision-Request")
            .map(String::as_str),
        Some("true")
    );
    assert_eq!(
        completions_headers.get("Openai-Intent").map(String::as_str),
        Some("conversation-edits")
    );
    assert_eq!(
        completions_headers.get("X-Initiator").map(String::as_str),
        Some("override")
    );
}

#[test]
fn openai_responses_payload_sends_default_none_reasoning_for_supported_openai_models() {
    for model_id in [
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
    ] {
        let model = get_model("openai", model_id).expect("model");
        let payload = build_openai_responses_payload(
            &model,
            &user_context("hi"),
            OpenAIResponsesPayloadOptions::default(),
        );

        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "none" }),
            "{model_id}"
        );
    }
}

#[test]
fn openai_responses_payload_omits_default_reasoning_when_off_is_unsupported() {
    for model_id in [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-pro",
        "gpt-5.2-pro",
        "gpt-5.4-pro",
        "gpt-5.5-pro",
    ] {
        let model = get_model("openai", model_id).expect("model");
        let payload = build_openai_responses_payload(
            &model,
            &user_context("hi"),
            OpenAIResponsesPayloadOptions::default(),
        );

        assert!(payload.get("reasoning").is_none(), "{model_id}");
    }
}

#[test]
fn openai_responses_payload_omits_default_reasoning_for_github_copilot() {
    let model = get_model("github-copilot", "gpt-5-mini").expect("model");
    let payload = build_openai_responses_payload(
        &model,
        &user_context("hi"),
        OpenAIResponsesPayloadOptions::default(),
    );

    assert!(payload.get("reasoning").is_none());
}

#[test]
fn openai_responses_payload_maps_explicit_reasoning_and_includes_encrypted_content() {
    let model = get_model("openai", "gpt-5.5").expect("model");
    let payload = build_openai_responses_payload(
        &model,
        &user_context("hi"),
        OpenAIResponsesPayloadOptions {
            reasoning_effort: Some(ThinkingLevel::XHigh),
            reasoning_summary: Some("detailed".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(
        payload["reasoning"],
        json!({ "effort": "xhigh", "summary": "detailed" })
    );
    assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn openai_responses_payload_omits_zero_max_tokens_and_defaults_empty_reasoning_summary() {
    let model = get_model("openai", "gpt-5.5").expect("model");
    let payload = build_openai_responses_payload(
        &model,
        &user_context("hi"),
        OpenAIResponsesPayloadOptions {
            max_tokens: Some(0),
            reasoning_effort: Some(ThinkingLevel::High),
            reasoning_summary: Some(String::new()),
            ..Default::default()
        },
    );

    assert!(payload.get("max_output_tokens").is_none());
    assert_eq!(
        payload["reasoning"],
        json!({ "effort": "high", "summary": "auto" })
    );
    assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
}

#[test]
fn openai_responses_payload_skips_aborted_reasoning_only_history() {
    let model = get_model("openai", "gpt-5-mini").expect("model");
    let reasoning_item = json!({
        "type": "reasoning",
        "id": "rs_aborted",
        "summary": [],
        "encrypted_content": "encrypted",
    });
    let aborted_reasoning = AssistantMessage {
        content: vec![AssistantContent::Thinking(ThinkingContent {
            thinking: "incomplete private reasoning".to_owned(),
            thinking_signature: Some(reasoning_item.to_string()),
            redacted: false,
        })],
        api: "openai-responses".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-5-mini".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Aborted,
        error_message: None,
        timestamp: 2,
    };

    let payload = build_openai_responses_payload(
        &model,
        &Context {
            messages: vec![
                Message::User(UserMessage {
                    content: "Use the tool.".into(),
                    timestamp: 1,
                }),
                Message::Assistant(aborted_reasoning),
                Message::User(UserMessage {
                    content: "Say hello.".into(),
                    timestamp: 3,
                }),
            ],
            ..Default::default()
        },
        OpenAIResponsesPayloadOptions {
            reasoning_effort: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    let input = payload["input"].as_array().expect("input array");
    assert_eq!(input.len(), 2);
    assert!(input.iter().all(|item| item["role"] == "user"));
    assert!(
        input
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"))
    );
}

#[test]
fn openai_responses_payload_omits_function_call_item_id_for_same_provider_model_handoff() {
    let target_model = get_model("openai", "gpt-5.2-codex").expect("target model");
    let reasoning_item = json!({
        "type": "reasoning",
        "id": "rs_1",
        "summary": [],
        "encrypted_content": "encrypted",
    });
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Need to use the tool.".to_owned(),
                thinking_signature: Some(reasoning_item.to_string()),
                redacted: false,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "call_1|fc_1".to_owned(),
                name: "double_number".to_owned(),
                arguments: object(json!({ "value": 21 })),
                thought_signature: None,
            }),
        ],
        api: "openai-responses".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-5-mini".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 2,
    };

    let payload = build_openai_responses_payload(
        &target_model,
        &Context {
            messages: vec![
                Message::User(UserMessage {
                    content: "Double 21.".into(),
                    timestamp: 1,
                }),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1|fc_1".to_owned(),
                    tool_name: "double_number".to_owned(),
                    content: vec![ToolResultContent::text("42")],
                    details: None,
                    is_error: false,
                    timestamp: 3,
                }),
                Message::User(UserMessage {
                    content: "What was the result?".into(),
                    timestamp: 4,
                }),
            ],
            ..Default::default()
        },
        OpenAIResponsesPayloadOptions {
            reasoning_effort: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    let input = payload["input"].as_array().expect("input array");
    assert!(
        input
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"))
    );
    let function_call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call");
    assert_eq!(function_call["call_id"], "call_1");
    assert!(function_call.get("id").is_none() || function_call["id"].is_null());
    let tool_result = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("tool result");
    assert_eq!(tool_result["call_id"], "call_1");
    assert_eq!(tool_result["output"], "42");
}

#[test]
fn openai_responses_payload_handles_cross_provider_anthropic_tool_handoff() {
    let target_model = get_model("openai", "gpt-5.2-codex").expect("target model");
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Need to use the tool.".to_owned(),
                thinking_signature: None,
                redacted: false,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "toolu_01abc".to_owned(),
                name: "double_number".to_owned(),
                arguments: object(json!({ "value": 21 })),
                thought_signature: Some("anthropic-only".to_owned()),
            }),
        ],
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        model: "claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 2,
    };

    let payload = build_openai_responses_payload(
        &target_model,
        &Context {
            messages: vec![
                Message::User(UserMessage {
                    content: "Double 21.".into(),
                    timestamp: 1,
                }),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "toolu_01abc".to_owned(),
                    tool_name: "double_number".to_owned(),
                    content: vec![ToolResultContent::text("42")],
                    details: None,
                    is_error: false,
                    timestamp: 3,
                }),
                Message::User(UserMessage {
                    content: "What was the result?".into(),
                    timestamp: 4,
                }),
            ],
            ..Default::default()
        },
        OpenAIResponsesPayloadOptions {
            reasoning_effort: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );

    let input = payload["input"].as_array().expect("input array");
    assert!(
        input
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("reasoning"))
    );
    let thinking_text = input
        .iter()
        .find(|item| item["type"] == "message")
        .expect("converted thinking text");
    assert_eq!(thinking_text["content"][0]["text"], "Need to use the tool.");
    let function_call = input
        .iter()
        .find(|item| item["type"] == "function_call")
        .expect("function call");
    assert_eq!(function_call["call_id"], "toolu_01abc");
    assert!(function_call.get("id").is_none() || function_call["id"].is_null());
    let tool_result = input
        .iter()
        .find(|item| item["type"] == "function_call_output")
        .expect("tool result");
    assert_eq!(tool_result["call_id"], "toolu_01abc");
    assert_eq!(tool_result["output"], "42");
}

#[test]
fn openai_responses_usage_applies_service_tier_cost_multiplier() {
    for (model_id, service_tier, multiplier) in [
        ("gpt-5.4", "priority", 2.0),
        ("gpt-5.5", "priority", 2.5),
        ("gpt-5.5", "flex", 0.5),
    ] {
        let model = get_model("openai", model_id).expect("model");
        let usage = parse_openai_responses_usage(
            &json!({
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "input_tokens_details": { "cached_tokens": 0 }
            }),
            &model,
            Some(service_tier),
        );

        assert_eq!(usage.input, 1_000_000);
        assert_eq!(usage.output, 1_000_000);
        assert_eq!(usage.total_tokens, 2_000_000);
        assert_usage_total_matches_components("openai responses service tier", &usage);
        assert_eq!(usage.cost.input, model.cost.input * multiplier);
        assert_eq!(usage.cost.output, model.cost.output * multiplier);
        assert_eq!(
            usage.cost.total,
            (model.cost.input + model.cost.output) * multiplier
        );
    }
}

#[test]
fn usage_total_tokens_match_components_for_provider_parsers() {
    let anthropic = process_anthropic_sse_body(
        &get_model("anthropic", "claude-haiku-4-5").expect("anthropic model"),
        &anthropic_sse_body(minimal_anthropic_sse_events()),
    )
    .expect("anthropic sse")
    .usage;
    assert_usage_total_matches_components("anthropic", &anthropic);

    let openai = get_model("openai", "gpt-5-mini").expect("openai model");
    let openai_responses = parse_openai_responses_usage(
        &json!({
            "input_tokens": 100,
            "output_tokens": 7,
            "total_tokens": 107,
            "input_tokens_details": { "cached_tokens": 20 }
        }),
        &openai,
        None,
    );
    assert_usage_total_matches_components("openai responses", &openai_responses);

    let mut openai_completions_model =
        get_model("openrouter", "google/gemini-2.5-flash").expect("openrouter model");
    openai_completions_model.api = "openai-completions".to_owned();
    let openai_completions = parse_openai_completions_chunk_usage(
        &json!({
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            },
        }),
        &openai_completions_model,
    );
    assert_usage_total_matches_components("openai completions", &openai_completions);

    let openrouter_images = parse_openrouter_images_usage(
        &json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "prompt_tokens_details": {
                "cached_tokens": 40,
                "cache_write_tokens": 15
            }
        }),
        &get_image_model("openrouter", "google/gemini-2.5-flash-image").expect("image model"),
    );
    assert_usage_total_matches_components("openrouter images", &openrouter_images);
}

#[test]
fn openai_responses_stream_maps_text_deltas_and_replays_text_signature() {
    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, _stream) = assistant_message_event_stream();

    process_openai_responses_events(
        [
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }),
            json!({ "type": "response.output_text.delta", "delta": "Hello" }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": "Hello" }]
                }
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process text stream");

    let AssistantContent::Text(text) = &output.content[0] else {
        panic!("text block");
    };
    assert_eq!(text.text, "Hello");
    assert_eq!(text.text_signature, Some(json!({ "v": 1, "id": "msg_1" })));

    let replay_items = openai_codex_response_items_for_continuation(&model, &output);
    assert_eq!(replay_items[0]["id"], "msg_1");
    assert_eq!(replay_items[0]["content"][0]["text"], "Hello");
}

#[tokio::test]
async fn openai_responses_stream_maps_reasoning_summary_events() {
    let model = get_model("openai", "gpt-5.5").expect("openai model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_openai_responses_events(
        [
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "encrypted"
                }
            }),
            json!({
                "type": "response.reasoning_summary_part.added",
                "part": { "type": "summary_text", "text": "" }
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "Plan the answer"
            }),
            json!({ "type": "response.reasoning_summary_part.done" }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{ "type": "summary_text", "text": "Plan the answer\n\n" }],
                    "content": [{ "type": "reasoning_text", "text": "private chain" }],
                    "encrypted_content": "encrypted"
                }
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process reasoning stream");
    drop(sender);

    let AssistantContent::Thinking(thinking) = &output.content[0] else {
        panic!("thinking block");
    };
    assert_eq!(thinking.thinking, "Plan the answer\n\n");
    let signature: Value =
        serde_json::from_str(thinking.thinking_signature.as_deref().expect("signature"))
            .expect("reasoning item signature");
    assert_eq!(signature["id"], "rs_1");
    assert_eq!(signature["encrypted_content"], "encrypted");

    let events = collect_events(stream).await;
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::ThinkingStart {
            content_index: 0,
            ..
        })
    ));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "Plan the answer"
    )));
    assert!(events.iter().any(|event| matches!(
        event,
        AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "\n\n"
    )));
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::ThinkingEnd { content, .. }) if content == "Plan the answer\n\n"
    ));
}

#[test]
fn openai_responses_stream_preserves_text_phase_and_refusal_content() {
    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, _stream) = assistant_message_event_stream();

    process_openai_responses_events(
        [
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "id": "msg_refusal",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }),
            json!({ "type": "response.refusal.delta", "delta": "No." }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_refusal",
                    "phase": "final_answer",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "refusal", "refusal": "No." }]
                }
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process refusal stream");

    let AssistantContent::Text(text) = &output.content[0] else {
        panic!("text block");
    };
    assert_eq!(text.text, "No.");
    assert_eq!(
        text.text_signature,
        Some(json!({ "v": 1, "id": "msg_refusal", "phase": "final_answer" }))
    );

    let replay_items = openai_codex_response_items_for_continuation(&model, &output);
    assert_eq!(replay_items[0]["id"], "msg_refusal");
    assert_eq!(replay_items[0]["phase"], "final_answer");
    assert_eq!(replay_items[0]["content"][0]["text"], "No.");
}

#[test]
fn openai_responses_message_conversion_preserves_text_phase_and_hashes_long_ids() {
    let model = get_model("openai", "gpt-5.5").expect("openai model");
    let long_id = "msg_".to_owned() + &"x".repeat(80);
    let mut assistant = empty_assistant_for_model(&model);
    assistant.content.push(AssistantContent::Text(TextContent {
        text: "Visible commentary".to_owned(),
        text_signature: Some(json!({
            "v": 1,
            "id": long_id,
            "phase": "commentary",
        })),
    }));

    let input = convert_openai_responses_messages(
        &model,
        &Context {
            messages: vec![Message::Assistant(assistant)],
            ..Default::default()
        },
        &["openai", "openai-codex"],
        false,
    );

    assert_eq!(input.len(), 1);
    assert_eq!(input[0]["id"], format!("msg_{}", short_hash(&long_id)));
    assert_eq!(input[0]["phase"], "commentary");
    assert_eq!(input[0]["content"][0]["text"], "Visible commentary");
}

#[test]
fn openai_responses_stream_formats_failed_response_errors_like_source() {
    let model = get_model("openai", "gpt-5.5").expect("openai model");
    let (sender, _stream) = assistant_message_event_stream();

    let error = process_openai_responses_events(
        [json!({
            "type": "response.failed",
            "response": {
                "error": { "code": "invalid_request", "message": "bad input" }
            }
        })],
        &mut empty_assistant_for_model(&model),
        &sender,
        &model,
    )
    .expect_err("failed response with error");
    assert_eq!(error, "invalid_request: bad input");

    let incomplete = process_openai_responses_events(
        [json!({
            "type": "response.failed",
            "response": {
                "incomplete_details": { "reason": "max_output_tokens" }
            }
        })],
        &mut empty_assistant_for_model(&model),
        &sender,
        &model,
    )
    .expect_err("failed response with incomplete details");
    assert_eq!(incomplete, "incomplete: max_output_tokens");
}

const CODEX_TEST_JWT_PAYLOAD: &str =
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjX3Rlc3QifX0=";

fn codex_test_token() -> String {
    format!("aaa.{CODEX_TEST_JWT_PAYLOAD}.bbb")
}

fn codex_oauth_token_response(access_token: &str, refresh_token: &str, expires_in: i64) -> String {
    json!({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
    })
    .to_string()
}

#[test]
fn openai_codex_responses_extracts_account_id_and_builds_transport_headers() {
    let token = codex_test_token();
    assert_eq!(
        extract_openai_codex_account_id(&token).as_deref(),
        Ok("acc_test")
    );
    let unpadded = format!("aaa.{}.bbb", CODEX_TEST_JWT_PAYLOAD.trim_end_matches('='));
    assert_eq!(
        extract_openai_codex_account_id(&unpadded).as_deref(),
        Ok("acc_test")
    );
    assert!(extract_openai_codex_account_id("not-a-jwt").is_err());

    let model_headers = BTreeMap::from([
        ("x-model".to_owned(), "model".to_owned()),
        ("accept".to_owned(), "application/json".to_owned()),
    ]);
    let option_headers = BTreeMap::from([
        ("x-option".to_owned(), "option".to_owned()),
        ("Authorization".to_owned(), "Bearer wrong".to_owned()),
    ]);
    let sse = build_openai_codex_sse_headers(
        &model_headers,
        &option_headers,
        "acc_test",
        &token,
        Some("session-123"),
    );

    assert_eq!(sse.get("Authorization"), Some(&format!("Bearer {token}")));
    assert_eq!(
        sse.get("chatgpt-account-id").map(String::as_str),
        Some("acc_test")
    );
    assert_eq!(
        sse.get("OpenAI-Beta").map(String::as_str),
        Some("responses=experimental")
    );
    assert_eq!(sse.get("originator").map(String::as_str), Some("pi"));
    assert_eq!(
        sse.get("accept").map(String::as_str),
        Some("text/event-stream")
    );
    assert_eq!(
        sse.get("content-type").map(String::as_str),
        Some("application/json")
    );
    assert_eq!(
        sse.get("session_id").map(String::as_str),
        Some("session-123")
    );
    assert_eq!(
        sse.get("x-client-request-id").map(String::as_str),
        Some("session-123")
    );
    assert!(!sse.contains_key("x-api-key"));
    assert_eq!(sse.get("x-model").map(String::as_str), Some("model"));
    assert_eq!(sse.get("x-option").map(String::as_str), Some("option"));

    let websocket = build_openai_codex_websocket_headers(
        &model_headers,
        &option_headers,
        "acc_test",
        &token,
        "request-456",
    );
    assert_eq!(
        websocket.get("OpenAI-Beta").map(String::as_str),
        Some(OPENAI_CODEX_WEBSOCKET_BETA)
    );
    assert!(!websocket.contains_key("accept"));
    assert!(!websocket.contains_key("content-type"));
    assert_eq!(
        websocket.get("session_id").map(String::as_str),
        Some("request-456")
    );
    assert_eq!(
        websocket.get("x-client-request-id").map(String::as_str),
        Some("request-456")
    );
}

#[test]
fn openai_codex_responses_omits_session_affinity_without_session_id() {
    let token = codex_test_token();
    let headers = build_openai_codex_sse_headers(
        &BTreeMap::new(),
        &BTreeMap::new(),
        "acc_test",
        &token,
        None,
    );

    assert_eq!(
        headers.get("Authorization"),
        Some(&format!("Bearer {token}"))
    );
    assert_eq!(
        headers.get("chatgpt-account-id").map(String::as_str),
        Some("acc_test")
    );
    assert!(!headers.contains_key("session_id"));
    assert!(!headers.contains_key("x-client-request-id"));

    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    let payload = build_openai_codex_responses_payload(
        &model,
        &user_context("Say hello"),
        OpenAICodexResponsesPayloadOptions::default(),
    );
    assert!(payload.get("prompt_cache_key").is_none());
}

#[test]
fn openai_codex_responses_resolves_urls() {
    assert_eq!(
        resolve_openai_codex_url(None),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        resolve_openai_codex_url(Some("https://chatgpt.com/backend-api/codex")),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        resolve_openai_codex_url(Some("https://chatgpt.com/backend-api/codex/responses/")),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        resolve_openai_codex_websocket_url(Some("http://localhost:3000/backend-api")),
        "ws://localhost:3000/backend-api/codex/responses"
    );
}

#[test]
fn openai_codex_responses_payload_matches_request_body_defaults_and_session() {
    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    let context = Context {
        system_prompt: Some("System instruction".to_owned()),
        messages: vec![Message::User(UserMessage::text("Say hello"))],
        tools: vec![Tool {
            name: "inspect".to_owned(),
            description: "Inspect a file".to_owned(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" }
                }
            }),
        }],
    };

    let payload = build_openai_codex_responses_payload(
        &model,
        &context,
        OpenAICodexResponsesPayloadOptions {
            session_id: Some("session-123".to_owned()),
            temperature: Some(0.2),
            service_tier: Some("priority".to_owned()),
            reasoning_effort: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );

    assert_eq!(payload["model"], "gpt-5.5");
    assert_eq!(payload["store"], false);
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["instructions"], "System instruction");
    assert_eq!(payload["input"][0]["role"], "user");
    assert_eq!(payload["input"][0]["content"][0]["text"], "Say hello");
    assert_eq!(payload["text"], json!({ "verbosity": "low" }));
    assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
    assert_eq!(payload["prompt_cache_key"], "session-123");
    assert_eq!(payload["tool_choice"], "auto");
    assert_eq!(payload["parallel_tool_calls"], true);
    assert_eq!(payload["temperature"], 0.2);
    assert_eq!(payload["service_tier"], "priority");
    assert_eq!(
        payload["reasoning"],
        json!({ "effort": "xhigh", "summary": "auto" })
    );
    assert_eq!(payload["tools"][0]["type"], "function");
    assert_eq!(payload["tools"][0]["name"], "inspect");
    assert_eq!(payload["tools"][0]["strict"], Value::Null);

    let default_instruction = build_openai_codex_responses_payload(
        &model,
        &user_context("Hi"),
        OpenAICodexResponsesPayloadOptions::default(),
    );
    assert_eq!(
        default_instruction["instructions"],
        "You are a helpful assistant."
    );
    assert!(default_instruction.get("prompt_cache_key").is_none());
}

#[test]
fn openai_codex_responses_payload_maps_minimal_reasoning_to_low() {
    for model_id in ["gpt-5.3-codex", "gpt-5.4", "gpt-5.5"] {
        let model = get_model("openai-codex", model_id).expect("codex model");
        let payload = build_openai_codex_responses_payload(
            &model,
            &user_context("hi"),
            OpenAICodexResponsesPayloadOptions {
                reasoning_effort: Some(ThinkingLevel::Minimal),
                ..Default::default()
            },
        );

        assert_eq!(
            payload["reasoning"],
            json!({ "effort": "low", "summary": "auto" }),
            "{model_id}"
        );
        assert!(get_supported_thinking_levels(&model).contains(&ThinkingLevel::XHigh));
    }
}

#[tokio::test]
async fn openai_codex_responses_sse_parser_maps_text_and_terminal_statuses() {
    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");

    for (terminal_type, status, expected_stop, include_done) in [
        ("response.completed", "completed", StopReason::Stop, true),
        (
            "response.incomplete",
            "incomplete",
            StopReason::Length,
            false,
        ),
    ] {
        let incomplete_details = if status == "incomplete" {
            json!({ "reason": "max_output_tokens" })
        } else {
            Value::Null
        };
        let mut frames = vec![
            format!(
                "data: {}",
                json!({
                    "type": "response.output_item.added",
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": []
                    }
                })
            ),
            format!(
                "data: {}",
                json!({
                    "type": "response.content_part.added",
                    "part": { "type": "output_text", "text": "" }
                })
            ),
            format!(
                "data: {}",
                json!({ "type": "response.output_text.delta", "delta": "Hello" })
            ),
            format!(
                "data: {}",
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{ "type": "output_text", "text": "Hello" }]
                    }
                })
            ),
            format!(
                "data: {}",
                json!({
                    "type": terminal_type,
                    "response": {
                        "id": "resp_1",
                        "status": status,
                        "incomplete_details": incomplete_details,
                        "usage": {
                            "input_tokens": 5,
                            "output_tokens": 3,
                            "total_tokens": 8,
                            "input_tokens_details": { "cached_tokens": 0 }
                        }
                    }
                })
            ),
        ];
        if include_done {
            frames.push("data: [DONE]".to_owned());
        }
        let sse = format!("{}\n\n", frames.join("\n\n"));
        let events = parse_openai_codex_sse_events(&sse).expect("sse events");
        assert_eq!(events.len(), 5, "{status}");

        let mut output = empty_assistant_for_model(&model);
        let (sender, stream) = assistant_message_event_stream();
        process_openai_responses_events(events, &mut output, &sender, &model)
            .expect("process codex sse events");
        drop(sender);
        let emitted = collect_events(stream).await;

        assert!(emitted.iter().any(|event| matches!(
            event,
            AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hello"
        )));
        assert!(matches!(
            emitted.last(),
            Some(AssistantMessageEvent::TextEnd { content, .. }) if content == "Hello"
        ));
        assert_eq!(text_of(&output), Some("Hello"), "{status}");
        assert_eq!(output.response_id.as_deref(), Some("resp_1"), "{status}");
        assert_eq!(output.stop_reason, expected_stop, "{status}");
        assert_eq!(output.usage.input, 5, "{status}");
        assert_eq!(output.usage.output, 3, "{status}");
        assert_eq!(output.usage.total_tokens, 8, "{status}");
    }
}

#[test]
fn openai_codex_responses_cached_websocket_request_sends_only_input_delta() {
    let model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    let first_context = Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Say hello"))],
        ..Default::default()
    };
    let first_body = build_openai_codex_responses_payload(
        &model,
        &first_context,
        OpenAICodexResponsesPayloadOptions {
            session_id: Some("session-1".to_owned()),
            ..Default::default()
        },
    );

    let mut first_response = empty_assistant_for_model(&model);
    let (sender, _stream) = assistant_message_event_stream();
    process_openai_responses_events(
        [
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }),
            json!({ "type": "response.output_text.delta", "delta": "Hello" }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{ "type": "output_text", "text": "Hello" }]
                }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 3,
                        "total_tokens": 8,
                        "input_tokens_details": { "cached_tokens": 0 }
                    }
                }
            }),
        ],
        &mut first_response,
        &sender,
        &model,
    )
    .expect("first response");

    let continuation = build_openai_codex_cached_websocket_continuation(
        &model,
        first_body.clone(),
        &first_response,
    )
    .expect("continuation");
    let second_context = Context {
        system_prompt: first_context.system_prompt.clone(),
        messages: vec![
            Message::User(UserMessage::text("Say hello")),
            Message::Assistant(first_response),
            Message::User(UserMessage::text("Now finish")),
        ],
        ..Default::default()
    };
    let second_body = build_openai_codex_responses_payload(
        &model,
        &second_context,
        OpenAICodexResponsesPayloadOptions {
            session_id: Some("session-1".to_owned()),
            ..Default::default()
        },
    );

    let cached =
        build_openai_codex_cached_websocket_request_body(&second_body, Some(&continuation));
    assert!(cached.used_delta);
    assert!(!cached.invalidated_continuation);
    assert_eq!(cached.body["store"], false);
    assert_eq!(cached.body["previous_response_id"], "resp_1");
    assert_eq!(
        cached.body["input"],
        json!([{ "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] }])
    );

    let mut mismatched_body = second_body.clone();
    mismatched_body["service_tier"] = json!("priority");
    let mismatched =
        build_openai_codex_cached_websocket_request_body(&mismatched_body, Some(&continuation));
    assert!(!mismatched.used_delta);
    assert!(mismatched.invalidated_continuation);
    assert!(mismatched.body.get("previous_response_id").is_none());
}

#[test]
fn openai_codex_responses_websocket_debug_stats_match_cached_request_accounting() {
    let mut stats = OpenAICodexWebSocketDebugStats::default();
    let full_request = json!({
        "store": false,
        "input": [
            { "role": "user", "content": [{ "type": "input_text", "text": "Say hello" }] }
        ]
    });
    record_openai_codex_websocket_request_stats(&mut stats, &full_request, false, true);

    assert_eq!(stats.requests, 1);
    assert_eq!(stats.connections_created, 1);
    assert_eq!(stats.connections_reused, 0);
    assert_eq!(stats.cached_context_requests, 1);
    assert_eq!(stats.store_true_requests, 0);
    assert_eq!(stats.full_context_requests, 1);
    assert_eq!(stats.delta_requests, 0);
    assert_eq!(stats.last_input_items, 1);
    assert_eq!(stats.last_delta_input_items, None);
    assert_eq!(stats.last_previous_response_id, None);

    let delta_request = json!({
        "store": false,
        "previous_response_id": "resp_1",
        "input": [
            { "role": "user", "content": [{ "type": "input_text", "text": "Now finish" }] }
        ]
    });
    record_openai_codex_websocket_request_stats(&mut stats, &delta_request, true, true);

    assert_eq!(stats.requests, 2);
    assert_eq!(stats.connections_created, 1);
    assert_eq!(stats.connections_reused, 1);
    assert_eq!(stats.cached_context_requests, 2);
    assert_eq!(stats.full_context_requests, 1);
    assert_eq!(stats.delta_requests, 1);
    assert_eq!(stats.last_delta_input_items, Some(1));
    assert_eq!(stats.last_previous_response_id.as_deref(), Some("resp_1"));

    record_openai_codex_websocket_failure(&mut stats, "connection refused");
    assert_eq!(stats.websocket_failures, 1);
    assert_eq!(
        stats.last_websocket_error.as_deref(),
        Some("connection refused")
    );
    assert_eq!(stats.websocket_fallback_active, Some(true));

    record_openai_codex_websocket_sse_fallback(&mut stats, true);
    assert_eq!(stats.sse_fallbacks, 1);
    assert_eq!(stats.websocket_fallback_active, Some(true));
}

#[test]
fn openai_responses_stream_maps_response_incomplete_event_to_length() {
    let model = get_model("openai", "gpt-5-mini").expect("model");
    let mut output = empty_assistant_for_model(&model);
    let (sender, _stream) = assistant_message_event_stream();

    process_openai_responses_events(
        [json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": { "reason": "max_output_tokens" },
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                    "input_tokens_details": { "cached_tokens": 0 }
                }
            }
        })],
        &mut output,
        &sender,
        &model,
    )
    .expect("process incomplete");

    assert_eq!(output.response_id.as_deref(), Some("resp_incomplete"));
    assert_eq!(output.stop_reason, StopReason::Length);
    assert_eq!(output.usage.input, 5);
    assert_eq!(output.usage.output, 3);
}

#[test]
fn openai_codex_responses_usage_uses_client_tier_when_response_echoes_default() {
    for (model_id, request_tier, multiplier) in [
        ("gpt-5.1-codex", "flex", 0.5),
        ("gpt-5.1-codex", "priority", 2.0),
        ("gpt-5.5", "flex", 0.5),
        ("gpt-5.5", "priority", 2.5),
    ] {
        let mut model = get_model("openai-codex", model_id).expect("codex model");
        model.cost = ModelCost {
            input: 1.0,
            output: 2.0,
            cache_read: 0.0,
            cache_write: 0.0,
        };
        let usage = parse_openai_codex_responses_usage(
            &json!({
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "input_tokens_details": { "cached_tokens": 0 }
            }),
            &model,
            Some("default"),
            Some(request_tier),
        );

        assert_eq!(
            resolve_openai_codex_service_tier(Some("default"), Some(request_tier)),
            Some(request_tier.to_owned())
        );
        assert_eq!(usage.cost.input, 1.0 * multiplier, "{model_id}");
        assert_eq!(usage.cost.output, 2.0 * multiplier, "{model_id}");
        assert_eq!(usage.cost.total, 3.0 * multiplier, "{model_id}");
    }
}

#[test]
fn openai_codex_responses_retry_delay_respects_headers_and_backoff() {
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", Some("1500"), None, 0, 0),
        Some(1500)
    );
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", None, Some("60"), 0, 0),
        Some(60_000)
    );
    assert_eq!(
        openai_codex_retry_delay_ms(
            429,
            "rate limited",
            None,
            Some("Thu, 01 Jan 1970 00:00:45 GMT"),
            0,
            0
        ),
        Some(45_000)
    );
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", None, None, 0, 0),
        Some(1000)
    );
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", None, None, 1, 0),
        Some(2000)
    );
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", None, None, 2, 0),
        Some(4000)
    );
    assert_eq!(
        openai_codex_retry_delay_ms(429, "rate limited", None, None, 3, 0),
        None
    );
    assert_eq!(
        openai_codex_retry_delay_ms(400, "plain bad request", None, None, 0, 0),
        None
    );
    assert_eq!(
        openai_codex_retry_delay_ms(400, "upstream connect refused", None, None, 0, 0),
        Some(1000)
    );
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "rate limited", None, None, 0, 0, 1, Some(0)),
        Some(0)
    );
    assert_eq!(
        openai_codex_retry_delay_ms_with_limits(429, "rate limited", None, None, 1, 0, 1, None),
        None
    );
}

#[test]
fn openai_completions_payload_omits_empty_tools_unless_tool_history_exists() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();

    let empty_tools = build_openai_completions_payload(
        &model,
        &Context {
            messages: vec![Message::User(UserMessage::text("hi"))],
            tools: Vec::new(),
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(empty_tools.get("tools").is_none());

    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: "t1".to_owned(),
            name: "noop".to_owned(),
            arguments: Map::new(),
            thought_signature: None,
        })],
        api: "openai-completions".to_owned(),
        provider: "openai".to_owned(),
        model: "gpt-4o-mini".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let tool_history = build_openai_completions_payload(
        &model,
        &Context {
            messages: vec![
                Message::User(UserMessage::text("use the tool")),
                Message::Assistant(assistant),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "t1".to_owned(),
                    tool_name: "noop".to_owned(),
                    content: vec![ToolResultContent::text("done")],
                    details: None,
                    is_error: false,
                    timestamp: 2,
                }),
            ],
            tools: Vec::new(),
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(tool_history["tools"], json!([]));
}

#[test]
fn openai_completions_cloudflare_gateway_compat_uses_conservative_payload_and_headers() {
    let model = get_model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    )
    .expect("cloudflare gateway workers model");
    let payload = build_openai_completions_payload(
        &model,
        &Context {
            system_prompt: Some("You are helpful.".to_owned()),
            messages: vec![Message::User(UserMessage::text("hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            max_tokens: Some(64),
            session_id: Some("session-1".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(payload["messages"][0]["role"], "system");
    assert_eq!(payload["stream"], true);
    assert_eq!(payload["stream_options"], json!({ "include_usage": true }));
    assert!(payload.get("store").is_none());
    assert_eq!(payload["max_tokens"], 64);
    assert!(payload.get("max_completion_tokens").is_none());
    assert!(payload.get("reasoning_effort").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-1"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("session-1")
    );
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-1")
    );
    assert_eq!(
        headers.get("x-session-affinity").map(String::as_str),
        Some("session-1")
    );

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-1"),
        CacheRetention::None,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());
    assert!(headers.get("x-session-affinity").is_none());
}

#[test]
fn openai_completions_payload_sets_stream_usage_store_and_omits_falsey_options() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let payload = build_openai_completions_payload(
        &model,
        &user_context("hi"),
        OpenAICompletionsPayloadOptions {
            max_tokens: Some(0),
            tool_choice: Some(String::new()),
            ..Default::default()
        },
    );

    assert_eq!(payload["stream"], true);
    assert_eq!(payload["stream_options"], json!({ "include_usage": true }));
    assert_eq!(payload["store"], false);
    assert!(payload.get("max_tokens").is_none());
    assert!(payload.get("max_completion_tokens").is_none());
    assert!(payload.get("tool_choice").is_none());
}

#[test]
fn openai_completions_system_prompt_uses_developer_role_for_standard_reasoning_models() {
    let context = Context {
        system_prompt: Some("Follow policy.".to_owned()),
        messages: vec![Message::User(UserMessage::text("hi"))],
        ..Default::default()
    };
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    model.reasoning = true;

    let messages = convert_openai_completions_messages(&model, &context);
    assert_eq!(messages[0]["role"], "developer");

    let mut non_reasoning_model = model.clone();
    non_reasoning_model.reasoning = false;
    let messages = convert_openai_completions_messages(&non_reasoning_model, &context);
    assert_eq!(messages[0]["role"], "system");

    let mut compat_disabled_model = model.clone();
    compat_disabled_model.compat = Some(json!({ "supportsDeveloperRole": false }));
    let messages = convert_openai_completions_messages(&compat_disabled_model, &context);
    assert_eq!(messages[0]["role"], "system");

    let mut cloudflare_model = get_model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    )
    .expect("cloudflare gateway workers model");
    cloudflare_model.reasoning = true;
    let messages = convert_openai_completions_messages(&cloudflare_model, &context);
    assert_eq!(messages[0]["role"], "system");
}

#[test]
fn openai_completions_payload_forwards_tool_choice_and_strict_compat() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let tool = Tool {
        name: "ping".to_owned(),
        description: "Ping tool".to_owned(),
        parameters: json!({ "type": "object", "properties": { "ok": { "type": "boolean" } } }),
    };

    let payload = build_openai_completions_payload(
        &model,
        &Context {
            messages: vec![Message::User(UserMessage::text("Call ping"))],
            tools: vec![tool.clone()],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            tool_choice: Some("required".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["tool_choice"], "required");
    assert_eq!(payload["tools"][0]["function"]["strict"], false);

    let together_model = get_model("together", "openai/gpt-oss-120b").expect("together model");
    let payload = build_openai_completions_payload(
        &together_model,
        &Context {
            messages: vec![Message::User(UserMessage::text("Call ping"))],
            tools: vec![tool.clone()],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload["tools"][0]["function"].get("strict").is_none());

    model.compat = Some(json!({ "supportsStrictMode": false }));
    let payload = build_openai_completions_payload(
        &model,
        &Context {
            messages: vec![Message::User(UserMessage::text("Call ping"))],
            tools: vec![tool],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload["tools"][0]["function"].get("strict").is_none());
}

#[test]
fn openai_completions_payload_maps_reasoning_and_zai_tool_stream_compat() {
    let groq_qwen = get_model("groq", "qwen/qwen3-32b").expect("groq qwen");
    assert!(groq_qwen.reasoning);
    let payload = build_openai_completions_payload(
        &groq_qwen,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(payload["reasoning_effort"], "default");

    let openrouter = get_model("openrouter", "deepseek/deepseek-r1").expect("openrouter");
    let payload = build_openai_completions_payload(
        &openrouter,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );
    assert_eq!(payload["reasoning"], json!({ "effort": "high" }));
    assert!(payload.get("reasoning_effort").is_none());
    let payload = build_openai_completions_payload(
        &openrouter,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["reasoning"], json!({ "effort": "none" }));
    assert!(payload.get("reasoning_effort").is_none());

    let openrouter_opus =
        get_model("openrouter", "anthropic/claude-opus-4.6").expect("openrouter opus");
    let payload = build_openai_completions_payload(
        &openrouter_opus,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );
    assert_eq!(payload["reasoning"], json!({ "effort": "max" }));
    assert!(payload.get("reasoning_effort").is_none());

    let deepseek_v4 = get_model("deepseek", "deepseek-v4-flash").expect("deepseek v4");
    let payload = build_openai_completions_payload(
        &deepseek_v4,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::XHigh),
            ..Default::default()
        },
    );
    assert_eq!(payload["thinking"], json!({ "type": "enabled" }));
    assert_eq!(payload["reasoning_effort"], "max");

    let zai = get_model("zai", "glm-5.1").expect("zai");
    let payload = build_openai_completions_payload(
        &zai,
        &Context {
            messages: vec![Message::User(UserMessage::text("Call ping"))],
            tools: vec![Tool {
                name: "ping".to_owned(),
                description: "Ping".to_owned(),
                parameters: json!({ "type": "object" }),
            }],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["enable_thinking"], false);
    assert_eq!(payload["tool_stream"], true);

    let unsupported = get_model("zai", "glm-4.5-air").expect("zai unsupported");
    let payload = build_openai_completions_payload(
        &unsupported,
        &Context {
            messages: vec![Message::User(UserMessage::text("Call ping"))],
            tools: vec![Tool {
                name: "ping".to_owned(),
                description: "Ping".to_owned(),
                parameters: json!({ "type": "object" }),
            }],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload.get("tool_stream").is_none());
}

#[test]
fn openrouter_qwen_payload_uses_openrouter_reasoning_format_and_disables_by_default() {
    let model = get_model("openrouter", "qwen/qwen3.5-plus-02-15").expect("openrouter qwen");
    assert!(model.reasoning);
    assert_eq!(model.input, vec![InputKind::Text, InputKind::Image]);
    let context = user_context("Reply with pong.");

    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["model"], "qwen/qwen3.5-plus-02-15");
    assert_eq!(payload["reasoning"], json!({ "effort": "none" }));
    assert!(payload.get("reasoning_effort").is_none());

    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );
    assert_eq!(payload["reasoning"], json!({ "effort": "high" }));
    assert!(payload.get("reasoning_effort").is_none());
}

#[test]
fn openai_completions_payload_keeps_normal_groq_reasoning_effort_without_mapping() {
    let groq_gpt_oss = get_model("groq", "openai/gpt-oss-20b").expect("groq gpt oss");
    assert!(groq_gpt_oss.reasoning);
    let payload = build_openai_completions_payload(
        &groq_gpt_oss,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );

    assert_eq!(payload["reasoning_effort"], "medium");
    assert!(payload.get("reasoning").is_none());
}

#[test]
fn openai_completions_detects_openai_compatible_payload_defaults_from_base_url() {
    let context = Context {
        system_prompt: Some("Follow policy.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hi"))],
        ..Default::default()
    };
    let together_model = Model {
        id: "custom-together".to_owned(),
        name: "Custom Together".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "custom".to_owned(),
        base_url: "https://api.together.ai/v1".to_owned(),
        reasoning: true,
        thinking_level_map: Default::default(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 32_000,
        headers: Default::default(),
        compat: None,
    };

    let payload = build_openai_completions_payload(
        &together_model,
        &context,
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            max_tokens: Some(64),
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-together".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["messages"][0]["role"], "system");
    assert_eq!(payload["max_tokens"], 64);
    assert!(payload.get("max_completion_tokens").is_none());
    assert!(payload.get("store").is_none());
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
    assert_eq!(payload["reasoning"], json!({ "enabled": true }));
    assert!(payload.get("reasoning_effort").is_none());

    let payload = build_openai_completions_payload(
        &together_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["reasoning"], json!({ "enabled": false }));

    let mut explicit_reasoning_model = together_model.clone();
    explicit_reasoning_model.compat = Some(json!({ "supportsReasoningEffort": true }));
    let payload = build_openai_completions_payload(
        &explicit_reasoning_model,
        &context,
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    );
    assert_eq!(payload["reasoning"], json!({ "enabled": true }));
    assert_eq!(payload["reasoning_effort"], "high");
}

#[test]
fn openai_completions_payload_forwards_provider_routing_for_matching_gateways() {
    let context = user_context("Hi");
    let openrouter_routing = json!({
        "only": ["anthropic"],
        "allow_fallbacks": false,
        "sort": { "by": "latency", "partition": "none" },
    });
    let mut openrouter_model = Model {
        id: "custom-openrouter".to_owned(),
        name: "Custom OpenRouter".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openrouter".to_owned(),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        reasoning: false,
        thinking_level_map: Default::default(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 32_000,
        headers: Default::default(),
        compat: Some(json!({ "openRouterRouting": openrouter_routing.clone() })),
    };

    let payload = build_openai_completions_payload(
        &openrouter_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["provider"], openrouter_routing);

    openrouter_model.base_url = "https://proxy.example.com/v1".to_owned();
    let payload = build_openai_completions_payload(
        &openrouter_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload.get("provider").is_none());

    let mut vercel_model = Model {
        id: "custom-vercel".to_owned(),
        name: "Custom Vercel".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "vercel-ai-gateway".to_owned(),
        base_url: "https://ai-gateway.vercel.sh/v1".to_owned(),
        reasoning: false,
        thinking_level_map: Default::default(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 32_000,
        headers: Default::default(),
        compat: Some(json!({
            "vercelGatewayRouting": {
                "only": ["anthropic"],
                "order": ["anthropic", "openai"],
            },
        })),
    };
    let payload = build_openai_completions_payload(
        &vercel_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(
        payload["providerOptions"],
        json!({
            "gateway": {
                "only": ["anthropic"],
                "order": ["anthropic", "openai"],
            },
        })
    );

    vercel_model.compat = Some(json!({ "vercelGatewayRouting": {} }));
    let payload = build_openai_completions_payload(
        &vercel_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload.get("providerOptions").is_none());
}

#[test]
fn openai_completions_payload_maps_detected_and_explicit_thinking_formats() {
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Hi"))],
        ..Default::default()
    };
    let zai_model = get_model("zai", "glm-5.1").expect("zai model");
    let payload = build_openai_completions_payload(
        &zai_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["enable_thinking"], false);
    assert!(payload.get("reasoning_effort").is_none());

    let payload = build_openai_completions_payload(
        &zai_model,
        &context,
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    );
    assert_eq!(payload["enable_thinking"], true);
    assert!(payload.get("reasoning_effort").is_none());

    let qwen_template_model = Model {
        id: "custom-qwen-chat-template".to_owned(),
        name: "Custom Qwen Chat Template".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "custom".to_owned(),
        base_url: "https://example.com/v1".to_owned(),
        reasoning: true,
        thinking_level_map: Default::default(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 32_000,
        headers: Default::default(),
        compat: Some(json!({ "thinkingFormat": "qwen-chat-template" })),
    };
    let payload = build_openai_completions_payload(
        &qwen_template_model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "enable_thinking": false, "preserve_thinking": true })
    );

    let payload = build_openai_completions_payload(
        &qwen_template_model,
        &context,
        OpenAICompletionsPayloadOptions {
            reasoning: Some(ThinkingLevel::Low),
            ..Default::default()
        },
    );
    assert_eq!(
        payload["chat_template_kwargs"],
        json!({ "enable_thinking": true, "preserve_thinking": true })
    );
}

#[test]
fn openai_completions_zai_tool_stream_metadata_override_and_no_tools_match_provider() {
    for model_id in ["glm-5.1", "glm-4.7", "glm-5-turbo"] {
        let model = get_model("zai", model_id).expect("zai model");
        assert!(model.reasoning, "{model_id}");
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.get("supportsDeveloperRole"))
                .and_then(Value::as_bool),
            Some(false),
            "{model_id}"
        );
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.get("thinkingFormat"))
                .and_then(Value::as_str),
            Some("zai"),
            "{model_id}"
        );
        assert_eq!(
            model
                .compat
                .as_ref()
                .and_then(|compat| compat.get("zaiToolStream"))
                .and_then(Value::as_bool),
            Some(true),
            "{model_id}"
        );
    }
    let unsupported_model = get_model("zai", "glm-4.5-air").expect("unsupported zai model");
    assert!(unsupported_model.reasoning);
    assert_eq!(
        unsupported_model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("supportsDeveloperRole"))
            .and_then(Value::as_bool),
        Some(false)
    );
    assert_eq!(
        unsupported_model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("thinkingFormat"))
            .and_then(Value::as_str),
        Some("zai")
    );
    assert_eq!(
        unsupported_model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("zaiToolStream"))
            .and_then(Value::as_bool),
        None
    );

    let supported_without_tools = get_model("zai", "glm-5.1").expect("zai");
    let payload = build_openai_completions_payload(
        &supported_without_tools,
        &Context {
            messages: vec![Message::User(UserMessage::text("Hi"))],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload.get("tool_stream").is_none());

    let mut override_model = get_model("zai", "glm-4.5-air").expect("zai unsupported");
    override_model.compat = Some(json!({ "zaiToolStream": true }));
    let payload = build_openai_completions_payload(
        &override_model,
        &Context {
            messages: vec![Message::User(UserMessage::text("Call ping"))],
            tools: vec![Tool {
                name: "ping".to_owned(),
                description: "Ping".to_owned(),
                parameters: json!({ "type": "object" }),
            }],
            ..Default::default()
        },
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(payload["tool_stream"], true);
}

#[test]
fn openai_completions_payload_applies_anthropic_cache_control_format() {
    let model = Model {
        id: "custom-qwen".to_owned(),
        name: "Custom Qwen".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openrouter".to_owned(),
        base_url: "https://example.com/v1".to_owned(),
        reasoning: true,
        thinking_level_map: Default::default(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 32_000,
        headers: Default::default(),
        compat: Some(json!({ "cacheControlFormat": "anthropic" })),
    };
    let context = Context {
        system_prompt: Some("System prompt".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        tools: vec![Tool {
            name: "read".to_owned(),
            description: "Read a file".to_owned(),
            parameters: json!({ "type": "object" }),
        }],
    };
    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(
        payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        payload["tools"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        payload["messages"][1]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let assistant_tail_context = Context {
        system_prompt: Some("System prompt".to_owned()),
        messages: vec![
            Message::User(UserMessage::text("Hello")),
            Message::Assistant(AssistantMessage {
                content: vec![AssistantContent::Text(TextContent::new("Assistant replay"))],
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: Vec::new(),
                usage: Usage::zero(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 2,
            }),
        ],
        tools: vec![
            Tool {
                name: "read".to_owned(),
                description: "Read a file".to_owned(),
                parameters: json!({ "type": "object" }),
            },
            Tool {
                name: "write".to_owned(),
                description: "Write a file".to_owned(),
                parameters: json!({ "type": "object" }),
            },
        ],
    };
    let payload = build_openai_completions_payload(
        &model,
        &assistant_tail_context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert!(payload["tools"][0].get("cache_control").is_none());
    assert_eq!(
        payload["tools"][1]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert!(payload["messages"][1]["content"].as_str().is_some());
    assert_eq!(
        payload["messages"][2]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let openrouter_anthropic =
        get_model("openrouter", "anthropic/claude-sonnet-4").expect("openrouter anthropic model");
    let openrouter_payload = build_openai_completions_payload(
        &openrouter_anthropic,
        &context,
        OpenAICompletionsPayloadOptions::default(),
    );
    assert_eq!(
        openrouter_payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        openrouter_payload["tools"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        openrouter_payload["messages"][1]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let long_payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        },
    );
    assert_eq!(
        long_payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    assert_eq!(
        long_payload["tools"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );
    assert_eq!(
        long_payload["messages"][1]["content"][0]["cache_control"],
        json!({ "type": "ephemeral", "ttl": "1h" })
    );

    let mut no_long_cache_model = model.clone();
    no_long_cache_model.compat = Some(json!({
        "cacheControlFormat": "anthropic",
        "supportsLongCacheRetention": false,
    }));
    let long_payload = build_openai_completions_payload(
        &no_long_cache_model,
        &context,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            ..Default::default()
        },
    );
    assert_eq!(
        long_payload["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );

    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        },
    );
    assert!(payload["messages"][0]["content"].as_str().is_some());
    assert!(payload["tools"][0].get("cache_control").is_none());
    assert!(payload["messages"][1]["content"].as_str().is_some());
}

#[test]
fn openai_completions_payload_sets_prompt_cache_fields_for_direct_openai() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["PI_CACHE_RETENTION"]);
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let context = user_context("hi");

    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            session_id: Some("session-123".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["prompt_cache_key"], "session-123");
    assert!(payload.get("prompt_cache_retention").is_none());

    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-456".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["prompt_cache_key"], "session-456");
    assert_eq!(payload["prompt_cache_retention"], "24h");

    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::None),
            session_id: Some("session-789".to_owned()),
            ..Default::default()
        },
    );
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());
}

#[test]
fn openai_completions_payload_omits_proxy_cache_fields_and_uses_env_retention() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["PI_CACHE_RETENTION"]);
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let context = user_context("hi");

    let mut proxy_model = model.clone();
    proxy_model.base_url = "https://proxy.example.com/v1".to_owned();
    proxy_model.compat = Some(json!({ "supportsLongCacheRetention": false }));
    let payload = build_openai_completions_payload(
        &proxy_model,
        &context,
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-proxy".to_owned()),
            ..Default::default()
        },
    );
    assert!(payload.get("prompt_cache_key").is_none());
    assert!(payload.get("prompt_cache_retention").is_none());

    set_env("PI_CACHE_RETENTION", "long");
    let payload = build_openai_completions_payload(
        &model,
        &context,
        OpenAICompletionsPayloadOptions {
            session_id: Some("session-env".to_owned()),
            ..Default::default()
        },
    );
    assert_eq!(payload["prompt_cache_key"], "session-env");
    assert_eq!(payload["prompt_cache_retention"], "24h");
}

#[test]
fn openai_completions_payload_sets_long_retention_for_proxy_when_supported() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    model.base_url = "https://my-proxy.example.com/v1".to_owned();

    let payload = build_openai_completions_payload(
        &model,
        &user_context("hi"),
        OpenAICompletionsPayloadOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("session-completions".to_owned()),
            ..Default::default()
        },
    );

    assert_eq!(payload["prompt_cache_key"], "session-completions");
    assert_eq!(payload["prompt_cache_retention"], "24h");
}

#[test]
fn openai_completions_default_headers_apply_session_affinity_and_overrides() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    model.base_url = "https://proxy.example.com/v1".to_owned();
    model.compat = Some(json!({ "sendSessionAffinityHeaders": true }));

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-affinity"),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("session-affinity")
    );
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("session-affinity")
    );
    assert_eq!(
        headers.get("x-session-affinity").map(String::as_str),
        Some("session-affinity")
    );

    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-affinity"),
        CacheRetention::None,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());
    assert!(headers.get("x-session-affinity").is_none());

    let headers = build_openai_completions_default_headers(
        &model,
        Some(""),
        CacheRetention::Short,
        &BTreeMap::new(),
    );
    assert!(headers.get("session_id").is_none());
    assert!(headers.get("x-client-request-id").is_none());
    assert!(headers.get("x-session-affinity").is_none());

    let override_headers = BTreeMap::from([
        ("session_id".to_owned(), "override-session".to_owned()),
        (
            "x-client-request-id".to_owned(),
            "override-request".to_owned(),
        ),
        (
            "x-session-affinity".to_owned(),
            "override-affinity".to_owned(),
        ),
    ]);
    let headers = build_openai_completions_default_headers(
        &model,
        Some("session-affinity"),
        CacheRetention::Short,
        &override_headers,
    );
    assert_eq!(
        headers.get("session_id").map(String::as_str),
        Some("override-session")
    );
    assert_eq!(
        headers.get("x-client-request-id").map(String::as_str),
        Some("override-request")
    );
    assert_eq!(
        headers.get("x-session-affinity").map(String::as_str),
        Some("override-affinity")
    );
}

#[test]
fn openai_completions_chunk_usage_preserves_cache_write_tokens_and_totals() {
    let mut model = get_model("openrouter", "google/gemini-2.5-flash").expect("model");
    model.api = "openai-completions".to_owned();
    let usage = parse_openai_completions_chunk_usage(
        &json!({
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {
                "cached_tokens": 20,
                "cache_write_tokens": 30,
            },
        }),
        &model,
    );

    assert_eq!(usage.input, 50);
    assert_eq!(usage.output, 5);
    assert_eq!(usage.cache_read, 20);
    assert_eq!(usage.cache_write, 30);
    assert_usage_total_matches_components("openai completions cache write", &usage);

    let legacy_usage = parse_openai_completions_chunk_usage(
        &json!({
            "prompt_tokens": 15,
            "completion_tokens": 2,
            "prompt_cache_hit_tokens": 4,
        }),
        &model,
    );
    assert_eq!(legacy_usage.input, 11);
    assert_eq!(legacy_usage.cache_read, 4);
    assert_eq!(legacy_usage.cache_write, 0);
    assert_eq!(legacy_usage.total_tokens, 17);
    assert_usage_total_matches_components("openai completions legacy cache hit", &legacy_usage);
}

#[test]
fn openai_completions_chunk_usage_does_not_double_count_reasoning_tokens() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let usage = parse_openai_completions_chunk_usage(
        &json!({
            "prompt_tokens": 10,
            "completion_tokens": 33,
            "prompt_tokens_details": { "cached_tokens": 0 },
            "completion_tokens_details": { "reasoning_tokens": 21 },
        }),
        &model,
    );

    assert_eq!(usage.input, 10);
    assert_eq!(usage.output, 33);
    assert_eq!(usage.total_tokens, 43);
    assert_usage_total_matches_components("openai completions reasoning usage", &usage);
}

#[test]
fn openai_completions_chunk_metadata_preserves_choice_usage_cache_write_tokens() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let mut output = empty_assistant_for_model(&model);

    apply_openai_completions_chunk_metadata(
        &mut output,
        &model,
        &json!({
            "id": "chatcmpl-cache-write-choice",
            "choices": [{
                "delta": {},
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {
                        "cached_tokens": 50,
                        "cache_write_tokens": 30,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                    },
                },
            }],
        }),
    );

    assert_eq!(
        output.response_id.as_deref(),
        Some("chatcmpl-cache-write-choice")
    );
    assert_eq!(output.usage.input, 20);
    assert_eq!(output.usage.output, 5);
    assert_eq!(output.usage.cache_read, 50);
    assert_eq!(output.usage.cache_write, 30);
    assert_eq!(output.usage.total_tokens, 105);
    assert_usage_total_matches_components("openai completions choice usage", &output.usage);
}

#[tokio::test]
async fn openai_completions_stream_coalesces_tool_call_deltas_by_stable_index() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_openai_completions_chunks(
        [
            json!({
                "id": "chatcmpl-kimi-bad-stream",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "functions.read:0",
                            "type": "function",
                            "function": { "name": "read", "arguments": "" },
                        }],
                    },
                    "finish_reason": null,
                }],
            }),
            json!({
                "id": "chatcmpl-kimi-bad-stream",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "chatcmpl-tool-a",
                            "type": "function",
                            "function": { "name": null, "arguments": "{\"path\":\"README" },
                        }],
                    },
                    "finish_reason": null,
                }],
            }),
            json!({
                "id": "chatcmpl-kimi-bad-stream",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "chatcmpl-tool-b",
                            "type": "function",
                            "function": { "name": null, "arguments": ".md\"}" },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 },
                },
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process chunks");
    drop(sender);
    let events = collect_events(stream).await;

    let tool_call_content_indexes: Vec<usize> = events
        .iter()
        .filter_map(|event| match event {
            AssistantMessageEvent::ToolcallStart { content_index, .. }
            | AssistantMessageEvent::ToolcallDelta { content_index, .. }
            | AssistantMessageEvent::ToolcallEnd { content_index, .. } => Some(*content_index),
            _ => None,
        })
        .collect();

    assert_eq!(output.stop_reason, StopReason::ToolUse);
    assert_eq!(tool_call_content_indexes, vec![0, 0, 0, 0, 0]);
    assert_eq!(
        output.response_id.as_deref(),
        Some("chatcmpl-kimi-bad-stream")
    );
    assert_eq!(output.content.len(), 1);
    let AssistantContent::ToolCall(tool_call) = &output.content[0] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.id, "functions.read:0");
    assert_eq!(tool_call.name, "read");
    assert_eq!(tool_call.arguments, object(json!({ "path": "README.md" })));
}

#[tokio::test]
async fn openai_completions_stream_accumulates_mixed_deltas_independently() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_openai_completions_chunks(
        [
            json!({
                "id": "chatcmpl-mixed-deltas",
                "choices": [{
                    "delta": {
                        "content": "answer 1",
                        "reasoning_content": "think 1",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_read_initial",
                                "type": "function",
                                "function": { "name": "read", "arguments": "{\"path\":\"README" },
                            },
                            {
                                "index": 1,
                                "id": "tc_grep_initial",
                                "type": "function",
                                "function": { "name": "grep", "arguments": "{\"pattern\":\"TODO" },
                            },
                            {
                                "id": "tc_list_no_index",
                                "type": "function",
                                "function": { "name": "list", "arguments": "{\"path\":\"packages" },
                            },
                            {
                                "id": "tc_write_no_index",
                                "type": "function",
                                "function": { "name": "write", "arguments": "{\"path\":\"out" },
                            },
                        ],
                    },
                    "finish_reason": null,
                }],
            }),
            json!({
                "id": "chatcmpl-mixed-deltas",
                "choices": [{
                    "delta": {
                        "content": " answer 2",
                        "tool_calls": [
                            {
                                "index": 1,
                                "id": "tc_grep_changed",
                                "type": "function",
                                "function": { "arguments": "\",\"path\":\"src" },
                            },
                            {
                                "id": "tc_write_no_index",
                                "type": "function",
                                "function": { "arguments": ".txt\",\"content\":\"ok\"}" },
                            },
                            {
                                "id": "tc_list_no_index",
                                "type": "function",
                                "function": { "arguments": "/ai\"}" },
                            },
                        ],
                    },
                    "finish_reason": null,
                }],
            }),
            json!({
                "id": "chatcmpl-mixed-deltas",
                "choices": [{
                    "delta": {
                        "content": "\n",
                        "reasoning_content": " think 2",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "tc_read_changed",
                                "type": "function",
                                "function": { "arguments": ".md\"}" },
                            },
                            {
                                "index": 1,
                                "type": "function",
                                "function": { "arguments": "\"}" },
                            },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 8,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 2 },
                },
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process chunks");
    drop(sender);
    let events = collect_events(stream).await;

    assert_eq!(output.stop_reason, StopReason::ToolUse);
    assert_eq!(output.content.len(), 6);
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::TextStart { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::TextDelta { .. }))
            .count(),
        3
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ThinkingDelta { .. }))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ToolcallStart { .. }))
            .count(),
        4
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ToolcallDelta { .. }))
            .count(),
        9
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, AssistantMessageEvent::ToolcallEnd { .. }))
            .count(),
        4
    );

    let mut tool_events_by_content_index: BTreeMap<usize, Vec<&'static str>> = BTreeMap::new();
    for event in &events {
        match event {
            AssistantMessageEvent::ToolcallStart { content_index, .. } => {
                tool_events_by_content_index
                    .entry(*content_index)
                    .or_default()
                    .push("toolcall_start");
            }
            AssistantMessageEvent::ToolcallDelta { content_index, .. } => {
                tool_events_by_content_index
                    .entry(*content_index)
                    .or_default()
                    .push("toolcall_delta");
            }
            AssistantMessageEvent::ToolcallEnd { content_index, .. } => {
                tool_events_by_content_index
                    .entry(*content_index)
                    .or_default()
                    .push("toolcall_end");
            }
            _ => {}
        }
    }
    assert_eq!(
        tool_events_by_content_index.get(&2).cloned(),
        Some(vec![
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
        ])
    );
    assert_eq!(
        tool_events_by_content_index.get(&3).cloned(),
        Some(vec![
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
        ])
    );
    assert_eq!(
        tool_events_by_content_index.get(&4).cloned(),
        Some(vec![
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
        ])
    );
    assert_eq!(
        tool_events_by_content_index.get(&5).cloned(),
        Some(vec![
            "toolcall_start",
            "toolcall_delta",
            "toolcall_delta",
            "toolcall_end",
        ])
    );

    let AssistantContent::Text(text) = &output.content[0] else {
        panic!("text");
    };
    assert_eq!(text.text, "answer 1 answer 2\n");
    let AssistantContent::Thinking(thinking) = &output.content[1] else {
        panic!("thinking");
    };
    assert_eq!(thinking.thinking, "think 1 think 2");
    assert_eq!(
        thinking.thinking_signature.as_deref(),
        Some("reasoning_content")
    );
    let tool_calls: Vec<&ToolCall> = output
        .content
        .iter()
        .skip(2)
        .map(|content| match content {
            AssistantContent::ToolCall(tool_call) => tool_call,
            _ => panic!("tool call"),
        })
        .collect();
    assert_eq!(tool_calls[0].id, "tc_read_initial");
    assert_eq!(tool_calls[0].name, "read");
    assert_eq!(
        tool_calls[0].arguments,
        object(json!({ "path": "README.md" }))
    );
    assert_eq!(tool_calls[1].id, "tc_grep_initial");
    assert_eq!(tool_calls[1].name, "grep");
    assert_eq!(
        tool_calls[1].arguments,
        object(json!({ "pattern": "TODO", "path": "src" }))
    );
    assert_eq!(tool_calls[2].id, "tc_list_no_index");
    assert_eq!(tool_calls[2].name, "list");
    assert_eq!(
        tool_calls[2].arguments,
        object(json!({ "path": "packages/ai" }))
    );
    assert_eq!(tool_calls[3].id, "tc_write_no_index");
    assert_eq!(tool_calls[3].name, "write");
    assert_eq!(
        tool_calls[3].arguments,
        object(json!({ "path": "out.txt", "content": "ok" }))
    );
}

#[tokio::test]
async fn openai_completions_stream_normalizes_reasoning_field_by_provider() {
    let mut opencode_go = get_model("opencode-go", "kimi-k2.6").expect("opencode go");
    opencode_go.api = "openai-completions".to_owned();
    let mut opencode_output = empty_assistant_for_model(&opencode_go);
    let (sender, stream) = assistant_message_event_stream();
    process_openai_completions_chunks(
        [json!({
            "id": "chatcmpl-opencode-go-reasoning",
            "choices": [{ "delta": { "reasoning": "think" }, "finish_reason": "stop" }],
        })],
        &mut opencode_output,
        &sender,
        &opencode_go,
    )
    .expect("process opencode reasoning");
    drop(sender);
    let _events = collect_events(stream).await;

    assert_eq!(opencode_output.content.len(), 1);
    let AssistantContent::Thinking(opencode_thinking) = &opencode_output.content[0] else {
        panic!("thinking");
    };
    assert_eq!(opencode_thinking.thinking, "think");
    assert_eq!(
        opencode_thinking.thinking_signature.as_deref(),
        Some("reasoning_content")
    );

    let mut openai = get_model("openai", "gpt-4o-mini").expect("openai model");
    openai.api = "openai-completions".to_owned();
    let mut openai_output = empty_assistant_for_model(&openai);
    let (sender, stream) = assistant_message_event_stream();
    process_openai_completions_chunks(
        [json!({
            "id": "chatcmpl-reasoning",
            "choices": [{ "delta": { "reasoning": "think" }, "finish_reason": "stop" }],
        })],
        &mut openai_output,
        &sender,
        &openai,
    )
    .expect("process openai reasoning");
    drop(sender);
    let _events = collect_events(stream).await;

    assert_eq!(openai_output.content.len(), 1);
    let AssistantContent::Thinking(openai_thinking) = &openai_output.content[0] else {
        panic!("thinking");
    };
    assert_eq!(openai_thinking.thinking, "think");
    assert_eq!(
        openai_thinking.thinking_signature.as_deref(),
        Some("reasoning")
    );
}

#[tokio::test]
async fn openai_completions_stream_ignores_null_chunks_and_finishes() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    process_openai_completions_chunks(
        [
            Value::Null,
            json!({
                "id": "chatcmpl-test",
                "choices": [{ "delta": { "content": "OK" }, "finish_reason": null }],
            }),
            json!({
                "id": "chatcmpl-test",
                "choices": [{ "delta": {}, "finish_reason": "stop" }],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "prompt_tokens_details": { "cached_tokens": 0 },
                    "completion_tokens_details": { "reasoning_tokens": 0 },
                },
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process chunks");
    drop(sender);
    let _events = collect_events(stream).await;

    assert_eq!(output.stop_reason, StopReason::Stop);
    assert_eq!(output.error_message, None);
    assert_eq!(output.response_id.as_deref(), Some("chatcmpl-test"));
    assert_eq!(output.usage.total_tokens, 4);
    assert_eq!(text_of(&output), Some("OK"));
}

#[tokio::test]
async fn openai_completions_stream_maps_finish_reason_errors_and_requires_terminal_reason() {
    let mut model = get_model("zai", "glm-5.1").expect("zai model");
    model.api = "openai-completions".to_owned();
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();

    let error = process_openai_completions_chunks(
        [
            json!({
                "id": "chatcmpl-error",
                "choices": [{ "delta": { "content": "partial" }, "finish_reason": null }],
            }),
            json!({
                "id": "chatcmpl-error",
                "choices": [{ "delta": {}, "finish_reason": "network_error" }],
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect_err("provider finish error");
    drop(sender);
    let _events = collect_events(stream).await;
    assert_eq!(error, "Provider finish_reason: network_error");
    assert_eq!(output.stop_reason, StopReason::Error);
    assert_eq!(
        output.error_message.as_deref(),
        Some("Provider finish_reason: network_error")
    );

    let mut truncated = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();
    let error = process_openai_completions_chunks(
        [
            json!({
                "id": "chatcmpl-truncated",
                "choices": [{ "delta": { "content": "partial answer" }, "finish_reason": null }],
            }),
            json!({
                "id": "chatcmpl-truncated",
                "choices": [{ "delta": { "content": "partial answer" }, "finish_reason": null }],
            }),
        ],
        &mut truncated,
        &sender,
        &model,
    )
    .expect_err("missing finish reason");
    drop(sender);
    let _events = collect_events(stream).await;
    assert_eq!(error, "Stream ended without finish_reason");
}

#[tokio::test]
async fn openai_completions_stream_attaches_reasoning_details_to_tool_calls() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    let mut output = empty_assistant_for_model(&model);
    let (sender, stream) = assistant_message_event_stream();
    let reasoning_detail = json!({
        "type": "reasoning.encrypted",
        "id": "call_1",
        "data": "encrypted",
    });

    process_openai_completions_chunks(
        [
            json!({
                "id": "chatcmpl-reasoning-detail",
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read",
                                "arguments": "{\"path\":\"README.md\"}",
                            },
                        }],
                    },
                    "finish_reason": null,
                }],
            }),
            json!({
                "id": "chatcmpl-reasoning-detail",
                "choices": [{
                    "delta": {
                        "reasoning_details": [
                            reasoning_detail,
                            { "type": "summary", "id": "call_1", "data": "ignored" },
                            { "type": "reasoning.encrypted", "id": "missing", "data": "ignored" },
                        ],
                    },
                    "finish_reason": "tool_calls",
                }],
            }),
        ],
        &mut output,
        &sender,
        &model,
    )
    .expect("process chunks");
    drop(sender);
    let events = collect_events(stream).await;

    assert_eq!(output.stop_reason, StopReason::ToolUse);
    let AssistantContent::ToolCall(tool_call) = &output.content[0] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.id, "call_1");
    assert_eq!(tool_call.name, "read");
    assert_eq!(tool_call.arguments, object(json!({ "path": "README.md" })));
    assert_eq!(
        tool_call.thought_signature.as_deref(),
        Some(reasoning_detail.to_string().as_str())
    );

    let tool_call_end = events
        .iter()
        .find_map(|event| match event {
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => Some(tool_call),
            _ => None,
        })
        .expect("toolcall_end");
    assert_eq!(tool_call_end.thought_signature, tool_call.thought_signature);
}

#[test]
fn openai_completions_chunk_metadata_sets_response_model_without_changing_requested_model() {
    let model = Model {
        id: "openrouter/auto".to_owned(),
        name: "OpenRouter Auto".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "openrouter".to_owned(),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        reasoning: false,
        thinking_level_map: BTreeMap::new(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 200_000,
        max_tokens: 8_192,
        headers: BTreeMap::new(),
        compat: None,
    };

    let mut routed = empty_assistant_for_model(&model);
    apply_openai_completions_chunk_metadata(
        &mut routed,
        &model,
        &json!({
            "id": "chatcmpl-1",
            "model": "anthropic/claude-opus-4.7",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                },
            },
        }),
    );
    assert_eq!(routed.model, "openrouter/auto");
    assert_eq!(
        routed.response_model.as_deref(),
        Some("anthropic/claude-opus-4.7")
    );
    assert_eq!(routed.provider, "openrouter");
    assert_eq!(routed.response_id.as_deref(), Some("chatcmpl-1"));

    let mut echoed = empty_assistant_for_model(&model);
    apply_openai_completions_chunk_metadata(
        &mut echoed,
        &model,
        &json!({
            "id": "chatcmpl-2",
            "model": "openrouter/auto",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
            },
        }),
    );
    assert_eq!(echoed.model, "openrouter/auto");
    assert_eq!(echoed.response_model, None);

    let mut missing = empty_assistant_for_model(&model);
    apply_openai_completions_chunk_metadata(
        &mut missing,
        &model,
        &json!({ "id": "chatcmpl-3", "choices": [{ "delta": { "content": "hi" } }] }),
    );
    apply_openai_completions_chunk_metadata(
        &mut missing,
        &model,
        &json!({ "id": "chatcmpl-3", "model": "" }),
    );
    assert_eq!(missing.model, "openrouter/auto");
    assert_eq!(missing.response_model, None);
}

#[test]
fn openai_completions_messages_keep_null_content_for_standard_tool_call_turns() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("openai model");
    model.api = "openai-completions".to_owned();
    let context = |model: &Model| Context {
        messages: vec![
            Message::User(UserMessage::text("Read README.md")),
            Message::Assistant(AssistantMessage {
                content: vec![AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_owned(),
                    name: "read".to_owned(),
                    arguments: object(json!({ "path": "README.md" })),
                    thought_signature: None,
                })],
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
            }),
        ],
        ..Default::default()
    };

    let messages = convert_openai_completions_messages(&model, &context(&model));
    assert!(messages[1]["content"].is_null());
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");

    let mut bridge_model = model.clone();
    bridge_model.compat = Some(json!({ "requiresAssistantAfterToolResult": true }));
    let messages = convert_openai_completions_messages(&bridge_model, &context(&bridge_model));
    assert_eq!(messages[1]["content"], "");
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
}

#[test]
fn openai_completions_messages_batch_tool_result_images_after_tool_results() {
    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    model.input = vec![InputKind::Text, InputKind::Image];
    let assistant = AssistantMessage {
        content: vec![
            AssistantContent::ToolCall(ToolCall {
                id: "tool-1".to_owned(),
                name: "read".to_owned(),
                arguments: object(json!({ "path": "img-1.png" })),
                thought_signature: None,
            }),
            AssistantContent::ToolCall(ToolCall {
                id: "tool-2".to_owned(),
                name: "read".to_owned(),
                arguments: object(json!({ "path": "img-2.png" })),
                thought_signature: None,
            }),
        ],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 1,
    };
    let tool_result = |tool_call_id: &str, timestamp: i64| {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: tool_call_id.to_owned(),
            tool_name: "read".to_owned(),
            content: vec![
                ToolResultContent::text("Read image file [image/png]"),
                ToolResultContent::Image(ImageContent {
                    data: "ZmFrZQ==".to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ],
            details: None,
            is_error: false,
            timestamp,
        })
    };
    let context = Context {
        messages: vec![
            Message::User(UserMessage::text("Read the images")),
            Message::Assistant(assistant),
            tool_result("tool-1", 2),
            tool_result("tool-2", 3),
        ],
        ..Default::default()
    };

    let messages = convert_openai_completions_messages(&model, &context);
    let roles = messages
        .iter()
        .map(|message| message["role"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(roles, vec!["user", "assistant", "tool", "tool", "user"]);
    let image_message = messages.last().expect("image message");
    assert_eq!(image_message["role"], "user");
    let content_parts = image_message["content"].as_array().expect("content parts");
    assert_eq!(
        content_parts.first().and_then(|part| part["text"].as_str()),
        Some("Attached image(s) from tool result:")
    );
    let image_parts = content_parts
        .iter()
        .filter(|part| part.get("type").and_then(Value::as_str) == Some("image_url"))
        .collect::<Vec<_>>();
    assert_eq!(image_parts.len(), 2);
    assert_eq!(
        image_parts
            .iter()
            .map(|part| part["image_url"]["url"].as_str())
            .collect::<Vec<_>>(),
        vec![
            Some("data:image/png;base64,ZmFrZQ=="),
            Some("data:image/png;base64,ZmFrZQ=="),
        ]
    );
}

fn openai_completions_thinking_text_model() -> Model {
    Model {
        id: "repro-model".to_owned(),
        name: "Repro Model".to_owned(),
        api: "openai-completions".to_owned(),
        provider: "repro-provider".to_owned(),
        base_url: "http://127.0.0.1:1".to_owned(),
        reasoning: true,
        thinking_level_map: BTreeMap::new(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 4_096,
        headers: BTreeMap::new(),
        compat: Some(json!({ "requiresThinkingAsText": true })),
    }
}

fn openai_completions_replay_context(model: &Model, content: Vec<AssistantContent>) -> Context {
    Context {
        messages: vec![
            Message::User(UserMessage::text("hello")),
            Message::Assistant(AssistantMessage {
                content,
                api: model.api.clone(),
                provider: model.provider.clone(),
                model: model.id.clone(),
                response_model: None,
                response_id: None,
                diagnostics: Vec::new(),
                usage: Usage::zero(),
                stop_reason: StopReason::Stop,
                error_message: None,
                timestamp: 2,
            }),
            Message::User(UserMessage::text("continue")),
        ],
        ..Default::default()
    }
}

#[test]
fn openai_completions_messages_replay_thinking_as_text_parts() {
    let model = openai_completions_thinking_text_model();
    let messages = convert_openai_completions_messages(
        &model,
        &openai_completions_replay_context(
            &model,
            vec![
                AssistantContent::Thinking(ThinkingContent::new("internal reasoning")),
                AssistantContent::Text(TextContent::new("visible answer")),
            ],
        ),
    );

    assert_eq!(
        messages[1],
        json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "internal reasoning" },
                { "type": "text", "text": "visible answer" },
            ],
        })
    );

    let messages = convert_openai_completions_messages(
        &model,
        &openai_completions_replay_context(
            &model,
            vec![AssistantContent::Thinking(ThinkingContent::new(
                "internal reasoning",
            ))],
        ),
    );

    assert_eq!(
        messages[1],
        json!({
            "role": "assistant",
            "content": [
                { "type": "text", "text": "internal reasoning" },
            ],
        })
    );
}

#[tokio::test]
async fn openai_completions_thinking_as_text_replay_posts_text_parts() {
    let sse = concat!(
        "data: {\"id\":\"chatcmpl-repro\",\"model\":\"repro-model\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = openai_completions_thinking_text_model();
    model.base_url = base_url;
    let context = openai_completions_replay_context(
        &model,
        vec![
            AssistantContent::Thinking(ThinkingContent::new("internal reasoning")),
            AssistantContent::Text(TextContent::new("visible answer")),
        ],
    );
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = complete_simple(&model, context, options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("ok"));
    assert!(request.contains("\"role\":\"assistant\""));
    assert!(request.contains("\"content\":[{\"type\":\"text\",\"text\":\"internal reasoning\"},{\"type\":\"text\",\"text\":\"visible answer\"}]"));
}

#[test]
fn openai_completions_messages_replay_reasoning_signature_and_details() {
    let model = get_model("opencode-go", "kimi-k2.6").expect("opencode go model");
    let reasoning_detail = json!({
        "type": "reasoning.encrypted",
        "id": "call_1",
        "data": "encrypted",
    });
    let messages = convert_openai_completions_messages(
        &model,
        &openai_completions_replay_context(
            &model,
            vec![
                AssistantContent::Thinking(ThinkingContent {
                    thinking: "think".to_owned(),
                    thinking_signature: Some("reasoning".to_owned()),
                    redacted: false,
                }),
                AssistantContent::ToolCall(ToolCall {
                    id: "call_1".to_owned(),
                    name: "read".to_owned(),
                    arguments: object(json!({ "path": "README.md" })),
                    thought_signature: Some(reasoning_detail.to_string()),
                }),
            ],
        ),
    );

    assert_eq!(messages[1]["reasoning_content"], "think");
    assert!(messages[1].get("reasoning").is_none());
    assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
    assert_eq!(messages[1]["reasoning_details"], json!([reasoning_detail]));
}

#[test]
fn openai_completions_messages_add_empty_reasoning_content_when_required() {
    let mut model = get_model("opencode-go", "deepseek-v4-flash").expect("deepseek compat model");
    model.reasoning = true;
    model.compat = Some(json!({
        "requiresReasoningContentOnAssistantMessages": true,
    }));
    let messages = convert_openai_completions_messages(
        &model,
        &openai_completions_replay_context(
            &model,
            vec![AssistantContent::Text(TextContent::new("visible answer"))],
        ),
    );

    assert_eq!(messages[1]["content"], "visible answer");
    assert_eq!(messages[1]["reasoning_content"], "");

    let mut inferred_model = Model::faux("openai-completions", "deepseek", "deepseek-v4-pro");
    inferred_model.base_url = "https://api.deepseek.com".to_owned();
    inferred_model.reasoning = true;
    let messages = convert_openai_completions_messages(
        &inferred_model,
        &openai_completions_replay_context(
            &inferred_model,
            vec![AssistantContent::Text(TextContent::new("visible answer"))],
        ),
    );

    assert_eq!(messages[1]["content"], "visible answer");
    assert_eq!(messages[1]["reasoning_content"], "");
}

#[tokio::test]
async fn faux_provider_registers_and_estimates_usage() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("hello world", Default::default()).into(),
    ]);

    let response = complete(
        &registration.get_model(),
        user_context("hi there"),
        Default::default(),
    )
    .await
    .expect("complete");

    assert_eq!(text_of(&response), Some("hello world"));
    assert!(response.usage.input > 0);
    assert!(response.usage.output > 0);
    assert_eq!(
        response.usage.total_tokens,
        response.usage.input + response.usage.output
    );
    assert_eq!(registration.state().call_count(), 1);

    let tool = Tool {
        name: "echo".to_owned(),
        description: "Echo back text".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" }
            }
        }),
    };
    let rich_context = Context {
        system_prompt: Some("sys".to_owned()),
        messages: vec![
            Message::User(UserMessage {
                content: UserContentValue::Blocks(vec![
                    UserContent::Text(TextContent::new("hello")),
                    UserContent::Image(ImageContent {
                        data: "abcd".to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ]),
                timestamp: 1,
            }),
            Message::Assistant(faux_assistant_message("prior", Default::default())),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: "tool-1".to_owned(),
                tool_name: "echo".to_owned(),
                content: vec![ToolResultContent::text("tool out")],
                details: None,
                is_error: false,
                timestamp: 2,
            }),
        ],
        tools: vec![tool],
    };
    registration.append_responses(vec![
        faux_assistant_message("done", Default::default()).into(),
    ]);
    let rich_response = complete(
        &registration.get_model(),
        rich_context.clone(),
        Default::default(),
    )
    .await
    .expect("rich complete");
    let prompt_text = [
        "system:sys".to_owned(),
        "user:hello\n[image:image/png:4]".to_owned(),
        "assistant:prior".to_owned(),
        "toolResult:echo\ntool out".to_owned(),
        format!(
            "tools:{}",
            serde_json::to_string(&rich_context.tools).expect("tools")
        ),
    ]
    .join("\n\n");
    let estimate_tokens = |text: &str| text.chars().count().div_ceil(4) as u64;
    assert_eq!(rich_response.usage.input, estimate_tokens(&prompt_text));
    assert_eq!(rich_response.usage.output, estimate_tokens("done"));
    assert_eq!(rich_response.usage.cache_read, 0);
    assert_eq!(rich_response.usage.cache_write, 0);
    assert_eq!(
        rich_response.usage.total_tokens,
        rich_response.usage.input + rich_response.usage.output
    );
    registration.unregister();
}

#[tokio::test]
async fn faux_provider_supports_helper_blocks_and_stream_order() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_thinking("go"),
                faux_text("ok"),
                faux_tool_call("echo", Map::new(), Some("tool-1".to_owned())),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);

    let events = collect_events(
        stream(
            &registration.get_model(),
            user_context("hi"),
            Default::default(),
        )
        .expect("stream"),
    )
    .await;

    let event_names: Vec<&'static str> = events
        .iter()
        .map(|event| match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        })
        .collect();

    assert_eq!(
        event_names,
        vec![
            "start",
            "thinking_start",
            "thinking_delta",
            "thinking_end",
            "text_start",
            "text_delta",
            "text_end",
            "toolcall_start",
            "toolcall_delta",
            "toolcall_end",
            "done"
        ]
    );
    registration.unregister();
}

#[tokio::test]
async fn faux_provider_supports_multiple_models_factories_and_message_rewrites() {
    let mut fast = FauxModelDefinition::new("faux-fast");
    fast.name = Some("Faux Fast".to_owned());
    fast.reasoning = false;
    let mut thinker = FauxModelDefinition::new("faux-thinker");
    thinker.name = Some("Faux Thinker".to_owned());
    thinker.reasoning = true;

    let registration = register_faux_provider(RegisterFauxProviderOptions {
        api: Some("faux:test-rewrite".to_owned()),
        provider: Some("faux-provider".to_owned()),
        models: vec![fast, thinker],
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_response_factory(|_context, _options, state, model| {
            faux_assistant_message(
                format!("{}:{}:{}", model.id, model.reasoning, state.call_count()),
                Default::default(),
            )
        }),
        faux_response_factory(|_context, _options, state, model| {
            faux_assistant_message(
                format!("{}:{}:{}", model.id, model.reasoning, state.call_count()),
                Default::default(),
            )
        }),
    ]);

    assert_eq!(
        registration
            .models
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["faux-fast", "faux-thinker"]
    );
    assert!(
        !registration
            .get_model_by_id("faux-fast")
            .expect("fast")
            .reasoning
    );
    assert!(
        registration
            .get_model_by_id("faux-thinker")
            .expect("thinker")
            .reasoning
    );

    let fast_response = complete(
        &registration.get_model_by_id("faux-fast").expect("fast"),
        user_context("hi"),
        Default::default(),
    )
    .await
    .expect("fast response");
    assert_eq!(text_of(&fast_response), Some("faux-fast:false:1"));
    assert_eq!(fast_response.api, "faux:test-rewrite");
    assert_eq!(fast_response.provider, "faux-provider");
    assert_eq!(fast_response.model, "faux-fast");

    let thinker_response = complete(
        &registration
            .get_model_by_id("faux-thinker")
            .expect("thinker"),
        user_context("hi"),
        Default::default(),
    )
    .await
    .expect("thinker response");
    assert_eq!(text_of(&thinker_response), Some("faux-thinker:true:2"));
    assert_eq!(thinker_response.api, "faux:test-rewrite");
    assert_eq!(thinker_response.provider, "faux-provider");
    assert_eq!(thinker_response.model, "faux-thinker");

    registration.unregister();
}

#[tokio::test]
async fn faux_provider_supports_async_response_factories() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![faux_async_response_factory(
        |context, options, state, model| async move {
            tokio::task::yield_now().await;
            faux_assistant_message(
                format!(
                    "{}:{}:{}:{}",
                    context.messages.len(),
                    options.stream.session_id.as_deref().unwrap_or("none"),
                    state.call_count(),
                    model.id
                ),
                Default::default(),
            )
        },
    )]);

    let response = complete(
        &registration.get_model(),
        user_context("hi"),
        StreamOptions {
            session_id: Some("session-async".to_owned()),
            ..Default::default()
        },
    )
    .await
    .expect("async factory response");

    assert_eq!(text_of(&response), Some("1:session-async:1:faux-1"));
    registration.unregister();
}

#[tokio::test]
async fn faux_provider_emits_error_when_response_factory_panics() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![faux_response_factory(|_, _, _, _| {
        panic!("boom");
    })]);
    let seen_responses = Arc::new(Mutex::new(Vec::<ProviderResponse>::new()));
    let mut options = SimpleStreamOptions::default();
    options
        .response_hooks
        .push(Arc::new(RecordingProviderResponseHook {
            seen: seen_responses.clone(),
        }));

    let events = collect_events(
        stream_simple(&registration.get_model(), user_context("hi"), options).expect("stream"),
    )
    .await;

    assert_eq!(events.len(), 1);
    match &events[0] {
        AssistantMessageEvent::Error { reason, error } => {
            assert_eq!(*reason, StopReason::Error);
            assert_eq!(error.stop_reason, StopReason::Error);
            assert_eq!(error.error_message.as_deref(), Some("boom"));
        }
        event => panic!("expected error event, got {event:?}"),
    }
    let seen_responses = seen_responses.lock().expect("seen responses");
    assert_eq!(seen_responses.len(), 1);
    assert_eq!(seen_responses[0].status, 200);
    assert!(seen_responses[0].headers.is_empty());
    registration.unregister();
}

#[tokio::test]
async fn faux_provider_replaces_appends_exhausts_and_unregisters() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    let model = registration.get_model();
    let context = user_context("hi");

    registration.set_responses(vec![
        faux_assistant_message("first", Default::default()).into(),
    ]);
    let first = complete(&model, context.clone(), Default::default())
        .await
        .expect("first");
    assert_eq!(text_of(&first), Some("first"));
    assert_eq!(registration.pending_response_count(), 0);

    registration.set_responses(vec![
        faux_assistant_message("second", Default::default()).into(),
    ]);
    assert_eq!(registration.pending_response_count(), 1);
    let second = complete(&model, context.clone(), Default::default())
        .await
        .expect("second");
    assert_eq!(text_of(&second), Some("second"));

    registration.append_responses(vec![
        faux_assistant_message("third", Default::default()).into(),
        faux_assistant_message("fourth", Default::default()).into(),
    ]);
    assert_eq!(registration.pending_response_count(), 2);
    let third = complete(&model, context.clone(), Default::default())
        .await
        .expect("third");
    let fourth = complete(&model, context.clone(), Default::default())
        .await
        .expect("fourth");
    assert_eq!(text_of(&third), Some("third"));
    assert_eq!(text_of(&fourth), Some("fourth"));

    let exhausted_events = collect_events(
        stream(&model, context.clone(), Default::default()).expect("exhausted stream"),
    )
    .await;
    assert_eq!(exhausted_events.len(), 1);
    match &exhausted_events[0] {
        AssistantMessageEvent::Error { reason, error } => {
            assert_eq!(*reason, StopReason::Error);
            assert_eq!(error.stop_reason, StopReason::Error);
            assert_eq!(
                error.error_message.as_deref(),
                Some("No more faux responses queued")
            );
        }
        event => panic!("expected exhausted faux response to emit only error, got {event:?}"),
    }
    assert_eq!(registration.pending_response_count(), 0);

    let api = registration.api().to_owned();
    registration.unregister();
    let error = complete(&model, context, Default::default())
        .await
        .expect_err("unregistered provider");
    assert_eq!(
        error.to_string(),
        format!("No API provider registered for api: {api}")
    );
}

#[tokio::test]
async fn faux_provider_streams_multiple_tool_call_deltas() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message(
            vec![
                faux_tool_call(
                    "echo",
                    object(json!({ "text": "one", "count": 12 })),
                    Some("tool-1".to_owned()),
                ),
                faux_tool_call(
                    "echo",
                    object(json!({ "text": "two", "count": 34 })),
                    Some("tool-2".to_owned()),
                ),
            ],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);

    let events = collect_events(
        stream(
            &registration.get_model(),
            user_context("hi"),
            Default::default(),
        )
        .expect("stream"),
    )
    .await;

    let mut starts = 0;
    let mut ends = 0;
    let mut deltas: BTreeMap<usize, String> = BTreeMap::new();
    for event in &events {
        match event {
            AssistantMessageEvent::ToolcallStart { .. } => starts += 1,
            AssistantMessageEvent::ToolcallDelta {
                content_index,
                delta,
                ..
            } => {
                deltas
                    .entry(*content_index)
                    .or_default()
                    .push_str(delta.as_str());
            }
            AssistantMessageEvent::ToolcallEnd { .. } => ends += 1,
            _ => {}
        }
    }

    assert_eq!(starts, 2);
    assert_eq!(ends, 2);
    assert!(deltas.values().all(|delta| delta.len() > 4));
    assert_eq!(
        serde_json::from_str::<Value>(&deltas[&0]).expect("tool-1 args"),
        json!({ "count": 12, "text": "one" })
    );
    assert_eq!(
        serde_json::from_str::<Value>(&deltas[&1]).expect("tool-2 args"),
        json!({ "count": 34, "text": "two" })
    );
    registration.unregister();
}

#[tokio::test]
async fn faux_provider_streams_terminal_error_and_aborted_messages() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        token_size: Some(TokenSize { min: 2, max: 2 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message(
            "partial",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("upstream failed".to_owned()),
                ..Default::default()
            },
        )
        .into(),
        faux_assistant_message(
            "partial",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("Request was aborted".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);

    for (expected_reason, expected_message) in [
        (StopReason::Error, "upstream failed"),
        (StopReason::Aborted, "Request was aborted"),
    ] {
        let events = collect_events(
            stream(
                &registration.get_model(),
                user_context("hi"),
                Default::default(),
            )
            .expect("stream"),
        )
        .await;

        let event_names: Vec<&'static str> = events
            .iter()
            .map(|event| match event {
                AssistantMessageEvent::Start { .. } => "start",
                AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
                AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
                AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
                AssistantMessageEvent::TextStart { .. } => "text_start",
                AssistantMessageEvent::TextDelta { .. } => "text_delta",
                AssistantMessageEvent::TextEnd { .. } => "text_end",
                AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
                AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
                AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
                AssistantMessageEvent::Done { .. } => "done",
                AssistantMessageEvent::Error { .. } => "error",
            })
            .collect();
        assert_eq!(
            event_names,
            vec!["start", "text_start", "text_delta", "text_end", "error"]
        );
        match events.last().expect("terminal event") {
            AssistantMessageEvent::Error { reason, error } => {
                assert_eq!(*reason, expected_reason);
                assert_eq!(error.stop_reason, expected_reason);
                assert_eq!(error.error_message.as_deref(), Some(expected_message));
            }
            event => panic!("expected terminal error event, got {event:?}"),
        }
    }

    registration.unregister();
}

#[tokio::test]
async fn faux_provider_respects_abort_flag_before_and_during_streaming() {
    fn event_type(event: &AssistantMessageEvent) -> &'static str {
        match event {
            AssistantMessageEvent::Start { .. } => "start",
            AssistantMessageEvent::ThinkingStart { .. } => "thinking_start",
            AssistantMessageEvent::ThinkingDelta { .. } => "thinking_delta",
            AssistantMessageEvent::ThinkingEnd { .. } => "thinking_end",
            AssistantMessageEvent::TextStart { .. } => "text_start",
            AssistantMessageEvent::TextDelta { .. } => "text_delta",
            AssistantMessageEvent::TextEnd { .. } => "text_end",
            AssistantMessageEvent::ToolcallStart { .. } => "toolcall_start",
            AssistantMessageEvent::ToolcallDelta { .. } => "toolcall_delta",
            AssistantMessageEvent::ToolcallEnd { .. } => "toolcall_end",
            AssistantMessageEvent::Done { .. } => "done",
            AssistantMessageEvent::Error { .. } => "error",
        }
    }

    let registration = register_faux_provider(RegisterFauxProviderOptions {
        tokens_per_second: Some(100.0),
        token_size: Some(TokenSize { min: 1, max: 1 }),
        ..Default::default()
    });
    registration.set_responses(vec![
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
        faux_assistant_message("abcdefghijklmnopqrstuvwxyz", Default::default()).into(),
        faux_assistant_message(
            vec![faux_thinking("abcdefghijklmnopqrstuvwxyz")],
            Default::default(),
        )
        .into(),
        faux_assistant_message(
            vec![faux_tool_call(
                "echo",
                object(json!({
                    "text": "abcdefghijklmnopqrstuvwxyz",
                    "count": 123456789
                })),
                Some("tool-1".to_owned()),
            )],
            FauxAssistantOptions {
                stop_reason: Some(StopReason::ToolUse),
                ..Default::default()
            },
        )
        .into(),
    ]);

    let pre_aborted = Arc::new(AtomicBool::new(true));
    let events = collect_events(
        stream(
            &registration.get_model(),
            user_context("hi"),
            StreamOptions {
                abort_flag: Some(pre_aborted),
                ..Default::default()
            },
        )
        .expect("pre-aborted stream"),
    )
    .await;
    assert_eq!(
        events.iter().map(event_type).collect::<Vec<_>>(),
        vec!["error"]
    );
    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
    ));

    let text_abort = Arc::new(AtomicBool::new(false));
    let mut text_events = Vec::new();
    let mut text_stream = stream(
        &registration.get_model(),
        user_context("hi"),
        StreamOptions {
            abort_flag: Some(text_abort.clone()),
            ..Default::default()
        },
    )
    .expect("text stream");
    while let Some(event) = text_stream.next().await {
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            text_abort.store(true, Ordering::SeqCst);
        }
        text_events.push(event);
    }
    let text_event_types = text_events.iter().map(event_type).collect::<Vec<_>>();
    assert!(text_event_types.contains(&"text_start"));
    assert!(text_event_types.contains(&"text_delta"));
    assert!(text_event_types.contains(&"error"));
    assert!(!text_event_types.contains(&"text_end"));

    let thinking_abort = Arc::new(AtomicBool::new(false));
    let mut thinking_events = Vec::new();
    let mut thinking_stream = stream(
        &registration.get_model(),
        user_context("hi"),
        StreamOptions {
            abort_flag: Some(thinking_abort.clone()),
            ..Default::default()
        },
    )
    .expect("thinking stream");
    while let Some(event) = thinking_stream.next().await {
        if matches!(event, AssistantMessageEvent::ThinkingDelta { .. }) {
            thinking_abort.store(true, Ordering::SeqCst);
        }
        thinking_events.push(event);
    }
    let thinking_event_types = thinking_events.iter().map(event_type).collect::<Vec<_>>();
    assert!(thinking_event_types.contains(&"thinking_start"));
    assert!(thinking_event_types.contains(&"thinking_delta"));
    assert!(thinking_event_types.contains(&"error"));
    assert!(!thinking_event_types.contains(&"thinking_end"));

    let tool_abort = Arc::new(AtomicBool::new(false));
    let mut tool_events = Vec::new();
    let mut tool_stream = stream(
        &registration.get_model(),
        user_context("hi"),
        StreamOptions {
            abort_flag: Some(tool_abort.clone()),
            ..Default::default()
        },
    )
    .expect("tool stream");
    while let Some(event) = tool_stream.next().await {
        if matches!(event, AssistantMessageEvent::ToolcallDelta { .. }) {
            tool_abort.store(true, Ordering::SeqCst);
        }
        tool_events.push(event);
    }
    let tool_event_types = tool_events.iter().map(event_type).collect::<Vec<_>>();
    assert!(tool_event_types.contains(&"toolcall_start"));
    assert!(tool_event_types.contains(&"toolcall_delta"));
    assert!(tool_event_types.contains(&"error"));
    assert!(!tool_event_types.contains(&"toolcall_end"));

    registration.unregister();
}

#[tokio::test]
async fn faux_provider_consumes_responses_and_caches_per_session() {
    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("first", Default::default()).into(),
        faux_assistant_message("second", Default::default()).into(),
        faux_assistant_message("third", Default::default()).into(),
        faux_assistant_message("fourth", Default::default()).into(),
        faux_assistant_message("fifth", Default::default()).into(),
    ]);

    let mut context = Context {
        system_prompt: Some("Be concise.".to_owned()),
        messages: vec![Message::User(UserMessage::text("hello"))],
        tools: Vec::new(),
    };

    let first = complete(
        &registration.get_model(),
        context.clone(),
        StreamOptions {
            session_id: Some("session-1".to_owned()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        },
    )
    .await
    .expect("first");
    assert_eq!(first.usage.cache_read, 0);
    assert!(first.usage.cache_write > 0);

    context.messages.push(Message::Assistant(first));
    context
        .messages
        .push(Message::User(UserMessage::text("follow up")));

    let second = complete(
        &registration.get_model(),
        context.clone(),
        StreamOptions {
            session_id: Some("session-1".to_owned()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        },
    )
    .await
    .expect("second");
    assert!(second.usage.cache_read > 0);

    let third = complete(
        &registration.get_model(),
        context.clone(),
        StreamOptions {
            session_id: Some("session-2".to_owned()),
            cache_retention: Some(CacheRetention::Short),
            ..Default::default()
        },
    )
    .await
    .expect("third");
    assert_eq!(third.usage.cache_read, 0);
    assert!(third.usage.cache_write > 0);

    let fourth = complete(
        &registration.get_model(),
        user_context("hi"),
        Default::default(),
    )
    .await
    .expect("fourth");
    assert_eq!(text_of(&fourth), Some("fourth"));
    assert_eq!(fourth.usage.cache_read, 0);
    assert_eq!(fourth.usage.cache_write, 0);

    let fifth = complete(
        &registration.get_model(),
        context,
        StreamOptions {
            session_id: Some("session-1".to_owned()),
            cache_retention: Some(CacheRetention::None),
            ..Default::default()
        },
    )
    .await
    .expect("fifth");
    assert_eq!(text_of(&fifth), Some("fifth"));
    assert_eq!(fifth.usage.cache_read, 0);
    assert_eq!(fifth.usage.cache_write, 0);
    assert_eq!(registration.pending_response_count(), 0);
    registration.unregister();
}

#[test]
fn validation_coerces_json_schema_primitives() {
    let tool = Tool {
        name: "mix".to_owned(),
        description: "mixed values".to_owned(),
        parameters: json!({
            "type": "object",
            "required": ["count", "flag", "label"],
            "properties": {
                "count": { "type": "integer" },
                "flag": { "type": "boolean" },
                "label": { "type": "string" }
            }
        }),
    };
    let call = ToolCall {
        id: "tool-1".to_owned(),
        name: "mix".to_owned(),
        arguments: object(json!({ "count": "7", "flag": "true", "label": 5 })),
        thought_signature: None,
    };

    let args = validate_tool_arguments(&tool, &call).expect("valid");
    assert_eq!(args["count"], 7);
    assert_eq!(args["flag"], true);
    assert_eq!(args["label"], "5");
}

#[test]
fn string_enum_schema_matches_pi_typebox_helper_shape() {
    let schema = string_enum_schema(
        ["add", "subtract", "multiply", "divide"],
        StringEnumOptions::new()
            .description("The operation to perform")
            .default_value("add"),
    );

    assert_eq!(
        schema,
        json!({
            "type": "string",
            "enum": ["add", "subtract", "multiply", "divide"],
            "description": "The operation to perform",
            "default": "add",
        })
    );
    assert_eq!(
        string_enum_schema(
            ["red", "blue"],
            StringEnumOptions::new().description("").default_value(""),
        ),
        json!({
            "type": "string",
            "enum": ["red", "blue"],
        })
    );

    let tool = Tool {
        name: "calculate".to_owned(),
        description: "Run a calculation".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "operation": schema,
            },
            "required": ["operation"],
        }),
    };
    let call = ToolCall {
        id: "call_enum".to_owned(),
        name: "calculate".to_owned(),
        arguments: object(json!({ "operation": "subtract" })),
        thought_signature: None,
    };
    let validated = validate_tool_arguments(&tool, &call).expect("valid enum");
    assert_eq!(validated["operation"], "subtract");

    let invalid = ToolCall {
        arguments: object(json!({ "operation": "divide-and-round" })),
        ..call
    };
    assert!(validate_tool_arguments(&tool, &invalid).is_err());
}

#[test]
fn validation_matches_ajv_style_plain_schema_coercions() {
    let passing_cases = vec![
        (json!({ "type": "number" }), json!("42"), json!(42.0)),
        (json!({ "type": "number" }), json!(true), json!(1.0)),
        (json!({ "type": "number" }), Value::Null, json!(0.0)),
        (json!({ "type": "integer" }), json!("42"), json!(42)),
        (json!({ "type": "integer" }), json!("42.0"), json!(42)),
        (json!({ "type": "integer" }), json!("1e3"), json!(1000)),
        (json!({ "type": "integer" }), json!(7.0), json!(7)),
        (json!({ "type": "boolean" }), json!("true"), json!(true)),
        (json!({ "type": "boolean" }), json!("false"), json!(false)),
        (json!({ "type": "boolean" }), json!(1), json!(true)),
        (json!({ "type": "boolean" }), json!(1.0), json!(true)),
        (json!({ "type": "boolean" }), json!(0), json!(false)),
        (json!({ "type": "boolean" }), json!(0.0), json!(false)),
        (json!({ "type": "string" }), Value::Null, json!("")),
        (json!({ "type": "string" }), json!(true), json!("true")),
        (json!({ "type": "string" }), json!(7.0), json!("7")),
        (json!({ "type": "null" }), json!(""), Value::Null),
        (json!({ "type": "null" }), json!(0), Value::Null),
        (json!({ "type": "null" }), json!(0.0), Value::Null),
        (json!({ "type": "null" }), json!(false), Value::Null),
        (
            json!({ "type": ["number", "string"] }),
            json!("1"),
            json!("1"),
        ),
        (
            json!({ "type": ["boolean", "number"] }),
            json!("1"),
            json!(1.0),
        ),
    ];

    for (schema, input, expected) in passing_cases {
        assert_eq!(
            validate_single_value(schema, input).expect("valid"),
            expected
        );
    }

    for (schema, input) in [
        (json!({ "type": "boolean" }), json!("1")),
        (json!({ "type": "boolean" }), json!("0")),
        (json!({ "type": "null" }), json!("null")),
        (json!({ "type": "integer" }), json!("42.1")),
    ] {
        assert!(validate_single_value(schema, input).is_err());
    }
}

#[test]
fn validation_enforces_json_schema_object_array_and_constraint_keywords() {
    let tool = Tool {
        name: "shape".to_owned(),
        description: "Schema coverage".to_owned(),
        parameters: json!({
            "type": "object",
            "required": ["name", "pair", "mode", "marker"],
            "additionalProperties": false,
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 2,
                    "maxLength": 6,
                    "pattern": "^[a-z]+$"
                },
                "pair": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": [
                        { "type": "string" },
                        { "type": "integer" }
                    ],
                    "additionalItems": false
                },
                "mode": { "enum": ["fast", "slow"] },
                "marker": { "const": true }
            }
        }),
    };
    let valid = ToolCall {
        id: "tool-1".to_owned(),
        name: "shape".to_owned(),
        arguments: object(json!({
            "name": "valid",
            "pair": ["left", "7"],
            "mode": "fast",
            "marker": true
        })),
        thought_signature: None,
    };

    let args = validate_tool_arguments(&tool, &valid).expect("valid");
    assert_eq!(args["pair"][1], 7);

    for (arguments, expected_error) in [
        (
            json!({
                "name": "valid",
                "pair": ["left", "7"],
                "mode": "fast",
                "marker": true,
                "extra": true
            }),
            "extra: unexpected property",
        ),
        (
            json!({
                "name": "valid",
                "pair": ["left", "7", "extra"],
                "mode": "fast",
                "marker": true
            }),
            "pair.2: unexpected item",
        ),
        (
            json!({
                "name": "Bad7",
                "pair": ["left", "7"],
                "mode": "medium",
                "marker": false
            }),
            "mode: expected one of",
        ),
    ] {
        let call = ToolCall {
            id: "tool-1".to_owned(),
            name: "shape".to_owned(),
            arguments: object(arguments),
            thought_signature: None,
        };
        let error = validate_tool_arguments(&tool, &call).expect_err("invalid");
        assert!(
            error.to_string().contains(expected_error),
            "{error:?} should contain {expected_error}"
        );
    }
}

#[test]
fn validation_enforces_combinators_and_additional_property_schemas() {
    assert_eq!(
        validate_single_value(
            json!({ "allOf": [{ "type": "integer" }, { "minimum": 2 }, { "maximum": 4 }] }),
            json!("3")
        )
        .expect("valid"),
        json!(3)
    );
    assert!(
        validate_single_value(
            json!({ "allOf": [{ "type": "integer" }, { "maximum": 4 }] }),
            json!("5")
        )
        .expect_err("invalid")
        .to_string()
        .contains("must be <= 4")
    );
    assert!(
        validate_single_value(
            json!({ "oneOf": [{ "type": "number" }, { "minimum": 0 }] }),
            json!(3)
        )
        .expect_err("ambiguous oneOf")
        .to_string()
        .contains("instead of exactly one")
    );

    let tool = Tool {
        name: "extras".to_owned(),
        description: "Additional properties schema".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "label": { "type": "string" }
            },
            "additionalProperties": { "type": "integer" }
        }),
    };
    let call = ToolCall {
        id: "tool-1".to_owned(),
        name: "extras".to_owned(),
        arguments: object(json!({ "label": "ok", "count": "9" })),
        thought_signature: None,
    };
    let args = validate_tool_arguments(&tool, &call).expect("valid");
    assert_eq!(args["count"], 9);
}

#[test]
fn env_api_keys_ignore_generic_github_tokens_for_copilot() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&["COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"]);

    set_env("GH_TOKEN", "gh-token");
    set_env("GITHUB_TOKEN", "github-token");

    assert_eq!(find_env_keys("github-copilot"), None);
    assert_eq!(get_env_api_key("github-copilot"), None);

    set_env("COPILOT_GITHUB_TOKEN", "copilot-token");
    assert_eq!(
        find_env_keys("github-copilot"),
        Some(vec!["COPILOT_GITHUB_TOKEN".to_owned()])
    );
    assert_eq!(
        get_env_api_key("github-copilot"),
        Some("copilot-token".to_owned())
    );
}

#[test]
fn node_http_proxy_respects_no_proxy_and_rejects_unsupported_protocols() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let target = "https://bedrock-runtime.us-east-1.amazonaws.com";
    set_env("HTTPS_PROXY", "http://proxy.example:8080");
    set_env("NO_PROXY", "bedrock-runtime.us-east-1.amazonaws.com");
    assert_eq!(resolve_http_proxy_url_for_target(target), Ok(None));

    remove_env("NO_PROXY");
    let proxy = resolve_http_proxy_url_for_target(target)
        .expect("proxy resolution")
        .expect("proxy url");
    assert_eq!(proxy.to_string(), "http://proxy.example:8080/");

    set_env("HTTPS_PROXY", "socks5://proxy.example:1080");
    let error = resolve_http_proxy_url_for_target(target).expect_err("unsupported proxy protocol");
    assert!(error.contains(UNSUPPORTED_PROXY_PROTOCOL_MESSAGE));
}

#[test]
fn node_http_proxy_respects_npm_config_proxy_envs() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let target = "https://api.example.com/v1/messages";
    set_env("npm_config_https_proxy", "npm-proxy.example:8080");
    let proxy = resolve_http_proxy_url_for_target(target)
        .expect("npm proxy resolution")
        .expect("npm proxy url");
    assert_eq!(proxy.to_string(), "https://npm-proxy.example:8080/");

    set_env("npm_config_no_proxy", ".example.com");
    assert_eq!(resolve_http_proxy_url_for_target(target), Ok(None));

    remove_env("npm_config_https_proxy");
    remove_env("npm_config_no_proxy");
    set_env("npm_config_proxy", "http://generic-npm-proxy.example:3128");
    let proxy = resolve_http_proxy_url_for_target(target)
        .expect("generic npm proxy resolution")
        .expect("generic npm proxy url");
    assert_eq!(proxy.to_string(), "http://generic-npm-proxy.example:3128/");
}

#[test]
fn node_http_proxy_resolves_websocket_targets_from_http_proxy_envs() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    set_env("HTTP_PROXY", "http://proxy.example:8080");
    let proxy = resolve_http_proxy_url_for_websocket_target("ws://codex.example/codex/responses")
        .expect("websocket proxy resolution")
        .expect("proxy url");
    assert_eq!(proxy.to_string(), "http://proxy.example:8080/");

    remove_env("HTTP_PROXY");
    set_env("HTTPS_PROXY", "http://secure-proxy.example:8443");
    let proxy = resolve_http_proxy_url_for_websocket_target(
        "wss://chatgpt.com/backend-api/codex/responses",
    )
    .expect("secure websocket proxy resolution")
    .expect("proxy url");
    assert_eq!(proxy.to_string(), "http://secure-proxy.example:8443/");

    set_env("NO_PROXY", "chatgpt.com");
    assert_eq!(
        resolve_http_proxy_url_for_websocket_target(
            "wss://chatgpt.com/backend-api/codex/responses"
        ),
        Ok(None)
    );
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_http_provider_routes_reqwest_requests_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_proxy\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_proxy\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Proxy\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_proxy\",\"content\":[{\"type\":\"output_text\",\"text\":\"Proxy\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_proxy\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
    );
    let (proxy_url, proxy_request_task) = mock_sse_server(sse).await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let mut model = Model::faux("openai-responses", "openai", "mock-gpt-5");
    model.base_url = "http://provider.example/v1".to_owned();
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = tokio::time::timeout(
        Duration::from_secs(1),
        complete_simple(&model, user_context("hello"), options),
    )
    .await
    .expect("proxied provider request should complete")
    .expect("complete");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(text_of(&message), Some("Proxy"));
    assert_eq!(message.response_id.as_deref(), Some("resp_proxy"));
    assert!(
        proxy_request.starts_with("POST http://provider.example/v1/responses HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_http_provider_routes_reqwest_requests_through_npm_config_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_npm_proxy\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_npm_proxy\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"NPM Proxy\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_npm_proxy\",\"content\":[{\"type\":\"output_text\",\"text\":\"NPM Proxy\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_npm_proxy\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":2,\"total_tokens\":4}}}\n\n",
    );
    let (proxy_url, proxy_request_task) = mock_sse_server(sse).await;
    set_env("npm_config_http_proxy", &proxy_url);
    set_env("npm_config_no_proxy", "127.0.0.1,localhost");

    let mut model = Model::faux("openai-responses", "openai", "mock-gpt-5");
    model.base_url = "http://provider.example/v1".to_owned();
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = tokio::time::timeout(
        Duration::from_secs(1),
        complete_simple(&model, user_context("hello"), options),
    )
    .await
    .expect("npm proxied provider request should complete")
    .expect("complete");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(text_of(&message), Some("NPM Proxy"));
    assert_eq!(message.response_id.as_deref(), Some("resp_npm_proxy"));
    assert!(
        proxy_request.starts_with("POST http://provider.example/v1/responses HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_openrouter_images_provider_routes_reqwest_requests_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let body = concat!(
        "{\"id\":\"img-proxy\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1},",
        "\"choices\":[{\"message\":{\"content\":\"Done\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,ZmFrZQ==\"}]}}]}"
    );
    let (proxy_url, proxy_request_task) = mock_json_server(body).await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = "http://images.example/v1".to_owned();
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate an icon")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-proxy"));
    assert!(
        proxy_request.starts_with("POST http://images.example/v1/chat/completions HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: bearer openrouter-key")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_openrouter_images_provider_routes_reqwest_requests_through_npm_config_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let body = concat!(
        "{\"id\":\"img-npm-proxy\",\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1},",
        "\"choices\":[{\"message\":{\"content\":\"NPM Done\",",
        "\"images\":[{\"image_url\":\"data:image/png;base64,bnBt\"}]}}]}"
    );
    let (proxy_url, proxy_request_task) = mock_json_server(body).await;
    set_env("npm_config_proxy", &proxy_url);
    set_env("npm_config_no_proxy", "127.0.0.1,localhost");

    let mut model =
        get_image_model("openrouter", "google/gemini-3.1-flash-image-preview").expect("model");
    model.base_url = "http://images.example/v1".to_owned();
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text("Generate an npm proxy icon")],
        },
        ImagesOptions {
            api_key: Some("openrouter-key".to_owned()),
            ..Default::default()
        },
    )
    .await
    .expect("generate images");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(output.stop_reason, ImagesStopReason::Stop);
    assert_eq!(output.response_id.as_deref(), Some("img-npm-proxy"));
    assert!(matches!(
        output.output.get(1),
        Some(ImagesContent::Image(image)) if image.data == "bnBt"
    ));
    assert!(
        proxy_request.starts_with("POST http://images.example/v1/chat/completions HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: bearer openrouter-key")
    );
}

#[test]
fn overflow_detects_error_and_silent_overflow_modes() {
    let mut message = error_assistant("prompt is too long: 10 tokens > 5 maximum");
    assert!(is_context_overflow(&message, None));

    message.stop_reason = StopReason::Stop;
    message.error_message = None;
    message.usage.input = 101;
    assert!(is_context_overflow(&message, Some(100)));

    message.stop_reason = StopReason::Length;
    message.usage.output = 0;
    message.usage.input = 99;
    assert!(is_context_overflow(&message, Some(100)));
}

#[test]
fn overflow_matches_provider_error_shapes_without_rate_limit_false_positives() {
    for error in [
        "400 `prompt too long; exceeded max context length by 100918 tokens`",
        "400 The input (516368 tokens) is longer than the model's context length (262144 tokens).",
        "Error: 503 litellm.ServiceUnavailableError: OpenAIException - Requested token count exceeds the model's maximum context length of 131072 tokens.",
    ] {
        assert!(
            is_context_overflow(&error_assistant(error), Some(262_144)),
            "{error}"
        );
    }

    for error in [
        "500 `model runner crashed unexpectedly`",
        "Throttling error: Too many tokens, please wait before trying again.",
        "Service unavailable: The service is temporarily unavailable.",
        "Rate limit exceeded, please retry after 30 seconds.",
        "Too many requests. Please slow down.",
        "429 status code (no body)",
    ] {
        assert!(
            !is_context_overflow(&error_assistant(error), Some(200_000)),
            "{error}"
        );
    }

    let mut xiaomi_length = empty_assistant(StopReason::Length);
    xiaomi_length.usage.input = 58;
    xiaomi_length.usage.cache_read = 1_048_512;
    xiaomi_length.usage.output = 0;
    assert!(is_context_overflow(&xiaomi_length, Some(1_048_576)));

    let mut normal_length = empty_assistant(StopReason::Length);
    normal_length.usage.input = 1_000;
    normal_length.usage.output = 4_096;
    assert!(!is_context_overflow(&normal_length, Some(200_000)));

    let mut small_length = empty_assistant(StopReason::Length);
    small_length.usage.input = 100;
    assert!(!is_context_overflow(&small_length, Some(200_000)));
}

#[test]
fn overflow_matches_context_overflow_provider_error_corpus() {
    assert_eq!(overflow_pattern_count(), 22);

    for (provider, error) in [
        (
            "anthropic token limit",
            "prompt is too long: 213462 tokens > 200000 maximum",
        ),
        (
            "anthropic request size",
            r#"413 {"error":{"type":"request_too_large","message":"Request exceeds the maximum size"}}"#,
        ),
        ("amazon bedrock", "Input is too long for requested model"),
        (
            "openai responses",
            "Your input exceeds the context window of this model",
        ),
        (
            "openai compatible proxy",
            "Requested token count exceeds the model's maximum context length of 131,072 tokens.",
        ),
        (
            "google gemini",
            "The input token count (1196265) exceeds the maximum number of tokens allowed (1048575)",
        ),
        (
            "xai",
            "This model's maximum prompt length is 131072 but the request contains 537812 tokens",
        ),
        (
            "groq",
            "Please reduce the length of the messages or completion",
        ),
        (
            "openrouter",
            "This endpoint's maximum context length is 200000 tokens. However, you requested about 210000 tokens",
        ),
        (
            "together",
            "The input (516368 tokens) is longer than the model's context length (262144 tokens).",
        ),
        (
            "github copilot",
            "prompt token count of 140000 exceeds the limit of 128000",
        ),
        (
            "llama.cpp",
            "the request exceeds the available context size, try increasing it",
        ),
        (
            "lm studio",
            "tokens to keep from the initial prompt is greater than the context length",
        ),
        ("minimax", "invalid params, context window exceeds limit"),
        (
            "kimi coding",
            "Your request exceeded model token limit: 131072 (requested: 140000)",
        ),
        (
            "mistral",
            "Prompt contains 140000 tokens and is too large for model with 128000 maximum context length",
        ),
        ("z.ai", "model_context_window_exceeded"),
        (
            "ollama",
            "400 `prompt too long; exceeded context length by 100918 tokens`",
        ),
        ("generic underscore", "context_length_exceeded"),
        ("generic spaced", "context length exceeded"),
        ("generic tokens", "The request has too many tokens"),
        ("generic token limit", "token limit exceeded"),
        ("cerebras 400", "400 status code (no body)"),
        ("cerebras 413", "413 (no body)"),
    ] {
        assert!(
            is_context_overflow(&error_assistant(error), Some(200_000)),
            "{provider}: {error}"
        );
    }
}

#[test]
fn json_repair_and_hash_match_core_semantics() {
    assert_eq!(repair_json("{\"x\":\"a\nb\"}"), "{\"x\":\"a\\nb\"}");
    let value: Value = parse_json_with_repair("{\"x\":\"a\nb\"}").expect("json");
    assert_eq!(value["x"], "a\nb");
    assert_eq!(short_hash(""), "k4n83c7h0j2b");
    assert_eq!(short_hash("hello"), "1h6qa0qrowduu");
    assert_eq!(short_hash("world"), "yoqfis1dkxj7l");
    assert_eq!(short_hash("emoji 🙈"), "8f4fmk10p65ud");
}

#[test]
fn parse_streaming_json_recovers_common_partial_tool_arguments() {
    assert_eq!(
        parse_streaming_json(Some(r#"{"path":"src/main.rs""#)),
        json!({ "path": "src/main.rs" })
    );
    assert_eq!(
        parse_streaming_json(Some(r#"{"count":2,"nested":{"ok":true}"#)),
        json!({ "count": 2, "nested": { "ok": true } })
    );
    assert_eq!(
        parse_streaming_json(Some(r#"{"items":[1,2,]"#)),
        json!({ "items": [1, 2] })
    );
    assert_eq!(
        parse_streaming_json(Some(r#"{"path":"src/main.rs","content":"#)),
        json!({ "path": "src/main.rs" })
    );
    assert_eq!(
        parse_streaming_json(Some(r#"{"outer":{"complete":1,"pending":tru"#)),
        json!({ "outer": { "complete": 1 } })
    );
    assert_eq!(parse_streaming_json(Some(r#"[1,2,tr"#)), json!([1, 2]));
    assert_eq!(
        parse_streaming_json(Some(r#"{"unfinished":"value"#)),
        json!({ "unfinished": "value" })
    );
    assert_eq!(parse_streaming_json(Some(r#"{"waiting":"#)), json!({}));
}

#[test]
fn unicode_surrogate_repair_preserves_pairs_and_replaces_unpaired_escapes() {
    let paired: Value =
        parse_json_with_repair(r#"{"text":"emoji \uD83D\uDE48 stays"}"#).expect("paired");
    assert_eq!(paired["text"], "emoji 🙈 stays");

    let high: Value =
        parse_json_with_repair(r#"{"text":"bad \uD83D high"}"#).expect("high surrogate");
    assert_eq!(high["text"], "bad � high");

    let low: Value = parse_json_with_repair(r#"{"text":"bad \uDE48 low"}"#).expect("low surrogate");
    assert_eq!(low["text"], "bad � low");
}

#[tokio::test]
async fn builtin_openai_completions_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"id\":\"chatcmpl_1\",\"model\":\"mock-response\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("openai-completions", "openai", "mock-model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.session_id = Some("session-1".to_owned());
    options.stream.cache_retention = Some(CacheRetention::Long);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hello"));
    assert_eq!(message.response_id.as_deref(), Some("chatcmpl_1"));
    assert_eq!(message.response_model.as_deref(), Some("mock-response"));
    assert_eq!(message.usage.input, 4);
    assert_eq!(message.usage.output, 2);
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key")
    );
    assert!(request.contains("\"stream\":true"));
    assert!(request.contains("\"prompt_cache_key\":\"session-1\""));
}

#[tokio::test]
async fn builtin_openai_completions_provider_applies_response_hooks() {
    let sse = concat!(
        "data: {\"id\":\"chatcmpl_hook\",\"choices\":[{\"delta\":{\"content\":\"Hooked\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, request_task) = mock_http_sequence_server(vec![MockHttpSequenceResponse {
        status: 200,
        reason: "OK",
        content_type: "text/event-stream",
        headers: vec![("x-ri-response-hook", "seen")],
        body: sse,
    }])
    .await;
    let mut model = Model::faux("openai-completions", "openai", "mock-model");
    model.base_url = base_url;
    let seen_responses = Arc::new(Mutex::new(Vec::<ProviderResponse>::new()));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options
        .response_hooks
        .push(Arc::new(RecordingProviderResponseHook {
            seen: seen_responses.clone(),
        }));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let requests = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hooked"));
    assert_eq!(requests.len(), 1);
    let seen_responses = seen_responses.lock().expect("seen responses");
    assert_eq!(seen_responses.len(), 1);
    assert_eq!(seen_responses[0].status, 200);
    assert_eq!(
        seen_responses[0]
            .headers
            .get("x-ri-response-hook")
            .map(String::as_str),
        Some("seen")
    );
}

#[tokio::test]
async fn builtin_openai_responses_simple_provider_applies_default_max_output_tokens() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_default_max\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_default_max\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Default max\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_default_max\",\"content\":[{\"type\":\"output_text\",\"text\":\"Default max\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_default_max\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":2,\"total_tokens\":4}}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("openai-responses", "openai", "full-window-output");
    model.base_url = base_url;
    model.context_window = 128_000;
    model.max_tokens = 128_000;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Default max"));
    assert!(request.contains("\"max_output_tokens\":32000"));
}

#[tokio::test]
async fn builtin_cloudflare_ai_gateway_completions_uses_cf_aig_auth_without_upstream_authorization()
{
    let sse = concat!(
        "data: {\"id\":\"chatcmpl_cf\",\"choices\":[{\"delta\":{\"content\":\"Cloudflare\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":1,\"total_tokens\":4}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = get_model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    )
    .expect("cloudflare gateway workers model");
    assert_eq!(model.api, "openai-completions");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("cf-token".to_owned());
    options.stream.session_id = Some("session-cf".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Cloudflare"));
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(lower_request.contains("cf-aig-authorization: bearer cf-token"));
    assert!(!lower_request.contains("\r\nauthorization: bearer cf-token"));
    assert!(lower_request.contains("x-session-affinity: session-cf"));
}

#[tokio::test]
async fn builtin_openai_completions_provider_emits_sse_events_incrementally() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"id\":\"chatcmpl_stream\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ],
        Duration::from_millis(150),
    )
    .await;
    let mut model = Model::faux("openai-completions", "openai", "mock-model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let first_delta = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let event = stream.next().await.expect("stream event");
            if let AssistantMessageEvent::TextDelta { delta, .. } = &event {
                assert_eq!(delta, "Hel");
                return event;
            }
        }
    })
    .await
    .expect("first text delta before full SSE response completes");

    assert!(
        !request_task.is_finished(),
        "first text delta should arrive while the mock server is still writing SSE chunks"
    );

    let mut events = vec![first_delta];
    events.extend(collect_events(stream).await);
    let message = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message),
            _ => None,
        })
        .expect("done event");
    assert_eq!(text_of(message), Some("Hello"));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_openai_completions_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"id\":\"chatcmpl_abort\",\"choices\":[{\"delta\":{\"content\":\"Hel\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux("openai-completions", "openai", "mock-model");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.abort_flag = Some(abort_flag.clone());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Hel")
            && error.usage.input == 0
            && error.usage.output == 0
            && error.usage.total_tokens == 0
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_openai_responses_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_1\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4}}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("openai-responses", "openai", "mock-gpt-5");
    model.base_url = base_url;
    model.reasoning = true;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.session_id = Some("session-2".to_owned());
    options.reasoning = Some(ThinkingLevel::Low);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hi"));
    assert_eq!(message.response_id.as_deref(), Some("resp_1"));
    assert_eq!(message.usage.input, 3);
    assert_eq!(message.usage.output, 1);
    assert!(request.starts_with("POST /responses HTTP/1.1"));
    assert!(request.contains("\"stream\":true"));
    assert!(request.contains("\"prompt_cache_key\":\"session-2\""));
    assert!(request.contains("\"reasoning\":{\"effort\":\"low\""));
}

#[tokio::test]
async fn builtin_cloudflare_ai_gateway_openai_responses_preserves_byok_authorization_and_adds_cf_aig_auth()
 {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_cf\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_cf\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Gateway\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_cf\",\"content\":[{\"type\":\"output_text\",\"text\":\"Gateway\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_cf\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model =
        get_model("cloudflare-ai-gateway", "gpt-5.1").expect("cloudflare gateway openai model");
    assert_eq!(model.api, "openai-responses");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("cf-token".to_owned());
    options.stream.headers.insert(
        "Authorization".to_owned(),
        "Bearer upstream-token".to_owned(),
    );

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Gateway"));
    assert!(request.starts_with("POST /responses HTTP/1.1"));
    assert!(lower_request.contains("authorization: bearer upstream-token"));
    assert!(lower_request.contains("cf-aig-authorization: bearer cf-token"));
    assert!(!lower_request.contains("\r\nauthorization: bearer cf-token"));
}

#[tokio::test]
async fn builtin_openai_responses_provider_emits_sse_events_incrementally() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_stream\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_stream\"}}\n\n\
             data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_stream\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1,\"total_tokens\":4}}}\n\n",
        ],
        Duration::from_millis(150),
    )
    .await;
    let mut model = Model::faux("openai-responses", "openai", "mock-gpt-5");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let first_delta = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let event = stream.next().await.expect("stream event");
            if let AssistantMessageEvent::TextDelta { delta, .. } = &event {
                assert_eq!(delta, "Hi");
                return event;
            }
        }
    })
    .await
    .expect("first text delta before full SSE response completes");

    assert!(
        !request_task.is_finished(),
        "first text delta should arrive while the mock server is still writing SSE chunks"
    );

    let mut events = vec![first_delta];
    events.extend(collect_events(stream).await);
    let message = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message),
            _ => None,
        })
        .expect("done event");
    assert_eq!(message.response_id.as_deref(), Some("resp_stream"));
    assert_eq!(text_of(message), Some("Hi"));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_openai_responses_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_abort\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_abort\"}}\n\n\
             data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_abort\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hi\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_abort\",\"status\":\"completed\"}}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux("openai-responses", "openai", "mock-gpt-5");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.abort_flag = Some(abort_flag.clone());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hi")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Hi")
            && error.usage.input == 0
            && error.usage.output == 0
            && error.usage.total_tokens == 0
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_github_copilot_openai_provider_exposes_response_id() {
    let sse = concat!(
        "data: {\"id\":\"chatcmpl_copilot_openai\",\"model\":\"gpt-5.3-codex\",\"choices\":[{\"delta\":{\"content\":\"Copilot\"},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = get_model("github-copilot", "gpt-5.3-codex").expect("copilot openai model");
    assert_eq!(model.api, "openai-completions");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("copilot-openai-token".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Copilot"));
    assert_eq!(
        message.response_id.as_deref(),
        Some("chatcmpl_copilot_openai")
    );
    assert_eq!(message.usage.input, 2);
    assert_eq!(message.usage.output, 1);
    assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(lower_request.contains("authorization: bearer copilot-openai-token"));
    assert!(lower_request.contains("copilot-integration-id: vscode-chat"));
    assert!(request.contains("\"model\":\"gpt-5.3-codex\""));
}

#[tokio::test]
async fn builtin_azure_openai_responses_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_azure\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_azure\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Azure\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_azure\",\"content\":[{\"type\":\"output_text\",\"text\":\"Azure\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_azure\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux(
        "azure-openai-responses",
        "azure-openai-responses",
        "gpt-4o-mini",
    );
    model.base_url = format!("{base_url}/openai/v1");
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("azure-key".to_owned());
    options.stream.session_id = Some("session-azure".to_owned());
    options
        .stream
        .extra
        .insert("azureDeploymentName".to_owned(), json!("deploy-gpt4o"));
    options
        .stream
        .extra
        .insert("azureApiVersion".to_owned(), json!("2024-10-21"));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Azure"));
    assert_eq!(message.response_id.as_deref(), Some("resp_azure"));
    assert_eq!(message.usage.input, 2);
    assert_eq!(message.usage.output, 1);
    assert!(request.starts_with("POST /openai/v1/responses?api-version=2024-10-21 HTTP/1.1"));
    assert!(request.to_ascii_lowercase().contains("api-key: azure-key"));
    assert!(
        !request
            .to_ascii_lowercase()
            .contains("authorization: bearer")
    );
    assert!(request.contains("\"model\":\"deploy-gpt4o\""));
    assert!(request.contains("\"prompt_cache_key\":\"session-azure\""));
}

#[tokio::test]
async fn builtin_azure_openai_responses_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_azure_abort\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_azure_abort\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Azur\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_azure_abort\",\"content\":[{\"type\":\"output_text\",\"text\":\"Azure\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_azure_abort\",\"status\":\"completed\",\"usage\":{\"input_tokens\":2,\"output_tokens\":1,\"total_tokens\":3}}}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux(
        "azure-openai-responses",
        "azure-openai-responses",
        "gpt-4o-mini",
    );
    model.base_url = format!("{base_url}/openai/v1");
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("azure-key".to_owned());
    options.stream.abort_flag = Some(abort_flag.clone());
    options
        .stream
        .extra
        .insert("azureDeploymentName".to_owned(), json!("deploy-gpt4o"));
    options
        .stream
        .extra
        .insert("azureApiVersion".to_owned(), json!("2024-10-21"));

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Azur")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Azur")
            && error.usage.input == 0
            && error.usage.output == 0
            && error.usage.total_tokens == 0
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_mistral_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"id\":\"mistral_1\",\"choices\":[{\"delta\":{\"content\":\"Bon\"},\"finishReason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"jour\"},\"finishReason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{},\"finishReason\":\"stop\"}],\"usage\":{\"promptTokens\":3,\"completionTokens\":1,\"totalTokens\":4}}\n\n",
        "data: [DONE]\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("mistral-conversations", "mistral", "mistral-small-latest");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.session_id = Some("session-mistral".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Bonjour"));
    assert_eq!(message.response_id.as_deref(), Some("mistral_1"));
    assert_eq!(message.usage.input, 3);
    assert_eq!(message.usage.output, 1);
    assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer test-key")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-affinity: session-mistral")
    );
    assert!(request.contains("\"stream\":true"));
}

#[tokio::test]
async fn builtin_mistral_provider_emits_sse_events_incrementally() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"id\":\"mistral_stream\",\"choices\":[{\"delta\":{\"content\":\"Bon\"},\"finishReason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"jour\"},\"finishReason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finishReason\":\"stop\"}]}\n\n",
        ],
        Duration::from_millis(150),
    )
    .await;
    let mut model = Model::faux("mistral-conversations", "mistral", "mistral-small-latest");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let first_delta = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let event = stream.next().await.expect("stream event");
            if let AssistantMessageEvent::TextDelta { delta, .. } = &event {
                assert_eq!(delta, "Bon");
                return event;
            }
        }
    })
    .await
    .expect("first text delta before full SSE response completes");

    assert!(
        !request_task.is_finished(),
        "first text delta should arrive while the mock server is still writing SSE chunks"
    );

    let mut events = vec![first_delta];
    events.extend(collect_events(stream).await);
    let message = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message),
            _ => None,
        })
        .expect("done event");
    assert_eq!(text_of(message), Some("Bonjour"));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_mistral_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"id\":\"mistral_abort\",\"choices\":[{\"delta\":{\"content\":\"Bon\"},\"finishReason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"jour\"},\"finishReason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finishReason\":\"stop\"}]}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux("mistral-conversations", "mistral", "mistral-small-latest");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.abort_flag = Some(abort_flag.clone());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Bon")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Bon")
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_anthropic_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_http\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("anthropic-messages", "anthropic", "claude-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("anthropic-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hi"));
    assert_eq!(message.response_id.as_deref(), Some("msg_http"));
    assert_eq!(message.usage.input, 3);
    assert_eq!(message.usage.output, 1);
    assert!(request.starts_with("POST /messages HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-api-key: anthropic-key")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("anthropic-version: 2023-06-01")
    );
    assert!(
        !request
            .to_ascii_lowercase()
            .contains("authorization: bearer")
    );
    assert!(request.contains("\"stream\":true"));
    assert!(request.contains("\"model\":\"claude-test\""));
}

#[tokio::test]
async fn builtin_anthropic_oauth_provider_normalizes_claude_code_tool_names_end_to_end() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_oauth_tool\",\"usage\":{\"input_tokens\":4,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_todo\",\"name\":\"TodoWrite\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"todos\\\":[]}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"input_tokens\":4,\"output_tokens\":2}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("anthropic-messages", "anthropic", "claude-test");
    model.base_url = base_url;
    let context = Context {
        messages: vec![Message::User(UserMessage::text("update todos"))],
        tools: vec![Tool {
            name: "todowrite".to_owned(),
            description: "Write todos".to_owned(),
            parameters: json!({ "type": "object" }),
        }],
        ..Default::default()
    };
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("sk-ant-oat-test-token".to_owned());

    let message = complete_simple(&model, context, options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(message.response_id.as_deref(), Some("msg_oauth_tool"));
    assert_eq!(message.stop_reason, StopReason::ToolUse);
    let AssistantContent::ToolCall(tool_call) = &message.content[0] else {
        panic!("tool call");
    };
    assert_eq!(tool_call.name, "todowrite");
    assert_eq!(tool_call.arguments, object(json!({ "todos": [] })));
    assert!(request.contains("\"name\":\"TodoWrite\""));
    assert!(!request.contains("\"name\":\"todowrite\""));
    assert!(lower_request.contains("authorization: bearer sk-ant-oat-test-token"));
    assert!(lower_request.contains("anthropic-beta: claude-code-20250219"));
    assert!(!lower_request.contains("x-api-key:"));
}

#[tokio::test]
async fn builtin_cloudflare_ai_gateway_anthropic_preserves_byok_authorization_and_adds_cf_aig_auth()
{
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_cf_anthropic\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Gateway Claude\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = get_model("cloudflare-ai-gateway", "claude-sonnet-4-5")
        .expect("cloudflare gateway anthropic model");
    assert_eq!(model.api, "anthropic-messages");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("cf-token".to_owned());
    options.stream.headers.insert(
        "Authorization".to_owned(),
        "Bearer upstream-token".to_owned(),
    );

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Gateway Claude"));
    assert_eq!(message.response_id.as_deref(), Some("msg_cf_anthropic"));
    assert!(request.starts_with("POST /messages HTTP/1.1"));
    assert!(lower_request.contains("cf-aig-authorization: bearer cf-token"));
    assert!(lower_request.contains("authorization: bearer upstream-token"));
    assert!(!lower_request.contains("\r\nx-api-key:"));
    assert!(!lower_request.contains("\r\nauthorization: bearer cf-token"));
}

#[tokio::test]
async fn builtin_github_copilot_anthropic_provider_exposes_response_id() {
    let sse = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_copilot_anthropic\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Claude\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = get_model("github-copilot", "claude-sonnet-4.6").expect("copilot claude model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("copilot-anthropic-token".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower_request = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Claude"));
    assert_eq!(
        message.response_id.as_deref(),
        Some("msg_copilot_anthropic")
    );
    assert_eq!(message.usage.input, 3);
    assert_eq!(message.usage.output, 1);
    assert!(request.starts_with("POST /messages HTTP/1.1"));
    assert!(lower_request.contains("authorization: bearer copilot-anthropic-token"));
    assert!(lower_request.contains("copilot-integration-id: vscode-chat"));
    assert!(!lower_request.contains("x-api-key:"));
    assert!(request.contains("\"model\":\"claude-sonnet-4.6\""));
}

#[tokio::test]
async fn builtin_anthropic_provider_emits_sse_events_incrementally() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "event: message_start\n\
             data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_stream\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\n\
             data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n\
             data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_stop\n\
             data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n\
             data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}\n\n",
            "event: message_stop\n\
             data: {\"type\":\"message_stop\"}\n\n",
        ],
        Duration::from_millis(150),
    )
    .await;
    let mut model = Model::faux("anthropic-messages", "anthropic", "claude-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("anthropic-key".to_owned());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let first_delta = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let event = stream.next().await.expect("stream event");
            if let AssistantMessageEvent::TextDelta { delta, .. } = &event {
                assert_eq!(delta, "Hi");
                return event;
            }
        }
    })
    .await
    .expect("first text delta before full SSE response completes");

    assert!(
        !request_task.is_finished(),
        "first text delta should arrive while the mock server is still writing SSE chunks"
    );

    let mut events = vec![first_delta];
    events.extend(collect_events(stream).await);
    let message = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message),
            _ => None,
        })
        .expect("done event");
    assert_eq!(message.response_id.as_deref(), Some("msg_stream"));
    assert_eq!(text_of(message), Some("Hi"));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_anthropic_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "event: message_start\n\
             data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_abort\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\n\
             data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\n\
             data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
            "event: content_block_stop\n\
             data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\n\
             data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux("anthropic-messages", "anthropic", "claude-test");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("anthropic-key".to_owned());
    options.stream.abort_flag = Some(abort_flag.clone());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hi")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Hi")
            && error.usage.input == 3
            && error.usage.output == 1
            && error.usage.total_tokens == 4
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_google_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"responseId\":\"google_http\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":2,\"totalTokenCount\":6}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("google-generative-ai", "google", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("gemini-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hello"));
    assert_eq!(message.response_id.as_deref(), Some("google_http"));
    assert_eq!(message.usage.input, 4);
    assert_eq!(message.usage.output, 2);
    assert!(
        request
            .starts_with("POST /v1beta/models/gemini-test:streamGenerateContent?alt=sse HTTP/1.1")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-goog-api-key: gemini-key")
    );
    assert!(
        !request
            .to_ascii_lowercase()
            .contains("authorization: bearer")
    );
    assert!(request.contains("\"model\":\"gemini-test\""));
    assert!(request.contains("\"contents\""));
}

#[tokio::test]
async fn builtin_google_provider_emits_sse_events_incrementally() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"responseId\":\"google_stream\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":2,\"totalTokenCount\":6}}\n\n",
        ],
        Duration::from_millis(150),
    )
    .await;
    let mut model = Model::faux("google-generative-ai", "google", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("gemini-key".to_owned());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let first_delta = tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let event = stream.next().await.expect("stream event");
            if let AssistantMessageEvent::TextDelta { delta, .. } = &event {
                assert_eq!(delta, "Hel");
                return event;
            }
        }
    })
    .await
    .expect("first text delta before full SSE response completes");

    assert!(
        !request_task.is_finished(),
        "first text delta should arrive while the mock server is still writing SSE chunks"
    );

    let mut events = vec![first_delta];
    events.extend(collect_events(stream).await);
    let message = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AssistantMessageEvent::Done { message, .. } => Some(message),
            _ => None,
        })
        .expect("done event");
    assert_eq!(message.response_id.as_deref(), Some("google_stream"));
    assert_eq!(text_of(message), Some("Hello"));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_google_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"responseId\":\"google_abort\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}],\"usageMetadata\":{\"promptTokenCount\":4,\"candidatesTokenCount\":1,\"totalTokenCount\":5}}\n\n",
            "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux("google-generative-ai", "google", "gemini-test");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("gemini-key".to_owned());
    options.stream.abort_flag = Some(abort_flag.clone());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Hel")
            && error.usage.input == 4
            && error.usage.output == 1
            && error.usage.total_tokens == 5
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_google_vertex_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"responseId\":\"vertex_http\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Ver\"}]}}]}\n\n",
        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"tex\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":2,\"totalTokenCount\":7}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("google-vertex", "google-vertex", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("vertex-key".to_owned());

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Vertex"));
    assert_eq!(message.response_id.as_deref(), Some("vertex_http"));
    assert_eq!(message.usage.input, 5);
    assert_eq!(message.usage.output, 2);
    assert!(
        request.starts_with("POST /v1/models/gemini-test:streamGenerateContent?alt=sse HTTP/1.1")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-goog-api-key: vertex-key")
    );
    assert!(request.contains("\"model\":\"gemini-test\""));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_google_vertex_provider_uses_adc_access_token_env() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]);
    set_env("GOOGLE_OAUTH_ACCESS_TOKEN", "adc-token");

    let sse = concat!(
        "data: {\"responseId\":\"vertex_adc\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"ADC\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":2,\"candidatesTokenCount\":1,\"totalTokenCount\":3}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("google-vertex", "google-vertex", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options
        .stream
        .extra
        .insert("project".to_owned(), json!("test-project"));
    options
        .stream
        .extra
        .insert("location".to_owned(), json!("us-central1"));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("ADC"));
    assert_eq!(message.response_id.as_deref(), Some("vertex_adc"));
    assert!(lower.contains("authorization: bearer adc-token"));
    assert!(!lower.contains("x-goog-api-key:"));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_google_vertex_provider_refreshes_authorized_user_adc_credentials() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]);

    let (token_url, token_request_task) =
        mock_json_server("{\"access_token\":\"refreshed-token\",\"expires_in\":3600}").await;
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let credentials_path = std::env::temp_dir().join(format!("ri-google-adc-{unique}.json"));
    std::fs::write(
        &credentials_path,
        json!({
            "type": "authorized_user",
            "client_id": "client id",
            "client_secret": "client secret",
            "refresh_token": "refresh token",
            "token_uri": token_url,
        })
        .to_string(),
    )
    .expect("write adc credentials");
    set_env(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials_path.to_str().expect("credentials path"),
    );

    let sse = concat!(
        "data: {\"responseId\":\"vertex_refreshed\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Refreshed\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
    );
    let (base_url, vertex_request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("google-vertex", "google-vertex", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options
        .stream
        .extra
        .insert("project".to_owned(), json!("test-project"));
    options
        .stream
        .extra
        .insert("location".to_owned(), json!("us-central1"));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let token_request = token_request_task.await.expect("token request task");
    let vertex_request = vertex_request_task.await.expect("vertex request task");
    let _ = std::fs::remove_file(credentials_path);
    let vertex_lower = vertex_request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Refreshed"));
    assert!(token_request.starts_with("POST / HTTP/1.1"));
    assert!(token_request.contains("grant_type=refresh_token"));
    assert!(token_request.contains("client_id=client%20id"));
    assert!(token_request.contains("client_secret=client%20secret"));
    assert!(token_request.contains("refresh_token=refresh%20token"));
    assert!(vertex_lower.contains("authorization: bearer refreshed-token"));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_google_vertex_adc_refresh_routes_token_request_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let mut env_keys = vec![
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ];
    env_keys.extend_from_slice(PROXY_ENV_KEYS);
    let _guard = EnvGuard::clearing(&env_keys);

    let (proxy_url, token_request_task) =
        mock_json_server("{\"access_token\":\"proxied-token\",\"expires_in\":3600}").await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let credentials_path = std::env::temp_dir().join(format!("ri-google-adc-proxy-{unique}.json"));
    std::fs::write(
        &credentials_path,
        json!({
            "type": "authorized_user",
            "client_id": "client id",
            "client_secret": "client secret",
            "refresh_token": "refresh token",
            "token_uri": "http://oauth.example/token",
        })
        .to_string(),
    )
    .expect("write adc credentials");
    set_env(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials_path.to_str().expect("credentials path"),
    );

    let sse = concat!(
        "data: {\"responseId\":\"vertex_proxy_adc\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Proxy ADC\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
    );
    let (base_url, vertex_request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("google-vertex", "google-vertex", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options
        .stream
        .extra
        .insert("project".to_owned(), json!("test-project"));
    options
        .stream
        .extra
        .insert("location".to_owned(), json!("us-central1"));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let token_request = token_request_task.await.expect("token request task");
    let vertex_request = vertex_request_task.await.expect("vertex request task");
    let _ = std::fs::remove_file(credentials_path);

    assert_eq!(text_of(&message), Some("Proxy ADC"));
    assert!(
        token_request.starts_with("POST http://oauth.example/token HTTP/1.1"),
        "{token_request}"
    );
    assert!(token_request.contains("grant_type=refresh_token"));
    assert!(
        vertex_request
            .starts_with("POST /v1/models/gemini-test:streamGenerateContent?alt=sse HTTP/1.1"),
        "{vertex_request}"
    );
    assert!(
        vertex_request
            .to_ascii_lowercase()
            .contains("authorization: bearer proxied-token")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_google_vertex_provider_refreshes_service_account_adc_credentials() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "GOOGLE_CLOUD_API_KEY",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ]);

    const TEST_PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----\n\
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCbDHU1DjkP+quE\n\
fcO+jc+LfIzlVzG5+43XHIvDfbZeVa7t/Y0FhWtMVp13fT3BoYEegC0l0SDFuoEn\n\
qvuwhM9BQIc6dwt3LjUEDRaHTY89EeER782pt20CiB9OTIq8mGsKHWkVxZWyPM0x\n\
lXd6rwY1ewCPgMhQOrPjpL17UgYzn/Z1iWRNMV5Y1iGgqnnyFDpxwO0u2Wc8/jO1\n\
wvvI4cGxlrJaDPj5YhbYNEZHd5KOrsKiLSpxJ8ARhIbH36tM5r3A+Ujm7i94FgRU\n\
baYtHX5T41lU2HQl5SK1K27uV1bBGiG6iQfDfv1OQA+hVtpx3UQJ3KDT5d5MFWI3\n\
45lS0RoHAgMBAAECggEACXxaIVxU5mLeInV1R8+yPmTo09EhVDENLPdsn5Gt2hig\n\
4qOMAKX4egukh55sbE++sAiEepdQS3iNFUmzK0n8yg+yFkQZOfnkOXK5iZ6XoFNb\n\
Mzc6HGOB8lE0pRwusroadlx1ROU5MtXgceOkkydpGFWFo8Hrv2jP/6HhC83pXjgX\n\
NiifGsYbrQRIMb9Rxag+6PCYtymxaGuYqavQifXjf/3hK62SJ/zDbycYYXsL+OBQ\n\
JxF6kLjd7gv0WjWcHiCgvkqDGzyzOjj6EqXsOtN+sbzEHTTH19KWBqbNJzQ/SxDw\n\
aCNE69JW2NVFDydub0YOhnIWQMoa/EAaDiOhNXa+AQKBgQDTwcA4CcAK/1H732/G\n\
6xip2ly/Cp/XtTnT6dBAU/CplLyA8WcgdPV6CFNiqvy1Bl948bAt+fsgf5pMLUxW\n\
x/q63KVQPbAFO+nNp/aByJxps44eKlPwOUuw21k5e4tqPJF+KjZSsqu0D19eNQN1\n\
9fDrEIwoa9bXxhu+q79q9dzYBwKBgQC7cY3G8Voest/qgGwd4QYQRrfweraaaMS4\n\
VldHrqIl8Yn1CVa3YFK6H6/iI0zlxhjCbym4Krhgr1MTCg6h42bdIXp6exfDMFDX\n\
+oUyU9t9EUBUjC4wb5a9p9bBJfYgucS40gMFUTtQ9dPl+aeaU7syeSGwUEjCPylo\n\
61PSWgUuAQKBgQCn9zeRO6qpDnzpXQI8tp6JnDuVDchcQdPs07nsTKjI2sHrRZCX\n\
ni5Y7eG2kgqBTNzOAmfNEEyyKoUph4TWESpArmQykbvdavi5uFFAAPCQp2xDYS/T\n\
jJ8NWfAcOHMNgZ2mhbUxQ6gO22K6RzLHjp3a1vVV2rQ/01SOmYzsOrlCYQKBgF/6\n\
G+NS75cqdhbn3PRLpUQuQb7jxp43qQrOQvCUTbhp/f624m0Q6CsfUHrVImnAziq4\n\
qr7/ONtgyoPEMYvZGXF+0+zlHFy4X5zHTO5hG9DlRXBFOt1YNfI0f3T00Bsfo8gS\n\
2LMfTeT9ipuGAri1yPNmLMbPxQGZP8XWQVxC9cYBAoGAHOK1aFwtN/CcBKcmRLJC\n\
0Ck/X2Y+bNFqKItMSzDLAyPMz8WFtSOdoSmDh3pozS/SfmIGcVmFJAeoUTGx4BnM\n\
M5nx2MdEMom48pYAHq1/tvFRJyjy8JbuNchswtG5Qx4ckncl1fywnewgndudtAvc\n\
0vpvqC4VV+Z21uFA1ZzRn7o=\n\
-----END PRIVATE KEY-----\n";

    let (token_url, token_request_task) =
        mock_json_server("{\"access_token\":\"service-token\",\"expires_in\":3600}").await;
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let credentials_path =
        std::env::temp_dir().join(format!("ri-google-service-account-adc-{unique}.json"));
    std::fs::write(
        &credentials_path,
        json!({
            "type": "service_account",
            "client_email": "svc@test-project.iam.gserviceaccount.com",
            "private_key": TEST_PRIVATE_KEY,
            "token_uri": token_url,
        })
        .to_string(),
    )
    .expect("write service account adc credentials");
    set_env(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials_path.to_str().expect("credentials path"),
    );

    let sse = concat!(
        "data: {\"responseId\":\"vertex_service\",\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Service\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2,\"totalTokenCount\":5}}\n\n",
    );
    let (base_url, vertex_request_task) = mock_sse_server(sse).await;
    let mut model = Model::faux("google-vertex", "google-vertex", "gemini-test");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options
        .stream
        .extra
        .insert("project".to_owned(), json!("test-project"));
    options
        .stream
        .extra
        .insert("location".to_owned(), json!("us-central1"));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let token_request = token_request_task.await.expect("token request task");
    let vertex_request = vertex_request_task.await.expect("vertex request task");
    let _ = std::fs::remove_file(credentials_path);
    let vertex_lower = vertex_request.to_ascii_lowercase();
    let token_body = token_request
        .split("\r\n\r\n")
        .nth(1)
        .expect("token request body");
    let assertion = token_body
        .split('&')
        .find_map(|field| field.strip_prefix("assertion="))
        .expect("jwt assertion");
    let jwt_parts = assertion.split('.').collect::<Vec<_>>();
    assert_eq!(jwt_parts.len(), 3);
    let claims_bytes = URL_SAFE_NO_PAD.decode(jwt_parts[1]).expect("decode claims");
    let claims: Value = serde_json::from_slice(&claims_bytes).expect("claims json");

    assert_eq!(text_of(&message), Some("Service"));
    assert!(token_request.starts_with("POST / HTTP/1.1"));
    assert!(
        token_body.contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer")
    );
    assert_eq!(
        claims["iss"],
        json!("svc@test-project.iam.gserviceaccount.com")
    );
    assert_eq!(
        claims["scope"],
        json!("https://www.googleapis.com/auth/cloud-platform")
    );
    assert_eq!(claims["aud"], json!(token_url));
    assert!(
        claims["exp"].as_i64().expect("exp") > claims["iat"].as_i64().expect("iat"),
        "service-account JWT should have a positive lifetime"
    );
    assert!(vertex_lower.contains("authorization: bearer service-token"));
}

#[tokio::test]
async fn builtin_openai_codex_provider_posts_json_and_parses_sse() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_codex\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_codex\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Code\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_codex\",\"content\":[{\"type\":\"output_text\",\"text\":\"Code\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_codex\",\"status\":\"completed\",\"usage\":{\"input_tokens\":6,\"output_tokens\":2,\"total_tokens\":8}}}\n\n",
    );
    let (base_url, request_task) = mock_sse_server(sse).await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let token = codex_test_token();
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(token.clone());
    options.stream.session_id = Some("codex-session".to_owned());
    options.stream.transport = Some(Transport::Sse);
    options
        .stream
        .extra
        .insert("textVerbosity".to_owned(), json!("medium"));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Code"));
    assert_eq!(message.response_id.as_deref(), Some("resp_codex"));
    assert_eq!(message.usage.input, 6);
    assert_eq!(message.usage.output, 2);
    assert!(request.starts_with("POST /codex/responses HTTP/1.1"));
    assert!(
        request
            .to_ascii_lowercase()
            .contains(&format!("authorization: bearer {}", token).to_ascii_lowercase())
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("chatgpt-account-id: acc_test")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("openai-beta: responses=experimental")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("session_id: codex-session")
    );
    assert!(request.contains("\"store\":false"));
    assert!(request.contains("\"stream\":true"));
    assert!(request.contains("\"prompt_cache_key\":\"codex-session\""));
    assert!(request.contains("\"verbosity\":\"medium\""));
}

#[tokio::test]
async fn builtin_openai_codex_provider_retries_sse_rate_limits_before_streaming_success() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_codex_retry\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_codex_retry\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Retry OK\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_codex_retry\",\"content\":[{\"type\":\"output_text\",\"text\":\"Retry OK\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_codex_retry\",\"status\":\"completed\",\"usage\":{\"input_tokens\":6,\"output_tokens\":2,\"total_tokens\":8}}}\n\n",
    );
    let (base_url, request_task) = mock_http_sequence_server(vec![
        MockHttpSequenceResponse {
            status: 429,
            reason: "Too Many Requests",
            content_type: "application/json",
            headers: vec![("retry-after-ms", "0")],
            body: "{\"error\":{\"code\":\"rate_limit_exceeded\",\"message\":\"rate limited\"}}",
        },
        MockHttpSequenceResponse {
            status: 200,
            reason: "OK",
            content_type: "text/event-stream",
            headers: Vec::new(),
            body: sse,
        },
    ])
    .await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("codex-retry-session".to_owned());
    options.stream.transport = Some(Transport::Sse);
    options.stream.max_retries = Some(1);
    options.stream.max_retry_delay_ms = Some(0);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete after retry");
    let requests = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Retry OK"));
    assert_eq!(message.response_id.as_deref(), Some("resp_codex_retry"));
    assert_eq!(message.usage.total_tokens, 8);
    assert_eq!(requests.len(), 2);
    for request in &requests {
        assert!(request.starts_with("POST /codex/responses HTTP/1.1"));
        assert!(request.contains("\"prompt_cache_key\":\"codex-retry-session\""));
    }
}

#[tokio::test]
async fn builtin_openai_codex_provider_respects_abort_flag_while_streaming() {
    let (base_url, request_task) = mock_delayed_sse_server(
        vec![
            "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_codex_abort\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_codex_abort\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Code\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_codex_abort\",\"content\":[{\"type\":\"output_text\",\"text\":\"Code\"}]}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_codex_abort\",\"status\":\"completed\",\"usage\":{\"input_tokens\":6,\"output_tokens\":2,\"total_tokens\":8}}}\n\n",
        ],
        Duration::from_millis(100),
    )
    .await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.abort_flag = Some(abort_flag.clone());
    options.stream.transport = Some(Transport::Sse);

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Code")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Code")
            && error.usage.input == 0
            && error.usage.output == 0
            && error.usage.total_tokens == 0
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test]
async fn builtin_openai_codex_provider_completes_after_terminal_sse_without_eof() {
    for (terminal_event, expected_stop_reason) in [
        ("response.completed", StopReason::Stop),
        ("response.incomplete", StopReason::Length),
    ] {
        let status = if terminal_event == "response.incomplete" {
            "incomplete"
        } else {
            "completed"
        };
        let sse = format!(
            "data: {{\"type\":\"response.output_item.added\",\"item\":{{\"type\":\"message\",\"id\":\"msg_open\",\"role\":\"assistant\",\"content\":[]}}}}\n\n\
             data: {{\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}}\n\n\
             data: {{\"type\":\"response.output_item.done\",\"item\":{{\"type\":\"message\",\"id\":\"msg_open\",\"content\":[{{\"type\":\"output_text\",\"text\":\"Hello\"}}]}}}}\n\n\
             data: {{\"type\":\"{terminal_event}\",\"response\":{{\"id\":\"resp_open\",\"status\":\"{status}\",\"usage\":{{\"input_tokens\":5,\"output_tokens\":3,\"total_tokens\":8}}}}}}\n\n"
        );
        let (base_url, request_task) = mock_open_sse_server(sse.into_bytes()).await;
        let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
        model.base_url = base_url;
        let mut options = SimpleStreamOptions::default();
        options.stream.api_key = Some(codex_test_token());
        options.stream.transport = Some(Transport::Sse);

        let message = tokio::time::timeout(
            Duration::from_secs(1),
            complete_simple(&model, user_context("hello"), options),
        )
        .await
        .unwrap_or_else(|_| panic!("{terminal_event} should finish without waiting for EOF"))
        .expect("complete");

        assert_eq!(text_of(&message), Some("Hello"), "{terminal_event}");
        assert_eq!(
            message.stop_reason, expected_stop_reason,
            "{terminal_event}"
        );
        assert_eq!(message.response_id.as_deref(), Some("resp_open"));
        let request = tokio::time::timeout(Duration::from_secs(1), request_task)
            .await
            .expect("client should close open SSE connection after terminal event")
            .expect("request task");
        assert!(request.starts_with("POST /codex/responses HTTP/1.1"));
    }
}

#[tokio::test]
async fn builtin_openai_codex_provider_uses_websocket_transport_and_parses_frames() {
    let events = vec![
        json!({ "type": "response.created", "response": { "id": "resp_ws" } }),
        json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_ws", "role": "assistant", "content": [] }
        }),
        json!({ "type": "response.output_text.delta", "delta": "Hello" }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_ws",
                "content": [{ "type": "output_text", "text": "Hello" }]
            }
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_ws",
                "status": "completed",
                "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 }
            }
        }),
    ];
    let (base_url, request_task) = mock_websocket_server(events).await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("ws-session".to_owned());
    options.stream.transport = Some(Transport::Websocket);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hello"));
    assert_eq!(message.response_id.as_deref(), Some("resp_ws"));
    assert_eq!(message.usage.input, 5);
    assert_eq!(message.usage.output, 3);
    assert!(
        request
            .handshake
            .starts_with("GET /codex/responses HTTP/1.1")
    );
    assert!(
        request
            .handshake
            .to_ascii_lowercase()
            .contains("authorization: bearer")
    );
    assert!(
        request
            .handshake
            .to_ascii_lowercase()
            .contains("chatgpt-account-id: acc_test")
    );
    assert!(
        request
            .handshake
            .to_ascii_lowercase()
            .contains("session_id: ws-session")
    );
    let body: Value = serde_json::from_str(&request.message).expect("websocket request JSON");
    assert_eq!(body["type"], "response.create");
    assert_eq!(body["model"], "gpt-5.5");
    assert_eq!(body["stream"], true);
    assert_eq!(body["prompt_cache_key"], "ws-session");
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_openai_codex_websocket_routes_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let events = vec![
        json!({ "type": "response.created", "response": { "id": "resp_ws_proxy" } }),
        json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_ws_proxy", "role": "assistant", "content": [] }
        }),
        json!({ "type": "response.output_text.delta", "delta": "Proxy WS" }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_ws_proxy",
                "content": [{ "type": "output_text", "text": "Proxy WS" }]
            }
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_ws_proxy",
                "status": "completed",
                "usage": { "input_tokens": 7, "output_tokens": 2, "total_tokens": 9 }
            }
        }),
    ];
    let (proxy_url, request_task) = mock_websocket_connect_proxy_server(events).await;
    let authenticated_proxy_url = proxy_url.replacen("http://", "http://user:pass@", 1);
    set_env("HTTP_PROXY", &authenticated_proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = "http://codex.example/backend-api".to_owned();
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("ws-proxy-session".to_owned());
    options.stream.transport = Some(Transport::Websocket);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Proxy WS"));
    assert_eq!(message.response_id.as_deref(), Some("resp_ws_proxy"));
    assert_eq!(message.usage.input, 7);
    assert_eq!(message.usage.output, 2);
    assert!(
        request
            .connect
            .starts_with("CONNECT codex.example:80 HTTP/1.1"),
        "{}",
        request.connect
    );
    assert!(
        request
            .connect
            .to_ascii_lowercase()
            .contains("host: codex.example:80")
    );
    assert!(
        request
            .connect
            .to_ascii_lowercase()
            .contains("proxy-authorization: basic dxnlcjpwyxnz")
    );
    assert!(
        request
            .handshake
            .starts_with("GET /backend-api/codex/responses HTTP/1.1"),
        "{}",
        request.handshake
    );
    assert!(
        request
            .handshake
            .to_ascii_lowercase()
            .contains("host: codex.example")
    );
    assert!(
        request
            .handshake
            .to_ascii_lowercase()
            .contains("session_id: ws-proxy-session")
    );
    let body: Value = serde_json::from_str(&request.message).expect("websocket request JSON");
    assert_eq!(body["type"], "response.create");
    assert_eq!(body["prompt_cache_key"], "ws-proxy-session");
}

#[tokio::test]
async fn builtin_openai_codex_provider_auto_falls_back_to_sse_when_websocket_fails_before_events() {
    let sse = concat!(
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_fallback\"}}\n\n",
        "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_fallback\"}}\n\n",
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Fallback\"}\n\n",
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"id\":\"msg_fallback\",\"content\":[{\"type\":\"output_text\",\"text\":\"Fallback\"}]}}\n\n",
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_fallback\",\"status\":\"completed\",\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":6}}}\n\n",
    );
    let (base_url, request_task) = mock_websocket_reject_then_sse_server(sse).await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("ws-fallback-session".to_owned());
    options.stream.transport = Some(Transport::Auto);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Fallback"));
    assert_eq!(message.response_id.as_deref(), Some("resp_fallback"));
    assert_eq!(message.usage.input, 4);
    assert_eq!(message.usage.output, 2);
    let diagnostic = message
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic["type"] == "provider_transport_failure")
        .expect("transport failure diagnostic");
    assert!(diagnostic["timestamp"].as_i64().is_some());
    assert_eq!(diagnostic["error"]["name"].as_str(), Some("ThrownValue"));
    assert!(
        diagnostic["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("404"))
    );
    assert_eq!(diagnostic["details"]["configuredTransport"], "auto");
    assert_eq!(diagnostic["details"]["fallbackTransport"], "sse");
    assert_eq!(diagnostic["details"]["eventsEmitted"], false);
    assert_eq!(
        diagnostic["details"]["phase"],
        "before_message_stream_start"
    );
    assert!(diagnostic["details"]["requestBytes"].as_u64().is_some());
    assert!(diagnostic.get("configuredTransport").is_none());
    assert!(
        request
            .websocket_handshake
            .starts_with("GET /codex/responses HTTP/1.1")
    );
    assert!(
        request
            .sse_request
            .starts_with("POST /codex/responses HTTP/1.1")
    );
    assert!(
        request
            .sse_request
            .to_ascii_lowercase()
            .contains("openai-beta: responses=experimental")
    );
}

#[tokio::test]
async fn builtin_openai_codex_provider_reuses_websocket_cached_context_for_session() {
    let responses = vec![
        vec![
            json!({ "type": "response.created", "response": { "id": "resp_1" } }),
            json!({
                "type": "response.output_item.added",
                "item": { "type": "message", "id": "msg_1", "role": "assistant", "content": [] }
            }),
            json!({ "type": "response.output_text.delta", "delta": "Hello" }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{ "type": "output_text", "text": "Hello" }]
                }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_1",
                    "status": "completed",
                    "usage": { "input_tokens": 5, "output_tokens": 3, "total_tokens": 8 }
                }
            }),
        ],
        vec![
            json!({ "type": "response.created", "response": { "id": "resp_2" } }),
            json!({
                "type": "response.output_item.added",
                "item": { "type": "message", "id": "msg_2", "role": "assistant", "content": [] }
            }),
            json!({ "type": "response.output_text.delta", "delta": "Done" }),
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "msg_2",
                    "content": [{ "type": "output_text", "text": "Done" }]
                }
            }),
            json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_2",
                    "status": "completed",
                    "usage": { "input_tokens": 6, "output_tokens": 2, "total_tokens": 8 }
                }
            }),
        ],
    ];
    let (base_url, request_task) = mock_reusable_websocket_server(responses).await;
    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some("ws-cache-session".to_owned());
    options.stream.transport = Some(Transport::Auto);

    let first_context = user_context("Say hello");
    let first = complete_simple(&model, first_context.clone(), options.clone())
        .await
        .expect("first complete");
    assert_eq!(text_of(&first), Some("Hello"));
    assert_eq!(first.response_id.as_deref(), Some("resp_1"));

    let second_context = Context {
        messages: vec![
            Message::User(UserMessage::text("Say hello")),
            Message::Assistant(first),
            Message::User(UserMessage::text("Continue")),
        ],
        ..Default::default()
    };
    let second = complete_simple(&model, second_context, options)
        .await
        .expect("second complete");
    assert_eq!(text_of(&second), Some("Done"));
    assert_eq!(second.response_id.as_deref(), Some("resp_2"));

    let request = request_task.await.expect("request task");
    assert!(
        request
            .handshake
            .starts_with("GET /codex/responses HTTP/1.1")
    );
    assert_eq!(request.messages.len(), 2);
    let first_body: Value = serde_json::from_str(&request.messages[0]).expect("first body");
    let second_body: Value = serde_json::from_str(&request.messages[1]).expect("second body");
    assert!(first_body.get("previous_response_id").is_none());
    assert_eq!(second_body["previous_response_id"], "resp_1");
    assert_eq!(
        second_body["input"].as_array().map(Vec::len),
        Some(1),
        "second websocket request should send only the new user input"
    );
    assert_eq!(second_body["input"][0]["role"], "user");
}

#[tokio::test]
async fn session_resource_cleanup_removes_openai_codex_websocket_cache_for_session() {
    let session_id = "ws-cleanup-session";
    let (first_base_url, first_request_task) = mock_reusable_websocket_server(vec![vec![
        json!({ "type": "response.created", "response": { "id": "resp_cleanup_1" } }),
        json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_cleanup_1", "role": "assistant", "content": [] }
        }),
        json!({ "type": "response.output_text.delta", "delta": "Cached" }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_cleanup_1",
                "content": [{ "type": "output_text", "text": "Cached" }]
            }
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_cleanup_1",
                "status": "completed",
                "usage": { "input_tokens": 3, "output_tokens": 2, "total_tokens": 5 }
            }
        }),
    ]])
    .await;

    let mut model = get_model("openai-codex", "gpt-5.5").expect("codex model");
    model.base_url = first_base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some(codex_test_token());
    options.stream.session_id = Some(session_id.to_owned());
    options.stream.transport = Some(Transport::Auto);

    let first = complete_simple(&model, user_context("First"), options.clone())
        .await
        .expect("first complete");
    assert_eq!(text_of(&first), Some("Cached"));
    let first_request = first_request_task.await.expect("first request task");
    assert_eq!(first_request.messages.len(), 1);

    let report = cleanup_session_resources(Some(session_id))
        .await
        .expect("cleanup session resources");
    assert_eq!(report.openai_codex_websocket_sessions, 1);
    assert_eq!(report.cleaned_count(), 1);

    let (second_base_url, second_request_task) = mock_reusable_websocket_server(vec![vec![
        json!({ "type": "response.created", "response": { "id": "resp_cleanup_2" } }),
        json!({
            "type": "response.output_item.added",
            "item": { "type": "message", "id": "msg_cleanup_2", "role": "assistant", "content": [] }
        }),
        json!({ "type": "response.output_text.delta", "delta": "Fresh" }),
        json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "id": "msg_cleanup_2",
                "content": [{ "type": "output_text", "text": "Fresh" }]
            }
        }),
        json!({
            "type": "response.completed",
            "response": {
                "id": "resp_cleanup_2",
                "status": "completed",
                "usage": { "input_tokens": 4, "output_tokens": 2, "total_tokens": 6 }
            }
        }),
    ]])
    .await;
    model.base_url = second_base_url;

    let second = complete_simple(&model, user_context("Second"), options.clone())
        .await
        .expect("second complete");
    assert_eq!(text_of(&second), Some("Fresh"));
    let second_request = second_request_task.await.expect("second request task");
    assert_eq!(second_request.messages.len(), 1);
    let second_body: Value =
        serde_json::from_str(&second_request.messages[0]).expect("second body");
    assert!(
        second_body.get("previous_response_id").is_none(),
        "cleanup must remove the cached continuation for the session"
    );

    let report = cleanup_session_resources(Some(session_id))
        .await
        .expect("cleanup session resources again");
    assert_eq!(report.openai_codex_websocket_sessions, 1);
}

#[tokio::test]
async fn session_resource_cleanup_runs_registered_hooks_and_aggregates_errors() {
    let calls = Arc::new(Mutex::new(Vec::<String>::new()));
    let ok_calls = calls.clone();
    let ok_registration = register_session_resource_cleanup(move |session_id| {
        ok_calls
            .lock()
            .expect("calls lock")
            .push(format!("ok:{}", session_id.unwrap_or("<all>")));
        Ok(())
    });
    let first_error_calls = calls.clone();
    let first_error_registration = register_session_resource_cleanup(move |session_id| {
        first_error_calls
            .lock()
            .expect("calls lock")
            .push(format!("err1:{}", session_id.unwrap_or("<all>")));
        Err("first cleanup failed".to_owned())
    });
    let second_error_calls = calls.clone();
    let second_error_registration = register_session_resource_cleanup(move |session_id| {
        second_error_calls
            .lock()
            .expect("calls lock")
            .push(format!("err2:{}", session_id.unwrap_or("<all>")));
        Err("second cleanup failed".to_owned())
    });

    let error = cleanup_session_resources(Some("registered-session"))
        .await
        .expect_err("registered cleanup failures should aggregate");
    assert_eq!(
        error.errors,
        vec![
            "first cleanup failed".to_owned(),
            "second cleanup failed".to_owned()
        ]
    );
    assert_eq!(
        calls.lock().expect("calls lock").as_slice(),
        [
            "ok:registered-session",
            "err1:registered-session",
            "err2:registered-session"
        ]
    );

    drop(first_error_registration);
    drop(second_error_registration);
    let report = cleanup_session_resources(None)
        .await
        .expect("remaining registered cleanup succeeds");
    let _ = report.cleaned_count();
    assert_eq!(
        calls.lock().expect("calls lock").as_slice(),
        [
            "ok:registered-session",
            "err1:registered-session",
            "err2:registered-session",
            "ok:<all>"
        ]
    );

    drop(ok_registration);
    cleanup_session_resources(Some("registered-session"))
        .await
        .expect("dropping registration unregisters cleanup");
    assert_eq!(
        calls.lock().expect("calls lock").as_slice(),
        [
            "ok:registered-session",
            "err1:registered-session",
            "err2:registered-session",
            "ok:<all>"
        ]
    );
}

#[tokio::test]
async fn builtin_bedrock_provider_posts_json_and_parses_eventstream() {
    let body = aws_eventstream_body(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "Hi" }
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
        json!({
            "metadata": {
                "usage": {
                    "inputTokens": 3,
                    "outputTokens": 1,
                    "totalTokens": 4
                }
            }
        }),
    ]);
    let (base_url, request_task) =
        mock_binary_server(body, "application/vnd.amazon.eventstream").await;
    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("bedrock-token".to_owned());
    options.stream.cache_retention = Some(CacheRetention::None);
    options
        .stream
        .extra
        .insert("requestMetadata".to_owned(), json!({ "app": "pi-test" }));

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");

    assert_eq!(text_of(&message), Some("Hi"));
    assert_eq!(message.usage.input, 3);
    assert_eq!(message.usage.output, 1);
    assert!(
        request.starts_with("POST /model/anthropic.claude-test-v1%3A0/converse-stream HTTP/1.1")
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer bedrock-token")
    );
    assert!(request.contains("\"modelId\":\"anthropic.claude-test-v1:0\""));
    assert!(request.contains("\"requestMetadata\":{\"app\":\"pi-test\"}"));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_bedrock_provider_routes_runtime_request_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);

    let body = aws_eventstream_body(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "Bedrock proxy" }
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
        json!({
            "metadata": {
                "usage": {
                    "inputTokens": 5,
                    "outputTokens": 2,
                    "totalTokens": 7
                }
            }
        }),
    ]);
    let (proxy_url, proxy_request_task) =
        mock_binary_server(body, "application/vnd.amazon.eventstream").await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");

    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = "http://bedrock-runtime.us-east-1.amazonaws.com".to_owned();
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("bedrock-token".to_owned());
    options.stream.cache_retention = Some(CacheRetention::None);

    let message = tokio::time::timeout(
        Duration::from_secs(1),
        complete_simple(&model, user_context("hello"), options),
    )
    .await
    .expect("proxied Bedrock runtime request should complete")
    .expect("complete");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(text_of(&message), Some("Bedrock proxy"));
    assert_eq!(message.usage.input, 5);
    assert_eq!(message.usage.output, 2);
    assert!(
        proxy_request.starts_with(
            "POST http://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-test-v1%3A0/converse-stream HTTP/1.1"
        ),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: bearer bedrock-token")
    );
}

#[tokio::test]
async fn builtin_bedrock_provider_respects_abort_flag_while_streaming() {
    let chunks = vec![
        aws_eventstream_frame(
            json!({ "messageStart": { "role": "assistant" } })
                .to_string()
                .as_bytes(),
        ),
        aws_eventstream_frame(
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "text": "Hel" }
                }
            })
            .to_string()
            .as_bytes(),
        ),
        aws_eventstream_frame(
            json!({
                "contentBlockDelta": {
                    "contentBlockIndex": 0,
                    "delta": { "text": "lo" }
                }
            })
            .to_string()
            .as_bytes(),
        ),
        aws_eventstream_frame(
            json!({ "contentBlockStop": { "contentBlockIndex": 0 } })
                .to_string()
                .as_bytes(),
        ),
        aws_eventstream_frame(
            json!({ "messageStop": { "stopReason": "end_turn" } })
                .to_string()
                .as_bytes(),
        ),
    ];
    let (base_url, request_task) = mock_delayed_binary_server(
        chunks,
        "application/vnd.amazon.eventstream",
        Duration::from_millis(100),
    )
    .await;
    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = base_url;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("bedrock-token".to_owned());
    options.stream.cache_retention = Some(CacheRetention::None);
    options.stream.abort_flag = Some(abort_flag.clone());

    let mut stream = stream_simple(&model, user_context("hello"), options).expect("stream");
    let mut events = Vec::new();
    loop {
        let event = tokio::time::timeout(Duration::from_secs(2), stream.next())
            .await
            .expect("stream should produce terminal event")
            .expect("stream event");
        if matches!(event, AssistantMessageEvent::TextDelta { .. }) {
            abort_flag.store(true, Ordering::SeqCst);
        }
        let terminal = matches!(
            event,
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. }
        );
        events.push(event);
        if terminal {
            break;
        }
    }

    assert!(events.iter().any(
        |event| matches!(event, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel")
    ));
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AssistantMessageEvent::TextEnd { .. }))
    );
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error,
        }) if error.stop_reason == StopReason::Aborted
            && error.error_message.as_deref() == Some("Request was aborted")
            && text_of(error) == Some("Hel")
            && error.usage.input == 0
            && error.usage.output == 0
            && error.usage.total_tokens == 0
    ));
    let _request = request_task.await.expect("request task");
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_bedrock_provider_signs_with_sigv4_env_credentials() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _guard = EnvGuard::clearing(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_BEDROCK_SKIP_AUTH",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
    ]);
    set_env("AWS_ACCESS_KEY_ID", "AKIDENV");
    set_env("AWS_SECRET_ACCESS_KEY", "env-secret");
    set_env("AWS_SESSION_TOKEN", "env-session");
    set_env("AWS_REGION", "us-west-2");

    let body = aws_eventstream_body(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "Signed" }
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
        json!({
            "metadata": {
                "usage": {
                    "inputTokens": 7,
                    "outputTokens": 2,
                    "totalTokens": 9
                }
            }
        }),
    ]);
    let (base_url, request_task) =
        mock_binary_server(body, "application/vnd.amazon.eventstream").await;
    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.cache_retention = Some(CacheRetention::None);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let request = request_task.await.expect("request task");
    let lower = request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Signed"));
    assert_eq!(message.usage.input, 7);
    assert_eq!(message.usage.output, 2);
    assert!(lower.contains("authorization: aws4-hmac-sha256 credential=akidenv/"));
    assert!(request.contains("/us-west-2/bedrock/aws4_request"));
    assert!(lower.contains("x-amz-date:"));
    assert!(lower.contains("x-amz-content-sha256:"));
    assert!(lower.contains("x-amz-security-token: env-session"));
    assert!(lower.contains(
        "signedheaders=accept;content-type;host;x-amz-content-sha256;x-amz-date;x-amz-security-token"
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_bedrock_provider_signs_with_web_identity_credentials() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _proxy_guard = EnvGuard::clearing(PROXY_ENV_KEYS);
    let _guard = EnvGuard::clearing(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_BEDROCK_SKIP_AUTH",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_STS",
    ]);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ri-aws-web-identity-{unique}"));
    std::fs::create_dir_all(&dir).expect("create web identity temp dir");
    let token_path = dir.join("token.jwt");
    std::fs::write(&token_path, "web-identity-token").expect("write web identity token");
    let (sts_url, sts_requests_task) = mock_http_sequence_server(vec![MockHttpSequenceResponse {
        status: 200,
        reason: "OK",
        content_type: "text/xml",
        headers: vec![],
        body: concat!(
            "<AssumeRoleWithWebIdentityResponse>",
            "<AssumeRoleWithWebIdentityResult><Credentials>",
            "<AccessKeyId>ASIAWEBIDENTITY</AccessKeyId>",
            "<SecretAccessKey>web-identity-secret</SecretAccessKey>",
            "<SessionToken>web-identity-session</SessionToken>",
            "</Credentials></AssumeRoleWithWebIdentityResult>",
            "</AssumeRoleWithWebIdentityResponse>"
        ),
    }])
    .await;
    set_env(
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        token_path.to_str().expect("token path"),
    );
    set_env("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/ri-test");
    set_env("AWS_ROLE_SESSION_NAME", "ri-test-session");
    set_env("AWS_ENDPOINT_URL_STS", &sts_url);
    set_env("AWS_REGION", "us-east-1");

    let body = aws_eventstream_body(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "Web identity signed" }
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]);
    let (base_url, bedrock_request_task) =
        mock_binary_server(body, "application/vnd.amazon.eventstream").await;
    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.cache_retention = Some(CacheRetention::None);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let sts_requests = sts_requests_task.await.expect("sts request task");
    let bedrock_request = bedrock_request_task.await.expect("bedrock request task");
    let sts_request = sts_requests.first().expect("sts request");
    let lower = bedrock_request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Web identity signed"));
    assert_eq!(sts_requests.len(), 1);
    assert!(sts_request.starts_with("POST / HTTP/1.1"), "{sts_request}");
    assert!(sts_request.contains("Action=AssumeRoleWithWebIdentity"));
    assert!(sts_request.contains("RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Fri-test"));
    assert!(sts_request.contains("RoleSessionName=ri-test-session"));
    assert!(sts_request.contains("WebIdentityToken=web-identity-token"));
    assert!(lower.contains("authorization: aws4-hmac-sha256 credential=asiawebidentity/"));
    assert!(bedrock_request.contains("/us-east-1/bedrock/aws4_request"));
    assert!(lower.contains("x-amz-security-token: web-identity-session"));

    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test(flavor = "current_thread")]
async fn builtin_bedrock_provider_signs_with_ecs_container_credentials() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _proxy_guard = EnvGuard::clearing(PROXY_ENV_KEYS);
    let _guard = EnvGuard::clearing(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_BEDROCK_SKIP_AUTH",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ]);
    let (credentials_url, credentials_request_task) = mock_json_server(
        r#"{"AccessKeyId":"AKIDCONTAINER","SecretAccessKey":"container-secret","Token":"container-session"}"#,
    )
    .await;
    set_env("AWS_CONTAINER_CREDENTIALS_FULL_URI", &credentials_url);
    set_env("AWS_CONTAINER_AUTHORIZATION_TOKEN", "container-auth");
    set_env("AWS_REGION", "us-east-1");

    let body = aws_eventstream_body(vec![
        json!({ "messageStart": { "role": "assistant" } }),
        json!({
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": { "text": "Container signed" }
            }
        }),
        json!({ "contentBlockStop": { "contentBlockIndex": 0 } }),
        json!({ "messageStop": { "stopReason": "end_turn" } }),
    ]);
    let (base_url, bedrock_request_task) =
        mock_binary_server(body, "application/vnd.amazon.eventstream").await;
    let mut model = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    model.base_url = base_url;
    let mut options = SimpleStreamOptions::default();
    options.stream.cache_retention = Some(CacheRetention::None);

    let message = complete_simple(&model, user_context("hello"), options)
        .await
        .expect("complete");
    let credentials_request = credentials_request_task
        .await
        .expect("credentials request task");
    let bedrock_request = bedrock_request_task.await.expect("bedrock request task");
    let lower = bedrock_request.to_ascii_lowercase();

    assert_eq!(text_of(&message), Some("Container signed"));
    assert!(
        credentials_request
            .to_ascii_lowercase()
            .contains("authorization: container-auth")
    );
    assert!(lower.contains("authorization: aws4-hmac-sha256 credential=akidcontainer/"));
    assert!(bedrock_request.contains("/us-east-1/bedrock/aws4_request"));
    assert!(lower.contains("x-amz-security-token: container-session"));
}

#[tokio::test(flavor = "current_thread")]
async fn bedrock_ecs_container_credentials_route_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _proxy_guard = EnvGuard::clearing(PROXY_ENV_KEYS);
    let _guard = EnvGuard::clearing(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_BEDROCK_SKIP_AUTH",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ]);
    let (proxy_url, proxy_request_task) = mock_json_server(
        r#"{"AccessKeyId":"AKIDPROXY","SecretAccessKey":"proxy-secret","Token":"proxy-session"}"#,
    )
    .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");
    set_env(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "http://ecs.example/credentials",
    );
    set_env("AWS_CONTAINER_AUTHORIZATION_TOKEN", "container-proxy-auth");

    let credentials = resolve_aws_credentials_with_container(None)
        .await
        .expect("resolve container credentials")
        .expect("container credentials");
    let proxy_request = proxy_request_task.await.expect("proxy request task");

    assert_eq!(credentials.access_key_id, "AKIDPROXY");
    assert_eq!(credentials.secret_access_key, "proxy-secret");
    assert_eq!(credentials.session_token.as_deref(), Some("proxy-session"));
    assert!(
        proxy_request.starts_with("GET http://ecs.example/credentials HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(
        proxy_request
            .to_ascii_lowercase()
            .contains("authorization: container-proxy-auth")
    );
}

#[tokio::test(flavor = "current_thread")]
async fn bedrock_web_identity_credentials_route_sts_through_resolved_proxy() {
    let _lock = ENV_LOCK.lock().expect("env lock");
    let _proxy_guard = EnvGuard::clearing(PROXY_ENV_KEYS);
    let _guard = EnvGuard::clearing(&[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_BEDROCK_SKIP_AUTH",
        "AWS_PROFILE",
        "AWS_SHARED_CREDENTIALS_FILE",
        "AWS_CONFIG_FILE",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_STS",
    ]);
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("ri-aws-web-identity-proxy-{unique}"));
    std::fs::create_dir_all(&dir).expect("create web identity temp dir");
    let token_path = dir.join("token.jwt");
    std::fs::write(&token_path, "proxied-web-identity-token").expect("write token");
    let (proxy_url, proxy_request_task) =
        mock_http_sequence_server(vec![MockHttpSequenceResponse {
            status: 200,
            reason: "OK",
            content_type: "text/xml",
            headers: vec![],
            body: concat!(
                "<AssumeRoleWithWebIdentityResponse>",
                "<AssumeRoleWithWebIdentityResult><Credentials>",
                "<AccessKeyId>ASIAPROXYSTS</AccessKeyId>",
                "<SecretAccessKey>proxy-sts-secret</SecretAccessKey>",
                "<SessionToken>proxy-sts-session</SessionToken>",
                "</Credentials></AssumeRoleWithWebIdentityResult>",
                "</AssumeRoleWithWebIdentityResponse>"
            ),
        }])
        .await;
    set_env("HTTP_PROXY", &proxy_url);
    set_env("NO_PROXY", "127.0.0.1,localhost");
    set_env(
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        token_path.to_str().expect("token path"),
    );
    set_env("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/proxy-test");
    set_env("AWS_ROLE_SESSION_NAME", "ri-proxy-session");
    set_env("AWS_ENDPOINT_URL_STS", "http://sts.example/");

    let credentials = resolve_aws_credentials_with_runtime(None)
        .await
        .expect("resolve web identity credentials")
        .expect("web identity credentials");
    let proxy_requests = proxy_request_task.await.expect("proxy request task");
    let proxy_request = proxy_requests.first().expect("proxy request");

    assert_eq!(credentials.access_key_id, "ASIAPROXYSTS");
    assert_eq!(credentials.secret_access_key, "proxy-sts-secret");
    assert_eq!(
        credentials.session_token.as_deref(),
        Some("proxy-sts-session")
    );
    assert_eq!(proxy_requests.len(), 1);
    assert!(
        proxy_request.starts_with("POST http://sts.example/ HTTP/1.1"),
        "{proxy_request}"
    );
    assert!(proxy_request.contains("Action=AssumeRoleWithWebIdentity"));
    assert!(
        proxy_request.contains("RoleArn=arn%3Aaws%3Aiam%3A%3A123456789012%3Arole%2Fproxy-test")
    );
    assert!(proxy_request.contains("RoleSessionName=ri-proxy-session"));
    assert!(proxy_request.contains("WebIdentityToken=proxied-web-identity-token"));

    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn builtin_http_providers_respect_pre_aborted_flag_before_request() {
    let mut openai_completions = Model::faux("openai-completions", "openai", "mock-model");
    openai_completions.base_url = "http://127.0.0.1:9".to_owned();

    let mut openai_responses = Model::faux("openai-responses", "openai", "mock-gpt-5");
    openai_responses.base_url = "http://127.0.0.1:9".to_owned();

    let mut azure = Model::faux(
        "azure-openai-responses",
        "azure-openai-responses",
        "gpt-4o-mini",
    );
    azure.base_url = "http://127.0.0.1:9/openai/v1".to_owned();

    let mut mistral = Model::faux("mistral-conversations", "mistral", "mistral-small-latest");
    mistral.base_url = "http://127.0.0.1:9".to_owned();

    let mut anthropic = Model::faux("anthropic-messages", "anthropic", "claude-test");
    anthropic.base_url = "http://127.0.0.1:9".to_owned();

    let mut google = Model::faux("google-generative-ai", "google", "gemini-test");
    google.base_url = "http://127.0.0.1:9".to_owned();

    let mut vertex = Model::faux("google-vertex", "google-vertex", "gemini-test");
    vertex.base_url = "http://127.0.0.1:9".to_owned();

    let mut codex = get_model("openai-codex", "gpt-5.5").expect("codex model");
    codex.base_url = "http://127.0.0.1:9".to_owned();

    let mut bedrock = Model::faux(
        "bedrock-converse-stream",
        "amazon-bedrock",
        "anthropic.claude-test-v1:0",
    );
    bedrock.base_url = "http://127.0.0.1:9".to_owned();

    let cases = vec![
        (openai_completions, "test-key".to_owned()),
        (openai_responses, "test-key".to_owned()),
        (azure, "azure-key".to_owned()),
        (mistral, "mistral-key".to_owned()),
        (anthropic, "anthropic-key".to_owned()),
        (google, "gemini-key".to_owned()),
        (vertex, "vertex-key".to_owned()),
        (codex, codex_test_token()),
        (bedrock, "bedrock-token".to_owned()),
    ];

    for (model, api_key) in cases {
        let abort_flag = Arc::new(AtomicBool::new(true));
        let mut options = SimpleStreamOptions::default();
        options.stream.api_key = Some(api_key);
        options.stream.abort_flag = Some(abort_flag);

        let message = tokio::time::timeout(
            Duration::from_secs(1),
            complete_simple(&model, user_context("hello"), options),
        )
        .await
        .unwrap_or_else(|_| panic!("{} should abort before any HTTP request", model.api))
        .unwrap_or_else(|error| panic!("{} returned provider error: {error}", model.api));

        assert_eq!(message.stop_reason, StopReason::Aborted, "{}", model.api);
        assert_eq!(
            message.error_message.as_deref(),
            Some("Request was aborted"),
            "{}",
            model.api
        );
        assert!(
            message.content.is_empty(),
            "{} should abort before receiving content",
            model.api
        );
    }
}

async fn mock_sse_server(body: &'static str) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

async fn mock_open_sse_server(body: Vec<u8>) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: keep-alive\r\n\r\n",
            )
            .await
            .expect("write headers");
        socket.write_all(&body).await.expect("write body");
        socket.flush().await.expect("flush body");
        loop {
            let n = socket.read(&mut buf).await.unwrap_or(0);
            if n == 0 {
                break;
            }
        }
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

struct MockWebSocketRequest {
    handshake: String,
    message: String,
    messages: Vec<String>,
}

struct MockWebSocketProxyRequest {
    connect: String,
    handshake: String,
    message: String,
}

struct MockCodexFallbackRequest {
    websocket_handshake: String,
    sse_request: String,
}

async fn mock_websocket_server(
    events: Vec<Value>,
) -> (String, tokio::task::JoinHandle<MockWebSocketRequest>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind websocket mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut handshake = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read handshake");
            if n == 0 {
                break;
            }
            handshake.extend_from_slice(&buf[..n]);
            if request_is_complete(&handshake) {
                break;
            }
        }
        socket
            .write_all(
                b"HTTP/1.1 101 Switching Protocols\r\n\
                  Upgrade: websocket\r\n\
                  Connection: Upgrade\r\n\
                  Sec-WebSocket-Accept: test\r\n\r\n",
            )
            .await
            .expect("write handshake response");
        let message = read_client_websocket_text_frame(&mut socket).await;
        for event in events {
            socket
                .write_all(&server_websocket_text_frame(&event.to_string()))
                .await
                .expect("write websocket frame");
        }
        let _ = socket.shutdown().await;
        MockWebSocketRequest {
            handshake: String::from_utf8_lossy(&handshake).into_owned(),
            message: message.clone(),
            messages: vec![message],
        }
    });
    (format!("http://{addr}"), task)
}

async fn mock_websocket_connect_proxy_server(
    events: Vec<Value>,
) -> (String, tokio::task::JoinHandle<MockWebSocketProxyRequest>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind websocket proxy mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept proxy request");
        let mut connect = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read CONNECT");
            if n == 0 {
                break;
            }
            connect.extend_from_slice(&buf[..n]);
            if request_is_complete(&connect) {
                break;
            }
        }
        socket
            .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            .await
            .expect("write CONNECT response");

        let mut handshake = Vec::new();
        loop {
            let n = socket
                .read(&mut buf)
                .await
                .expect("read websocket handshake");
            if n == 0 {
                break;
            }
            handshake.extend_from_slice(&buf[..n]);
            if request_is_complete(&handshake) {
                break;
            }
        }
        socket
            .write_all(
                b"HTTP/1.1 101 Switching Protocols\r\n\
                  Upgrade: websocket\r\n\
                  Connection: Upgrade\r\n\
                  Sec-WebSocket-Accept: test\r\n\r\n",
            )
            .await
            .expect("write websocket handshake response");
        let message = read_client_websocket_text_frame(&mut socket).await;
        for event in events {
            socket
                .write_all(&server_websocket_text_frame(&event.to_string()))
                .await
                .expect("write websocket frame");
        }
        let _ = socket.shutdown().await;
        MockWebSocketProxyRequest {
            connect: String::from_utf8_lossy(&connect).into_owned(),
            handshake: String::from_utf8_lossy(&handshake).into_owned(),
            message,
        }
    });
    (format!("http://{addr}"), task)
}

async fn mock_websocket_reject_then_sse_server(
    body: &'static str,
) -> (String, tokio::task::JoinHandle<MockCodexFallbackRequest>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind websocket fallback mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let mut buf = [0u8; 1024];
        let (mut socket, _) = listener.accept().await.expect("accept websocket request");
        let mut websocket_handshake = Vec::new();
        loop {
            let n = socket.read(&mut buf).await.expect("read handshake");
            if n == 0 {
                break;
            }
            websocket_handshake.extend_from_slice(&buf[..n]);
            if request_is_complete(&websocket_handshake) {
                break;
            }
        }
        socket
            .write_all(b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await
            .expect("write websocket rejection");
        let _ = socket.shutdown().await;

        let (mut socket, _) = listener.accept().await.expect("accept sse request");
        let mut sse_request = Vec::new();
        loop {
            let n = socket.read(&mut buf).await.expect("read sse request");
            if n == 0 {
                break;
            }
            sse_request.extend_from_slice(&buf[..n]);
            if request_is_complete(&sse_request) {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write sse response");

        MockCodexFallbackRequest {
            websocket_handshake: String::from_utf8_lossy(&websocket_handshake).into_owned(),
            sse_request: String::from_utf8_lossy(&sse_request).into_owned(),
        }
    });
    (format!("http://{addr}"), task)
}

async fn mock_reusable_websocket_server(
    responses: Vec<Vec<Value>>,
) -> (String, tokio::task::JoinHandle<MockWebSocketRequest>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind reusable websocket mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut handshake = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read handshake");
            if n == 0 {
                break;
            }
            handshake.extend_from_slice(&buf[..n]);
            if request_is_complete(&handshake) {
                break;
            }
        }
        socket
            .write_all(
                b"HTTP/1.1 101 Switching Protocols\r\n\
                  Upgrade: websocket\r\n\
                  Connection: Upgrade\r\n\
                  Sec-WebSocket-Accept: test\r\n\r\n",
            )
            .await
            .expect("write handshake response");

        let mut messages = Vec::new();
        for response_events in responses {
            messages.push(read_client_websocket_text_frame(&mut socket).await);
            for event in response_events {
                socket
                    .write_all(&server_websocket_text_frame(&event.to_string()))
                    .await
                    .expect("write websocket frame");
            }
        }
        let _ = socket.shutdown().await;
        MockWebSocketRequest {
            handshake: String::from_utf8_lossy(&handshake).into_owned(),
            message: messages.first().cloned().unwrap_or_default(),
            messages,
        }
    });
    (format!("http://{addr}"), task)
}

async fn read_client_websocket_text_frame(socket: &mut tokio::net::TcpStream) -> String {
    use tokio::io::AsyncReadExt;

    let mut header = [0u8; 2];
    socket.read_exact(&mut header).await.expect("frame header");
    assert_eq!(header[0] & 0x0f, 0x1);
    assert_ne!(header[1] & 0x80, 0, "client frames must be masked");
    let mut len = usize::from(header[1] & 0x7f);
    if len == 126 {
        let mut extended = [0u8; 2];
        socket
            .read_exact(&mut extended)
            .await
            .expect("extended len");
        len = usize::from(u16::from_be_bytes(extended));
    } else if len == 127 {
        let mut extended = [0u8; 8];
        socket
            .read_exact(&mut extended)
            .await
            .expect("extended len");
        len = usize::try_from(u64::from_be_bytes(extended)).expect("frame len");
    }
    let mut mask = [0u8; 4];
    socket.read_exact(&mut mask).await.expect("mask");
    let mut payload = vec![0u8; len];
    socket.read_exact(&mut payload).await.expect("payload");
    for (index, byte) in payload.iter_mut().enumerate() {
        *byte ^= mask[index % 4];
    }
    String::from_utf8(payload).expect("text frame")
}

fn server_websocket_text_frame(text: &str) -> Vec<u8> {
    let payload = text.as_bytes();
    let mut frame = Vec::with_capacity(payload.len() + 10);
    frame.push(0x81);
    if payload.len() <= 125 {
        frame.push(payload.len() as u8);
    } else if payload.len() <= u16::MAX as usize {
        frame.push(126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    } else {
        frame.push(127);
        frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    }
    frame.extend_from_slice(payload);
    frame
}

async fn mock_json_server(body: impl Into<String>) -> (String, tokio::task::JoinHandle<String>) {
    mock_json_status_server(200, "OK", body).await
}

async fn mock_json_status_server(
    status: u16,
    reason: &'static str,
    body: impl Into<String>,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let body = body.into();
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body,
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write response");
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

async fn mock_json_sequence_server(
    bodies: Vec<&'static str>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for body in bodies {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = socket.read(&mut buf).await.expect("read request");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request_is_complete(&request) {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            requests.push(String::from_utf8_lossy(&request).into_owned());
        }
        requests
    });
    (format!("http://{addr}"), task)
}

async fn mock_json_status_sequence_server(
    responses: Vec<(u16, &'static str, &'static str)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for (status, reason, body) in responses {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = socket.read(&mut buf).await.expect("read request");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request_is_complete(&request) {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            requests.push(String::from_utf8_lossy(&request).into_owned());
        }
        requests
    });
    (format!("http://{addr}"), task)
}

struct MockHttpSequenceResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    headers: Vec<(&'static str, &'static str)>,
    body: &'static str,
}

async fn mock_http_sequence_server(
    responses: Vec<MockHttpSequenceResponse>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for response in responses {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = socket.read(&mut buf).await.expect("read request");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request_is_complete(&request) {
                    break;
                }
            }
            let mut header_text = format!(
                "HTTP/1.1 {} {}\r\ncontent-type: {}\r\ncontent-length: {}\r\nconnection: close\r\n",
                response.status,
                response.reason,
                response.content_type,
                response.body.len(),
            );
            for (name, value) in response.headers {
                header_text.push_str(name);
                header_text.push_str(": ");
                header_text.push_str(value);
                header_text.push_str("\r\n");
            }
            header_text.push_str("\r\n");
            socket
                .write_all(header_text.as_bytes())
                .await
                .expect("write headers");
            socket
                .write_all(response.body.as_bytes())
                .await
                .expect("write body");
            requests.push(String::from_utf8_lossy(&request).into_owned());
        }
        requests
    });
    (format!("http://{addr}"), task)
}

async fn mock_hanging_response_server() -> (
    String,
    tokio::sync::oneshot::Receiver<String>,
    tokio::task::JoinHandle<()>,
) {
    use tokio::{io::AsyncReadExt, net::TcpListener, sync::oneshot, time::sleep};

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind hanging mock server");
    let addr = listener.local_addr().expect("local addr");
    let (request_tx, request_rx) = oneshot::channel();
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        let _ = request_tx.send(String::from_utf8_lossy(&request).into_owned());
        sleep(Duration::from_secs(60)).await;
    });
    (format!("http://{addr}"), request_rx, task)
}

async fn oauth_callback_get(port: u16, target: &str) -> String {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpStream,
    };

    let mut socket = TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect callback server");
    let request = format!("GET {target} HTTP/1.1\r\nhost: localhost\r\nconnection: close\r\n\r\n");
    socket
        .write_all(request.as_bytes())
        .await
        .expect("write callback request");
    let mut response = String::new();
    socket
        .read_to_string(&mut response)
        .await
        .expect("read callback response");
    response
}

async fn mock_binary_server(
    body: Vec<u8>,
    content_type: &'static str,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len(),
        );
        socket
            .write_all(headers.as_bytes())
            .await
            .expect("write headers");
        socket.write_all(&body).await.expect("write response");
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

async fn mock_delayed_binary_server(
    chunks: Vec<Vec<u8>>,
    content_type: &'static str,
    delay: Duration,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        time::sleep,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        let headers = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\nconnection: close\r\n\r\n",
        );
        if socket.write_all(headers.as_bytes()).await.is_err() {
            return String::from_utf8_lossy(&request).into_owned();
        }
        for (index, chunk) in chunks.iter().enumerate() {
            if socket.write_all(chunk).await.is_err() {
                break;
            }
            if socket.flush().await.is_err() {
                break;
            }
            if index + 1 < chunks.len() {
                sleep(delay).await;
            }
        }
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

async fn mock_delayed_sse_server(
    chunks: Vec<&'static str>,
    delay: Duration,
) -> (String, tokio::task::JoinHandle<String>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
        time::sleep,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept request");
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        loop {
            let n = socket.read(&mut buf).await.expect("read request");
            if n == 0 {
                break;
            }
            request.extend_from_slice(&buf[..n]);
            if request_is_complete(&request) {
                break;
            }
        }
        let headers =
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\nconnection: close\r\n\r\n";
        if socket.write_all(headers.as_bytes()).await.is_err() {
            return String::from_utf8_lossy(&request).into_owned();
        }
        for (index, chunk) in chunks.iter().enumerate() {
            if socket.write_all(chunk.as_bytes()).await.is_err() {
                break;
            }
            if socket.flush().await.is_err() {
                break;
            }
            if index + 1 < chunks.len() {
                sleep(delay).await;
            }
        }
        String::from_utf8_lossy(&request).into_owned()
    });
    (format!("http://{addr}"), task)
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|value| value.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    request.len() >= header_end + 4 + content_length
}

fn aws_eventstream_body(events: Vec<Value>) -> Vec<u8> {
    events
        .into_iter()
        .flat_map(|event| aws_eventstream_frame(event.to_string().as_bytes()))
        .collect()
}

fn aws_eventstream_frame(payload: &[u8]) -> Vec<u8> {
    let total_len = 16 + payload.len();
    let mut frame = Vec::with_capacity(total_len);
    frame.extend_from_slice(&(total_len as u32).to_be_bytes());
    frame.extend_from_slice(&0_u32.to_be_bytes());
    frame.extend_from_slice(&0_u32.to_be_bytes());
    frame.extend_from_slice(payload);
    frame.extend_from_slice(&0_u32.to_be_bytes());
    frame
}
