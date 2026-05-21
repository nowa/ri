use crate::types::{Model, ModelCost, ThinkingLevel, Usage, UsageCost};
use parking_lot::RwLock;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

static MODEL_REGISTRY: std::sync::LazyLock<RwLock<BTreeMap<String, BTreeMap<String, Model>>>> =
    std::sync::LazyLock::new(|| RwLock::new(seed_models()));

pub const CLOUDFLARE_WORKERS_AI_BASE_URL: &str =
    "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1";
pub const CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat";
pub const CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL: &str =
    "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai";
pub const CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL: &str = "https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/anthropic";

pub fn get_model(provider: &str, model_id: &str) -> Option<Model> {
    MODEL_REGISTRY
        .read()
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
        .or_else(|| synthesize_known_provider_model(provider, model_id))
}

pub fn get_providers() -> Vec<String> {
    let mut providers: BTreeSet<String> = MODEL_REGISTRY.read().keys().cloned().collect();
    providers.extend(
        KNOWN_PROVIDERS
            .iter()
            .map(|provider| (*provider).to_owned()),
    );
    providers.into_iter().collect()
}

pub fn get_models(provider: &str) -> Vec<Model> {
    MODEL_REGISTRY
        .read()
        .get(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

pub fn register_model(model: Model) {
    MODEL_REGISTRY
        .write()
        .entry(model.provider.clone())
        .or_default()
        .insert(model.id.clone(), model);
}

pub fn clear_models() {
    *MODEL_REGISTRY.write() = BTreeMap::new();
}

pub fn reset_models() {
    *MODEL_REGISTRY.write() = seed_models();
}

pub fn calculate_cost(model: &Model, usage: &mut Usage) -> UsageCost {
    usage.cost.input = (model.cost.input / 1_000_000.0) * usage.input as f64;
    usage.cost.output = (model.cost.output / 1_000_000.0) * usage.output as f64;
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage.cost.clone()
}

pub fn get_supported_thinking_levels(model: &Model) -> Vec<ThinkingLevel> {
    if !model.reasoning {
        return vec![ThinkingLevel::Off];
    }

    ThinkingLevel::EXTENDED
        .into_iter()
        .filter(|level| match model.thinking_level_map.get(level) {
            Some(None) => false,
            Some(Some(_)) => true,
            None => *level != ThinkingLevel::XHigh,
        })
        .collect()
}

pub fn clamp_thinking_level(model: &Model, level: ThinkingLevel) -> ThinkingLevel {
    let available = get_supported_thinking_levels(model);
    if available.contains(&level) {
        return level;
    }

    let requested_index = ThinkingLevel::EXTENDED
        .iter()
        .position(|candidate| *candidate == level)
        .unwrap_or(0);

    for candidate in ThinkingLevel::EXTENDED.iter().skip(requested_index) {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    for candidate in ThinkingLevel::EXTENDED.iter().take(requested_index).rev() {
        if available.contains(candidate) {
            return *candidate;
        }
    }
    available.first().copied().unwrap_or(ThinkingLevel::Off)
}

pub fn models_are_equal(a: Option<&Model>, b: Option<&Model>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => a.id == b.id && a.provider == b.provider,
        _ => false,
    }
}

const KNOWN_PROVIDERS: &[&str] = &[
    "amazon-bedrock",
    "anthropic",
    "google",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
];

fn seed_models() -> BTreeMap<String, BTreeMap<String, Model>> {
    let mut registry = BTreeMap::new();
    for (provider, id) in COMMON_MODELS {
        let model = synthesize_model(provider, id);
        registry
            .entry((*provider).to_owned())
            .or_insert_with(BTreeMap::new)
            .insert((*id).to_owned(), model);
    }
    registry
}

const COMMON_MODELS: &[(&str, &str)] = &[
    ("anthropic", "claude-3-5-haiku-20241022"),
    ("anthropic", "claude-3-5-haiku-latest"),
    ("anthropic", "claude-3-5-sonnet-20240620"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
    ("anthropic", "claude-3-7-sonnet-20250219"),
    ("anthropic", "claude-3-haiku-20240307"),
    ("anthropic", "claude-3-opus-20240229"),
    ("anthropic", "claude-3-sonnet-20240229"),
    ("anthropic", "claude-haiku-4-5"),
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("anthropic", "claude-opus-4-0"),
    ("anthropic", "claude-opus-4-1"),
    ("anthropic", "claude-opus-4-1-20250805"),
    ("anthropic", "claude-opus-4-20250514"),
    ("anthropic", "claude-opus-4-5"),
    ("anthropic", "claude-opus-4-5-20251101"),
    ("anthropic", "claude-opus-4-6"),
    ("anthropic", "claude-opus-4-7"),
    ("anthropic", "claude-sonnet-4-0"),
    ("anthropic", "claude-sonnet-4-20250514"),
    ("anthropic", "claude-sonnet-4-5"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
    ("anthropic", "claude-sonnet-4-6"),
    (
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    (
        "amazon-bedrock",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    (
        "amazon-bedrock",
        "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ),
    (
        "amazon-bedrock",
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
    ),
    ("amazon-bedrock", "global.anthropic.claude-opus-4-6-v1"),
    ("amazon-bedrock", "us.anthropic.claude-opus-4-7"),
    ("openai", "gpt-4"),
    ("openai", "gpt-4-turbo"),
    ("openai", "gpt-4.1"),
    ("openai", "gpt-4.1-mini"),
    ("openai", "gpt-4.1-nano"),
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-2024-05-13"),
    ("openai", "gpt-4o-2024-08-06"),
    ("openai", "gpt-4o-2024-11-20"),
    ("openai", "gpt-4o-mini"),
    ("openai", "gpt-5"),
    ("openai", "gpt-5-chat-latest"),
    ("openai", "gpt-5-codex"),
    ("openai", "gpt-5-mini"),
    ("openai", "gpt-5-nano"),
    ("openai", "gpt-5-pro"),
    ("openai", "gpt-5.1"),
    ("openai", "gpt-5.1-chat-latest"),
    ("openai", "gpt-5.1-codex"),
    ("openai", "gpt-5.1-codex-max"),
    ("openai", "gpt-5.1-codex-mini"),
    ("openai", "gpt-5.2"),
    ("openai", "gpt-5.2-chat-latest"),
    ("openai", "gpt-5.2-codex"),
    ("openai", "gpt-5.2-pro"),
    ("openai", "gpt-5.3-chat-latest"),
    ("openai", "gpt-5.3-codex"),
    ("openai", "gpt-5.3-codex-spark"),
    ("openai", "gpt-5.4"),
    ("openai", "gpt-5.4-mini"),
    ("openai", "gpt-5.4-nano"),
    ("openai", "gpt-5.4-pro"),
    ("openai", "gpt-5.5"),
    ("openai", "gpt-5.5-pro"),
    ("openai", "o1"),
    ("openai", "o1-pro"),
    ("openai", "o3"),
    ("openai", "o3-deep-research"),
    ("openai", "o3-mini"),
    ("openai", "o3-pro"),
    ("openai", "o4-mini"),
    ("openai", "o4-mini-deep-research"),
    ("openai-codex", "gpt-5.2"),
    ("openai-codex", "gpt-5.3-codex"),
    ("openai-codex", "gpt-5.3-codex-spark"),
    ("openai-codex", "gpt-5.4"),
    ("openai-codex", "gpt-5.4-mini"),
    ("openai-codex", "gpt-5.5"),
    ("azure-openai-responses", "gpt-4"),
    ("azure-openai-responses", "gpt-4-turbo"),
    ("azure-openai-responses", "gpt-4.1"),
    ("azure-openai-responses", "gpt-4.1-mini"),
    ("azure-openai-responses", "gpt-4.1-nano"),
    ("azure-openai-responses", "gpt-4o"),
    ("azure-openai-responses", "gpt-4o-2024-05-13"),
    ("azure-openai-responses", "gpt-4o-2024-08-06"),
    ("azure-openai-responses", "gpt-4o-2024-11-20"),
    ("azure-openai-responses", "gpt-4o-mini"),
    ("azure-openai-responses", "gpt-5"),
    ("azure-openai-responses", "gpt-5-chat-latest"),
    ("azure-openai-responses", "gpt-5-codex"),
    ("azure-openai-responses", "gpt-5-mini"),
    ("azure-openai-responses", "gpt-5-nano"),
    ("azure-openai-responses", "gpt-5-pro"),
    ("azure-openai-responses", "gpt-5.1"),
    ("azure-openai-responses", "gpt-5.1-chat-latest"),
    ("azure-openai-responses", "gpt-5.1-codex"),
    ("azure-openai-responses", "gpt-5.1-codex-max"),
    ("azure-openai-responses", "gpt-5.1-codex-mini"),
    ("azure-openai-responses", "gpt-5.2"),
    ("azure-openai-responses", "gpt-5.2-chat-latest"),
    ("azure-openai-responses", "gpt-5.2-codex"),
    ("azure-openai-responses", "gpt-5.2-pro"),
    ("azure-openai-responses", "gpt-5.3-chat-latest"),
    ("azure-openai-responses", "gpt-5.3-codex"),
    ("azure-openai-responses", "gpt-5.3-codex-spark"),
    ("azure-openai-responses", "gpt-5.4"),
    ("azure-openai-responses", "gpt-5.4-mini"),
    ("azure-openai-responses", "gpt-5.4-nano"),
    ("azure-openai-responses", "gpt-5.4-pro"),
    ("azure-openai-responses", "gpt-5.5"),
    ("azure-openai-responses", "gpt-5.5-pro"),
    ("azure-openai-responses", "o1"),
    ("azure-openai-responses", "o1-pro"),
    ("azure-openai-responses", "o3"),
    ("azure-openai-responses", "o3-deep-research"),
    ("azure-openai-responses", "o3-mini"),
    ("azure-openai-responses", "o3-pro"),
    ("azure-openai-responses", "o4-mini"),
    ("azure-openai-responses", "o4-mini-deep-research"),
    ("github-copilot", "gpt-4o"),
    ("github-copilot", "gpt-5-mini"),
    ("github-copilot", "gpt-5.2-codex"),
    ("github-copilot", "gpt-5.3-codex"),
    ("github-copilot", "claude-sonnet-4.6"),
    ("google", "gemini-2.0-flash"),
    ("google", "gemini-2.5-flash"),
    ("google", "gemini-3-flash-preview"),
    ("google", "gemini-3.1-pro-preview"),
    ("google-vertex", "gemini-2.5-flash"),
    ("google-vertex", "gemini-3-flash-preview"),
    ("deepseek", "deepseek-v4-flash"),
    ("opencode", "big-pickle"),
    ("opencode", "claude-sonnet-4-5"),
    ("opencode", "gemini-3-flash"),
    ("opencode", "gpt-5.2-codex"),
    ("opencode", "kimi-k2.5"),
    ("opencode", "minimax-m2.5"),
    ("opencode-go", "deepseek-v4-flash"),
    ("opencode-go", "kimi-k2.5"),
    ("opencode-go", "kimi-k2.6"),
    ("opencode-go", "minimax-m2.5"),
    ("openrouter", "deepseek/deepseek-v4-flash"),
    ("openrouter", "deepseek/deepseek-v3.2"),
    ("openrouter", "deepseek/deepseek-chat"),
    ("openrouter", "deepseek/deepseek-r1"),
    ("openrouter", "anthropic/claude-opus-4.6"),
    ("openrouter", "anthropic/claude-sonnet-4"),
    ("openrouter", "google/gemini-2.0-flash-001"),
    ("openrouter", "google/gemini-2.5-flash"),
    ("openrouter", "meta-llama/llama-4-scout"),
    ("openrouter", "mistralai/mistral-small-3.2-24b-instruct"),
    ("openrouter", "mistralai/mistral-large-2512"),
    ("openrouter", "openai/gpt-5.2-codex"),
    ("openrouter", "qwen/qwen3.5-plus-02-15"),
    ("openrouter", "z-ai/glm-4.5v"),
    ("vercel-ai-gateway", "google/gemini-2.5-flash"),
    ("vercel-ai-gateway", "anthropic/claude-opus-4.5"),
    ("vercel-ai-gateway", "openai/gpt-5.1-codex-max"),
    ("groq", "llama-3.3-70b-versatile"),
    ("groq", "openai/gpt-oss-20b"),
    ("groq", "openai/gpt-oss-120b"),
    ("groq", "qwen/qwen3-32b"),
    ("cerebras", "gpt-oss-120b"),
    ("cerebras", "qwen-3-235b-a22b-instruct-2507"),
    ("xai", "grok-3"),
    ("xai", "grok-3-fast"),
    ("xai", "grok-code-fast-1"),
    ("mistral", "devstral-medium-latest"),
    ("mistral", "magistral-medium-latest"),
    ("mistral", "mistral-medium-3.5"),
    ("mistral", "mistral-small-2603"),
    ("mistral", "pixtral-12b"),
    ("minimax", "MiniMax-M2.7"),
    ("kimi-coding", "kimi-for-coding"),
    ("kimi-coding", "kimi-k2-thinking"),
    ("huggingface", "moonshotai/Kimi-K2.5"),
    ("fireworks", "accounts/fireworks/models/kimi-k2p6"),
    ("fireworks", "accounts/fireworks/routers/kimi-k2p5-turbo"),
    ("together", "moonshotai/Kimi-K2.6"),
    ("together", "MiniMaxAI/MiniMax-M2.7"),
    ("together", "deepseek-ai/DeepSeek-V4-Pro"),
    ("together", "openai/gpt-oss-120b"),
    ("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6"),
    (
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    ),
    ("cloudflare-ai-gateway", "gpt-5.1"),
    ("cloudflare-ai-gateway", "gpt-5.1-codex"),
    ("cloudflare-ai-gateway", "gpt-5.2"),
    ("cloudflare-ai-gateway", "gpt-5.2-codex"),
    ("cloudflare-ai-gateway", "gpt-5.3-codex"),
    ("cloudflare-ai-gateway", "gpt-5.4"),
    ("cloudflare-ai-gateway", "gpt-5.5"),
    ("cloudflare-ai-gateway", "claude-sonnet-4-5"),
    ("zai", "glm-4.5-air"),
    ("zai", "glm-4.7"),
    ("zai", "glm-5-turbo"),
    ("zai", "glm-5.1"),
    ("xiaomi", "mimo-v2.5-pro"),
    ("xiaomi-token-plan-cn", "mimo-v2.5-pro"),
    ("xiaomi-token-plan-ams", "mimo-v2.5-pro"),
    ("xiaomi-token-plan-sgp", "mimo-v2.5-pro"),
];

fn synthesize_known_provider_model(provider: &str, model_id: &str) -> Option<Model> {
    KNOWN_PROVIDERS
        .contains(&provider)
        .then(|| synthesize_model(provider, model_id))
}

fn synthesize_model(provider: &str, model_id: &str) -> Model {
    let mut model = Model {
        id: model_id.to_owned(),
        name: model_id.to_owned(),
        api: api_for_provider(provider, model_id).to_owned(),
        provider: provider.to_owned(),
        base_url: base_url_for_model(provider, model_id),
        reasoning: reasoning_for_model(provider, model_id),
        thinking_level_map: BTreeMap::new(),
        input: vec![crate::types::InputKind::Text],
        cost: ModelCost::default(),
        context_window: context_window_for_model(provider, model_id),
        max_tokens: 16_384,
        headers: BTreeMap::new(),
        compat: compat_for_model(provider, model_id),
    };

    if supports_images(provider, model_id) {
        model.input.push(crate::types::InputKind::Image);
    }

    if is_deepseek_v4_flash(provider, model_id) {
        model.reasoning = true;
        model
            .thinking_level_map
            .insert(ThinkingLevel::Minimal, None);
        model.thinking_level_map.insert(ThinkingLevel::Low, None);
        model.thinking_level_map.insert(ThinkingLevel::Medium, None);
        model.thinking_level_map.insert(
            ThinkingLevel::XHigh,
            Some(
                xhigh_effort_for_model(provider, model_id)
                    .unwrap_or("max")
                    .to_owned(),
            ),
        );
    }

    if let Some(effort) = xhigh_effort_for_model(provider, model_id) {
        model.reasoning = true;
        model
            .thinking_level_map
            .insert(ThinkingLevel::XHigh, Some(effort.to_owned()));
    }

    apply_known_model_overrides(&mut model);

    model
}

fn apply_known_model_overrides(model: &mut Model) {
    if model.provider == "github-copilot" {
        model.base_url = "https://api.individual.githubcopilot.com".to_owned();
        model.headers = github_copilot_headers();
        model.cost = ModelCost::default();
        if model.id == "claude-sonnet-4.6" {
            model.context_window = 1_000_000;
            model.max_tokens = 32_000;
            model.reasoning = true;
            ensure_image_input(model);
        }
    }

    if model.provider == "cloudflare-workers-ai" && model.id == "@cf/moonshotai/kimi-k2.6" {
        model.reasoning = true;
        ensure_image_input(model);
        model.context_window = 256_000;
        model.max_tokens = 256_000;
        model.cost = ModelCost {
            input: 0.95,
            output: 4.0,
            cache_read: 0.16,
            cache_write: 0.0,
        };
    }

    if model.provider == "amazon-bedrock" && model.id.contains("opus-4-6") {
        model.reasoning = true;
        ensure_image_input(model);
        model
            .thinking_level_map
            .insert(ThinkingLevel::XHigh, Some("max".to_owned()));
        model.context_window = 1_000_000;
        model.max_tokens = 128_000;
    }

    if model.provider == "cloudflare-ai-gateway" {
        if model.id == "workers-ai/@cf/moonshotai/kimi-k2.6" {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 256_000;
            model.max_tokens = 256_000;
            model.cost = ModelCost {
                input: 0.95,
                output: 4.0,
                cache_read: 0.16,
                cache_write: 0.0,
            };
        } else if matches!(
            model.id.as_str(),
            "gpt-5.1"
                | "gpt-5.1-codex"
                | "gpt-5.2"
                | "gpt-5.2-codex"
                | "gpt-5.3-codex"
                | "gpt-5.4"
                | "gpt-5.5"
        ) {
            model.reasoning = true;
            model.thinking_level_map.clear();
            model.thinking_level_map.insert(ThinkingLevel::Off, None);
            if !matches!(model.id.as_str(), "gpt-5.1" | "gpt-5.1-codex") {
                model
                    .thinking_level_map
                    .insert(ThinkingLevel::XHigh, Some("xhigh".to_owned()));
            }
            ensure_image_input(model);
            model.context_window = if matches!(model.id.as_str(), "gpt-5.4" | "gpt-5.5") {
                1_050_000
            } else {
                400_000
            };
            model.max_tokens = 128_000;
            model.cost = match model.id.as_str() {
                "gpt-5.1" => ModelCost {
                    input: 1.25,
                    output: 10.0,
                    cache_read: 0.13,
                    cache_write: 0.0,
                },
                "gpt-5.1-codex" => ModelCost {
                    input: 1.25,
                    output: 10.0,
                    cache_read: 0.125,
                    cache_write: 0.0,
                },
                "gpt-5.4" => ModelCost {
                    input: 2.5,
                    output: 15.0,
                    cache_read: 0.25,
                    cache_write: 0.0,
                },
                "gpt-5.5" => ModelCost {
                    input: 5.0,
                    output: 30.0,
                    cache_read: 0.5,
                    cache_write: 0.0,
                },
                _ => ModelCost {
                    input: 1.75,
                    output: 14.0,
                    cache_read: 0.175,
                    cache_write: 0.0,
                },
            };
        } else if model.id == "claude-sonnet-4-5" {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 200_000;
            model.max_tokens = 64_000;
            model.cost = ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            };
        }
    }

    if model.provider == "opencode" {
        apply_opencode_model_overrides(model);
    }
    if model.provider == "opencode-go" {
        apply_opencode_go_model_overrides(model);
    }

    if model.provider == "kimi-coding" {
        model
            .headers
            .insert("User-Agent".to_owned(), "KimiCLI/1.5".to_owned());
        model.reasoning = true;
        model.context_window = 262_144;
        model.max_tokens = 32_768;
        model.cost = ModelCost::default();
        if model.id == "kimi-for-coding" {
            ensure_image_input(model);
        }
    }

    if matches!(model.provider.as_str(), "minimax" | "minimax-cn") {
        model.reasoning = true;
        model.context_window = 204_800;
        model.max_tokens = 131_072;
    }

    if model.provider == "vercel-ai-gateway" {
        model.reasoning = true;
        ensure_image_input(model);
    }

    if model.provider == "anthropic" && apply_anthropic_generated_metadata(model) {
        return;
    }
    if model.provider == "openai" && apply_openai_gpt4_generated_metadata(model) {
        return;
    }
    if model.provider == "azure-openai-responses"
        && apply_azure_openai_gpt4_generated_metadata(model)
    {
        return;
    }
    if model.provider == "openai" && apply_openai_gpt5_generated_metadata(model) {
        return;
    }
    if model.provider == "azure-openai-responses"
        && apply_azure_openai_gpt5_generated_metadata(model)
    {
        return;
    }
    if model.provider == "openai" && apply_openai_o_series_generated_metadata(model) {
        return;
    }
    if model.provider == "azure-openai-responses"
        && apply_azure_openai_o_series_generated_metadata(model)
    {
        return;
    }

    match (model.provider.as_str(), model.id.as_str()) {
        ("mistral", "mistral-small-2603") | ("mistral", "mistral-small-latest") => {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 256_000;
            model.max_tokens = 256_000;
            model.cost = ModelCost {
                input: 0.15,
                output: 0.6,
                cache_read: 0.0,
                cache_write: 0.0,
            };
        }
        ("mistral", "mistral-medium-2604")
        | ("mistral", "mistral-medium-3.5")
        | ("mistral", "mistral-medium-latest") => {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 262_144;
            model.max_tokens = 262_144;
            model.cost = ModelCost {
                input: 1.5,
                output: 7.5,
                cache_read: 0.0,
                cache_write: 0.0,
            };
        }
        ("fireworks", "accounts/fireworks/models/kimi-k2p6") => {
            model.api = "anthropic-messages".to_owned();
            model.base_url = "https://api.fireworks.ai/inference".to_owned();
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 262_000;
            model.max_tokens = 262_000;
            model.cost = ModelCost {
                input: 0.95,
                output: 4.0,
                cache_read: 0.16,
                cache_write: 0.0,
            };
            model.compat = Some(json!({
                "sendSessionAffinityHeaders": true,
                "supportsEagerToolInputStreaming": false,
                "supportsCacheControlOnTools": false,
                "supportsLongCacheRetention": false,
            }));
        }
        ("fireworks", "accounts/fireworks/routers/kimi-k2p5-turbo") => {
            model.api = "anthropic-messages".to_owned();
            model.base_url = "https://api.fireworks.ai/inference".to_owned();
            model.reasoning = true;
            ensure_image_input(model);
        }
        ("groq", "openai/gpt-oss-20b" | "openai/gpt-oss-120b") => {
            model.reasoning = true;
        }
        ("groq", "qwen/qwen3-32b") => {
            model.reasoning = true;
            model.thinking_level_map.clear();
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, None);
            model.thinking_level_map.insert(ThinkingLevel::Low, None);
            model.thinking_level_map.insert(ThinkingLevel::Medium, None);
            model
                .thinking_level_map
                .insert(ThinkingLevel::High, Some("default".to_owned()));
        }
        ("openrouter", "qwen/qwen3.5-plus-02-15") => {
            model.reasoning = true;
            ensure_image_input(model);
        }
        ("together", "moonshotai/Kimi-K2.6") => {
            model.base_url = "https://api.together.ai/v1".to_owned();
            model.reasoning = true;
            model.thinking_level_map.clear();
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, None);
            model.thinking_level_map.insert(ThinkingLevel::Low, None);
            model.thinking_level_map.insert(ThinkingLevel::Medium, None);
            ensure_image_input(model);
            model.context_window = 262_144;
            model.max_tokens = 131_000;
            model.cost = ModelCost {
                input: 1.2,
                output: 4.5,
                cache_read: 0.2,
                cache_write: 0.0,
            };
            model.compat = Some(json!({
                "supportsStore": false,
                "supportsDeveloperRole": false,
                "supportsReasoningEffort": false,
                "maxTokensField": "max_tokens",
                "thinkingFormat": "together",
                "supportsStrictMode": false,
                "supportsLongCacheRetention": false,
            }));
        }
        ("together", "openai/gpt-oss-120b") => {
            model.base_url = "https://api.together.ai/v1".to_owned();
            model.reasoning = true;
            model.thinking_level_map.clear();
            model.thinking_level_map.insert(ThinkingLevel::Off, None);
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, None);
            model.compat = Some(json!({
                "supportsReasoningEffort": true,
                "thinkingFormat": "openai",
            }));
        }
        ("together", "deepseek-ai/DeepSeek-V4-Pro") => {
            model.base_url = "https://api.together.ai/v1".to_owned();
            model.reasoning = true;
            model.thinking_level_map.clear();
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, None);
            model.thinking_level_map.insert(ThinkingLevel::Low, None);
            model.thinking_level_map.insert(ThinkingLevel::Medium, None);
            model
                .thinking_level_map
                .insert(ThinkingLevel::High, Some("high".to_owned()));
            model.thinking_level_map.insert(ThinkingLevel::XHigh, None);
            model.compat = Some(json!({
                "supportsReasoningEffort": true,
                "thinkingFormat": "together",
            }));
        }
        ("together", "MiniMaxAI/MiniMax-M2.7") => {
            model.base_url = "https://api.together.ai/v1".to_owned();
            model.reasoning = true;
            model.thinking_level_map.clear();
            model.thinking_level_map.insert(ThinkingLevel::Off, None);
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, None);
            model.thinking_level_map.insert(ThinkingLevel::Low, None);
            model.thinking_level_map.insert(ThinkingLevel::Medium, None);
            model.compat = Some(json!({
                "supportsReasoningEffort": false,
            }));
        }
        ("together", _) => {
            model.base_url = "https://api.together.ai/v1".to_owned();
        }
        (
            "openai-codex",
            "gpt-5.2"
            | "gpt-5.3-codex"
            | "gpt-5.3-codex-spark"
            | "gpt-5.4"
            | "gpt-5.4-mini"
            | "gpt-5.5",
        ) => {
            model.base_url = "https://chatgpt.com/backend-api".to_owned();
            model.reasoning = true;
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, Some("low".to_owned()));
            model
                .thinking_level_map
                .insert(ThinkingLevel::XHigh, Some("xhigh".to_owned()));
            model.context_window = 272_000;
            model.max_tokens = 128_000;
            model.cost = match model.id.as_str() {
                "gpt-5.4" => ModelCost {
                    input: 2.5,
                    output: 15.0,
                    cache_read: 0.25,
                    cache_write: 0.0,
                },
                "gpt-5.4-mini" => ModelCost {
                    input: 0.75,
                    output: 4.5,
                    cache_read: 0.075,
                    cache_write: 0.0,
                },
                "gpt-5.5" => ModelCost {
                    input: 5.0,
                    output: 30.0,
                    cache_read: 0.5,
                    cache_write: 0.0,
                },
                _ => ModelCost {
                    input: 1.75,
                    output: 14.0,
                    cache_read: 0.175,
                    cache_write: 0.0,
                },
            };
            if model.id == "gpt-5.3-codex-spark" {
                model.input = vec![crate::types::InputKind::Text];
            } else {
                ensure_image_input(model);
            }
        }
        _ => {}
    }
}

struct AnthropicGeneratedMetadata {
    reasoning: bool,
    xhigh_effort: Option<&'static str>,
    context_window: u64,
    max_tokens: u64,
    cost: ModelCost,
}

fn anthropic_generated_metadata(model_id: &str) -> Option<AnthropicGeneratedMetadata> {
    match model_id {
        "claude-3-haiku-20240307" => Some(AnthropicGeneratedMetadata {
            reasoning: false,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 4_096,
            cost: ModelCost {
                input: 0.25,
                output: 1.25,
                cache_read: 0.03,
                cache_write: 0.3,
            },
        }),
        "claude-3-opus-20240229" => Some(AnthropicGeneratedMetadata {
            reasoning: false,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 4_096,
            cost: ModelCost {
                input: 15.0,
                output: 75.0,
                cache_read: 1.5,
                cache_write: 18.75,
            },
        }),
        "claude-3-sonnet-20240229" => Some(AnthropicGeneratedMetadata {
            reasoning: false,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 4_096,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 0.3,
            },
        }),
        "claude-3-5-haiku-20241022" | "claude-3-5-haiku-latest" => {
            Some(AnthropicGeneratedMetadata {
                reasoning: false,
                xhigh_effort: None,
                context_window: 200_000,
                max_tokens: 8_192,
                cost: ModelCost {
                    input: 0.8,
                    output: 4.0,
                    cache_read: 0.08,
                    cache_write: 1.0,
                },
            })
        }
        "claude-3-5-sonnet-20240620" | "claude-3-5-sonnet-20241022" => {
            Some(AnthropicGeneratedMetadata {
                reasoning: false,
                xhigh_effort: None,
                context_window: 200_000,
                max_tokens: 8_192,
                cost: ModelCost {
                    input: 3.0,
                    output: 15.0,
                    cache_read: 0.3,
                    cache_write: 3.75,
                },
            })
        }
        "claude-3-7-sonnet-20250219" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 64_000,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
        }),
        "claude-haiku-4-5" | "claude-haiku-4-5-20251001" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 64_000,
            cost: ModelCost {
                input: 1.0,
                output: 5.0,
                cache_read: 0.1,
                cache_write: 1.25,
            },
        }),
        "claude-opus-4-0"
        | "claude-opus-4-1"
        | "claude-opus-4-1-20250805"
        | "claude-opus-4-20250514" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 32_000,
            cost: ModelCost {
                input: 15.0,
                output: 75.0,
                cache_read: 1.5,
                cache_write: 18.75,
            },
        }),
        "claude-opus-4-5" | "claude-opus-4-5-20251101" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 64_000,
            cost: ModelCost {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
        }),
        "claude-opus-4-6" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: Some("max"),
            context_window: 1_000_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
        }),
        "claude-opus-4-7" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: Some("xhigh"),
            context_window: 1_000_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 5.0,
                output: 25.0,
                cache_read: 0.5,
                cache_write: 6.25,
            },
        }),
        "claude-sonnet-4-0"
        | "claude-sonnet-4-20250514"
        | "claude-sonnet-4-5"
        | "claude-sonnet-4-5-20250929" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: None,
            context_window: 200_000,
            max_tokens: 64_000,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
        }),
        "claude-sonnet-4-6" => Some(AnthropicGeneratedMetadata {
            reasoning: true,
            xhigh_effort: None,
            context_window: 1_000_000,
            max_tokens: 64_000,
            cost: ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            },
        }),
        _ => None,
    }
}

struct Gpt4GeneratedMetadata {
    text_only: bool,
    context_window: u64,
    max_tokens: u64,
    cost: ModelCost,
}

fn gpt4_generated_metadata(model_id: &str) -> Option<Gpt4GeneratedMetadata> {
    match model_id {
        "gpt-4" => Some(Gpt4GeneratedMetadata {
            text_only: true,
            context_window: 8_192,
            max_tokens: 8_192,
            cost: ModelCost {
                input: 30.0,
                output: 60.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "gpt-4-turbo" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 128_000,
            max_tokens: 4_096,
            cost: ModelCost {
                input: 10.0,
                output: 30.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "gpt-4.1" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 1_047_576,
            max_tokens: 32_768,
            cost: ModelCost {
                input: 2.0,
                output: 8.0,
                cache_read: 0.5,
                cache_write: 0.0,
            },
        }),
        "gpt-4.1-mini" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 1_047_576,
            max_tokens: 32_768,
            cost: ModelCost {
                input: 0.4,
                output: 1.6,
                cache_read: 0.1,
                cache_write: 0.0,
            },
        }),
        "gpt-4.1-nano" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 1_047_576,
            max_tokens: 32_768,
            cost: ModelCost {
                input: 0.1,
                output: 0.4,
                cache_read: 0.03,
                cache_write: 0.0,
            },
        }),
        "gpt-4o" | "gpt-4o-2024-08-06" | "gpt-4o-2024-11-20" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 128_000,
            max_tokens: 16_384,
            cost: ModelCost {
                input: 2.5,
                output: 10.0,
                cache_read: 1.25,
                cache_write: 0.0,
            },
        }),
        "gpt-4o-2024-05-13" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 128_000,
            max_tokens: 4_096,
            cost: ModelCost {
                input: 5.0,
                output: 15.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "gpt-4o-mini" => Some(Gpt4GeneratedMetadata {
            text_only: false,
            context_window: 128_000,
            max_tokens: 16_384,
            cost: ModelCost {
                input: 0.15,
                output: 0.6,
                cache_read: 0.08,
                cache_write: 0.0,
            },
        }),
        _ => None,
    }
}

struct Gpt5GeneratedMetadata {
    reasoning: bool,
    openai_off_effort: Option<&'static str>,
    supports_xhigh: bool,
    context_window: u64,
    max_tokens: u64,
    cost: ModelCost,
}

fn gpt5_generated_metadata(model_id: &str) -> Option<Gpt5GeneratedMetadata> {
    match model_id {
        "gpt-5" | "gpt-5-codex" | "gpt-5.1-codex" | "gpt-5.1-codex-max" => {
            Some(Gpt5GeneratedMetadata {
                reasoning: true,
                openai_off_effort: None,
                supports_xhigh: false,
                context_window: 400_000,
                max_tokens: 128_000,
                cost: ModelCost {
                    input: 1.25,
                    output: 10.0,
                    cache_read: 0.125,
                    cache_write: 0.0,
                },
            })
        }
        "gpt-5-chat-latest" => Some(Gpt5GeneratedMetadata {
            reasoning: false,
            openai_off_effort: None,
            supports_xhigh: false,
            context_window: 128_000,
            max_tokens: 16_384,
            cost: ModelCost {
                input: 1.25,
                output: 10.0,
                cache_read: 0.125,
                cache_write: 0.0,
            },
        }),
        "gpt-5-mini" | "gpt-5.1-codex-mini" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: false,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 0.25,
                output: 2.0,
                cache_read: 0.025,
                cache_write: 0.0,
            },
        }),
        "gpt-5-nano" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: false,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 0.05,
                output: 0.4,
                cache_read: 0.005,
                cache_write: 0.0,
            },
        }),
        "gpt-5-pro" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: false,
            context_window: 400_000,
            max_tokens: 272_000,
            cost: ModelCost {
                input: 15.0,
                output: 120.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "gpt-5.1" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: false,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 1.25,
                output: 10.0,
                cache_read: 0.13,
                cache_write: 0.0,
            },
        }),
        "gpt-5.1-chat-latest" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: false,
            context_window: 128_000,
            max_tokens: 16_384,
            cost: ModelCost {
                input: 1.25,
                output: 10.0,
                cache_read: 0.125,
                cache_write: 0.0,
            },
        }),
        "gpt-5.2" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: true,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 1.75,
                output: 14.0,
                cache_read: 0.175,
                cache_write: 0.0,
            },
        }),
        "gpt-5.2-chat-latest" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: true,
            context_window: 128_000,
            max_tokens: 16_384,
            cost: ModelCost {
                input: 1.75,
                output: 14.0,
                cache_read: 0.175,
                cache_write: 0.0,
            },
        }),
        "gpt-5.2-codex" | "gpt-5.3-codex" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: true,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 1.75,
                output: 14.0,
                cache_read: 0.175,
                cache_write: 0.0,
            },
        }),
        "gpt-5.2-pro" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: true,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 21.0,
                output: 168.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "gpt-5.3-chat-latest" => Some(Gpt5GeneratedMetadata {
            reasoning: false,
            openai_off_effort: None,
            supports_xhigh: true,
            context_window: 128_000,
            max_tokens: 16_384,
            cost: ModelCost {
                input: 1.75,
                output: 14.0,
                cache_read: 0.175,
                cache_write: 0.0,
            },
        }),
        "gpt-5.3-codex-spark" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: true,
            context_window: 128_000,
            max_tokens: 32_000,
            cost: ModelCost {
                input: 1.75,
                output: 14.0,
                cache_read: 0.175,
                cache_write: 0.0,
            },
        }),
        "gpt-5.4" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: true,
            context_window: 272_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 2.5,
                output: 15.0,
                cache_read: 0.25,
                cache_write: 0.0,
            },
        }),
        "gpt-5.4-mini" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: true,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 0.75,
                output: 4.5,
                cache_read: 0.075,
                cache_write: 0.0,
            },
        }),
        "gpt-5.4-nano" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: true,
            context_window: 400_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 0.2,
                output: 1.25,
                cache_read: 0.02,
                cache_write: 0.0,
            },
        }),
        "gpt-5.4-pro" | "gpt-5.5-pro" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: None,
            supports_xhigh: true,
            context_window: 1_050_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 30.0,
                output: 180.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "gpt-5.5" => Some(Gpt5GeneratedMetadata {
            reasoning: true,
            openai_off_effort: Some("none"),
            supports_xhigh: true,
            context_window: 272_000,
            max_tokens: 128_000,
            cost: ModelCost {
                input: 5.0,
                output: 30.0,
                cache_read: 0.5,
                cache_write: 0.0,
            },
        }),
        _ => None,
    }
}

struct OSeriesGeneratedMetadata {
    text_only: bool,
    cost: ModelCost,
}

fn o_series_generated_metadata(model_id: &str) -> Option<OSeriesGeneratedMetadata> {
    match model_id {
        "o1" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 15.0,
                output: 60.0,
                cache_read: 7.5,
                cache_write: 0.0,
            },
        }),
        "o1-pro" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 150.0,
                output: 600.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "o3" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 2.0,
                output: 8.0,
                cache_read: 0.5,
                cache_write: 0.0,
            },
        }),
        "o3-deep-research" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 10.0,
                output: 40.0,
                cache_read: 2.5,
                cache_write: 0.0,
            },
        }),
        "o3-mini" => Some(OSeriesGeneratedMetadata {
            text_only: true,
            cost: ModelCost {
                input: 1.1,
                output: 4.4,
                cache_read: 0.55,
                cache_write: 0.0,
            },
        }),
        "o3-pro" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 20.0,
                output: 80.0,
                cache_read: 0.0,
                cache_write: 0.0,
            },
        }),
        "o4-mini" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 1.1,
                output: 4.4,
                cache_read: 0.28,
                cache_write: 0.0,
            },
        }),
        "o4-mini-deep-research" => Some(OSeriesGeneratedMetadata {
            text_only: false,
            cost: ModelCost {
                input: 2.0,
                output: 8.0,
                cache_read: 0.5,
                cache_write: 0.0,
            },
        }),
        _ => None,
    }
}

fn apply_anthropic_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = anthropic_generated_metadata(model.id.as_str()) else {
        return false;
    };
    model.api = "anthropic-messages".to_owned();
    model.base_url = "https://api.anthropic.com".to_owned();
    model.reasoning = metadata.reasoning;
    model.thinking_level_map.clear();
    if let Some(effort) = metadata.xhigh_effort {
        model
            .thinking_level_map
            .insert(ThinkingLevel::XHigh, Some(effort.to_owned()));
    }
    model.input = vec![crate::types::InputKind::Text];
    ensure_image_input(model);
    model.context_window = metadata.context_window;
    model.max_tokens = metadata.max_tokens;
    model.cost = metadata.cost;
    true
}

fn apply_openai_gpt4_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = gpt4_generated_metadata(model.id.as_str()) else {
        return false;
    };
    model.api = "openai-responses".to_owned();
    apply_gpt4_generated_metadata(model, metadata);
    true
}

fn apply_azure_openai_gpt4_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = gpt4_generated_metadata(model.id.as_str()) else {
        return false;
    };
    model.base_url.clear();
    apply_gpt4_generated_metadata(model, metadata);
    true
}

fn apply_gpt4_generated_metadata(model: &mut Model, metadata: Gpt4GeneratedMetadata) {
    model.reasoning = false;
    model.thinking_level_map.clear();
    model.input = vec![crate::types::InputKind::Text];
    if !metadata.text_only {
        ensure_image_input(model);
    }
    model.context_window = metadata.context_window;
    model.max_tokens = metadata.max_tokens;
    model.cost = metadata.cost;
}

fn apply_openai_gpt5_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = gpt5_generated_metadata(model.id.as_str()) else {
        return false;
    };
    apply_gpt5_generated_metadata(model, &metadata, metadata.openai_off_effort);
    true
}

fn apply_azure_openai_gpt5_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = gpt5_generated_metadata(model.id.as_str()) else {
        return false;
    };
    model.base_url.clear();
    apply_gpt5_generated_metadata(model, &metadata, None);
    true
}

fn apply_openai_o_series_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = o_series_generated_metadata(model.id.as_str()) else {
        return false;
    };
    model.api = "openai-responses".to_owned();
    apply_o_series_generated_metadata(model, metadata);
    true
}

fn apply_azure_openai_o_series_generated_metadata(model: &mut Model) -> bool {
    let Some(metadata) = o_series_generated_metadata(model.id.as_str()) else {
        return false;
    };
    model.base_url.clear();
    apply_o_series_generated_metadata(model, metadata);
    true
}

fn apply_o_series_generated_metadata(model: &mut Model, metadata: OSeriesGeneratedMetadata) {
    model.reasoning = true;
    model.thinking_level_map.clear();
    model.input = vec![crate::types::InputKind::Text];
    if !metadata.text_only {
        ensure_image_input(model);
    }
    model.context_window = 200_000;
    model.max_tokens = 100_000;
    model.cost = metadata.cost;
}

fn apply_gpt5_generated_metadata(
    model: &mut Model,
    metadata: &Gpt5GeneratedMetadata,
    off_effort: Option<&str>,
) {
    model.reasoning = metadata.reasoning;
    model.thinking_level_map.clear();
    model
        .thinking_level_map
        .insert(ThinkingLevel::Off, off_effort.map(str::to_owned));
    if metadata.supports_xhigh {
        model
            .thinking_level_map
            .insert(ThinkingLevel::XHigh, Some("xhigh".to_owned()));
    }
    ensure_image_input(model);
    model.context_window = metadata.context_window;
    model.max_tokens = metadata.max_tokens;
    model.cost = metadata.cost.clone();
}

fn github_copilot_headers() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "User-Agent".to_owned(),
            "GitHubCopilotChat/0.35.0".to_owned(),
        ),
        ("Editor-Version".to_owned(), "vscode/1.107.0".to_owned()),
        (
            "Editor-Plugin-Version".to_owned(),
            "copilot-chat/0.35.0".to_owned(),
        ),
        (
            "Copilot-Integration-Id".to_owned(),
            "vscode-chat".to_owned(),
        ),
    ])
}

fn ensure_image_input(model: &mut Model) {
    if !model.input.contains(&crate::types::InputKind::Image) {
        model.input.push(crate::types::InputKind::Image);
    }
}

fn apply_opencode_model_overrides(model: &mut Model) {
    match model.id.as_str() {
        "big-pickle" => {
            model.reasoning = true;
            model.context_window = 200_000;
            model.max_tokens = 128_000;
        }
        "claude-sonnet-4-5" => {
            model.api = "anthropic-messages".to_owned();
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 200_000;
            model.max_tokens = 64_000;
            model.cost = ModelCost {
                input: 3.0,
                output: 15.0,
                cache_read: 0.3,
                cache_write: 3.75,
            };
        }
        "gemini-3-flash" => {
            model.api = "google-generative-ai".to_owned();
            model.reasoning = true;
            model.thinking_level_map.clear();
            model.thinking_level_map.insert(ThinkingLevel::Off, None);
            ensure_image_input(model);
            model.context_window = 1_048_576;
            model.max_tokens = 65_536;
            model.cost = ModelCost {
                input: 0.5,
                output: 3.0,
                cache_read: 0.05,
                cache_write: 0.0,
            };
        }
        "gpt-5.2-codex" => {
            model.api = "openai-responses".to_owned();
            model.reasoning = true;
            model.thinking_level_map.clear();
            model.thinking_level_map.insert(ThinkingLevel::Off, None);
            model
                .thinking_level_map
                .insert(ThinkingLevel::XHigh, Some("xhigh".to_owned()));
            ensure_image_input(model);
            model.context_window = 400_000;
            model.max_tokens = 128_000;
            model.cost = ModelCost {
                input: 1.75,
                output: 14.0,
                cache_read: 0.175,
                cache_write: 0.0,
            };
        }
        "kimi-k2.5" => {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 262_144;
            model.max_tokens = 65_536;
            model.cost = ModelCost {
                input: 0.6,
                output: 3.0,
                cache_read: 0.08,
                cache_write: 0.0,
            };
        }
        "minimax-m2.5" => {
            model.reasoning = true;
            model.context_window = 204_800;
            model.max_tokens = 131_072;
            model.cost = ModelCost {
                input: 0.3,
                output: 1.2,
                cache_read: 0.06,
                cache_write: 0.0,
            };
        }
        _ => {}
    }
}

fn apply_opencode_go_model_overrides(model: &mut Model) {
    match model.id.as_str() {
        "deepseek-v4-flash" => {
            model.reasoning = true;
            model.thinking_level_map.clear();
            model
                .thinking_level_map
                .insert(ThinkingLevel::Minimal, None);
            model.thinking_level_map.insert(ThinkingLevel::Low, None);
            model.thinking_level_map.insert(ThinkingLevel::Medium, None);
            model
                .thinking_level_map
                .insert(ThinkingLevel::High, Some("high".to_owned()));
            model
                .thinking_level_map
                .insert(ThinkingLevel::XHigh, Some("max".to_owned()));
            model.context_window = 1_000_000;
            model.max_tokens = 384_000;
            model.cost = ModelCost {
                input: 0.14,
                output: 0.28,
                cache_read: 0.0028,
                cache_write: 0.0,
            };
            model.compat = Some(json!({
                "requiresReasoningContentOnAssistantMessages": true,
                "thinkingFormat": "deepseek",
            }));
        }
        "kimi-k2.5" => {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 262_144;
            model.max_tokens = 65_536;
            model.cost = ModelCost {
                input: 0.6,
                output: 3.0,
                cache_read: 0.1,
                cache_write: 0.0,
            };
        }
        "kimi-k2.6" => {
            model.reasoning = true;
            ensure_image_input(model);
            model.context_window = 262_144;
            model.max_tokens = 65_536;
            model.cost = ModelCost {
                input: 0.95,
                output: 4.0,
                cache_read: 0.16,
                cache_write: 0.0,
            };
        }
        "minimax-m2.5" => {
            model.api = "anthropic-messages".to_owned();
            model.reasoning = true;
            model.context_window = 204_800;
            model.max_tokens = 65_536;
            model.cost = ModelCost {
                input: 0.3,
                output: 1.2,
                cache_read: 0.03,
                cache_write: 0.0,
            };
        }
        _ => {}
    }
}

fn base_url_for_model(provider: &str, model_id: &str) -> String {
    if provider == "amazon-bedrock" {
        return crate::bedrock::bedrock_base_url_for_model(model_id).to_owned();
    }
    if provider == "opencode" {
        if model_id.contains("claude") {
            return "https://opencode.ai/zen".to_owned();
        }
        return "https://opencode.ai/zen/v1".to_owned();
    }
    if provider == "opencode-go" {
        if model_id == "minimax-m2.5" {
            return "https://opencode.ai/zen/go".to_owned();
        }
        return "https://opencode.ai/zen/go/v1".to_owned();
    }
    if provider == "cloudflare-workers-ai" {
        return CLOUDFLARE_WORKERS_AI_BASE_URL.to_owned();
    }
    if provider == "cloudflare-ai-gateway" {
        if model_id.starts_with("workers-ai/") {
            return CLOUDFLARE_AI_GATEWAY_COMPAT_BASE_URL.to_owned();
        }
        if model_id.contains("claude") {
            return CLOUDFLARE_AI_GATEWAY_ANTHROPIC_BASE_URL.to_owned();
        }
        return CLOUDFLARE_AI_GATEWAY_OPENAI_BASE_URL.to_owned();
    }
    base_url_for_provider(provider).to_owned()
}

fn api_for_provider(provider: &str, model_id: &str) -> &'static str {
    if provider == "openai"
        && (model_id.starts_with("gpt-5")
            || gpt4_generated_metadata(model_id).is_some()
            || o_series_generated_metadata(model_id).is_some())
    {
        return "openai-responses";
    }

    match provider {
        "anthropic" | "github-copilot" if model_id.contains("claude") => "anthropic-messages",
        "opencode" if model_id.contains("claude") => "anthropic-messages",
        "opencode-go" if model_id == "minimax-m2.5" => "anthropic-messages",
        "opencode" if model_id.contains("gemini") => "google-generative-ai",
        "opencode" if model_id.starts_with("gpt-") => "openai-responses",
        "cloudflare-ai-gateway" if model_id.contains("claude") => "anthropic-messages",
        "cloudflare-ai-gateway" if model_id.starts_with("gpt-") => "openai-responses",
        "fireworks" => "anthropic-messages",
        "amazon-bedrock" => "bedrock-converse-stream",
        "google" => "google-generative-ai",
        "google-vertex" => "google-vertex",
        "azure-openai-responses" => "azure-openai-responses",
        "openai-codex" => "openai-codex-responses",
        "mistral" => "mistral-conversations",
        "kimi-coding" | "minimax" | "minimax-cn" | "vercel-ai-gateway" => "anthropic-messages",
        _ => "openai-completions",
    }
}

fn base_url_for_provider(provider: &str) -> &'static str {
    match provider {
        "anthropic" => "https://api.anthropic.com",
        "amazon-bedrock" => "https://bedrock-runtime.us-east-1.amazonaws.com",
        "google" => "https://generativelanguage.googleapis.com",
        "google-vertex" => "https://{location}-aiplatform.googleapis.com",
        "openai" => "https://api.openai.com/v1",
        "azure-openai-responses" => "",
        "openai-codex" => "https://chatgpt.com/backend-api",
        "mistral" => "https://api.mistral.ai",
        "fireworks" => "https://api.fireworks.ai/inference",
        "together" => "https://api.together.ai/v1",
        "openrouter" => "https://openrouter.ai/api/v1",
        "xai" => "https://api.x.ai/v1",
        "groq" => "https://api.groq.com/openai/v1",
        "cerebras" => "https://api.cerebras.ai/v1",
        "huggingface" => "https://router.huggingface.co/v1",
        "minimax" => "https://api.minimax.io/anthropic",
        "minimax-cn" => "https://api.minimaxi.com/anthropic",
        "kimi-coding" => "https://api.kimi.com/coding",
        "vercel-ai-gateway" => "https://ai-gateway.vercel.sh",
        "zai" => "https://open.bigmodel.cn/api/paas/v4",
        "xiaomi" => "https://api.xiaomimimo.com/v1",
        "xiaomi-token-plan-cn" => "https://token-plan-cn.xiaomimimo.com/v1",
        "xiaomi-token-plan-ams" => "https://token-plan-ams.xiaomimimo.com/v1",
        "xiaomi-token-plan-sgp" => "https://token-plan-sgp.xiaomimimo.com/v1",
        _ => "https://example.invalid",
    }
}

fn reasoning_for_model(provider: &str, model_id: &str) -> bool {
    provider == "openai-codex"
        || provider == "zai"
        || model_id.starts_with("gpt-5")
        || model_id.contains("opus")
        || model_id.contains("sonnet")
        || model_id.contains("deepseek-r1")
        || model_id.contains("thinking")
        || model_id.contains("magistral")
        || model_id.contains("gemini-2.5")
        || model_id.contains("gemini-3")
        || is_deepseek_v4_flash(provider, model_id)
}

fn xhigh_effort_for_model(provider: &str, model_id: &str) -> Option<&'static str> {
    match (provider, model_id) {
        ("anthropic", "claude-opus-4-6") => Some("max"),
        ("anthropic", "claude-opus-4-7") => Some("xhigh"),
        ("openai-codex", "gpt-5.4" | "gpt-5.5") => Some("xhigh"),
        ("openrouter", "anthropic/claude-opus-4.6") => Some("max"),
        _ if is_deepseek_v4_flash(provider, model_id) => Some("max"),
        _ => None,
    }
}

fn is_deepseek_v4_flash(provider: &str, model_id: &str) -> bool {
    model_id.ends_with("deepseek-v4-flash")
        && matches!(provider, "deepseek" | "opencode-go" | "openrouter")
}

fn context_window_for_model(provider: &str, model_id: &str) -> u64 {
    if provider == "anthropic" || model_id.contains("claude") {
        200_000
    } else if model_id.contains("gemini") {
        1_000_000
    } else if model_id.contains("gpt-5") {
        400_000
    } else {
        128_000
    }
}

fn supports_images(provider: &str, model_id: &str) -> bool {
    provider == "google"
        || provider == "google-vertex"
        || model_id.contains("gpt-4o")
        || model_id.contains("gpt-5")
        || model_id.contains("gemini")
        || model_id.contains("claude")
        || model_id.contains("pixtral")
        || model_id.contains("glm-4.5v")
}

fn compat_for_model(provider: &str, model_id: &str) -> Option<Value> {
    if provider == "zai" {
        let mut compat = json!({
            "supportsDeveloperRole": false,
            "thinkingFormat": "zai",
        });
        if matches!(model_id, "glm-4.7" | "glm-5-turbo" | "glm-5.1") {
            compat["zaiToolStream"] = Value::Bool(true);
        }
        Some(compat)
    } else if provider == "cloudflare-workers-ai"
        || (provider == "cloudflare-ai-gateway" && model_id.starts_with("workers-ai/"))
    {
        Some(json!({
            "sendSessionAffinityHeaders": true,
            "supportsReasoningEffort": false,
            "supportsStrictMode": false,
            "supportsLongCacheRetention": false,
            "maxTokensField": "max_tokens",
        }))
    } else {
        None
    }
}

pub fn is_cloudflare_provider(provider: &str) -> bool {
    provider == "cloudflare-workers-ai" || provider == "cloudflare-ai-gateway"
}

pub fn resolve_cloudflare_base_url(model: &Model) -> Result<String, String> {
    if !model.base_url.contains('{') {
        return Ok(model.base_url.clone());
    }

    let mut resolved = String::new();
    let mut remaining = model.base_url.as_str();
    while let Some(start) = remaining.find('{') {
        resolved.push_str(&remaining[..start]);
        let after_start = &remaining[start + 1..];
        let Some(end) = after_start.find('}') else {
            resolved.push_str(&remaining[start..]);
            return Ok(resolved);
        };
        let name = &after_start[..end];
        if !is_cloudflare_env_placeholder(name) {
            resolved.push_str(&remaining[start..start + end + 2]);
            remaining = &after_start[end + 1..];
            continue;
        }
        let value = std::env::var(name).map_err(|_| {
            format!(
                "{name} is required for provider {} but is not set.",
                model.provider
            )
        })?;
        resolved.push_str(&value);
        remaining = &after_start[end + 1..];
    }
    resolved.push_str(remaining);
    Ok(resolved)
}

fn is_cloudflare_env_placeholder(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_uppercase())
        && chars.all(|ch| ch == '_' || ch.is_ascii_uppercase() || ch.is_ascii_digit())
}
