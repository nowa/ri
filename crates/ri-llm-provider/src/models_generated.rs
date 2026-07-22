//! Generated model catalog data, embedded from pi's generator output
//! (`packages/ai/scripts/generate-models.ts`, data regenerated 2026-07-23
//! from models.dev plus pi's overrides).
//!
//! The generated catalog seeds models that ri's hand-written metadata does
//! not cover; hand-written entries win so existing test baselines stay
//! authoritative for the models they assert on.

use crate::types::Model;
use std::collections::BTreeMap;

static GENERATED_MODEL_DATA: &[(&str, &str)] = &[
    (
        "amazon-bedrock",
        include_str!("models_generated/amazon-bedrock.json"),
    ),
    ("ant-ling", include_str!("models_generated/ant-ling.json")),
    ("anthropic", include_str!("models_generated/anthropic.json")),
    (
        "azure-openai-responses",
        include_str!("models_generated/azure-openai-responses.json"),
    ),
    ("cerebras", include_str!("models_generated/cerebras.json")),
    (
        "cloudflare-ai-gateway",
        include_str!("models_generated/cloudflare-ai-gateway.json"),
    ),
    (
        "cloudflare-workers-ai",
        include_str!("models_generated/cloudflare-workers-ai.json"),
    ),
    ("deepseek", include_str!("models_generated/deepseek.json")),
    ("fireworks", include_str!("models_generated/fireworks.json")),
    (
        "github-copilot",
        include_str!("models_generated/github-copilot.json"),
    ),
    ("google", include_str!("models_generated/google.json")),
    (
        "google-vertex",
        include_str!("models_generated/google-vertex.json"),
    ),
    ("groq", include_str!("models_generated/groq.json")),
    (
        "huggingface",
        include_str!("models_generated/huggingface.json"),
    ),
    (
        "kimi-coding",
        include_str!("models_generated/kimi-coding.json"),
    ),
    ("minimax", include_str!("models_generated/minimax.json")),
    (
        "minimax-cn",
        include_str!("models_generated/minimax-cn.json"),
    ),
    ("mistral", include_str!("models_generated/mistral.json")),
    (
        "moonshotai",
        include_str!("models_generated/moonshotai.json"),
    ),
    (
        "moonshotai-cn",
        include_str!("models_generated/moonshotai-cn.json"),
    ),
    ("nvidia", include_str!("models_generated/nvidia.json")),
    ("openai", include_str!("models_generated/openai.json")),
    (
        "openai-codex",
        include_str!("models_generated/openai-codex.json"),
    ),
    ("opencode", include_str!("models_generated/opencode.json")),
    (
        "opencode-go",
        include_str!("models_generated/opencode-go.json"),
    ),
    (
        "openrouter",
        include_str!("models_generated/openrouter.json"),
    ),
    (
        "qwen-token-plan",
        include_str!("models_generated/qwen-token-plan.json"),
    ),
    (
        "qwen-token-plan-cn",
        include_str!("models_generated/qwen-token-plan-cn.json"),
    ),
    ("together", include_str!("models_generated/together.json")),
    (
        "vercel-ai-gateway",
        include_str!("models_generated/vercel-ai-gateway.json"),
    ),
    ("xai", include_str!("models_generated/xai.json")),
    ("xiaomi", include_str!("models_generated/xiaomi.json")),
    (
        "xiaomi-token-plan-ams",
        include_str!("models_generated/xiaomi-token-plan-ams.json"),
    ),
    (
        "xiaomi-token-plan-cn",
        include_str!("models_generated/xiaomi-token-plan-cn.json"),
    ),
    (
        "xiaomi-token-plan-sgp",
        include_str!("models_generated/xiaomi-token-plan-sgp.json"),
    ),
    ("zai", include_str!("models_generated/zai.json")),
    (
        "zai-coding-cn",
        include_str!("models_generated/zai-coding-cn.json"),
    ),
];

/// Provider id -> models from the embedded generator output. Entries that no
/// longer deserialize (future schema drift) are skipped rather than failing
/// the whole catalog.
pub fn generated_catalog() -> &'static BTreeMap<String, Vec<Model>> {
    static CATALOG: std::sync::LazyLock<BTreeMap<String, Vec<Model>>> =
        std::sync::LazyLock::new(|| {
            let mut catalog = BTreeMap::new();
            for (provider_id, data) in GENERATED_MODEL_DATA {
                let Ok(entries) = serde_json::from_str::<BTreeMap<String, serde_json::Value>>(data)
                else {
                    continue;
                };
                let models: Vec<Model> = entries
                    .into_values()
                    .filter_map(|entry| serde_json::from_value(entry).ok())
                    .collect();
                catalog.insert((*provider_id).to_owned(), models);
            }
            catalog
        });
    &CATALOG
}

/// Providers present in the generated catalog, in sorted order.
pub fn generated_catalog_provider_ids() -> Vec<String> {
    generated_catalog().keys().cloned().collect()
}
