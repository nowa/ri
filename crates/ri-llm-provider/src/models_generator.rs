//! ri-owned model catalog generator (replaces re-running pi's
//! `scripts/generate-models.ts` for routine refreshes).
//!
//! v1 scope and policy — deliberately conservative so a refresh can never
//! silently regress the curated catalog:
//! - Source: models.dev `api.json` (the dominant upstream pi also uses).
//! - Existing models: only "market data" fields refresh (name, reasoning,
//!   input modalities, cost, context window, max tokens). Curated fields
//!   (api, base URL, thinking maps, compat, headers, pricing tiers) always
//!   come from the embedded snapshot. Guards skip cost updates that would
//!   zero out subscription pricing or clobber tiered pricing.
//! - New models: added only for providers whose pi mapping rules are ported
//!   (Anthropic, Google, Google Vertex, Amazon Bedrock); elsewhere they are
//!   reported for review instead of added.
//! - Removed upstream models are kept and reported as stale, never dropped.

use crate::types::{InputKind, Model, ModelCost};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub const MODELS_DEV_API_URL: &str = "https://models.dev/api.json";

/// Providers whose pi mapping rules are ported well enough to add new models.
const ADDITION_PROVIDERS: &[&str] = &["anthropic", "google", "google-vertex", "amazon-bedrock"];

/// Providers pi actually sources from models.dev. Everything else (OpenRouter
/// and gateway APIs, derived aliases, subscription token plans) has a
/// different upstream and never matches against models.dev data.
const MODELS_DEV_SOURCED_PROVIDERS: &[&str] = &[
    "amazon-bedrock",
    "anthropic",
    "cerebras",
    "deepseek",
    "fireworks",
    "google",
    "google-vertex",
    "groq",
    "huggingface",
    "minimax",
    "minimax-cn",
    "mistral",
    "moonshotai",
    "moonshotai-cn",
    "openai",
    "together",
    "xai",
    "xiaomi",
    "zai",
];

/// Models whose context/output limits carry explicit pi corrections over
/// models.dev (e.g. gpt-5-pro's max output duplicates the input sub-limit
/// upstream); their limits never refresh automatically.
const LIMIT_CORRECTED_MODELS: &[(&str, &str)] = &[("openai", "gpt-5-pro")];

/// A models.dev model entry, reduced to the fields the mapping uses.
#[derive(Debug, Clone, Default)]
pub struct ModelsDevModel {
    pub name: Option<String>,
    pub tool_call: bool,
    pub reasoning: bool,
    pub image_input: bool,
    pub cost: Option<ModelCost>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
}

fn parse_models_dev_model(value: &Value) -> ModelsDevModel {
    let cost = value.get("cost").and_then(Value::as_object).map(|cost| {
        let read = |key: &str| cost.get(key).and_then(Value::as_f64).unwrap_or(0.0);
        ModelCost {
            input: read("input"),
            output: read("output"),
            cache_read: read("cache_read"),
            cache_write: read("cache_write"),
            tiers: Vec::new(),
        }
    });
    ModelsDevModel {
        name: value.get("name").and_then(Value::as_str).map(str::to_owned),
        tool_call: value.get("tool_call").and_then(Value::as_bool) == Some(true),
        reasoning: value.get("reasoning").and_then(Value::as_bool) == Some(true),
        image_input: value
            .pointer("/modalities/input")
            .and_then(Value::as_array)
            .is_some_and(|input| input.iter().any(|kind| kind.as_str() == Some("image"))),
        cost,
        context_window: value.pointer("/limit/context").and_then(Value::as_u64),
        max_tokens: value.pointer("/limit/output").and_then(Value::as_u64),
    }
}

/// Parsed models.dev catalog: provider id -> model id -> entry.
pub fn parse_models_dev_catalog(
    api_json: &Value,
) -> BTreeMap<String, BTreeMap<String, ModelsDevModel>> {
    let mut catalog = BTreeMap::new();
    let Some(providers) = api_json.as_object() else {
        return catalog;
    };
    for (provider_id, provider) in providers {
        let Some(models) = provider.get("models").and_then(Value::as_object) else {
            continue;
        };
        let entries = models
            .iter()
            .map(|(model_id, model)| (model_id.clone(), parse_models_dev_model(model)))
            .collect();
        catalog.insert(provider_id.clone(), entries);
    }
    catalog
}

pub async fn fetch_models_dev_catalog()
-> Result<BTreeMap<String, BTreeMap<String, ModelsDevModel>>, String> {
    let response = reqwest::get(MODELS_DEV_API_URL)
        .await
        .map_err(|error| format!("models.dev fetch failed: {error}"))?;
    if !response.status().is_success() {
        return Err(format!("models.dev API returned {}", response.status()));
    }
    let body: Value = response
        .json()
        .await
        .map_err(|error| format!("models.dev API returned invalid JSON: {error}"))?;
    Ok(parse_models_dev_catalog(&body))
}

/// Skip rules ported from pi's generator for the addition providers.
fn skip_new_model(provider_id: &str, model_id: &str) -> bool {
    match provider_id {
        // No tool use in streaming mode / no system messages.
        "amazon-bedrock" => {
            model_id.starts_with("ai21.jamba")
                || model_id.starts_with("mistral.mistral-7b-instruct-v0")
        }
        // Only the Gemini streaming path is implemented for Vertex.
        "google-vertex" => {
            !model_id.starts_with("gemini-") || model_id == "gemini-3.1-flash-lite-preview"
        }
        _ => false,
    }
}

fn new_model_base(provider_id: &str, model_id: &str) -> Option<(String, String)> {
    let (api, base_url) = match provider_id {
        "anthropic" => ("anthropic-messages", "https://api.anthropic.com".to_owned()),
        "google" => (
            "google-generative-ai",
            "https://generativelanguage.googleapis.com/v1beta".to_owned(),
        ),
        "google-vertex" => (
            "google-vertex",
            "https://{location}-aiplatform.googleapis.com".to_owned(),
        ),
        "amazon-bedrock" => (
            "bedrock-converse-stream",
            crate::bedrock::bedrock_base_url_for_model(model_id).to_owned(),
        ),
        _ => return None,
    };
    Some((api.to_owned(), base_url))
}

fn build_new_model(provider_id: &str, model_id: &str, entry: &ModelsDevModel) -> Option<Model> {
    let (api, base_url) = new_model_base(provider_id, model_id)?;
    let mut model = Model {
        id: model_id.to_owned(),
        name: entry.name.clone().unwrap_or_else(|| model_id.to_owned()),
        api,
        provider: provider_id.to_owned(),
        base_url,
        reasoning: entry.reasoning,
        thinking_level_map: BTreeMap::new(),
        input: if entry.image_input {
            vec![InputKind::Text, InputKind::Image]
        } else {
            vec![InputKind::Text]
        },
        cost: entry.cost.clone().unwrap_or_default(),
        context_window: entry.context_window.unwrap_or(4096),
        max_tokens: entry.max_tokens.unwrap_or(4096),
        headers: BTreeMap::new(),
        compat: None,
    };
    if provider_id == "anthropic" {
        crate::models::apply_anthropic_adaptive_catalog_metadata(&mut model);
    }
    apply_cost_corrections(&mut model);
    Some(model)
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct ProviderRefresh {
    /// Final model list for the provider after the refresh.
    pub models: Vec<Model>,
    pub updated: Vec<String>,
    pub added: Vec<String>,
    /// Present in the snapshot but no longer reported upstream (kept).
    pub stale: Vec<String>,
    /// New upstream models for providers whose rules are not ported yet
    /// (reported, not added).
    pub skipped_additions: Vec<String>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct CatalogRefreshPlan {
    pub providers: BTreeMap<String, ProviderRefresh>,
}

impl CatalogRefreshPlan {
    pub fn has_changes(&self) -> bool {
        self.providers
            .values()
            .any(|provider| !provider.updated.is_empty() || !provider.added.is_empty())
    }

    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        for (provider_id, refresh) in &self.providers {
            if refresh.updated.is_empty()
                && refresh.added.is_empty()
                && refresh.stale.is_empty()
                && refresh.skipped_additions.is_empty()
            {
                continue;
            }
            lines.push(format!(
                "{provider_id}: {} updated, {} added, {} stale kept, {} additions skipped",
                refresh.updated.len(),
                refresh.added.len(),
                refresh.stale.len(),
                refresh.skipped_additions.len(),
            ));
            for id in &refresh.added {
                lines.push(format!("  + {id}"));
            }
            for id in &refresh.skipped_additions {
                lines.push(format!("  ? {id} (rules not ported; review manually)"));
            }
            for id in &refresh.stale {
                lines.push(format!("  ~ {id} (no longer upstream; kept)"));
            }
        }
        if lines.is_empty() {
            "catalog is up to date".to_owned()
        } else {
            lines.join("\n")
        }
    }
}

/// Update one cost component, refusing to zero out a curated non-zero value
/// (pi adds e.g. Mistral cache pricing that models.dev reports as zero).
fn update_cost_component(embedded: &mut f64, fetched: f64) -> bool {
    if fetched == *embedded || (fetched == 0.0 && *embedded != 0.0) {
        return false;
    }
    *embedded = fetched;
    true
}

/// Cost corrections pi applies over models.dev data.
fn apply_cost_corrections(model: &mut Model) {
    if model.provider == "google-vertex" {
        // models.dev Vertex cache values do not match the official Gemini
        // pricing table; pi only accounts cachedContentTokenCount as
        // cacheRead.
        model.cost.cache_write = 0.0;
        if model.id == "gemini-2.5-flash" {
            model.cost.cache_read = 0.03;
        }
    }
}

fn apply_market_update(model: &mut Model, entry: &ModelsDevModel) -> bool {
    // Tiered pricing (and its capped limits) is derived by curation rules;
    // never partially update such models.
    if !model.cost.tiers.is_empty() {
        return false;
    }
    let original = model.clone();
    let mut changed = false;
    if let Some(name) = &entry.name
        && *name != model.name
    {
        model.name = name.clone();
        changed = true;
    }
    if entry.reasoning != model.reasoning {
        model.reasoning = entry.reasoning;
        changed = true;
    }
    let input = if entry.image_input {
        vec![InputKind::Text, InputKind::Image]
    } else {
        vec![InputKind::Text]
    };
    if input != model.input {
        model.input = input;
        changed = true;
    }
    if let Some(cost) = &entry.cost {
        changed |= update_cost_component(&mut model.cost.input, cost.input);
        changed |= update_cost_component(&mut model.cost.output, cost.output);
        changed |= update_cost_component(&mut model.cost.cache_read, cost.cache_read);
        changed |= update_cost_component(&mut model.cost.cache_write, cost.cache_write);
    }
    let limits_corrected =
        LIMIT_CORRECTED_MODELS.contains(&(model.provider.as_str(), model.id.as_str()));
    if !limits_corrected {
        if let Some(context_window) = entry.context_window
            && context_window != model.context_window
        {
            model.context_window = context_window;
            changed = true;
        }
        if let Some(max_tokens) = entry.max_tokens
            && max_tokens != model.max_tokens
        {
            model.max_tokens = max_tokens;
            changed = true;
        }
    }
    apply_cost_corrections(model);
    let _ = changed;
    *model != original
}

/// Plan a refresh of `embedded` (provider id -> models) from a models.dev
/// catalog, under the conservative v1 policy.
pub fn plan_catalog_refresh(
    models_dev: &BTreeMap<String, BTreeMap<String, ModelsDevModel>>,
    embedded: &BTreeMap<String, Vec<Model>>,
) -> CatalogRefreshPlan {
    let mut plan = CatalogRefreshPlan::default();
    for (provider_id, embedded_models) in embedded {
        let mut refresh = ProviderRefresh {
            models: embedded_models.clone(),
            ..Default::default()
        };
        let sourced = MODELS_DEV_SOURCED_PROVIDERS.contains(&provider_id.as_str());
        if let Some(fetched) = models_dev.get(provider_id).filter(|_| sourced) {
            let known: BTreeSet<&str> = embedded_models
                .iter()
                .map(|model| model.id.as_str())
                .collect();
            for model in &mut refresh.models {
                match fetched.get(&model.id) {
                    Some(entry) if entry.tool_call => {
                        if apply_market_update(model, entry) {
                            refresh.updated.push(model.id.clone());
                        }
                    }
                    _ => refresh.stale.push(model.id.clone()),
                }
            }
            for (model_id, entry) in fetched {
                if !entry.tool_call
                    || known.contains(model_id.as_str())
                    || skip_new_model(provider_id, model_id)
                {
                    continue;
                }
                if ADDITION_PROVIDERS.contains(&provider_id.as_str()) {
                    if let Some(model) = build_new_model(provider_id, model_id, entry) {
                        refresh.added.push(model_id.clone());
                        refresh.models.push(model);
                    }
                } else {
                    refresh.skipped_additions.push(model_id.clone());
                }
            }
            refresh.models.sort_by(|a, b| a.id.cmp(&b.id));
        }
        plan.providers.insert(provider_id.clone(), refresh);
    }
    plan
}

/// Serialize one provider's refreshed models in the embedded snapshot format
/// (an object keyed by model id).
pub fn render_generated_provider_json(models: &[Model]) -> String {
    let mut object = serde_json::Map::new();
    for model in models {
        object.insert(
            model.id.clone(),
            serde_json::to_value(model).expect("serialize model"),
        );
    }
    serde_json::to_string_pretty(&Value::Object(object)).expect("serialize catalog")
}
