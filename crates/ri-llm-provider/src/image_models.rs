use crate::types::{ImagesModel, InputKind, ModelCost, OutputKind};
use parking_lot::RwLock;
use std::collections::{BTreeMap, BTreeSet};

static IMAGE_MODEL_REGISTRY: std::sync::LazyLock<
    RwLock<BTreeMap<String, BTreeMap<String, ImagesModel>>>,
> = std::sync::LazyLock::new(|| RwLock::new(seed_image_models()));

pub fn get_image_model(provider: &str, model_id: &str) -> Option<ImagesModel> {
    IMAGE_MODEL_REGISTRY
        .read()
        .get(provider)
        .and_then(|models| models.get(model_id))
        .cloned()
}

pub fn get_image_providers() -> Vec<String> {
    let mut providers: BTreeSet<String> = IMAGE_MODEL_REGISTRY.read().keys().cloned().collect();
    providers.extend(
        KNOWN_IMAGE_PROVIDERS
            .iter()
            .map(|provider| (*provider).to_owned()),
    );
    providers.into_iter().collect()
}

pub fn get_image_models(provider: &str) -> Vec<ImagesModel> {
    IMAGE_MODEL_REGISTRY
        .read()
        .get(provider)
        .map(|models| models.values().cloned().collect())
        .unwrap_or_default()
}

pub fn register_image_model(model: ImagesModel) {
    IMAGE_MODEL_REGISTRY
        .write()
        .entry(model.provider.clone())
        .or_default()
        .insert(model.id.clone(), model);
}

pub fn clear_image_models() {
    *IMAGE_MODEL_REGISTRY.write() = BTreeMap::new();
}

pub fn reset_image_models() {
    *IMAGE_MODEL_REGISTRY.write() = seed_image_models();
}

const KNOWN_IMAGE_PROVIDERS: &[&str] = &["openrouter"];

const TEXT_IMAGE_INPUT: &[InputKind] = &[InputKind::Text, InputKind::Image];
const IMAGE_TEXT_INPUT: &[InputKind] = &[InputKind::Image, InputKind::Text];
const IMAGE_OUTPUT: &[OutputKind] = &[OutputKind::Image];
const IMAGE_TEXT_OUTPUT: &[OutputKind] = &[OutputKind::Image, OutputKind::Text];
const TEXT_IMAGE_OUTPUT: &[OutputKind] = &[OutputKind::Text, OutputKind::Image];

struct ImageModelSeed {
    id: &'static str,
    name: &'static str,
    input: &'static [InputKind],
    output: &'static [OutputKind],
    cost: ModelCost,
}

fn seed_image_models() -> BTreeMap<String, BTreeMap<String, ImagesModel>> {
    let mut registry = BTreeMap::new();
    for seed in OPENROUTER_IMAGE_MODELS {
        let model = image_model_from_seed("openrouter", seed);
        registry
            .entry(model.provider.clone())
            .or_insert_with(BTreeMap::new)
            .insert(model.id.clone(), model);
    }
    registry
}

fn image_model_from_seed(provider: &str, seed: &ImageModelSeed) -> ImagesModel {
    ImagesModel {
        id: seed.id.to_owned(),
        name: seed.name.to_owned(),
        api: "openrouter-images".to_owned(),
        provider: provider.to_owned(),
        base_url: "https://openrouter.ai/api/v1".to_owned(),
        input: seed.input.to_vec(),
        output: seed.output.to_vec(),
        cost: seed.cost.clone(),
        headers: BTreeMap::new(),
    }
}

const fn cost(input: f64, output: f64, cache_read: f64, cache_write: f64) -> ModelCost {
    ModelCost {
        input,
        output,
        cache_read,
        cache_write,
    }
}

const OPENROUTER_IMAGE_MODELS: &[ImageModelSeed] = &[
    ImageModelSeed {
        id: "black-forest-labs/flux.2-flex",
        name: "Black Forest Labs: FLUX.2 Flex",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "black-forest-labs/flux.2-klein-4b",
        name: "Black Forest Labs: FLUX.2 Klein 4B",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "black-forest-labs/flux.2-max",
        name: "Black Forest Labs: FLUX.2 Max",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "black-forest-labs/flux.2-pro",
        name: "Black Forest Labs: FLUX.2 Pro",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "bytedance-seed/seedream-4.5",
        name: "ByteDance Seed: Seedream 4.5",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "google/gemini-2.5-flash-image",
        name: "Google: Nano Banana (Gemini 2.5 Flash Image)",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_TEXT_OUTPUT,
        cost: cost(0.3, 2.5, 0.03, 0.08333333333333334),
    },
    ImageModelSeed {
        id: "google/gemini-3-pro-image-preview",
        name: "Google: Nano Banana Pro (Gemini 3 Pro Image Preview)",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_TEXT_OUTPUT,
        cost: cost(2.0, 12.0, 0.19999999999999998, 0.375),
    },
    ImageModelSeed {
        id: "google/gemini-3.1-flash-image-preview",
        name: "Google: Nano Banana 2 (Gemini 3.1 Flash Image Preview)",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_TEXT_OUTPUT,
        cost: cost(0.5, 3.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "openai/gpt-5-image",
        name: "OpenAI: GPT-5 Image",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_TEXT_OUTPUT,
        cost: cost(10.0, 10.0, 1.25, 0.0),
    },
    ImageModelSeed {
        id: "openai/gpt-5-image-mini",
        name: "OpenAI: GPT-5 Image Mini",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_TEXT_OUTPUT,
        cost: cost(2.5, 2.0, 0.25, 0.0),
    },
    ImageModelSeed {
        id: "openai/gpt-5.4-image-2",
        name: "OpenAI: GPT-5.4 Image 2",
        input: IMAGE_TEXT_INPUT,
        output: IMAGE_TEXT_OUTPUT,
        cost: cost(8.0, 15.0, 2.0, 0.0),
    },
    ImageModelSeed {
        id: "openrouter/auto",
        name: "Auto Router",
        input: TEXT_IMAGE_INPUT,
        output: TEXT_IMAGE_OUTPUT,
        cost: cost(-1_000_000.0, -1_000_000.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v3",
        name: "Recraft: Recraft V3",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4",
        name: "Recraft: Recraft V4",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4-pro",
        name: "Recraft: Recraft V4 Pro",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4-pro-vector",
        name: "Recraft: Recraft V4 Pro Vector",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4-vector",
        name: "Recraft: Recraft V4 Vector",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4.1",
        name: "Recraft: Recraft V4.1",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4.1-pro",
        name: "Recraft: Recraft V4.1 Pro",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4.1-pro-vector",
        name: "Recraft: Recraft V4.1 Pro Vector",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4.1-utility",
        name: "Recraft: Recraft V4.1 Utility",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4.1-utility-pro",
        name: "Recraft: Recraft V4.1 Utility Pro",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "recraft/recraft-v4.1-vector",
        name: "Recraft: Recraft V4.1 Vector",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "sourceful/riverflow-v2-fast",
        name: "Sourceful: Riverflow V2 Fast",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "sourceful/riverflow-v2-fast-preview",
        name: "Sourceful: Riverflow V2 Fast Preview",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "sourceful/riverflow-v2-max-preview",
        name: "Sourceful: Riverflow V2 Max Preview",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "sourceful/riverflow-v2-pro",
        name: "Sourceful: Riverflow V2 Pro",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
    ImageModelSeed {
        id: "sourceful/riverflow-v2-standard-preview",
        name: "Sourceful: Riverflow V2 Standard Preview",
        input: TEXT_IMAGE_INPUT,
        output: IMAGE_OUTPUT,
        cost: cost(0.0, 0.0, 0.0, 0.0),
    },
];
