use crate::{
    api_registry::ProviderError,
    get_env_api_key,
    images_api_registry::{ImagesApiProvider, register_images_api_provider},
    json_repair::{parse_json_with_repair, sanitize_surrogates},
    types::{
        AssistantImages, ImageContent, ImagesContent, ImagesContext, ImagesModel, ImagesOptions,
        ImagesStopReason, OutputKind, Usage, UsageCost, now_millis,
    },
};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{Arc, Once, atomic::Ordering},
};

static REGISTER_BUILTIN_IMAGES: Once = Once::new();

pub fn ensure_builtin_images_api_providers() {
    REGISTER_BUILTIN_IMAGES.call_once(|| {
        register_images_api_provider(
            Arc::new(OpenRouterImagesHttpProvider),
            Some("builtin-http".to_owned()),
        );
    });
}

struct OpenRouterImagesHttpProvider;

#[async_trait]
impl ImagesApiProvider for OpenRouterImagesHttpProvider {
    fn api(&self) -> &str {
        "openrouter-images"
    }

    async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Result<AssistantImages, ProviderError> {
        if abort_requested(&options) {
            return Ok(openrouter_images_error(model, "Request was aborted", true));
        }

        let payload = build_openrouter_images_payload(model, &context);
        let headers = build_openrouter_images_default_headers(model, &options.headers);
        let client = reqwest::Client::new();
        let mut request = client
            .post(format!(
                "{}/chat/completions",
                model.base_url.trim_end_matches('/')
            ))
            .json(&payload);
        for (name, value) in &headers {
            request = request.header(name, value);
        }
        if !headers_contain(&headers, "authorization")
            && let Some(api_key) = options
                .api_key
                .clone()
                .or_else(|| get_env_api_key(&model.provider))
        {
            request = request.bearer_auth(api_key);
        }

        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => return Ok(openrouter_images_error(model, error.to_string(), false)),
        };
        if abort_requested(&options) {
            return Ok(openrouter_images_error(model, "Request was aborted", true));
        }
        let status = response.status();
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => return Ok(openrouter_images_error(model, error.to_string(), false)),
        };
        if !status.is_success() {
            return Ok(openrouter_images_error(
                model,
                provider_error_from_body(status.as_u16(), &body),
                false,
            ));
        }
        let value = match parse_json_with_repair::<Value>(&body) {
            Ok(value) => value,
            Err(error) => {
                return Ok(openrouter_images_error(
                    model,
                    format!("Could not parse OpenRouter image response: {error}"),
                    false,
                ));
            }
        };
        Ok(parse_openrouter_images_response(model, &value))
    }
}

pub fn build_openrouter_images_payload(model: &ImagesModel, context: &ImagesContext) -> Value {
    let content = context
        .input
        .iter()
        .map(|item| match item {
            ImagesContent::Text(text) => json!({
                "type": "text",
                "text": sanitize_surrogates(&text.text),
            }),
            ImagesContent::Image(image) => json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:{};base64,{}", image.mime_type, image.data),
                },
            }),
        })
        .collect::<Vec<_>>();

    let modalities = if model.output.contains(&OutputKind::Text) {
        json!(["image", "text"])
    } else {
        json!(["image"])
    };

    json!({
        "model": model.id,
        "messages": [
            {
                "role": "user",
                "content": content,
            },
        ],
        "stream": false,
        "modalities": modalities,
    })
}

pub fn build_openrouter_images_default_headers(
    model: &ImagesModel,
    option_headers: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut headers = model.headers.clone();
    headers.extend(option_headers.clone());
    headers
}

pub fn parse_openrouter_images_response(model: &ImagesModel, response: &Value) -> AssistantImages {
    let mut output = empty_openrouter_images_result(model);
    output.response_id = response
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_owned);

    if let Some(usage) = response.get("usage") {
        output.usage = Some(parse_openrouter_images_usage(usage, model));
    }

    let Some(choice) = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
    else {
        return output;
    };

    let Some(message) = choice.get("message") else {
        return output;
    };

    if let Some(content) = message.get("content").and_then(Value::as_str)
        && !content.is_empty()
    {
        output.output.push(ImagesContent::text(content));
    }

    if let Some(images) = message.get("images").and_then(Value::as_array) {
        for image in images {
            if let Some(image_url) = openrouter_image_url(image)
                && let Some(content) = parse_data_image_url(image_url)
            {
                output.output.push(ImagesContent::Image(content));
            }
        }
    }

    output
}

pub fn parse_openrouter_images_usage(raw_usage: &Value, model: &ImagesModel) -> Usage {
    let prompt_tokens = raw_usage
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let output = raw_usage
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let details = raw_usage.get("prompt_tokens_details");
    let reported_cached_tokens = details
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_write = details
        .and_then(|details| details.get("cache_write_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let cache_read = if cache_write > 0 {
        reported_cached_tokens.saturating_sub(cache_write)
    } else {
        reported_cached_tokens
    };
    let input = prompt_tokens
        .saturating_sub(cache_read)
        .saturating_sub(cache_write);

    let mut usage = Usage {
        input,
        output,
        cache_read,
        cache_write,
        total_tokens: input + output + cache_read + cache_write,
        cost: UsageCost::default(),
    };
    usage.cost.input = (model.cost.input / 1_000_000.0) * usage.input as f64;
    usage.cost.output = (model.cost.output / 1_000_000.0) * usage.output as f64;
    usage.cost.cache_read = (model.cost.cache_read / 1_000_000.0) * usage.cache_read as f64;
    usage.cost.cache_write = (model.cost.cache_write / 1_000_000.0) * usage.cache_write as f64;
    usage.cost.total =
        usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write;
    usage
}

pub fn openrouter_images_error(
    model: &ImagesModel,
    error_message: impl Into<String>,
    aborted: bool,
) -> AssistantImages {
    AssistantImages {
        stop_reason: if aborted {
            ImagesStopReason::Aborted
        } else {
            ImagesStopReason::Error
        },
        error_message: Some(error_message.into()),
        ..empty_openrouter_images_result(model)
    }
}

fn abort_requested(options: &ImagesOptions) -> bool {
    options
        .abort_flag
        .as_ref()
        .is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst))
}

fn headers_contain(headers: &BTreeMap<String, String>, name: &str) -> bool {
    headers.keys().any(|key| key.eq_ignore_ascii_case(name))
}

fn provider_error_from_body(status: u16, body: &str) -> String {
    parse_json_with_repair::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .or_else(|| {
                    value
                        .get("message")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
        })
        .unwrap_or_else(|| format!("Provider returned HTTP {status}: {body}"))
}

fn empty_openrouter_images_result(model: &ImagesModel) -> AssistantImages {
    AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: ImagesStopReason::Stop,
        error_message: None,
        timestamp: now_millis(),
    }
}

fn openrouter_image_url(image: &Value) -> Option<&str> {
    match image.get("image_url")? {
        Value::String(url) => Some(url.as_str()),
        Value::Object(object) => object.get("url").and_then(Value::as_str),
        _ => None,
    }
}

fn parse_data_image_url(image_url: &str) -> Option<ImageContent> {
    let payload = image_url.strip_prefix("data:")?;
    let (mime_type, data) = payload.split_once(";base64,")?;
    if mime_type.is_empty() || data.is_empty() {
        return None;
    }
    Some(ImageContent {
        mime_type: mime_type.to_owned(),
        data: data.to_owned(),
    })
}
