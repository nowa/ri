use crate::{
    api_registry::ProviderError,
    get_env_api_key,
    images_api_registry::{ImagesApiProvider, register_images_api_provider},
    json_repair::{parse_json_with_repair, sanitize_surrogates},
    node_http_proxy::reqwest_client_for_target,
    types::{
        AssistantImages, ImageContent, ImagesContent, ImagesContext, ImagesModel, ImagesOptions,
        ImagesStopReason, OutputKind, ProviderResponse, Usage, UsageCost, now_millis,
    },
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    future::Future,
    sync::{Arc, Once, atomic::Ordering},
    time::Duration,
};

static REGISTER_BUILTIN_IMAGES: Once = Once::new();
const OPENROUTER_IMAGES_BASE_RETRY_DELAY_MS: u64 = 1000;

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

        let payload = match options
            .apply_payload_hooks(model, build_openrouter_images_payload(model, &context))
        {
            Ok(payload) => payload,
            Err(error) => return Ok(openrouter_images_error(model, error, false)),
        };
        let headers = build_openrouter_images_default_headers(model, &options.headers);
        let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
        let client = match reqwest_client_for_target(&url) {
            Ok(client) => client,
            Err(error) => return Ok(openrouter_images_error(model, error, false)),
        };
        let max_retries = options.max_retries.unwrap_or(0) as usize;

        for attempt in 0..=max_retries {
            if abort_requested(&options) {
                return Ok(openrouter_images_error(model, "Request was aborted", true));
            }

            let request =
                build_openrouter_images_request(&client, &url, &payload, &headers, model, &options);
            let response = match await_openrouter_images_abortable(request.send(), &options).await {
                Some(Ok(response)) => response,
                Some(Err(error)) => {
                    let error = error.to_string();
                    if let Some(delay_ms) = openrouter_images_request_retry_delay_ms(
                        attempt,
                        max_retries,
                        options.max_retry_delay_ms,
                    ) {
                        if !sleep_openrouter_images_retry(delay_ms, &options).await {
                            return Ok(openrouter_images_error(model, "Request was aborted", true));
                        }
                        continue;
                    }
                    return Ok(openrouter_images_error(model, error, false));
                }
                None => return Ok(openrouter_images_error(model, "Request was aborted", true)),
            };
            if abort_requested(&options) {
                return Ok(openrouter_images_error(model, "Request was aborted", true));
            }
            let status = response.status();
            let response_headers = response
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.as_str().to_owned(), value.to_owned()))
                })
                .collect::<BTreeMap<_, _>>();
            let retry_after_ms =
                header_value(&response_headers, "retry-after-ms").map(str::to_owned);
            let retry_after = header_value(&response_headers, "retry-after").map(str::to_owned);
            if let Err(error) = options.emit_response_hooks(
                model,
                ProviderResponse {
                    status: status.as_u16(),
                    headers: response_headers,
                },
            ) {
                return Ok(openrouter_images_error(model, error, false));
            }
            let body = match await_openrouter_images_abortable(response.text(), &options).await {
                Some(Ok(body)) => body,
                Some(Err(error)) => {
                    if let Some(delay_ms) = openrouter_images_request_retry_delay_ms(
                        attempt,
                        max_retries,
                        options.max_retry_delay_ms,
                    ) {
                        if !sleep_openrouter_images_retry(delay_ms, &options).await {
                            return Ok(openrouter_images_error(model, "Request was aborted", true));
                        }
                        continue;
                    }
                    return Ok(openrouter_images_error(model, error.to_string(), false));
                }
                None => return Ok(openrouter_images_error(model, "Request was aborted", true)),
            };
            if !status.is_success() {
                let error = provider_error_from_body(status.as_u16(), &body);
                if let Some(delay_ms) = openrouter_images_retry_delay_ms(
                    status.as_u16(),
                    &error,
                    retry_after_ms.as_deref(),
                    retry_after.as_deref(),
                    attempt,
                    max_retries as u32,
                    options.max_retry_delay_ms,
                    now_millis() as i64,
                ) {
                    if !sleep_openrouter_images_retry(delay_ms, &options).await {
                        return Ok(openrouter_images_error(model, "Request was aborted", true));
                    }
                    continue;
                }
                return Ok(openrouter_images_error(model, error, false));
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
            return Ok(parse_openrouter_images_response(model, &value));
        }

        Ok(openrouter_images_error(
            model,
            "OpenRouter image request exhausted retries",
            false,
        ))
    }
}

fn build_openrouter_images_request(
    client: &reqwest::Client,
    url: &str,
    payload: &Value,
    headers: &BTreeMap<String, String>,
    model: &ImagesModel,
    options: &ImagesOptions,
) -> reqwest::RequestBuilder {
    let mut request = client.post(url).json(payload);
    for (name, value) in headers {
        request = request.header(name, value);
    }
    if let Some(timeout_ms) = options.timeout_ms {
        request = request.timeout(Duration::from_millis(timeout_ms));
    }
    if !headers_contain(headers, "authorization")
        && let Some(api_key) = options
            .api_key
            .clone()
            .or_else(|| get_env_api_key(&model.provider))
    {
        request = request.bearer_auth(api_key);
    }
    request
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

fn header_value<'a>(headers: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
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

pub fn openrouter_images_retry_delay_ms(
    status: u16,
    error_text: &str,
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    attempt: usize,
    max_retries: u32,
    max_retry_delay_ms: Option<u64>,
    now_ms: i64,
) -> Option<u64> {
    if attempt >= max_retries as usize || !is_openrouter_images_retryable_error(status, error_text)
    {
        return None;
    }

    let attempt = u32::try_from(attempt).unwrap_or(u32::MAX);
    let mut delay_ms =
        OPENROUTER_IMAGES_BASE_RETRY_DELAY_MS.saturating_mul(2_u64.saturating_pow(attempt));
    if let Some(retry_after_ms) = retry_after_ms {
        if let Ok(millis) = retry_after_ms.parse::<f64>()
            && millis.is_finite()
        {
            delay_ms = millis.max(0.0) as u64;
        }
    } else if let Some(retry_after) = retry_after {
        if let Ok(seconds) = retry_after.parse::<f64>()
            && seconds.is_finite()
        {
            delay_ms = (seconds.max(0.0) * 1000.0) as u64;
        } else if let Ok(date) = DateTime::parse_from_rfc2822(retry_after) {
            delay_ms = date
                .with_timezone(&Utc)
                .timestamp_millis()
                .saturating_sub(now_ms)
                .max(0) as u64;
        }
    }

    Some(cap_openrouter_images_retry_delay(
        delay_ms,
        max_retry_delay_ms,
    ))
}

fn openrouter_images_request_retry_delay_ms(
    attempt: usize,
    max_retries: usize,
    max_retry_delay_ms: Option<u64>,
) -> Option<u64> {
    if attempt >= max_retries {
        return None;
    }
    let attempt = u32::try_from(attempt).unwrap_or(u32::MAX);
    Some(cap_openrouter_images_retry_delay(
        OPENROUTER_IMAGES_BASE_RETRY_DELAY_MS.saturating_mul(2_u64.saturating_pow(attempt)),
        max_retry_delay_ms,
    ))
}

fn cap_openrouter_images_retry_delay(delay_ms: u64, max_retry_delay_ms: Option<u64>) -> u64 {
    max_retry_delay_ms.map_or(delay_ms, |max| delay_ms.min(max))
}

fn is_openrouter_images_retryable_error(status: u16, error_text: &str) -> bool {
    if matches!(status, 408 | 409 | 425 | 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    let normalized = error_text
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    normalized.contains("ratelimit")
        || normalized.contains("overloaded")
        || normalized.contains("serviceunavailable")
        || normalized.contains("upstreamconnect")
        || normalized.contains("connectionrefused")
}

async fn await_openrouter_images_abortable<T, E, F>(
    future: F,
    options: &ImagesOptions,
) -> Option<Result<T, E>>
where
    F: Future<Output = Result<T, E>>,
{
    let Some(abort_flag) = options.abort_flag.as_ref().cloned() else {
        return Some(future.await);
    };
    tokio::pin!(future);
    loop {
        if abort_flag.load(Ordering::SeqCst) {
            return None;
        }
        tokio::select! {
            result = &mut future => return Some(result),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
}

async fn sleep_openrouter_images_retry(delay_ms: u64, options: &ImagesOptions) -> bool {
    if delay_ms == 0 {
        return !abort_requested(options);
    }
    let Some(abort_flag) = options.abort_flag.as_ref().cloned() else {
        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        return true;
    };
    let sleep = tokio::time::sleep(Duration::from_millis(delay_ms));
    tokio::pin!(sleep);
    loop {
        if abort_flag.load(Ordering::SeqCst) {
            return false;
        }
        tokio::select! {
            _ = &mut sleep => return !abort_flag.load(Ordering::SeqCst),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
    }
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
