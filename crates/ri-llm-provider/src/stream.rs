use crate::{
    api_registry::{ProviderError, api_provider_source_matches, get_api_provider},
    event_stream::{AssistantMessageEventStream, assistant_message_event_stream},
    http_api_provider::{BUILTIN_API_PROVIDER_SOURCE_ID, ensure_builtin_api_providers},
    models::get_supported_thinking_levels,
    models_runtime::Provider,
    providers_all::compat_models,
    simple_options::apply_simple_stream_defaults,
    types::{
        AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, StopReason,
        StreamOptions, ThinkingLevel, Usage, now_millis,
    },
};
use std::sync::Arc;

/// The runtime provider for a built-in catalog model, unless the api-registry
/// entry for its api has been overridden (tests, extensions) — mirrors pi
/// compat's `getBuiltinProviderForModel`.
fn builtin_provider_for_model(model: &Model) -> Option<Arc<dyn Provider>> {
    if !api_provider_source_matches(&model.api, BUILTIN_API_PROVIDER_SOURCE_ID) {
        return None;
    }
    let provider = compat_models().get_provider(&model.provider)?;
    provider
        .get_models()
        .iter()
        .any(|candidate| candidate.api == model.api)
        .then_some(provider)
}

fn has_resolved_cloudflare_auth(options: &StreamOptions) -> bool {
    options
        .api_key
        .as_deref()
        .is_some_and(|api_key| !api_key.trim().is_empty())
        || options.headers.contains_key("cf-aig-authorization")
}

/// Cloudflare gateway models need credential-store/provider-env resolution;
/// without explicit auth the request routes through the compat `Models`
/// collection.
fn routes_through_compat_models(model: &Model, options: &StreamOptions) -> bool {
    model.provider.starts_with("cloudflare-") && !has_resolved_cloudflare_auth(options)
}

pub fn stream(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> Result<AssistantMessageEventStream, ProviderError> {
    ensure_builtin_api_providers();
    if let Some(error) = unsupported_reasoning_error(
        model,
        SimpleStreamOptions::reasoning_from_stream_options(&options),
    ) {
        return Ok(error_stream(model, error));
    }
    if let Some(provider) = builtin_provider_for_model(model) {
        if routes_through_compat_models(model, &options) {
            return Ok(compat_models().stream(model, context, options));
        }
        return Ok(provider.stream(model, context, options));
    }
    let provider =
        get_api_provider(&model.api).ok_or_else(|| ProviderError::MissingApi(model.api.clone()))?;
    provider.stream(model, context, options)
}

pub async fn complete(
    model: &Model,
    context: Context,
    options: StreamOptions,
) -> Result<AssistantMessage, ProviderError> {
    Ok(stream(model, context, options)?.result().await)
}

pub fn stream_simple(
    model: &Model,
    context: Context,
    options: SimpleStreamOptions,
) -> Result<AssistantMessageEventStream, ProviderError> {
    ensure_builtin_api_providers();
    if let Some(error) = unsupported_reasoning_error(model, options.reasoning) {
        return Ok(error_stream(model, error));
    }
    let options = apply_simple_stream_defaults(model, &context, options);
    if let Some(provider) = builtin_provider_for_model(model) {
        if routes_through_compat_models(model, &options.stream) {
            return Ok(compat_models().stream_simple(model, context, options));
        }
        return Ok(provider.stream_simple(model, context, options));
    }
    let provider =
        get_api_provider(&model.api).ok_or_else(|| ProviderError::MissingApi(model.api.clone()))?;
    provider.stream_simple(model, context, options)
}

pub async fn complete_simple(
    model: &Model,
    context: Context,
    options: SimpleStreamOptions,
) -> Result<AssistantMessage, ProviderError> {
    Ok(stream_simple(model, context, options)?.result().await)
}

fn unsupported_reasoning_error(model: &Model, reasoning: Option<ThinkingLevel>) -> Option<String> {
    if reasoning == Some(ThinkingLevel::XHigh)
        && !get_supported_thinking_levels(model).contains(&ThinkingLevel::XHigh)
    {
        return Some(format!(
            "Model {}/{} does not support xhigh reasoning",
            model.provider, model.id
        ));
    }
    None
}

fn error_stream(model: &Model, error_message: String) -> AssistantMessageEventStream {
    let (sender, stream) = assistant_message_event_stream();
    let message = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Error,
        error_message: Some(error_message),
        timestamp: now_millis(),
    };
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message,
    });
    stream
}
