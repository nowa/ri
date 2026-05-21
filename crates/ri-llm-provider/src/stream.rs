use crate::{
    api_registry::{ProviderError, get_api_provider},
    event_stream::{AssistantMessageEventStream, assistant_message_event_stream},
    http_api_provider::ensure_builtin_api_providers,
    models::get_supported_thinking_levels,
    simple_options::apply_simple_stream_defaults,
    types::{
        AssistantMessage, AssistantMessageEvent, Context, Model, SimpleStreamOptions, StopReason,
        StreamOptions, ThinkingLevel, Usage, now_millis,
    },
};

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
    let options = apply_simple_stream_defaults(model, options);
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
