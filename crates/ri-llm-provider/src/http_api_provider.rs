use crate::{
    anthropic::{
        AnthropicClientOptions, AnthropicStreamProcessor, build_anthropic_client_config,
        build_anthropic_simple_payload_for_client,
    },
    api_registry::{
        ApiProvider, ProviderError, clear_api_providers, ensure_model_api, get_api_provider,
        register_api_provider,
    },
    azure_openai::{
        AzureOpenAIConfigOptions, AzureOpenAIResponsesPayloadOptions,
        build_azure_openai_responses_payload, resolve_azure_openai_config,
    },
    bedrock::{
        BedrockClientConfig, BedrockClientOptions, BedrockConverseStreamProcessor,
        BedrockPayloadOptions, build_bedrock_payload, parse_bedrock_tool_choice,
        resolve_aws_credentials_with_runtime, resolve_aws_profile_region,
        resolve_bedrock_client_config, sign_aws_sigv4_headers, standard_bedrock_endpoint_region,
    },
    diagnostics::{
        AssistantMessageDiagnostic, append_assistant_message_diagnostic,
        create_assistant_message_diagnostic,
    },
    event_stream::{AssistantMessageEventStream, assistant_message_event_stream},
    get_env_api_key_scoped,
    google_shared::{GoogleStreamProcessor, build_google_simple_payload},
    google_vertex::{
        GoogleVertexClientConfig, GoogleVertexOptions, resolve_google_vertex_adc_access_token,
        resolve_google_vertex_client_config,
    },
    json_repair::parse_json_with_repair,
    mistral::{
        MistralChatStreamProcessor, build_mistral_request_headers, build_mistral_simple_payload,
        format_mistral_http_error,
    },
    models::{is_cloudflare_provider, resolve_cloudflare_base_url},
    node_http_proxy::reqwest_client_for_target,
    openai_codex_responses::{
        OpenAICodexCachedWebSocketContinuation, OpenAICodexResponsesPayloadOptions,
        OpenAICodexWebSocket, build_openai_codex_cached_websocket_continuation,
        build_openai_codex_cached_websocket_request_body, build_openai_codex_responses_payload,
        build_openai_codex_sse_headers, build_openai_codex_websocket_headers,
        extract_openai_codex_account_id, openai_codex_error_message_from_response,
        openai_codex_retry_delay_ms_with_limits, openai_codex_websocket_sse_fallback_active,
        record_openai_codex_websocket_failure_for_session,
        record_openai_codex_websocket_request_stats_for_session,
        record_openai_codex_websocket_sse_fallback_for_session, resolve_openai_codex_url,
        resolve_openai_codex_websocket_url,
    },
    openai_completions::{
        OpenAICompletionsPayloadOptions, OpenAICompletionsStreamProcessor,
        build_openai_completions_default_headers_with_context, build_openai_completions_payload,
    },
    openai_responses::{
        OpenAIResponsesPayloadOptions, OpenAIResponsesStreamProcessor,
        build_openai_responses_default_headers_with_context, build_openai_responses_payload,
    },
    types::{
        AssistantMessage, AssistantMessageEvent, Context, Model, ProviderResponse,
        SimpleStreamOptions, StopReason, StreamOptions, Tool, Transport, Usage, now_millis,
    },
};
use futures::StreamExt;
use serde_json::{Map, Value};
use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock, atomic::Ordering},
};

static OPENAI_CODEX_WS_CACHE: OnceLock<tokio::sync::Mutex<BTreeMap<String, CachedCodexWebSocket>>> =
    OnceLock::new();
pub const BUILTIN_API_PROVIDER_SOURCE_ID: &str = "builtin-http";

struct CachedCodexWebSocket {
    socket: OpenAICodexWebSocket,
    continuation: Option<OpenAICodexCachedWebSocketContinuation>,
    created_at_ms: i64,
    /// When the socket was last released back to the cache; drives the
    /// 5-minute idle TTL (pi SESSION_WEBSOCKET_CACHE_TTL_MS).
    last_used_at_ms: i64,
}

enum CodexWebSocketStreamError {
    Transport(String),
    NonTransport(String),
    /// The server refused the connection with
    /// `websocket_connection_limit_reached`; the caller retries once before
    /// falling back to SSE (pi #5973).
    ConnectionLimit(String),
}

impl CodexWebSocketStreamError {
    fn transport(error: impl Into<String>) -> Self {
        Self::Transport(error.into())
    }

    fn non_transport(error: impl Into<String>) -> Self {
        Self::NonTransport(error.into())
    }

    fn is_non_transport(&self) -> bool {
        matches!(self, Self::NonTransport(_))
    }

    fn is_connection_limit(&self) -> bool {
        matches!(self, Self::ConnectionLimit(_))
    }

    fn into_message(self) -> String {
        match self {
            Self::Transport(error) | Self::NonTransport(error) | Self::ConnectionLimit(error) => {
                error
            }
        }
    }
}

pub async fn cleanup_openai_codex_websocket_sessions(session_id: Option<&str>) -> usize {
    let entries = {
        let mut cache = codex_ws_cache().lock().await;
        if let Some(session_id) = session_id {
            cache.remove(session_id).into_iter().collect::<Vec<_>>()
        } else {
            std::mem::take(&mut *cache)
                .into_values()
                .collect::<Vec<_>>()
        }
    };
    let cleaned = entries.len();
    for mut entry in entries {
        let _ = entry.socket.close().await;
    }
    cleaned
}

pub fn ensure_builtin_api_providers() {
    register_missing_builtin_api_providers();
}

pub fn register_builtin_api_providers() {
    register_builtin_api_provider(Arc::new(AnthropicMessagesHttpProvider));
    register_builtin_api_provider(Arc::new(PiMessagesHttpProvider));
    register_builtin_api_provider(Arc::new(OpenAICompletionsHttpProvider));
    register_builtin_api_provider(Arc::new(MistralHttpProvider));
    register_builtin_api_provider(Arc::new(OpenAIResponsesHttpProvider));
    register_builtin_api_provider(Arc::new(AzureOpenAIResponsesHttpProvider));
    register_builtin_api_provider(Arc::new(OpenAICodexResponsesHttpProvider));
    register_builtin_api_provider(Arc::new(GoogleGenerativeAiHttpProvider));
    register_builtin_api_provider(Arc::new(GoogleVertexHttpProvider));
    register_builtin_api_provider(Arc::new(BedrockConverseStreamHttpProvider));
}

pub fn reset_api_providers() {
    clear_api_providers();
    register_builtin_api_providers();
}

fn register_missing_builtin_api_providers() {
    register_builtin_api_provider_if_missing(
        "anthropic-messages",
        Arc::new(AnthropicMessagesHttpProvider),
    );
    register_builtin_api_provider_if_missing(
        "openai-completions",
        Arc::new(OpenAICompletionsHttpProvider),
    );
    register_builtin_api_provider_if_missing(
        "mistral-conversations",
        Arc::new(MistralHttpProvider),
    );
    register_builtin_api_provider_if_missing(
        "openai-responses",
        Arc::new(OpenAIResponsesHttpProvider),
    );
    register_builtin_api_provider_if_missing(
        "azure-openai-responses",
        Arc::new(AzureOpenAIResponsesHttpProvider),
    );
    register_builtin_api_provider_if_missing(
        "openai-codex-responses",
        Arc::new(OpenAICodexResponsesHttpProvider),
    );
    register_builtin_api_provider_if_missing(
        "google-generative-ai",
        Arc::new(GoogleGenerativeAiHttpProvider),
    );
    register_builtin_api_provider_if_missing("google-vertex", Arc::new(GoogleVertexHttpProvider));
    register_builtin_api_provider_if_missing(
        "bedrock-converse-stream",
        Arc::new(BedrockConverseStreamHttpProvider),
    );
    register_builtin_api_provider_if_missing("pi-messages", Arc::new(PiMessagesHttpProvider));
}

fn register_builtin_api_provider_if_missing(api: &str, provider: Arc<dyn ApiProvider>) {
    if get_api_provider(api).is_none() {
        register_builtin_api_provider(provider);
    }
}

fn register_builtin_api_provider(provider: Arc<dyn ApiProvider>) {
    register_api_provider(provider, Some(BUILTIN_API_PROVIDER_SOURCE_ID.to_owned()));
}

struct PiMessagesHttpProvider;

impl ApiProvider for PiMessagesHttpProvider {
    fn api(&self) -> &str {
        "pi-messages"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions {
                stream: options,
                ..Default::default()
            },
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        let api_key = options
            .stream
            .api_key
            .clone()
            .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env))
            .ok_or_else(|| {
                ProviderError::Provider(format!(
                    "No API key provided for provider \"{}\"",
                    model.provider
                ))
            })?;
        let debug = options.stream.extra.get("debug").and_then(Value::as_bool) == Some(true);
        let url = crate::pi_messages::pi_messages_url(model, debug);
        let payload = build_pi_messages_payload_checked(model, &context, &options)?;
        let (sender, stream) = assistant_message_event_stream();
        let model = model.clone();
        tokio::spawn(async move {
            let mut output = empty_assistant_message(&model);
            if let Err(error) = stream_pi_messages_sse(
                &model,
                &options,
                &url,
                &api_key,
                payload,
                &sender,
                &mut output,
            )
            .await
            {
                push_provider_error(&sender, &mut output, StopReason::Error, error);
            }
        });
        Ok(stream)
    }
}

fn build_pi_messages_payload_checked(
    model: &Model,
    context: &Context,
    options: &SimpleStreamOptions,
) -> Result<Value, ProviderError> {
    let payload = crate::pi_messages::build_pi_messages_payload(model, context, options);
    options
        .apply_payload_hooks(model, payload)
        .map_err(ProviderError::Provider)
}

async fn stream_pi_messages_sse(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    api_key: &str,
    payload: Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    let client = reqwest_client_for_target(url)?;
    let mut request = client
        .post(url)
        .json(&payload)
        .header("authorization", format!("Bearer {api_key}"))
        .header("accept", "text/event-stream");
    for (name, value) in &options.stream.headers {
        request = request.header(name, value);
    }
    if let Some(timeout_ms) = options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms));
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let status_text = status.canonical_reason().unwrap_or_default().to_owned();
        let retry_after = retry_after_hint_ms(&response);
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(append_retry_after_hint(
            crate::pi_messages::append_pi_messages_response_failure(
                output,
                model,
                url,
                status.as_u16(),
                &status_text,
                &body,
            ),
            retry_after,
        ));
    }

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor = crate::pi_messages::PiMessagesStreamProcessor::new();
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events);
        for event in events.drain(..) {
            processor.process_event(event, output, sender);
            if processor.is_terminal() {
                return Ok(());
            }
        }
    }
    parser.finish(&mut events);
    for event in events.drain(..) {
        processor.process_event(event, output, sender);
        if processor.is_terminal() {
            return Ok(());
        }
    }
    Err(format!(
        "{} stream ended without a terminal event",
        model.provider
    ))
}

struct OpenAICompletionsHttpProvider;

impl ApiProvider for OpenAICompletionsHttpProvider {
    fn api(&self) -> &str {
        "openai-completions"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let cache_retention =
            crate::openai_completions::resolve_openai_completions_cache_retention_scoped(
                options.stream.cache_retention,
                &options.stream.env,
            );
        let mut payload = build_openai_completions_payload(
            model,
            &context,
            OpenAICompletionsPayloadOptions {
                tool_choice: options.stream.extra.get("toolChoice").cloned(),
                reasoning: options.reasoning,
                cache_retention: Some(cache_retention),
                session_id: options.stream.session_id.clone(),
                max_tokens: options.stream.max_tokens,
                temperature: options.stream.temperature,
                headers: options.stream.headers.clone(),
            },
        );
        payload["stream"] = Value::Bool(true);
        let mut headers = build_openai_completions_default_headers_with_context(
            model,
            Some(&context),
            options.stream.session_id.as_deref(),
            cache_retention,
            &options.stream.headers,
        );
        remove_null_provider_headers(&mut headers, &options.stream);
        let base_url = resolved_model_base_url(model).map_err(ProviderError::Provider)?;
        spawn_openai_completions_sse_request(
            model.clone(),
            options.clone(),
            endpoint_url(&base_url, "chat/completions"),
            headers,
            payload,
        )
    }
}

struct OpenAIResponsesHttpProvider;

impl ApiProvider for OpenAIResponsesHttpProvider {
    fn api(&self) -> &str {
        "openai-responses"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let cache_retention =
            crate::openai_responses::resolve_openai_responses_cache_retention_scoped(
                options.stream.cache_retention,
                &options.stream.env,
            );
        let payload = build_openai_responses_payload(
            model,
            &context,
            OpenAIResponsesPayloadOptions {
                cache_retention: Some(cache_retention),
                session_id: options.stream.session_id.clone(),
                max_tokens: options.stream.max_tokens,
                temperature: options.stream.temperature,
                tool_choice: options.stream.extra.get("toolChoice").cloned(),
                service_tier: options
                    .stream
                    .extra
                    .get("serviceTier")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                reasoning_effort: options.reasoning,
                reasoning_summary: options
                    .stream
                    .extra
                    .get("reasoningSummary")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
        );
        let mut headers = build_openai_responses_default_headers_with_context(
            model,
            Some(&context),
            options.stream.session_id.as_deref(),
            cache_retention,
            &options.stream.headers,
        );
        remove_null_provider_headers(&mut headers, &options.stream);
        let base_url = resolved_model_base_url(model).map_err(ProviderError::Provider)?;
        spawn_openai_responses_sse_request(
            model.clone(),
            options.clone(),
            endpoint_url(&base_url, "responses"),
            headers,
            payload,
        )
    }
}

struct MistralHttpProvider;

impl ApiProvider for MistralHttpProvider {
    fn api(&self) -> &str {
        "mistral-conversations"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let payload = build_mistral_simple_payload(model, &context, options.clone());
        let mut headers = build_mistral_request_headers(
            model,
            options.stream.session_id.as_deref(),
            &options.stream.headers,
        );
        remove_null_provider_headers(&mut headers, &options.stream);
        spawn_mistral_sse_request(
            model.clone(),
            options,
            mistral_chat_completions_url(&model.base_url),
            headers,
            payload,
        )
    }
}

struct AzureOpenAIResponsesHttpProvider;

impl ApiProvider for AzureOpenAIResponsesHttpProvider {
    fn api(&self) -> &str {
        "azure-openai-responses"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let config = resolve_azure_openai_config(
            model,
            AzureOpenAIConfigOptions {
                env: options.stream.env.clone(),
                azure_base_url: options
                    .stream
                    .extra
                    .get("azureBaseUrl")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                azure_resource_name: options
                    .stream
                    .extra
                    .get("azureResourceName")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                azure_api_version: options
                    .stream
                    .extra
                    .get("azureApiVersion")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
        )
        .map_err(ProviderError::Provider)?;
        let payload = build_azure_openai_responses_payload(
            model,
            &context,
            AzureOpenAIResponsesPayloadOptions {
                session_id: options.stream.session_id.clone(),
                max_tokens: options.stream.max_tokens,
                temperature: options.stream.temperature,
                reasoning_effort: options.reasoning,
                reasoning_summary: options
                    .stream
                    .extra
                    .get("reasoningSummary")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                azure_deployment_name: options
                    .stream
                    .extra
                    .get("azureDeploymentName")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
        );
        let mut headers = model.headers.clone();
        headers.extend(options.stream.headers.clone());
        remove_null_provider_headers(&mut headers, &options.stream);
        if !headers_contain(&headers, "api-key")
            && let Some(api_key) = options
                .stream
                .api_key
                .clone()
                .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env))
        {
            headers.insert("api-key".to_owned(), api_key);
        }
        spawn_openai_responses_sse_request(
            model.clone(),
            options,
            azure_openai_responses_url(&config.base_url, &config.api_version),
            headers,
            payload,
        )
    }
}

struct AnthropicMessagesHttpProvider;

impl ApiProvider for AnthropicMessagesHttpProvider {
    fn api(&self) -> &str {
        "anthropic-messages"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let api_key = options
            .stream
            .api_key
            .clone()
            .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env))
            .filter(|api_key| !api_key.is_empty());
        // pi assertRequestAuth: fail fast when neither an API key nor an auth
        // header is present instead of sending an unauthenticated request.
        if api_key.is_none() && !headers_carry_request_auth(&options.stream.headers) {
            return Err(ProviderError::Provider(format!(
                "No API key for provider: {}",
                model.provider
            )));
        }
        let api_key = api_key.unwrap_or_default();
        let config = build_anthropic_client_config(
            model,
            &context,
            AnthropicClientOptions {
                api_key,
                headers: options.stream.headers.clone(),
                session_id: options.stream.session_id.clone(),
                cache_retention: Some(crate::anthropic::resolve_anthropic_cache_retention_scoped(
                    options.stream.cache_retention,
                    &options.stream.env,
                )),
                interleaved_thinking: options
                    .stream
                    .extra
                    .get("interleavedThinking")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                ..Default::default()
            },
        );
        let payload = build_anthropic_simple_payload_for_client(
            model,
            &context,
            options.clone(),
            config.is_oauth_token,
        );
        let mut headers = config.default_headers.clone();
        remove_null_provider_headers(&mut headers, &options.stream);
        headers
            .entry("anthropic-version".to_owned())
            .or_insert_with(|| "2023-06-01".to_owned());
        if let Some(api_key) = config.api_key
            && !headers_contain(&headers, "x-api-key")
        {
            headers.insert("x-api-key".to_owned(), api_key);
        }
        if let Some(auth_token) = config.auth_token
            && !headers_contain(&headers, "authorization")
        {
            headers.insert("authorization".to_owned(), format!("Bearer {auth_token}"));
        }
        let base_url = resolved_model_base_url(model).map_err(ProviderError::Provider)?;
        spawn_anthropic_sse_request(
            model.clone(),
            options,
            // The Anthropic SDK posts to `{baseUrl}/v1/messages` (the catalog
            // base URLs carry no /v1 segment).
            endpoint_url(&base_url, "v1/messages"),
            headers,
            payload,
            context.tools.clone(),
            config.is_oauth_token,
        )
    }
}

/// pi `assertRequestAuth`: accepted request-auth headers with a non-blank
/// value.
fn headers_carry_request_auth(headers: &BTreeMap<String, String>) -> bool {
    ["authorization", "x-api-key", "cf-aig-authorization"]
        .iter()
        .any(|name| {
            headers
                .iter()
                .any(|(key, value)| key.eq_ignore_ascii_case(name) && !value.trim().is_empty())
        })
}

struct GoogleGenerativeAiHttpProvider;

impl ApiProvider for GoogleGenerativeAiHttpProvider {
    fn api(&self) -> &str {
        "google-generative-ai"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let payload = build_google_simple_payload(model, &context, options.clone());
        let mut headers = model.headers.clone();
        headers.extend(options.stream.headers.clone());
        remove_null_provider_headers(&mut headers, &options.stream);
        if !headers_contain(&headers, "x-goog-api-key") {
            let api_key = options
                .stream
                .api_key
                .clone()
                .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env))
                .ok_or_else(|| {
                    ProviderError::Provider(format!("No API key for provider: {}", model.provider))
                })?;
            headers.insert("x-goog-api-key".to_owned(), api_key);
        }
        spawn_google_sse_request(
            model.clone(),
            options,
            google_generative_ai_stream_url(&model.base_url, &model.id),
            headers,
            payload,
        )
    }
}

struct GoogleVertexHttpProvider;

impl ApiProvider for GoogleVertexHttpProvider {
    fn api(&self) -> &str {
        "google-vertex"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let config = resolve_google_vertex_client_config(
            model,
            GoogleVertexOptions {
                api_key: options
                    .stream
                    .api_key
                    .clone()
                    .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env)),
                project: options
                    .stream
                    .extra
                    .get("project")
                    .or_else(|| options.stream.extra.get("vertexProject"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                location: options
                    .stream
                    .extra
                    .get("location")
                    .or_else(|| options.stream.extra.get("vertexLocation"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                headers: options.stream.headers.clone(),
            },
        )
        .map_err(ProviderError::Provider)?;
        let payload = build_google_simple_payload(model, &context, options.clone());
        let mut headers = config
            .http_options
            .as_ref()
            .map(|http_options| http_options.headers.clone())
            .unwrap_or_else(|| {
                let mut headers = model.headers.clone();
                headers.extend(options.stream.headers.clone());
                headers
            });
        remove_null_provider_headers(&mut headers, &options.stream);
        if !headers_contain(&headers, "x-goog-api-key")
            && !headers_contain(&headers, "authorization")
            && let Some(api_key) = config.api_key.clone()
        {
            headers.insert("x-goog-api-key".to_owned(), api_key);
        }
        let url = google_vertex_stream_url(&config, model)?;
        spawn_google_sse_request(model.clone(), options, url, headers, payload)
    }
}

struct OpenAICodexResponsesHttpProvider;

impl ApiProvider for OpenAICodexResponsesHttpProvider {
    fn api(&self) -> &str {
        "openai-codex-responses"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let token = options
            .stream
            .api_key
            .clone()
            .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env))
            .ok_or_else(|| {
                ProviderError::Provider(format!("No API key for provider: {}", model.provider))
            })?;
        let account_id =
            extract_openai_codex_account_id(&token).map_err(ProviderError::Provider)?;
        let payload = build_openai_codex_responses_payload(
            model,
            &context,
            OpenAICodexResponsesPayloadOptions {
                session_id: options.stream.session_id.clone(),
                temperature: options.stream.temperature,
                tool_choice: options
                    .stream
                    .extra
                    .get("toolChoice")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                service_tier: options
                    .stream
                    .extra
                    .get("serviceTier")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                text_verbosity: options
                    .stream
                    .extra
                    .get("textVerbosity")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                reasoning_effort: options.reasoning,
                reasoning_summary: options
                    .stream
                    .extra
                    .get("reasoningSummary")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
        );
        // Null-valued options headers unset model defaults before the codex
        // builders add their own required headers (pi buildBaseCodexHeaders
        // deletes null entries from the Headers object).
        let mut codex_model_headers = model.headers.clone();
        remove_null_provider_headers(&mut codex_model_headers, &options.stream);
        let headers = build_openai_codex_sse_headers(
            &codex_model_headers,
            &options.stream.headers,
            &account_id,
            &token,
            options.stream.session_id.as_deref(),
        );
        let websocket_request_id = options
            .stream
            .session_id
            .clone()
            .unwrap_or_else(crate::uuidv7);
        let websocket_headers = build_openai_codex_websocket_headers(
            &codex_model_headers,
            &options.stream.headers,
            &account_id,
            &token,
            &websocket_request_id,
        );
        spawn_openai_codex_responses_request(
            model.clone(),
            options,
            resolve_openai_codex_url(Some(&model.base_url)),
            resolve_openai_codex_websocket_url(Some(&model.base_url)),
            headers,
            websocket_headers,
            payload,
        )
    }
}

struct BedrockConverseStreamHttpProvider;

impl ApiProvider for BedrockConverseStreamHttpProvider {
    fn api(&self) -> &str {
        "bedrock-converse-stream"
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: crate::StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream_simple(
            model,
            context,
            SimpleStreamOptions::from_stream_options(options),
        )
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        ensure_model_api(model, self.api())?;
        let options = apply_bedrock_simple_stream_defaults(model, &context, options);
        let config = resolve_bedrock_client_config(
            model,
            BedrockClientOptions {
                env: options.stream.env.clone(),
                region: options
                    .stream
                    .extra
                    .get("region")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                profile: options
                    .stream
                    .extra
                    .get("profile")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            },
        );
        let payload = build_bedrock_payload(
            model,
            &context,
            BedrockPayloadOptions {
                cache_retention: Some(crate::bedrock::resolve_bedrock_cache_retention_scoped(
                    options.stream.cache_retention,
                    &options.stream.env,
                )),
                max_tokens: options.stream.max_tokens,
                temperature: options.stream.temperature,
                tool_choice: options
                    .stream
                    .extra
                    .get("toolChoice")
                    .and_then(parse_bedrock_tool_choice),
                reasoning: options.reasoning,
                region: config.region.clone(),
                thinking_budgets: options.thinking_budgets.clone(),
                interleaved_thinking: options
                    .stream
                    .extra
                    .get("interleavedThinking")
                    .and_then(Value::as_bool),
                thinking_display: options
                    .stream
                    .extra
                    .get("thinkingDisplay")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                request_metadata: options.stream.extra.get("requestMetadata").cloned(),
            },
        );
        let mut headers = model.headers.clone();
        // `host` and `x-amz-*` participate in the SigV4 canonical request and
        // `authorization` is owned by SigV4 or the bearer-token path; none may
        // be overwritten by caller-supplied headers (pi isReservedHeader).
        headers.extend(options.stream.headers.iter().filter_map(|(name, value)| {
            let lower = name.to_ascii_lowercase();
            (!lower.starts_with("x-amz-") && lower != "host" && lower != "authorization")
                .then(|| (name.clone(), value.clone()))
        }));
        remove_null_provider_headers(&mut headers, &options.stream);
        headers
            .entry("accept".to_owned())
            .or_insert_with(|| "application/vnd.amazon.eventstream".to_owned());
        headers
            .entry("content-type".to_owned())
            .or_insert_with(|| "application/json".to_owned());
        let skip_auth = crate::get_provider_env_value("AWS_BEDROCK_SKIP_AUTH", &options.stream.env)
            .as_deref()
            == Some("1");
        let bearer = (!skip_auth)
            .then(|| {
                options.stream.api_key.clone().or_else(|| {
                    crate::get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", &options.stream.env)
                })
            })
            .flatten();
        if !headers_contain(&headers, "authorization")
            && let Some(token) = bearer
        {
            headers.insert("authorization".to_owned(), format!("Bearer {token}"));
        }
        spawn_bedrock_eventstream_request(
            model.clone(),
            options,
            bedrock_converse_stream_url(
                config.endpoint.as_deref().unwrap_or(&model.base_url),
                &model.id,
            ),
            headers,
            payload,
            config,
        )
    }
}

/// pi bedrock `streamSimple`: an unset maxTokens falls back to the clamped
/// model cap, and non-adaptive Claude thinking grows/clamps maxTokens so the
/// budget fits inside `maxTokens - 1024` (pi adjustMaxTokensForThinking).
fn apply_bedrock_simple_stream_defaults(
    model: &Model,
    context: &Context,
    mut options: SimpleStreamOptions,
) -> SimpleStreamOptions {
    let base_max_tokens = crate::simple_options::clamp_max_tokens_to_context(
        model,
        context,
        options.stream.max_tokens.unwrap_or(model.max_tokens),
    );
    options.stream.max_tokens = Some(base_max_tokens);

    let Some(reasoning) = options.reasoning else {
        return options;
    };
    if !crate::bedrock::is_bedrock_anthropic_claude_model(model)
        || crate::bedrock::supports_bedrock_adaptive_thinking(model)
    {
        return options;
    }

    let adjusted = crate::simple_options::adjust_max_tokens_for_thinking(
        base_max_tokens,
        model.max_tokens,
        reasoning,
        options.thinking_budgets.as_ref(),
    );
    let max_tokens =
        crate::simple_options::clamp_max_tokens_to_context(model, context, adjusted.max_tokens);
    options.stream.max_tokens = Some(max_tokens);
    let budget = adjusted
        .thinking_budget
        .min(max_tokens.saturating_sub(1024));
    let mut budgets = options.thinking_budgets.clone().unwrap_or_default();
    match crate::simple_options::clamp_reasoning_for_budget(reasoning) {
        crate::types::ThinkingLevel::Minimal => budgets.minimal = Some(budget),
        crate::types::ThinkingLevel::Low => budgets.low = Some(budget),
        crate::types::ThinkingLevel::Medium => budgets.medium = Some(budget),
        crate::types::ThinkingLevel::High => budgets.high = Some(budget),
        _ => {}
    }
    options.thinking_budgets = Some(budgets);
    options
}

fn spawn_openai_codex_responses_request(
    model: Model,
    options: SimpleStreamOptions,
    sse_url: String,
    websocket_url: String,
    sse_headers: BTreeMap<String, String>,
    websocket_headers: BTreeMap<String, String>,
    payload: Value,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        let transport = options.stream.transport.unwrap_or(Transport::Auto);
        let session_id = options.stream.session_id.clone();
        let websocket_disabled_for_session = transport != Transport::Sse
            && openai_codex_websocket_sse_fallback_active(session_id.as_deref());
        if websocket_disabled_for_session {
            record_openai_codex_websocket_sse_fallback_for_session(session_id.as_deref());
        }
        if transport != Transport::Sse && !websocket_disabled_for_session {
            let mut retried_websocket_connection_limit = false;
            loop {
                let mut websocket_started = false;
                match stream_openai_codex_websocket_json(
                    &model,
                    &options,
                    &websocket_url,
                    &websocket_headers,
                    &payload,
                    &sender,
                    &mut output,
                    &mut websocket_started,
                )
                .await
                {
                    Ok(()) => return,
                    Err(error)
                        if !websocket_started
                            && error.is_connection_limit()
                            && !retried_websocket_connection_limit =>
                    {
                        // Reconnect once when the server rejects the socket for
                        // hitting its connection limit (pi #5973).
                        retried_websocket_connection_limit = true;
                        continue;
                    }
                    Err(error) if error.is_non_transport() => {
                        push_provider_error(
                            &sender,
                            &mut output,
                            StopReason::Error,
                            error.into_message(),
                        );
                        return;
                    }
                    Err(error) if websocket_started => {
                        let error = error.into_message();
                        record_openai_codex_websocket_failure_for_session(
                            session_id.as_deref(),
                            &error,
                        );
                        append_assistant_message_diagnostic(
                            &mut output,
                            provider_transport_failure_diagnostic(
                                transport,
                                None,
                                error.clone(),
                                true,
                                payload.to_string().len(),
                            ),
                        );
                        push_provider_error(&sender, &mut output, StopReason::Error, error);
                        return;
                    }
                    Err(error) => {
                        let error = error.into_message();
                        record_openai_codex_websocket_failure_for_session(
                            session_id.as_deref(),
                            &error,
                        );
                        append_assistant_message_diagnostic(
                            &mut output,
                            provider_transport_failure_diagnostic(
                                transport,
                                Some("sse"),
                                error,
                                false,
                                payload.to_string().len(),
                            ),
                        );
                        record_openai_codex_websocket_sse_fallback_for_session(
                            session_id.as_deref(),
                        );
                        break;
                    }
                }
            }
        }

        if let Err(error) = stream_openai_codex_sse_json(
            &model,
            &options,
            &sse_url,
            &sse_headers,
            payload,
            &sender,
            &mut output,
        )
        .await
        {
            push_provider_error(&sender, &mut output, StopReason::Error, error);
        }
    });
    Ok(stream)
}

fn spawn_openai_responses_sse_request(
    model: Model,
    options: SimpleStreamOptions,
    url: String,
    headers: BTreeMap<String, String>,
    payload: Value,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        if let Err(error) = stream_openai_responses_sse_json(
            &model,
            &options,
            &url,
            &headers,
            payload,
            &sender,
            &mut output,
        )
        .await
        {
            push_provider_error(&sender, &mut output, StopReason::Error, error);
        }
    });
    Ok(stream)
}

fn spawn_google_sse_request(
    model: Model,
    options: SimpleStreamOptions,
    url: String,
    headers: BTreeMap<String, String>,
    payload: Value,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        if let Err(error) = stream_google_sse_json(
            &model,
            &options,
            &url,
            &headers,
            payload,
            &sender,
            &mut output,
        )
        .await
            && !processor_already_pushed_error(&output, &error)
        {
            push_provider_error(&sender, &mut output, StopReason::Error, error);
        }
    });
    Ok(stream)
}

fn spawn_bedrock_eventstream_request(
    model: Model,
    options: SimpleStreamOptions,
    url: String,
    headers: BTreeMap<String, String>,
    payload: Value,
    config: BedrockClientConfig,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        if let Err(error) = stream_bedrock_eventstream_json(
            &model,
            &options,
            &url,
            &headers,
            payload,
            &config,
            &sender,
            &mut output,
        )
        .await
            && !processor_already_pushed_error(&output, &error)
        {
            push_provider_error(&sender, &mut output, StopReason::Error, error);
        }
    });
    Ok(stream)
}

fn spawn_anthropic_sse_request(
    model: Model,
    options: SimpleStreamOptions,
    url: String,
    headers: BTreeMap<String, String>,
    payload: Value,
    tools: Vec<Tool>,
    use_claude_code_tool_names: bool,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        if let Err(error) = stream_anthropic_sse_json(
            &model,
            &options,
            &url,
            &headers,
            payload,
            tools,
            use_claude_code_tool_names,
            &sender,
            &mut output,
        )
        .await
        {
            push_provider_error(&sender, &mut output, StopReason::Error, error);
        }
    });
    Ok(stream)
}

fn spawn_mistral_sse_request(
    model: Model,
    options: SimpleStreamOptions,
    url: String,
    headers: BTreeMap<String, String>,
    payload: Value,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        if let Err(error) = stream_mistral_sse_json(
            &model,
            &options,
            &url,
            &headers,
            payload,
            &sender,
            &mut output,
        )
        .await
        {
            if !processor_already_pushed_error(&output, &error) {
                push_provider_error(&sender, &mut output, StopReason::Error, error);
            }
        }
    });
    Ok(stream)
}

fn spawn_openai_completions_sse_request(
    model: Model,
    options: SimpleStreamOptions,
    url: String,
    headers: BTreeMap<String, String>,
    payload: Value,
) -> Result<AssistantMessageEventStream, ProviderError> {
    let (sender, stream) = assistant_message_event_stream();
    let payload = options
        .apply_payload_hooks(&model, payload)
        .map_err(ProviderError::Provider)?;
    tokio::spawn(async move {
        let mut output = empty_assistant_message(&model);
        if let Err(error) = stream_openai_completions_sse_json(
            &model,
            &options,
            &url,
            &headers,
            payload,
            &sender,
            &mut output,
        )
        .await
            && !processor_already_pushed_error(&output, &error)
        {
            push_provider_error(&sender, &mut output, StopReason::Error, error);
        }
    });
    Ok(stream)
}

async fn stream_openai_codex_websocket_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: &Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
    websocket_started: &mut bool,
) -> Result<(), CodexWebSocketStreamError> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let transport = options.stream.transport.unwrap_or(Transport::Auto);
    let use_cached_context = matches!(transport, Transport::Auto | Transport::WebsocketCached);
    let session_id = options.stream.session_id.clone();
    let mut cached = if let Some(session_id) = session_id.as_deref() {
        codex_ws_cache().lock().await.remove(session_id)
    } else {
        None
    };
    // Rotate sessions past their server-side lifetime instead of reusing a
    // socket the server is about to close (pi #6268), and drop sockets that
    // sat idle past the cache TTL (pi SESSION_WEBSOCKET_CACHE_TTL_MS) so a
    // fresh connection is dialed instead.
    if let Some(entry) = cached.take_if(|entry| {
        crate::openai_codex_responses::openai_codex_websocket_session_expired(
            entry.created_at_ms,
            now_millis() as i64,
        ) || crate::openai_codex_responses::openai_codex_websocket_session_idle_expired(
            entry.last_used_at_ms,
            now_millis() as i64,
        )
    }) {
        let mut socket = entry.socket;
        let _ = socket.close().await;
    }
    let reused_connection = cached.is_some();
    let (mut socket, continuation, connection_created_at_ms) = if let Some(cached) = cached {
        (cached.socket, cached.continuation, cached.created_at_ms)
    } else {
        // A user abort interrupts the connect wait promptly (pi
        // connectWebSocket observes the abort signal).
        let socket = tokio::select! {
            result = connect_openai_codex_websocket(
                url,
                headers,
                openai_codex_websocket_connect_timeout_ms(options),
            ) => result?,
            _ = wait_for_abort_flag(options) => {
                push_abort_if_requested(sender, options, output);
                return Ok(());
            }
        };
        (socket, None, now_millis() as i64)
    };

    let cached_request = if use_cached_context {
        build_openai_codex_cached_websocket_request_body(payload, continuation.as_ref())
    } else {
        build_openai_codex_cached_websocket_request_body(payload, None)
    };
    let mut request_body = cached_request.body;
    record_openai_codex_websocket_request_stats_for_session(
        session_id.as_deref(),
        &request_body,
        reused_connection,
        use_cached_context,
    );
    insert_object_field(
        &mut request_body,
        "type",
        Value::String("response.create".to_owned()),
    );
    socket
        .send_json_text(&request_body)
        .await
        .map_err(CodexWebSocketStreamError::transport)?;

    let mut processor = OpenAIResponsesStreamProcessor::with_request_service_tier(
        request_service_tier_for_usage(model, options),
    );
    // The stream timeout acts as an idle timeout between websocket events (pi
    // 7c02a556): before the first event the transport error falls back to
    // SSE; after the stream started it surfaces as an error.
    let idle_timeout_ms = options
        .stream
        .timeout_ms
        .filter(|timeout_ms| *timeout_ms > 0);
    loop {
        if push_abort_if_requested(sender, options, output) {
            let _ = socket.close().await;
            return Ok(());
        }
        enum CodexWebSocketRead {
            Event(Result<Option<Value>, String>),
            IdleTimeout,
            Aborted,
        }
        // The read wait observes both the idle timeout and the user abort
        // flag (pi parseWebSocket wakes on abort immediately).
        let read_outcome = {
            let read_future = async {
                match idle_timeout_ms {
                    Some(timeout_ms) => {
                        match tokio::time::timeout(
                            std::time::Duration::from_millis(timeout_ms),
                            socket.read_json_text(),
                        )
                        .await
                        {
                            Ok(result) => CodexWebSocketRead::Event(result),
                            Err(_) => CodexWebSocketRead::IdleTimeout,
                        }
                    }
                    None => CodexWebSocketRead::Event(socket.read_json_text().await),
                }
            };
            tokio::select! {
                outcome = read_future => outcome,
                _ = wait_for_abort_flag(options) => CodexWebSocketRead::Aborted,
            }
        };
        let read_result = match read_outcome {
            CodexWebSocketRead::Event(result) => result,
            CodexWebSocketRead::IdleTimeout => {
                let _ = socket.close().await;
                return Err(CodexWebSocketStreamError::transport(format!(
                    "WebSocket idle timeout after {}ms",
                    idle_timeout_ms.unwrap_or_default()
                )));
            }
            CodexWebSocketRead::Aborted => {
                let _ = socket.close().await;
                push_abort_if_requested(sender, options, output);
                return Ok(());
            }
        };
        let event = read_result
            .map_err(|error| {
                if error.starts_with("Invalid Codex WebSocket JSON:") {
                    CodexWebSocketStreamError::non_transport(error)
                } else {
                    CodexWebSocketStreamError::transport(error)
                }
            })?
            .ok_or_else(|| {
                CodexWebSocketStreamError::transport(
                    "WebSocket stream closed before response.completed",
                )
            })?;
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        // Codex error events fail the stream before it counts as started
        // (pi maps them ahead of the first-event marker), so a
        // connection-limit rejection can be retried on a fresh socket.
        if event_type == "error" {
            let code = crate::openai_codex_responses::openai_codex_error_event_code(&event)
                .unwrap_or("unknown")
                .to_owned();
            let message = event
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| event.pointer("/error/message").and_then(Value::as_str))
                .unwrap_or("Unknown error");
            let message = format!("Error Code {code}: {message}");
            let _ = socket.close().await;
            if code
                == crate::openai_codex_responses::OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE
            {
                return Err(CodexWebSocketStreamError::ConnectionLimit(message));
            }
            return Err(CodexWebSocketStreamError::non_transport(message));
        }
        if !*websocket_started {
            *websocket_started = true;
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        processor
            .process_event(event, output, sender, model)
            .map_err(|error| {
                if matches!(event_type.as_str(), "error" | "response.failed") {
                    CodexWebSocketStreamError::non_transport(error)
                } else {
                    CodexWebSocketStreamError::transport(error)
                }
            })?;
        if processor.is_terminal() {
            let should_cache =
                !matches!(output.stop_reason, StopReason::Error | StopReason::Aborted);
            finish_openai_responses_processor(processor, sender, output);
            if let Some(session_id) = session_id.filter(|_| should_cache) {
                let continuation = if use_cached_context {
                    build_openai_codex_cached_websocket_continuation(model, payload.clone(), output)
                } else {
                    continuation
                };
                codex_ws_cache().lock().await.insert(
                    session_id,
                    CachedCodexWebSocket {
                        socket,
                        continuation,
                        created_at_ms: connection_created_at_ms,
                        last_used_at_ms: now_millis() as i64,
                    },
                );
            } else {
                let _ = socket.close().await;
            }
            return Ok(());
        }
    }
}

async fn stream_openai_completions_sse_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let request = build_json_request(model, options, url, headers, payload)?;
    let Some(response) = send_with_sdk_retries(request, options, sender, output).await? else {
        return Ok(());
    };
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = retry_after_hint_ms(&response);
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(append_retry_after_hint(
            openai_completions_provider_error_from_body(status.as_u16(), &body),
            retry_after,
        ));
    }

    sender.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor = OpenAICompletionsStreamProcessor::started();
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events);
        for event in events.drain(..) {
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
            processor.process_chunk(event, output, sender, model)?;
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
        }
    }

    parser.finish(&mut events);
    for event in events.drain(..) {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        processor.process_chunk(event, output, sender, model)?;
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    processor.finish(output, sender)
}

async fn stream_openai_responses_sse_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let request = build_json_request(model, options, url, headers, payload)?;
    let Some(response) = send_with_sdk_retries(request, options, sender, output).await? else {
        return Ok(());
    };
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = retry_after_hint_ms(&response);
        let body = response.text().await.map_err(|error| error.to_string())?;
        let prefix = if model.api == "azure-openai-responses" {
            "Azure OpenAI API error"
        } else {
            "OpenAI API error"
        };
        return Err(append_retry_after_hint(
            openai_provider_error_from_body(status.as_u16(), &body, Some(prefix)),
            retry_after,
        ));
    }

    stream_openai_responses_sse_response(model, options, response, sender, output).await
}

async fn stream_openai_codex_sse_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    // Compress the request body once for every retry of the SSE path; the
    // Codex backend decodes `Content-Encoding: zstd` (pi 0ac3cfe0).
    let body_json = payload.to_string();
    let compressed_body =
        crate::openai_codex_responses::compress_openai_codex_request_body(&body_json);
    // The stream timeout only guards SSE response-header arrival (pi
    // 493efd42); a whole-request reqwest timeout would kill long-lived
    // streaming bodies, so strip it from the request builder.
    let request_options = {
        let mut request_options = options.clone();
        request_options.stream.timeout_ms = None;
        request_options
    };
    let header_timeout_ms = options
        .stream
        .timeout_ms
        .filter(|timeout_ms| *timeout_ms > 0);
    let mut attempt = 0usize;
    loop {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }

        let request = match &compressed_body {
            Some(compressed) => {
                let mut request =
                    build_json_request(model, &request_options, url, headers, payload.clone())?;
                request = request
                    .header("content-encoding", "zstd")
                    .body(compressed.clone());
                request
            }
            None => build_json_request(model, &request_options, url, headers, payload.clone())?,
        };
        // The user abort flag cancels the in-flight header wait (pi combines
        // the user signal with the header-timeout signal in the fetch).
        let send_result = tokio::select! {
            result = async {
                match header_timeout_ms {
                    Some(timeout_ms) => {
                        match tokio::time::timeout(
                            std::time::Duration::from_millis(timeout_ms),
                            request.send(),
                        )
                        .await
                        {
                            Ok(result) => result.map_err(|error| error.to_string()),
                            Err(_) => Err(format!(
                                "Codex SSE response headers timed out after {timeout_ms}ms"
                            )),
                        }
                    }
                    None => request.send().await.map_err(|error| error.to_string()),
                }
            } => result,
            _ = wait_for_abort_flag(options) => {
                push_abort_if_requested(sender, options, output);
                return Ok(());
            }
        };
        let response = match send_result {
            Ok(response) => response,
            Err(error) => {
                let max_retries = openai_codex_max_retries(options);
                if attempt >= max_retries || error.contains("usage limit") {
                    return Err(error);
                }
                // Network backoff is never capped by maxRetryDelayMs (pi
                // caps only 429 retry-after delays).
                let delay_ms = openai_codex_network_retry_delay_ms(attempt);
                if !sleep_observing_abort(delay_ms, options).await {
                    push_abort_if_requested(sender, options, output);
                    return Ok(());
                }
                attempt += 1;
                continue;
            }
        };
        emit_simple_response_hooks(model, options, &response).await?;
        let status = response.status();
        if status.is_success() {
            return stream_openai_responses_sse_response(model, options, response, sender, output)
                .await;
        }

        let status_text = status.canonical_reason().unwrap_or_default().to_owned();
        let status = status.as_u16();
        let retry_after_ms = response
            .headers()
            .get("retry-after-ms")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = response.text().await.map_err(|error| error.to_string())?;
        let max_retries = openai_codex_max_retries(options);
        // Retryability (including the terminal 429 rate-limit gate) tests the
        // RAW response body; the friendly message is only for the final error.
        let Some(delay_ms) = openai_codex_retry_delay_ms_with_limits(
            status,
            &body,
            retry_after_ms.as_deref(),
            retry_after.as_deref(),
            attempt,
            now_millis(),
            max_retries,
            options.stream.max_retry_delay_ms,
        ) else {
            return Err(openai_codex_error_message_from_response(
                status,
                &status_text,
                &body,
                now_millis(),
            ));
        };

        if !sleep_observing_abort(delay_ms, options).await {
            push_abort_if_requested(sender, options, output);
            return Ok(());
        }
        attempt += 1;
    }
}

const ABORT_FLAG_POLL_INTERVAL_MS: u64 = 10;

/// Resolves once the user abort flag is raised; pends forever when the
/// request has no abort flag.
async fn wait_for_abort_flag(options: &SimpleStreamOptions) {
    match options.stream.abort_flag.as_ref() {
        Some(abort_flag) => {
            while !abort_flag.load(Ordering::SeqCst) {
                tokio::time::sleep(std::time::Duration::from_millis(
                    ABORT_FLAG_POLL_INTERVAL_MS,
                ))
                .await;
            }
        }
        None => std::future::pending::<()>().await,
    }
}

/// Provider-SDK retry loop for the paths where pi forwards
/// `options.maxRetries` to the OpenAI/Anthropic SDKs (anthropic-messages,
/// openai-completions, openai-responses, azure-openai-responses). Defaults to
/// a single attempt (`maxRetries` unset = 0), matching pi.
///
/// Returns `Ok(None)` when the abort flag fired during a retry wait; the
/// caller emits the abort event and stops, mirroring the codex loop.
async fn send_with_sdk_retries(
    request: reqwest::RequestBuilder,
    options: &SimpleStreamOptions,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<Option<reqwest::Response>, String> {
    fn header_value(response: &reqwest::Response, name: &str) -> Option<String> {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    }

    let max_retries = options.stream.max_retries.unwrap_or(0);
    let mut pending = Some(request);
    let mut attempt: u32 = 0;
    loop {
        let current = pending.take().ok_or("request builder consumed")?;
        // Keep a clone for the next attempt while retries remain; a
        // non-cloneable body (streaming) simply cannot be retried.
        let (attempt_request, next_request) = if attempt < max_retries {
            match current.try_clone() {
                Some(clone) => (clone, Some(current)),
                None => (current, None),
            }
        } else {
            (current, None)
        };
        let can_retry = next_request.is_some();

        let delay_ms = match attempt_request.send().await {
            Ok(response) => {
                let status = response.status();
                if status.is_success() || !can_retry {
                    return Ok(Some(response));
                }
                if !crate::retry::sdk_should_retry_status(
                    status.as_u16(),
                    header_value(&response, "x-should-retry").as_deref(),
                ) {
                    return Ok(Some(response));
                }
                crate::retry::sdk_retry_delay_ms(
                    attempt,
                    header_value(&response, "retry-after-ms").as_deref(),
                    header_value(&response, "retry-after").as_deref(),
                    now_millis(),
                    crate::retry::sdk_retry_jitter_sample(),
                )
            }
            Err(error) => {
                if !can_retry {
                    return Err(error.to_string());
                }
                // Connection failures and timeouts retry on the default
                // backoff (no response headers to consult).
                crate::retry::sdk_default_retry_delay_ms(
                    attempt,
                    crate::retry::sdk_retry_jitter_sample(),
                )
            }
        };

        if !sleep_observing_abort(delay_ms, options).await {
            push_abort_if_requested(sender, options, output);
            return Ok(None);
        }
        pending = next_request;
        attempt += 1;
    }
}

/// Sleep that observes the abort flag every 10ms, mirroring pi's abortable
/// retry sleeps; returns `false` when aborted before the delay elapsed.
async fn sleep_observing_abort(delay_ms: u64, options: &SimpleStreamOptions) -> bool {
    let mut remaining = delay_ms;
    loop {
        if options
            .stream
            .abort_flag
            .as_ref()
            .is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst))
        {
            return false;
        }
        if remaining == 0 {
            return true;
        }
        let step = remaining.min(ABORT_FLAG_POLL_INTERVAL_MS);
        tokio::time::sleep(std::time::Duration::from_millis(step)).await;
        remaining -= step;
    }
}

const DEFAULT_OPENAI_CODEX_WEBSOCKET_CONNECT_TIMEOUT_MS: u64 = 15_000;

fn openai_codex_websocket_connect_timeout_ms(options: &SimpleStreamOptions) -> u64 {
    options
        .stream
        .extra
        .get("websocketConnectTimeoutMs")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_OPENAI_CODEX_WEBSOCKET_CONNECT_TIMEOUT_MS)
}

/// A connect timeout of 0 disables the guard, matching pi's `connectTimeoutMs
/// > 0` check (pi be7d5cf5).
async fn connect_openai_codex_websocket(
    url: &str,
    headers: &BTreeMap<String, String>,
    connect_timeout_ms: u64,
) -> Result<OpenAICodexWebSocket, CodexWebSocketStreamError> {
    if connect_timeout_ms == 0 {
        return OpenAICodexWebSocket::connect(url, headers)
            .await
            .map_err(CodexWebSocketStreamError::transport);
    }
    match tokio::time::timeout(
        std::time::Duration::from_millis(connect_timeout_ms),
        OpenAICodexWebSocket::connect(url, headers),
    )
    .await
    {
        Ok(result) => result.map_err(CodexWebSocketStreamError::transport),
        Err(_) => Err(CodexWebSocketStreamError::transport(format!(
            "WebSocket connect timeout after {connect_timeout_ms}ms"
        ))),
    }
}

fn openai_codex_max_retries(options: &SimpleStreamOptions) -> usize {
    options
        .stream
        .max_retries
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(crate::openai_codex_responses::OPENAI_CODEX_DEFAULT_MAX_RETRIES)
}

fn openai_codex_network_retry_delay_ms(attempt: usize) -> u64 {
    let shift = attempt.min(63) as u32;
    let multiplier = 1u64.checked_shl(shift).unwrap_or(u64::MAX);
    crate::openai_codex_responses::OPENAI_CODEX_BASE_RETRY_DELAY_MS.saturating_mul(multiplier)
}

async fn stream_openai_responses_sse_response(
    model: &Model,
    options: &SimpleStreamOptions,
    response: reqwest::Response,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    sender.push(AssistantMessageEvent::Start {
        partial: output.clone(),
    });

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor = OpenAIResponsesStreamProcessor::with_request_service_tier(
        request_service_tier_for_usage(model, options),
    );
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events);
        for event in events.drain(..) {
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
            processor.process_event(event, output, sender, model)?;
            if processor.is_terminal() {
                finish_openai_responses_processor(processor, sender, output);
                return Ok(());
            }
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
        }
    }

    parser.finish(&mut events);
    for event in events.drain(..) {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        processor.process_event(event, output, sender, model)?;
        if processor.is_terminal() {
            finish_openai_responses_processor(processor, sender, output);
            return Ok(());
        }
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    if !processor.is_terminal() {
        return Err("OpenAI Responses stream ended before a terminal response event".to_owned());
    }
    finish_openai_responses_processor(processor, sender, output);
    Ok(())
}

fn finish_openai_responses_processor(
    processor: OpenAIResponsesStreamProcessor,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) {
    if matches!(output.stop_reason, StopReason::Error | StopReason::Aborted) {
        let reason = output.stop_reason;
        let error = output
            .error_message
            .clone()
            .unwrap_or_else(|| "An unknown error occurred".to_owned());
        push_provider_error(sender, output, reason, error);
        return;
    }
    processor.finish(output, sender);
}

fn request_service_tier_for_usage(model: &Model, options: &SimpleStreamOptions) -> Option<String> {
    if !matches!(
        model.api.as_str(),
        "openai-responses" | "openai-codex-responses"
    ) {
        return None;
    }
    options
        .stream
        .extra
        .get("serviceTier")
        .and_then(Value::as_str)
        .map(str::to_owned)
}

async fn stream_google_sse_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let mut request_headers = headers.clone();
    if model.api == "google-vertex"
        && !headers_contain(&request_headers, "x-goog-api-key")
        && !headers_contain(&request_headers, "authorization")
    {
        let token = resolve_google_vertex_adc_access_token()
            .await?
            .ok_or_else(|| {
                "Vertex AI HTTP provider requires GOOGLE_CLOUD_API_KEY, an api_key option, an Authorization header, or Google ADC credentials"
                    .to_owned()
            })?;
        request_headers.insert("authorization".to_owned(), format!("Bearer {token}"));
    }
    let request = build_json_request_without_default_auth(options, url, &request_headers, payload)?;
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let json_content_type = response
            .headers()
            .get("content-type")
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.contains("application/json"));
        let status_text = status.canonical_reason().unwrap_or_default().to_owned();
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(google_provider_error_from_body(
            status.as_u16(),
            &status_text,
            json_content_type,
            &body,
        ));
    }

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor = GoogleStreamProcessor::new();
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events);
        for event in events.drain(..) {
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
            processor.process_chunk(event, output, sender, model);
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
        }
    }

    parser.finish(&mut events);
    for event in events.drain(..) {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        processor.process_chunk(event, output, sender, model);
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    processor.finish(output, sender)
}

async fn stream_bedrock_eventstream_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    config: &BedrockClientConfig,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let body = serde_json::to_vec(&payload).map_err(|error| error.to_string())?;
    let mut request_headers = headers.clone();
    if !headers_contain(&request_headers, "authorization") {
        let credentials = resolve_aws_credentials_with_runtime(config.profile.as_deref())
            .await?
            .ok_or_else(|| {
            "Bedrock HTTP provider requires AWS credentials from AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, AWS_WEB_IDENTITY_TOKEN_FILE/AWS_ROLE_ARN, AWS_CONTAINER_CREDENTIALS_RELATIVE_URI/FULL_URI, AWS_BEARER_TOKEN_BEDROCK, an api_key option, or an Authorization header"
                .to_owned()
        })?;
        let region = bedrock_signing_region(config, url);
        sign_aws_sigv4_headers(
            "POST",
            url,
            "bedrock",
            &region,
            &mut request_headers,
            &body,
            &credentials,
            chrono::Utc::now(),
        )?;
    }
    let client = reqwest_client_for_target(url)?;
    let mut request = client.post(url).body(body);
    for (name, value) in &request_headers {
        request = request.header(name, value);
    }
    if let Some(timeout_ms) = options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms));
    }
    // Some custom endpoints require HTTP/1.1 instead of HTTP/2 (pi honors the
    // AWS_BEDROCK_FORCE_HTTP1 escape hatch by swapping the request handler).
    if crate::get_provider_env_value("AWS_BEDROCK_FORCE_HTTP1", &options.stream.env).as_deref()
        == Some("1")
    {
        request = request.version(reqwest::Version::HTTP_11);
    }
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        // The AWS exception-name prefix ("Throttling error: " etc.) keeps
        // downstream retry/overflow classification aligned with pi.
        let error_type_header = response
            .headers()
            .get("x-amzn-errortype")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(crate::bedrock::format_bedrock_http_error(
            status.as_u16(),
            error_type_header.as_deref(),
            &body,
        ));
    }

    let mut byte_stream = response.bytes_stream();
    let mut parser = AwsEventStreamJsonParser::default();
    let mut events = Vec::new();
    let mut processor = BedrockConverseStreamProcessor::new();
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events)?;
        for event in events.drain(..) {
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
            processor.process_event(event, output, sender, model)?;
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
        }
    }

    parser.finish(&mut events)?;
    for event in events.drain(..) {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        processor.process_event(event, output, sender, model)?;
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    processor.finish(output, sender)
}

async fn stream_anthropic_sse_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    tools: Vec<Tool>,
    use_claude_code_tool_names: bool,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let request = build_json_request_without_default_auth(options, url, headers, payload)?;
    let Some(response) = send_with_sdk_retries(request, options, sender, output).await? else {
        return Ok(());
    };
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let retry_after = retry_after_hint_ms(&response);
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(append_retry_after_hint(
            anthropic_provider_error_from_body(status.as_u16(), &body),
            retry_after,
        ));
    }

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor =
        AnthropicStreamProcessor::with_tool_name_options(tools, use_claude_code_tool_names);
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events);
        for event in events.drain(..) {
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
            processor.process_event(model, event, output, sender)?;
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
        }
    }

    parser.finish(&mut events);
    for event in events.drain(..) {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        processor.process_event(model, event, output, sender)?;
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    processor.finish(output, sender);
    Ok(())
}

async fn stream_mistral_sse_json(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let request = build_json_request(model, options, url, headers, payload)?;
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(format_mistral_http_error(status.as_u16(), &body));
    }

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor = MistralChatStreamProcessor::new();
    while let Some(chunk) = byte_stream.next().await {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        let chunk = chunk.map_err(|error| error.to_string())?;
        parser.push_bytes(&chunk, &mut events);
        for event in events.drain(..) {
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
            processor.process_chunk(event, output, sender, model)?;
            if push_abort_if_requested(sender, options, output) {
                return Ok(());
            }
        }
    }

    parser.finish(&mut events);
    for event in events.drain(..) {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }
        processor.process_chunk(event, output, sender, model)?;
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    processor.finish(output, sender)
}

fn codex_ws_cache() -> &'static tokio::sync::Mutex<BTreeMap<String, CachedCodexWebSocket>> {
    OPENAI_CODEX_WS_CACHE.get_or_init(|| tokio::sync::Mutex::new(BTreeMap::new()))
}

fn insert_object_field(value: &mut Value, key: &str, field: Value) {
    match value {
        Value::Object(object) => {
            object.insert(key.to_owned(), field);
        }
        _ => {
            let mut object = Map::new();
            object.insert(key.to_owned(), field);
            *value = Value::Object(object);
        }
    }
}

fn provider_transport_failure_diagnostic(
    configured_transport: Transport,
    fallback_transport: Option<&str>,
    error: String,
    events_emitted: bool,
    request_bytes: usize,
) -> AssistantMessageDiagnostic {
    let mut details = Map::new();
    details.insert(
        "configuredTransport".to_owned(),
        Value::String(transport_name(configured_transport).to_owned()),
    );
    if let Some(fallback_transport) = fallback_transport {
        details.insert(
            "fallbackTransport".to_owned(),
            Value::String(fallback_transport.to_owned()),
        );
    }
    details.insert("eventsEmitted".to_owned(), Value::Bool(events_emitted));
    details.insert(
        "phase".to_owned(),
        Value::String(
            if events_emitted {
                "after_message_stream_start"
            } else {
                "before_message_stream_start"
            }
            .to_owned(),
        ),
    );
    details.insert(
        "requestBytes".to_owned(),
        Value::Number(serde_json::Number::from(request_bytes)),
    );
    create_assistant_message_diagnostic(
        "provider_transport_failure",
        error,
        Some(Value::Object(details)),
    )
}

fn transport_name(transport: Transport) -> &'static str {
    match transport {
        Transport::Sse => "sse",
        Transport::Websocket => "websocket",
        Transport::WebsocketCached => "websocket-cached",
        Transport::Auto => "auto",
    }
}

fn build_json_request(
    model: &Model,
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
) -> Result<reqwest::RequestBuilder, String> {
    let client = reqwest_client_for_target(url)?;
    let mut request = client.post(url).json(&payload);
    let api_key = options
        .stream
        .api_key
        .clone()
        .or_else(|| get_env_api_key_scoped(&model.provider, &options.stream.env));
    for (name, value) in headers {
        request = request.header(name, value);
    }
    if model.provider == "cloudflare-ai-gateway" {
        if !headers_contain(headers, "cf-aig-authorization")
            && let Some(api_key) = api_key
        {
            request = request.header("cf-aig-authorization", format!("Bearer {api_key}"));
        }
    } else if !headers_contain(headers, "authorization")
        && !headers_contain(headers, "api-key")
        && !headers_contain(headers, "x-api-key")
        && let Some(api_key) = api_key
    {
        request = request.bearer_auth(api_key);
    }
    if let Some(timeout_ms) = options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms));
    }
    Ok(request)
}

fn build_json_request_without_default_auth(
    options: &SimpleStreamOptions,
    url: &str,
    headers: &BTreeMap<String, String>,
    payload: Value,
) -> Result<reqwest::RequestBuilder, String> {
    let client = reqwest_client_for_target(url)?;
    let mut request = client.post(url).json(&payload);
    for (name, value) in headers {
        request = request.header(name, value);
    }
    if let Some(timeout_ms) = options.stream.timeout_ms {
        request = request.timeout(std::time::Duration::from_millis(timeout_ms));
    }
    Ok(request)
}

async fn emit_simple_response_hooks(
    model: &Model,
    options: &SimpleStreamOptions,
    response: &reqwest::Response,
) -> Result<(), String> {
    options
        .emit_response_hooks(
            model,
            ProviderResponse {
                status: response.status().as_u16(),
                headers: response
                    .headers()
                    .iter()
                    .filter_map(|(name, value)| {
                        Some((name.as_str().to_owned(), value.to_str().ok()?.to_owned()))
                    })
                    .collect(),
            },
        )
        .await
}

fn resolved_model_base_url(model: &Model) -> Result<String, String> {
    if is_cloudflare_provider(&model.provider) {
        resolve_cloudflare_base_url(model)
    } else {
        Ok(model.base_url.clone())
    }
}

fn endpoint_url(base_url: &str, path: &str) -> String {
    format!("{}/{}", base_url.trim_end_matches('/'), path)
}

fn azure_openai_responses_url(base_url: &str, api_version: &str) -> String {
    format!(
        "{}?api-version={}",
        endpoint_url(base_url, "responses"),
        api_version
    )
}

fn mistral_chat_completions_url(base_url: &str) -> String {
    crate::mistral::mistral_chat_completions_url(base_url)
}

fn google_generative_ai_stream_url(base_url: &str, model_id: &str) -> String {
    let mut base_url = base_url.trim_end_matches('/').to_owned();
    if !base_url_path_has_version(&base_url) {
        base_url.push_str("/v1beta");
    }
    format!(
        "{}/models/{}:streamGenerateContent?alt=sse",
        base_url,
        url_encode_path_segment(model_id)
    )
}

fn google_vertex_stream_url(
    config: &GoogleVertexClientConfig,
    model: &Model,
) -> Result<String, ProviderError> {
    let escaped_model = url_encode_path_segment(&model.id);
    if let Some(http_options) = &config.http_options
        && !http_options.base_url.trim().is_empty()
    {
        let mut base_url = http_options
            .base_url
            .trim()
            .trim_end_matches('/')
            .to_owned();
        let base_includes_version =
            http_options.api_version.as_deref() == Some("") || base_url_path_has_version(&base_url);
        if !base_includes_version {
            base_url.push('/');
            base_url.push_str(&config.api_version);
        }
        return Ok(format!(
            "{base_url}/models/{escaped_model}:streamGenerateContent?alt=sse"
        ));
    }

    let project = config.project.as_deref().ok_or_else(|| {
        ProviderError::Provider(
            "Vertex AI requires a project ID for the default endpoint".to_owned(),
        )
    })?;
    let location = config.location.as_deref().ok_or_else(|| {
        ProviderError::Provider("Vertex AI requires a location for the default endpoint".to_owned())
    })?;
    let base_url = model
        .base_url
        .replace("{location}", location)
        .trim_end_matches('/')
        .to_owned();
    Ok(format!(
        "{base_url}/{}/projects/{project}/locations/{location}/publishers/google/models/{escaped_model}:streamGenerateContent?alt=sse",
        config.api_version
    ))
}

fn bedrock_converse_stream_url(base_url: &str, model_id: &str) -> String {
    format!(
        "{}/model/{}/converse-stream",
        base_url.trim_end_matches('/'),
        url_encode_path_segment(model_id)
    )
}

fn bedrock_signing_region(config: &BedrockClientConfig, url: &str) -> String {
    config
        .region
        .clone()
        .or_else(|| resolve_aws_profile_region(config.profile.as_deref()))
        .or_else(|| standard_bedrock_endpoint_region(url))
        .unwrap_or_else(|| "us-east-1".to_owned())
}

fn base_url_path_has_version(base_url: &str) -> bool {
    let path = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url)
        .split_once('/')
        .map(|(_, path)| path)
        .unwrap_or_default();
    path.split('/').any(|part| {
        let Some(rest) = part.strip_prefix('v') else {
            return false;
        };
        let digits = rest
            .chars()
            .take_while(|ch| ch.is_ascii_digit())
            .collect::<String>();
        !digits.is_empty()
            && rest[digits.len()..]
                .chars()
                .all(|ch| ch.is_ascii_alphabetic() || ch.is_ascii_digit())
    })
}

fn url_encode_path_segment(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        let allowed = byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~');
        if allowed {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn headers_contain(headers: &BTreeMap<String, String>, name: &str) -> bool {
    headers.keys().any(|key| key.eq_ignore_ascii_case(name))
}

/// Reserved `StreamOptions.extra` key listing header names whose JSON value
/// was `null`. pi models provider headers as `Record<string, string | null>`
/// where a null options header unsets the same-named model default header
/// (pi `providerHeadersToRecord`); ri's typed `BTreeMap<String, String>`
/// cannot hold nulls, so [`stream_options_from_json`] strips them into this
/// marker for the merge boundary to consume.
pub const NULL_PROVIDER_HEADERS_EXTRA_KEY: &str = "__nullProviderHeaders";

/// Pre-filter raw JSON options before `StreamOptions` deserialization:
/// null-valued headers are removed (instead of failing deserialization) and
/// recorded under [`NULL_PROVIDER_HEADERS_EXTRA_KEY`].
pub fn strip_null_provider_headers_json(options_json: &mut Value) {
    let Some(object) = options_json.as_object_mut() else {
        return;
    };
    let mut removed = Vec::new();
    if let Some(headers) = object.get_mut("headers").and_then(Value::as_object_mut) {
        headers.retain(|name, value| {
            if value.is_null() {
                removed.push(Value::String(name.clone()));
                false
            } else {
                true
            }
        });
    }
    if !removed.is_empty() {
        object.insert(
            NULL_PROVIDER_HEADERS_EXTRA_KEY.to_owned(),
            Value::Array(removed),
        );
    }
}

/// Deserialize `StreamOptions` from raw JSON, tolerating pi-style null header
/// values (which unset model default headers rather than erroring).
pub fn stream_options_from_json(mut value: Value) -> Result<StreamOptions, String> {
    strip_null_provider_headers_json(&mut value);
    serde_json::from_value(value).map_err(|error| error.to_string())
}

/// Deserialize `SimpleStreamOptions` from raw JSON, tolerating pi-style null
/// header values.
pub fn simple_stream_options_from_json(mut value: Value) -> Result<SimpleStreamOptions, String> {
    strip_null_provider_headers_json(&mut value);
    serde_json::from_value(value).map_err(|error| error.to_string())
}

fn null_provider_header_names(options: &StreamOptions) -> Vec<&str> {
    options
        .extra
        .get(NULL_PROVIDER_HEADERS_EXTRA_KEY)
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default()
}

/// Remove headers that a null-valued options header unsets
/// (pi `providerHeadersToRecord`). Names compare case-insensitively, matching
/// the `Headers`-based providers.
pub fn remove_null_provider_headers(
    headers: &mut BTreeMap<String, String>,
    options: &StreamOptions,
) {
    for name in null_provider_header_names(options) {
        let keys = headers
            .keys()
            .filter(|key| key.eq_ignore_ascii_case(name))
            .cloned()
            .collect::<Vec<_>>();
        for key in keys {
            headers.remove(&key);
        }
    }
}

#[cfg(test)]
fn parse_sse_json_body(body: &str) -> Vec<Value> {
    let mut events = Vec::new();
    let mut parser = SseJsonParser::default();
    parser.push_bytes(body.as_bytes(), &mut events);
    parser.finish(&mut events);
    events
}

const UTF8_BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

#[derive(Debug, Default)]
struct SseJsonParser {
    line_buffer: Vec<u8>,
    data_lines: Vec<String>,
    /// Leading UTF-8 BOM handling (pi's TextDecoder strips a single BOM at
    /// the start of the stream before line parsing).
    bom_checked: bool,
    bom_matched: u8,
}

impl SseJsonParser {
    fn push_bytes(&mut self, bytes: &[u8], events: &mut Vec<Value>) {
        let bytes = self.strip_leading_bom(bytes);
        for byte in bytes {
            if *byte == b'\n' {
                self.push_line(events);
            } else {
                self.line_buffer.push(*byte);
            }
        }
    }

    /// Consume one leading UTF-8 BOM, tolerating a BOM split across chunks;
    /// a partial match that turns out not to be a BOM is replayed as content
    /// (BOM bytes are never `\n`, so the line buffer is safe to extend).
    fn strip_leading_bom<'bytes>(&mut self, bytes: &'bytes [u8]) -> &'bytes [u8] {
        if self.bom_checked {
            return bytes;
        }
        let mut offset = 0;
        while offset < bytes.len() {
            if bytes[offset] == UTF8_BOM[self.bom_matched as usize] {
                self.bom_matched += 1;
                offset += 1;
                if self.bom_matched as usize == UTF8_BOM.len() {
                    self.bom_checked = true;
                    return &bytes[offset..];
                }
            } else {
                self.bom_checked = true;
                self.line_buffer
                    .extend_from_slice(&UTF8_BOM[..self.bom_matched as usize]);
                return &bytes[offset..];
            }
        }
        // Chunk ended mid-BOM: wait for more bytes.
        &[]
    }

    fn finish(&mut self, events: &mut Vec<Value>) {
        if !self.line_buffer.is_empty() {
            self.push_line(events);
        }
        push_sse_event(events, &mut self.data_lines);
    }

    fn push_line(&mut self, events: &mut Vec<Value>) {
        if self.line_buffer.ends_with(b"\r") {
            self.line_buffer.pop();
        }
        let line = String::from_utf8_lossy(&self.line_buffer).into_owned();
        self.line_buffer.clear();
        if line.is_empty() {
            push_sse_event(events, &mut self.data_lines);
            return;
        }
        if let Some(data) = line.strip_prefix("data:") {
            self.data_lines.push(data.trim_start().to_owned());
        }
    }
}

#[derive(Debug, Default)]
struct AwsEventStreamJsonParser {
    buffer: Vec<u8>,
}

impl AwsEventStreamJsonParser {
    fn push_bytes(&mut self, bytes: &[u8], events: &mut Vec<Value>) -> Result<(), String> {
        self.buffer.extend_from_slice(bytes);
        self.drain_events(events)
    }

    fn finish(&mut self, events: &mut Vec<Value>) -> Result<(), String> {
        self.drain_events(events)?;
        if self.buffer.is_empty() {
            Ok(())
        } else {
            Err("Incomplete AWS EventStream frame".to_owned())
        }
    }

    fn drain_events(&mut self, events: &mut Vec<Value>) -> Result<(), String> {
        loop {
            if self.buffer.len() < 12 {
                return Ok(());
            }
            let total_len = u32::from_be_bytes([
                self.buffer[0],
                self.buffer[1],
                self.buffer[2],
                self.buffer[3],
            ]) as usize;
            let headers_len = u32::from_be_bytes([
                self.buffer[4],
                self.buffer[5],
                self.buffer[6],
                self.buffer[7],
            ]) as usize;
            if total_len < 16 || headers_len > total_len.saturating_sub(16) {
                return Err("Invalid AWS EventStream frame length".to_owned());
            }
            if self.buffer.len() < total_len {
                return Ok(());
            }
            let payload_start = 12 + headers_len;
            let payload_end = total_len - 4;
            let payload = &self.buffer[payload_start..payload_end];
            if !payload.is_empty() {
                let payload_text = String::from_utf8_lossy(payload);
                let value = parse_json_with_repair::<Value>(payload_text.as_ref())
                    .map_err(|error| format!("Could not parse AWS EventStream JSON: {error}"))?;
                events.push(value);
            }
            self.buffer.drain(..total_len);
        }
    }
}

fn push_sse_event(events: &mut Vec<Value>, data_lines: &mut Vec<String>) {
    if data_lines.is_empty() {
        return;
    }
    let data = data_lines.join("\n");
    data_lines.clear();
    if data.trim() == "[DONE]" {
        return;
    }
    if let Ok(value) = parse_json_with_repair::<Value>(&data) {
        events.push(value);
    }
}

pub const MAX_PROVIDER_ERROR_BODY_CHARS: usize = 4000;

/// pi `truncateErrorText` counts UTF-16 code units (JS string length); the
/// reported truncated-character count uses the same arithmetic.
pub fn truncate_provider_error_text(text: &str, max_chars: usize) -> String {
    let total_units: usize = text.chars().map(char::len_utf16).sum();
    if total_units <= max_chars {
        return text.to_owned();
    }
    let mut used = 0usize;
    let mut truncated = String::new();
    for ch in text.chars() {
        let units = ch.len_utf16();
        if used + units > max_chars {
            break;
        }
        used += units;
        truncated.push(ch);
    }
    format!(
        "{truncated}... [truncated {} chars]",
        total_units - max_chars
    )
}

/// JS truthiness for parsed JSON values (`null`, `false`, `0`, and `""` are
/// falsy); pi's SDK error composition branches on it.
fn js_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_none_or(|number| number != 0.0),
        Value::String(text) => !text.is_empty(),
        _ => true,
    }
}

/// The openai/anthropic SDK `APIError.makeMessage(status, error, message)`:
/// `error.message` when present, otherwise the JSON of `error`, otherwise the
/// raw text; composed as `"<status> <msg>"` or
/// `"<status> status code (no body)"`.
fn sdk_api_error_message(status: u16, error_field: Option<&Value>, raw_message: &str) -> String {
    let msg = match error_field {
        Some(field) if js_truthy(field) => match field.get("message") {
            Some(message) if js_truthy(message) => match message {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            },
            _ => field.to_string(),
        },
        _ => raw_message.to_owned(),
    };
    if msg.is_empty() {
        format!("{status} status code (no body)")
    } else {
        format!("{status} {msg}")
    }
}

/// Gateway retry hint from a non-2xx response's headers, resolved to a
/// positive delay in ms (`retry-after-ms` wins, then `retry-after` as seconds
/// or an HTTP date). Must be read before the body consumes the response.
fn retry_after_hint_ms(response: &reqwest::Response) -> Option<u64> {
    let header = |name: &str| {
        response
            .headers()
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned)
    };
    let retry_after_ms = header("retry-after-ms");
    let retry_after = header("retry-after");
    crate::retry::sdk_retry_after_delay_ms(
        retry_after_ms.as_deref(),
        retry_after.as_deref(),
        now_millis() as i64,
    )
    .filter(|delay_ms| *delay_ms > 0.0)
    .map(|delay_ms| delay_ms as u64)
}

/// Provider errors cross API boundaries as plain strings, so the retry hint
/// rides inside the message where outer retry loops can parse it back out
/// (see `parse_retry_after_hint_ms`) instead of falling back to blind backoff.
fn append_retry_after_hint(message: String, hint_ms: Option<u64>) -> String {
    match hint_ms {
        Some(delay_ms) => format!("{message} (retry-after-ms: {delay_ms})"),
        None => message,
    }
}

/// Mirrors pi's Anthropic SDK error surface (anthropic-messages.ts:752 keeps
/// `error.message` verbatim): the SDK folds the WHOLE parsed body into
/// `"<status> <json>"`, an empty body yields
/// `"<status> status code (no body)"`, and a non-JSON body passes through.
pub fn anthropic_provider_error_from_body(status: u16, body: &str) -> String {
    // Strict JSON.parse like the SDK; JS-falsy parses behave as unparsed.
    let parsed = serde_json::from_str::<Value>(body).ok().filter(js_truthy);
    sdk_api_error_message(status, parsed.as_ref(), body)
}

/// Mirrors pi's openai SDK error + `formatProviderError`: the SDK extracts
/// the parsed body's `error` field into `"<status> <msg>"`, and
/// `normalizeProviderError` surfaces the field's JSON (truncated, empty
/// objects excluded) when the message does not already carry it.
pub fn openai_provider_error_from_body(status: u16, body: &str, prefix: Option<&str>) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok().filter(js_truthy);
    let error_field = parsed.as_ref().and_then(|value| value.get("error"));
    // JS `errMessage` is the raw text only when the body did not parse.
    let raw_message = if parsed.is_some() { "" } else { body };
    let sdk_message = sdk_api_error_message(status, error_field, raw_message);
    // pi isNonEmptyObject: empty parsed bodies never surface as "{}".
    let body_text = error_field.and_then(|field| match field {
        Value::Object(object) if !object.is_empty() => Some(field.to_string()),
        Value::Array(items) if !items.is_empty() => Some(field.to_string()),
        _ => None,
    });
    let body_text =
        body_text.map(|text| truncate_provider_error_text(&text, MAX_PROVIDER_ERROR_BODY_CHARS));
    match body_text {
        Some(body_text) if !sdk_message.contains(&body_text) => match prefix {
            Some(prefix) => format!("{prefix} ({status}): {body_text}"),
            None => format!("{status}: {body_text}"),
        },
        _ => match prefix {
            Some(prefix) => format!("{prefix} ({status}): {sdk_message}"),
            None => sdk_message,
        },
    }
}

pub fn openai_completions_provider_error_from_body(status: u16, body: &str) -> String {
    let mut message = openai_provider_error_from_body(status, body, None);
    // OpenRouter tucks the upstream reason under error.metadata.raw; append it
    // only when the surfaced body does not already carry it (pi dedups,
    // coercing scalar raw values like JS template interpolation).
    if let Some(raw) = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|value| {
            value
                .pointer("/error/metadata/raw")
                .and_then(|raw| match raw {
                    Value::String(text) => Some(text.clone()),
                    Value::Number(number) => Some(number.to_string()),
                    Value::Bool(flag) => Some(flag.to_string()),
                    _ => None,
                })
        })
        .filter(|raw| !raw.is_empty())
        && !message.contains(&raw)
    {
        message.push('\n');
        message.push_str(&raw);
    }
    message
}

/// Mirrors @google/genai `throwErrorIfNotOK` (pi surfaces the SDK message
/// unchanged via `formatProviderError`): JSON responses re-serialize the
/// whole body, other responses wrap the raw text with the status/statusText.
pub fn google_provider_error_from_body(
    status: u16,
    status_text: &str,
    json_content_type: bool,
    body: &str,
) -> String {
    if json_content_type {
        return match serde_json::from_str::<Value>(body) {
            Ok(value) => value.to_string(),
            // `response.json()` rejection surfaces as the parse error text.
            Err(error) => error.to_string(),
        };
    }
    serde_json::json!({
        "error": { "message": body, "code": status, "status": status_text },
    })
    .to_string()
}

fn push_abort_if_requested(
    sender: &crate::AssistantMessageEventSender,
    options: &SimpleStreamOptions,
    output: &mut AssistantMessage,
) -> bool {
    // A dropped consumer can never observe this stream again; keeping the
    // socket open only pins an upstream concurrency slot, so treat it like an
    // abort even without an abort_flag.
    if sender.is_abandoned() {
        push_provider_error(
            sender,
            output,
            StopReason::Aborted,
            "Request was aborted: stream consumer dropped".to_owned(),
        );
        return true;
    }
    if !options
        .stream
        .abort_flag
        .as_ref()
        .is_some_and(|abort_flag| abort_flag.load(Ordering::SeqCst))
    {
        return false;
    }

    push_provider_error(
        sender,
        output,
        StopReason::Aborted,
        "Request was aborted".to_owned(),
    );
    true
}

fn push_provider_error(
    sender: &crate::AssistantMessageEventSender,
    output: &mut AssistantMessage,
    reason: StopReason,
    error: String,
) {
    output.stop_reason = reason;
    output.error_message = Some(error);
    output.timestamp = now_millis();
    sender.push(AssistantMessageEvent::Error {
        reason,
        error: output.clone(),
    });
}

fn processor_already_pushed_error(output: &AssistantMessage, error: &str) -> bool {
    output.stop_reason == StopReason::Error && output.error_message.as_deref() == Some(error)
}

fn empty_assistant_message(model: &Model) -> AssistantMessage {
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
        timestamp: now_millis(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::parse_retry_after_hint_ms;

    #[test]
    fn parses_sse_json_body_and_ignores_done() {
        let body = "event: message\ndata: {\"a\":1}\n\ndata: [DONE]\n\n";
        assert_eq!(
            parse_sse_json_body(body),
            vec![serde_json::json!({ "a": 1 })]
        );
    }

    #[test]
    fn retry_after_hint_round_trips_through_error_text() {
        let message = append_retry_after_hint("429: busy".to_owned(), Some(12_500));
        assert_eq!(message, "429: busy (retry-after-ms: 12500)");
        assert_eq!(parse_retry_after_hint_ms(&message), Some(12_500));
        // Outer layers may wrap the message; trailing text must not break it.
        let wrapped = format!("transient LLM error: {message} [attempt 3]");
        assert_eq!(parse_retry_after_hint_ms(&wrapped), Some(12_500));
        assert_eq!(parse_retry_after_hint_ms("429: busy"), None);
        assert_eq!(
            append_retry_after_hint("500: oops".to_owned(), None),
            "500: oops"
        );
    }

    #[test]
    fn abandoned_stream_consumer_aborts_the_request_loop() {
        let (sender, stream) = crate::assistant_message_event_stream();
        assert!(!sender.is_abandoned());
        drop(stream);
        assert!(sender.is_abandoned());

        let options = SimpleStreamOptions::default();
        let mut output = empty_assistant_message(&Model {
            id: "test-model".to_owned(),
            name: "test-model".to_owned(),
            api: "openai-completions".to_owned(),
            provider: "test".to_owned(),
            base_url: "https://example.invalid".to_owned(),
            reasoning: false,
            thinking_level_map: BTreeMap::new(),
            input: Vec::new(),
            cost: Default::default(),
            context_window: 0,
            max_tokens: 0,
            headers: BTreeMap::new(),
            compat: None,
        });
        assert!(push_abort_if_requested(&sender, &options, &mut output));
        assert_eq!(output.stop_reason, StopReason::Aborted);
    }
}
