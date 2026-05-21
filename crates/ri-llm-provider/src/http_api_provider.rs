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
    get_env_api_key,
    google_shared::{GoogleStreamProcessor, build_google_simple_payload},
    google_vertex::{
        GoogleVertexClientConfig, GoogleVertexOptions, resolve_google_vertex_adc_access_token,
        resolve_google_vertex_client_config,
    },
    json_repair::parse_json_with_repair,
    mistral::{
        MistralChatStreamProcessor, build_mistral_request_headers, build_mistral_simple_payload,
    },
    node_http_proxy::reqwest_client_for_target,
    openai_codex_responses::{
        OpenAICodexCachedWebSocketContinuation, OpenAICodexResponsesPayloadOptions,
        OpenAICodexWebSocket, build_openai_codex_cached_websocket_continuation,
        build_openai_codex_cached_websocket_request_body, build_openai_codex_responses_payload,
        build_openai_codex_sse_headers, build_openai_codex_websocket_headers,
        extract_openai_codex_account_id, openai_codex_retry_delay_ms_with_limits,
        resolve_openai_codex_url, resolve_openai_codex_websocket_url,
    },
    openai_completions::{
        OpenAICompletionsPayloadOptions, OpenAICompletionsStreamProcessor,
        build_openai_completions_default_headers_with_context, build_openai_completions_payload,
        resolve_openai_completions_cache_retention,
    },
    openai_responses::{
        OpenAIResponsesPayloadOptions, OpenAIResponsesStreamProcessor,
        build_openai_responses_default_headers_with_context, build_openai_responses_payload,
        resolve_openai_responses_cache_retention,
    },
    types::{
        AssistantMessage, AssistantMessageEvent, Context, Model, ProviderResponse,
        SimpleStreamOptions, StopReason, Tool, Transport, Usage, now_millis,
    },
};
use futures::StreamExt;
use serde_json::{Map, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock, atomic::Ordering},
};

static OPENAI_CODEX_WS_CACHE: OnceLock<tokio::sync::Mutex<BTreeMap<String, CachedCodexWebSocket>>> =
    OnceLock::new();
static OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS: OnceLock<tokio::sync::Mutex<BTreeSet<String>>> =
    OnceLock::new();
const BUILTIN_API_PROVIDER_SOURCE_ID: &str = "builtin-http";

struct CachedCodexWebSocket {
    socket: OpenAICodexWebSocket,
    continuation: Option<OpenAICodexCachedWebSocketContinuation>,
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
}

fn register_builtin_api_provider_if_missing(api: &str, provider: Arc<dyn ApiProvider>) {
    if get_api_provider(api).is_none() {
        register_builtin_api_provider(provider);
    }
}

fn register_builtin_api_provider(provider: Arc<dyn ApiProvider>) {
    register_api_provider(provider, Some(BUILTIN_API_PROVIDER_SOURCE_ID.to_owned()));
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
            resolve_openai_completions_cache_retention(options.stream.cache_retention);
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
        let headers = build_openai_completions_default_headers_with_context(
            model,
            Some(&context),
            options.stream.session_id.as_deref(),
            cache_retention,
            &options.stream.headers,
        );
        spawn_openai_completions_sse_request(
            model.clone(),
            options.clone(),
            endpoint_url(&model.base_url, "chat/completions"),
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
            resolve_openai_responses_cache_retention(options.stream.cache_retention);
        let payload = build_openai_responses_payload(
            model,
            &context,
            OpenAIResponsesPayloadOptions {
                cache_retention: Some(cache_retention),
                session_id: options.stream.session_id.clone(),
                max_tokens: options.stream.max_tokens,
                temperature: options.stream.temperature,
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
        let headers = build_openai_responses_default_headers_with_context(
            model,
            Some(&context),
            options.stream.session_id.as_deref(),
            cache_retention,
            &options.stream.headers,
        );
        spawn_openai_responses_sse_request(
            model.clone(),
            options.clone(),
            endpoint_url(&model.base_url, "responses"),
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
        let headers = build_mistral_request_headers(
            model,
            options.stream.session_id.as_deref(),
            &options.stream.headers,
        );
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
        if !headers_contain(&headers, "api-key")
            && let Some(api_key) = options
                .stream
                .api_key
                .clone()
                .or_else(|| get_env_api_key(&model.provider))
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
            .or_else(|| get_env_api_key(&model.provider))
            .unwrap_or_default();
        let config = build_anthropic_client_config(
            model,
            &context,
            AnthropicClientOptions {
                api_key,
                headers: options.stream.headers.clone(),
                session_id: options.stream.session_id.clone(),
                cache_retention: options.stream.cache_retention,
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
        spawn_anthropic_sse_request(
            model.clone(),
            options,
            endpoint_url(&config.base_url, "messages"),
            headers,
            payload,
            context.tools.clone(),
            config.is_oauth_token,
        )
    }
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
        if !headers_contain(&headers, "x-goog-api-key") {
            let api_key = options
                .stream
                .api_key
                .clone()
                .or_else(|| get_env_api_key(&model.provider))
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
                    .or_else(|| get_env_api_key(&model.provider)),
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
            .or_else(|| get_env_api_key(&model.provider))
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
        let headers = build_openai_codex_sse_headers(
            &model.headers,
            &options.stream.headers,
            &account_id,
            &token,
            options.stream.session_id.as_deref(),
        );
        let websocket_request_id = options
            .stream
            .session_id
            .clone()
            .unwrap_or_else(|| format!("ri-codex-{}", now_millis()));
        let websocket_headers = build_openai_codex_websocket_headers(
            &model.headers,
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
        let config = resolve_bedrock_client_config(
            model,
            BedrockClientOptions {
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
                cache_retention: options.stream.cache_retention,
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
        headers.extend(options.stream.headers.clone());
        headers
            .entry("accept".to_owned())
            .or_insert_with(|| "application/vnd.amazon.eventstream".to_owned());
        headers
            .entry("content-type".to_owned())
            .or_insert_with(|| "application/json".to_owned());
        let bearer = (std::env::var("AWS_BEDROCK_SKIP_AUTH").ok().as_deref() != Some("1"))
            .then(|| {
                options
                    .stream
                    .api_key
                    .clone()
                    .or_else(|| std::env::var("AWS_BEARER_TOKEN_BEDROCK").ok())
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
            && codex_ws_sse_fallback_active(session_id.as_deref()).await;
        if transport != Transport::Sse && !websocket_disabled_for_session {
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
                Err(error) if websocket_started => {
                    record_codex_ws_failure(session_id.as_deref()).await;
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
                    record_codex_ws_failure(session_id.as_deref()).await;
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
) -> Result<(), String> {
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }

    let transport = options.stream.transport.unwrap_or(Transport::Auto);
    let use_cached_context = matches!(transport, Transport::Auto | Transport::WebsocketCached);
    let session_id = options.stream.session_id.clone();
    let cached = if use_cached_context {
        if let Some(session_id) = session_id.as_deref() {
            codex_ws_cache().lock().await.remove(session_id)
        } else {
            None
        }
    } else {
        None
    };
    let (mut socket, continuation) = if let Some(cached) = cached {
        (cached.socket, cached.continuation)
    } else {
        (OpenAICodexWebSocket::connect(url, headers).await?, None)
    };

    let cached_request = if use_cached_context {
        build_openai_codex_cached_websocket_request_body(payload, continuation.as_ref())
    } else {
        build_openai_codex_cached_websocket_request_body(payload, None)
    };
    let mut request_body = cached_request.body;
    insert_object_field(
        &mut request_body,
        "type",
        Value::String("response.create".to_owned()),
    );
    socket.send_json_text(&request_body).await?;

    let mut processor = OpenAIResponsesStreamProcessor::new();
    loop {
        if push_abort_if_requested(sender, options, output) {
            let _ = socket.close().await;
            return Ok(());
        }
        let event = socket
            .read_json_text()
            .await?
            .ok_or_else(|| "WebSocket stream closed before response.completed".to_owned())?;
        if !*websocket_started {
            *websocket_started = true;
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
        }
        processor.process_event(event, output, sender, model)?;
        if processor.is_terminal() {
            processor.finish(output, sender);
            if use_cached_context {
                if let Some(session_id) = session_id {
                    let continuation = build_openai_codex_cached_websocket_continuation(
                        model,
                        payload.clone(),
                        output,
                    );
                    codex_ws_cache().lock().await.insert(
                        session_id,
                        CachedCodexWebSocket {
                            socket,
                            continuation,
                        },
                    );
                } else {
                    let _ = socket.close().await;
                }
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
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(provider_error_from_body(status.as_u16(), &body));
    }

    let mut byte_stream = response.bytes_stream();
    let mut parser = SseJsonParser::default();
    let mut events = Vec::new();
    let mut processor = OpenAICompletionsStreamProcessor::new();
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
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(provider_error_from_body(status.as_u16(), &body));
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
    let mut attempt = 0usize;
    loop {
        if push_abort_if_requested(sender, options, output) {
            return Ok(());
        }

        let request = build_json_request(model, options, url, headers, payload.clone())?;
        let response = request.send().await.map_err(|error| error.to_string())?;
        emit_simple_response_hooks(model, options, &response).await?;
        let status = response.status();
        if status.is_success() {
            return stream_openai_responses_sse_response(model, options, response, sender, output)
                .await;
        }

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
        let error = provider_error_from_body(status, &body);
        let max_retries = options
            .stream
            .max_retries
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or(crate::openai_codex_responses::OPENAI_CODEX_MAX_RETRIES);
        let Some(delay_ms) = openai_codex_retry_delay_ms_with_limits(
            status,
            &error,
            retry_after_ms.as_deref(),
            retry_after.as_deref(),
            attempt,
            now_millis(),
            max_retries,
            options.stream.max_retry_delay_ms,
        ) else {
            return Err(error);
        };

        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
        attempt += 1;
    }
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
    let mut processor = OpenAIResponsesStreamProcessor::new();
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
                processor.finish(output, sender);
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
            processor.finish(output, sender);
            return Ok(());
        }
    }
    if push_abort_if_requested(sender, options, output) {
        return Ok(());
    }
    processor.finish(output, sender);
    Ok(())
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
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(provider_error_from_body(status.as_u16(), &body));
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
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(provider_error_from_body(status.as_u16(), &body));
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
    let response = request.send().await.map_err(|error| error.to_string())?;
    emit_simple_response_hooks(model, options, &response).await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.map_err(|error| error.to_string())?;
        return Err(provider_error_from_body(status.as_u16(), &body));
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
            processor.process_event(event, output, sender)?;
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
        processor.process_event(event, output, sender)?;
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
        return Err(provider_error_from_body(status.as_u16(), &body));
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

fn codex_ws_sse_fallback_sessions() -> &'static tokio::sync::Mutex<BTreeSet<String>> {
    OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS.get_or_init(|| tokio::sync::Mutex::new(BTreeSet::new()))
}

async fn codex_ws_sse_fallback_active(session_id: Option<&str>) -> bool {
    let Some(session_id) = session_id else {
        return false;
    };
    codex_ws_sse_fallback_sessions()
        .lock()
        .await
        .contains(session_id)
}

async fn record_codex_ws_failure(session_id: Option<&str>) {
    if let Some(session_id) = session_id {
        codex_ws_sse_fallback_sessions()
            .lock()
            .await
            .insert(session_id.to_owned());
    }
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
        .or_else(|| get_env_api_key(&model.provider));
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
    let base_url = base_url.trim_end_matches('/');
    if base_url.ends_with("/v1") {
        format!("{base_url}/chat/completions")
    } else {
        format!("{base_url}/v1/chat/completions")
    }
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

#[cfg(test)]
fn parse_sse_json_body(body: &str) -> Vec<Value> {
    let mut events = Vec::new();
    let mut parser = SseJsonParser::default();
    parser.push_bytes(body.as_bytes(), &mut events);
    parser.finish(&mut events);
    events
}

#[derive(Debug, Default)]
struct SseJsonParser {
    line_buffer: Vec<u8>,
    data_lines: Vec<String>,
}

impl SseJsonParser {
    fn push_bytes(&mut self, bytes: &[u8], events: &mut Vec<Value>) {
        for byte in bytes {
            if *byte == b'\n' {
                self.push_line(events);
            } else {
                self.line_buffer.push(*byte);
            }
        }
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

fn push_abort_if_requested(
    sender: &crate::AssistantMessageEventSender,
    options: &SimpleStreamOptions,
    output: &mut AssistantMessage,
) -> bool {
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

    #[test]
    fn parses_sse_json_body_and_ignores_done() {
        let body = "event: message\ndata: {\"a\":1}\n\ndata: [DONE]\n\n";
        assert_eq!(
            parse_sse_json_body(body),
            vec![serde_json::json!({ "a": 1 })]
        );
    }
}
