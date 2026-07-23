//! The Models runtime, mirroring pi `models.ts` (v0.80.0 refactor).
//!
//! # Architecture
//!
//! - A [`Provider`] is the concrete runtime unit. It owns id/name/base
//!   metadata, auth methods ([`ProviderAuth`]), model listing, and stream
//!   behavior. [`create_provider`] builds one from parts: a static baseline
//!   catalog, an optional dynamic fetcher (with in-flight dedupe and
//!   [`ModelsStore`] persistence), and one API implementation or a map
//!   dispatched on `model.api`.
//! - [`Models`] is a runtime collection of providers plus auth application
//!   and stream convenience. Providers own stream behavior; `Models` resolves
//!   auth through [`resolve_provider_auth`] and delegates each request to the
//!   provider that owns the model.
//!
//! # State
//!
//! `Models` is a cheap-to-clone handle (`Arc` inner). Provider registration
//! order is preserved; ids are unique. Injected dependencies — the
//! [`CredentialStore`], [`ModelsStore`], and [`AuthContext`] — default to
//! in-memory/process implementations.
//!
//! # Data flow
//!
//! ```text
//! Models::stream(model, context, options)
//!   -> returns an event stream immediately; a spawned task then:
//!      1. looks up the owning provider (unknown provider -> error stream)
//!      2. resolves auth (stored credential owns the provider; double-checked
//!         OAuth refresh under the store lock; ambient env only when nothing
//!         is stored)
//!      3. merges auth into the request: explicit options win per field,
//!         headers merge case-insensitively, auth base URLs override the
//!         model, provider-scoped env merges into request env
//!      4. delegates to Provider::stream and forwards all events
//! ```

use crate::auth::{
    AuthCheck, AuthContext, AuthInteraction, AuthResolutionOverrides, AuthResult, AuthType,
    Credential, CredentialStore, DefaultAuthContext, InMemoryCredentialStore, ModelsError,
    ModelsErrorCode, ProviderAuth, resolve_provider_auth,
};
use crate::models_store::{
    InMemoryModelsStore, ModelsStore, ModelsStoreEntry, ProviderModelsStore,
};
use crate::{
    Api, ApiProvider, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    AssistantMessageEventStream, Context, Model, SimpleStreamOptions, StopReason, StreamOptions,
    assistant_message_event_stream, now_millis,
};
use async_trait::async_trait;
use futures::future::{BoxFuture, FutureExt as _, Shared};
use futures::stream::StreamExt as _;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Context passed to a dynamic provider's model refresh.
#[derive(Clone)]
pub struct RefreshModelsContext {
    /// Effective configured credential. OAuth credentials are refreshed
    /// before network access.
    pub credential: Option<Credential>,
    /// Persistent model storage scoped to this provider id.
    pub store: ProviderModelsStore,
    /// False during offline/cache-only initialization.
    pub allow_network: bool,
    /// Bypass provider freshness checks and fetch immediately when network
    /// access is allowed.
    pub force: bool,
    /// Cooperative cancellation for network requests.
    pub abort_flag: Option<Arc<AtomicBool>>,
}

impl RefreshModelsContext {
    fn aborted(&self) -> bool {
        self.abort_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelsRefreshOptions {
    /// Default true.
    pub allow_network: Option<bool>,
    pub force: bool,
    pub abort_flag: Option<Arc<AtomicBool>>,
}

#[derive(Debug, Default)]
pub struct ModelsRefreshResult {
    pub aborted: bool,
    pub errors: BTreeMap<String, ModelsError>,
}

/// A provider is the concrete runtime unit: id/name/base metadata, auth
/// methods, model listing, and stream behavior.
#[async_trait]
pub trait Provider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;

    fn base_url(&self) -> Option<&str> {
        None
    }

    fn headers(&self) -> BTreeMap<String, String> {
        BTreeMap::new()
    }

    /// Required: at least one of `api_key`/`oauth` — even ambient-credential
    /// providers and keyless local servers provide `api_key` auth whose
    /// `resolve()` reports whether the provider is configured.
    fn auth(&self) -> &ProviderAuth;

    /// Current known models, sync. Static providers return their catalog;
    /// dynamic providers return the list as of the last refresh (empty
    /// before the first).
    fn get_models(&self) -> Vec<Model>;

    /// Whether [`Provider::refresh_models`] does anything. Static providers
    /// report false and are skipped by `Models::refresh`.
    fn supports_refresh(&self) -> bool {
        false
    }

    /// Dynamic providers only: restore the provider-scoped stored catalog and
    /// optionally fetch a newer list using the effective credential.
    /// Implementations must retain their previous list on failure.
    async fn refresh_models(&self, _context: RefreshModelsContext) -> Result<(), ModelsError> {
        Ok(())
    }

    /// Optional provider policy for credential-specific model availability.
    fn filter_models(&self, models: Vec<Model>, _credential: Option<&Credential>) -> Vec<Model> {
        models
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream;

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream;
}

/// API implementation dispatch for [`create_provider`].
#[derive(Clone)]
pub enum ProviderApiDispatch {
    /// A single implementation streams all models.
    Single(Arc<dyn ApiProvider>),
    /// Dispatch on `model.api`; a model whose api has no entry produces a
    /// stream error.
    ByApi(BTreeMap<Api, Arc<dyn ApiProvider>>),
    /// Late-bound lookup (e.g. through the built-in API registry).
    Resolver(Arc<dyn Fn(&str) -> Option<Arc<dyn ApiProvider>> + Send + Sync>),
}

/// Fetcher for a dynamic model overlay.
pub type FetchModelsFn = Arc<
    dyn Fn(RefreshModelsContext) -> BoxFuture<'static, Result<Vec<Model>, String>> + Send + Sync,
>;

pub type FilterModelsFn = Arc<dyn Fn(Vec<Model>, Option<&Credential>) -> Vec<Model> + Send + Sync>;

pub struct CreateProviderOptions {
    pub id: String,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    pub base_url: Option<String>,
    pub headers: BTreeMap<String, String>,
    /// Required — every provider has auth semantics, even ambient/keyless
    /// ones.
    pub auth: ProviderAuth,
    /// Static baseline model list (empty for purely dynamic providers).
    pub models: Vec<Model>,
    /// Fetch a dynamic model overlay. `create_provider` restores/persists it
    /// through the [`ModelsStore`].
    pub fetch_models: Option<FetchModelsFn>,
    pub filter_models: Option<FilterModelsFn>,
    pub api: ProviderApiDispatch,
}

impl CreateProviderOptions {
    pub fn new(id: impl Into<String>, auth: ProviderAuth, api: ProviderApiDispatch) -> Self {
        Self {
            id: id.into(),
            name: None,
            base_url: None,
            headers: BTreeMap::new(),
            auth,
            models: Vec::new(),
            fetch_models: None,
            filter_models: None,
            api,
        }
    }
}

/// Builds a provider from parts. Built-in provider factories and custom
/// providers both go through this.
pub fn create_provider(options: CreateProviderOptions) -> Arc<dyn Provider> {
    Arc::new(PartsProvider {
        name: options.name.unwrap_or_else(|| options.id.clone()),
        id: options.id,
        base_url: options.base_url,
        headers: options.headers,
        auth: options.auth,
        baseline_models: options.models,
        dynamic_models: Arc::new(parking_lot::RwLock::new(Vec::new())),
        inflight_refresh: parking_lot::Mutex::new(None),
        fetch_models: options.fetch_models,
        filter: options.filter_models,
        api: options.api,
    })
}

type SharedRefresh = Shared<BoxFuture<'static, Result<(), String>>>;

struct PartsProvider {
    id: String,
    name: String,
    base_url: Option<String>,
    headers: BTreeMap<String, String>,
    auth: ProviderAuth,
    baseline_models: Vec<Model>,
    /// Dynamic overlay behind an `Arc` so 'static refresh futures can publish
    /// results the provider reads through the same handle.
    dynamic_models: Arc<parking_lot::RwLock<Vec<Model>>>,
    /// Concurrent refreshes share one in-flight future.
    inflight_refresh: parking_lot::Mutex<Option<SharedRefresh>>,
    fetch_models: Option<FetchModelsFn>,
    filter: Option<FilterModelsFn>,
    api: ProviderApiDispatch,
}

impl PartsProvider {
    fn api_for(&self, api: &str) -> Option<Arc<dyn ApiProvider>> {
        match &self.api {
            ProviderApiDispatch::Single(provider) => Some(provider.clone()),
            ProviderApiDispatch::ByApi(map) => map.get(api).cloned(),
            ProviderApiDispatch::Resolver(resolver) => resolver(api),
        }
    }

    fn dispatch(
        &self,
        model: &Model,
        run: impl FnOnce(
            Arc<dyn ApiProvider>,
        ) -> Result<AssistantMessageEventStream, crate::ProviderError>,
    ) -> AssistantMessageEventStream {
        let Some(streams) = self.api_for(&model.api) else {
            return error_stream(
                model,
                ModelsError::new(
                    ModelsErrorCode::Stream,
                    format!(
                        "Provider {} has no API implementation for \"{}\"",
                        self.id, model.api
                    ),
                )
                .to_string(),
            );
        };
        match run(streams) {
            Ok(stream) => stream,
            Err(error) => error_stream(model, error.to_string()),
        }
    }
}

#[async_trait]
impl Provider for PartsProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn base_url(&self) -> Option<&str> {
        self.base_url.as_deref()
    }

    fn headers(&self) -> BTreeMap<String, String> {
        self.headers.clone()
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Vec<Model> {
        let mut merged = self.baseline_models.clone();
        for model in self.dynamic_models.read().iter() {
            if let Some(existing) = merged.iter_mut().find(|entry| entry.id == model.id) {
                *existing = model.clone();
            } else {
                merged.push(model.clone());
            }
        }
        merged
    }

    fn supports_refresh(&self) -> bool {
        self.fetch_models.is_some()
    }

    async fn refresh_models(&self, context: RefreshModelsContext) -> Result<(), ModelsError> {
        let Some(fetch_models) = self.fetch_models.clone() else {
            return Ok(());
        };
        let shared = {
            let mut inflight = self.inflight_refresh.lock();
            if let Some(shared) = inflight.clone() {
                shared
            } else {
                let shared: SharedRefresh = refresh_parts_models(
                    self.id.clone(),
                    fetch_models,
                    context,
                    self.dynamic_models.clone(),
                )
                .shared();
                *inflight = Some(shared.clone());
                shared
            }
        };
        let result = shared.clone().await;
        // Only the future we awaited may clear the slot; a refresh started
        // after ours finished must not be discarded.
        let mut inflight = self.inflight_refresh.lock();
        if inflight
            .as_ref()
            .is_some_and(|current| SharedRefresh::ptr_eq(current, &shared))
        {
            inflight.take();
        }
        drop(inflight);
        result.map_err(|error| {
            ModelsError::new(
                ModelsErrorCode::ModelSource,
                format!("Model refresh failed for {}", self.id),
            )
            .with_cause(error)
        })
    }

    fn filter_models(&self, models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
        match &self.filter {
            Some(filter) => filter(models, credential),
            None => models,
        }
    }

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        self.dispatch(model, |streams| streams.stream(model, context, options))
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        self.dispatch(model, |streams| {
            streams.stream_simple(model, context, options)
        })
    }
}

fn refresh_parts_models(
    provider_id: String,
    fetch_models: FetchModelsFn,
    context: RefreshModelsContext,
    target: Arc<parking_lot::RwLock<Vec<Model>>>,
) -> BoxFuture<'static, Result<(), String>> {
    Box::pin(async move {
        if let Ok(Some(stored)) = context.store.read().await {
            let restored = stored
                .models
                .into_iter()
                .filter(|model| model.provider == provider_id)
                .collect::<Vec<_>>();
            *target.write() = restored;
        }
        if !context.allow_network || context.aborted() {
            return Ok(());
        }
        let refreshed = fetch_models(context.clone()).await?;
        if context.aborted() {
            return Ok(());
        }
        *target.write() = refreshed.clone();
        context
            .store
            .write(ModelsStoreEntry {
                models: refreshed,
                last_modified: None,
                checked_at: Some(now_millis()),
            })
            .await?;
        Ok(())
    })
}

/// Build an error assistant message stream for `model`.
pub(crate) fn error_stream(model: &Model, error: String) -> AssistantMessageEventStream {
    let (sender, stream) = assistant_message_event_stream();
    push_models_error(&sender, model, error);
    stream
}

fn push_models_error(sender: &AssistantMessageEventSender, model: &Model, error: String) {
    let message = AssistantMessage {
        content: Vec::new(),
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: crate::Usage::zero(),
        stop_reason: StopReason::Error,
        error_message: Some(error),
        timestamp: now_millis(),
    };
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: message,
    });
}

/// Transform fully assembled model/auth/request headers before provider
/// dispatch, mirroring pi `ModelsStreamTransforms.transformHeaders`. The hook
/// runs exactly once per request, after resolved auth headers (including
/// model-level headers, which apply only for model-scoped auth) merge with
/// explicit request headers.
pub type TransformHeadersFn = Arc<
    dyn Fn(BTreeMap<String, String>) -> BoxFuture<'static, BTreeMap<String, String>> + Send + Sync,
>;

/// Options for [`create_models`].
#[derive(Default)]
pub struct CreateModelsOptions {
    pub credentials: Option<Arc<dyn CredentialStore>>,
    pub models_store: Option<Arc<dyn ModelsStore>>,
    pub auth_context: Option<Arc<dyn AuthContext>>,
    /// Optional header-transform hook applied once over assembled headers.
    pub transform_headers: Option<TransformHeadersFn>,
}

/// Runtime collection of providers plus auth application and stream
/// convenience. Cheap to clone; clones share providers and state.
#[derive(Clone)]
pub struct Models {
    inner: Arc<ModelsInner>,
}

struct ModelsInner {
    /// Insertion-ordered, unique by id.
    providers: parking_lot::RwLock<Vec<Arc<dyn Provider>>>,
    credentials: Arc<dyn CredentialStore>,
    models_store: Arc<dyn ModelsStore>,
    auth_context: Arc<dyn AuthContext>,
    transform_headers: Option<TransformHeadersFn>,
}

pub fn create_models(options: CreateModelsOptions) -> Models {
    Models {
        inner: Arc::new(ModelsInner {
            providers: parking_lot::RwLock::new(Vec::new()),
            credentials: options
                .credentials
                .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::new())),
            models_store: options
                .models_store
                .unwrap_or_else(|| Arc::new(InMemoryModelsStore::new())),
            auth_context: options
                .auth_context
                .unwrap_or_else(|| Arc::new(DefaultAuthContext)),
            transform_headers: options.transform_headers,
        }),
    }
}

impl Models {
    /// Upsert/replace by provider id. Provider ids are unique.
    pub fn set_provider(&self, provider: Arc<dyn Provider>) {
        let mut providers = self.inner.providers.write();
        if let Some(existing) = providers
            .iter_mut()
            .find(|existing| existing.id() == provider.id())
        {
            *existing = provider;
        } else {
            providers.push(provider);
        }
    }

    pub fn delete_provider(&self, id: &str) {
        self.inner
            .providers
            .write()
            .retain(|provider| provider.id() != id);
    }

    pub fn clear_providers(&self) {
        self.inner.providers.write().clear();
    }

    pub fn get_providers(&self) -> Vec<Arc<dyn Provider>> {
        self.inner.providers.read().clone()
    }

    pub fn get_provider(&self, id: &str) -> Option<Arc<dyn Provider>> {
        self.inner
            .providers
            .read()
            .iter()
            .find(|provider| provider.id() == id)
            .cloned()
    }

    pub fn credentials(&self) -> Arc<dyn CredentialStore> {
        self.inner.credentials.clone()
    }

    /// Sync read of last-known models from one provider or all providers.
    pub fn get_models(&self, provider: Option<&str>) -> Vec<Model> {
        match provider {
            Some(id) => self
                .get_provider(id)
                .map(|provider| provider.get_models())
                .unwrap_or_default(),
            None => self
                .get_providers()
                .iter()
                .flat_map(|provider| provider.get_models())
                .collect(),
        }
    }

    /// Sync runtime model lookup against last-known lists.
    pub fn get_model(&self, provider: &str, id: &str) -> Option<Model> {
        self.get_models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Refresh every configured dynamic provider concurrently. Provider
    /// errors and cancellation are returned without failing; static and
    /// unconfigured providers are skipped.
    pub async fn refresh(&self, options: ModelsRefreshOptions) -> ModelsRefreshResult {
        let allow_network = options.allow_network.unwrap_or(true);
        let refreshable = self
            .get_providers()
            .into_iter()
            .filter(|provider| provider.supports_refresh())
            .collect::<Vec<_>>();
        let mut result = ModelsRefreshResult::default();

        let refreshes = refreshable.into_iter().map(|provider| {
            let models = self.clone();
            let abort_flag = options.abort_flag.clone();
            let force = options.force;
            async move {
                if abort_flag
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::SeqCst))
                {
                    return None;
                }
                let store = ProviderModelsStore::new(
                    models.inner.models_store.clone(),
                    provider.id().to_owned(),
                );
                let stored = match crate::auth::read_credential(
                    models.inner.credentials.as_ref(),
                    provider.id(),
                )
                .await
                {
                    Ok(stored) => stored,
                    Err(error) => return Some((provider.id().to_owned(), error)),
                };
                let credential = match models
                    .resolve_refresh_credential(
                        provider.as_ref(),
                        stored.clone(),
                        allow_network,
                        abort_flag.clone(),
                    )
                    .await
                {
                    Ok(Some(credential)) => Some(credential),
                    Ok(None) => return None,
                    Err(error) => return Some((provider.id().to_owned(), error)),
                };
                let refresh_result = provider
                    .refresh_models(RefreshModelsContext {
                        credential,
                        store: store.clone(),
                        allow_network,
                        force,
                        abort_flag: abort_flag.clone(),
                    })
                    .await;
                match refresh_result {
                    Ok(()) => None,
                    Err(error) => {
                        let aborted = abort_flag
                            .as_ref()
                            .is_some_and(|flag| flag.load(Ordering::SeqCst));
                        // Preserve the original auth/network error; cache
                        // restoration is best-effort here.
                        let _ = provider
                            .refresh_models(RefreshModelsContext {
                                credential: stored,
                                store,
                                allow_network: false,
                                force: false,
                                abort_flag,
                            })
                            .await;
                        (!aborted).then(|| (provider.id().to_owned(), error))
                    }
                }
            }
        });
        let errors = futures::future::join_all(refreshes).await;
        for (provider_id, error) in errors.into_iter().flatten() {
            result.errors.insert(provider_id, error);
        }
        result.aborted = options
            .abort_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst));
        result
    }

    async fn resolve_refresh_credential(
        &self,
        provider: &dyn Provider,
        stored: Option<Credential>,
        allow_network: bool,
        abort_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Option<Credential>, ModelsError> {
        match stored {
            Some(Credential::OAuth(stored_oauth)) => {
                let Some(oauth) = provider.auth().oauth.clone() else {
                    return Ok(None);
                };
                if !allow_network || now_millis() < stored_oauth.expires {
                    return Ok(Some(Credential::OAuth(stored_oauth)));
                }
                if abort_flag
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::SeqCst))
                {
                    return Ok(None);
                }
                let post = self
                    .inner
                    .credentials
                    .modify(
                        provider.id(),
                        Box::new(move |current| {
                            let oauth = oauth.clone();
                            Box::pin(async move {
                                let Some(Credential::OAuth(current)) = current else {
                                    return Ok(None);
                                };
                                if now_millis() < current.expires {
                                    return Ok(None);
                                }
                                oauth
                                    .refresh(&current)
                                    .await
                                    .map(|refreshed| Some(Credential::OAuth(refreshed)))
                            })
                        }),
                    )
                    .await
                    .map_err(|error| {
                        ModelsError::new(
                            ModelsErrorCode::OAuth,
                            format!("OAuth refresh failed for {}", provider.id()),
                        )
                        .with_cause(error)
                    })?;
                Ok(match post {
                    Some(Credential::OAuth(post)) => Some(Credential::OAuth(post)),
                    _ => None,
                })
            }
            stored => {
                let Some(api_key) = provider.auth().api_key.clone() else {
                    return Ok(None);
                };
                let credential = match &stored {
                    Some(Credential::ApiKey(credential)) => Some(credential.clone()),
                    _ => None,
                };
                let result = api_key
                    .resolve(self.inner.auth_context.as_ref(), credential.as_ref())
                    .await
                    .map_err(|error| {
                        ModelsError::new(
                            ModelsErrorCode::Auth,
                            format!("API key auth failed for provider {}", provider.id()),
                        )
                        .with_cause(error)
                    })?;
                Ok(result.map(|result| {
                    Credential::ApiKey(crate::auth::ApiKeyCredential {
                        key: result.auth.api_key,
                        env: result.env,
                    })
                }))
            }
        }
    }

    async fn check_provider_auth(
        &self,
        provider: &dyn Provider,
        credential: Option<&Credential>,
    ) -> Result<Option<AuthCheck>, ModelsError> {
        if let Some(Credential::OAuth(_)) = credential {
            return Ok(provider.auth().oauth.as_ref().map(|_| AuthCheck {
                source: Some("OAuth".to_owned()),
                auth_type: AuthType::OAuth,
            }));
        }
        let Some(api_key) = provider.auth().api_key.clone() else {
            return Ok(None);
        };
        if api_key.supports_check() {
            let api_key_credential = match credential {
                Some(Credential::ApiKey(credential)) => Some(credential.clone()),
                _ => None,
            };
            return api_key
                .check(
                    self.inner.auth_context.as_ref(),
                    api_key_credential.as_ref(),
                )
                .await
                .map_err(|error| {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("API key auth check failed for provider {}", provider.id()),
                    )
                    .with_cause(error)
                });
        }

        let resolution = resolve_provider_auth(
            provider.id(),
            provider.auth(),
            self.inner.credentials.as_ref(),
            self.inner.auth_context.as_ref(),
            &AuthResolutionOverrides::default(),
        )
        .await?;
        Ok(resolution.map(|resolution| AuthCheck {
            source: resolution.source,
            auth_type: AuthType::ApiKey,
        }))
    }

    /// Check whether a provider has complete auth configuration without
    /// refreshing OAuth.
    pub async fn check_auth(&self, provider_id: &str) -> Result<Option<AuthCheck>, ModelsError> {
        let Some(provider) = self.get_provider(provider_id) else {
            return Ok(None);
        };
        let credential =
            crate::auth::read_credential(self.inner.credentials.as_ref(), provider_id).await?;
        self.check_provider_auth(provider.as_ref(), credential.as_ref())
            .await
    }

    /// Return models whose providers have complete auth configuration.
    pub async fn get_available(
        &self,
        provider_id: Option<&str>,
    ) -> Result<Vec<Model>, ModelsError> {
        let providers = match provider_id {
            Some(id) => self.get_provider(id).into_iter().collect::<Vec<_>>(),
            None => self.get_providers(),
        };
        let mut models = Vec::new();
        for provider in providers {
            let credential =
                crate::auth::read_credential(self.inner.credentials.as_ref(), provider.id())
                    .await?;
            let auth = self
                .check_provider_auth(provider.as_ref(), credential.as_ref())
                .await?;
            if auth.is_none() {
                continue;
            }
            models.extend(provider.filter_models(provider.get_models(), credential.as_ref()));
        }
        Ok(models)
    }

    /// Resolve provider-scoped auth by provider id. `Ok(None)` when the
    /// provider is unknown or unconfigured.
    pub async fn get_auth(
        &self,
        provider_id: &str,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, ModelsError> {
        let Some(provider) = self.get_provider(provider_id) else {
            return Ok(None);
        };
        resolve_provider_auth(
            provider.id(),
            provider.auth(),
            self.inner.credentials.as_ref(),
            self.inner.auth_context.as_ref(),
            overrides,
        )
        .await
    }

    /// Resolve provider auth plus static model headers.
    pub async fn get_auth_for_model(
        &self,
        model: &Model,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, ModelsError> {
        let result = self.get_auth(&model.provider, overrides).await?;
        Ok(result.map(|mut result| {
            if !model.headers.is_empty() {
                result.auth.headers = merge_headers(&result.auth.headers, &model.headers);
            }
            result
        }))
    }

    /// Run a provider-owned login flow and persist its returned credential.
    pub async fn login(
        &self,
        provider_id: &str,
        auth_type: AuthType,
        interaction: &dyn AuthInteraction,
    ) -> Result<Credential, ModelsError> {
        let provider = self.get_provider(provider_id).ok_or_else(|| {
            ModelsError::new(
                ModelsErrorCode::Provider,
                format!("Unknown provider: {provider_id}"),
            )
        })?;
        let credential = match auth_type {
            AuthType::OAuth => {
                let oauth = provider.auth().oauth.clone().ok_or_else(|| {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("{} does not support oauth login", provider.name()),
                    )
                })?;
                Credential::OAuth(
                    oauth
                        .login(interaction)
                        .await
                        .map_err(|error| ModelsError::new(ModelsErrorCode::Auth, error))?,
                )
            }
            AuthType::ApiKey => {
                let api_key = provider
                    .auth()
                    .api_key
                    .clone()
                    .filter(|auth| auth.supports_login());
                let api_key = api_key.ok_or_else(|| {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("{} does not support api_key login", provider.name()),
                    )
                })?;
                Credential::ApiKey(
                    api_key
                        .login(interaction)
                        .await
                        .map_err(|error| ModelsError::new(ModelsErrorCode::Auth, error))?,
                )
            }
        };
        let persisted = credential.clone();
        self.inner
            .credentials
            .modify(
                provider_id,
                Box::new(move |_| Box::pin(async move { Ok(Some(persisted)) })),
            )
            .await
            .map_err(|error| {
                ModelsError::new(
                    ModelsErrorCode::Auth,
                    format!("Credential store modify failed for {provider_id}"),
                )
                .with_cause(error)
            })?;
        Ok(credential)
    }

    /// Remove the stored credential for a provider.
    pub async fn logout(&self, provider_id: &str) -> Result<(), ModelsError> {
        self.inner
            .credentials
            .delete(provider_id)
            .await
            .map_err(|error| {
                ModelsError::new(
                    ModelsErrorCode::Auth,
                    format!("Credential store delete failed for {provider_id}"),
                )
                .with_cause(error)
            })
    }

    /// Merge resolved auth into request options. Explicit request options win
    /// per field.
    async fn apply_auth(
        &self,
        model: &Model,
        mut options: StreamOptions,
    ) -> Result<(Model, StreamOptions), ModelsError> {
        let overrides = AuthResolutionOverrides {
            api_key: options.api_key.clone(),
            env: options.env.clone(),
        };
        let resolution = self
            .get_auth_for_model(model, &overrides)
            .await?
            .ok_or_else(|| {
                ModelsError::new(
                    ModelsErrorCode::Auth,
                    format!("Provider is not configured: {}", model.provider),
                )
            })?;
        let auth = resolution.auth;

        if options.api_key.is_none() {
            options.api_key = auth.api_key;
        }
        options.headers = merge_headers(&auth.headers, &options.headers);
        // The Models-only transform runs last, once over assembled headers.
        if let Some(transform) = &self.inner.transform_headers {
            options.headers = transform(std::mem::take(&mut options.headers)).await;
        }
        let mut env = resolution.env;
        env.extend(options.env.clone());
        options.env = env;

        let mut request_model = model.clone();
        if let Some(base_url) = auth.base_url {
            request_model.base_url = base_url;
        }
        Ok((request_model, options))
    }

    pub fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let model = model.clone();
        lazy_stream(model.clone(), async move {
            let provider = models.get_provider(&model.provider).ok_or_else(|| {
                ModelsError::new(
                    ModelsErrorCode::Provider,
                    format!("Unknown provider: {}", model.provider),
                )
                .to_string()
            })?;
            let (request_model, request_options) = models
                .apply_auth(&model, options)
                .await
                .map_err(|error| error.to_string())?;
            Ok(provider.stream(&request_model, context, request_options))
        })
    }

    pub async fn complete(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> AssistantMessage {
        self.stream(model, context, options).result().await
    }

    pub fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessageEventStream {
        let models = self.clone();
        let model = model.clone();
        lazy_stream(model.clone(), async move {
            let provider = models.get_provider(&model.provider).ok_or_else(|| {
                ModelsError::new(
                    ModelsErrorCode::Provider,
                    format!("Unknown provider: {}", model.provider),
                )
                .to_string()
            })?;
            let mut options = options;
            let (request_model, stream_options) = models
                .apply_auth(&model, options.stream)
                .await
                .map_err(|error| error.to_string())?;
            options.stream = stream_options;
            Ok(provider.stream_simple(&request_model, context, options))
        })
    }

    pub async fn complete_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> AssistantMessage {
        self.stream_simple(model, context, options).result().await
    }
}

/// Defer async setup into a returned stream: setup errors surface as stream
/// error events, and inner stream events forward verbatim.
fn lazy_stream(
    model: Model,
    setup: impl std::future::Future<Output = Result<AssistantMessageEventStream, String>>
    + Send
    + 'static,
) -> AssistantMessageEventStream {
    let (sender, stream) = assistant_message_event_stream();
    tokio::spawn(async move {
        match setup.await {
            Ok(mut inner) => {
                while let Some(event) = inner.next().await {
                    sender.push(event);
                }
                if let Ok(result) = inner.try_result().await {
                    sender.end(result);
                }
            }
            Err(error) => push_models_error(&sender, &model, error),
        }
    });
    stream
}

/// Merge headers case-insensitively; `override_headers` wins.
pub fn merge_headers(
    base: &BTreeMap<String, String>,
    override_headers: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut merged = base.clone();
    for (name, value) in override_headers {
        let lower = name.to_ascii_lowercase();
        merged.retain(|existing, _| existing.to_ascii_lowercase() != lower);
        merged.insert(name.clone(), value.clone());
    }
    merged
}

/// Runtime-checked narrowing helper for dynamically looked-up models.
pub fn has_api(model: &Model, api: &str) -> bool {
    model.api == api
}
