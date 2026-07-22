//! Image-generation runtime collection, mirroring pi `images-models.ts`.
//!
//! [`ImagesProvider`] is the image-side counterpart of the chat
//! [`crate::models_runtime::Provider`]: it owns id/name metadata, auth, model
//! listing, and generation behavior. [`ImagesModels`] resolves provider auth
//! through the shared [`crate::auth`] substrate and merges it into request
//! options (explicit options win per field); generation never rejects —
//! failures come back as an error [`AssistantImages`].

use crate::auth::{
    AuthContext, AuthResolutionOverrides, AuthResult, CredentialStore, DefaultAuthContext,
    InMemoryCredentialStore, ModelsError, ModelsErrorCode, ProviderAuth, resolve_provider_auth,
};
use crate::images_api_registry::get_images_api_provider;
use crate::models_runtime::{CreateModelsOptions, merge_headers};
use crate::openrouter_images::ensure_builtin_images_api_providers;
use crate::types::{
    AssistantImages, ImagesContext, ImagesModel, ImagesOptions, ImagesStopReason, now_millis,
};
use async_trait::async_trait;
use futures::FutureExt as _;
use futures::future::{BoxFuture, Shared};
use std::collections::BTreeMap;
use std::sync::Arc;

/// An image-generation provider: id/name metadata, auth, model listing, and
/// generation behavior.
#[async_trait]
pub trait ImagesProvider: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;

    /// At least one of `api_key`/`oauth`; `ImagesModels::get_auth` returns
    /// `None` when the provider is unconfigured.
    fn auth(&self) -> &ProviderAuth;

    /// Current known models, sync and non-throwing. Static providers return
    /// their catalog; dynamic providers the list as of the last refresh.
    fn get_models(&self) -> Vec<ImagesModel>;

    fn supports_refresh(&self) -> bool {
        false
    }

    /// Dynamic providers only: fetch and update the model list. On failure
    /// the list stays at its last-known state and a later call retries.
    async fn refresh_models(&self) -> Result<(), String> {
        Ok(())
    }

    async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages;
}

/// Options for [`create_images_provider`].
pub struct CreateImagesProviderOptions {
    pub id: String,
    /// Display name. Default: `id`.
    pub name: Option<String>,
    /// Required — every provider has auth semantics, even ambient ones.
    pub auth: ProviderAuth,
    /// Initial model list (empty for purely dynamic providers).
    pub models: Vec<ImagesModel>,
    /// Dynamic providers: fetch the current list. Concurrent refreshes share
    /// one in-flight fetch; failures keep the last-known list.
    pub fetch_models: Option<FetchImagesModelsFn>,
    /// Generation implementation. `None` resolves through the images
    /// api-registry by `model.api` (the built-in path).
    pub api: Option<Arc<dyn crate::images_api_registry::ImagesApiProvider>>,
}

pub type FetchImagesModelsFn =
    Arc<dyn Fn() -> BoxFuture<'static, Result<Vec<ImagesModel>, String>> + Send + Sync>;

impl CreateImagesProviderOptions {
    pub fn new(id: impl Into<String>, auth: ProviderAuth) -> Self {
        Self {
            id: id.into(),
            name: None,
            auth,
            models: Vec::new(),
            fetch_models: None,
            api: None,
        }
    }
}

type SharedImagesRefresh = Shared<BoxFuture<'static, Result<(), String>>>;

struct PartsImagesProvider {
    id: String,
    name: String,
    auth: ProviderAuth,
    models: Arc<parking_lot::RwLock<Vec<ImagesModel>>>,
    fetch_models: Option<FetchImagesModelsFn>,
    api: Option<Arc<dyn crate::images_api_registry::ImagesApiProvider>>,
    inflight_refresh: parking_lot::Mutex<Option<SharedImagesRefresh>>,
}

#[async_trait]
impl ImagesProvider for PartsImagesProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn auth(&self) -> &ProviderAuth {
        &self.auth
    }

    fn get_models(&self) -> Vec<ImagesModel> {
        self.models.read().clone()
    }

    fn supports_refresh(&self) -> bool {
        self.fetch_models.is_some()
    }

    async fn refresh_models(&self) -> Result<(), String> {
        let Some(fetch_models) = self.fetch_models.clone() else {
            return Ok(());
        };
        let shared = {
            let mut inflight = self.inflight_refresh.lock();
            if let Some(shared) = inflight.clone() {
                shared
            } else {
                let target = self.models.clone();
                let shared: SharedImagesRefresh = async move {
                    let refreshed = fetch_models().await?;
                    *target.write() = refreshed;
                    Ok(())
                }
                .boxed()
                .shared();
                *inflight = Some(shared.clone());
                shared
            }
        };
        let result = shared.clone().await;
        // Only the future we awaited may clear the slot.
        let mut inflight = self.inflight_refresh.lock();
        if inflight
            .as_ref()
            .is_some_and(|current| SharedImagesRefresh::ptr_eq(current, &shared))
        {
            inflight.take();
        }
        drop(inflight);
        result
    }

    async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages {
        let api = match &self.api {
            Some(api) => api.clone(),
            None => {
                ensure_builtin_images_api_providers();
                match get_images_api_provider(&model.api) {
                    Some(api) => api,
                    None => {
                        return error_images(
                            model,
                            format!("No API provider registered for api: {}", model.api),
                        );
                    }
                }
            }
        };
        match api.generate_images(model, context, options).await {
            Ok(images) => images,
            Err(error) => error_images(model, error.to_string()),
        }
    }
}

/// Builds an image-generation provider from parts.
pub fn create_images_provider(options: CreateImagesProviderOptions) -> Arc<dyn ImagesProvider> {
    Arc::new(PartsImagesProvider {
        name: options.name.unwrap_or_else(|| options.id.clone()),
        id: options.id,
        auth: options.auth,
        models: Arc::new(parking_lot::RwLock::new(options.models)),
        fetch_models: options.fetch_models,
        api: options.api,
        inflight_refresh: parking_lot::Mutex::new(None),
    })
}

fn error_images(model: &ImagesModel, error_message: String) -> AssistantImages {
    AssistantImages {
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        output: Vec::new(),
        response_id: None,
        usage: None,
        stop_reason: ImagesStopReason::Error,
        error_message: Some(error_message),
        timestamp: now_millis(),
    }
}

struct ImagesModelsInner {
    providers: parking_lot::RwLock<Vec<Arc<dyn ImagesProvider>>>,
    credentials: Arc<dyn CredentialStore>,
    auth_context: Arc<dyn AuthContext>,
}

/// Runtime collection of image-generation providers plus auth application and
/// generation convenience: the image-side counterpart of `Models`.
#[derive(Clone)]
pub struct ImagesModels {
    inner: Arc<ImagesModelsInner>,
}

pub fn create_images_models(options: CreateModelsOptions) -> ImagesModels {
    ImagesModels {
        inner: Arc::new(ImagesModelsInner {
            providers: parking_lot::RwLock::new(Vec::new()),
            credentials: options
                .credentials
                .unwrap_or_else(|| Arc::new(InMemoryCredentialStore::default())),
            auth_context: options
                .auth_context
                .unwrap_or_else(|| Arc::new(DefaultAuthContext)),
        }),
    }
}

impl ImagesModels {
    /// Upsert/replace by provider id. Insertion order is preserved.
    pub fn set_provider(&self, provider: Arc<dyn ImagesProvider>) {
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

    pub fn get_providers(&self) -> Vec<Arc<dyn ImagesProvider>> {
        self.inner.providers.read().clone()
    }

    pub fn get_provider(&self, id: &str) -> Option<Arc<dyn ImagesProvider>> {
        self.inner
            .providers
            .read()
            .iter()
            .find(|provider| provider.id() == id)
            .cloned()
    }

    /// Sync read of last-known models from one provider or all providers.
    pub fn get_models(&self, provider: Option<&str>) -> Vec<ImagesModel> {
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

    pub fn get_model(&self, provider: &str, id: &str) -> Option<ImagesModel> {
        self.get_models(Some(provider))
            .into_iter()
            .find(|model| model.id == id)
    }

    /// Ask dynamic providers to re-fetch their model lists. With a provider
    /// id, failures surface as `ModelsError` ("model_source"); without one,
    /// all providers refresh concurrently best-effort.
    pub async fn refresh(&self, provider: Option<&str>) -> Result<(), ModelsError> {
        if let Some(id) = provider {
            let Some(provider) = self.get_provider(id) else {
                return Ok(());
            };
            if !provider.supports_refresh() {
                return Ok(());
            }
            return provider.refresh_models().await.map_err(|error| {
                ModelsError::new(
                    ModelsErrorCode::ModelSource,
                    format!("Model refresh failed for {id}"),
                )
                .with_cause(error)
            });
        }
        let providers = self.get_providers();
        futures::future::join_all(
            providers
                .iter()
                .map(|provider| async move { provider.refresh_models().await }),
        )
        .await;
        Ok(())
    }

    /// Resolve request auth by provider id. `None` when unknown/unconfigured;
    /// errors on real failures (OAuth refresh, store access).
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

    pub async fn get_auth_for_model(
        &self,
        model: &ImagesModel,
        overrides: &AuthResolutionOverrides,
    ) -> Result<Option<AuthResult>, ModelsError> {
        self.get_auth(&model.provider, overrides).await
    }

    /// Generate images through the owning provider with auth resolved and
    /// merged (explicit options win per field). Never rejects; failures come
    /// back as an error `AssistantImages`.
    pub async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> AssistantImages {
        let Some(provider) = self.get_provider(&model.provider) else {
            return error_images(model, format!("Unknown provider: {}", model.provider));
        };
        let overrides = AuthResolutionOverrides {
            api_key: options.api_key.clone(),
            env: options.env.clone(),
        };
        let resolution = match self.get_auth_for_model(model, &overrides).await {
            Ok(resolution) => resolution,
            Err(error) => return error_images(model, error.to_string()),
        };
        let Some(resolution) = resolution else {
            return provider.generate_images(model, context, options).await;
        };

        let auth = resolution.auth;
        let mut request_model = model.clone();
        if let Some(base_url) = auth.base_url {
            request_model.base_url = base_url;
        }
        let mut options = options;
        if options.api_key.is_none() {
            options.api_key = auth.api_key;
        }
        options.headers = merge_headers(&auth.headers, &options.headers);
        let mut env: BTreeMap<String, String> = resolution.env;
        env.extend(options.env.clone());
        options.env = env;

        provider
            .generate_images(&request_model, context, options)
            .await
    }
}
