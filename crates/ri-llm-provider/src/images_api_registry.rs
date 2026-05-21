use crate::{
    api_registry::ProviderError,
    openrouter_images::ensure_builtin_images_api_providers,
    types::{AssistantImages, ImagesApi, ImagesContext, ImagesModel, ImagesOptions},
};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::{collections::BTreeMap, sync::Arc};

#[async_trait]
pub trait ImagesApiProvider: Send + Sync {
    fn api(&self) -> &str;

    async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Result<AssistantImages, ProviderError>;
}

#[derive(Clone)]
struct ImagesRegistryEntry {
    provider: Arc<dyn ImagesApiProvider>,
    source_id: Option<String>,
}

struct CheckedImagesApiProvider {
    api: ImagesApi,
    inner: Arc<dyn ImagesApiProvider>,
}

#[async_trait]
impl ImagesApiProvider for CheckedImagesApiProvider {
    fn api(&self) -> &str {
        &self.api
    }

    async fn generate_images(
        &self,
        model: &ImagesModel,
        context: ImagesContext,
        options: ImagesOptions,
    ) -> Result<AssistantImages, ProviderError> {
        ensure_images_model_api(model, &self.api)?;
        self.inner.generate_images(model, context, options).await
    }
}

static IMAGES_API_PROVIDER_REGISTRY: std::sync::LazyLock<
    RwLock<BTreeMap<ImagesApi, ImagesRegistryEntry>>,
> = std::sync::LazyLock::new(|| RwLock::new(BTreeMap::new()));

pub fn register_images_api_provider(
    provider: Arc<dyn ImagesApiProvider>,
    source_id: Option<String>,
) {
    let api = provider.api().to_owned();
    let provider = Arc::new(CheckedImagesApiProvider {
        api: api.clone(),
        inner: provider,
    });
    IMAGES_API_PROVIDER_REGISTRY.write().insert(
        api,
        ImagesRegistryEntry {
            provider,
            source_id,
        },
    );
}

pub fn get_images_api_provider(api: &str) -> Option<Arc<dyn ImagesApiProvider>> {
    IMAGES_API_PROVIDER_REGISTRY
        .read()
        .get(api)
        .map(|entry| entry.provider.clone())
}

pub fn get_images_api_providers() -> Vec<Arc<dyn ImagesApiProvider>> {
    IMAGES_API_PROVIDER_REGISTRY
        .read()
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn unregister_images_api_providers(source_id: &str) {
    IMAGES_API_PROVIDER_REGISTRY
        .write()
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_images_api_providers() {
    IMAGES_API_PROVIDER_REGISTRY.write().clear();
}

pub async fn generate_images(
    model: &ImagesModel,
    context: ImagesContext,
    options: ImagesOptions,
) -> Result<AssistantImages, ProviderError> {
    ensure_builtin_images_api_providers();
    let provider = get_images_api_provider(&model.api)
        .ok_or_else(|| ProviderError::MissingApi(model.api.clone()))?;
    ensure_images_model_api(model, provider.api())?;
    provider.generate_images(model, context, options).await
}

pub(crate) fn ensure_images_model_api(model: &ImagesModel, api: &str) -> Result<(), ProviderError> {
    if model.api == api {
        Ok(())
    } else {
        Err(ProviderError::MismatchedApi {
            actual: model.api.clone(),
            expected: api.to_owned(),
        })
    }
}
