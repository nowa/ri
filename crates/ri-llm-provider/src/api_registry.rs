use crate::{
    event_stream::AssistantMessageEventStream,
    types::{Api, Context, Model, SimpleStreamOptions, StreamOptions},
};
use parking_lot::RwLock;
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("No API provider registered for api: {0}")]
    MissingApi(String),
    #[error("Mismatched api: {actual} expected {expected}")]
    MismatchedApi { actual: String, expected: String },
    #[error("{0}")]
    Provider(String),
}

pub trait ApiProvider: Send + Sync {
    fn api(&self) -> &str;

    fn stream(
        &self,
        model: &Model,
        context: Context,
        options: StreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError>;

    fn stream_simple(
        &self,
        model: &Model,
        context: Context,
        options: SimpleStreamOptions,
    ) -> Result<AssistantMessageEventStream, ProviderError> {
        self.stream(model, context, options.stream)
    }
}

#[derive(Clone)]
struct RegistryEntry {
    provider: Arc<dyn ApiProvider>,
    source_id: Option<String>,
}

static API_PROVIDER_REGISTRY: std::sync::LazyLock<RwLock<BTreeMap<Api, RegistryEntry>>> =
    std::sync::LazyLock::new(|| RwLock::new(BTreeMap::new()));

pub fn register_api_provider(provider: Arc<dyn ApiProvider>, source_id: Option<String>) {
    API_PROVIDER_REGISTRY.write().insert(
        provider.api().to_owned(),
        RegistryEntry {
            provider,
            source_id,
        },
    );
}

pub fn get_api_provider(api: &str) -> Option<Arc<dyn ApiProvider>> {
    API_PROVIDER_REGISTRY
        .read()
        .get(api)
        .map(|entry| entry.provider.clone())
}

pub fn get_api_providers() -> Vec<Arc<dyn ApiProvider>> {
    API_PROVIDER_REGISTRY
        .read()
        .values()
        .map(|entry| entry.provider.clone())
        .collect()
}

pub fn unregister_api_providers(source_id: &str) {
    API_PROVIDER_REGISTRY
        .write()
        .retain(|_, entry| entry.source_id.as_deref() != Some(source_id));
}

pub fn clear_api_providers() {
    API_PROVIDER_REGISTRY.write().clear();
}

pub(crate) fn ensure_model_api(model: &Model, api: &str) -> Result<(), ProviderError> {
    if model.api == api {
        Ok(())
    } else {
        Err(ProviderError::MismatchedApi {
            actual: model.api.clone(),
            expected: api.to_owned(),
        })
    }
}
