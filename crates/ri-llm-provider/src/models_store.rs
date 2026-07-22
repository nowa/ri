//! Persistent dynamic model catalogs, mirroring pi `models-store.ts`.

use crate::Model;
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelsStoreEntry {
    pub models: Vec<Model>,
    /// Unix timestamp from the remote catalog's Last-Modified header.
    pub last_modified: Option<i64>,
    /// Unix timestamp of the last completed remote check.
    pub checked_at: Option<i64>,
}

/// Persistent model catalogs keyed by provider id.
#[async_trait]
pub trait ModelsStore: Send + Sync {
    async fn read(&self, provider_id: &str) -> Result<Option<ModelsStoreEntry>, String>;
    async fn write(&self, provider_id: &str, entry: ModelsStoreEntry) -> Result<(), String>;
    async fn delete(&self, provider_id: &str) -> Result<(), String>;
}

/// [`ModelsStore`] scoped to one provider. Providers cannot access other
/// providers' catalogs.
#[derive(Clone)]
pub struct ProviderModelsStore {
    store: Arc<dyn ModelsStore>,
    provider_id: String,
}

impl ProviderModelsStore {
    pub fn new(store: Arc<dyn ModelsStore>, provider_id: impl Into<String>) -> Self {
        Self {
            store,
            provider_id: provider_id.into(),
        }
    }

    pub async fn read(&self) -> Result<Option<ModelsStoreEntry>, String> {
        self.store.read(&self.provider_id).await
    }

    pub async fn write(&self, entry: ModelsStoreEntry) -> Result<(), String> {
        self.store.write(&self.provider_id, entry).await
    }

    pub async fn delete(&self) -> Result<(), String> {
        self.store.delete(&self.provider_id).await
    }
}

#[derive(Default)]
pub struct InMemoryModelsStore {
    entries: parking_lot::Mutex<BTreeMap<String, ModelsStoreEntry>>,
}

impl InMemoryModelsStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl ModelsStore for InMemoryModelsStore {
    async fn read(&self, provider_id: &str) -> Result<Option<ModelsStoreEntry>, String> {
        Ok(self.entries.lock().get(provider_id).cloned())
    }

    async fn write(&self, provider_id: &str, entry: ModelsStoreEntry) -> Result<(), String> {
        self.entries.lock().insert(provider_id.to_owned(), entry);
        Ok(())
    }

    async fn delete(&self, provider_id: &str) -> Result<(), String> {
        self.entries.lock().remove(provider_id);
        Ok(())
    }
}
