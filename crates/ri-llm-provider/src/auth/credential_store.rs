//! App-owned credential storage, mirroring pi `auth/credential-store.ts`.

use super::types::{Credential, CredentialInfo};
use async_trait::async_trait;
use futures::future::BoxFuture;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Serialized read-modify-write closure. Sees the current credential; returns
/// the new credential, or `None` to leave the entry unchanged.
pub type CredentialModifyFn<'a> = Box<
    dyn FnOnce(Option<Credential>) -> BoxFuture<'a, Result<Option<Credential>, String>> + Send + 'a,
>;

/// App-owned credential storage, keyed by provider id, one credential per
/// provider. `modify` is the only write path, so every mutation is a
/// serialized read-modify-write; `Models::get_auth` runs OAuth refresh inside
/// `modify` so concurrent requests cannot double-refresh a rotated token.
///
/// Error semantics: `read` resolves `None` for missing entries. Methods error
/// only on storage failure; `Models` wraps such errors in [`super::ModelsError`]
/// with code `Auth`.
#[async_trait]
pub trait CredentialStore: Send + Sync {
    /// Read the stored credential, possibly expired. Display/status use;
    /// resolved request auth comes from `Models::get_auth`.
    async fn read(&self, provider_id: &str) -> Result<Option<Credential>, String>;

    /// List stored credential metadata without resolving or exposing secrets.
    async fn list(&self) -> Result<Vec<CredentialInfo>, String>;

    /// Serialized write — the only write path. Mutual exclusion per provider
    /// id. Resolves with the post-write credential. Errors from the closure
    /// propagate.
    async fn modify(
        &self,
        provider_id: &str,
        f: CredentialModifyFn<'_>,
    ) -> Result<Option<Credential>, String>;

    /// Remove a credential (logout). Serialized against `modify`.
    async fn delete(&self, provider_id: &str) -> Result<(), String>;
}

/// Default in-memory credential store. Apps inject persistent stores.
#[derive(Default)]
pub struct InMemoryCredentialStore {
    credentials: parking_lot::Mutex<BTreeMap<String, Credential>>,
    /// Per-provider write locks so `modify`/`delete` serialize per provider
    /// without blocking other providers.
    locks: parking_lot::Mutex<BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl InMemoryCredentialStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock_for(&self, provider_id: &str) -> Arc<tokio::sync::Mutex<()>> {
        self.locks
            .lock()
            .entry(provider_id.to_owned())
            .or_default()
            .clone()
    }
}

#[async_trait]
impl CredentialStore for InMemoryCredentialStore {
    async fn read(&self, provider_id: &str) -> Result<Option<Credential>, String> {
        Ok(self.credentials.lock().get(provider_id).cloned())
    }

    async fn list(&self) -> Result<Vec<CredentialInfo>, String> {
        Ok(self
            .credentials
            .lock()
            .iter()
            .map(|(provider_id, credential)| CredentialInfo {
                provider_id: provider_id.clone(),
                auth_type: credential.auth_type(),
            })
            .collect())
    }

    async fn modify(
        &self,
        provider_id: &str,
        f: CredentialModifyFn<'_>,
    ) -> Result<Option<Credential>, String> {
        let lock = self.lock_for(provider_id);
        let _guard = lock.lock().await;
        let current = self.credentials.lock().get(provider_id).cloned();
        let next = f(current.clone()).await?;
        if let Some(next) = &next {
            self.credentials
                .lock()
                .insert(provider_id.to_owned(), next.clone());
        }
        Ok(next.or(current))
    }

    async fn delete(&self, provider_id: &str) -> Result<(), String> {
        let lock = self.lock_for(provider_id);
        let _guard = lock.lock().await;
        self.credentials.lock().remove(provider_id);
        Ok(())
    }
}
