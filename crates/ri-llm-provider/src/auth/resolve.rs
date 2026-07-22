//! Stateless auth resolution shared by the `Models` and images collections,
//! mirroring pi `auth/resolve.ts`.
//!
//! A stored credential owns the provider: ambient/env is consulted only when
//! nothing is stored. There is no silent env fallback after a failed refresh
//! or for a credential type without a matching handler.

use super::credential_store::CredentialStore;
use super::types::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult, Credential, ModelAuth, OAuthAuth,
    ProviderAuth,
};
use crate::now_millis;
use async_trait::async_trait;
use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelsErrorCode {
    ModelSource,
    ModelValidation,
    Provider,
    Stream,
    Auth,
    OAuth,
}

/// Typed error surface of the Models runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsError {
    pub code: ModelsErrorCode,
    pub message: String,
    pub cause: Option<String>,
}

impl ModelsError {
    pub fn new(code: ModelsErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            cause: None,
        }
    }

    pub fn with_cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }
}

impl Display for ModelsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)?;
        if let Some(cause) = &self.cause {
            write!(f, ": {cause}")?;
        }
        Ok(())
    }
}

impl Error for ModelsError {}

/// Request-time auth overrides (explicit api key / scoped env).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuthResolutionOverrides {
    pub api_key: Option<String>,
    pub env: BTreeMap<String, String>,
}

/// Resolve provider-scoped auth from a stored credential and/or ambient
/// sources.
pub async fn resolve_provider_auth(
    provider_id: &str,
    auth: &ProviderAuth,
    credentials: &dyn CredentialStore,
    auth_context: &dyn AuthContext,
    overrides: &AuthResolutionOverrides,
) -> Result<Option<AuthResult>, ModelsError> {
    let overlay;
    let request_context: &dyn AuthContext = if overrides.env.is_empty() {
        auth_context
    } else {
        overlay = EnvOverlayAuthContext {
            base: auth_context,
            env: overrides.env.clone(),
        };
        &overlay
    };

    if let (Some(api_key_override), Some(api_key_auth)) = (&overrides.api_key, &auth.api_key) {
        let credential = ApiKeyCredential {
            key: Some(api_key_override.clone()),
            env: overrides.env.clone(),
        };
        return resolve_api_key(
            request_context,
            api_key_auth.as_ref(),
            provider_id,
            Some(&credential),
        )
        .await;
    }

    let stored = read_credential(credentials, provider_id).await?;
    if let Some(stored) = stored {
        match (&stored, &auth.oauth, &auth.api_key) {
            (Credential::OAuth(oauth_credential), Some(oauth), _) => {
                return resolve_stored_oauth(
                    credentials,
                    provider_id,
                    oauth.as_ref(),
                    oauth_credential,
                )
                .await;
            }
            (Credential::ApiKey(api_key_credential), _, Some(api_key_auth)) => {
                let mut credential = api_key_credential.clone();
                credential.env.extend(
                    overrides
                        .env
                        .iter()
                        .map(|(name, value)| (name.clone(), value.clone())),
                );
                return resolve_api_key(
                    request_context,
                    api_key_auth.as_ref(),
                    provider_id,
                    Some(&credential),
                )
                .await;
            }
            // A stored credential without a matching handler blocks ambient
            // fallback.
            _ => return Ok(None),
        }
    }

    // Ambient (env vars, AWS profiles, ADC files).
    match &auth.api_key {
        Some(api_key_auth) => {
            resolve_api_key(request_context, api_key_auth.as_ref(), provider_id, None).await
        }
        None => Ok(None),
    }
}

struct EnvOverlayAuthContext<'a> {
    base: &'a dyn AuthContext,
    env: BTreeMap<String, String>,
}

#[async_trait]
impl AuthContext for EnvOverlayAuthContext<'_> {
    async fn env(&self, name: &str) -> Option<String> {
        match self.env.get(name).filter(|value| !value.is_empty()) {
            Some(value) => Some(value.clone()),
            None => self.base.env(name).await,
        }
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.base.file_exists(path).await
    }
}

/// OAuth resolution with double-checked locking: valid tokens cost zero
/// locks; expired tokens lock, re-check expiry under the lock, refresh once
/// globally, and persist the rotated credential before release.
async fn resolve_stored_oauth(
    credentials: &dyn CredentialStore,
    provider_id: &str,
    oauth: &dyn OAuthAuth,
    stored: &super::types::OAuthCredential,
) -> Result<Option<AuthResult>, ModelsError> {
    let mut credential = stored.clone();

    if now_millis() >= credential.expires {
        // Optimistic check said expired; the authoritative check runs under
        // the lock.
        let refresh_error = std::sync::Arc::new(parking_lot::Mutex::new(None::<String>));
        let refresh_error_slot = refresh_error.clone();
        let post = credentials
            .modify(
                provider_id,
                Box::new(move |current| {
                    Box::pin(async move {
                        let Some(Credential::OAuth(current)) = current else {
                            return Ok(None); // logged out meanwhile
                        };
                        if now_millis() < current.expires {
                            return Ok(None); // another request refreshed
                        }
                        match oauth.refresh(&current).await {
                            Ok(refreshed) => Ok(Some(Credential::OAuth(refreshed))),
                            Err(error) => {
                                *refresh_error_slot.lock() = Some(error.clone());
                                Err(error)
                            }
                        }
                    })
                }),
            )
            .await;
        let post = match post {
            Ok(post) => post,
            Err(error) => {
                return Err(if refresh_error.lock().is_some() {
                    ModelsError::new(
                        ModelsErrorCode::OAuth,
                        format!("OAuth refresh failed for {provider_id}"),
                    )
                    .with_cause(error)
                } else {
                    ModelsError::new(
                        ModelsErrorCode::Auth,
                        format!("Credential store modify failed for {provider_id}"),
                    )
                    .with_cause(error)
                });
            }
        };
        match post {
            Some(Credential::OAuth(post)) => credential = post,
            _ => return Ok(None), // logged out meanwhile
        }
    }

    match oauth.to_auth(&credential).await {
        Ok(auth) => Ok(Some(AuthResult {
            auth,
            env: Default::default(),
            source: Some("OAuth".to_owned()),
        })),
        Err(error) => Err(ModelsError::new(
            ModelsErrorCode::OAuth,
            format!("OAuth auth derivation failed for {provider_id}"),
        )
        .with_cause(error)),
    }
}

async fn resolve_api_key(
    auth_context: &dyn AuthContext,
    api_key: &dyn ApiKeyAuth,
    provider_id: &str,
    credential: Option<&ApiKeyCredential>,
) -> Result<Option<AuthResult>, ModelsError> {
    api_key
        .resolve(auth_context, credential)
        .await
        .map_err(|error| {
            ModelsError::new(
                ModelsErrorCode::Auth,
                format!("API key auth failed for provider {provider_id}"),
            )
            .with_cause(error)
        })
}

pub async fn read_credential(
    credentials: &dyn CredentialStore,
    provider_id: &str,
) -> Result<Option<Credential>, ModelsError> {
    credentials.read(provider_id).await.map_err(|error| {
        ModelsError::new(
            ModelsErrorCode::Auth,
            format!("Credential store read failed for {provider_id}"),
        )
        .with_cause(error)
    })
}

/// Derive a [`ModelAuth`] from a stored api-key override — used by request
/// paths that carry an explicit key.
pub fn model_auth_from_api_key(api_key: impl Into<String>) -> ModelAuth {
    ModelAuth {
        api_key: Some(api_key.into()),
        ..Default::default()
    }
}
