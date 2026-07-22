//! Auth data model and behavior contracts for the Models runtime, mirroring
//! pi `auth/types.ts`.
//!
//! Data flows in one direction: a [`super::CredentialStore`] holds at most one
//! type-tagged [`Credential`] per provider; [`ApiKeyAuth`]/[`OAuthAuth`]
//! implementations turn a credential (or ambient environment) into an
//! [`AuthResult`]; the `Models` collection merges the resulting [`ModelAuth`]
//! into per-request stream options. Anything that cannot be expressed as
//! `api_key`, `headers`, or `base_url` is provider configuration, not auth.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;

/// Request auth for a single model request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelAuth {
    pub api_key: Option<String>,
    pub headers: BTreeMap<String, String>,
    pub base_url: Option<String>,
}

/// Stored api-key credential. `env` holds provider-scoped environment/config
/// values such as Cloudflare account/gateway ids.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ApiKeyCredential {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
}

/// Stored canonical OAuth credential. Extra fields (e.g. `accountId`,
/// `enterpriseUrl`) round-trip through `extra`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OAuthCredential {
    pub refresh: String,
    pub access: String,
    /// Unix milliseconds expiry of `access`.
    pub expires: i64,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// One type-tagged credential per provider — the shape of today's auth.json.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Credential {
    ApiKey(ApiKeyCredential),
    OAuth(OAuthCredential),
}

impl Credential {
    pub fn auth_type(&self) -> AuthType {
        match self {
            Self::ApiKey(_) => AuthType::ApiKey,
            Self::OAuth(_) => AuthType::OAuth,
        }
    }
}

/// Non-secret credential metadata for account/status enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CredentialInfo {
    pub provider_id: String,
    pub auth_type: AuthType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthType {
    ApiKey,
    OAuth,
}

/// Result of resolving auth for a model.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuthResult {
    pub auth: ModelAuth,
    /// Provider-scoped environment/config values resolved from credentials
    /// and ambient context.
    pub env: BTreeMap<String, String>,
    /// Human-readable label for status UI: "ANTHROPIC_API_KEY", "OAuth",
    /// "~/.aws/credentials".
    pub source: Option<String>,
}

/// Side-effect-free auth availability information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthCheck {
    pub source: Option<String>,
    pub auth_type: AuthType,
}

/// Prompt shown to the user during login.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthPrompt {
    Text {
        message: String,
        placeholder: Option<String>,
    },
    Secret {
        message: String,
        placeholder: Option<String>,
    },
    Select {
        message: String,
        options: Vec<AuthPromptOption>,
    },
    ManualCode {
        message: String,
        placeholder: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthPromptOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthInfoLink {
    pub url: String,
    pub label: Option<String>,
}

/// Out-of-band login progress events.
#[derive(Debug, Clone, PartialEq)]
pub enum AuthEvent {
    Info {
        message: String,
        links: Vec<AuthInfoLink>,
    },
    AuthUrl {
        url: String,
        instructions: Option<String>,
    },
    DeviceCode {
        user_code: String,
        verification_uri: String,
        interval_seconds: Option<u64>,
        expires_in_seconds: Option<u64>,
    },
    Progress {
        message: String,
    },
}

/// Login interaction callbacks serving both api-key and OAuth flows.
///
/// `prompt()` returns the entered/selected string (`Select` returns the
/// option id) and errors on cancel/abort.
#[async_trait]
pub trait AuthInteraction: Send + Sync {
    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, String>;
    fn notify(&self, event: AuthEvent);
}

/// Environment access for auth resolution. Injectable for tests.
#[async_trait]
pub trait AuthContext: Send + Sync {
    /// Non-empty environment value, or `None`.
    async fn env(&self, name: &str) -> Option<String>;
    /// Check whether a file exists. Supports a leading `~`.
    async fn file_exists(&self, path: &str) -> bool;
}

/// Api-key auth: stored key/provider env plus ambient sources (env vars, AWS
/// profiles, ADC files). Ambient-only providers report
/// `supports_login() == false`.
#[async_trait]
pub trait ApiKeyAuth: Send + Sync {
    /// Display name, e.g. "Anthropic API key".
    fn name(&self) -> &str;

    /// Whether [`ApiKeyAuth::login`] is implemented.
    fn supports_login(&self) -> bool {
        false
    }

    /// Interactive setup (prompt for key/provider env).
    async fn login(&self, _interaction: &dyn AuthInteraction) -> Result<ApiKeyCredential, String> {
        Err(format!("{} does not support api_key login", self.name()))
    }

    /// Whether a side-effect-free [`ApiKeyAuth::check`] is implemented. When
    /// false, availability is checked by resolving auth.
    fn supports_check(&self) -> bool {
        false
    }

    /// Optional side-effect-free availability check for `resolve()`
    /// implementations that may execute commands or do request-time work.
    async fn check(
        &self,
        _ctx: &dyn AuthContext,
        _credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthCheck>, String> {
        Ok(None)
    }

    /// Resolve auth from the stored credential and/or ambient sources,
    /// merging per field. `None` = not configured.
    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthResult>, String>;
}

/// OAuth auth. The `refresh`/`to_auth` split lets `Models` own the locked
/// refresh pattern: `refresh` produces a credential, `to_auth` derives
/// request auth from whatever credential ends up stored.
#[async_trait]
pub trait OAuthAuth: Send + Sync {
    /// Display name, e.g. "Anthropic (Claude Pro/Max)".
    fn name(&self) -> &str;

    /// Selector label for the subscription login option.
    fn login_label(&self) -> Option<&str> {
        None
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, String>;

    /// Exchange the refresh token. Network call; errors on failure
    /// (invalid_grant etc.). `Models` runs this under the store lock.
    async fn refresh(&self, credential: &OAuthCredential) -> Result<OAuthCredential, String>;

    /// Side-effect-free derivation of request auth from a valid credential.
    /// Covers per-credential base URLs (GitHub Copilot).
    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, String>;
}

/// Provider auth. At least one of `api_key`/`oauth` must be present: even
/// ambient-credential providers and keyless local servers provide `api_key`
/// auth whose `resolve()` reports whether the provider is configured.
#[derive(Clone, Default)]
pub struct ProviderAuth {
    pub api_key: Option<std::sync::Arc<dyn ApiKeyAuth>>,
    pub oauth: Option<std::sync::Arc<dyn OAuthAuth>>,
}

impl std::fmt::Debug for ProviderAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderAuth")
            .field("api_key", &self.api_key.as_ref().map(|auth| auth.name()))
            .field("oauth", &self.oauth.as_ref().map(|auth| auth.name()))
            .finish()
    }
}
