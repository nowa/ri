//! Provider-owned auth substrate for the Models runtime, mirroring pi
//! `src/auth/*`.
//!
//! Layering:
//! - [`types`] — credential/auth data model plus the [`ApiKeyAuth`],
//!   [`OAuthAuth`], [`AuthContext`], and [`AuthInteraction`] behavior
//!   contracts.
//! - [`credential_store`] — app-owned credential persistence with serialized
//!   per-provider writes; [`InMemoryCredentialStore`] is the default.
//! - [`context`] — the process-environment [`DefaultAuthContext`].
//! - [`helpers`] — [`env_api_key_auth`] for the standard stored-key/env-var
//!   resolution.
//! - [`resolve`] — the stateless stored-credential-owns-provider resolution
//!   algorithm with double-checked OAuth refresh, shared by every collection.

mod context;
mod credential_store;
mod helpers;
mod resolve;
mod types;

pub use context::DefaultAuthContext;
pub use credential_store::{CredentialModifyFn, CredentialStore, InMemoryCredentialStore};
pub use helpers::env_api_key_auth;
pub use resolve::{
    AuthResolutionOverrides, ModelsError, ModelsErrorCode, model_auth_from_api_key,
    read_credential, resolve_provider_auth,
};
pub use types::{
    ApiKeyAuth, ApiKeyCredential, AuthCheck, AuthContext, AuthEvent, AuthInfoLink, AuthInteraction,
    AuthPrompt, AuthPromptOption, AuthResult, AuthType, Credential, CredentialInfo, ModelAuth,
    OAuthAuth, OAuthCredential, ProviderAuth,
};
