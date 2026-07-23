//! Built-in runtime provider factories, mirroring pi `providers/all.ts`.
//!
//! Each factory wraps three existing pieces of this crate:
//! - the generated catalog seed (baseline model lists),
//! - the built-in HTTP API implementations (resolved lazily through the API
//!   registry, so importing one provider does not construct the others), and
//! - the auth substrate: env-var api-key auth by default, ambient auth for
//!   AWS Bedrock / Google Vertex, and OAuth adapters over the existing token
//!   primitives for Anthropic, GitHub Copilot, and OpenAI Codex.

use crate::auth::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult, ModelAuth, OAuthAuth, OAuthCredential,
    ProviderAuth, env_api_key_auth,
};
use crate::models_runtime::{
    CreateModelsOptions, CreateProviderOptions, Models, Provider, ProviderApiDispatch,
    create_models, create_provider,
};
use crate::oauth_auth_storage::{StoredOAuthCredentials, get_oauth_provider, refresh_oauth_token};
use crate::{
    ensure_builtin_api_providers, get_api_provider, github_copilot_base_url, seed_provider_ids,
    seed_provider_models,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Lazily resolve API implementations through the built-in registry.
fn builtin_api_dispatch() -> ProviderApiDispatch {
    ProviderApiDispatch::Resolver(Arc::new(|api| {
        ensure_builtin_api_providers();
        get_api_provider(api)
    }))
}

/// Ambient auth for providers whose credentials come from runtime context
/// (AWS profiles, ADC files) rather than a single api key.
struct AmbientApiKeyAuth {
    provider_id: String,
    display: String,
}

#[async_trait]
impl ApiKeyAuth for AmbientApiKeyAuth {
    fn name(&self) -> &str {
        &self.display
    }

    async fn resolve(
        &self,
        _ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthResult>, String> {
        if let Some(key) = credential.and_then(|credential| credential.key.clone()) {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    ..Default::default()
                },
                env: credential.map(|c| c.env.clone()).unwrap_or_default(),
                source: Some("stored credential".to_owned()),
            }));
        }
        Ok(
            crate::get_env_api_key(&self.provider_id).map(|api_key| AuthResult {
                auth: ModelAuth {
                    api_key: Some(api_key),
                    ..Default::default()
                },
                env: Default::default(),
                source: Some("ambient credentials".to_owned()),
            }),
        )
    }
}

/// OAuth adapter over the existing built-in token primitives.
struct BuiltInOAuthAdapter {
    provider_id: String,
    display: String,
}

fn to_stored(credential: &OAuthCredential) -> StoredOAuthCredentials {
    StoredOAuthCredentials {
        refresh: credential.refresh.clone(),
        access: credential.access.clone(),
        expires: credential.expires,
        extra: credential.extra.clone().into_iter().collect(),
    }
}

fn from_stored(credential: StoredOAuthCredentials) -> OAuthCredential {
    OAuthCredential {
        refresh: credential.refresh,
        access: credential.access,
        expires: credential.expires,
        extra: credential.extra.into_iter().collect(),
    }
}

#[async_trait]
impl OAuthAuth for BuiltInOAuthAdapter {
    fn name(&self) -> &str {
        &self.display
    }

    async fn login(
        &self,
        interaction: &dyn crate::auth::AuthInteraction,
    ) -> Result<OAuthCredential, String> {
        crate::auth::login_builtin_oauth_provider(&self.provider_id, interaction).await
    }

    async fn refresh(&self, credential: &OAuthCredential) -> Result<OAuthCredential, String> {
        refresh_oauth_token(&self.provider_id, &to_stored(credential))
            .await
            .map(from_stored)
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, String> {
        let mut auth = ModelAuth {
            api_key: Some(credential.access.clone()),
            ..Default::default()
        };
        if self.provider_id == "github-copilot" {
            let stored = to_stored(credential);
            auth.base_url = Some(github_copilot_base_url(
                Some(&credential.access),
                stored.enterprise_domain(),
            ));
        }
        Ok(auth)
    }
}

fn oauth_adapter(provider_id: &str) -> Option<Arc<dyn OAuthAuth>> {
    // xAI carries its own device-code OAuth on the provider (pi
    // `providers/xai.ts` `auth.oauth: xaiOAuth`).
    if provider_id == "xai" {
        return Some(Arc::new(crate::xai_oauth::XaiOAuth::new()));
    }
    let info = get_oauth_provider(provider_id)?;
    Some(Arc::new(BuiltInOAuthAdapter {
        provider_id: info.id,
        display: info.name,
    }))
}

fn builtin_provider_auth(provider_id: &str) -> ProviderAuth {
    let api_key: Arc<dyn ApiKeyAuth> = match provider_id {
        "amazon-bedrock" | "google-vertex" => Arc::new(AmbientApiKeyAuth {
            provider_id: provider_id.to_owned(),
            display: format!("{provider_id} ambient credentials"),
        }),
        _ => match crate::api_key_env_vars(provider_id) {
            Some(env_vars) => env_api_key_auth(
                format!("{provider_id} API key"),
                env_vars.iter().map(|name| (*name).to_owned()),
            ),
            None => Arc::new(AmbientApiKeyAuth {
                provider_id: provider_id.to_owned(),
                display: format!("{provider_id} credentials"),
            }),
        },
    };
    ProviderAuth {
        api_key: Some(api_key),
        oauth: oauth_adapter(provider_id),
    }
}

/// Build the runtime provider for one built-in catalog provider.
pub fn builtin_provider(provider_id: &str) -> Option<Arc<dyn Provider>> {
    let models = seed_provider_models(provider_id);
    if models.is_empty() {
        return None;
    }
    let base_url = models.first().map(|model| model.base_url.clone());
    let mut options = CreateProviderOptions::new(
        provider_id,
        builtin_provider_auth(provider_id),
        builtin_api_dispatch(),
    );
    options.base_url = base_url;
    options.models = models;
    Some(create_provider(options))
}

/// All built-in catalog providers, in catalog order, plus the Radius
/// gateway provider (dynamic catalog, no static seed).
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    let mut providers: Vec<Arc<dyn Provider>> = seed_provider_ids()
        .into_iter()
        .filter_map(|provider_id| builtin_provider(&provider_id))
        .collect();
    providers.push(radius_provider(RadiusProviderOptions::default()));
    providers
}

#[derive(Debug, Clone, Default)]
pub struct RadiusProviderOptions {
    /// Provider id. Default: `radius`.
    pub id: Option<String>,
    /// Display name. Default: `Radius`.
    pub name: Option<String>,
    /// Gateway URL. Default: [`crate::radius::DEFAULT_RADIUS_GATEWAY`].
    pub gateway: Option<String>,
}

/// Radius gateway provider with a persisted, dynamically refreshed catalog
/// (pi `providers/radius.ts`).
pub fn radius_provider(options: RadiusProviderOptions) -> Arc<dyn Provider> {
    use crate::radius::{
        DEFAULT_RADIUS_GATEWAY, RadiusOAuth, get_radius_models, get_radius_models_from_config,
        load_radius_gateway_config, normalize_radius_gateway_url,
    };
    let id = options.id.unwrap_or_else(|| "radius".to_owned());
    let name = options.name.unwrap_or_else(|| "Radius".to_owned());
    let gateway = normalize_radius_gateway_url(
        &options
            .gateway
            .unwrap_or_else(|| DEFAULT_RADIUS_GATEWAY.to_owned()),
    );

    let mut create = CreateProviderOptions::new(
        id.clone(),
        ProviderAuth {
            api_key: Some(env_api_key_auth(
                "Radius API key",
                ["RADIUS_API_KEY".to_owned()],
            )),
            oauth: Some(Arc::new(RadiusOAuth::new(name.clone(), &gateway))),
        },
        builtin_api_dispatch(),
    );
    create.name = Some(name);
    let fetch_id = id.clone();
    create.fetch_models = Some(Arc::new(move |context| {
        let gateway = gateway.clone();
        let id = fetch_id.clone();
        Box::pin(async move {
            let (api_key, oauth_credential) = match &context.credential {
                Some(crate::auth::Credential::OAuth(oauth)) => {
                    (Some(oauth.access.clone()), Some(oauth.clone()))
                }
                Some(crate::auth::Credential::ApiKey(key)) => (key.key.clone(), None),
                None => (None, None),
            };
            match load_radius_gateway_config(&gateway, api_key.as_deref()).await {
                Ok(config) => Ok(get_radius_models_from_config(&id, &config)),
                Err(error) => {
                    // Catalogs embedded in pre-store credentials keep working
                    // when the gateway is unreachable.
                    let legacy = get_radius_models(&id, oauth_credential.as_ref());
                    if legacy.is_empty() {
                        Err(error)
                    } else {
                        Ok(legacy)
                    }
                }
            }
        })
    }));
    create_provider(create)
}

/// A `Models` collection preloaded with every built-in provider.
pub fn builtin_models(options: CreateModelsOptions) -> Models {
    let models = create_models(options);
    for provider in builtin_providers() {
        models.set_provider(provider);
    }
    models
}

/// Process-wide `Models` collection backing the compat stream surface (pi
/// `compat.ts` `compatModels`). Lazily initialized on first use; clones share
/// providers and credential state.
pub fn compat_models() -> Models {
    static COMPAT_MODELS: std::sync::OnceLock<Models> = std::sync::OnceLock::new();
    COMPAT_MODELS
        .get_or_init(|| builtin_models(CreateModelsOptions::default()))
        .clone()
}

/// All built-in image-generation providers, freshly constructed.
pub fn builtin_images_providers() -> Vec<Arc<dyn crate::images_models::ImagesProvider>> {
    let mut options = crate::images_models::CreateImagesProviderOptions::new(
        "openrouter",
        builtin_provider_auth("openrouter"),
    );
    options.models = crate::image_models::get_image_models("openrouter");
    vec![crate::images_models::create_images_provider(options)]
}

/// An `ImagesModels` collection with every built-in image provider registered.
pub fn builtin_images_models(options: CreateModelsOptions) -> crate::images_models::ImagesModels {
    let models = crate::images_models::create_images_models(options);
    for provider in builtin_images_providers() {
        models.set_provider(provider);
    }
    models
}
