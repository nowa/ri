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
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthEvent, AuthInfoLink, AuthInteraction,
    AuthPrompt, AuthPromptOption, AuthResult, Credential, ModelAuth, OAuthAuth, OAuthCredential,
    ProviderAuth, env_api_key_auth,
};
use crate::models_runtime::{
    CreateModelsOptions, CreateProviderOptions, Models, Provider, ProviderApiDispatch,
    create_models, create_provider,
};
use crate::oauth_auth_storage::{StoredOAuthCredentials, get_oauth_provider, refresh_oauth_token};
use crate::{
    Model, ensure_builtin_api_providers, get_api_provider, github_copilot_base_url,
    seed_provider_ids, seed_provider_models,
};
use async_trait::async_trait;
use serde_json::Value;
use std::{collections::BTreeSet, sync::Arc};

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

/// Amazon Bedrock auth (pi `providers/amazon-bedrock.ts`): a stored bearer
/// token or AWS profile, or the ambient AWS credential chain.
struct AmazonBedrockApiKeyAuth;

#[async_trait]
impl ApiKeyAuth for AmazonBedrockApiKeyAuth {
    fn name(&self) -> &str {
        "AWS credentials or bearer token"
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<ApiKeyCredential, String> {
        let method = interaction
            .prompt(AuthPrompt::Select {
                message: "Select Amazon Bedrock authentication method:".to_owned(),
                options: vec![
                    AuthPromptOption {
                        id: "bearer-token".to_owned(),
                        label: "Bearer token".to_owned(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: "aws-profile".to_owned(),
                        label: "AWS profile".to_owned(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: "credential-chain".to_owned(),
                        label: "Existing AWS credential chain".to_owned(),
                        description: None,
                    },
                ],
            })
            .await?;
        if method == "bearer-token" {
            let key = interaction
                .prompt(AuthPrompt::Secret {
                    message: "Enter Amazon Bedrock bearer token".to_owned(),
                    placeholder: None,
                })
                .await?;
            return Ok(ApiKeyCredential {
                key: Some(key),
                env: Default::default(),
            });
        }
        interaction.notify(AuthEvent::Info {
            message:
                "Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials."
                    .to_owned(),
            links: vec![AuthInfoLink {
                url:
                    "https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html"
                        .to_owned(),
                label: Some("AWS credential provider chain".to_owned()),
            }],
        });
        if method == "aws-profile" {
            let profile = interaction
                .prompt(AuthPrompt::Text {
                    message: "Enter AWS profile name".to_owned(),
                    placeholder: None,
                })
                .await?;
            return Ok(ApiKeyCredential {
                key: None,
                env: [("AWS_PROFILE".to_owned(), profile)].into_iter().collect(),
            });
        }
        if method != "credential-chain" {
            return Err(format!("Unknown Amazon Bedrock auth method: {method}"));
        }
        interaction
            .prompt(AuthPrompt::Text {
                message: "Configure AWS credentials, then press Enter to continue".to_owned(),
                placeholder: None,
            })
            .await?;
        Ok(ApiKeyCredential::default())
    }

    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthResult>, String> {
        let stored_env = credential.map(|credential| credential.env.clone());
        if let Some(key) = credential.and_then(|credential| credential.key.clone()) {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    ..Default::default()
                },
                env: stored_env.unwrap_or_default(),
                source: Some("stored credential".to_owned()),
            }));
        }
        if ctx.env("AWS_BEARER_TOKEN_BEDROCK").await.is_some() {
            return Ok(Some(ambient_auth_result(None, "AWS_BEARER_TOKEN_BEDROCK")));
        }
        let stored_profile = credential.and_then(|credential| credential.env.get("AWS_PROFILE"));
        if stored_profile.is_some() {
            return Ok(Some(ambient_auth_result(stored_env, "stored credential")));
        }
        if ctx.env("AWS_PROFILE").await.is_some() {
            return Ok(Some(ambient_auth_result(stored_env, "AWS_PROFILE")));
        }
        if ctx.env("AWS_ACCESS_KEY_ID").await.is_some()
            && ctx.env("AWS_SECRET_ACCESS_KEY").await.is_some()
        {
            return Ok(Some(ambient_auth_result(None, "AWS access keys")));
        }
        if ctx
            .env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
            .await
            .is_some()
            || ctx
                .env("AWS_CONTAINER_CREDENTIALS_FULL_URI")
                .await
                .is_some()
        {
            return Ok(Some(ambient_auth_result(None, "ECS task role")));
        }
        if ctx.env("AWS_WEB_IDENTITY_TOKEN_FILE").await.is_some() {
            return Ok(Some(ambient_auth_result(None, "web identity token")));
        }
        Ok(None)
    }
}

/// Google Vertex AI auth (pi `providers/google-vertex.ts`): an explicit API key
/// or Application Default Credentials plus project/location configuration.
struct GoogleVertexApiKeyAuth;

const VERTEX_ADC_PATH: &str = "~/.config/gcloud/application_default_credentials.json";

#[async_trait]
impl ApiKeyAuth for GoogleVertexApiKeyAuth {
    fn name(&self) -> &str {
        "Google Cloud credentials"
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<ApiKeyCredential, String> {
        let method = interaction
            .prompt(AuthPrompt::Select {
                message: "Select Google Vertex AI authentication method:".to_owned(),
                options: vec![
                    AuthPromptOption {
                        id: "api-key".to_owned(),
                        label: "Google Cloud API key".to_owned(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: "adc".to_owned(),
                        label: "Application Default Credentials".to_owned(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: "service-account".to_owned(),
                        label: "Service account credentials file".to_owned(),
                        description: None,
                    },
                ],
            })
            .await?;
        if method == "api-key" {
            let key = interaction
                .prompt(AuthPrompt::Secret {
                    message: "Enter Google Cloud API key".to_owned(),
                    placeholder: None,
                })
                .await?;
            return Ok(ApiKeyCredential {
                key: Some(key),
                env: Default::default(),
            });
        }
        if method != "adc" && method != "service-account" {
            return Err(format!("Unknown Google Vertex AI auth method: {method}"));
        }
        interaction.notify(AuthEvent::Info {
            message: if method == "adc" {
                "Run `gcloud auth application-default login`, then provide the project and location."
                    .to_owned()
            } else {
                "Provide a service account credentials file, project, and location.".to_owned()
            },
            links: vec![AuthInfoLink {
                url: "https://cloud.google.com/docs/authentication/provide-credentials-adc"
                    .to_owned(),
                label: Some("Application Default Credentials".to_owned()),
            }],
        });
        let project = interaction
            .prompt(AuthPrompt::Text {
                message: "Enter Google Cloud project ID".to_owned(),
                placeholder: None,
            })
            .await?;
        let location = interaction
            .prompt(AuthPrompt::Text {
                message: "Enter Google Cloud location".to_owned(),
                placeholder: None,
            })
            .await?;
        let mut env: std::collections::BTreeMap<String, String> = [
            ("GOOGLE_CLOUD_PROJECT".to_owned(), project),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), location),
        ]
        .into_iter()
        .collect();
        if method == "service-account" {
            let credentials_path = interaction
                .prompt(AuthPrompt::Text {
                    message: "Enter service account credentials file path".to_owned(),
                    placeholder: None,
                })
                .await?;
            env.insert(
                "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
                credentials_path,
            );
        }
        Ok(ApiKeyCredential { key: None, env })
    }

    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
        credential: Option<&ApiKeyCredential>,
    ) -> Result<Option<AuthResult>, String> {
        let stored =
            |name: &str| credential.and_then(|credential| credential.env.get(name).cloned());
        if let Some(key) = credential.and_then(|credential| credential.key.clone()) {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    ..Default::default()
                },
                env: Default::default(),
                source: Some("stored credential".to_owned()),
            }));
        }
        if let Some(key) = ctx.env("GOOGLE_CLOUD_API_KEY").await {
            return Ok(Some(AuthResult {
                auth: ModelAuth {
                    api_key: Some(key),
                    ..Default::default()
                },
                env: Default::default(),
                source: Some("GOOGLE_CLOUD_API_KEY".to_owned()),
            }));
        }

        let adc_path = match stored("GOOGLE_APPLICATION_CREDENTIALS") {
            Some(path) => Some(path),
            None => ctx.env("GOOGLE_APPLICATION_CREDENTIALS").await,
        };
        let has_credentials = ctx
            .file_exists(adc_path.as_deref().unwrap_or(VERTEX_ADC_PATH))
            .await;
        let project = match stored("GOOGLE_CLOUD_PROJECT") {
            Some(project) => Some(project),
            None => match ctx.env("GOOGLE_CLOUD_PROJECT").await {
                Some(project) => Some(project),
                None => ctx.env("GCLOUD_PROJECT").await,
            },
        };
        let location = match stored("GOOGLE_CLOUD_LOCATION") {
            Some(location) => Some(location),
            None => ctx.env("GOOGLE_CLOUD_LOCATION").await,
        };
        if has_credentials && project.is_some() && location.is_some() {
            return Ok(Some(ambient_auth_result(
                credential.map(|credential| credential.env.clone()),
                if credential.is_some() {
                    "stored credential"
                } else {
                    "gcloud application default credentials"
                },
            )));
        }
        Ok(None)
    }
}

fn ambient_auth_result(
    env: Option<std::collections::BTreeMap<String, String>>,
    source: &str,
) -> AuthResult {
    AuthResult {
        auth: ModelAuth::default(),
        env: env.unwrap_or_default(),
        source: Some(source.to_owned()),
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
        "amazon-bedrock" => Arc::new(AmazonBedrockApiKeyAuth),
        "google-vertex" => Arc::new(GoogleVertexApiKeyAuth),
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

/// pi `githubCopilotProvider.filterModels`: an OAuth credential carrying a
/// string `availableModelIds` array restricts the catalog to entitled models;
/// anything else leaves the list untouched.
fn filter_github_copilot_models(models: Vec<Model>, credential: Option<&Credential>) -> Vec<Model> {
    let Some(Credential::OAuth(credential)) = credential else {
        return models;
    };
    let Some(available) = credential.extra.get("availableModelIds") else {
        return models;
    };
    let Some(entries) = available.as_array() else {
        return models;
    };
    if !entries.iter().all(Value::is_string) {
        return models;
    }
    let available: BTreeSet<&str> = entries.iter().filter_map(Value::as_str).collect();
    models
        .into_iter()
        .filter(|model| available.contains(model.id.as_str()))
        .collect()
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
    if provider_id == "github-copilot" {
        options.filter_models = Some(Arc::new(filter_github_copilot_models));
    }
    Some(create_provider(options))
}

/// All built-in catalog providers, in catalog order, plus the Radius
/// gateway provider (dynamic catalog, no static seed).
pub fn builtin_providers() -> Vec<Arc<dyn Provider>> {
    let mut providers: Vec<Arc<dyn Provider>> = seed_provider_ids()
        .into_iter()
        .filter_map(|provider_id| builtin_provider(&provider_id))
        .collect();
    // pi registers providers alphabetically (all.ts), radius included.
    let radius = radius_provider(RadiusProviderOptions::default());
    let position = providers
        .iter()
        .position(|provider| provider.id() > radius.id())
        .unwrap_or(providers.len());
    providers.insert(position, radius);
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
