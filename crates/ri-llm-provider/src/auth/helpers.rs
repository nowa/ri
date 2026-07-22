//! Reusable auth building blocks, mirroring pi `auth/helpers.ts`.

use super::types::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthInteraction, AuthPrompt, AuthResult, ModelAuth,
};
use async_trait::async_trait;
use std::sync::Arc;

/// Standard api-key auth: a stored credential key wins, otherwise the first
/// set env var resolves. Includes a `login` that prompts for the key.
/// Providers with non-standard resolution (provider env, ambient files, IAM)
/// write their own [`ApiKeyAuth`].
pub fn env_api_key_auth(
    name: impl Into<String>,
    env_vars: impl IntoIterator<Item = impl Into<String>>,
) -> Arc<dyn ApiKeyAuth> {
    Arc::new(EnvApiKeyAuth {
        name: name.into(),
        env_vars: env_vars.into_iter().map(Into::into).collect(),
    })
}

struct EnvApiKeyAuth {
    name: String,
    env_vars: Vec<String>,
}

#[async_trait]
impl ApiKeyAuth for EnvApiKeyAuth {
    fn name(&self) -> &str {
        &self.name
    }

    fn supports_login(&self) -> bool {
        true
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<ApiKeyCredential, String> {
        let key = interaction
            .prompt(AuthPrompt::Secret {
                message: format!("Enter {}", self.name),
                placeholder: None,
            })
            .await?;
        Ok(ApiKeyCredential {
            key: Some(key),
            env: Default::default(),
        })
    }

    async fn resolve(
        &self,
        ctx: &dyn AuthContext,
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
        for env_var in &self.env_vars {
            if let Some(value) = ctx.env(env_var).await {
                return Ok(Some(AuthResult {
                    auth: ModelAuth {
                        api_key: Some(value),
                        ..Default::default()
                    },
                    env: Default::default(),
                    source: Some(env_var.clone()),
                }));
            }
        }
        Ok(None)
    }
}
