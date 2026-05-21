use crate::{
    anthropic_oauth::{OAuthCredentials, refresh_anthropic_token_at},
    github_copilot_oauth::{GitHubCopilotCredentials, refresh_github_copilot_token_at},
    openai_codex_oauth::refresh_openai_codex_token_at,
    types::now_millis,
};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::LazyLock,
};

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

pub type AuthStorage = BTreeMap<String, AuthCredential>;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthCredential {
    ApiKey {
        key: String,
    },
    #[serde(rename = "oauth")]
    OAuth {
        #[serde(flatten)]
        credentials: StoredOAuthCredentials,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoredOAuthCredentials {
    pub refresh: String,
    pub access: String,
    pub expires: i64,
    #[serde(default, flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthApiKeyResolution {
    pub api_key: String,
    pub credentials: Option<StoredOAuthCredentials>,
    pub refreshed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OAuthProviderInfo {
    pub id: String,
    pub name: String,
    pub uses_callback_server: bool,
}

#[async_trait]
pub trait OAuthTokenRefresher: Send + Sync {
    async fn refresh_token(
        &self,
        provider_id: &str,
        credentials: &StoredOAuthCredentials,
        now_millis: i64,
    ) -> Result<StoredOAuthCredentials, String>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct BuiltInOAuthTokenRefresher;

static OAUTH_PROVIDER_REGISTRY: LazyLock<RwLock<BTreeMap<String, OAuthProviderInfo>>> =
    LazyLock::new(|| RwLock::new(built_in_oauth_provider_map()));

impl StoredOAuthCredentials {
    pub fn enterprise_domain(&self) -> Option<&str> {
        ["enterpriseUrl", "enterprise_url"]
            .into_iter()
            .find_map(|key| self.extra.get(key).and_then(Value::as_str))
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    fn with_enterprise_domain(mut self, enterprise_domain: Option<String>) -> Self {
        self.extra.remove("enterprise_url");
        if let Some(enterprise_domain) = enterprise_domain.filter(|value| !value.trim().is_empty())
        {
            self.extra
                .insert("enterpriseUrl".to_owned(), Value::String(enterprise_domain));
        } else {
            self.extra.remove("enterpriseUrl");
        }
        self
    }
}

impl From<OAuthCredentials> for StoredOAuthCredentials {
    fn from(credentials: OAuthCredentials) -> Self {
        Self {
            refresh: credentials.refresh,
            access: credentials.access,
            expires: credentials.expires,
            extra: credentials.extra,
        }
    }
}

impl From<GitHubCopilotCredentials> for StoredOAuthCredentials {
    fn from(credentials: GitHubCopilotCredentials) -> Self {
        StoredOAuthCredentials {
            refresh: credentials.refresh,
            access: credentials.access,
            expires: credentials.expires,
            extra: BTreeMap::new(),
        }
        .with_enterprise_domain(credentials.enterprise_url)
    }
}

pub fn built_in_oauth_providers() -> Vec<OAuthProviderInfo> {
    vec![
        OAuthProviderInfo {
            id: "anthropic".to_owned(),
            name: "Anthropic (Claude Pro/Max)".to_owned(),
            uses_callback_server: true,
        },
        OAuthProviderInfo {
            id: "github-copilot".to_owned(),
            name: "GitHub Copilot".to_owned(),
            uses_callback_server: false,
        },
        OAuthProviderInfo {
            id: "openai-codex".to_owned(),
            name: "ChatGPT Plus/Pro (Codex Subscription)".to_owned(),
            uses_callback_server: true,
        },
    ]
}

pub fn get_oauth_provider(id: &str) -> Option<OAuthProviderInfo> {
    OAUTH_PROVIDER_REGISTRY.read().get(id).cloned()
}

pub fn get_oauth_providers() -> Vec<OAuthProviderInfo> {
    OAUTH_PROVIDER_REGISTRY.read().values().cloned().collect()
}

pub fn get_oauth_provider_info_list() -> Vec<OAuthProviderInfo> {
    get_oauth_providers()
}

pub fn register_oauth_provider(info: OAuthProviderInfo) {
    OAUTH_PROVIDER_REGISTRY
        .write()
        .insert(info.id.clone(), info);
}

pub fn unregister_oauth_provider(id: &str) {
    let mut registry = OAUTH_PROVIDER_REGISTRY.write();
    if let Some(info) = built_in_oauth_providers()
        .into_iter()
        .find(|provider| provider.id == id)
    {
        registry.insert(id.to_owned(), info);
    } else {
        registry.remove(id);
    }
}

pub fn reset_oauth_providers() {
    *OAUTH_PROVIDER_REGISTRY.write() = built_in_oauth_provider_map();
}

fn built_in_oauth_provider_map() -> BTreeMap<String, OAuthProviderInfo> {
    built_in_oauth_providers()
        .into_iter()
        .map(|info| (info.id.clone(), info))
        .collect()
}

#[async_trait]
impl OAuthTokenRefresher for BuiltInOAuthTokenRefresher {
    async fn refresh_token(
        &self,
        provider_id: &str,
        credentials: &StoredOAuthCredentials,
        now_millis: i64,
    ) -> Result<StoredOAuthCredentials, String> {
        match provider_id {
            "anthropic" => refresh_anthropic_token_at(&credentials.refresh, now_millis)
                .await
                .map(StoredOAuthCredentials::from),
            "github-copilot" => {
                let refreshed = refresh_github_copilot_token_at(
                    &credentials.refresh,
                    credentials.enterprise_domain(),
                    now_millis,
                )
                .await?;
                Ok(StoredOAuthCredentials::from(refreshed))
            }
            "openai-codex" => refresh_openai_codex_token_at(&credentials.refresh, now_millis)
                .await
                .map(StoredOAuthCredentials::from),
            _ => Err(format!("Unknown OAuth provider: {provider_id}")),
        }
    }
}

pub fn default_auth_storage_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".pi").join("agent").join("auth.json"))
}

pub fn load_auth_storage_from_path(path: impl AsRef<Path>) -> Result<AuthStorage, String> {
    let path = path.as_ref();
    let Ok(content) = fs::read_to_string(path) else {
        return Ok(AuthStorage::new());
    };
    serde_json::from_str(&content).or_else(|_| Ok(AuthStorage::new()))
}

pub fn save_auth_storage_to_path(
    path: impl AsRef<Path>,
    storage: &AuthStorage,
) -> Result<(), String> {
    let path = path.as_ref();
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "Failed to create auth storage directory {}: {error}",
                parent.display()
            )
        })?;
        set_private_dir_permissions(parent)?;
    }
    let content = serde_json::to_string_pretty(storage)
        .map_err(|error| format!("Failed to serialize auth storage: {error}"))?;
    fs::write(path, content)
        .map_err(|error| format!("Failed to write auth storage {}: {error}", path.display()))?;
    set_private_file_permissions(path)?;
    Ok(())
}

pub async fn resolve_auth_storage_api_key(
    provider_id: &str,
) -> Result<Option<OAuthApiKeyResolution>, String> {
    let Some(path) = default_auth_storage_path() else {
        return Ok(None);
    };
    resolve_auth_storage_api_key_from_path(provider_id, path).await
}

pub async fn resolve_auth_storage_api_key_from_path(
    provider_id: &str,
    path: impl AsRef<Path>,
) -> Result<Option<OAuthApiKeyResolution>, String> {
    resolve_auth_storage_api_key_from_path_with_refresher_at(
        provider_id,
        path,
        now_millis() as i64,
        &BuiltInOAuthTokenRefresher,
    )
    .await
}

pub async fn resolve_auth_storage_api_key_from_path_with_refresher_at<R>(
    provider_id: &str,
    path: impl AsRef<Path>,
    now_millis: i64,
    refresher: &R,
) -> Result<Option<OAuthApiKeyResolution>, String>
where
    R: OAuthTokenRefresher,
{
    let path = path.as_ref();
    let mut storage = load_auth_storage_from_path(path)?;
    let Some(entry) = storage.get(provider_id).cloned() else {
        return Ok(None);
    };

    match entry {
        AuthCredential::ApiKey { key } => Ok(Some(OAuthApiKeyResolution {
            api_key: key,
            credentials: None,
            refreshed: false,
        })),
        AuthCredential::OAuth { mut credentials } => {
            if get_oauth_provider(provider_id).is_none() {
                return Err(format!("Unknown OAuth provider: {provider_id}"));
            }
            let mut refreshed = false;
            if now_millis >= credentials.expires {
                credentials = refresher
                    .refresh_token(provider_id, &credentials, now_millis)
                    .await
                    .map_err(|error| {
                        format!("Failed to refresh OAuth token for {provider_id}: {error}")
                    })?;
                storage.insert(
                    provider_id.to_owned(),
                    AuthCredential::OAuth {
                        credentials: credentials.clone(),
                    },
                );
                save_auth_storage_to_path(path, &storage)?;
                refreshed = true;
            }
            Ok(Some(OAuthApiKeyResolution {
                api_key: credentials.access.clone(),
                credentials: Some(credentials),
                refreshed,
            }))
        }
    }
}

fn set_private_dir_permissions(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(|error| {
            format!(
                "Failed to set auth storage directory permissions {}: {error}",
                path.display()
            )
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

fn set_private_file_permissions(path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600)).map_err(|error| {
            format!(
                "Failed to set auth storage file permissions {}: {error}",
                path.display()
            )
        })?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CurrentDirGuard {
        original: PathBuf,
    }

    impl CurrentDirGuard {
        fn enter(path: &Path) -> Self {
            let original = std::env::current_dir().expect("current dir");
            std::env::set_current_dir(path).expect("set test current dir");
            Self { original }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            std::env::set_current_dir(&self.original).expect("restore current dir");
        }
    }

    #[test]
    fn save_auth_storage_supports_current_directory_auth_file() {
        let dir = std::env::temp_dir().join(format!(
            "ri-auth-storage-current-dir-{}-{}",
            std::process::id(),
            now_millis()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        let _guard = CurrentDirGuard::enter(&dir);

        let mut storage = AuthStorage::new();
        storage.insert(
            "anthropic".to_owned(),
            AuthCredential::OAuth {
                credentials: StoredOAuthCredentials {
                    refresh: "refresh".to_owned(),
                    access: "access".to_owned(),
                    expires: 123,
                    extra: BTreeMap::new(),
                },
            },
        );

        save_auth_storage_to_path("auth.json", &storage).expect("save auth.json");
        let loaded = load_auth_storage_from_path("auth.json").expect("load auth.json");
        assert_eq!(loaded, storage);
    }
}
