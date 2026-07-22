//! Default auth context, mirroring pi `auth/context.ts`.

use super::types::AuthContext;
use async_trait::async_trait;

/// Default auth context: env vars from the process environment, file
/// existence via the filesystem with a leading `~` expanded to the home
/// directory.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultAuthContext;

#[async_trait]
impl AuthContext for DefaultAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
    }

    async fn file_exists(&self, path: &str) -> bool {
        let resolved = if let Some(rest) = path.strip_prefix('~') {
            match dirs_home() {
                Some(home) => format!("{home}{rest}"),
                None => return false,
            }
        } else {
            path.to_owned()
        };
        tokio::fs::metadata(&resolved).await.is_ok()
    }
}

fn dirs_home() -> Option<String> {
    std::env::var("HOME")
        .ok()
        .filter(|home| !home.is_empty())
        .or_else(|| {
            std::env::var("USERPROFILE")
                .ok()
                .filter(|home| !home.is_empty())
        })
}
