//! Session-scoped provider resource cleanup.
//!
//! Pi exposes a process-global cleanup hook registry for provider resources.
//! The Rust API keeps the public behavior explicit: callers ask the crate to
//! clean resources for one session or all sessions, and the implementation
//! releases the resources owned by Rust-native providers.

use crate::http_api_provider::cleanup_openai_codex_websocket_sessions;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionResourceCleanupReport {
    pub openai_codex_websocket_sessions: usize,
}

impl SessionResourceCleanupReport {
    pub fn cleaned_count(self) -> usize {
        self.openai_codex_websocket_sessions
    }
}

pub async fn cleanup_session_resources(session_id: Option<&str>) -> SessionResourceCleanupReport {
    SessionResourceCleanupReport {
        openai_codex_websocket_sessions: cleanup_openai_codex_websocket_sessions(session_id).await,
    }
}
