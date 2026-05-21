//! Session-scoped provider resource cleanup.
//!
//! Pi exposes a process-global cleanup hook registry for provider resources.
//! The Rust API keeps the public behavior explicit: callers ask the crate to
//! clean resources for one session or all sessions, and the implementation
//! releases the resources owned by Rust-native providers.

use crate::http_api_provider::cleanup_openai_codex_websocket_sessions;
use std::{
    collections::BTreeMap,
    fmt,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, LazyLock,
        atomic::{AtomicU64, Ordering},
    },
};

use parking_lot::RwLock;

type SessionResourceCleanup = Arc<dyn Fn(Option<&str>) -> Result<(), String> + Send + Sync>;

static SESSION_RESOURCE_CLEANUPS: LazyLock<RwLock<BTreeMap<u64, SessionResourceCleanup>>> =
    LazyLock::new(|| RwLock::new(BTreeMap::new()));
static NEXT_SESSION_RESOURCE_CLEANUP_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SessionResourceCleanupReport {
    pub openai_codex_websocket_sessions: usize,
}

impl SessionResourceCleanupReport {
    pub fn cleaned_count(self) -> usize {
        self.openai_codex_websocket_sessions
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionResourceCleanupError {
    pub report: SessionResourceCleanupReport,
    pub errors: Vec<String>,
}

impl fmt::Display for SessionResourceCleanupError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Failed to cleanup session resources")?;
        if !self.errors.is_empty() {
            write!(formatter, ": {}", self.errors.join("; "))?;
        }
        Ok(())
    }
}

impl std::error::Error for SessionResourceCleanupError {}

#[derive(Debug)]
pub struct SessionResourceCleanupRegistration {
    id: u64,
}

impl SessionResourceCleanupRegistration {
    pub fn unregister(self) {
        unregister_session_resource_cleanup(self.id);
    }
}

impl Drop for SessionResourceCleanupRegistration {
    fn drop(&mut self) {
        unregister_session_resource_cleanup(self.id);
    }
}

pub fn register_session_resource_cleanup<F>(cleanup: F) -> SessionResourceCleanupRegistration
where
    F: Fn(Option<&str>) -> Result<(), String> + Send + Sync + 'static,
{
    let id = NEXT_SESSION_RESOURCE_CLEANUP_ID.fetch_add(1, Ordering::Relaxed);
    SESSION_RESOURCE_CLEANUPS
        .write()
        .insert(id, Arc::new(cleanup));
    SessionResourceCleanupRegistration { id }
}

pub async fn cleanup_session_resources(
    session_id: Option<&str>,
) -> Result<SessionResourceCleanupReport, SessionResourceCleanupError> {
    let report = SessionResourceCleanupReport {
        openai_codex_websocket_sessions: cleanup_openai_codex_websocket_sessions(session_id).await,
    };
    let errors = run_registered_session_resource_cleanups(session_id);

    if errors.is_empty() {
        Ok(report)
    } else {
        Err(SessionResourceCleanupError { report, errors })
    }
}

fn unregister_session_resource_cleanup(id: u64) -> bool {
    SESSION_RESOURCE_CLEANUPS.write().remove(&id).is_some()
}

fn run_registered_session_resource_cleanups(session_id: Option<&str>) -> Vec<String> {
    let cleanups = SESSION_RESOURCE_CLEANUPS
        .read()
        .values()
        .cloned()
        .collect::<Vec<_>>();
    let mut errors = Vec::new();

    for cleanup in cleanups {
        match catch_unwind(AssertUnwindSafe(|| cleanup(session_id))) {
            Ok(Ok(())) => {}
            Ok(Err(error)) => errors.push(error),
            Err(payload) => errors.push(format_panic_payload(payload)),
        }
    }

    errors
}

fn format_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "session resource cleanup panicked".to_owned()
    }
}
