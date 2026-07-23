//! SQLite session storage (pi `@earendil-works/pi-storage-sqlite-node`).
//!
//! Full pi architecture: reads are lazy and writes materialize derived state.
//!
//! Data flow on append (one transaction, mirroring pi `appendEntry`):
//! `session_entries` row + `session_sequences` advance, the session-level
//! reduced state into `session_materialized`, per-entry derived rows into
//! `entry_materialized` (labels), the durable leaf pointer onto
//! `sessions.active_leaf_id`, and the active branch path into
//! `branch_entries` — rebuilt only when branch membership changes (leaf
//! navigation, or a fork from a parent that already has a child); linear
//! appends extend the active branch by one row. On failure the transaction
//! rolls back and the in-memory state snapshot is restored.
//!
//! Data flow on open: only the `sessions` row, the materialized summary, the
//! label rows, and the newest branch id load — entries stay in the database
//! and are fetched on demand (`get_entry` point queries with a decode cache,
//! `find_entries` by type, `entries_page` by sequence cursor). The active
//! leaf's path resolves through the materialized branch table.

use super::session::{
    SessionEntryCursorOptions, SessionEntryMessage, SessionError, SessionErrorCode,
    SessionForkOptions, SessionModelSelection, SessionStorageKind, SessionTreeEntry,
    entries_to_fork, generate_entry_id, get_path_to_root, get_path_to_root_or_compaction,
    leaf_id_after_entry, now_iso, storage_error,
};
use ri_llm_provider::Message;
use rusqlite::Connection;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

const SQLITE_MIGRATIONS: &[(&str, &str)] = &[(
    "001_initial.sql",
    include_str!("sqlite_migrations/001_initial.sql"),
)];

// ============================================================================
// Materialized session state (pi `session-materialized.ts`)
// ============================================================================

/// A (provider, model, thinking level) combination seen in the session.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ModelThinkingConfig {
    pub provider: String,
    pub model_id: String,
    pub thinking_level: String,
}

/// Aggregated session statistics derived from the materialized state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SessionStats {
    pub message_count: u64,
    pub cached_tokens: u64,
    pub uncached_tokens: u64,
    pub total_tokens: u64,
    pub cost_total: f64,
}

/// Session-level state reduced from appended entries, persisted as the
/// `session_materialized` summary plus per-entry `entry_materialized` rows.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionMaterializedState {
    pub name: Option<String>,
    pub message_count: u64,
    pub cached_tokens: u64,
    pub uncached_tokens: u64,
    pub total_tokens: u64,
    pub cost_total: f64,
    pub labels_by_id: BTreeMap<String, String>,
    pub model_thinking_configs: Vec<ModelThinkingConfig>,
    pub current_model: Option<SessionModelSelection>,
    pub current_thinking_level: Option<String>,
}

fn is_thinking_level(value: &str) -> bool {
    matches!(
        value,
        "off" | "minimal" | "low" | "medium" | "high" | "xhigh"
    )
}

fn add_model_thinking_config(
    state: &mut SessionMaterializedState,
    provider: &str,
    model_id: &str,
    thinking_level: &str,
) {
    let config = ModelThinkingConfig {
        provider: provider.to_owned(),
        model_id: model_id.to_owned(),
        thinking_level: thinking_level.to_owned(),
    };
    if !state.model_thinking_configs.contains(&config) {
        state.model_thinking_configs.push(config);
        state.model_thinking_configs.sort();
    }
}

/// Reduce one appended entry into the session-level state.
pub fn apply_entry_to_materialized_state(
    state: &mut SessionMaterializedState,
    entry: &SessionTreeEntry,
) {
    match entry {
        SessionTreeEntry::SessionInfo { name, .. } => {
            state.name = name
                .as_deref()
                .map(str::trim)
                .filter(|name| !name.is_empty())
                .map(ToOwned::to_owned);
        }
        SessionTreeEntry::Label {
            target_id, label, ..
        } => {
            let label = label.as_deref().map(str::trim).unwrap_or_default();
            if label.is_empty() {
                state.labels_by_id.remove(target_id);
            } else {
                state
                    .labels_by_id
                    .insert(target_id.clone(), label.to_owned());
            }
        }
        SessionTreeEntry::ModelChange {
            provider, model_id, ..
        } => {
            state.current_model = Some(SessionModelSelection {
                provider: provider.clone(),
                model_id: model_id.clone(),
            });
            if let Some(level) = state.current_thinking_level.clone() {
                add_model_thinking_config(state, provider, model_id, &level);
            }
        }
        SessionTreeEntry::ThinkingLevelChange { thinking_level, .. } => {
            if !is_thinking_level(thinking_level) {
                return;
            }
            state.current_thinking_level = Some(thinking_level.clone());
            if let Some(model) = state.current_model.clone() {
                add_model_thinking_config(state, &model.provider, &model.model_id, thinking_level);
            }
        }
        SessionTreeEntry::Message { message, .. } => {
            state.message_count += 1;
            let SessionEntryMessage::Llm(Message::Assistant(assistant)) = message else {
                return;
            };
            let usage = &assistant.usage;
            state.cached_tokens += usage.cache_read;
            state.uncached_tokens += usage.input + usage.cache_write;
            state.total_tokens += usage.input + usage.output + usage.cache_read + usage.cache_write;
            state.cost_total += usage.cost.total;
            state.current_model = Some(SessionModelSelection {
                provider: assistant.provider.clone(),
                model_id: assistant.model.clone(),
            });
            if let Some(level) = state.current_thinking_level.clone() {
                add_model_thinking_config(state, &assistant.provider, &assistant.model, &level);
            }
        }
        SessionTreeEntry::Compaction { usage, .. }
        | SessionTreeEntry::BranchSummary { usage, .. } => {
            let Some(usage) = usage else {
                return;
            };
            state.cached_tokens += usage.cache_read;
            state.uncached_tokens += usage.input + usage.cache_write;
            state.total_tokens += usage.input + usage.output + usage.cache_read + usage.cache_write;
            state.cost_total += usage.cost.total;
        }
        SessionTreeEntry::ActiveToolsChange { .. }
        | SessionTreeEntry::Custom { .. }
        | SessionTreeEntry::CustomMessage { .. }
        | SessionTreeEntry::Leaf { .. } => {}
    }
}

/// Reduce a full entry list into session statistics (in-memory backends).
pub fn session_stats_from_entries(entries: &[SessionTreeEntry]) -> SessionStats {
    let mut state = SessionMaterializedState::default();
    for entry in entries {
        apply_entry_to_materialized_state(&mut state, entry);
    }
    session_stats_from_materialized_state(&state)
}

pub fn session_stats_from_materialized_state(state: &SessionMaterializedState) -> SessionStats {
    SessionStats {
        message_count: state.message_count,
        cached_tokens: state.cached_tokens,
        uncached_tokens: state.uncached_tokens,
        total_tokens: state.total_tokens,
        cost_total: state.cost_total,
    }
}

fn invalid_session(message: impl std::fmt::Display) -> SessionError {
    SessionError::new(
        SessionErrorCode::InvalidSession,
        format!("Invalid SQLite session: {message}"),
    )
}

/// The `session_materialized` summary payload, in pi's wire format.
fn serialize_summary(state: &SessionMaterializedState) -> String {
    let mut summary = serde_json::Map::new();
    if let Some(name) = &state.name {
        summary.insert("name".to_owned(), Value::String(name.clone()));
    }
    summary.insert("messageCount".to_owned(), state.message_count.into());
    summary.insert("cachedTokens".to_owned(), state.cached_tokens.into());
    summary.insert("uncachedTokens".to_owned(), state.uncached_tokens.into());
    summary.insert("totalTokens".to_owned(), state.total_tokens.into());
    summary.insert(
        "costTotal".to_owned(),
        serde_json::Number::from_f64(state.cost_total)
            .map(Value::Number)
            .unwrap_or_else(|| 0.into()),
    );
    summary.insert(
        "currentModel".to_owned(),
        match &state.current_model {
            Some(model) => serde_json::json!({
                "provider": model.provider,
                "modelId": model.model_id,
            }),
            None => Value::Null,
        },
    );
    summary.insert(
        "currentThinkingLevel".to_owned(),
        match &state.current_thinking_level {
            Some(level) => Value::String(level.clone()),
            None => Value::Null,
        },
    );
    Value::Object(summary).to_string()
}

fn parse_summary(payload: &str) -> Result<SessionMaterializedState, SessionError> {
    let parsed: Value = serde_json::from_str(payload).map_err(|error| {
        invalid_session(format!("materialized summary is not valid JSON: {error}"))
    })?;
    let object = parsed
        .as_object()
        .ok_or_else(|| invalid_session("materialized summary is not an object"))?;
    let read_u64 = |key: &str| -> Result<u64, SessionError> {
        object
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| invalid_session("materialized summary has invalid fields"))
    };
    let current_model = match object.get("currentModel") {
        None | Some(Value::Null) => None,
        Some(model) => Some(SessionModelSelection {
            provider: model
                .get("provider")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_session("materialized summary has invalid fields"))?
                .to_owned(),
            model_id: model
                .get("modelId")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid_session("materialized summary has invalid fields"))?
                .to_owned(),
        }),
    };
    let current_thinking_level = match object.get("currentThinkingLevel") {
        None | Some(Value::Null) => None,
        Some(level) => {
            let level = level
                .as_str()
                .filter(|level| is_thinking_level(level))
                .ok_or_else(|| invalid_session("materialized summary has invalid fields"))?;
            Some(level.to_owned())
        }
    };
    Ok(SessionMaterializedState {
        name: object
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(ToOwned::to_owned),
        message_count: read_u64("messageCount")?,
        cached_tokens: read_u64("cachedTokens")?,
        uncached_tokens: read_u64("uncachedTokens")?,
        total_tokens: read_u64("totalTokens")?,
        cost_total: object
            .get("costTotal")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid_session("materialized summary has invalid fields"))?,
        labels_by_id: BTreeMap::new(),
        model_thinking_configs: Vec::new(),
        current_model,
        current_thinking_level,
    })
}

/// Per-entry materialized rows: `(type, payload)` pairs, currently labels.
fn entry_materialized_values(entry: &SessionTreeEntry) -> Vec<(&'static str, String)> {
    match entry {
        SessionTreeEntry::Label {
            target_id, label, ..
        } => vec![(
            "label",
            serde_json::json!({
                "targetId": target_id,
                "label": label.clone().map(Value::String).unwrap_or(Value::Null),
            })
            .to_string(),
        )],
        _ => Vec::new(),
    }
}

fn apply_entry_materialized_row(
    state: &mut SessionMaterializedState,
    entry_seq: i64,
    row_type: &str,
    payload: &str,
) -> Result<(), SessionError> {
    if row_type != "label" {
        return Ok(());
    }
    let parsed: Value = serde_json::from_str(payload).map_err(|error| {
        invalid_session(format!(
            "materialized entry row {entry_seq} is not valid JSON: {error}"
        ))
    })?;
    let target_id = parsed
        .get("targetId")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            invalid_session(format!(
                "materialized label row {entry_seq} is missing targetId"
            ))
        })?;
    let label = match parsed.get("label") {
        None | Some(Value::Null) => "",
        Some(Value::String(label)) => label.trim(),
        Some(_) => {
            return Err(invalid_session(format!(
                "materialized label row {entry_seq} has invalid label"
            )));
        }
    };
    if label.is_empty() {
        state.labels_by_id.remove(target_id);
    } else {
        state
            .labels_by_id
            .insert(target_id.to_owned(), label.to_owned());
    }
    Ok(())
}

// ============================================================================
// Metadata / repo
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SqliteSessionMetadata {
    pub id: String,
    pub created_at: String,
    pub cwd: String,
    /// Path of the SQLite database file holding this session.
    pub path: String,
    pub parent_session_id: Option<String>,
    /// Opaque caller-provided metadata stored on the session row.
    pub metadata: Option<serde_json::Map<String, Value>>,
}

#[derive(Debug, Clone, Default)]
pub struct SqliteSessionCreateOptions {
    /// Explicit session id; a UUIDv7 is generated when absent.
    pub id: Option<String>,
    pub cwd: String,
    pub parent_session_id: Option<String>,
    pub metadata: Option<serde_json::Map<String, Value>>,
}

#[derive(Debug, Clone, Default)]
pub struct SqliteSessionListOptions {
    pub cwd: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SqliteSessionForkOptions {
    pub cwd: String,
    /// Defaults to the source session id.
    pub parent_session_id: Option<String>,
    /// Defaults to the source session's metadata.
    pub metadata: Option<serde_json::Map<String, Value>>,
    pub fork: SessionForkOptions,
}

/// Repository over one SQLite database file containing many sessions.
#[derive(Debug, Clone)]
pub struct SqliteSessionRepo {
    database_path: PathBuf,
}

impl SqliteSessionRepo {
    pub fn new(database_path: impl Into<PathBuf>) -> Self {
        Self {
            database_path: database_path.into(),
        }
    }

    pub fn database_path(&self) -> &Path {
        &self.database_path
    }

    fn open_database(&self) -> Result<Connection, SessionError> {
        if let Some(parent) = self.database_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(storage_error)?;
        }
        let connection = Connection::open(&self.database_path).map_err(storage_error)?;
        configure_sqlite_database(&connection)?;
        apply_sqlite_migrations(&connection)?;
        Ok(connection)
    }

    pub fn create(
        &self,
        options: SqliteSessionCreateOptions,
    ) -> Result<SqliteSessionStorage, SessionError> {
        let connection = self.open_database()?;
        let id = options.id.unwrap_or_else(ri_llm_provider::uuidv7);
        let created_at = now_iso();
        let metadata_json = options
            .metadata
            .as_ref()
            .map(|metadata| serde_json::to_string(metadata).map_err(storage_error))
            .transpose()?;
        connection
            .execute(
                "INSERT INTO sessions (id, created_at, cwd, parent_session_id, metadata, active_leaf_id) VALUES (?1, ?2, ?3, ?4, ?5, NULL)",
                rusqlite::params![
                    id,
                    created_at,
                    options.cwd,
                    options.parent_session_id,
                    metadata_json
                ],
            )
            .map_err(storage_error)?;
        connection
            .execute(
                "INSERT INTO session_sequences (session_id, next_seq) VALUES (?1, 1)",
                rusqlite::params![id],
            )
            .map_err(storage_error)?;
        let state = SessionMaterializedState::default();
        connection
            .execute(
                "INSERT INTO session_materialized (session_id, payload) VALUES (?1, ?2)",
                rusqlite::params![id, serialize_summary(&state)],
            )
            .map_err(storage_error)?;
        let metadata = SqliteSessionMetadata {
            id,
            created_at,
            cwd: options.cwd,
            path: self.database_path.to_string_lossy().into_owned(),
            parent_session_id: options.parent_session_id,
            metadata: options.metadata,
        };
        Ok(SqliteSessionStorage::from_parts(
            connection, metadata, None, None, state,
        ))
    }

    pub fn open(&self, session_id: &str) -> Result<SqliteSessionStorage, SessionError> {
        if !self.database_path.exists() {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {session_id}"),
            ));
        }
        let connection = self.open_database()?;
        let row = load_session_row(&connection, session_id)?.ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {session_id}"),
            )
        })?;
        let metadata = session_row_to_metadata(row.clone(), &self.database_path)?;
        // Lazy open: derived state comes from the materialized rows; entries
        // stay in the database until they are asked for.
        let state = load_materialized_state(&connection, session_id)?;
        let active_branch_id = load_active_branch_id(&connection, session_id)?;
        Ok(SqliteSessionStorage::from_parts(
            connection,
            metadata,
            row.active_leaf_id,
            active_branch_id,
            state,
        ))
    }

    pub fn list(
        &self,
        options: SqliteSessionListOptions,
    ) -> Result<Vec<SqliteSessionMetadata>, SessionError> {
        if !self.database_path.exists() {
            return Ok(Vec::new());
        }
        let connection = self.open_database()?;
        let mut rows = Vec::new();
        let collect =
            |row: &rusqlite::Row<'_>| -> rusqlite::Result<SessionRow> { session_row_from_sql(row) };
        if let Some(cwd) = &options.cwd {
            let mut statement = connection
                .prepare(
                    "SELECT id, created_at, metadata, cwd, parent_session_id, active_leaf_id FROM sessions WHERE cwd = ?1 ORDER BY created_at DESC",
                )
                .map_err(storage_error)?;
            let mapped = statement
                .query_map(rusqlite::params![cwd], collect)
                .map_err(storage_error)?;
            for row in mapped {
                rows.push(row.map_err(storage_error)?);
            }
        } else {
            let mut statement = connection
                .prepare(
                    "SELECT id, created_at, metadata, cwd, parent_session_id, active_leaf_id FROM sessions ORDER BY created_at DESC",
                )
                .map_err(storage_error)?;
            let mapped = statement.query_map([], collect).map_err(storage_error)?;
            for row in mapped {
                rows.push(row.map_err(storage_error)?);
            }
        }
        rows.into_iter()
            .map(|row| session_row_to_metadata(row, &self.database_path))
            .collect()
    }

    /// Fork a session: copy the branch selected by `options.fork` into a new
    /// session row (pi `SqliteSessionRepo.fork`).
    pub fn fork(
        &self,
        source_session_id: &str,
        options: SqliteSessionForkOptions,
    ) -> Result<SqliteSessionStorage, SessionError> {
        let source = self.open(source_session_id)?;
        let source_metadata = source.metadata().clone();
        let source_storage = SessionStorageKind::Sqlite(source);
        let entries = entries_to_fork(&source_storage, &options.fork)?;
        drop(source_storage);
        let mut storage = self.create(SqliteSessionCreateOptions {
            id: options.fork.id.clone(),
            cwd: options.cwd,
            parent_session_id: options
                .parent_session_id
                .or_else(|| Some(source_metadata.id.clone())),
            metadata: options.metadata.or(source_metadata.metadata),
        })?;
        for entry in entries {
            storage.append_entry(entry)?;
        }
        Ok(storage)
    }

    pub fn delete(&self, session_id: &str) -> Result<(), SessionError> {
        let connection = self.open_database()?;
        let changed: Result<usize, SessionError> = (|| {
            connection.execute("BEGIN", []).map_err(storage_error)?;
            for statement in [
                "DELETE FROM branch_entries WHERE session_id = ?1",
                "DELETE FROM session_entries WHERE session_id = ?1",
                "DELETE FROM entry_materialized WHERE session_id = ?1",
                "DELETE FROM session_materialized WHERE session_id = ?1",
                "DELETE FROM session_sequences WHERE session_id = ?1",
            ] {
                connection
                    .execute(statement, rusqlite::params![session_id])
                    .map_err(storage_error)?;
            }
            let changed = connection
                .execute(
                    "DELETE FROM sessions WHERE id = ?1",
                    rusqlite::params![session_id],
                )
                .map_err(storage_error)?;
            connection.execute("COMMIT", []).map_err(storage_error)?;
            Ok(changed)
        })();
        let changed = match changed {
            Ok(changed) => changed,
            Err(error) => {
                let _ = connection.execute("ROLLBACK", []);
                return Err(error);
            }
        };
        if changed == 0 {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Session not found: {session_id}"),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Lazy storage
// ============================================================================

#[derive(Debug, Clone)]
pub struct SqliteSessionStorage {
    connection: Arc<Mutex<Connection>>,
    metadata: SqliteSessionMetadata,
    /// Decode cache over `session_entries`; never authoritative.
    by_id: Arc<Mutex<BTreeMap<String, SessionTreeEntry>>>,
    materialized: SessionMaterializedState,
    leaf_id: Option<String>,
    active_branch_id: Option<String>,
}

impl SqliteSessionStorage {
    fn from_parts(
        connection: Connection,
        metadata: SqliteSessionMetadata,
        leaf_id: Option<String>,
        active_branch_id: Option<String>,
        materialized: SessionMaterializedState,
    ) -> Self {
        Self {
            connection: Arc::new(Mutex::new(connection)),
            metadata,
            by_id: Arc::new(Mutex::new(BTreeMap::new())),
            materialized,
            leaf_id,
            active_branch_id,
        }
    }

    pub fn metadata(&self) -> &SqliteSessionMetadata {
        &self.metadata
    }

    pub fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        if let Some(leaf_id) = self.leaf_id.as_deref()
            && self.get_entry(leaf_id).is_none()
        {
            return Err(SessionError::new(
                SessionErrorCode::InvalidSession,
                format!("Entry {leaf_id} not found"),
            ));
        }
        Ok(self.leaf_id.clone())
    }

    pub fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        if let Some(leaf_id) = leaf_id.as_deref()
            && self.get_entry(leaf_id).is_none()
        {
            return Err(SessionError::new(
                SessionErrorCode::NotFound,
                format!("Entry {leaf_id} not found"),
            ));
        }
        let entry = SessionTreeEntry::Leaf {
            id: self.create_entry_id(),
            parent_id: self.leaf_id.clone(),
            timestamp: now_iso(),
            target_id: leaf_id,
        };
        self.append_entry(entry)
    }

    /// Short-id generation with a database uniqueness check (pi
    /// `createEntryId`).
    pub fn create_entry_id(&self) -> String {
        let connection = self.connection.lock().expect("sqlite connection lock");
        let by_id = self.by_id.lock().expect("sqlite cache lock");
        for _ in 0..100 {
            let id = generate_entry_id(&by_id);
            let exists = connection
                .query_row(
                    "SELECT 1 FROM session_entries WHERE session_id = ?1 AND id = ?2 LIMIT 1",
                    rusqlite::params![self.metadata.id, id],
                    |_| Ok(()),
                )
                .is_ok();
            if !exists {
                return id;
            }
        }
        ri_llm_provider::uuidv7()
    }

    pub fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        let encoded = encode_sqlite_entry(&entry)?;
        // Snapshot the in-memory state for rollback (pi `appendEntry`).
        let previous_materialized = self.materialized.clone();
        let previous_leaf = self.leaf_id.clone();
        let previous_branch = self.active_branch_id.clone();

        apply_entry_to_materialized_state(&mut self.materialized, &entry);
        let new_leaf = leaf_id_after_entry(&entry);

        let result: Result<(), SessionError> = {
            let connection = self.connection.lock().expect("sqlite connection lock");
            let transaction: Result<(), SessionError> = (|| {
                connection.execute("BEGIN", []).map_err(storage_error)?;
                let parent_had_existing_child =
                    has_existing_child(&connection, &self.metadata.id, entry.parent_id())?;
                let next_seq: i64 = connection
                    .query_row(
                        "SELECT next_seq FROM session_sequences WHERE session_id = ?1",
                        rusqlite::params![self.metadata.id],
                        |row| row.get(0),
                    )
                    .map_err(|_| {
                        invalid_session(format!(
                            "missing sequence row for session {}",
                            self.metadata.id
                        ))
                    })?;
                connection
                    .execute(
                        "INSERT INTO session_entries (session_id, id, entry_seq, parent_id, type, timestamp, payload) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        rusqlite::params![
                            self.metadata.id,
                            entry.id(),
                            next_seq,
                            entry.parent_id(),
                            entry.entry_type(),
                            encoded.timestamp,
                            encoded.payload
                        ],
                    )
                    .map_err(storage_error)?;
                connection
                    .execute(
                        "UPDATE session_sequences SET next_seq = ?1 WHERE session_id = ?2",
                        rusqlite::params![next_seq + 1, self.metadata.id],
                    )
                    .map_err(storage_error)?;
                connection
                    .execute(
                        "UPDATE session_materialized SET payload = ?1 WHERE session_id = ?2",
                        rusqlite::params![serialize_summary(&self.materialized), self.metadata.id],
                    )
                    .map_err(storage_error)?;
                for (row_type, payload) in entry_materialized_values(&entry) {
                    connection
                        .execute(
                            "INSERT INTO entry_materialized (session_id, entry_seq, type, payload) VALUES (?1, ?2, ?3, ?4)",
                            rusqlite::params![self.metadata.id, next_seq, row_type, payload],
                        )
                        .map_err(storage_error)?;
                }
                connection
                    .execute(
                        "UPDATE sessions SET active_leaf_id = ?1 WHERE id = ?2",
                        rusqlite::params![new_leaf, self.metadata.id],
                    )
                    .map_err(storage_error)?;

                // Branch materialization (pi): leaf navigation always
                // re-materializes the target branch; a fork from a parent
                // that already has a child starts a new branch; linear
                // appends extend the active branch.
                let mut active_branch = self.active_branch_id.clone();
                if let SessionTreeEntry::Leaf { target_id, .. } = &entry {
                    active_branch = Some(materialize_branch(
                        &connection,
                        &self.metadata.id,
                        target_id.as_deref(),
                        &self.by_id,
                    )?);
                } else if parent_had_existing_child || active_branch.is_none() {
                    active_branch = Some(materialize_branch(
                        &connection,
                        &self.metadata.id,
                        entry.parent_id(),
                        &self.by_id,
                    )?);
                }
                let active_branch = active_branch.ok_or_else(|| {
                    invalid_session(format!(
                        "active branch missing for session {}",
                        self.metadata.id
                    ))
                })?;
                connection
                    .execute(
                        "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq) VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![self.metadata.id, active_branch, entry.id(), next_seq],
                    )
                    .map_err(storage_error)?;
                self.active_branch_id = Some(active_branch);
                connection.execute("COMMIT", []).map_err(storage_error)?;
                Ok(())
            })();
            if let Err(error) = &transaction {
                let _ = error;
                let _ = connection.execute("ROLLBACK", []);
            }
            transaction
        };

        match result {
            Ok(()) => {
                self.leaf_id = new_leaf;
                self.by_id
                    .lock()
                    .expect("sqlite cache lock")
                    .insert(entry.id().to_owned(), entry);
                Ok(())
            }
            Err(error) => {
                self.materialized = previous_materialized;
                self.leaf_id = previous_leaf;
                self.active_branch_id = previous_branch;
                Err(error)
            }
        }
    }

    /// Point lookup with a decode cache; malformed rows read as absent.
    pub fn get_entry(&self, id: &str) -> Option<SessionTreeEntry> {
        if let Some(cached) = self.by_id.lock().expect("sqlite cache lock").get(id) {
            return Some(cached.clone());
        }
        let row = {
            let connection = self.connection.lock().expect("sqlite connection lock");
            connection
                .query_row(
                    "SELECT id, parent_id, type, timestamp, payload FROM session_entries WHERE session_id = ?1 AND id = ?2",
                    rusqlite::params![self.metadata.id, id],
                    session_entry_row_from_sql,
                )
                .ok()?
        };
        let entry = decode_sqlite_entry(&row).ok()?;
        self.by_id
            .lock()
            .expect("sqlite cache lock")
            .insert(entry.id().to_owned(), entry.clone());
        Some(entry)
    }

    pub fn find_entries(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        let rows = {
            let connection = self.connection.lock().expect("sqlite connection lock");
            load_entry_rows(
                &connection,
                "SELECT id, parent_id, type, timestamp, payload FROM session_entries WHERE session_id = ?1 AND type = ?2 ORDER BY entry_seq",
                rusqlite::params![self.metadata.id, entry_type],
            )
        };
        self.decode_and_cache(rows)
    }

    pub fn label(&self, id: &str) -> Option<String> {
        self.materialized.labels_by_id.get(id).cloned()
    }

    pub fn session_name(&self) -> Option<String> {
        self.materialized.name.clone()
    }

    pub fn session_stats(&self) -> SessionStats {
        session_stats_from_materialized_state(&self.materialized)
    }

    pub fn materialized_state(&self) -> &SessionMaterializedState {
        &self.materialized
    }

    /// Plain path to root (ri storage contract), resolved by lazy parent
    /// walking.
    pub fn path_to_root(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let by_id = self.load_path_closure(leaf_id)?;
        get_path_to_root(&by_id, leaf_id)
    }

    /// Compaction-stopped path (pi `getPathToRootOrCompaction`): the active
    /// leaf reads the materialized branch table; other leaves walk parents.
    pub fn path_to_root_or_compaction(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let Some(leaf) = leaf_id else {
            return Ok(Vec::new());
        };
        if Some(leaf) == self.leaf_id.as_deref() {
            let branch_id = self.active_branch_id.clone().ok_or_else(|| {
                invalid_session(format!(
                    "missing active branch for session {} leaf {leaf}",
                    self.metadata.id
                ))
            })?;
            return self.materialized_branch_path(&branch_id);
        }
        get_path_to_root_or_compaction(|id| self.get_entry(id), leaf_id)
    }

    /// Cursor-paged listing (pi `getEntries`): up to `limit` entries at or
    /// before the anchor sequence, ascending.
    pub fn entries_page(
        &self,
        options: &SessionEntryCursorOptions,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let Some(limit) = options.limit else {
            return Ok(self.entries());
        };
        let rows = {
            let connection = self.connection.lock().expect("sqlite connection lock");
            let anchor: Option<i64> = match options.after_entry_seq {
                Some(seq) => Some(seq as i64),
                None => connection
                    .query_row(
                        "SELECT entry_seq FROM session_entries WHERE session_id = ?1 ORDER BY entry_seq DESC LIMIT 1",
                        rusqlite::params![self.metadata.id],
                        |row| row.get(0),
                    )
                    .ok(),
            };
            let Some(anchor) = anchor else {
                return Ok(Vec::new());
            };
            load_entry_rows(
                &connection,
                "SELECT id, parent_id, type, timestamp, payload FROM session_entries WHERE session_id = ?1 AND entry_seq <= ?2 ORDER BY entry_seq DESC LIMIT ?3",
                rusqlite::params![self.metadata.id, anchor, limit as i64],
            )
        };
        let mut entries = self.decode_and_cache(rows);
        entries.reverse();
        Ok(entries)
    }

    pub fn entries(&self) -> Vec<SessionTreeEntry> {
        let rows = {
            let connection = self.connection.lock().expect("sqlite connection lock");
            load_entry_rows(
                &connection,
                "SELECT id, parent_id, type, timestamp, payload FROM session_entries WHERE session_id = ?1 ORDER BY entry_seq",
                rusqlite::params![self.metadata.id],
            )
        };
        self.decode_and_cache(rows)
    }

    fn decode_and_cache(&self, rows: Vec<SessionEntryRow>) -> Vec<SessionTreeEntry> {
        let mut entries = Vec::with_capacity(rows.len());
        let mut cache = self.by_id.lock().expect("sqlite cache lock");
        for row in rows {
            // Keep JSONL-like permissive resume behavior: skip malformed rows.
            if let Ok(entry) = decode_sqlite_entry(&row) {
                cache.insert(entry.id().to_owned(), entry.clone());
                entries.push(entry);
            }
        }
        entries
    }

    /// Load every entry on the parent chain of `leaf_id` into a map so the
    /// shared in-memory walkers can run over lazy storage.
    fn load_path_closure(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<BTreeMap<String, SessionTreeEntry>, SessionError> {
        let mut by_id = BTreeMap::new();
        let Some(leaf_id) = leaf_id else {
            return Ok(by_id);
        };
        let mut current_id = leaf_id.to_owned();
        loop {
            let Some(entry) = self.get_entry(&current_id) else {
                // Leave resolution errors to the shared walker so messages
                // match the in-memory backends.
                return Ok(by_id);
            };
            let parent = entry.parent_id().map(str::to_owned);
            by_id.insert(entry.id().to_owned(), entry);
            match parent {
                Some(parent_id) => current_id = parent_id,
                None => break,
            }
        }
        Ok(by_id)
    }

    fn materialized_branch_path(
        &self,
        branch_id: &str,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        let rows = {
            let connection = self.connection.lock().expect("sqlite connection lock");
            let mut statement = connection
                .prepare(
                    "SELECT session_entries.id, session_entries.parent_id, session_entries.type, session_entries.timestamp, session_entries.payload \
                     FROM branch_entries JOIN session_entries \
                       ON session_entries.session_id = branch_entries.session_id \
                      AND session_entries.id = branch_entries.entry_id \
                     WHERE branch_entries.session_id = ?1 AND branch_entries.branch_id = ?2 \
                     ORDER BY branch_entries.entry_seq",
                )
                .map_err(storage_error)?;
            let mapped = statement
                .query_map(
                    rusqlite::params![self.metadata.id, branch_id],
                    session_entry_row_from_sql,
                )
                .map_err(storage_error)?;
            let mut rows = Vec::new();
            for row in mapped {
                rows.push(row.map_err(storage_error)?);
            }
            rows
        };
        let mut entries = Vec::with_capacity(rows.len());
        let mut cache = self.by_id.lock().expect("sqlite cache lock");
        for row in rows {
            let entry = decode_sqlite_entry(&row).map_err(|_| {
                invalid_session(format!("invalid entry row for branch entry {}", row.id))
            })?;
            cache.insert(entry.id().to_owned(), entry.clone());
            // Leaf entries are navigation markers, not part of the path.
            if !matches!(entry, SessionTreeEntry::Leaf { .. }) {
                entries.push(entry);
            }
        }
        Ok(entries)
    }
}

/// Materialize the branch ending at `leaf_id` under a fresh branch id and
/// return it. Runs inside the caller's transaction.
fn materialize_branch(
    connection: &Connection,
    session_id: &str,
    leaf_id: Option<&str>,
    cache: &Arc<Mutex<BTreeMap<String, SessionTreeEntry>>>,
) -> Result<String, SessionError> {
    let branch_id = ri_llm_provider::uuidv7();
    let lookup = |id: &str| -> Option<SessionTreeEntry> {
        if let Some(cached) = cache.lock().expect("sqlite cache lock").get(id) {
            return Some(cached.clone());
        }
        let row = connection
            .query_row(
                "SELECT id, parent_id, type, timestamp, payload FROM session_entries WHERE session_id = ?1 AND id = ?2",
                rusqlite::params![session_id, id],
                session_entry_row_from_sql,
            )
            .ok()?;
        let entry = decode_sqlite_entry(&row).ok()?;
        cache
            .lock()
            .expect("sqlite cache lock")
            .insert(entry.id().to_owned(), entry.clone());
        Some(entry)
    };
    let path = get_path_to_root_or_compaction(lookup, leaf_id)?;
    for entry in &path {
        let entry_seq: i64 = connection
            .query_row(
                "SELECT entry_seq FROM session_entries WHERE session_id = ?1 AND id = ?2",
                rusqlite::params![session_id, entry.id()],
                |row| row.get(0),
            )
            .map_err(|_| {
                invalid_session(format!(
                    "missing entry row for session {session_id} entry {}",
                    entry.id()
                ))
            })?;
        connection
            .execute(
                "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![session_id, branch_id, entry.id(), entry_seq],
            )
            .map_err(storage_error)?;
    }
    Ok(branch_id)
}

fn has_existing_child(
    connection: &Connection,
    session_id: &str,
    parent_id: Option<&str>,
) -> Result<bool, SessionError> {
    let found = match parent_id {
        Some(parent_id) => connection.query_row(
            "SELECT 1 FROM session_entries WHERE session_id = ?1 AND parent_id = ?2 LIMIT 1",
            rusqlite::params![session_id, parent_id],
            |_| Ok(()),
        ),
        None => connection.query_row(
            "SELECT 1 FROM session_entries WHERE session_id = ?1 AND parent_id IS NULL LIMIT 1",
            rusqlite::params![session_id],
            |_| Ok(()),
        ),
    };
    Ok(found.is_ok())
}

// ============================================================================
// Row encoding / loading
// ============================================================================

struct EncodedSqliteEntry {
    timestamp: String,
    payload: String,
}

/// Base fields become columns; the remaining variant fields are the payload.
fn encode_sqlite_entry(entry: &SessionTreeEntry) -> Result<EncodedSqliteEntry, SessionError> {
    let mut value = serde_json::to_value(entry).map_err(storage_error)?;
    let object = value.as_object_mut().ok_or_else(|| {
        SessionError::new(
            SessionErrorCode::InvalidEntry,
            "Invalid SQLite session entry: entry did not serialize to an object",
        )
    })?;
    object.remove("type");
    object.remove("id");
    object.remove("parentId");
    let timestamp = object
        .remove("timestamp")
        .and_then(|timestamp| timestamp.as_str().map(ToOwned::to_owned))
        .ok_or_else(|| {
            SessionError::new(
                SessionErrorCode::InvalidEntry,
                format!(
                    "Invalid SQLite session entry: entry {} is missing timestamp",
                    entry.id()
                ),
            )
        })?;
    let payload = Value::Object(std::mem::take(object)).to_string();
    Ok(EncodedSqliteEntry { timestamp, payload })
}

fn decode_sqlite_entry(row: &SessionEntryRow) -> Result<SessionTreeEntry, SessionError> {
    let payload: Value = serde_json::from_str(&row.payload).map_err(|error| {
        SessionError::new(
            SessionErrorCode::InvalidEntry,
            format!(
                "Invalid SQLite session entry: entry {} payload is not valid JSON: {error}",
                row.id
            ),
        )
    })?;
    let Value::Object(mut object) = payload else {
        return Err(SessionError::new(
            SessionErrorCode::InvalidEntry,
            format!(
                "Invalid SQLite session entry: entry {} payload is not an object",
                row.id
            ),
        ));
    };
    object.insert("type".to_owned(), Value::String(row.entry_type.clone()));
    object.insert("id".to_owned(), Value::String(row.id.clone()));
    object.insert(
        "parentId".to_owned(),
        row.parent_id
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    object.insert("timestamp".to_owned(), Value::String(row.timestamp.clone()));
    serde_json::from_value(Value::Object(object)).map_err(|error| {
        SessionError::new(
            SessionErrorCode::InvalidEntry,
            format!("Invalid SQLite session entry: entry {}: {error}", row.id),
        )
    })
}

#[derive(Debug, Clone)]
struct SessionRow {
    id: String,
    created_at: String,
    metadata: Option<String>,
    cwd: String,
    parent_session_id: Option<String>,
    active_leaf_id: Option<String>,
}

struct SessionEntryRow {
    id: String,
    parent_id: Option<String>,
    entry_type: String,
    timestamp: String,
    payload: String,
}

fn session_row_from_sql(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRow> {
    Ok(SessionRow {
        id: row.get(0)?,
        created_at: row.get(1)?,
        metadata: row.get(2)?,
        cwd: row.get(3)?,
        parent_session_id: row.get(4)?,
        active_leaf_id: row.get(5)?,
    })
}

fn session_entry_row_from_sql(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionEntryRow> {
    Ok(SessionEntryRow {
        id: row.get(0)?,
        parent_id: row.get(1)?,
        entry_type: row.get(2)?,
        timestamp: row.get(3)?,
        payload: row.get(4)?,
    })
}

fn load_entry_rows(
    connection: &Connection,
    sql: &str,
    params: impl rusqlite::Params,
) -> Vec<SessionEntryRow> {
    let Ok(mut statement) = connection.prepare(sql) else {
        return Vec::new();
    };
    let Ok(mapped) = statement.query_map(params, session_entry_row_from_sql) else {
        return Vec::new();
    };
    mapped.filter_map(Result::ok).collect()
}

fn session_row_to_metadata(
    row: SessionRow,
    database_path: &Path,
) -> Result<SqliteSessionMetadata, SessionError> {
    let metadata = row
        .metadata
        .as_deref()
        .map(|metadata| {
            serde_json::from_str::<Value>(metadata)
                .ok()
                .and_then(|value| match value {
                    Value::Object(object) => Some(object),
                    _ => None,
                })
                .ok_or_else(|| {
                    SessionError::new(
                        SessionErrorCode::InvalidSession,
                        format!(
                            "Invalid SQLite session {}: metadata must be a JSON object",
                            row.id
                        ),
                    )
                })
        })
        .transpose()?;
    Ok(SqliteSessionMetadata {
        id: row.id,
        created_at: row.created_at,
        cwd: row.cwd,
        path: database_path.to_string_lossy().into_owned(),
        parent_session_id: row.parent_session_id,
        metadata,
    })
}

fn load_session_row(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<SessionRow>, SessionError> {
    let mut statement = connection
        .prepare(
            "SELECT id, created_at, metadata, cwd, parent_session_id, active_leaf_id FROM sessions WHERE id = ?1",
        )
        .map_err(storage_error)?;
    let mut rows = statement
        .query_map(rusqlite::params![session_id], session_row_from_sql)
        .map_err(storage_error)?;
    rows.next().transpose().map_err(storage_error)
}

/// Load the materialized state: the summary row (required — its absence
/// marks a corrupt/pre-materialization session, as in pi) plus label rows.
fn load_materialized_state(
    connection: &Connection,
    session_id: &str,
) -> Result<SessionMaterializedState, SessionError> {
    let payload: String = connection
        .query_row(
            "SELECT payload FROM session_materialized WHERE session_id = ?1",
            rusqlite::params![session_id],
            |row| row.get(0),
        )
        .map_err(|_| {
            invalid_session(format!("missing materialized row for session {session_id}"))
        })?;
    let mut state = parse_summary(&payload)?;
    let mut statement = connection
        .prepare(
            "SELECT entry_seq, type, payload FROM entry_materialized WHERE session_id = ?1 ORDER BY entry_seq, type",
        )
        .map_err(storage_error)?;
    let rows = statement
        .query_map(rusqlite::params![session_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(storage_error)?;
    for row in rows {
        let (entry_seq, row_type, payload) = row.map_err(storage_error)?;
        apply_entry_materialized_row(&mut state, entry_seq, &row_type, &payload)?;
    }
    Ok(state)
}

/// The newest branch row identifies the branch most recently made active.
fn load_active_branch_id(
    connection: &Connection,
    session_id: &str,
) -> Result<Option<String>, SessionError> {
    Ok(connection
        .query_row(
            "SELECT branch_id FROM branch_entries WHERE session_id = ?1 ORDER BY entry_seq DESC, branch_id DESC LIMIT 1",
            rusqlite::params![session_id],
            |row| row.get(0),
        )
        .ok())
}

fn configure_sqlite_database(connection: &Connection) -> Result<(), SessionError> {
    connection
        .pragma_update(None, "journal_mode", "WAL")
        .map_err(storage_error)?;
    connection
        .pragma_update(None, "synchronous", "FULL")
        .map_err(storage_error)?;
    connection
        .pragma_update(None, "busy_timeout", 5000)
        .map_err(storage_error)?;
    Ok(())
}

pub fn apply_sqlite_migrations(connection: &Connection) -> Result<(), SessionError> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS migrations (\n\tid TEXT PRIMARY KEY,\n\tapplied_at TEXT NOT NULL\n);",
        )
        .map_err(storage_error)?;
    let mut statement = connection
        .prepare("SELECT id FROM migrations ORDER BY applied_at, id")
        .map_err(storage_error)?;
    let applied = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(storage_error)?
        .collect::<Result<std::collections::BTreeSet<_>, _>>()
        .map_err(storage_error)?;
    drop(statement);
    for (id, sql) in SQLITE_MIGRATIONS {
        if applied.contains(*id) {
            continue;
        }
        let result: Result<(), SessionError> = (|| {
            connection.execute("BEGIN", []).map_err(storage_error)?;
            connection.execute_batch(sql).map_err(storage_error)?;
            connection
                .execute(
                    "INSERT INTO migrations (id, applied_at) VALUES (?1, ?2)",
                    rusqlite::params![id, now_iso()],
                )
                .map_err(storage_error)?;
            connection.execute("COMMIT", []).map_err(storage_error)?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = connection.execute("ROLLBACK", []);
            return Err(error);
        }
    }
    Ok(())
}

pub fn sqlite_applied_migration_ids(connection: &Connection) -> Result<Vec<String>, SessionError> {
    let mut statement = connection
        .prepare("SELECT id FROM migrations ORDER BY applied_at, id")
        .map_err(storage_error)?;
    let ids = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(storage_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(storage_error)?;
    Ok(ids)
}
