//! SQLite session storage (pi `@earendil-works/pi-storage-sqlite-node`, #6594).
//!
//! Sessions live in one database file using pi's schema: a `sessions` row per
//! session, ordered `session_entries` rows keyed by a monotonically increasing
//! `entry_seq` from `session_sequences`, and the `sessions.active_leaf_id`
//! column as the persisted leaf pointer. Entries store their base fields
//! (id/parent/type/timestamp) as columns and the remaining variant fields as a
//! JSON payload, matching pi's encode/decode. Like the JSONL backend, the
//! storage keeps an in-memory mirror for reads and writes through to the
//! database on every append.
//!
//! pi's materialized-state cache tables (`session_materialized`,
//! `entry_materialized`) and `branch_entries` acceleration exist in the schema
//! but are not yet populated; reads derive everything from `session_entries`.

use super::session::{
    SessionError, SessionErrorCode, SessionTreeEntry, build_labels_by_id, generate_entry_id,
    get_path_to_root, leaf_id_after_entry, now_iso, storage_error, update_label_cache,
    validate_existing_leaf, validate_leaf,
};
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
                "INSERT INTO session_sequences (session_id, next_seq) VALUES (?1, 0)",
                rusqlite::params![id],
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
        Ok(SqliteSessionStorage::from_loaded(
            connection,
            metadata,
            Vec::new(),
            None,
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
        let entries = load_session_entries(&connection, session_id)?;
        Ok(SqliteSessionStorage::from_loaded(
            connection,
            metadata,
            entries,
            row.active_leaf_id,
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

#[derive(Debug, Clone)]
pub struct SqliteSessionStorage {
    connection: Arc<Mutex<Connection>>,
    metadata: SqliteSessionMetadata,
    entries: Vec<SessionTreeEntry>,
    by_id: BTreeMap<String, SessionTreeEntry>,
    labels_by_id: BTreeMap<String, String>,
    leaf_id: Option<String>,
}

impl SqliteSessionStorage {
    fn from_loaded(
        connection: Connection,
        metadata: SqliteSessionMetadata,
        entries: Vec<SessionTreeEntry>,
        leaf_id: Option<String>,
    ) -> Self {
        let by_id = entries
            .iter()
            .map(|entry| (entry.id().to_owned(), entry.clone()))
            .collect();
        let labels_by_id = build_labels_by_id(&entries);
        Self {
            connection: Arc::new(Mutex::new(connection)),
            metadata,
            entries,
            by_id,
            labels_by_id,
            leaf_id,
        }
    }

    pub fn metadata(&self) -> &SqliteSessionMetadata {
        &self.metadata
    }

    pub fn leaf_id(&self) -> Result<Option<String>, SessionError> {
        validate_existing_leaf(&self.by_id, self.leaf_id.as_deref())?;
        Ok(self.leaf_id.clone())
    }

    pub fn set_leaf_id(&mut self, leaf_id: Option<String>) -> Result<(), SessionError> {
        validate_leaf(&self.by_id, leaf_id.as_deref())?;
        let entry = SessionTreeEntry::Leaf {
            id: generate_entry_id(&self.by_id),
            parent_id: self.leaf_id.clone(),
            timestamp: now_iso(),
            target_id: leaf_id.clone(),
        };
        self.append_entry(entry)?;
        self.leaf_id = leaf_id;
        Ok(())
    }

    pub fn create_entry_id(&self) -> String {
        generate_entry_id(&self.by_id)
    }

    pub fn append_entry(&mut self, entry: SessionTreeEntry) -> Result<(), SessionError> {
        let encoded = encode_sqlite_entry(&entry)?;
        {
            let connection = self.connection.lock().expect("sqlite connection lock");
            let result: Result<(), SessionError> = (|| {
                connection.execute("BEGIN", []).map_err(storage_error)?;
                let next_seq: i64 = connection
                    .query_row(
                        "SELECT next_seq FROM session_sequences WHERE session_id = ?1",
                        rusqlite::params![self.metadata.id],
                        |row| row.get(0),
                    )
                    .map_err(|_| {
                        SessionError::new(
                            SessionErrorCode::InvalidSession,
                            format!(
                                "Invalid SQLite session: missing sequence row for session {}",
                                self.metadata.id
                            ),
                        )
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
                        "UPDATE sessions SET active_leaf_id = ?1 WHERE id = ?2",
                        rusqlite::params![leaf_id_after_entry(&entry), self.metadata.id],
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
        update_label_cache(&mut self.labels_by_id, &entry);
        self.leaf_id = leaf_id_after_entry(&entry);
        self.by_id.insert(entry.id().to_owned(), entry.clone());
        self.entries.push(entry);
        Ok(())
    }

    pub fn get_entry(&self, id: &str) -> Option<&SessionTreeEntry> {
        self.by_id.get(id)
    }

    pub fn find_entries(&self, entry_type: &str) -> Vec<SessionTreeEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.entry_type() == entry_type)
            .cloned()
            .collect()
    }

    pub fn label(&self, id: &str) -> Option<&str> {
        self.labels_by_id.get(id).map(String::as_str)
    }

    pub fn path_to_root(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<SessionTreeEntry>, SessionError> {
        get_path_to_root(&self.by_id, leaf_id)
    }

    pub fn entries(&self) -> Vec<SessionTreeEntry> {
        self.entries.clone()
    }
}

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

fn load_session_entries(
    connection: &Connection,
    session_id: &str,
) -> Result<Vec<SessionTreeEntry>, SessionError> {
    let mut statement = connection
        .prepare(
            "SELECT id, parent_id, type, timestamp, payload FROM session_entries WHERE session_id = ?1 ORDER BY entry_seq",
        )
        .map_err(storage_error)?;
    let rows = statement
        .query_map(rusqlite::params![session_id], |row| {
            Ok(SessionEntryRow {
                id: row.get(0)?,
                parent_id: row.get(1)?,
                entry_type: row.get(2)?,
                timestamp: row.get(3)?,
                payload: row.get(4)?,
            })
        })
        .map_err(storage_error)?;
    let mut entries = Vec::new();
    for row in rows {
        let row = row.map_err(storage_error)?;
        // Keep JSONL-like permissive resume behavior: skip malformed entries.
        if let Ok(entry) = decode_sqlite_entry(&row) {
            entries.push(entry);
        }
    }
    Ok(entries)
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
