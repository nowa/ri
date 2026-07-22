use ri_agent_core::harness::{
    Session, SessionEntryMessage, SessionTreeEntry, SqliteSessionCreateOptions,
    SqliteSessionForkOptions, SqliteSessionListOptions, SqliteSessionRepo, apply_sqlite_migrations,
    sqlite_applied_migration_ids,
};
use ri_llm_provider::{Message, UserContentValue, UserMessage};
use serde_json::json;

fn temp_database_path(name: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "ri-sqlite-session-{name}-{}",
        ri_llm_provider::uuidv7()
    ));
    path.push("sessions.db");
    path
}

fn cleanup(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::remove_dir_all(parent);
    }
}

fn message_entry(id: &str, parent_id: Option<&str>, text: &str) -> SessionTreeEntry {
    SessionTreeEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(ToOwned::to_owned),
        timestamp: "2026-07-22T00:00:00.000Z".to_owned(),
        message: SessionEntryMessage::Llm(Message::User(UserMessage {
            content: UserContentValue::Plain(text.to_owned()),
            timestamp: 1,
        })),
    }
}

#[test]
fn sqlite_migrations_apply_once_and_are_recorded() {
    let path = temp_database_path("migrations");
    let repo = SqliteSessionRepo::new(&path);
    let storage = repo
        .create(SqliteSessionCreateOptions {
            cwd: "/tmp/project".to_owned(),
            ..Default::default()
        })
        .expect("create session");
    drop(storage);

    // Re-opening applies migrations idempotently.
    let connection = rusqlite::Connection::open(&path).expect("open raw connection");
    apply_sqlite_migrations(&connection).expect("reapply migrations");
    let ids = sqlite_applied_migration_ids(&connection).expect("migration ids");
    assert_eq!(ids, vec!["001_initial.sql".to_owned()]);

    let tables: Vec<String> = {
        let mut statement = connection
            .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
            .expect("prepare");
        statement
            .query_map([], |row| row.get(0))
            .expect("query")
            .collect::<Result<_, _>>()
            .expect("collect")
    };
    for table in [
        "branch_entries",
        "entry_materialized",
        "migrations",
        "session_entries",
        "session_materialized",
        "session_sequences",
        "sessions",
    ] {
        assert!(tables.contains(&table.to_owned()), "missing table {table}");
    }
    cleanup(&path);
}

#[test]
fn sqlite_sessions_persist_metadata_through_create_list_open_delete() {
    let path = temp_database_path("metadata");
    let repo = SqliteSessionRepo::new(&path);
    let first = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-one".to_owned()),
            cwd: "/tmp/project-a".to_owned(),
            metadata: Some(
                json!({ "name": "first" })
                    .as_object()
                    .cloned()
                    .expect("object"),
            ),
            ..Default::default()
        })
        .expect("create first");
    let second = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-two".to_owned()),
            cwd: "/tmp/project-b".to_owned(),
            parent_session_id: Some("session-one".to_owned()),
            ..Default::default()
        })
        .expect("create second");
    assert_eq!(first.metadata().id, "session-one");
    assert_eq!(
        second.metadata().parent_session_id.as_deref(),
        Some("session-one")
    );
    drop(first);
    drop(second);

    let all = repo
        .list(SqliteSessionListOptions::default())
        .expect("list");
    assert_eq!(all.len(), 2);
    let filtered = repo
        .list(SqliteSessionListOptions {
            cwd: Some("/tmp/project-a".to_owned()),
        })
        .expect("list cwd");
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].id, "session-one");
    assert_eq!(
        filtered[0]
            .metadata
            .as_ref()
            .and_then(|meta| meta.get("name")),
        Some(&json!("first"))
    );
    assert_eq!(filtered[0].path, path.to_string_lossy());

    let opened = repo.open("session-one").expect("open");
    assert_eq!(opened.metadata().cwd, "/tmp/project-a");
    drop(opened);

    repo.delete("session-two").expect("delete");
    let err = repo.delete("session-two").expect_err("double delete");
    assert!(err.to_string().contains("Session not found"));
    let remaining = repo
        .list(SqliteSessionListOptions::default())
        .expect("list");
    assert_eq!(remaining.len(), 1);
    cleanup(&path);
}

#[test]
fn sqlite_entries_round_trip_and_leaf_id_is_materialized() {
    let path = temp_database_path("entries");
    let repo = SqliteSessionRepo::new(&path);
    let mut storage = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-entries".to_owned()),
            cwd: "/tmp/project".to_owned(),
            ..Default::default()
        })
        .expect("create");

    storage
        .append_entry(message_entry("aaaa", None, "hello"))
        .expect("append root");
    storage
        .append_entry(message_entry("bbbb", Some("aaaa"), "world"))
        .expect("append child");
    storage
        .append_entry(SessionTreeEntry::Label {
            id: "cccc".to_owned(),
            parent_id: Some("bbbb".to_owned()),
            timestamp: "2026-07-22T00:00:01.000Z".to_owned(),
            target_id: "aaaa".to_owned(),
            label: Some("root label".to_owned()),
        })
        .expect("append label");
    assert_eq!(storage.leaf_id().expect("leaf"), Some("cccc".to_owned()));
    drop(storage);

    let reopened = repo.open("session-entries").expect("reopen");
    assert_eq!(reopened.leaf_id().expect("leaf"), Some("cccc".to_owned()));
    let entries = reopened.entries();
    assert_eq!(
        entries.iter().map(SessionTreeEntry::id).collect::<Vec<_>>(),
        vec!["aaaa", "bbbb", "cccc"]
    );
    assert_eq!(reopened.label("aaaa"), Some("root label"));
    let session_path = reopened
        .path_to_root(Some("bbbb"))
        .expect("path to root")
        .iter()
        .map(SessionTreeEntry::id)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(session_path, vec!["aaaa", "bbbb"]);
    cleanup(&path);
}

#[test]
fn sqlite_set_leaf_id_appends_navigation_entry_and_persists() {
    let path = temp_database_path("leaf");
    let repo = SqliteSessionRepo::new(&path);
    let mut storage = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-leaf".to_owned()),
            cwd: "/tmp/project".to_owned(),
            ..Default::default()
        })
        .expect("create");
    storage
        .append_entry(message_entry("aaaa", None, "hello"))
        .expect("append root");
    storage
        .append_entry(message_entry("bbbb", Some("aaaa"), "world"))
        .expect("append child");
    storage
        .set_leaf_id(Some("aaaa".to_owned()))
        .expect("set leaf");
    assert_eq!(storage.leaf_id().expect("leaf"), Some("aaaa".to_owned()));

    let err = storage
        .set_leaf_id(Some("missing".to_owned()))
        .expect_err("unknown leaf");
    assert!(err.to_string().contains("not found"));
    drop(storage);

    let reopened = repo.open("session-leaf").expect("reopen");
    assert_eq!(reopened.leaf_id().expect("leaf"), Some("aaaa".to_owned()));
    let leaf_entries = reopened.find_entries("leaf");
    assert_eq!(leaf_entries.len(), 1);
    cleanup(&path);
}

#[test]
fn sqlite_open_skips_malformed_entries_like_jsonl_resume() {
    let path = temp_database_path("malformed");
    let repo = SqliteSessionRepo::new(&path);
    let mut storage = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-malformed".to_owned()),
            cwd: "/tmp/project".to_owned(),
            ..Default::default()
        })
        .expect("create");
    storage
        .append_entry(message_entry("aaaa", None, "hello"))
        .expect("append root");
    drop(storage);

    let connection = rusqlite::Connection::open(&path).expect("raw connection");
    connection
        .execute(
            "INSERT INTO session_entries (session_id, id, entry_seq, parent_id, type, timestamp, payload) VALUES ('session-malformed', 'zzzz', 99, 'aaaa', 'message', '2026-07-22T00:00:02.000Z', 'not json')",
            [],
        )
        .expect("insert malformed");
    drop(connection);

    let reopened = repo.open("session-malformed").expect("reopen");
    assert_eq!(
        reopened
            .entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["aaaa"]
    );
    cleanup(&path);
}

#[test]
fn sqlite_storage_backs_session_context_builds() {
    let path = temp_database_path("session");
    let repo = SqliteSessionRepo::new(&path);
    let storage = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-context".to_owned()),
            cwd: "/tmp/project".to_owned(),
            ..Default::default()
        })
        .expect("create");
    let mut session = Session::new(storage);
    session
        .append_message(Message::User(UserMessage {
            content: UserContentValue::Plain("hello from sqlite".to_owned()),
            timestamp: 1,
        }))
        .expect("append message");
    let path_entries = session.branch(None).expect("branch entries");
    let context =
        ri_agent_core::harness::build_session_context(&path_entries).expect("session context");
    assert_eq!(context.messages.len(), 1);
    assert_eq!(
        session.sqlite_metadata().expect("metadata").id,
        "session-context"
    );

    // The same session is visible through a fresh repo handle.
    let reopened = repo.open("session-context").expect("reopen");
    assert_eq!(reopened.entries().len(), 1);
    cleanup(&path);
}

#[test]
fn sqlite_fork_copies_branch_and_inherits_metadata() {
    let path = temp_database_path("fork");
    let repo = SqliteSessionRepo::new(&path);
    let mut source = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-source".to_owned()),
            cwd: "/tmp/project".to_owned(),
            metadata: Some(
                serde_json::json!({ "name": "origin" })
                    .as_object()
                    .cloned()
                    .expect("object"),
            ),
            ..Default::default()
        })
        .expect("create source");
    source
        .append_entry(message_entry("aaaa", None, "hello"))
        .expect("append root");
    source
        .append_entry(message_entry("bbbb", Some("aaaa"), "world"))
        .expect("append child");
    source
        .append_entry(message_entry("cccc", Some("bbbb"), "tail"))
        .expect("append tail");
    drop(source);

    // Fork at an entry keeps the path up to and including it.
    let fork = repo
        .fork(
            "session-source",
            SqliteSessionForkOptions {
                cwd: "/tmp/project".to_owned(),
                parent_session_id: None,
                metadata: None,
                fork: ri_agent_core::harness::SessionForkOptions {
                    entry_id: Some("bbbb".to_owned()),
                    position: ri_agent_core::harness::ForkPosition::At,
                    id: Some("session-fork".to_owned()),
                },
            },
        )
        .expect("fork");
    assert_eq!(fork.metadata().id, "session-fork");
    assert_eq!(
        fork.metadata().parent_session_id.as_deref(),
        Some("session-source")
    );
    assert_eq!(
        fork.metadata()
            .metadata
            .as_ref()
            .and_then(|meta| meta.get("name")),
        Some(&serde_json::json!("origin"))
    );
    assert_eq!(
        fork.entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["aaaa", "bbbb"]
    );
    drop(fork);

    // The fork persists independently of the source.
    let reopened = repo.open("session-fork").expect("reopen fork");
    assert_eq!(reopened.entries().len(), 2);
    assert_eq!(reopened.leaf_id().expect("leaf"), Some("bbbb".to_owned()));

    // Unknown fork targets are rejected.
    let err = repo
        .fork(
            "session-source",
            SqliteSessionForkOptions {
                cwd: "/tmp/project".to_owned(),
                parent_session_id: None,
                metadata: None,
                fork: ri_agent_core::harness::SessionForkOptions {
                    entry_id: Some("missing".to_owned()),
                    position: ri_agent_core::harness::ForkPosition::At,
                    id: None,
                },
            },
        )
        .expect_err("invalid fork target");
    assert!(err.to_string().contains("Entry missing not found"));
    cleanup(&path);
}
