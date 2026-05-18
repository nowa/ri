use ri_agent_core::*;
use ri_llm_provider::Message;
use serde_json::json;
use std::{fs, path::PathBuf};

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ri-session-test-{}", uuidv7()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

fn entry(id: &str, parent_id: Option<&str>, message: Message) -> SessionTreeEntry {
    SessionTreeEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        message,
    }
}

#[test]
fn in_memory_storage_matches_core_storage_behaviour() {
    let metadata = SessionMetadata {
        id: "session-1".to_owned(),
        created_at: "2026-01-01T00:00:00.000Z".to_owned(),
    };
    let root = entry("entry-1", None, user_message_text("one"));
    let mut initial_entries = vec![root.clone()];
    let mut storage =
        InMemorySessionStorage::with_options(Some(initial_entries.clone()), Some(metadata.clone()));
    initial_entries.push(entry("entry-2", None, user_message_text("two")));

    assert_eq!(storage.metadata(), &metadata);
    assert_eq!(
        storage
            .entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["entry-1"]
    );
    assert_eq!(storage.leaf_id().expect("leaf"), Some("entry-1".to_owned()));

    storage.set_leaf_id(None).expect("set leaf");
    assert_eq!(storage.leaf_id().expect("leaf"), None);
    assert!(matches!(
        storage.entries().last(),
        Some(SessionTreeEntry::Leaf {
            target_id: None,
            ..
        })
    ));

    let err = storage
        .set_leaf_id(Some("missing".to_owned()))
        .expect_err("invalid leaf");
    assert_eq!(err.code, SessionErrorCode::NotFound);

    storage.append_entry(SessionTreeEntry::Label {
        id: "label-1".to_owned(),
        parent_id: Some("entry-1".to_owned()),
        timestamp: "2026-01-01T00:00:01.000Z".to_owned(),
        target_id: "entry-1".to_owned(),
        label: Some(" checkpoint ".to_owned()),
    });
    assert_eq!(storage.label("entry-1"), Some("checkpoint"));
    storage.append_entry(SessionTreeEntry::Label {
        id: "label-2".to_owned(),
        parent_id: Some("label-1".to_owned()),
        timestamp: "2026-01-01T00:00:02.000Z".to_owned(),
        target_id: "entry-1".to_owned(),
        label: None,
    });
    assert_eq!(storage.label("entry-1"), None);
}

#[test]
fn in_memory_storage_walks_paths_to_root() {
    let root = entry("root", None, user_message_text("root"));
    let child = entry("child", Some("root"), assistant_message_text("child"));
    let storage = InMemorySessionStorage::with_options(Some(vec![root, child]), None);

    assert_eq!(
        storage
            .path_to_root(Some("child"))
            .expect("path")
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["root", "child"]
    );
    assert!(storage.path_to_root(None).expect("root").is_empty());
}

#[test]
fn in_memory_storage_finds_entries_by_type() {
    let root = entry("entry-1", None, user_message_text("one"));
    let storage = InMemorySessionStorage::with_options(Some(vec![root]), None);

    assert_eq!(
        storage
            .find_entries("message")
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["entry-1"]
    );
    assert!(storage.find_entries("session_info").is_empty());
}

#[test]
fn jsonl_storage_writes_loads_metadata_entries_leaf_and_labels() {
    let dir = temp_dir();
    let path = dir.join("session.jsonl");
    let mut storage = JsonlSessionStorage::create(
        &path,
        dir.to_string_lossy(),
        "session-1",
        Some("/tmp/parent.jsonl".to_owned()),
    )
    .expect("create");

    assert!(path.exists());
    assert_eq!(
        fs::read_to_string(&path)
            .expect("read")
            .trim()
            .lines()
            .count(),
        1
    );
    assert_eq!(storage.leaf_id().expect("leaf"), None);

    storage
        .append_entry(entry("root", None, user_message_text("root")))
        .expect("append root");
    storage
        .append_entry(entry(
            "child",
            Some("root"),
            assistant_message_text("child"),
        ))
        .expect("append child");
    storage
        .set_leaf_id(Some("root".to_owned()))
        .expect("set leaf");
    storage
        .append_entry(SessionTreeEntry::Label {
            id: "label-1".to_owned(),
            parent_id: Some("root".to_owned()),
            timestamp: "2026-01-01T00:00:01.000Z".to_owned(),
            target_id: "root".to_owned(),
            label: Some("checkpoint".to_owned()),
        })
        .expect("label");

    let metadata = load_jsonl_session_metadata(&path).expect("metadata");
    assert_eq!(metadata.id, "session-1");
    assert_eq!(metadata.cwd, dir.to_string_lossy());
    assert_eq!(
        metadata.parent_session_path.as_deref(),
        Some("/tmp/parent.jsonl")
    );

    let loaded = JsonlSessionStorage::open(&path).expect("open");
    assert_eq!(loaded.leaf_id().expect("leaf"), Some("label-1".to_owned()));
    assert_eq!(loaded.label("root"), Some("checkpoint"));
    assert_eq!(
        loaded
            .path_to_root(Some("child"))
            .expect("path")
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["root", "child"]
    );
}

#[test]
fn jsonl_storage_rejects_missing_files_and_finds_entries_by_type() {
    let dir = temp_dir();
    let missing_path = dir.join("missing.jsonl");
    let err = JsonlSessionStorage::open(&missing_path).expect_err("missing");
    assert_eq!(err.code, SessionErrorCode::NotFound);

    let path = dir.join("session.jsonl");
    let mut storage = JsonlSessionStorage::create(&path, dir.to_string_lossy(), "session-1", None)
        .expect("create");
    storage
        .append_entry(entry("entry-1", None, user_message_text("one")))
        .expect("append");

    assert_eq!(
        storage
            .find_entries("message")
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec!["entry-1"]
    );
    assert!(storage.find_entries("session_info").is_empty());
}

#[test]
fn jsonl_storage_label_lookup_can_be_cleared_and_reloaded() {
    let dir = temp_dir();
    let path = dir.join("session.jsonl");
    let mut storage = JsonlSessionStorage::create(&path, dir.to_string_lossy(), "session-1", None)
        .expect("create");
    storage
        .append_entry(entry("entry-1", None, user_message_text("one")))
        .expect("append entry");
    assert_eq!(storage.label("entry-1"), None);
    storage
        .append_entry(SessionTreeEntry::Label {
            id: "label-1".to_owned(),
            parent_id: Some("entry-1".to_owned()),
            timestamp: "2026-01-01T00:00:01.000Z".to_owned(),
            target_id: "entry-1".to_owned(),
            label: Some("checkpoint".to_owned()),
        })
        .expect("label");
    assert_eq!(storage.label("entry-1"), Some("checkpoint"));
    storage
        .append_entry(SessionTreeEntry::Label {
            id: "label-2".to_owned(),
            parent_id: Some("label-1".to_owned()),
            timestamp: "2026-01-01T00:00:02.000Z".to_owned(),
            target_id: "entry-1".to_owned(),
            label: None,
        })
        .expect("clear label");
    assert_eq!(storage.label("entry-1"), None);

    let loaded = JsonlSessionStorage::open(&path).expect("reload");
    assert_eq!(loaded.label("entry-1"), None);
}

#[test]
fn jsonl_storage_rejects_malformed_headers_and_entries() {
    let dir = temp_dir();
    let path = dir.join("bad.jsonl");
    fs::write(&path, "not json\n").expect("write");
    let err = JsonlSessionStorage::open(&path).expect_err("bad header");
    assert_eq!(err.code, SessionErrorCode::InvalidSession);
    assert!(
        err.message
            .contains("first line is not a valid session header")
    );

    let header = json!({
        "type": "session",
        "version": 3,
        "id": "session-1",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": dir.to_string_lossy()
    });
    fs::write(&path, format!("{header}\nnot json\n")).expect("write");
    let err = JsonlSessionStorage::open(&path).expect_err("bad entry");
    assert_eq!(err.code, SessionErrorCode::InvalidEntry);
}

fn run_session_suite(storage: impl Into<SessionStorageKind>) {
    let mut session = Session::new(storage);
    let user1 = session
        .append_message(user_message_text("one"))
        .expect("user1");
    let assistant1 = session
        .append_message(assistant_message_text("two"))
        .expect("assistant1");
    session
        .append_message(user_message_text("three"))
        .expect("user2");
    session.move_to(Some(user1.clone()), None).expect("move");
    session
        .append_message(assistant_message_text("branched"))
        .expect("branch assistant");

    let branch = session.branch(None).expect("branch");
    assert!(branch.iter().any(|entry| entry.id() == user1));
    assert!(!branch.iter().any(|entry| entry.id() == assistant1));
    assert_eq!(
        session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
}

#[test]
fn session_supports_branching_for_memory_and_jsonl_storage() {
    run_session_suite(InMemorySessionStorage::new());

    let dir = temp_dir();
    let path = dir.join("session.jsonl");
    run_session_suite(
        JsonlSessionStorage::create(&path, dir.to_string_lossy(), "session-1", None)
            .expect("jsonl"),
    );
}

fn run_session_context_suite(storage: impl Into<SessionStorageKind>) {
    let mut session = Session::new(storage);
    session
        .append_message(user_message_text("one"))
        .expect("user");
    session
        .append_message(assistant_message_text("two"))
        .expect("assistant");

    assert_eq!(
        session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );

    session
        .append_model_change("openai", "gpt-4.1")
        .expect("model");
    session
        .append_thinking_level_change("high")
        .expect("thinking");
    let context = session.build_context().expect("context");
    assert_eq!(context.thinking_level, "high");
    assert_eq!(
        context.model,
        Some(SessionModelSelection {
            provider: "openai".to_owned(),
            model_id: "gpt-4.1".to_owned(),
        })
    );

    session.move_to(None, None).expect("move root");
    assert_eq!(session.leaf_id().expect("leaf"), None);
    assert!(
        session
            .build_context()
            .expect("context")
            .messages
            .is_empty()
    );
}

#[test]
fn session_builds_context_tracks_model_thinking_and_moves_to_root_for_storage_kinds() {
    run_session_context_suite(InMemorySessionStorage::new());

    let dir = temp_dir();
    let path = dir.join("session.jsonl");
    run_session_context_suite(
        JsonlSessionStorage::create(&path, dir.to_string_lossy(), "session-1", None)
            .expect("jsonl"),
    );
}

fn run_session_branch_summary_suite(storage: impl Into<SessionStorageKind>) {
    let mut session = Session::new(storage);
    let user1 = session
        .append_message(user_message_text("one"))
        .expect("user");
    let summary_id = session
        .move_to(
            Some(user1.clone()),
            Some(BranchMoveSummary {
                summary: "summary text".to_owned(),
                details: None,
                from_hook: None,
            }),
        )
        .expect("move summary")
        .expect("summary id");

    assert!(matches!(
        session.get_entry(&summary_id),
        Some(SessionTreeEntry::BranchSummary {
            parent_id,
            from_id,
            summary,
            ..
        }) if parent_id.as_deref() == Some(user1.as_str())
            && from_id == user1.as_str()
            && summary == "summary text"
    ));
    assert_eq!(
        session
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "branchSummary"]
    );
}

#[test]
fn session_move_with_branch_summary_appears_in_context_for_storage_kinds() {
    run_session_branch_summary_suite(InMemorySessionStorage::new());

    let dir = temp_dir();
    let path = dir.join("session.jsonl");
    run_session_branch_summary_suite(
        JsonlSessionStorage::create(&path, dir.to_string_lossy(), "session-1", None)
            .expect("jsonl"),
    );
}

#[test]
fn session_builds_model_thinking_compaction_custom_and_branch_summary_context() {
    let mut session = Session::new(InMemorySessionStorage::new());
    session
        .append_message(user_message_text("one"))
        .expect("one");
    session
        .append_model_change("openai", "gpt-4.1")
        .expect("model");
    session
        .append_thinking_level_change("high")
        .expect("thinking");
    session
        .append_message(assistant_message_text("two"))
        .expect("two");
    let user2 = session
        .append_message(user_message_text("three"))
        .expect("three");
    session
        .append_message(assistant_message_text("four"))
        .expect("four");
    session
        .append_compaction("summary", user2.clone(), 1234)
        .expect("compact");
    session
        .append_message(user_message_text("five"))
        .expect("five");
    session
        .append_custom_message_entry(
            "custom",
            custom_text_content("hello"),
            true,
            Some(json!({ "ok": true })),
        )
        .expect("custom");

    let context = session.build_context().expect("context");
    assert_eq!(context.thinking_level, "high");
    assert_eq!(context.model.expect("model").provider, "test");
    assert_eq!(
        context
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["compactionSummary", "user", "assistant", "user", "custom"]
    );

    let summary_id = session
        .move_to(
            Some(user2),
            Some(BranchMoveSummary {
                summary: "summary text".to_owned(),
                details: None,
                from_hook: None,
            }),
        )
        .expect("move summary")
        .expect("summary id");
    assert!(matches!(
        session.get_entry(&summary_id),
        Some(SessionTreeEntry::BranchSummary { .. })
    ));
}

#[test]
fn session_labels_and_session_info_do_not_affect_context() {
    let mut session = Session::new(InMemorySessionStorage::new());
    let user1 = session
        .append_message(user_message_text("one"))
        .expect("user");
    session
        .append_label(user1.clone(), Some("checkpoint".to_owned()))
        .expect("label");
    session
        .append_custom_entry("metadata", Some(json!({ "ok": true })))
        .expect("custom");
    session.append_session_name(" name ").expect("name");
    assert_eq!(session.label(&user1), Some("checkpoint".to_owned()));
    assert_eq!(session.session_name().as_deref(), Some("name"));
    assert_eq!(session.build_context().expect("context").messages.len(), 1);
    assert!(matches!(
        session.entries().get(2),
        Some(SessionTreeEntry::Custom {
            custom_type,
            data: Some(data),
            ..
        }) if custom_type == "metadata" && data == &json!({ "ok": true })
    ));

    let err = session
        .append_label("missing", Some("checkpoint".to_owned()))
        .expect_err("missing");
    assert_eq!(err.code, SessionErrorCode::NotFound);
}

#[test]
fn jsonl_session_persists_leaf_entries_and_wire_entry_types() {
    let dir = temp_dir();
    let path = dir.join("session.jsonl");
    let storage = JsonlSessionStorage::create(&path, dir.to_string_lossy(), "session-1", None)
        .expect("jsonl");
    let mut session = Session::new(storage);
    let user1 = session
        .append_message(user_message_text("one"))
        .expect("user");
    session
        .append_message(assistant_message_text("two"))
        .expect("assistant");
    session
        .append_label(user1.clone(), Some("checkpoint".to_owned()))
        .expect("label");
    session.append_session_name("name").expect("name");
    session.move_to(Some(user1.clone()), None).expect("move");
    session
        .append_message(assistant_message_text("branched"))
        .expect("branch");

    let session2 = Session::new(JsonlSessionStorage::open(&path).expect("reload"));
    assert_eq!(
        session2
            .build_context()
            .expect("context")
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user", "assistant"]
    );
    assert_eq!(session2.label(&user1), Some("checkpoint".to_owned()));
    assert_eq!(session2.session_name().as_deref(), Some("name"));

    let lines = fs::read_to_string(&path).expect("read");
    let json_lines = lines
        .trim()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("json line"))
        .collect::<Vec<_>>();
    assert!(json_lines.len() > 1);
    assert_eq!(json_lines[0]["type"], "session");
    assert_eq!(json_lines[0]["version"], 3);

    let entries = &json_lines[1..];
    assert!(entries.iter().any(|entry| entry["type"] == "leaf"));
    for entry in entries {
        assert_ne!(entry["type"], "entry");
        assert!(entry["id"].is_string());
    }
}

#[test]
fn in_memory_repo_opens_deletes_and_forks_by_metadata() {
    let mut repo = InMemorySessionRepo::new();
    let mut session = repo.create(Some("session-1".to_owned()));
    let metadata = session.in_memory_metadata().expect("metadata").clone();
    let user1 = session
        .append_message(user_message_text("one"))
        .expect("user1");
    let assistant1 = session
        .append_message(assistant_message_text("two"))
        .expect("assistant1");
    let user2 = session
        .append_message(user_message_text("three"))
        .expect("user2");

    assert_eq!(
        repo.open(&metadata).expect("open").metadata_id(),
        "session-1"
    );
    assert_eq!(
        repo.list()
            .iter()
            .map(|metadata| metadata.id.as_str())
            .collect::<Vec<_>>(),
        vec!["session-1"]
    );

    let fork = repo
        .fork(
            &metadata,
            SessionForkOptions {
                entry_id: Some(user2.clone()),
                id: Some("session-2".to_owned()),
                ..Default::default()
            },
        )
        .expect("fork");
    assert_eq!(
        fork.entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec![user1.as_str(), assistant1.as_str()]
    );

    let full_fork = repo
        .fork(
            &metadata,
            SessionForkOptions {
                id: Some("session-3".to_owned()),
                ..Default::default()
            },
        )
        .expect("full fork");
    assert_eq!(
        full_fork
            .entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec![user1.as_str(), assistant1.as_str(), user2.as_str()]
    );

    repo.delete(&metadata);
    assert!(
        repo.open(&metadata)
            .expect_err("deleted")
            .message
            .contains("Session not found")
    );
}

#[test]
fn jsonl_repo_stores_lists_opens_deletes_and_forks_by_metadata() {
    let root = temp_dir();
    let repo = JsonlSessionRepo::new(&root);
    let cwd = "/tmp/my-project";
    let other_cwd = "/tmp/other-project";

    let session = repo
        .create(cwd, Some("source-session".to_owned()), None)
        .expect("create");
    let other = repo
        .create(other_cwd, Some("other-session".to_owned()), None)
        .expect("other");
    let source_metadata = session.jsonl_metadata().expect("metadata").clone();
    let other_metadata = other.jsonl_metadata().expect("metadata").clone();
    assert!(source_metadata.path.contains("--tmp-my-project--"));
    assert!(other_metadata.path.contains("--tmp-other-project--"));
    assert!(PathBuf::from(&source_metadata.path).exists());
    assert_eq!(
        repo.list(Some(cwd))
            .expect("list cwd")
            .iter()
            .map(|metadata| metadata.id.as_str())
            .collect::<Vec<_>>(),
        vec!["source-session"]
    );
    let mut all = repo
        .list(None)
        .expect("list all")
        .into_iter()
        .map(|metadata| metadata.id)
        .collect::<Vec<_>>();
    all.sort();
    assert_eq!(
        all,
        vec!["other-session".to_owned(), "source-session".to_owned()]
    );

    let mut opened = repo.open(&source_metadata).expect("open");
    let user1 = opened
        .append_message(user_message_text("one"))
        .expect("user1");
    let assistant1 = opened
        .append_message(assistant_message_text("two"))
        .expect("assistant1");
    let user2 = opened
        .append_message(user_message_text("three"))
        .expect("user2");
    let updated_metadata = opened.jsonl_metadata().expect("metadata").clone();

    let fork = repo
        .fork(
            &updated_metadata,
            JsonlSessionForkOptions {
                cwd: "/tmp/target".to_owned(),
                parent_session_path: None,
                fork: SessionForkOptions {
                    entry_id: Some(user2.clone()),
                    id: Some("fork-session".to_owned()),
                    ..Default::default()
                },
            },
        )
        .expect("fork");
    let fork_metadata = fork.jsonl_metadata().expect("fork metadata");
    assert_eq!(fork_metadata.cwd, "/tmp/target");
    assert_eq!(
        fork_metadata.parent_session_path.as_deref(),
        Some(updated_metadata.path.as_str())
    );
    assert_eq!(
        fork.entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec![user1.as_str(), assistant1.as_str()]
    );

    let full_fork = repo
        .fork(
            &updated_metadata,
            JsonlSessionForkOptions {
                cwd: "/tmp/target".to_owned(),
                parent_session_path: None,
                fork: SessionForkOptions {
                    id: Some("full-fork-session".to_owned()),
                    ..Default::default()
                },
            },
        )
        .expect("full fork");
    assert_eq!(
        full_fork
            .entries()
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec![user1.as_str(), assistant1.as_str(), user2.as_str()]
    );

    repo.delete(&updated_metadata).expect("delete");
    assert!(!PathBuf::from(&updated_metadata.path).exists());
    assert!(
        repo.open(&updated_metadata)
            .expect_err("deleted")
            .message
            .contains("Session not found")
    );
}
