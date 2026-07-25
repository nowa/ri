//! Cross-implementation interop: ri reads session files that pi's own storage
//! code wrote.
//!
//! Fixtures under `tests/fixtures/` are produced by
//! `tools/interop/generate-pi-fixtures.ts`, which drives pi's
//! `JsonlSessionStorage` directly — so these are bytes pi actually emits, not
//! hand-written samples. See `docs/BEHAVIOR_PARITY_AUDIT.md`.

use ri_agent_core::*;
use serde_json::Value;
use std::{fs, path::PathBuf};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

/// Copy a read-only fixture into a temp dir: opening a session is a
/// read/append surface and tests must not mutate the committed file.
fn copy_fixture(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ri-interop-{}", uuidv7()));
    fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join(name);
    fs::copy(fixture(name), &path).expect("copy fixture");
    path
}

fn open_fixture(name: &str) -> JsonlSessionStorage {
    JsonlSessionStorage::open(copy_fixture(name)).expect("pi session opens in ri")
}

#[test]
fn reads_a_pi_written_session_tree() {
    let storage = open_fixture("pi-session-v3.jsonl");

    let metadata = storage.metadata();
    assert_eq!(metadata.id, "01998f1e-0000-7000-8000-00000000abcd");
    assert_eq!(metadata.cwd, "/workspace/project");
    // Opaque header metadata round-trips as written.
    assert_eq!(
        metadata
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("fixture"))
            .and_then(Value::as_str),
        Some("interop")
    );
    assert_eq!(
        metadata
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("nested"))
            .and_then(|nested| nested.pointer("/keep")),
        Some(&Value::Bool(true))
    );

    let entries = storage.entries();
    // 13 appended entries plus the leaf marker pi writes on setLeafId.
    assert_eq!(entries.len(), 14);
    assert_eq!(
        entries.iter().map(|entry| entry.id()).collect::<Vec<_>>()[..13],
        [
            "e0000001", "e0000002", "e0000003", "e0000004", "e0000005", "e0000006", "e0000007",
            "e0000008", "e0000009", "e0000010", "e0000011", "e0000012", "e0000013",
        ]
    );
    assert_eq!(
        storage.leaf_id().expect("leaf"),
        Some("e0000013".to_owned())
    );
    assert_eq!(storage.label("e0000005"), Some("first read".to_owned()));

    // Every pi entry type ri claims to support parsed into its own variant.
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000002"),
        Some(SessionTreeEntry::ModelChange { provider, model_id, .. })
            if provider == "anthropic" && model_id == "claude-sonnet-4-5"
    ));
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000003"),
        Some(SessionTreeEntry::ThinkingLevelChange { thinking_level, .. })
            if thinking_level == "high"
    ));
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000004"),
        Some(SessionTreeEntry::ActiveToolsChange { active_tool_names, .. })
            if active_tool_names == &["read".to_owned(), "bash".to_owned()]
    ));
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000008"),
        Some(SessionTreeEntry::Custom { custom_type, data, .. })
            if custom_type == "appState"
                && data.as_ref().and_then(|data| data.pointer("/scroll")) == Some(&Value::from(42))
    ));
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000009"),
        Some(SessionTreeEntry::BranchSummary { from_id, summary, usage, .. })
            if from_id == "e0000008"
                && summary.starts_with("The user explored")
                && usage.as_ref().is_some_and(|usage| usage.total_tokens == 180)
    ));
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000012"),
        Some(SessionTreeEntry::SessionInfo { name, .. })
            if name.as_deref() == Some("Parser work")
    ));

    // Assistant content survives with thinking signatures and tool calls.
    let Some(SessionTreeEntry::Message { message, .. }) =
        entries.iter().find(|entry| entry.id() == "e0000005")
    else {
        panic!("assistant entry");
    };
    let SessionEntryMessage::Llm(ri_llm_provider::Message::Assistant(assistant)) = message else {
        panic!("assistant message, got {message:?}");
    };
    assert_eq!(assistant.provider, "anthropic");
    assert_eq!(assistant.stop_reason, ri_llm_provider::StopReason::ToolUse);
    assert_eq!(assistant.usage.total_tokens, 180);
    assert!(matches!(
        &assistant.content[0],
        ri_llm_provider::AssistantContent::Thinking(thinking)
            if thinking.thinking_signature.as_deref() == Some("sig-abc")
    ));
    assert!(matches!(
        &assistant.content[2],
        ri_llm_provider::AssistantContent::ToolCall(tool_call) if tool_call.name == "read"
    ));

    // pi's bash-execution custom messages keep their custom type and details.
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "e0000007"),
        Some(SessionTreeEntry::CustomMessage { custom_type, display, details, .. })
            if custom_type == "bashExecution"
                && *display
                && details.as_ref().and_then(|d| d.pointer("/exitCode")) == Some(&Value::from(0))
    ));
}

#[test]
fn builds_context_from_a_pi_written_session() {
    let storage = open_fixture("pi-session-v3.jsonl");
    let session = Session::new(SessionStorageKind::Jsonl(storage));

    // Ground truth captured from pi via tools/interop/dump-pi-session.ts:
    // the retained-tail compaction roots the branch, so the pre-boundary
    // model/thinking/tool state is genuinely out of scope for both sides.
    let context = session.build_context().expect("context");
    assert_eq!(context.model, None);
    assert_eq!(context.thinking_level, "off");
    assert_eq!(context.active_tool_names, None);

    let roles = context
        .messages
        .iter()
        .map(|message| message.role())
        .collect::<Vec<_>>();
    assert_eq!(roles, ["compactionSummary", "user", "user"]);

    let texts = context
        .messages
        .iter()
        .map(|message| format!("{message:?}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        texts.contains("Earlier: the user asked about the parser"),
        "{texts}"
    );
    // The retained tail pi stored on the compaction entry is replayed.
    assert!(texts.contains("Keep going"), "{texts}");
    assert!(texts.contains("Now optimize it"), "{texts}");
}

#[test]
fn appending_to_a_pi_session_preserves_the_original_lines() {
    let path = copy_fixture("pi-session-v3.jsonl");
    let original = fs::read_to_string(&path).expect("read fixture");

    let mut storage = JsonlSessionStorage::open(&path).expect("open");
    let entry = SessionTreeEntry::Message {
        id: "ri000001".to_owned(),
        parent_id: Some("e0000013".to_owned()),
        timestamp: "2026-07-26T00:01:00.000Z".to_owned(),
        message: SessionEntryMessage::Llm(ri_llm_provider::Message::User(
            ri_llm_provider::UserMessage::text("appended by ri"),
        )),
    };
    storage.append_entry(entry).expect("append");

    // ri appends; it must never rewrite pi's bytes — fields ri does not model
    // (e.g. `retainedTail`) stay intact on disk for pi to read back.
    let updated = fs::read_to_string(&path).expect("read updated");
    assert!(
        updated.starts_with(&original),
        "existing lines were rewritten"
    );
    assert!(original.contains("\"retainedTail\""));
    assert!(updated.contains("\"retainedTail\""));
    assert_eq!(updated.lines().count(), original.lines().count() + 1);
}

#[test]
fn reads_pi_sessions_that_omit_optional_fields() {
    let storage = open_fixture("pi-session-v3-optional-fields.jsonl");

    let entries = storage.entries();
    assert_eq!(entries.len(), 6);

    // pi's `firstKeptEntryId` is optional; a compaction without it means
    // "nothing kept".
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "f0000002"),
        Some(SessionTreeEntry::Compaction { first_kept_entry_id, tokens_before, .. })
            if first_kept_entry_id.is_none() && *tokens_before == 999
    ));
    // Custom message content may be a plain string instead of blocks.
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "f0000003"),
        Some(SessionTreeEntry::CustomMessage { content, display, .. })
            if !*display && matches!(content, CustomMessageContent::Text(text) if text == "plain string content")
    ));
    // A label with no text clears the label.
    assert_eq!(storage.label("f0000001"), None);
    assert!(matches!(
        entries.iter().find(|entry| entry.id() == "f0000005"),
        Some(SessionTreeEntry::SessionInfo { name: None, .. })
    ));
    // `setLeafId(null)` resets navigation to the root.
    assert_eq!(storage.leaf_id().expect("leaf"), None);
}

#[test]
fn reads_pi_written_auth_json_credentials() {
    let body = fs::read_to_string(fixture("pi-auth.json")).expect("read auth fixture");
    let credentials: std::collections::BTreeMap<String, ri_llm_provider::auth::Credential> =
        serde_json::from_str(&body).expect("pi auth.json parses into ri credentials");

    assert_eq!(credentials.len(), 5);
    let Some(ri_llm_provider::auth::Credential::OAuth(anthropic)) = credentials.get("anthropic")
    else {
        panic!("anthropic oauth credential");
    };
    assert_eq!(anthropic.access, "access-token");
    assert_eq!(anthropic.expires, 1_784_000_000_000);

    // Provider-specific extras round-trip through `extra`.
    let Some(ri_llm_provider::auth::Credential::OAuth(codex)) = credentials.get("openai-codex")
    else {
        panic!("codex credential");
    };
    assert_eq!(
        codex.extra.get("accountId").and_then(Value::as_str),
        Some("acct_123")
    );
    let Some(ri_llm_provider::auth::Credential::OAuth(copilot)) = credentials.get("github-copilot")
    else {
        panic!("copilot credential");
    };
    assert_eq!(
        copilot.extra.get("enterpriseUrl").and_then(Value::as_str),
        Some("ghe.example.com")
    );
    assert_eq!(
        copilot
            .extra
            .get("availableModelIds")
            .and_then(Value::as_array)
            .map(|ids| ids.len()),
        Some(2)
    );

    // Api-key credentials carry either a key or provider env.
    let Some(ri_llm_provider::auth::Credential::ApiKey(openai)) = credentials.get("openai") else {
        panic!("openai api key credential");
    };
    assert_eq!(openai.key.as_deref(), Some("sk-test"));
    let Some(ri_llm_provider::auth::Credential::ApiKey(bedrock)) =
        credentials.get("amazon-bedrock")
    else {
        panic!("bedrock api key credential");
    };
    assert_eq!(bedrock.key, None);
    assert_eq!(
        bedrock.env.get("AWS_PROFILE").map(String::as_str),
        Some("prod")
    );

    // Re-serializing keeps pi's on-disk shape (type tag + camelCase extras).
    let round_tripped: Value =
        serde_json::from_str(&serde_json::to_string(&credentials).expect("serialize"))
            .expect("reparse");
    let original: Value = serde_json::from_str(&body).expect("parse original");
    assert_eq!(round_tripped, original);
}
