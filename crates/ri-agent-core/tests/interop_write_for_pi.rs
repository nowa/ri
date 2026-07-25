//! Reverse interop: ri writes a session that pi must be able to read.
//!
//! The Rust side writes the file into `target/interop/` and asserts the wire
//! shape ri emits; `tools/interop/dump-pi-session.ts` then loads the same file
//! with pi's own storage to confirm pi agrees (see docs/BEHAVIOR_PARITY_AUDIT.md).

use ri_agent_core::*;
use ri_llm_provider::{AssistantContent, Message, TextContent, ThinkingContent, ToolCall, Usage};
use serde_json::{Map, Value, json};
use std::{fs, path::PathBuf};

/// Written under `target/` so the bun verifier can pick it up after a run.
fn output_path() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/interop")
        .canonicalize()
        .unwrap_or_else(|_| {
            let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/interop");
            fs::create_dir_all(&path).expect("interop dir");
            path
        });
    fs::create_dir_all(&dir).expect("interop dir");
    dir.join("ri-session-v3.jsonl")
}

#[test]
fn writes_a_session_in_pis_on_disk_shape() {
    let path = output_path();
    let _ = fs::remove_file(&path);

    let mut metadata = Map::new();
    metadata.insert("writer".to_owned(), json!("ri"));
    let storage = JsonlSessionStorage::create_with_metadata(
        &path,
        "/workspace/project",
        "01998f1e-0000-7000-8000-00000000ri00",
        None,
        Some(metadata),
    )
    .expect("create session");
    let mut session = Session::new(SessionStorageKind::Jsonl(storage));

    session
        .append_message(Message::User(ri_llm_provider::UserMessage::text(
            "Port the parser",
        )))
        .expect("user message");
    session
        .append_model_change("anthropic", "claude-sonnet-4-5")
        .expect("model change");
    session
        .append_thinking_level_change("high")
        .expect("thinking level");
    session
        .append_active_tools_change(vec!["read".to_owned(), "bash".to_owned()])
        .expect("active tools");

    let mut assistant = ri_llm_provider::AssistantMessage {
        content: vec![
            AssistantContent::Thinking(ThinkingContent {
                thinking: "Planning".to_owned(),
                thinking_signature: Some("sig-ri".to_owned()),
                redacted: false,
            }),
            AssistantContent::Text(TextContent::new("Reading the file.")),
            AssistantContent::ToolCall(ToolCall {
                id: "call_ri_1".to_owned(),
                name: "read".to_owned(),
                arguments: json!({ "path": "parser.rs" })
                    .as_object()
                    .cloned()
                    .expect("object"),
                thought_signature: None,
            }),
        ],
        api: "anthropic-messages".to_owned(),
        provider: "anthropic".to_owned(),
        model: "claude-sonnet-4-5".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: ri_llm_provider::StopReason::ToolUse,
        error_message: None,
        timestamp: 1_784_000_100_000,
    };
    assistant.usage.total_tokens = 42;
    session
        .append_message(Message::Assistant(assistant))
        .expect("assistant message");
    session
        .append_bash_execution(BashExecutionMessage::new(
            "cargo test",
            "ok",
            1_784_000_110_000,
        ))
        .expect("bash execution");
    let kept = session
        .append_message(Message::User(ri_llm_provider::UserMessage::text(
            "Keep this one",
        )))
        .expect("kept message");
    session
        .append_custom_message_entry(
            "note",
            CustomMessageContent::Text("side note".to_owned()),
            true,
            None,
        )
        .expect("custom message");
    session
        .append_compaction_entry(
            "Summarized the earlier work.",
            Some(kept.clone()),
            9_000,
            None,
            None,
            None,
            None,
        )
        .expect("compaction");
    session
        .append_label(kept.clone(), Some("keep".to_owned()))
        .expect("label");
    session
        .append_session_name("ri interop")
        .expect("session info");

    // The file pi will read: a v3 header plus one JSON object per line.
    let contents = fs::read_to_string(&path).expect("read session");
    let mut lines = contents.lines();
    let header: Value = serde_json::from_str(lines.next().expect("header")).expect("header json");
    assert_eq!(header["type"], json!("session"));
    assert_eq!(header["version"], json!(3));
    assert_eq!(header["cwd"], json!("/workspace/project"));
    assert_eq!(header["metadata"]["writer"], json!("ri"));

    let entries = lines
        .map(|line| serde_json::from_str::<Value>(line).expect("entry json"))
        .collect::<Vec<_>>();
    let types = entries
        .iter()
        .map(|entry| entry["type"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(
        types,
        [
            "message",
            "model_change",
            "thinking_level_change",
            "active_tools_change",
            "message",
            "message",
            "message",
            "custom_message",
            "compaction",
            "label",
            "session_info",
        ]
    );

    // Field names must be pi's camelCase wire names, not Rust snake_case.
    for entry in &entries {
        assert!(entry.get("id").is_some_and(Value::is_string), "{entry}");
        assert!(entry.get("parentId").is_some(), "{entry}");
        assert!(
            entry.get("timestamp").is_some_and(Value::is_string),
            "{entry}"
        );
        assert!(
            entry
                .as_object()
                .expect("object")
                .keys()
                .all(|key| !key.contains('_')
                    || matches!(
                        key.as_str(),
                        "model_change" | "thinking_level_change" | "active_tools_change"
                    )),
            "snake_case field leaked: {entry}"
        );
    }
    let compaction = entries
        .iter()
        .find(|entry| entry["type"] == json!("compaction"))
        .expect("compaction entry");
    assert_eq!(compaction["firstKeptEntryId"], json!(kept));
    assert_eq!(compaction["tokensBefore"], json!(9_000));
    // Optional fields ri does not set stay absent rather than serializing null.
    assert!(compaction.get("retainedTail").is_none());
    assert!(compaction.get("details").is_none());

    // Assistant messages keep pi's role-tagged AgentMessage shape.
    let assistant = entries
        .iter()
        .filter(|entry| entry["type"] == json!("message"))
        .find(|entry| entry["message"]["role"] == json!("assistant"))
        .expect("assistant entry");
    assert_eq!(assistant["message"]["provider"], json!("anthropic"));
    assert_eq!(assistant["message"]["stopReason"], json!("toolUse"));
    assert_eq!(
        assistant["message"]["content"][0]["thinkingSignature"],
        json!("sig-ri")
    );
    assert_eq!(
        assistant["message"]["content"][2]["type"],
        json!("toolCall")
    );

    // Bash-execution messages round-trip through pi's custom-message shape.
    let bash = entries
        .iter()
        .filter(|entry| entry["type"] == json!("message"))
        .find(|entry| entry["message"].get("command").is_some())
        .expect("bash entry");
    assert_eq!(bash["message"]["command"], json!("cargo test"));

    // Re-opening our own file must reproduce the same tree, and the branch and
    // context must match what pi reports for the same bytes (verified by
    // tools/interop/verify.sh, which runs pi's reader over this file).
    let reopened = JsonlSessionStorage::open(&path).expect("reopen");
    assert_eq!(reopened.entries().len(), entries.len());
    assert_eq!(reopened.label(&kept), Some("keep".to_owned()));

    let session = Session::new(SessionStorageKind::Jsonl(reopened));
    let branch = session.branch(None).expect("branch");
    assert_eq!(branch.len(), 5, "pi walks 5 entries from the compaction");
    assert_eq!(branch[0].id(), kept, "the kept entry roots the branch");

    let context = session.build_context().expect("context");
    assert_eq!(
        context
            .messages
            .iter()
            .map(|message| message.role())
            .collect::<Vec<_>>(),
        ["compactionSummary", "user", "custom"]
    );
    let rendered = context
        .messages
        .iter()
        .map(|message| format!("{message:?}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(rendered.contains("Keep this one"), "{rendered}");
    assert!(rendered.contains("side note"), "{rendered}");
}
