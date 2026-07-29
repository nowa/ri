//! Behavior-parity audit regressions against pi v0.81.1: active-branch
//! compaction scope, `getBranch` compaction boundaries, forward cursor
//! paging, zero-usage assistant anchors, per-message token rounding, and
//! gitignore `**`/character-class patterns in the skills loader.

use ri_agent_core::*;
use ri_llm_provider::*;
// Disambiguate names exported by both glob imports.
use ri_agent_core::harness::compaction::estimate_context_tokens;
use ri_llm_provider::uuidv7;
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

fn usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
        cache_write_1h: None,
        reasoning: None,
        total_tokens: input + output + cache_read + cache_write,
        cost: UsageCost::default(),
    }
}

fn assistant_with_usage(text: &str, usage: Usage, stop_reason: StopReason) -> Message {
    let mut assistant = faux_assistant_message(text, Default::default());
    assistant.usage = usage;
    assistant.stop_reason = stop_reason;
    Message::Assistant(assistant)
}

fn message_entry(id: &str, parent_id: Option<&str>, message: Message) -> SessionTreeEntry {
    SessionTreeEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        message: message.into(),
    }
}

fn user_entry(id: &str, parent_id: Option<&str>, text: &str) -> SessionTreeEntry {
    message_entry(id, parent_id, Message::User(UserMessage::text(text)))
}

fn compaction_entry(
    id: &str,
    parent_id: Option<&str>,
    first_kept_entry_id: &str,
) -> SessionTreeEntry {
    SessionTreeEntry::Compaction {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        summary: "compacted history".to_owned(),
        first_kept_entry_id: Some(first_kept_entry_id.to_owned()),
        tokens_before: 1234,
        retained_tail: None,
        details: None,
        from_hook: None,
        usage: None,
    }
}

fn entry_ids(entries: &[SessionTreeEntry]) -> Vec<String> {
    entries
        .iter()
        .map(SessionTreeEntry::id)
        .map(ToOwned::to_owned)
        .collect()
}

fn temp_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ri-audit-behavior-{label}-{}", uuidv7()));
    fs::create_dir_all(&dir).expect("temp dir");
    dir
}

// pi session/session.ts getBranch(): the branch is the path from the leaf to
// the root, stopping at the previous compaction boundary
// (getPathToRootOrCompaction), not the full walk to the root.
#[test]
fn session_branch_stops_at_the_previous_compaction_boundary() {
    let storage = InMemorySessionStorage::with_options(
        Some(vec![
            user_entry("aaaa", None, "pre-compaction history"),
            message_entry(
                "bbbb",
                Some("aaaa"),
                Message::Assistant(faux_assistant_message("kept tail", Default::default())),
            ),
            compaction_entry("comp", Some("bbbb"), "bbbb"),
            user_entry("dddd", Some("comp"), "post-compaction"),
        ]),
        None,
    )
    .expect("storage");
    let session = Session::new(storage);

    let branch = session.branch(None).expect("branch");
    assert_eq!(entry_ids(&branch), vec!["bbbb", "comp", "dddd"]);
    let branch_from = session.branch(Some("dddd")).expect("branch from id");
    assert_eq!(entry_ids(&branch_from), vec!["bbbb", "comp", "dddd"]);

    // The raw append log still contains the full history.
    assert_eq!(
        entry_ids(&session.entries()),
        vec!["aaaa", "bbbb", "comp", "dddd"]
    );
}

// pi branch-summarization.ts collectEntriesForBranchSummary(): both paths are
// compaction-stopped branches, so an ancestor that lies before the target's
// compaction boundary is not a common ancestor.
#[test]
fn branch_summary_collection_ignores_ancestors_behind_a_compaction_boundary() {
    let storage = InMemorySessionStorage::with_options(
        Some(vec![
            user_entry("aaaa", None, "shared root"),
            user_entry("bbbb", Some("aaaa"), "old branch"),
            user_entry("cccc", Some("aaaa"), "target branch"),
            compaction_entry("comp", Some("cccc"), "cccc"),
            user_entry("ffff", Some("comp"), "target leaf"),
        ]),
        None,
    )
    .expect("storage");
    let session = Session::new(storage);

    let result =
        collect_entries_for_branch_summary(&session, Some("bbbb"), "ffff").expect("collect");
    // The target branch [cccc, comp, ffff] stops at the compaction boundary,
    // so "aaaa" is invisible and no common ancestor is found; the whole old
    // path is summarized.
    assert_eq!(result.common_ancestor_id, None);
    assert_eq!(entry_ids(&result.entries), vec!["aaaa", "bbbb"]);
}

// pi agent-harness.ts:720: compact() feeds session.getBranch() (the active
// branch) into prepareCompaction, so abandoned sibling branches neither reach
// the hook nor inflate tokensBefore.
#[tokio::test]
async fn harness_compaction_feeds_the_active_branch_only() {
    let registration = register_faux_provider(RegisterFauxProviderOptions {
        models: vec![FauxModelDefinition::new("audit-branch-compact-model")],
        ..Default::default()
    });
    let storage = InMemorySessionStorage::with_options(
        Some(vec![
            user_entry("m1", None, "first question"),
            message_entry(
                "m2",
                Some("m1"),
                Message::Assistant(faux_assistant_message(
                    "abandoned ".repeat(4_000),
                    Default::default(),
                )),
            ),
            user_entry("m3", Some("m1"), "second question"),
            message_entry(
                "m4",
                Some("m3"),
                Message::Assistant(faux_assistant_message("short answer", Default::default())),
            ),
            user_entry("m5", Some("m4"), "third question"),
        ]),
        None,
    )
    .expect("storage");
    let session = Session::new(storage);

    let options = AgentHarnessOptions::new(
        LocalExecutionEnv::new("/tmp"),
        session.clone(),
        registration.get_model(),
    );
    let harness = AgentHarness::new(options);
    let seen = Arc::new(Mutex::new(Vec::<(Vec<String>, u64)>::new()));
    let seen_ref = seen.clone();
    harness.on_session_before_compact(move |event| {
        seen_ref.lock().expect("seen").push((
            entry_ids(&event.branch_entries),
            event.preparation.tokens_before,
        ));
        Ok(Some(SessionBeforeCompactResult {
            cancel: false,
            compaction: Some(CompactionResult {
                summary: "hook summary".to_owned(),
                first_kept_entry_id: event.preparation.first_kept_entry_id,
                tokens_before: event.preparation.tokens_before,
                details: CompactionDetails {
                    read_files: Vec::new(),
                    modified_files: Vec::new(),
                },
                usage: None,
            }),
        }))
    });

    harness
        .compact_session(AgentHarnessCompactionOptions {
            settings: CompactionThresholdSettings {
                enabled: true,
                reserve_tokens: 128,
                keep_recent_tokens: 1,
            },
            custom_instructions: None,
        })
        .await
        .expect("compact")
        .expect("compaction result");

    let seen = seen.lock().expect("seen").clone();
    assert_eq!(seen.len(), 1);
    let (branch_ids, tokens_before) = &seen[0];
    // The abandoned sibling "m2" is not part of the active branch.
    assert_eq!(*branch_ids, vec!["m1", "m3", "m4", "m5"]);
    // tokensBefore covers only the active branch: the 40k-char abandoned
    // assistant (~10k tokens) must not be counted.
    assert!(
        *tokens_before < 1_000,
        "tokens_before overcounts: {tokens_before}"
    );
    registration.unregister();
}

// pi jsonl-storage.ts:371-375 getEntries(): forward paging via
// entries.slice(afterEntrySeq, afterEntrySeq + limit); afterEntrySeq is
// honored even without a limit.
#[test]
fn in_memory_entries_page_pages_forward_from_the_cursor() {
    let mut storage = InMemorySessionStorage::new();
    let mut parent: Option<String> = None;
    for id in ["e1", "e2", "e3", "e4", "e5"] {
        storage.append_entry(user_entry(id, parent.as_deref(), id));
        parent = Some(id.to_owned());
    }
    let storage = SessionStorageKind::from(storage);

    let page = |limit, after| {
        entry_ids(
            &storage
                .entries_page(&SessionEntryCursorOptions {
                    limit,
                    after_entry_seq: after,
                })
                .expect("page"),
        )
    };
    assert_eq!(page(Some(2), None), vec!["e1", "e2"]);
    assert_eq!(page(Some(2), Some(3)), vec!["e4", "e5"]);
    assert_eq!(page(Some(10), Some(2)), vec!["e3", "e4", "e5"]);
    // The anchor applies even without a limit.
    assert_eq!(page(None, Some(3)), vec!["e4", "e5"]);
    assert_eq!(page(None, None), vec!["e1", "e2", "e3", "e4", "e5"]);
    assert_eq!(page(Some(2), Some(10)), Vec::<String>::new());
}

#[test]
fn sqlite_entries_page_pages_forward_from_the_cursor() {
    let dir = temp_dir("sqlite-paging");
    let repo = SqliteSessionRepo::new(dir.join("sessions.db"));
    let mut storage = repo
        .create(SqliteSessionCreateOptions {
            id: Some("session-forward-page".to_owned()),
            cwd: "/tmp/project".to_owned(),
            ..Default::default()
        })
        .expect("create");
    let mut parent: Option<String> = None;
    for id in ["e1", "e2", "e3", "e4", "e5"] {
        storage
            .append_entry(user_entry(id, parent.as_deref(), id))
            .expect("append");
        parent = Some(id.to_owned());
    }

    let page = |limit, after| {
        entry_ids(
            &storage
                .entries_page(&SessionEntryCursorOptions {
                    limit,
                    after_entry_seq: after,
                })
                .expect("page"),
        )
    };
    assert_eq!(page(Some(2), None), vec!["e1", "e2"]);
    assert_eq!(page(Some(2), Some(3)), vec!["e4", "e5"]);
    assert_eq!(page(Some(10), Some(2)), vec!["e3", "e4", "e5"]);
    assert_eq!(page(None, Some(3)), vec!["e4", "e5"]);
    assert_eq!(page(None, None), vec!["e1", "e2", "e3", "e4", "e5"]);
    assert_eq!(page(Some(2), Some(10)), Vec::<String>::new());
    let _ = fs::remove_dir_all(&dir);
}

// pi compaction.ts:174-185 getAssistantUsage(): a usage anchor is only
// accepted when calculateContextTokens(usage) > 0; zero-usage assistant
// messages fall through to older ones.
#[test]
fn zero_usage_assistant_messages_do_not_anchor_context_estimates() {
    let messages = vec![
        assistant_with_usage("real", usage(100, 50, 25, 25), StopReason::Stop),
        assistant_with_usage("zero", usage(0, 0, 0, 0), StopReason::Stop),
        Message::User(UserMessage::text("tail")),
    ];
    assert_eq!(
        get_last_assistant_usage(&messages).map(|(index, usage)| (index, usage.total_tokens)),
        Some((0, 200))
    );
    // "zero" (4 chars) and "tail" (4 chars) are estimated as trailing tokens.
    assert_eq!(
        estimate_context_token_usage(&messages),
        ContextTokenEstimate {
            tokens: 202,
            usage_tokens: 200,
            last_usage_index: Some(0),
        }
    );

    // Only zero-usage anchors: no anchor at all, pure estimation.
    let only_zero = vec![assistant_with_usage(
        "zero",
        usage(0, 0, 0, 0),
        StopReason::Stop,
    )];
    assert!(get_last_assistant_usage(&only_zero).is_none());
    assert_eq!(
        estimate_context_token_usage(&only_zero).last_usage_index,
        None
    );

    let entries = vec![
        message_entry(
            "a1",
            None,
            assistant_with_usage("real", usage(100, 50, 25, 25), StopReason::Stop),
        ),
        message_entry(
            "a2",
            Some("a1"),
            assistant_with_usage("zero", usage(0, 0, 0, 0), StopReason::Stop),
        ),
    ];
    assert_eq!(
        get_last_assistant_usage_from_entries(&entries).map(|usage| usage.total_tokens),
        Some(200)
    );

    let session_messages = vec![
        SessionMessage::Llm {
            message: assistant_with_usage("real", usage(100, 50, 25, 25), StopReason::Stop),
        },
        SessionMessage::Llm {
            message: assistant_with_usage("zero", usage(0, 0, 0, 0), StopReason::Stop),
        },
    ];
    assert_eq!(
        get_last_session_assistant_usage(&session_messages)
            .map(|(index, usage)| (index, usage.total_tokens)),
        Some((0, 200))
    );
}

// pi compaction.ts:275-315 estimateTokens(): one ceil(chars / 4) per MESSAGE
// over the summed character estimate, not one ceil per content block.
#[test]
fn token_estimates_round_once_per_message() {
    let user_blocks = Message::User(UserMessage {
        content: UserContentValue::Blocks(vec![
            UserContent::Text(TextContent::new("a")),
            UserContent::Text(TextContent::new("b")),
            UserContent::Text(TextContent::new("c")),
        ]),
        timestamp: 0,
    });
    // 3 chars in one message: ceil(3 / 4) = 1 (per-block rounding gave 3).
    assert_eq!(
        estimate_context_tokens(std::slice::from_ref(&user_blocks)),
        1
    );

    let assistant = Message::Assistant(faux_assistant_message(
        vec![
            AssistantContent::Text(TextContent::new("a")),
            AssistantContent::Thinking(ThinkingContent::new("b")),
        ],
        Default::default(),
    ));
    assert_eq!(estimate_context_tokens(std::slice::from_ref(&assistant)), 1);

    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: "call-1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![ToolResultContent::text("a"), ToolResultContent::text("b")],
        details: None,
        usage: None,
        is_error: false,
        added_tool_names: None,
        timestamp: 0,
    });
    assert_eq!(
        estimate_context_tokens(std::slice::from_ref(&tool_result)),
        1
    );
    // Messages are still rounded independently of each other.
    assert_eq!(
        estimate_context_tokens(&[user_blocks, assistant, tool_result]),
        3
    );

    // Image chars (4800) join the same per-message sum before the ceil:
    // ceil((3 + 4800) / 4) = 1201, not 3 + 1200 = 1203.
    let user_with_image = Message::User(UserMessage {
        content: UserContentValue::Blocks(vec![
            UserContent::Text(TextContent::new("a")),
            UserContent::Text(TextContent::new("b")),
            UserContent::Text(TextContent::new("c")),
            UserContent::Image(ImageContent {
                data: "zz".to_owned(),
                mime_type: "image/png".to_owned(),
            }),
        ]),
        timestamp: 0,
    });
    assert_eq!(estimate_context_tokens(&[user_with_image]), 1201);

    // bashExecution: command.length + output.length under one ceil.
    let bash = SessionMessage::BashExecution(BashExecutionMessage::new("a", "b", 0));
    assert_eq!(estimate_session_context_tokens(&[bash]), 1);
}

// pi skills.ts uses the npm `ignore` package: `**` spans path components and
// `[...]` character classes match per gitignore rules.
#[test]
fn skill_ignore_patterns_support_globstar() {
    let root = temp_dir("skills-globstar");
    let skills_dir = root.join("skills");
    for dir in [
        "x/generated",
        "generated",
        "vendor/tool",
        "a/x/b",
        "a/b",
        "a/x/keep",
        "keep/generated-not",
    ] {
        fs::create_dir_all(skills_dir.join(dir)).expect("skill dir");
    }
    fs::write(
        skills_dir.join(".gitignore"),
        "**/generated/\nvendor/**\na/**/b/\n",
    )
    .expect("gitignore");
    let skill_dirs = [
        ("x/generated", "x-generated"),
        ("generated", "root-generated"),
        ("vendor/tool", "vendor-tool"),
        ("a/x/b", "a-x-b"),
        ("a/b", "a-b"),
        ("a/x/keep", "a-x-keep"),
        ("keep/generated-not", "keep-generated-not"),
    ];
    for (dir, description) in skill_dirs {
        fs::write(
            skills_dir.join(dir).join("SKILL.md"),
            format!("---\ndescription: {description}\n---\nBody"),
        )
        .expect("skill file");
    }
    // A SKILL.md directly inside a `/**`-ignored directory is also ignored.
    fs::write(
        skills_dir.join("vendor/SKILL.md"),
        "---\ndescription: vendor-root\n---\nBody",
    )
    .expect("vendor skill");

    let (skills, diagnostics) = load_skills([skills_dir]);
    assert!(diagnostics.is_empty());
    let mut descriptions = skills
        .iter()
        .map(|skill| skill.description.as_str())
        .collect::<Vec<_>>();
    descriptions.sort_unstable();
    assert_eq!(descriptions, vec!["a-x-keep", "keep-generated-not"]);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn skill_ignore_patterns_support_character_classes() {
    let root = temp_dir("skills-classes");
    let skills_dir = root.join("skills");
    let skill_dirs = [
        ("skill-1", "skill-1", true),
        ("skill-x", "skill-x", false),
        ("temp-a1", "temp-a1", true),
        ("temp-k1", "temp-k1", false),
        ("blpha", "blpha", true),
        ("dlpha", "dlpha", false),
    ];
    for (dir, _, _) in skill_dirs {
        fs::create_dir_all(skills_dir.join(dir)).expect("skill dir");
    }
    fs::write(
        skills_dir.join(".gitignore"),
        "skill-[0-9]/\ntemp-[!k]*/\n[a-c]lpha/\n",
    )
    .expect("gitignore");
    for (dir, description, _) in skill_dirs {
        fs::write(
            skills_dir.join(dir).join("SKILL.md"),
            format!("---\ndescription: {description}\n---\nBody"),
        )
        .expect("skill file");
    }

    let (skills, diagnostics) = load_skills([skills_dir]);
    assert!(diagnostics.is_empty());
    let mut descriptions = skills
        .iter()
        .map(|skill| skill.description.as_str())
        .collect::<Vec<_>>();
    descriptions.sort_unstable();
    let mut expected = skill_dirs
        .iter()
        .filter(|(_, _, ignored)| !ignored)
        .map(|(_, description, _)| *description)
        .collect::<Vec<_>>();
    expected.sort_unstable();
    assert_eq!(descriptions, expected);
    let _ = fs::remove_dir_all(&root);
}

/// longbridge/ri#7: the returned event log must not grow with streamed output.
/// Per-chunk progress events go to the sink; the log keeps the lifecycle only.
#[tokio::test]
async fn returned_event_log_omits_per_chunk_progress_events() {
    use ri_agent_core::*;
    use std::sync::{Arc, Mutex};

    struct Sink {
        events: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait::async_trait]
    impl AgentEventSink for Sink {
        async fn on_event(&self, event: &AgentEvent) {
            let name = match event {
                AgentEvent::MessageUpdate { .. } => "message_update",
                AgentEvent::ToolExecutionUpdate { .. } => "tool_execution_update",
                AgentEvent::MessageEnd { .. } => "message_end",
                _ => return,
            };
            self.events.lock().expect("events").push(name.to_owned());
        }
    }

    let registration = register_faux_provider(RegisterFauxProviderOptions::default());
    registration.set_responses(vec![
        faux_assistant_message("streamed reply", Default::default()).into(),
    ]);

    let sink_events = Arc::new(Mutex::new(Vec::new()));
    let mut config = AgentLoopConfig::new(registration.get_model());
    config.event_sink = Some(Arc::new(Sink {
        events: sink_events.clone(),
    }));
    let context = AgentContext {
        system_prompt: String::new(),
        messages: Vec::new(),
        tools: Vec::new(),
    };

    let (_messages, events) = agent_loop_prompt(context, "hello", config)
        .await
        .expect("loop");

    assert!(
        !events
            .iter()
            .any(|event| matches!(event, AgentEvent::MessageUpdate { .. })),
        "MessageUpdate must never be retained: each one owns two copies of the \
         partial message, so the log would grow quadratically with output"
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, AgentEvent::MessageEnd { .. })),
        "the final message content is still represented in the log"
    );
    // The sink saw the streamed deltas the log dropped.
    let observed = sink_events.lock().expect("events").clone();
    assert!(
        observed.contains(&"message_update".to_owned()),
        "{observed:?}"
    );
    assert!(observed.contains(&"message_end".to_owned()), "{observed:?}");
    registration.unregister();
}
