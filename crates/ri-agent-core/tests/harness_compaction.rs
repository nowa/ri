use ri_agent_core::*;
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, Mutex},
};

fn usage(input: u64, output: u64, cache_read: u64, cache_write: u64) -> Usage {
    Usage {
        input,
        output,
        cache_read,
        cache_write,
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

fn bash_execution_entry(
    id: &str,
    parent_id: Option<&str>,
    command: &str,
    output: &str,
) -> SessionTreeEntry {
    let mut message = BashExecutionMessage::new(command, output, 1_700_000_000_000);
    message.exit_code = Some(0);
    SessionTreeEntry::Message {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        message: message.into(),
    }
}

fn compaction_entry(
    id: &str,
    parent_id: Option<&str>,
    summary: &str,
    first_kept_entry_id: &str,
    details: Option<Value>,
) -> SessionTreeEntry {
    SessionTreeEntry::Compaction {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        summary: summary.to_owned(),
        first_kept_entry_id: first_kept_entry_id.to_owned(),
        tokens_before: 1234,
        details,
        from_hook: None,
    }
}

fn branch_summary_entry(id: &str, parent_id: Option<&str>) -> SessionTreeEntry {
    SessionTreeEntry::BranchSummary {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        from_id: "branch".to_owned(),
        summary: "branch summary".to_owned(),
        details: None,
        from_hook: None,
    }
}

fn branch_summary_entry_with_details(
    id: &str,
    parent_id: Option<&str>,
    details: Value,
) -> SessionTreeEntry {
    SessionTreeEntry::BranchSummary {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        from_id: "branch".to_owned(),
        summary: "branch summary".to_owned(),
        details: Some(details),
        from_hook: None,
    }
}

fn custom_message_entry(id: &str, parent_id: Option<&str>) -> SessionTreeEntry {
    SessionTreeEntry::CustomMessage {
        id: id.to_owned(),
        parent_id: parent_id.map(str::to_owned),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        custom_type: "note".to_owned(),
        content: custom_text_content("custom content"),
        display: true,
        details: None,
    }
}

fn assistant_tool_call(name: &str, path: &str) -> Message {
    let mut arguments = Map::new();
    arguments.insert("path".to_owned(), Value::String(path.to_owned()));
    Message::Assistant(faux_assistant_message(
        AssistantContent::ToolCall(ToolCall {
            id: format!("call-{name}"),
            name: name.to_owned(),
            arguments,
            thought_signature: None,
        }),
        Default::default(),
    ))
}

fn session_user(text: &str) -> SessionMessage {
    SessionMessage::Llm {
        message: user_message_text(text),
    }
}

fn first_user_text(context: &Context) -> String {
    let Some(Message::User(user)) = context.messages.first() else {
        return String::new();
    };
    match &user.content {
        UserContentValue::Plain(text) => text.clone(),
        UserContentValue::Blocks(blocks) => blocks
            .iter()
            .filter_map(|block| match block {
                UserContent::Text(text) => Some(text.text.as_str()),
                UserContent::Image(_) => None,
            })
            .collect(),
    }
}

fn register_faux_model(reasoning: bool, max_tokens: u64) -> FauxProviderRegistration {
    let mut definition = FauxModelDefinition::new(if reasoning {
        "reasoning-model"
    } else {
        "plain-model"
    });
    definition.reasoning = reasoning;
    definition.max_tokens = max_tokens;
    register_faux_provider(RegisterFauxProviderOptions {
        models: vec![definition],
        ..Default::default()
    })
}

#[test]
fn compaction_calculates_context_tokens_from_usage_and_thresholds() {
    assert_eq!(calculate_context_tokens(&usage(1000, 500, 200, 100)), 1800);
    assert_eq!(calculate_context_tokens(&usage(0, 0, 0, 0)), 0);

    let settings = CompactionThresholdSettings {
        enabled: true,
        reserve_tokens: 10_000,
        keep_recent_tokens: 20_000,
    };
    assert!(should_compact_tokens(95_000, 100_000, &settings));
    assert!(!should_compact_tokens(89_000, 100_000, &settings));
    assert!(!should_compact_tokens(
        95_000,
        100_000,
        &CompactionThresholdSettings {
            enabled: false,
            ..settings
        }
    ));
    assert!(!should_compact_tokens(90_000, 100_000, &settings));
}

#[test]
fn compaction_estimates_tokens_and_uses_latest_valid_assistant_usage() {
    let mut assistant_with_blocks = faux_assistant_message(
        vec![
            AssistantContent::Thinking(ThinkingContent::new("thinking")),
            AssistantContent::ToolCall(ToolCall {
                id: "call-1".to_owned(),
                name: "read".to_owned(),
                arguments: Map::new(),
                thought_signature: None,
            }),
        ],
        Default::default(),
    );
    assistant_with_blocks.usage = usage(10, 5, 3, 2);

    let tool_result_with_image = Message::ToolResult(ToolResultMessage {
        tool_call_id: "call-1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![
            ToolResultContent::text("tool text"),
            ToolResultContent::Image(ImageContent {
                data: "abc".repeat(2000),
                mime_type: "image/png".to_owned(),
            }),
        ],
        details: None,
        is_error: false,
        timestamp: 0,
    });

    assert!(estimate_context_tokens(&[Message::User(UserMessage::text("plain user"))]) > 0);
    assert!(estimate_context_tokens(&[Message::Assistant(assistant_with_blocks.clone())]) > 0);
    assert!(estimate_context_tokens(&[tool_result_with_image]) > 1000);

    let messages = vec![
        Message::Assistant(assistant_with_blocks),
        Message::User(UserMessage::text("tail")),
    ];
    assert_eq!(
        get_last_assistant_usage(&messages).map(|(index, usage)| (index, usage.total_tokens)),
        Some((0, 20))
    );
    assert_eq!(
        estimate_context_token_usage(&messages),
        ContextTokenEstimate {
            tokens: 21,
            usage_tokens: 20,
            last_usage_index: Some(0),
        }
    );

    let invalid = vec![
        assistant_with_usage("aborted", usage(1, 1, 0, 0), StopReason::Aborted),
        assistant_with_usage("error", usage(2, 2, 0, 0), StopReason::Error),
    ];
    assert!(get_last_assistant_usage(&invalid).is_none());
    assert_eq!(
        estimate_context_token_usage(&[Message::User(UserMessage::text("no usage"))])
            .last_usage_index,
        None
    );
}

#[test]
fn compaction_treats_bash_execution_as_user_visible_context() {
    let mut bash_message = BashExecutionMessage::new("cargo test", "ok", 1_700_000_000_000);
    bash_message.exit_code = Some(0);
    let session_messages = vec![SessionMessage::BashExecution(bash_message.clone())];
    assert!(estimate_session_context_tokens(&session_messages) > 0);

    let llm_messages = convert_session_messages_to_llm(&session_messages);
    assert_eq!(llm_messages.len(), 1);
    let text = first_user_text(&Context {
        system_prompt: None,
        messages: llm_messages,
        tools: Vec::new(),
    });
    assert_eq!(text, bash_execution_to_text(&bash_message));

    let bash = bash_execution_entry("bash", None, "cargo test", "ok");
    let assistant = message_entry("assistant", Some("bash"), assistant_message_text("done"));
    let entries = vec![bash, assistant];
    assert_eq!(find_entry_turn_start_index(&entries, 1, 0), Some(0));
    let cut = find_cut_point(&entries, 0, entries.len(), 1);
    assert_eq!(cut.first_kept_entry_index, 1);
    assert_eq!(cut.turn_start_index, Some(0));
    assert!(cut.is_split_turn);
}

#[test]
fn compaction_finds_cut_points_and_turn_start_edges() {
    let thinking = SessionTreeEntry::ThinkingLevelChange {
        id: "thinking".to_owned(),
        parent_id: None,
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        thinking_level: "high".to_owned(),
    };
    let model_change = SessionTreeEntry::ModelChange {
        id: "model".to_owned(),
        parent_id: Some("thinking".to_owned()),
        timestamp: "2026-01-01T00:00:00.000Z".to_owned(),
        provider: "openai".to_owned(),
        model_id: "gpt-4".to_owned(),
    };
    assert_eq!(
        find_cut_point(&[thinking.clone(), model_change.clone()], 0, 2, 1),
        CutPointResult {
            first_kept_entry_index: 0,
            turn_start_index: None,
            is_split_turn: false,
        }
    );

    let branch = branch_summary_entry("branch", Some("model"));
    let custom = custom_message_entry("custom", Some("branch"));
    assert_eq!(
        find_entry_turn_start_index(&[thinking.clone(), branch.clone()], 1, 0),
        Some(1)
    );
    assert_eq!(
        find_entry_turn_start_index(&[thinking.clone(), custom.clone()], 1, 0),
        Some(1)
    );
    assert_eq!(
        find_entry_turn_start_index(&[thinking.clone(), model_change], 1, 0),
        None
    );
    assert_eq!(
        find_cut_point(&[thinking, branch, custom], 0, 3, 1).first_kept_entry_index,
        0
    );

    let tool_result = message_entry(
        "tool-result",
        None,
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call-1".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultContent::text("tool output")],
            details: None,
            is_error: false,
            timestamp: 0,
        }),
    );
    assert_eq!(
        find_cut_point(&[tool_result], 0, 1, 1),
        CutPointResult {
            first_kept_entry_index: 0,
            turn_start_index: None,
            is_split_turn: false,
        }
    );

    let user = message_entry("user", None, user_message_text("user"));
    let compaction = compaction_entry("compact", Some("user"), "summary", "user", None);
    let assistant = message_entry(
        "assistant",
        Some("compact"),
        assistant_message_text("assistant"),
    );
    assert_eq!(
        find_cut_point(&[user, compaction, assistant], 0, 3, 1).first_kept_entry_index,
        2
    );
}

#[test]
fn compaction_prepares_entries_previous_summary_split_turn_and_file_ops() {
    let u1 = message_entry("u1", None, user_message_text("user msg 1"));
    let a1 = message_entry("a1", Some("u1"), assistant_tool_call("write", "written.ts"));
    let compaction = compaction_entry(
        "compact",
        Some("a1"),
        "First summary",
        "u1",
        Some(json!({
            "readFiles": ["old-read.ts"],
            "modifiedFiles": ["old-edit.ts"]
        })),
    );
    let u2 = message_entry("u2", Some("compact"), user_message_text("large turn"));
    let a2 = message_entry(
        "a2",
        Some("u2"),
        assistant_with_usage(
            "large assistant message",
            usage(50, 10, 0, 0),
            StopReason::Stop,
        ),
    );

    let preparation = prepare_compaction(
        &[u1, a1, compaction, u2, a2],
        CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 1,
        },
    )
    .expect("prepare")
    .expect("compaction");

    assert_eq!(
        preparation.previous_summary.as_deref(),
        Some("First summary")
    );
    assert!(preparation.is_split_turn);
    assert_eq!(
        preparation
            .turn_prefix_messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["user"]
    );
    assert!(preparation.file_ops.read.contains("old-read.ts"));
    assert!(preparation.file_ops.edited.contains("old-edit.ts"));
    assert!(preparation.file_ops.written.contains("written.ts"));

    let lists = compute_file_lists(&preparation.file_ops);
    assert_eq!(lists.read_files, vec!["old-read.ts"]);
    assert_eq!(lists.modified_files, vec!["old-edit.ts", "written.ts"]);
    assert!(
        format_file_operations(&lists.read_files, &lists.modified_files).contains("<read-files>")
    );
}

#[test]
fn compaction_prepares_custom_branch_messages_and_serializes_tool_results() {
    let branch = branch_summary_entry("branch", None);
    let custom = custom_message_entry("custom", Some("branch"));
    let user = message_entry("user", Some("custom"), user_message_text("keep"));
    let assistant = message_entry(
        "assistant",
        Some("user"),
        assistant_message_text("assistant"),
    );

    let preparation = prepare_compaction(
        &[branch, custom, user, assistant],
        CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 100,
            keep_recent_tokens: 1,
        },
    )
    .expect("prepare")
    .expect("compaction");

    assert_eq!(
        preparation
            .messages_to_summarize
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["branchSummary", "custom"]
    );

    let long_content = "x".repeat(5000);
    let serialized = serialize_conversation(&[Message::ToolResult(ToolResultMessage {
        tool_call_id: "tc1".to_owned(),
        tool_name: "read".to_owned(),
        content: vec![ToolResultContent::text(long_content)],
        details: None,
        is_error: false,
        timestamp: 0,
    })]);
    assert!(serialized.contains("[Tool result]:"));
    assert!(serialized.contains("[... 3000 more characters truncated]"));

    assert!(
        prepare_compaction(
            &[compaction_entry(
                "compact",
                None,
                "already compacted",
                "entry-keep",
                None
            )],
            CompactionThresholdSettings::default()
        )
        .expect("prepare")
        .is_none()
    );
    assert!(
        prepare_compaction(&[], CompactionThresholdSettings::default())
            .expect("prepare")
            .is_none()
    );
}

#[tokio::test]
async fn compaction_generate_summary_builds_prompt_and_passes_reasoning_options() {
    let registration = register_faux_model(true, 8_192);
    let model = registration.get_model();
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_prompts = Arc::new(Mutex::new(Vec::<String>::new()));
    let options_ref = seen_options.clone();
    let prompts_ref = seen_prompts.clone();
    registration.set_responses(vec![faux_response_factory(
        move |context, options, _, _| {
            options_ref.lock().expect("options").push(options.clone());
            prompts_ref
                .lock()
                .expect("prompts")
                .push(first_user_text(context));
            faux_assistant_message("## Goal\nTest summary", Default::default())
        },
    )]);

    let mut headers = BTreeMap::new();
    headers.insert("x-test".to_owned(), "yes".to_owned());
    let summary = generate_summary(
        &[session_user("Summarize this.")],
        &model,
        2_000,
        "test-key",
        Some(headers),
        Some("focus"),
        Some("old summary"),
        Some(ThinkingLevel::Medium),
    )
    .await
    .expect("summary");

    assert!(summary.contains("Test summary"));
    let options = seen_options.lock().expect("options");
    assert_eq!(options[0].reasoning, Some(ThinkingLevel::Medium));
    assert_eq!(options[0].stream.max_tokens, Some(1_600));
    assert_eq!(options[0].stream.api_key.as_deref(), Some("test-key"));
    assert_eq!(
        options[0].stream.headers.get("x-test").map(String::as_str),
        Some("yes")
    );
    let prompts = seen_prompts.lock().expect("prompts");
    assert!(prompts[0].contains("<previous-summary>\nold summary\n</previous-summary>"));
    assert!(prompts[0].contains("Additional focus: focus"));
    registration.unregister();

    let off_registration = register_faux_model(true, 8_192);
    let off_seen = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let off_seen_ref = off_seen.clone();
    off_registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        off_seen_ref.lock().expect("options").push(options.clone());
        faux_assistant_message("off", Default::default())
    })]);
    generate_summary(
        &[session_user("Summarize this.")],
        &off_registration.get_model(),
        2_000,
        "test-key",
        None,
        None,
        None,
        Some(ThinkingLevel::Off),
    )
    .await
    .expect("off");
    assert_eq!(off_seen.lock().expect("options")[0].reasoning, None);
    off_registration.unregister();

    let plain_registration = register_faux_model(false, 8_192);
    let plain_seen = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let plain_seen_ref = plain_seen.clone();
    plain_registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        plain_seen_ref
            .lock()
            .expect("options")
            .push(options.clone());
        faux_assistant_message("plain", Default::default())
    })]);
    generate_summary(
        &[session_user("Summarize this.")],
        &plain_registration.get_model(),
        2_000,
        "test-key",
        None,
        None,
        None,
        Some(ThinkingLevel::High),
    )
    .await
    .expect("plain");
    assert_eq!(plain_seen.lock().expect("options")[0].reasoning, None);
    plain_registration.unregister();
}

#[tokio::test]
async fn compaction_generate_summary_maps_error_and_aborted_results() {
    let error_registration = register_faux_model(false, 8_192);
    error_registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("boom".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let error = generate_summary(
        &[session_user("Summarize this.")],
        &error_registration.get_model(),
        2_000,
        "test-key",
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("error");
    assert_eq!(error.code, CompactionErrorCode::SummarizationFailed);
    assert_eq!(error.message, "Summarization failed: boom");
    error_registration.unregister();

    let aborted_registration = register_faux_model(false, 8_192);
    aborted_registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("stopped".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let aborted = generate_summary(
        &[session_user("Summarize this.")],
        &aborted_registration.get_model(),
        2_000,
        "test-key",
        None,
        None,
        None,
        None,
    )
    .await
    .expect_err("aborted");
    assert_eq!(aborted.code, CompactionErrorCode::Aborted);
    assert_eq!(aborted.message, "stopped");
    aborted_registration.unregister();
}

#[tokio::test]
async fn compaction_compact_returns_summary_details_and_clamps_max_tokens() {
    let registration = register_faux_model(false, 128_000);
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_ref = seen_options.clone();
    registration.set_responses(vec![faux_response_factory(move |_, options, _, _| {
        seen_ref.lock().expect("options").push(options.clone());
        faux_assistant_message("## Goal\nTest summary", Default::default())
    })]);
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_owned(),
        messages_to_summarize: vec![session_user("read a file")],
        turn_prefix_messages: Vec::new(),
        is_split_turn: false,
        tokens_before: 600_000,
        previous_summary: None,
        file_ops: FileOperations {
            read: BTreeSet::from(["src/index.ts".to_owned(), "src/lib.ts".to_owned()]),
            written: BTreeSet::from(["src/index.ts".to_owned()]),
            edited: BTreeSet::new(),
        },
        settings: CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 500_000,
            keep_recent_tokens: 20_000,
        },
    };

    let result = compact(
        &preparation,
        &registration.get_model(),
        "test-key",
        None,
        None,
        None,
    )
    .await
    .expect("compact");

    assert!(result.summary.contains("Test summary"));
    assert!(
        result
            .summary
            .contains("<read-files>\nsrc/lib.ts\n</read-files>")
    );
    assert!(
        result
            .summary
            .contains("<modified-files>\nsrc/index.ts\n</modified-files>")
    );
    assert_eq!(result.first_kept_entry_id, "entry-keep");
    assert_eq!(result.tokens_before, 600_000);
    assert_eq!(result.details.read_files, vec!["src/lib.ts"]);
    assert_eq!(result.details.modified_files, vec!["src/index.ts"]);
    assert_eq!(
        seen_options.lock().expect("options")[0].stream.max_tokens,
        Some(128_000)
    );
    registration.unregister();
}

#[tokio::test]
async fn compaction_compact_maps_history_summary_errors_without_throwing() {
    let registration = register_faux_model(false, 8_192);
    registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("history failed".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_owned(),
        messages_to_summarize: vec![session_user("Summarize this.")],
        turn_prefix_messages: Vec::new(),
        is_split_turn: false,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 2_000,
            keep_recent_tokens: 20,
        },
    };

    let error = compact(
        &preparation,
        &registration.get_model(),
        "test-key",
        None,
        None,
        None,
    )
    .await
    .expect_err("history summary error");

    assert_eq!(error.code, CompactionErrorCode::SummarizationFailed);
    assert_eq!(error.message, "Summarization failed: history failed");
    registration.unregister();

    let invalid_registration = register_faux_model(false, 8_192);
    let invalid = compact(
        &CompactionPreparation {
            first_kept_entry_id: String::new(),
            messages_to_summarize: Vec::new(),
            ..preparation
        },
        &invalid_registration.get_model(),
        "test-key",
        None,
        None,
        None,
    )
    .await
    .expect_err("invalid compaction");
    assert_eq!(invalid.code, CompactionErrorCode::InvalidSession);
    invalid_registration.unregister();
}

#[tokio::test]
async fn compaction_compact_summarizes_split_turn_and_maps_prefix_errors() {
    let registration = register_faux_model(true, 8_192);
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_ref = seen_options.clone();
    registration.set_responses(vec![faux_response_factory(
        move |context, options, _, _| {
            seen_ref.lock().expect("options").push(options.clone());
            assert!(first_user_text(context).contains("PREFIX of a turn"));
            faux_assistant_message("## Original Request\nPrefix summary", Default::default())
        },
    )]);
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_owned(),
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: vec![session_user("large turn prefix")],
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 2_000,
            keep_recent_tokens: 20,
        },
    };

    let result = compact(
        &preparation,
        &registration.get_model(),
        "test-key",
        None,
        None,
        Some(ThinkingLevel::High),
    )
    .await
    .expect("split compact");
    assert!(result.summary.contains("No prior history."));
    assert!(result.summary.contains("**Turn Context (split turn):**"));
    assert!(result.summary.contains("Prefix summary"));
    let options = seen_options.lock().expect("options");
    assert_eq!(options[0].reasoning, Some(ThinkingLevel::High));
    assert_eq!(options[0].stream.max_tokens, Some(1_000));
    registration.unregister();

    let error_registration = register_faux_model(false, 8_192);
    error_registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("prefix failed".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let error = compact(
        &preparation,
        &error_registration.get_model(),
        "test-key",
        None,
        None,
        None,
    )
    .await
    .expect_err("prefix error");
    assert_eq!(error.code, CompactionErrorCode::SummarizationFailed);
    assert_eq!(
        error.message,
        "Turn prefix summarization failed: prefix failed"
    );
    error_registration.unregister();

    let invalid_registration = register_faux_model(false, 8_192);
    let invalid = compact(
        &CompactionPreparation {
            first_kept_entry_id: String::new(),
            ..preparation
        },
        &invalid_registration.get_model(),
        "test-key",
        None,
        None,
        None,
    )
    .await
    .expect_err("invalid");
    assert_eq!(invalid.code, CompactionErrorCode::InvalidSession);
    invalid_registration.unregister();
}

#[tokio::test]
async fn compaction_compact_maps_aborted_turn_prefix_summary() {
    let registration = register_faux_model(false, 8_192);
    registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("prefix stopped".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let preparation = CompactionPreparation {
        first_kept_entry_id: "entry-keep".to_owned(),
        messages_to_summarize: Vec::new(),
        turn_prefix_messages: vec![session_user("large turn prefix")],
        is_split_turn: true,
        tokens_before: 100,
        previous_summary: None,
        file_ops: FileOperations::default(),
        settings: CompactionThresholdSettings {
            enabled: true,
            reserve_tokens: 2_000,
            keep_recent_tokens: 20,
        },
    };

    let error = compact(
        &preparation,
        &registration.get_model(),
        "test-key",
        None,
        None,
        None,
    )
    .await
    .expect_err("prefix aborted");

    assert_eq!(error.code, CompactionErrorCode::Aborted);
    assert_eq!(error.message, "prefix stopped");
    registration.unregister();
}

#[test]
fn branch_summary_collects_abandoned_branch_entries() {
    let mut session = Session::new(InMemorySessionStorage::new());
    let root = session
        .append_message(user_message_text("root"))
        .expect("root");
    let main_assistant = session
        .append_message(assistant_message_text("main"))
        .expect("main");
    let old_user = session
        .append_message(user_message_text("old branch"))
        .expect("old user");
    let old_leaf = session
        .append_message(assistant_message_text("old result"))
        .expect("old leaf");

    session
        .move_to(Some(root.clone()), None)
        .expect("move root");
    let target = session
        .append_message(assistant_message_text("target branch"))
        .expect("target");

    let collected =
        collect_entries_for_branch_summary(&session, Some(&old_leaf), &target).expect("collect");
    assert_eq!(collected.common_ancestor_id.as_deref(), Some(root.as_str()));
    assert_eq!(
        collected
            .entries
            .iter()
            .map(SessionTreeEntry::id)
            .collect::<Vec<_>>(),
        vec![
            main_assistant.as_str(),
            old_user.as_str(),
            old_leaf.as_str()
        ]
    );

    let empty = collect_entries_for_branch_summary(&session, None, &target).expect("empty");
    assert!(empty.entries.is_empty());
    assert_eq!(empty.common_ancestor_id, None);

    let missing = collect_entries_for_branch_summary(&session, Some("missing"), &target)
        .expect_err("missing");
    assert_eq!(missing.code, BranchSummaryErrorCode::InvalidSession);
}

#[test]
fn branch_summary_prepares_messages_budget_and_file_ops() {
    let branch = branch_summary_entry_with_details(
        "branch",
        None,
        json!({
            "readFiles": ["old-read.ts"],
            "modifiedFiles": ["old-edit.ts"]
        }),
    );
    let user = message_entry(
        "user",
        Some("branch"),
        user_message_text("keep this request"),
    );
    let read = message_entry(
        "read",
        Some("user"),
        assistant_tool_call("read", "src/read.ts"),
    );
    let write = message_entry(
        "write",
        Some("read"),
        assistant_tool_call("write", "src/write.ts"),
    );
    let tool_result = message_entry(
        "tool-result",
        Some("write"),
        Message::ToolResult(ToolResultMessage {
            tool_call_id: "call-read".to_owned(),
            tool_name: "read".to_owned(),
            content: vec![ToolResultContent::text("tool output")],
            details: None,
            is_error: false,
            timestamp: 0,
        }),
    );

    let preparation = prepare_branch_entries(
        &[
            branch.clone(),
            user.clone(),
            read.clone(),
            write.clone(),
            tool_result,
        ],
        0,
    );
    assert_eq!(
        preparation
            .messages
            .iter()
            .map(SessionMessage::role)
            .collect::<Vec<_>>(),
        vec!["branchSummary", "user", "assistant", "assistant"]
    );
    let lists = compute_file_lists(&preparation.file_ops);
    assert_eq!(lists.read_files, vec!["old-read.ts", "src/read.ts"]);
    assert_eq!(lists.modified_files, vec!["old-edit.ts", "src/write.ts"]);

    let budgeted = prepare_branch_entries(&[branch, user, read, write], 1);
    assert!(budgeted.messages.len() < 4);
}

#[tokio::test]
async fn branch_summary_generate_builds_prompt_options_and_file_details() {
    let registration = register_faux_model(true, 8_192);
    let seen_options = Arc::new(Mutex::new(Vec::<SimpleStreamOptions>::new()));
    let seen_prompts = Arc::new(Mutex::new(Vec::<String>::new()));
    let options_ref = seen_options.clone();
    let prompts_ref = seen_prompts.clone();
    registration.set_responses(vec![faux_response_factory(
        move |context, options, _, _| {
            options_ref.lock().expect("options").push(options.clone());
            prompts_ref
                .lock()
                .expect("prompts")
                .push(first_user_text(context));
            faux_assistant_message("## Goal\nBranch summary", Default::default())
        },
    )]);

    let mut headers = BTreeMap::new();
    headers.insert("x-branch".to_owned(), "yes".to_owned());
    let result = generate_branch_summary(
        &[message_entry(
            "assistant",
            None,
            assistant_tool_call("read", "src/read.ts"),
        )],
        &registration.get_model(),
        "branch-key",
        Some(headers),
        Some("only decisions"),
        false,
        Some(16_384),
    )
    .await
    .expect("branch summary");

    assert!(
        result.summary.starts_with(
            "The user explored a different conversation branch before returning here."
        )
    );
    assert!(result.summary.contains("Branch summary"));
    assert!(
        result
            .summary
            .contains("<read-files>\nsrc/read.ts\n</read-files>")
    );
    assert_eq!(result.read_files, vec!["src/read.ts"]);
    assert!(result.modified_files.is_empty());
    let options = seen_options.lock().expect("options");
    assert_eq!(options[0].stream.max_tokens, Some(2_048));
    assert_eq!(options[0].stream.api_key.as_deref(), Some("branch-key"));
    assert_eq!(options[0].reasoning, None);
    assert_eq!(
        options[0]
            .stream
            .headers
            .get("x-branch")
            .map(String::as_str),
        Some("yes")
    );
    let prompts = seen_prompts.lock().expect("prompts");
    assert!(prompts[0].contains("Additional focus: only decisions"));
    assert!(prompts[0].contains("[Assistant tool calls]: read(path=\"src/read.ts\")"));
    registration.unregister();

    let no_content_registration = register_faux_model(false, 8_192);
    let no_content = generate_branch_summary(
        &[],
        &no_content_registration.get_model(),
        "branch-key",
        None,
        None,
        false,
        None,
    )
    .await
    .expect("no content");
    assert_eq!(no_content.summary, "No content to summarize");
    no_content_registration.unregister();
}

#[tokio::test]
async fn branch_summary_generate_replaces_prompt_and_maps_errors() {
    let replace_registration = register_faux_model(false, 8_192);
    let prompt_ref = Arc::new(Mutex::new(String::new()));
    let prompt_sink = prompt_ref.clone();
    replace_registration.set_responses(vec![faux_response_factory(move |context, _, _, _| {
        *prompt_sink.lock().expect("prompt") = first_user_text(context);
        faux_assistant_message("replacement", Default::default())
    })]);
    generate_branch_summary(
        &[message_entry(
            "user",
            None,
            user_message_text("branch work"),
        )],
        &replace_registration.get_model(),
        "branch-key",
        None,
        Some("Use this exact prompt"),
        true,
        None,
    )
    .await
    .expect("replace");
    let prompt = prompt_ref.lock().expect("prompt");
    assert!(prompt.contains("Use this exact prompt"));
    assert!(!prompt.contains("Create a structured summary of this conversation branch"));
    replace_registration.unregister();

    let error_registration = register_faux_model(false, 8_192);
    error_registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Error),
                error_message: Some("branch failed".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let error = generate_branch_summary(
        &[message_entry(
            "user",
            None,
            user_message_text("branch work"),
        )],
        &error_registration.get_model(),
        "branch-key",
        None,
        None,
        false,
        None,
    )
    .await
    .expect_err("error");
    assert_eq!(error.code, BranchSummaryErrorCode::SummarizationFailed);
    assert_eq!(error.message, "Branch summary failed: branch failed");
    error_registration.unregister();

    let aborted_registration = register_faux_model(false, 8_192);
    aborted_registration.set_responses(vec![
        faux_assistant_message(
            "",
            FauxAssistantOptions {
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("branch stopped".to_owned()),
                ..Default::default()
            },
        )
        .into(),
    ]);
    let aborted = generate_branch_summary(
        &[message_entry(
            "user",
            None,
            user_message_text("branch work"),
        )],
        &aborted_registration.get_model(),
        "branch-key",
        None,
        None,
        false,
        None,
    )
    .await
    .expect_err("aborted");
    assert_eq!(aborted.code, BranchSummaryErrorCode::Aborted);
    assert_eq!(aborted.message, "branch stopped");
    aborted_registration.unregister();
}
