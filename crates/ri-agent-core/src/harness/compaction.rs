use super::session::{
    CustomMessageContent, Session, SessionError, SessionMessage, SessionTreeEntry,
    build_session_context,
};
use ri_llm_provider::{
    AssistantContent, Context, Message, Model, SimpleStreamOptions, StopReason, StreamOptions,
    TextContent, ThinkingLevel, ToolResultContent, Usage, UserContent, UserContentValue,
    complete_simple,
};
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt::{Display, Formatter},
};

pub const COMPACTION_SUMMARY_PREFIX: &str = "The conversation history before this point was compacted into the following summary:\n\n<summary>\n";
pub const COMPACTION_SUMMARY_SUFFIX: &str = "\n</summary>";
pub const BRANCH_SUMMARY_PREFIX: &str =
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n";
pub const BRANCH_SUMMARY_SUFFIX: &str = "</summary>";
pub const SUMMARIZATION_SYSTEM_PROMPT: &str = "You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary following the exact format specified.\n\nDo NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.";

const SUMMARIZATION_PROMPT: &str = "The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.\n\nUse this EXACT format:\n\n## Goal\n[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned by user]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Current work]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [Ordered list of what should happen next]\n\n## Critical Context\n- [Any data, examples, or references needed to continue]\n- [Or \"(none)\" if not applicable]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const UPDATE_SUMMARIZATION_PROMPT: &str = "The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.\n\nUpdate the existing structured summary with new information. RULES:\n- PRESERVE all existing information from the previous summary\n- ADD new progress, decisions, and context from the new messages\n- UPDATE the Progress section: move items from \"In Progress\" to \"Done\" when completed\n- UPDATE \"Next Steps\" based on what was accomplished\n- PRESERVE exact file paths, function names, and error messages\n- If something is no longer relevant, you may remove it\n\nUse this EXACT format:\n\n## Goal\n[Preserve existing goals, add new ones if the task expanded]\n\n## Constraints & Preferences\n- [Preserve existing, add new ones discovered]\n\n## Progress\n### Done\n- [x] [Include previously done items AND newly completed items]\n\n### In Progress\n- [ ] [Current work - update based on progress]\n\n### Blocked\n- [Current blockers - remove if resolved]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale] (preserve all previous, add new)\n\n## Next Steps\n1. [Update based on current state]\n\n## Critical Context\n- [Preserve important context, add new if needed]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

const TURN_PREFIX_SUMMARIZATION_PROMPT: &str = "This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.\n\nSummarize the prefix to provide context for the retained suffix:\n\n## Original Request\n[What did the user ask for in this turn?]\n\n## Early Progress\n- [Key decisions and work done in the prefix]\n\n## Context for Suffix\n- [Information needed to understand the retained recent work]\n\nBe concise. Focus on what's needed to understand the kept suffix.";

const BRANCH_SUMMARY_PREAMBLE: &str = "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n";

const BRANCH_SUMMARY_PROMPT: &str = "Create a structured summary of this conversation branch for context when returning later.\n\nUse this EXACT format:\n\n## Goal\n[What was the user trying to accomplish in this branch?]\n\n## Constraints & Preferences\n- [Any constraints, preferences, or requirements mentioned]\n- [Or \"(none)\" if none were mentioned]\n\n## Progress\n### Done\n- [x] [Completed tasks/changes]\n\n### In Progress\n- [ ] [Work that was started but not finished]\n\n### Blocked\n- [Issues preventing progress, if any]\n\n## Key Decisions\n- **[Decision]**: [Brief rationale]\n\n## Next Steps\n1. [What should happen next to continue this work]\n\nKeep each section concise. Preserve exact file paths, function names, and error messages.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionSettings {
    pub max_context_tokens: u64,
    pub threshold_percent: u8,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            max_context_tokens: 128_000,
            threshold_percent: 80,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionThresholdSettings {
    pub enabled: bool,
    pub reserve_tokens: u64,
    pub keep_recent_tokens: u64,
}

impl Default for CompactionThresholdSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 20_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextTokenEstimate {
    pub tokens: u64,
    pub usage_tokens: u64,
    pub last_usage_index: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionErrorCode {
    Aborted,
    SummarizationFailed,
    InvalidSession,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionError {
    pub code: CompactionErrorCode,
    pub message: String,
}

impl CompactionError {
    pub fn new(code: CompactionErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl Display for CompactionError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for CompactionError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BranchSummaryErrorCode {
    Aborted,
    SummarizationFailed,
    InvalidSession,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSummaryError {
    pub code: BranchSummaryErrorCode,
    pub message: String,
}

impl BranchSummaryError {
    pub fn new(code: BranchSummaryErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

impl Display for BranchSummaryError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for BranchSummaryError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutPointResult {
    pub first_kept_entry_index: usize,
    pub turn_start_index: Option<usize>,
    pub is_split_turn: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FileOperations {
    pub read: BTreeSet<String>,
    pub written: BTreeSet<String>,
    pub edited: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileOperationLists {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompactionPreparation {
    pub first_kept_entry_id: String,
    pub messages_to_summarize: Vec<SessionMessage>,
    pub turn_prefix_messages: Vec<SessionMessage>,
    pub is_split_turn: bool,
    pub tokens_before: u64,
    pub previous_summary: Option<String>,
    pub file_ops: FileOperations,
    pub settings: CompactionThresholdSettings,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionDetails {
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompactionResult {
    pub summary: String,
    pub first_kept_entry_id: String,
    pub tokens_before: u64,
    pub details: CompactionDetails,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BranchPreparation {
    pub messages: Vec<SessionMessage>,
    pub file_ops: FileOperations,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CollectEntriesResult {
    pub entries: Vec<SessionTreeEntry>,
    pub common_ancestor_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BranchSummaryResult {
    pub summary: String,
    pub read_files: Vec<String>,
    pub modified_files: Vec<String>,
}

pub fn estimate_tokens(text: &str) -> u64 {
    text.chars().count().div_ceil(4) as u64
}

pub fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

pub fn estimate_context_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_message_tokens).sum()
}

pub fn estimate_context_token_usage(messages: &[Message]) -> ContextTokenEstimate {
    let Some((last_usage_index, usage)) = get_last_assistant_usage(messages) else {
        return ContextTokenEstimate {
            tokens: estimate_context_tokens(messages),
            usage_tokens: 0,
            last_usage_index: None,
        };
    };

    let usage_tokens = calculate_context_tokens(usage);
    let tail_tokens = messages
        .iter()
        .skip(last_usage_index + 1)
        .map(estimate_message_tokens)
        .sum::<u64>();
    ContextTokenEstimate {
        tokens: usage_tokens + tail_tokens,
        usage_tokens,
        last_usage_index: Some(last_usage_index),
    }
}

pub fn estimate_session_context_tokens(messages: &[SessionMessage]) -> u64 {
    messages.iter().map(estimate_session_message_tokens).sum()
}

pub fn estimate_session_context_token_usage(messages: &[SessionMessage]) -> ContextTokenEstimate {
    let Some((last_usage_index, usage)) = get_last_session_assistant_usage(messages) else {
        let tokens = estimate_session_context_tokens(messages);
        return ContextTokenEstimate {
            tokens,
            usage_tokens: 0,
            last_usage_index: None,
        };
    };

    let usage_tokens = calculate_context_tokens(usage);
    let tail_tokens = messages
        .iter()
        .skip(last_usage_index + 1)
        .map(estimate_session_message_tokens)
        .sum::<u64>();
    ContextTokenEstimate {
        tokens: usage_tokens + tail_tokens,
        usage_tokens,
        last_usage_index: Some(last_usage_index),
    }
}

pub fn get_last_assistant_usage(messages: &[Message]) -> Option<(usize, &Usage)> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| match message {
            Message::Assistant(assistant)
                if assistant.stop_reason != StopReason::Error
                    && assistant.stop_reason != StopReason::Aborted =>
            {
                Some((index, &assistant.usage))
            }
            _ => None,
        })
}

pub fn get_last_assistant_usage_from_entries(entries: &[SessionTreeEntry]) -> Option<&Usage> {
    entries.iter().rev().find_map(|entry| match entry {
        SessionTreeEntry::Message {
            message: Message::Assistant(assistant),
            ..
        } if assistant.stop_reason != StopReason::Error
            && assistant.stop_reason != StopReason::Aborted =>
        {
            Some(&assistant.usage)
        }
        _ => None,
    })
}

pub fn get_last_session_assistant_usage(messages: &[SessionMessage]) -> Option<(usize, &Usage)> {
    messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| match message {
            SessionMessage::Llm {
                message: Message::Assistant(assistant),
            } if assistant.stop_reason != StopReason::Error
                && assistant.stop_reason != StopReason::Aborted =>
            {
                Some((index, &assistant.usage))
            }
            _ => None,
        })
}

pub fn should_compact(messages: &[Message], settings: &CompactionSettings) -> bool {
    estimate_context_tokens(messages) * 100
        >= settings.max_context_tokens * settings.threshold_percent as u64
}

pub fn should_compact_tokens(
    current_tokens: u64,
    context_window: u64,
    settings: &CompactionThresholdSettings,
) -> bool {
    settings.enabled && current_tokens.saturating_add(settings.reserve_tokens) > context_window
}

pub fn find_turn_start_index(messages: &[Message], from_index: usize) -> usize {
    let mut index = from_index.min(messages.len());
    while index > 0 {
        if matches!(messages.get(index), Some(Message::User(_))) {
            return index;
        }
        index -= 1;
    }
    0
}

pub fn find_entry_turn_start_index(
    entries: &[SessionTreeEntry],
    entry_index: usize,
    start_index: usize,
) -> Option<usize> {
    if entries.is_empty() {
        return None;
    }
    let mut index = entry_index.min(entries.len() - 1);
    loop {
        match &entries[index] {
            SessionTreeEntry::BranchSummary { .. } | SessionTreeEntry::CustomMessage { .. } => {
                return Some(index);
            }
            SessionTreeEntry::Message {
                message: Message::User(_),
                ..
            } => return Some(index),
            _ => {}
        }
        if index == start_index {
            break;
        }
        index -= 1;
    }
    None
}

pub fn find_cut_point(
    entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
    keep_recent_tokens: u64,
) -> CutPointResult {
    let bounded_start = start_index.min(entries.len());
    let bounded_end = end_index.min(entries.len());
    let cut_points = find_valid_cut_points(entries, bounded_start, bounded_end);

    if cut_points.is_empty() {
        return CutPointResult {
            first_kept_entry_index: bounded_start,
            turn_start_index: None,
            is_split_turn: false,
        };
    }

    let mut accumulated_tokens = 0;
    let mut cut_index = cut_points[0];

    for index in (bounded_start..bounded_end).rev() {
        let SessionTreeEntry::Message { message, .. } = &entries[index] else {
            continue;
        };
        accumulated_tokens += estimate_message_tokens(message);
        if accumulated_tokens >= keep_recent_tokens {
            if let Some(next_cut) = cut_points.iter().find(|cut| **cut >= index) {
                cut_index = *next_cut;
            }
            break;
        }
    }

    while cut_index > bounded_start {
        let prev_entry = &entries[cut_index - 1];
        if matches!(
            prev_entry,
            SessionTreeEntry::Compaction { .. } | SessionTreeEntry::Message { .. }
        ) {
            break;
        }
        cut_index -= 1;
    }

    let is_user_message = matches!(
        entries.get(cut_index),
        Some(SessionTreeEntry::Message {
            message: Message::User(_),
            ..
        })
    );
    let turn_start_index = if is_user_message {
        None
    } else {
        find_entry_turn_start_index(entries, cut_index, bounded_start)
    };

    CutPointResult {
        first_kept_entry_index: cut_index,
        turn_start_index,
        is_split_turn: !is_user_message && turn_start_index.is_some(),
    }
}

pub fn prepare_compaction(
    path_entries: &[SessionTreeEntry],
    settings: CompactionThresholdSettings,
) -> Result<Option<CompactionPreparation>, CompactionError> {
    if path_entries.is_empty()
        || matches!(
            path_entries.last(),
            Some(SessionTreeEntry::Compaction { .. })
        )
    {
        return Ok(None);
    }

    let prev_compaction_index = path_entries
        .iter()
        .rposition(|entry| matches!(entry, SessionTreeEntry::Compaction { .. }));

    let mut previous_summary = None;
    let mut boundary_start = 0;
    if let Some(index) = prev_compaction_index {
        if let SessionTreeEntry::Compaction {
            summary,
            first_kept_entry_id,
            ..
        } = &path_entries[index]
        {
            previous_summary = Some(summary.clone());
            boundary_start = path_entries
                .iter()
                .position(|entry| entry.id() == first_kept_entry_id)
                .unwrap_or(index + 1);
        }
    }

    let boundary_end = path_entries.len();
    let context = build_session_context(path_entries).map_err(compaction_session_error)?;
    let tokens_before = estimate_session_context_token_usage(&context.messages).tokens;
    let cut_point = find_cut_point(
        path_entries,
        boundary_start,
        boundary_end,
        settings.keep_recent_tokens,
    );
    let first_kept_entry = path_entries
        .get(cut_point.first_kept_entry_index)
        .ok_or_else(|| {
            CompactionError::new(
                CompactionErrorCode::InvalidSession,
                "First kept entry has no UUID - session may need migration",
            )
        })?;
    let first_kept_entry_id = first_kept_entry.id().to_owned();
    let history_end = if cut_point.is_split_turn {
        cut_point.turn_start_index.ok_or_else(|| {
            CompactionError::new(
                CompactionErrorCode::InvalidSession,
                "Split-turn compaction is missing turn start",
            )
        })?
    } else {
        cut_point.first_kept_entry_index
    };

    let messages_to_summarize = path_entries[boundary_start..history_end]
        .iter()
        .filter_map(get_message_from_entry_for_compaction)
        .collect::<Vec<_>>();
    let turn_prefix_messages = if let Some(turn_start_index) = cut_point.turn_start_index {
        path_entries[turn_start_index..cut_point.first_kept_entry_index]
            .iter()
            .filter_map(get_message_from_entry_for_compaction)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut file_ops =
        extract_file_operations(&messages_to_summarize, path_entries, prev_compaction_index);
    if cut_point.is_split_turn {
        for message in &turn_prefix_messages {
            extract_file_ops_from_message(message, &mut file_ops);
        }
    }

    Ok(Some(CompactionPreparation {
        first_kept_entry_id,
        messages_to_summarize,
        turn_prefix_messages,
        is_split_turn: cut_point.is_split_turn,
        tokens_before,
        previous_summary,
        file_ops,
        settings,
    }))
}

pub async fn generate_summary(
    current_messages: &[SessionMessage],
    model: &Model,
    reserve_tokens: u64,
    api_key: impl Into<String>,
    headers: Option<BTreeMap<String, String>>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
) -> Result<String, CompactionError> {
    generate_summary_with_prompt(
        current_messages,
        model,
        reserve_tokens,
        api_key.into(),
        headers,
        custom_instructions,
        previous_summary,
        thinking_level,
        false,
    )
    .await
}

pub async fn compact(
    preparation: &CompactionPreparation,
    model: &Model,
    api_key: impl Into<String>,
    headers: Option<BTreeMap<String, String>>,
    custom_instructions: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
) -> Result<CompactionResult, CompactionError> {
    if preparation.first_kept_entry_id.is_empty() {
        return Err(CompactionError::new(
            CompactionErrorCode::InvalidSession,
            "First kept entry has no UUID - session may need migration",
        ));
    }

    let api_key = api_key.into();
    let summary = if preparation.is_split_turn && !preparation.turn_prefix_messages.is_empty() {
        let history_summary = if preparation.messages_to_summarize.is_empty() {
            "No prior history.".to_owned()
        } else {
            generate_summary(
                &preparation.messages_to_summarize,
                model,
                preparation.settings.reserve_tokens,
                api_key.clone(),
                headers.clone(),
                custom_instructions,
                preparation.previous_summary.as_deref(),
                thinking_level,
            )
            .await?
        };
        let prefix_summary = generate_turn_prefix_summary(
            &preparation.turn_prefix_messages,
            model,
            preparation.settings.reserve_tokens,
            api_key,
            headers,
            thinking_level,
        )
        .await?;
        format!("{history_summary}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_summary}")
    } else {
        generate_summary(
            &preparation.messages_to_summarize,
            model,
            preparation.settings.reserve_tokens,
            api_key,
            headers,
            custom_instructions,
            preparation.previous_summary.as_deref(),
            thinking_level,
        )
        .await?
    };

    let file_lists = compute_file_lists(&preparation.file_ops);
    let summary = format!(
        "{}{}",
        summary,
        format_file_operations(&file_lists.read_files, &file_lists.modified_files)
    );

    Ok(CompactionResult {
        summary,
        first_kept_entry_id: preparation.first_kept_entry_id.clone(),
        tokens_before: preparation.tokens_before,
        details: CompactionDetails {
            read_files: file_lists.read_files,
            modified_files: file_lists.modified_files,
        },
    })
}

pub fn collect_entries_for_branch_summary(
    session: &Session,
    old_leaf_id: Option<&str>,
    target_id: &str,
) -> Result<CollectEntriesResult, BranchSummaryError> {
    let Some(old_leaf_id) = old_leaf_id else {
        return Ok(CollectEntriesResult {
            entries: Vec::new(),
            common_ancestor_id: None,
        });
    };
    let old_path = session
        .branch(Some(old_leaf_id))
        .map_err(branch_summary_session_error)?
        .into_iter()
        .map(|entry| entry.id().to_owned())
        .collect::<BTreeSet<_>>();
    let target_path = session
        .branch(Some(target_id))
        .map_err(branch_summary_session_error)?;
    let common_ancestor_id = target_path
        .iter()
        .rev()
        .find(|entry| old_path.contains(entry.id()))
        .map(|entry| entry.id().to_owned());

    let mut entries = Vec::new();
    let mut current = Some(old_leaf_id.to_owned());
    while let Some(current_id) = current {
        if common_ancestor_id.as_deref() == Some(current_id.as_str()) {
            break;
        }
        let entry = session.get_entry(&current_id).ok_or_else(|| {
            BranchSummaryError::new(
                BranchSummaryErrorCode::InvalidSession,
                format!("Entry {current_id} not found"),
            )
        })?;
        current = entry.parent_id().map(str::to_owned);
        entries.push(entry);
    }
    entries.reverse();

    Ok(CollectEntriesResult {
        entries,
        common_ancestor_id,
    })
}

pub fn prepare_branch_entries(
    entries: &[SessionTreeEntry],
    token_budget: u64,
) -> BranchPreparation {
    let mut messages = Vec::new();
    let mut file_ops = create_file_ops();
    let mut total_tokens = 0;

    for entry in entries {
        if let SessionTreeEntry::BranchSummary {
            details, from_hook, ..
        } = entry
        {
            if !from_hook.unwrap_or(false)
                && let Some(details) = details
            {
                add_string_array_field(details, "readFiles", &mut file_ops.read);
                add_string_array_field(details, "modifiedFiles", &mut file_ops.edited);
            }
        }
    }

    for entry in entries.iter().rev() {
        let Some(message) = get_message_from_entry_for_branch_summary(entry) else {
            continue;
        };
        extract_file_ops_from_message(&message, &mut file_ops);
        let tokens = estimate_session_message_tokens(&message);
        if token_budget > 0 && total_tokens + tokens > token_budget {
            if matches!(
                entry,
                SessionTreeEntry::Compaction { .. } | SessionTreeEntry::BranchSummary { .. }
            ) && total_tokens.saturating_mul(10) < token_budget.saturating_mul(9)
            {
                messages.insert(0, message);
                total_tokens += tokens;
            }
            break;
        }
        messages.insert(0, message);
        total_tokens += tokens;
    }

    BranchPreparation {
        messages,
        file_ops,
        total_tokens,
    }
}

pub async fn generate_branch_summary(
    entries: &[SessionTreeEntry],
    model: &Model,
    api_key: impl Into<String>,
    headers: Option<BTreeMap<String, String>>,
    custom_instructions: Option<&str>,
    replace_instructions: bool,
    reserve_tokens: Option<u64>,
) -> Result<BranchSummaryResult, BranchSummaryError> {
    let reserve_tokens = reserve_tokens.unwrap_or(16_384);
    let token_budget = model.context_window.saturating_sub(reserve_tokens);
    let preparation = prepare_branch_entries(entries, token_budget);

    if preparation.messages.is_empty() {
        return Ok(BranchSummaryResult {
            summary: "No content to summarize".to_owned(),
            read_files: Vec::new(),
            modified_files: Vec::new(),
        });
    }

    let llm_messages = convert_session_messages_to_llm(&preparation.messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let instructions = if replace_instructions {
        custom_instructions
            .unwrap_or(BRANCH_SUMMARY_PROMPT)
            .to_owned()
    } else if let Some(custom_instructions) = custom_instructions {
        format!("{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}")
    } else {
        BRANCH_SUMMARY_PROMPT.to_owned()
    };
    let prompt_text =
        format!("<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}");
    let context = Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![Message::User(ri_llm_provider::UserMessage {
            content: UserContentValue::Blocks(vec![UserContent::Text(TextContent::new(
                prompt_text,
            ))]),
            timestamp: ri_llm_provider::now_millis(),
        })],
        tools: Vec::new(),
    };
    let response = complete_simple(
        model,
        context,
        SimpleStreamOptions {
            stream: StreamOptions {
                max_tokens: Some(2_048),
                api_key: Some(api_key.into()),
                headers: headers.unwrap_or_default(),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .await
    .map_err(|error| {
        BranchSummaryError::new(
            BranchSummaryErrorCode::SummarizationFailed,
            format!("Branch summary failed: {error}"),
        )
    })?;

    match response.stop_reason {
        StopReason::Aborted => Err(BranchSummaryError::new(
            BranchSummaryErrorCode::Aborted,
            response
                .error_message
                .unwrap_or_else(|| "Branch summary aborted".to_owned()),
        )),
        StopReason::Error => Err(BranchSummaryError::new(
            BranchSummaryErrorCode::SummarizationFailed,
            format!(
                "Branch summary failed: {}",
                response
                    .error_message
                    .unwrap_or_else(|| "Unknown error".to_owned())
            ),
        )),
        _ => {
            let mut summary = response
                .content
                .into_iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            summary = format!("{BRANCH_SUMMARY_PREAMBLE}{summary}");
            let file_lists = compute_file_lists(&preparation.file_ops);
            summary.push_str(&format_file_operations(
                &file_lists.read_files,
                &file_lists.modified_files,
            ));

            Ok(BranchSummaryResult {
                summary: if summary.is_empty() {
                    "No summary generated".to_owned()
                } else {
                    summary
                },
                read_files: file_lists.read_files,
                modified_files: file_lists.modified_files,
            })
        }
    }
}

pub fn create_file_ops() -> FileOperations {
    FileOperations::default()
}

pub fn extract_file_ops_from_message(message: &SessionMessage, file_ops: &mut FileOperations) {
    let SessionMessage::Llm {
        message: Message::Assistant(assistant),
    } = message
    else {
        return;
    };
    for block in &assistant.content {
        let AssistantContent::ToolCall(tool_call) = block else {
            continue;
        };
        let Some(path) = tool_call.arguments.get("path").and_then(Value::as_str) else {
            continue;
        };
        match tool_call.name.as_str() {
            "read" => {
                file_ops.read.insert(path.to_owned());
            }
            "write" => {
                file_ops.written.insert(path.to_owned());
            }
            "edit" => {
                file_ops.edited.insert(path.to_owned());
            }
            _ => {}
        }
    }
}

pub fn compute_file_lists(file_ops: &FileOperations) -> FileOperationLists {
    let modified = file_ops
        .edited
        .union(&file_ops.written)
        .cloned()
        .collect::<BTreeSet<_>>();
    let read_files = file_ops
        .read
        .difference(&modified)
        .cloned()
        .collect::<Vec<_>>();
    let modified_files = modified.into_iter().collect::<Vec<_>>();
    FileOperationLists {
        read_files,
        modified_files,
    }
}

pub fn format_file_operations(read_files: &[String], modified_files: &[String]) -> String {
    let mut sections = Vec::new();
    if !read_files.is_empty() {
        sections.push(format!(
            "<read-files>\n{}\n</read-files>",
            read_files.join("\n")
        ));
    }
    if !modified_files.is_empty() {
        sections.push(format!(
            "<modified-files>\n{}\n</modified-files>",
            modified_files.join("\n")
        ));
    }
    if sections.is_empty() {
        String::new()
    } else {
        format!("\n\n{}", sections.join("\n\n"))
    }
}

pub fn serialize_conversation(messages: &[Message]) -> String {
    let mut parts = Vec::new();
    for message in messages {
        match message {
            Message::User(user) => {
                let content = match &user.content {
                    UserContentValue::Plain(text) => text.clone(),
                    UserContentValue::Blocks(blocks) => blocks
                        .iter()
                        .filter_map(|block| match block {
                            UserContent::Text(text) => Some(text.text.as_str()),
                            UserContent::Image(_) => None,
                        })
                        .collect::<String>(),
                };
                if !content.is_empty() {
                    parts.push(format!("[User]: {content}"));
                }
            }
            Message::Assistant(assistant) => {
                let mut text_parts = Vec::new();
                let mut thinking_parts = Vec::new();
                let mut tool_calls = Vec::new();
                for block in &assistant.content {
                    match block {
                        AssistantContent::Text(text) => text_parts.push(text.text.as_str()),
                        AssistantContent::Thinking(thinking) => {
                            thinking_parts.push(thinking.thinking.as_str());
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let args = tool_call
                                .arguments
                                .iter()
                                .map(|(key, value)| format!("{key}={}", safe_json_stringify(value)))
                                .collect::<Vec<_>>()
                                .join(", ");
                            tool_calls.push(format!("{}({args})", tool_call.name));
                        }
                    }
                }
                if !thinking_parts.is_empty() {
                    parts.push(format!(
                        "[Assistant thinking]: {}",
                        thinking_parts.join("\n")
                    ));
                }
                if !text_parts.is_empty() {
                    parts.push(format!("[Assistant]: {}", text_parts.join("\n")));
                }
                if !tool_calls.is_empty() {
                    parts.push(format!("[Assistant tool calls]: {}", tool_calls.join("; ")));
                }
            }
            Message::ToolResult(tool_result) => {
                let content = tool_result
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ToolResultContent::Text(text) => Some(text.text.as_str()),
                        ToolResultContent::Image(_) => None,
                    })
                    .collect::<String>();
                if !content.is_empty() {
                    parts.push(format!(
                        "[Tool result]: {}",
                        truncate_for_summary(&content, 2000)
                    ));
                }
            }
        }
    }
    parts.join("\n\n")
}

pub fn convert_session_messages_to_llm(messages: &[SessionMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter_map(convert_session_message_to_llm)
        .collect()
}

fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::User(message) => match &message.content {
            ri_llm_provider::UserContentValue::Plain(text) => estimate_tokens(text),
            ri_llm_provider::UserContentValue::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    ri_llm_provider::UserContent::Text(text) => estimate_tokens(&text.text),
                    ri_llm_provider::UserContent::Image(image) => estimate_tokens(&image.data),
                })
                .sum(),
        },
        Message::Assistant(message) => message
            .content
            .iter()
            .map(|content| match content {
                AssistantContent::Text(text) => estimate_tokens(&text.text),
                AssistantContent::Thinking(thinking) => estimate_tokens(&thinking.thinking),
                AssistantContent::ToolCall(tool_call) => {
                    estimate_tokens(&tool_call.name)
                        + estimate_tokens(&safe_json_stringify(&tool_call.arguments))
                }
            })
            .sum(),
        Message::ToolResult(message) => message
            .content
            .iter()
            .map(|content| match content {
                ri_llm_provider::ToolResultContent::Text(text) => estimate_tokens(&text.text),
                ri_llm_provider::ToolResultContent::Image(_) => estimate_tokens(&"x".repeat(4800)),
            })
            .sum(),
    }
}

fn estimate_session_message_tokens(message: &SessionMessage) -> u64 {
    match message {
        SessionMessage::Llm { message } => estimate_message_tokens(message),
        SessionMessage::Custom { content, .. } => match content {
            CustomMessageContent::Text(text) => estimate_tokens(text),
            CustomMessageContent::Blocks(blocks) => blocks
                .iter()
                .map(|block| match block {
                    UserContent::Text(text) => estimate_tokens(&text.text),
                    UserContent::Image(_) => estimate_tokens(&"x".repeat(4800)),
                })
                .sum(),
        },
        SessionMessage::BranchSummary { summary, .. }
        | SessionMessage::CompactionSummary { summary, .. } => estimate_tokens(summary),
    }
}

fn find_valid_cut_points(
    entries: &[SessionTreeEntry],
    start_index: usize,
    end_index: usize,
) -> Vec<usize> {
    let mut cut_points = Vec::new();
    for index in start_index..end_index {
        match &entries[index] {
            SessionTreeEntry::Message {
                message: Message::User(_) | Message::Assistant(_),
                ..
            }
            | SessionTreeEntry::BranchSummary { .. }
            | SessionTreeEntry::CustomMessage { .. } => cut_points.push(index),
            _ => {}
        }
    }
    cut_points
}

fn get_message_from_entry(entry: &SessionTreeEntry) -> Option<SessionMessage> {
    match entry {
        SessionTreeEntry::Message { message, .. } => Some(message.clone().into()),
        SessionTreeEntry::CustomMessage {
            custom_type,
            content,
            display,
            details,
            timestamp,
            ..
        } => Some(SessionMessage::Custom {
            custom_type: custom_type.clone(),
            content: content.clone(),
            display: *display,
            details: details.clone(),
            timestamp: parse_timestamp_millis(timestamp),
        }),
        SessionTreeEntry::BranchSummary {
            summary,
            from_id,
            timestamp,
            ..
        } => Some(SessionMessage::BranchSummary {
            summary: summary.clone(),
            from_id: from_id.clone(),
            timestamp: parse_timestamp_millis(timestamp),
        }),
        SessionTreeEntry::Compaction {
            summary,
            tokens_before,
            timestamp,
            ..
        } => Some(SessionMessage::CompactionSummary {
            summary: summary.clone(),
            tokens_before: *tokens_before,
            timestamp: parse_timestamp_millis(timestamp),
        }),
        _ => None,
    }
}

fn get_message_from_entry_for_compaction(entry: &SessionTreeEntry) -> Option<SessionMessage> {
    if matches!(entry, SessionTreeEntry::Compaction { .. }) {
        None
    } else {
        get_message_from_entry(entry)
    }
}

fn get_message_from_entry_for_branch_summary(entry: &SessionTreeEntry) -> Option<SessionMessage> {
    if matches!(
        entry,
        SessionTreeEntry::Message {
            message: Message::ToolResult(_),
            ..
        }
    ) {
        None
    } else {
        get_message_from_entry(entry)
    }
}

fn extract_file_operations(
    messages: &[SessionMessage],
    entries: &[SessionTreeEntry],
    prev_compaction_index: Option<usize>,
) -> FileOperations {
    let mut file_ops = create_file_ops();
    if let Some(index) = prev_compaction_index {
        if let SessionTreeEntry::Compaction {
            details, from_hook, ..
        } = &entries[index]
        {
            if !from_hook.unwrap_or(false) {
                if let Some(details) = details {
                    add_string_array_field(details, "readFiles", &mut file_ops.read);
                    add_string_array_field(details, "modifiedFiles", &mut file_ops.edited);
                }
            }
        }
    }
    for message in messages {
        extract_file_ops_from_message(message, &mut file_ops);
    }
    file_ops
}

fn add_string_array_field(value: &Value, field: &str, target: &mut BTreeSet<String>) {
    let Some(array) = value.get(field).and_then(Value::as_array) else {
        return;
    };
    for item in array {
        if let Some(path) = item.as_str() {
            target.insert(path.to_owned());
        }
    }
}

fn safe_json_stringify(value: &impl Serialize) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_owned())
}

fn truncate_for_summary(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text.to_owned();
    }
    let truncated_chars = char_count - max_chars;
    let prefix = text.chars().take(max_chars).collect::<String>();
    format!("{prefix}\n\n[... {truncated_chars} more characters truncated]")
}

fn compaction_session_error(error: SessionError) -> CompactionError {
    CompactionError::new(CompactionErrorCode::InvalidSession, error.message)
}

fn branch_summary_session_error(error: SessionError) -> BranchSummaryError {
    BranchSummaryError::new(BranchSummaryErrorCode::InvalidSession, error.message)
}

fn parse_timestamp_millis(timestamp: &str) -> i64 {
    chrono::DateTime::parse_from_rfc3339(timestamp)
        .map(|value| value.timestamp_millis())
        .unwrap_or_else(|_| ri_llm_provider::now_millis())
}

async fn generate_turn_prefix_summary(
    messages: &[SessionMessage],
    model: &Model,
    reserve_tokens: u64,
    api_key: String,
    headers: Option<BTreeMap<String, String>>,
    thinking_level: Option<ThinkingLevel>,
) -> Result<String, CompactionError> {
    generate_summary_with_prompt(
        messages,
        model,
        reserve_tokens,
        api_key,
        headers,
        None,
        None,
        thinking_level,
        true,
    )
    .await
}

async fn generate_summary_with_prompt(
    current_messages: &[SessionMessage],
    model: &Model,
    reserve_tokens: u64,
    api_key: String,
    headers: Option<BTreeMap<String, String>>,
    custom_instructions: Option<&str>,
    previous_summary: Option<&str>,
    thinking_level: Option<ThinkingLevel>,
    turn_prefix: bool,
) -> Result<String, CompactionError> {
    let max_tokens = if turn_prefix {
        summary_max_tokens(reserve_tokens, model.max_tokens, 5, 10)
    } else {
        summary_max_tokens(reserve_tokens, model.max_tokens, 8, 10)
    };
    let llm_messages = convert_session_messages_to_llm(current_messages);
    let conversation_text = serialize_conversation(&llm_messages);
    let prompt_text = if turn_prefix {
        format!(
            "<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
        )
    } else {
        let mut base_prompt = if previous_summary.is_some() {
            UPDATE_SUMMARIZATION_PROMPT.to_owned()
        } else {
            SUMMARIZATION_PROMPT.to_owned()
        };
        if let Some(custom_instructions) = custom_instructions {
            base_prompt.push_str("\n\nAdditional focus: ");
            base_prompt.push_str(custom_instructions);
        }
        let mut prompt = format!("<conversation>\n{conversation_text}\n</conversation>\n\n");
        if let Some(previous_summary) = previous_summary {
            prompt.push_str("<previous-summary>\n");
            prompt.push_str(previous_summary);
            prompt.push_str("\n</previous-summary>\n\n");
        }
        prompt.push_str(&base_prompt);
        prompt
    };
    let context = Context {
        system_prompt: Some(SUMMARIZATION_SYSTEM_PROMPT.to_owned()),
        messages: vec![Message::User(ri_llm_provider::UserMessage {
            content: UserContentValue::Blocks(vec![UserContent::Text(TextContent::new(
                prompt_text,
            ))]),
            timestamp: ri_llm_provider::now_millis(),
        })],
        tools: Vec::new(),
    };
    let response = complete_simple(
        model,
        context,
        summary_options(model, max_tokens, api_key, headers, thinking_level),
    )
    .await
    .map_err(|error| {
        CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            if turn_prefix {
                format!("Turn prefix summarization failed: {error}")
            } else {
                format!("Summarization failed: {error}")
            },
        )
    })?;

    match response.stop_reason {
        StopReason::Aborted => Err(CompactionError::new(
            CompactionErrorCode::Aborted,
            response.error_message.unwrap_or_else(|| {
                if turn_prefix {
                    "Turn prefix summarization aborted".to_owned()
                } else {
                    "Summarization aborted".to_owned()
                }
            }),
        )),
        StopReason::Error => Err(CompactionError::new(
            CompactionErrorCode::SummarizationFailed,
            if turn_prefix {
                format!(
                    "Turn prefix summarization failed: {}",
                    response
                        .error_message
                        .unwrap_or_else(|| "Unknown error".to_owned())
                )
            } else {
                format!(
                    "Summarization failed: {}",
                    response
                        .error_message
                        .unwrap_or_else(|| "Unknown error".to_owned())
                )
            },
        )),
        _ => Ok(response
            .content
            .into_iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")),
    }
}

fn summary_max_tokens(
    reserve_tokens: u64,
    model_max_tokens: u64,
    numerator: u64,
    denominator: u64,
) -> u64 {
    let requested = reserve_tokens.saturating_mul(numerator) / denominator;
    if model_max_tokens > 0 {
        requested.min(model_max_tokens)
    } else {
        requested
    }
}

fn summary_options(
    model: &Model,
    max_tokens: u64,
    api_key: String,
    headers: Option<BTreeMap<String, String>>,
    thinking_level: Option<ThinkingLevel>,
) -> SimpleStreamOptions {
    let reasoning = thinking_level.filter(|level| model.reasoning && *level != ThinkingLevel::Off);
    SimpleStreamOptions {
        stream: StreamOptions {
            max_tokens: Some(max_tokens),
            api_key: Some(api_key),
            headers: headers.unwrap_or_default(),
            ..Default::default()
        },
        reasoning,
        ..Default::default()
    }
}

fn convert_session_message_to_llm(message: &SessionMessage) -> Option<Message> {
    match message {
        SessionMessage::Llm { message } => Some(message.clone()),
        SessionMessage::Custom {
            content, timestamp, ..
        } => Some(Message::User(ri_llm_provider::UserMessage {
            content: match content {
                CustomMessageContent::Text(text) => {
                    UserContentValue::Blocks(vec![UserContent::Text(TextContent::new(
                        text.clone(),
                    ))])
                }
                CustomMessageContent::Blocks(blocks) => UserContentValue::Blocks(blocks.clone()),
            },
            timestamp: *timestamp,
        })),
        SessionMessage::BranchSummary {
            summary, timestamp, ..
        } => Some(Message::User(ri_llm_provider::UserMessage {
            content: UserContentValue::Blocks(vec![UserContent::Text(TextContent::new(format!(
                "{BRANCH_SUMMARY_PREFIX}{summary}{BRANCH_SUMMARY_SUFFIX}"
            )))]),
            timestamp: *timestamp,
        })),
        SessionMessage::CompactionSummary {
            summary, timestamp, ..
        } => Some(Message::User(ri_llm_provider::UserMessage {
            content: UserContentValue::Blocks(vec![UserContent::Text(TextContent::new(format!(
                "{COMPACTION_SUMMARY_PREFIX}{summary}{COMPACTION_SUMMARY_SUFFIX}"
            )))]),
            timestamp: *timestamp,
        })),
    }
}
