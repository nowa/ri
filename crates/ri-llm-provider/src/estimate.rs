//! Context token estimation, mirroring pi `utils/estimate.ts`.

use crate::{
    AssistantContent, Context, Message, StopReason, Tool, ToolResultContent, Usage, UserContent,
    UserContentValue,
};

/// Estimated context-token usage for a message list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextUsageEstimate {
    /// Estimated total context tokens.
    pub tokens: u64,
    /// Tokens reported by the most recent applicable assistant usage block.
    pub usage_tokens: u64,
    /// Estimated tokens after the most recent applicable assistant usage block.
    pub trailing_tokens: u64,
    /// Index of the applicable message that provided usage, or `None` when
    /// none exists.
    pub last_usage_index: Option<usize>,
}

const CHARS_PER_TOKEN: u64 = 4;
const ESTIMATED_IMAGE_CHARS: u64 = 4800;

pub fn calculate_context_tokens(usage: &Usage) -> u64 {
    if usage.total_tokens > 0 {
        usage.total_tokens
    } else {
        usage.input + usage.output + usage.cache_read + usage.cache_write
    }
}

fn safe_json_stringify<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "[unserializable]".to_owned())
}

pub fn estimate_text_tokens(text: &str) -> u64 {
    (text.encode_utf16().count() as u64).div_ceil(CHARS_PER_TOKEN)
}

fn estimate_user_content_chars(content: &UserContentValue) -> u64 {
    match content {
        UserContentValue::Plain(text) => text.encode_utf16().count() as u64,
        UserContentValue::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                UserContent::Text(text) => text.text.encode_utf16().count() as u64,
                UserContent::Image(_) => ESTIMATED_IMAGE_CHARS,
            })
            .sum(),
    }
}

fn estimate_tool_result_content_chars(content: &[ToolResultContent]) -> u64 {
    content
        .iter()
        .map(|block| match block {
            ToolResultContent::Text(text) => text.text.encode_utf16().count() as u64,
            ToolResultContent::Image(_) => ESTIMATED_IMAGE_CHARS,
        })
        .sum()
}

pub fn estimate_message_tokens(message: &Message) -> u64 {
    let chars = match message {
        Message::User(user) => estimate_user_content_chars(&user.content),
        Message::ToolResult(tool_result) => {
            estimate_tool_result_content_chars(&tool_result.content)
        }
        Message::Assistant(assistant) => assistant
            .content
            .iter()
            .map(|block| match block {
                AssistantContent::Text(text) => text.text.encode_utf16().count() as u64,
                AssistantContent::Thinking(thinking) => {
                    thinking.thinking.encode_utf16().count() as u64
                }
                AssistantContent::ToolCall(tool_call) => {
                    (tool_call.name.encode_utf16().count()
                        + safe_json_stringify(&tool_call.arguments)
                            .encode_utf16()
                            .count()) as u64
                }
            })
            .sum(),
    };
    chars.div_ceil(CHARS_PER_TOKEN)
}

fn message_timestamp(message: &Message) -> i64 {
    match message {
        Message::User(user) => user.timestamp,
        Message::Assistant(assistant) => assistant.timestamp,
        Message::ToolResult(tool_result) => tool_result.timestamp,
    }
}

fn get_last_assistant_usage_info(messages: &[Message]) -> Option<(usize, &Usage)> {
    let mut latest_prefix_timestamp = i64::MIN;
    let mut usage_info = None;

    for (index, message) in messages.iter().enumerate() {
        if let Message::Assistant(assistant) = message {
            // A newer prefix message was inserted after this response (for
            // example, a compaction summary), so its usage cannot describe the
            // current prefix.
            let usage_applies_to_prefix = assistant.timestamp >= latest_prefix_timestamp;
            if usage_applies_to_prefix
                && assistant.stop_reason != StopReason::Aborted
                && assistant.stop_reason != StopReason::Error
                && calculate_context_tokens(&assistant.usage) > 0
            {
                usage_info = Some((index, &assistant.usage));
            }
        }
        latest_prefix_timestamp = latest_prefix_timestamp.max(message_timestamp(message));
    }

    usage_info
}

fn estimate_messages(messages: &[Message]) -> ContextUsageEstimate {
    if let Some((index, usage)) = get_last_assistant_usage_info(messages) {
        let usage_tokens = calculate_context_tokens(usage);
        let trailing_tokens = messages
            .iter()
            .skip(index + 1)
            .map(estimate_message_tokens)
            .sum::<u64>();
        return ContextUsageEstimate {
            tokens: usage_tokens + trailing_tokens,
            usage_tokens,
            trailing_tokens,
            last_usage_index: Some(index),
        };
    }

    let tokens = messages.iter().map(estimate_message_tokens).sum();
    ContextUsageEstimate {
        tokens,
        usage_tokens: 0,
        trailing_tokens: tokens,
        last_usage_index: None,
    }
}

fn estimate_tools_tokens(tools: &[&Tool]) -> u64 {
    if tools.is_empty() {
        return 0;
    }
    estimate_text_tokens(&safe_json_stringify(&tools))
}

/// Estimate context tokens for a bare message list, mirroring pi
/// `estimateContextTokens(Message[])`.
pub fn estimate_context_messages_tokens(messages: &[Message]) -> ContextUsageEstimate {
    estimate_messages(messages)
}

/// Estimate context tokens for a full context, mirroring pi
/// `estimateContextTokens(Context)`.
pub fn estimate_context_tokens(context: &Context) -> ContextUsageEstimate {
    let estimate = estimate_messages(&context.messages);
    if let Some(last_usage_index) = estimate.last_usage_index {
        let added_names = context
            .messages
            .iter()
            .skip(last_usage_index + 1)
            .filter_map(|message| match message {
                Message::ToolResult(tool_result) => tool_result.added_tool_names.as_deref(),
                _ => None,
            })
            .flatten()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        let added_tools = context
            .tools
            .iter()
            .filter(|tool| added_names.contains(tool.name.as_str()))
            .collect::<Vec<_>>();
        let added_tool_tokens = estimate_tools_tokens(&added_tools);
        return ContextUsageEstimate {
            tokens: estimate.tokens + added_tool_tokens,
            usage_tokens: estimate.usage_tokens,
            trailing_tokens: estimate.trailing_tokens + added_tool_tokens,
            last_usage_index: estimate.last_usage_index,
        };
    }

    let prefix_tokens = context
        .system_prompt
        .as_deref()
        .map(estimate_text_tokens)
        .unwrap_or(0)
        + estimate_tools_tokens(&context.tools.iter().collect::<Vec<_>>());

    ContextUsageEstimate {
        tokens: estimate.tokens + prefix_tokens,
        usage_tokens: estimate.usage_tokens,
        trailing_tokens: estimate.trailing_tokens + prefix_tokens,
        last_usage_index: estimate.last_usage_index,
    }
}
