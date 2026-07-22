//! Text extraction from message content, mirroring pi `utils/text.ts`.

use crate::{AssistantContent, ToolResultContent, UserContent};

/// Extract and join text blocks from assistant content.
pub fn assistant_content_text(content: &[AssistantContent], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extract and join text blocks from tool-result content.
pub fn tool_result_content_text(content: &[ToolResultContent], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ToolResultContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}

/// Extract and join text blocks from user content blocks.
pub fn user_content_text(content: &[UserContent], separator: &str) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            UserContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(separator)
}
