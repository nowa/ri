use crate::{
    AssistantContent, AssistantMessage, ImageContent, InputKind, Message, Model, StopReason,
    TextContent, ToolCall, ToolResultContent, ToolResultMessage, UserContent, UserContentValue,
};
use std::collections::{HashMap, HashSet};

pub const NON_VISION_USER_IMAGE_PLACEHOLDER: &str =
    "(image omitted: model does not support images)";
pub const NON_VISION_TOOL_IMAGE_PLACEHOLDER: &str =
    "(tool image omitted: model does not support images)";

pub type ToolCallIdNormalizer<'a> = dyn Fn(&str, &Model, &AssistantMessage) -> String + 'a;

pub fn transform_messages(
    messages: &[Message],
    model: &Model,
    normalize_tool_call_id: Option<&ToolCallIdNormalizer<'_>>,
) -> Vec<Message> {
    let mut tool_call_id_map: HashMap<String, String> = HashMap::new();
    let image_aware_messages = downgrade_unsupported_images(messages, model);

    let transformed = image_aware_messages
        .into_iter()
        .map(|message| match message {
            Message::User(_) => message,
            Message::ToolResult(mut tool_result) => {
                if let Some(normalized) = tool_call_id_map.get(&tool_result.tool_call_id) {
                    tool_result.tool_call_id = normalized.clone();
                }
                Message::ToolResult(tool_result)
            }
            Message::Assistant(assistant) => Message::Assistant(transform_assistant_message(
                assistant,
                model,
                normalize_tool_call_id,
                &mut tool_call_id_map,
            )),
        })
        .collect::<Vec<_>>();

    insert_synthetic_tool_results(transformed)
}

pub fn normalize_openai_completions_tool_call_id(id: &str, model: &Model) -> String {
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or_default();
        return call_id
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .take(40)
            .collect();
    }

    if model.provider == "openai" && id.len() > 40 {
        id.chars().take(40).collect()
    } else {
        id.to_owned()
    }
}

pub fn downgrade_unsupported_images(messages: &[Message], model: &Model) -> Vec<Message> {
    if model.input.contains(&InputKind::Image) {
        return messages.to_vec();
    }

    messages
        .iter()
        .cloned()
        .map(|message| match message {
            Message::User(mut user) => {
                if let UserContentValue::Blocks(blocks) = user.content {
                    user.content = UserContentValue::Blocks(replace_user_images_with_placeholder(
                        &blocks,
                        NON_VISION_USER_IMAGE_PLACEHOLDER,
                    ));
                }
                Message::User(user)
            }
            Message::ToolResult(mut tool_result) => {
                tool_result.content = replace_tool_images_with_placeholder(
                    &tool_result.content,
                    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                );
                Message::ToolResult(tool_result)
            }
            Message::Assistant(_) => message,
        })
        .collect()
}

fn transform_assistant_message(
    assistant: AssistantMessage,
    model: &Model,
    normalize_tool_call_id: Option<&ToolCallIdNormalizer<'_>>,
    tool_call_id_map: &mut HashMap<String, String>,
) -> AssistantMessage {
    let is_same_model = assistant.provider == model.provider
        && assistant.api == model.api
        && assistant.model == model.id;
    let mut next = assistant.clone();
    next.content = assistant
        .content
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::Thinking(thinking) => {
                if thinking.redacted {
                    return is_same_model.then_some(AssistantContent::Thinking(thinking));
                }
                if is_same_model && thinking.thinking_signature.is_some() {
                    return Some(AssistantContent::Thinking(thinking));
                }
                if thinking.thinking.trim().is_empty() {
                    return None;
                }
                if is_same_model {
                    Some(AssistantContent::Thinking(thinking))
                } else {
                    Some(AssistantContent::Text(TextContent::new(thinking.thinking)))
                }
            }
            AssistantContent::Text(text) => {
                if is_same_model {
                    Some(AssistantContent::Text(text))
                } else {
                    Some(AssistantContent::Text(TextContent::new(text.text)))
                }
            }
            AssistantContent::ToolCall(mut tool_call) => {
                if !is_same_model {
                    tool_call.thought_signature = None;
                    if let Some(normalize) = normalize_tool_call_id {
                        let normalized = normalize(&tool_call.id, model, &next);
                        if normalized != tool_call.id {
                            tool_call_id_map.insert(tool_call.id.clone(), normalized.clone());
                            tool_call.id = normalized;
                        }
                    }
                }
                Some(AssistantContent::ToolCall(tool_call))
            }
        })
        .collect();
    next
}

fn insert_synthetic_tool_results(messages: Vec<Message>) -> Vec<Message> {
    let mut result = Vec::new();
    let mut pending_tool_calls: Vec<ToolCall> = Vec::new();
    let mut existing_tool_result_ids = HashSet::new();

    for message in messages {
        match message {
            Message::Assistant(assistant) => {
                insert_missing_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                );
                if assistant.stop_reason == StopReason::Error
                    || assistant.stop_reason == StopReason::Aborted
                {
                    continue;
                }

                pending_tool_calls = assistant
                    .content
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect();
                existing_tool_result_ids.clear();
                result.push(Message::Assistant(assistant));
            }
            Message::ToolResult(tool_result) => {
                existing_tool_result_ids.insert(tool_result.tool_call_id.clone());
                result.push(Message::ToolResult(tool_result));
            }
            Message::User(user) => {
                insert_missing_tool_results(
                    &mut result,
                    &mut pending_tool_calls,
                    &mut existing_tool_result_ids,
                );
                result.push(Message::User(user));
            }
        }
    }

    insert_missing_tool_results(
        &mut result,
        &mut pending_tool_calls,
        &mut existing_tool_result_ids,
    );
    result
}

fn insert_missing_tool_results(
    result: &mut Vec<Message>,
    pending_tool_calls: &mut Vec<ToolCall>,
    existing_tool_result_ids: &mut HashSet<String>,
) {
    for tool_call in pending_tool_calls.drain(..) {
        if existing_tool_result_ids.contains(&tool_call.id) {
            continue;
        }
        result.push(Message::ToolResult(ToolResultMessage {
            tool_call_id: tool_call.id,
            tool_name: tool_call.name,
            content: vec![ToolResultContent::text("No result provided")],
            details: None,
            is_error: true,
            timestamp: crate::types::now_millis(),
        }));
    }
    existing_tool_result_ids.clear();
}

fn replace_user_images_with_placeholder(
    blocks: &[UserContent],
    placeholder: &str,
) -> Vec<UserContent> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;
    for block in blocks {
        match block {
            UserContent::Image(ImageContent { .. }) => {
                if !previous_was_placeholder {
                    result.push(UserContent::Text(TextContent::new(placeholder)));
                }
                previous_was_placeholder = true;
            }
            UserContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(UserContent::Text(text.clone()));
            }
        }
    }
    result
}

fn replace_tool_images_with_placeholder(
    blocks: &[ToolResultContent],
    placeholder: &str,
) -> Vec<ToolResultContent> {
    let mut result = Vec::new();
    let mut previous_was_placeholder = false;
    for block in blocks {
        match block {
            ToolResultContent::Image(ImageContent { .. }) => {
                if !previous_was_placeholder {
                    result.push(ToolResultContent::Text(TextContent::new(placeholder)));
                }
                previous_was_placeholder = true;
            }
            ToolResultContent::Text(text) => {
                previous_was_placeholder = text.text == placeholder;
                result.push(ToolResultContent::Text(text.clone()));
            }
        }
    }
    result
}
