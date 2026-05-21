use crate::{Context, Message, ToolResultContent, UserContent, UserContentValue};
use std::collections::BTreeMap;

pub fn infer_copilot_initiator(messages: &[Message]) -> &'static str {
    match messages.last() {
        Some(Message::User(_)) | None => "user",
        Some(_) => "agent",
    }
}

pub fn has_copilot_vision_input(context: &Context) -> bool {
    context.messages.iter().any(|message| match message {
        Message::User(user) => match &user.content {
            UserContentValue::Blocks(blocks) => blocks
                .iter()
                .any(|block| matches!(block, UserContent::Image(_))),
            UserContentValue::Plain(_) => false,
        },
        Message::ToolResult(result) => result
            .content
            .iter()
            .any(|block| matches!(block, ToolResultContent::Image(_))),
        Message::Assistant(_) => false,
    })
}

pub fn build_copilot_dynamic_headers(context: &Context) -> BTreeMap<String, String> {
    let mut headers = BTreeMap::from([
        (
            "X-Initiator".to_owned(),
            infer_copilot_initiator(&context.messages).to_owned(),
        ),
        ("Openai-Intent".to_owned(), "conversation-edits".to_owned()),
    ]);
    if has_copilot_vision_input(context) {
        headers.insert("Copilot-Vision-Request".to_owned(), "true".to_owned());
    }
    headers
}
