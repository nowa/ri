//! Message-anchored (deferred) tool loading, mirroring pi
//! `utils/deferred-tools.ts`.

use crate::{AssistantContent, Context, Message, Tool};
use std::collections::BTreeSet;

/// Current tools split into prompt-prefix definitions and transcript-loaded
/// definitions. `deferred` preserves tool order and is keyed by the
/// normalized name.
#[derive(Debug, Clone, Default)]
pub struct DeferredToolPlacement {
    pub immediate: Vec<Tool>,
    pub deferred: Vec<(String, Tool)>,
}

impl DeferredToolPlacement {
    pub fn deferred_tool(&self, normalized_name: &str) -> Option<&Tool> {
        self.deferred
            .iter()
            .find(|(name, _)| name == normalized_name)
            .map(|(_, tool)| tool)
    }

    pub fn deferred_names(&self) -> BTreeSet<String> {
        self.deferred.iter().map(|(name, _)| name.clone()).collect()
    }
}

/// Split current tools into prefix and transcript-loaded definitions.
pub fn split_deferred_tools(
    context: &Context,
    enabled: bool,
    normalize_name: impl Fn(&str) -> String,
) -> DeferredToolPlacement {
    let mut unique_tools: Vec<(String, Tool)> = Vec::new();
    for tool in &context.tools {
        let name = normalize_name(&tool.name);
        if let Some(existing) = unique_tools
            .iter_mut()
            .find(|(existing, _)| *existing == name)
        {
            existing.1 = tool.clone();
        } else {
            unique_tools.push((name, tool.clone()));
        }
    }
    if !enabled {
        return DeferredToolPlacement {
            immediate: unique_tools.into_iter().map(|(_, tool)| tool).collect(),
            deferred: Vec::new(),
        };
    }

    let mut deferred_names = BTreeSet::new();
    let mut used_names = BTreeSet::new();
    for message in &context.messages {
        match message {
            Message::Assistant(assistant) => {
                for block in &assistant.content {
                    if let AssistantContent::ToolCall(tool_call) = block {
                        used_names.insert(normalize_name(&tool_call.name));
                    }
                }
            }
            Message::ToolResult(tool_result) => {
                for name in tool_result.added_tool_names.as_deref().unwrap_or_default() {
                    let normalized_name = normalize_name(name);
                    if !used_names.contains(&normalized_name) {
                        deferred_names.insert(normalized_name);
                    }
                }
            }
            Message::User(_) => {}
        }
    }

    let mut placement = DeferredToolPlacement::default();
    for (name, tool) in unique_tools {
        if deferred_names.contains(&name) {
            placement.deferred.push((name, tool));
        } else {
            placement.immediate.push(tool);
        }
    }
    placement
}
