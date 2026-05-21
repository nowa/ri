use crate::{AssistantMessage, now_millis};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiagnosticErrorInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stack: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<Value>,
}

impl DiagnosticErrorInfo {
    pub fn thrown_value(message: impl Into<String>) -> Self {
        Self {
            name: Some("ThrownValue".to_owned()),
            message: message.into(),
            stack: None,
            code: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantMessageDiagnostic {
    #[serde(rename = "type")]
    pub diagnostic_type: String,
    pub timestamp: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<DiagnosticErrorInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

pub fn format_thrown_value(value: impl ToString) -> String {
    value.to_string()
}

pub fn create_assistant_message_diagnostic(
    diagnostic_type: impl Into<String>,
    error: impl ToString,
    details: Option<Value>,
) -> AssistantMessageDiagnostic {
    AssistantMessageDiagnostic {
        diagnostic_type: diagnostic_type.into(),
        timestamp: now_millis(),
        error: Some(DiagnosticErrorInfo::thrown_value(format_thrown_value(
            error,
        ))),
        details,
    }
}

pub fn append_assistant_message_diagnostic(
    message: &mut AssistantMessage,
    diagnostic: AssistantMessageDiagnostic,
) {
    message
        .diagnostics
        .push(serde_json::to_value(diagnostic).expect("diagnostic serializes"));
}
