use crate::types::{Tool, ToolCall};
use serde_json::{Map, Value};
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum ValidationError {
    #[error("Tool \"{0}\" not found")]
    ToolNotFound(String),
    #[error("Validation failed for tool \"{tool}\":\n{details}\n\nReceived arguments:\n{received}")]
    InvalidArguments {
        tool: String,
        details: String,
        received: String,
    },
}

pub fn validate_tool_call(tools: &[Tool], tool_call: &ToolCall) -> Result<Value, ValidationError> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == tool_call.name)
        .ok_or_else(|| ValidationError::ToolNotFound(tool_call.name.clone()))?;
    validate_tool_arguments(tool, tool_call)
}

pub fn validate_tool_arguments(
    tool: &Tool,
    tool_call: &ToolCall,
) -> Result<Value, ValidationError> {
    let mut args = Value::Object(tool_call.arguments.clone());
    coerce_with_schema(&mut args, &tool.parameters);
    let mut errors = Vec::new();
    validate_value(&args, &tool.parameters, "root", &mut errors);
    if errors.is_empty() {
        Ok(args)
    } else {
        Err(ValidationError::InvalidArguments {
            tool: tool_call.name.clone(),
            details: errors
                .into_iter()
                .map(|error| format!("  - {error}"))
                .collect::<Vec<_>>()
                .join("\n"),
            received: serde_json::to_string_pretty(&Value::Object(tool_call.arguments.clone()))
                .unwrap_or_else(|_| "{}".to_owned()),
        })
    }
}

fn validate_value(value: &Value, schema: &Value, path: &str, errors: &mut Vec<String>) {
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        if let Some(object) = value.as_object() {
            for required_key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(required_key) {
                    errors.push(format!(
                        "{path}.{required_key}: required property is missing"
                    ));
                }
            }
        }
    }

    if let Some(schema_type) = schema.get("type") {
        if !matches_schema_type(value, schema_type) {
            errors.push(format!(
                "{path}: expected {}",
                render_schema_type(schema_type)
            ));
            return;
        }
    }

    if let (Some(object), Some(properties)) = (
        value.as_object(),
        schema.get("properties").and_then(Value::as_object),
    ) {
        for (key, property_schema) in properties {
            if let Some(property_value) = object.get(key) {
                validate_value(
                    property_value,
                    property_schema,
                    &format_path(path, key),
                    errors,
                );
            }
        }
    }

    if let (Some(array), Some(items)) = (value.as_array(), schema.get("items")) {
        for (index, item) in array.iter().enumerate() {
            validate_value(item, items, &format!("{path}.{index}"), errors);
        }
    }

    for key in ["anyOf", "oneOf"] {
        if let Some(candidates) = schema.get(key).and_then(Value::as_array) {
            if !candidates.iter().any(|candidate| {
                let mut nested = Vec::new();
                validate_value(value, candidate, path, &mut nested);
                nested.is_empty()
            }) {
                errors.push(format!("{path}: did not match any {key} schema"));
            }
        }
    }
}

fn format_path(base: &str, key: &str) -> String {
    if base == "root" {
        key.to_owned()
    } else {
        format!("{base}.{key}")
    }
}

fn matches_schema_type(value: &Value, schema_type: &Value) -> bool {
    match schema_type {
        Value::String(schema_type) => matches_one_type(value, schema_type),
        Value::Array(types) => types
            .iter()
            .filter_map(Value::as_str)
            .any(|schema_type| matches_one_type(value, schema_type)),
        _ => true,
    }
}

fn matches_one_type(value: &Value, schema_type: &str) -> bool {
    match schema_type {
        "number" => value.as_f64().is_some(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.as_bool().is_some(),
        "string" => value.as_str().is_some(),
        "null" => value.is_null(),
        "array" => value.as_array().is_some(),
        "object" => value.as_object().is_some(),
        _ => true,
    }
}

fn render_schema_type(schema_type: &Value) -> String {
    match schema_type {
        Value::String(schema_type) => schema_type.clone(),
        Value::Array(types) => types
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" | "),
        _ => "valid JSON schema type".to_owned(),
    }
}

fn coerce_with_schema(value: &mut Value, schema: &Value) {
    if let Some(all_of) = schema.get("allOf").and_then(Value::as_array) {
        for nested in all_of {
            coerce_with_schema(value, nested);
        }
    }

    for key in ["anyOf", "oneOf"] {
        if let Some(candidates) = schema.get(key).and_then(Value::as_array) {
            let original = value.clone();
            for candidate in candidates {
                let mut trial = original.clone();
                coerce_with_schema(&mut trial, candidate);
                let mut errors = Vec::new();
                validate_value(&trial, candidate, "root", &mut errors);
                if errors.is_empty() {
                    *value = trial;
                    break;
                }
            }
        }
    }

    let Some(schema_type) = schema.get("type") else {
        return;
    };

    if !matches_schema_type(value, schema_type) {
        if let Some(types) = schema_types(schema_type) {
            for schema_type in types {
                if coerce_primitive(value, schema_type) {
                    break;
                }
            }
        }
    }

    if schema_allows_type(schema_type, "object") {
        if let Some(object) = value.as_object_mut() {
            coerce_object(object, schema);
        }
    }

    if schema_allows_type(schema_type, "array") {
        if let Some(array) = value.as_array_mut() {
            if let Some(items) = schema.get("items") {
                for item in array {
                    coerce_with_schema(item, items);
                }
            }
        }
    }
}

fn coerce_object(object: &mut Map<String, Value>, schema: &Value) {
    let defined_keys: std::collections::BTreeSet<String> = schema
        .get("properties")
        .and_then(Value::as_object)
        .map(|properties| properties.keys().cloned().collect())
        .unwrap_or_default();

    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        for (key, property_schema) in properties {
            if let Some(property_value) = object.get_mut(key) {
                coerce_with_schema(property_value, property_schema);
            }
        }
    }

    if let Some(additional_schema) = schema
        .get("additionalProperties")
        .filter(|value| value.is_object())
    {
        for (key, property_value) in object.iter_mut() {
            if !defined_keys.contains(key) {
                coerce_with_schema(property_value, additional_schema);
            }
        }
    }
}

fn schema_types(schema_type: &Value) -> Option<Vec<&str>> {
    match schema_type {
        Value::String(schema_type) => Some(vec![schema_type.as_str()]),
        Value::Array(types) => Some(types.iter().filter_map(Value::as_str).collect()),
        _ => None,
    }
}

fn schema_allows_type(schema_type: &Value, expected: &str) -> bool {
    schema_types(schema_type)
        .map(|types| types.contains(&expected))
        .unwrap_or(false)
}

fn coerce_primitive(value: &mut Value, schema_type: &str) -> bool {
    let next = match schema_type {
        "number" => match value {
            Value::Null => Some(Value::from(0.0)),
            Value::String(text) => text
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|number| number.is_finite())
                .map(Value::from),
            Value::Bool(flag) => Some(Value::from(if *flag { 1.0 } else { 0.0 })),
            _ => None,
        },
        "integer" => match value {
            Value::Null => Some(Value::from(0)),
            Value::String(text) => text.trim().parse::<i64>().ok().map(Value::from),
            Value::Bool(flag) => Some(Value::from(if *flag { 1 } else { 0 })),
            _ => None,
        },
        "boolean" => match value {
            Value::Null => Some(Value::Bool(false)),
            Value::String(text) if text == "true" => Some(Value::Bool(true)),
            Value::String(text) if text == "false" => Some(Value::Bool(false)),
            Value::Number(number) if number.as_i64() == Some(1) => Some(Value::Bool(true)),
            Value::Number(number) if number.as_i64() == Some(0) => Some(Value::Bool(false)),
            _ => None,
        },
        "string" => match value {
            Value::Null => Some(Value::String(String::new())),
            Value::Bool(flag) => Some(Value::String(flag.to_string())),
            Value::Number(number) => Some(Value::String(number.to_string())),
            _ => None,
        },
        "null" => match value {
            Value::String(text) if text.is_empty() => Some(Value::Null),
            Value::Number(number) if number.as_i64() == Some(0) => Some(Value::Null),
            Value::Bool(false) => Some(Value::Null),
            _ => None,
        },
        _ => None,
    };

    if let Some(next) = next {
        *value = next;
        true
    } else {
        false
    }
}
