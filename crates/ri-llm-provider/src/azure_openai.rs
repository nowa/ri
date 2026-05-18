use crate::{
    Context, Model, ThinkingLevel, convert_openai_responses_messages,
    convert_openai_responses_tools,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const DEFAULT_AZURE_OPENAI_API_VERSION: &str = "v1";
const AZURE_OPENAI_TOOL_CALL_PROVIDERS: &[&str] = &[
    "openai",
    "openai-codex",
    "opencode",
    "azure-openai-responses",
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AzureOpenAIConfigOptions {
    pub azure_base_url: Option<String>,
    pub azure_resource_name: Option<String>,
    pub azure_api_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AzureOpenAIConfig {
    pub base_url: String,
    pub api_version: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct AzureOpenAIResponsesPayloadOptions {
    pub session_id: Option<String>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub reasoning_effort: Option<ThinkingLevel>,
    pub reasoning_summary: Option<String>,
    pub azure_deployment_name: Option<String>,
}

pub fn parse_azure_openai_deployment_name_map(value: Option<&str>) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    let Some(value) = value else {
        return map;
    };
    for entry in value.split(',') {
        let trimmed = entry.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some((model_id, deployment_name)) = trimmed.split_once('=') else {
            continue;
        };
        let model_id = model_id.trim();
        let deployment_name = deployment_name.trim();
        if model_id.is_empty() || deployment_name.is_empty() {
            continue;
        }
        map.insert(model_id.to_owned(), deployment_name.to_owned());
    }
    map
}

pub fn resolve_azure_openai_deployment_name(
    model: &Model,
    azure_deployment_name: Option<&str>,
) -> String {
    if let Some(deployment_name) = azure_deployment_name.filter(|value| !value.is_empty()) {
        return deployment_name.to_owned();
    }
    let mapped = std::env::var("AZURE_OPENAI_DEPLOYMENT_NAME_MAP")
        .ok()
        .and_then(|value| {
            parse_azure_openai_deployment_name_map(Some(&value))
                .get(&model.id)
                .cloned()
        });
    mapped.unwrap_or_else(|| model.id.clone())
}

pub fn build_azure_openai_responses_payload(
    model: &Model,
    context: &Context,
    options: AzureOpenAIResponsesPayloadOptions,
) -> Value {
    let deployment_name =
        resolve_azure_openai_deployment_name(model, options.azure_deployment_name.as_deref());
    let messages =
        convert_openai_responses_messages(model, context, AZURE_OPENAI_TOOL_CALL_PROVIDERS, true);
    let mut payload = json!({
        "model": deployment_name,
        "input": messages,
        "stream": true,
    });

    if let Some(session_id) = options.session_id {
        payload["prompt_cache_key"] = Value::String(session_id);
    }
    if let Some(max_tokens) = options.max_tokens {
        payload["max_output_tokens"] = Value::Number(max_tokens.into());
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if !context.tools.is_empty() {
        payload["tools"] = Value::Array(convert_openai_responses_tools(&context.tools, None));
    }
    if model.reasoning {
        if options.reasoning_effort.is_some() || options.reasoning_summary.is_some() {
            let effort = options
                .reasoning_effort
                .map(|level| azure_openai_reasoning_effort(model, level))
                .unwrap_or_else(|| "medium".to_owned());
            payload["reasoning"] = json!({
                "effort": effort,
                "summary": options.reasoning_summary.unwrap_or_else(|| "auto".to_owned()),
            });
            payload["include"] = json!(["reasoning.encrypted_content"]);
        } else if model.thinking_level_map.get(&ThinkingLevel::Off) != Some(&None) {
            let effort = model
                .thinking_level_map
                .get(&ThinkingLevel::Off)
                .and_then(Clone::clone)
                .unwrap_or_else(|| "none".to_owned());
            payload["reasoning"] = json!({ "effort": effort });
        }
    }

    payload
}

fn azure_openai_reasoning_effort(model: &Model, level: ThinkingLevel) -> String {
    if let Some(Some(mapped)) = model.thinking_level_map.get(&level) {
        return mapped.clone();
    }
    match level {
        ThinkingLevel::Off => "off",
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        ThinkingLevel::XHigh => "xhigh",
    }
    .to_owned()
}

pub fn normalize_azure_openai_base_url(base_url: &str) -> Result<String, String> {
    let trimmed = base_url.trim().trim_end_matches('/');
    let Some((scheme, rest)) = trimmed.split_once("://") else {
        return Err(format!("Invalid Azure OpenAI base URL: {base_url}"));
    };
    if scheme.is_empty() || rest.is_empty() {
        return Err(format!("Invalid Azure OpenAI base URL: {base_url}"));
    }

    let (host, tail) = split_host_and_tail(rest);
    if host.is_empty() {
        return Err(format!("Invalid Azure OpenAI base URL: {base_url}"));
    }

    let is_azure_host =
        host.ends_with(".openai.azure.com") || host.ends_with(".cognitiveservices.azure.com");
    let path = tail
        .split_once(['?', '#'])
        .map(|(path, _)| path)
        .unwrap_or(tail)
        .trim_end_matches('/');

    if is_azure_host && (path.is_empty() || path == "/" || path == "/openai") {
        return Ok(format!("{scheme}://{host}/openai/v1"));
    }

    Ok(trimmed.to_owned())
}

pub fn build_default_azure_openai_base_url(resource_name: &str) -> String {
    format!("https://{resource_name}.openai.azure.com/openai/v1")
}

pub fn resolve_azure_openai_config(
    model: &Model,
    options: AzureOpenAIConfigOptions,
) -> Result<AzureOpenAIConfig, String> {
    let api_version = options
        .azure_api_version
        .or_else(|| std::env::var("AZURE_OPENAI_API_VERSION").ok())
        .unwrap_or_else(|| DEFAULT_AZURE_OPENAI_API_VERSION.to_owned());

    let base_url = options
        .azure_base_url
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::env::var("AZURE_OPENAI_BASE_URL")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
        .or_else(|| {
            options
                .azure_resource_name
                .or_else(|| std::env::var("AZURE_OPENAI_RESOURCE_NAME").ok())
                .map(|resource| build_default_azure_openai_base_url(&resource))
        })
        .or_else(|| (!model.base_url.is_empty()).then(|| model.base_url.clone()))
        .ok_or_else(|| {
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, or pass azureBaseUrl, azureResourceName, or model.baseUrl.".to_owned()
        })?;

    Ok(AzureOpenAIConfig {
        base_url: normalize_azure_openai_base_url(&base_url)?,
        api_version,
    })
}

fn split_host_and_tail(rest: &str) -> (&str, &str) {
    let end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let host = &rest[..end];
    let tail = if end < rest.len() && rest.as_bytes()[end] == b'/' {
        &rest[end..]
    } else {
        ""
    };
    (host, tail)
}
