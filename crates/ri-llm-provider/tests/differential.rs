//! Differential payload dump against pi.
//!
//! Reads a neutral case matrix (RI_DIFF_CASES), replays each case through the
//! full `complete_simple` pipeline with a capture-then-abort payload hook —
//! the same seam pi's dumper uses via `onPayload` — and writes the captured
//! payloads (or errors) to RI_DIFF_OUT for comparison with pi's output.
//!
//! Skips silently when RI_DIFF_CASES is unset so normal test runs are
//! unaffected.

use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::sync::{Arc, Mutex};

const TS: i64 = 1000;

#[derive(Debug)]
struct CaptureThenAbortHook {
    captured: Arc<Mutex<Option<Value>>>,
}

impl ProviderPayloadHook for CaptureThenAbortHook {
    fn on_payload(&self, _model: &Model, payload: Value) -> Result<Value, String> {
        let mut captured = self
            .captured
            .lock()
            .map_err(|_| "capture lock poisoned".to_owned())?;
        *captured = Some(payload);
        Err("diff payload captured".to_owned())
    }
}

#[tokio::test]
async fn differential_payload_dump() {
    let Ok(cases_path) = std::env::var("RI_DIFF_CASES") else {
        return;
    };
    let out_path = std::env::var("RI_DIFF_OUT").expect("RI_DIFF_OUT required with RI_DIFF_CASES");
    let spec: Value =
        serde_json::from_str(&std::fs::read_to_string(&cases_path).expect("read RI_DIFF_CASES"))
            .expect("parse RI_DIFF_CASES");

    let mut results = Map::new();
    for case in spec["cases"].as_array().expect("cases array") {
        let id = case["id"].as_str().expect("case id").to_owned();
        let value = match run_case(case).await {
            Ok(payload) => json!({ "payload": payload }),
            Err(error) => json!({ "error": error }),
        };
        results.insert(id, value);
    }
    std::fs::write(
        &out_path,
        serde_json::to_string_pretty(&Value::Object(results)).expect("serialize results"),
    )
    .expect("write RI_DIFF_OUT");
}

async fn run_case(case: &Value) -> Result<Value, String> {
    let provider = case["provider"].as_str().expect("provider");
    let model_id = case["model"].as_str().expect("model");
    let mut model = get_model(provider, model_id)
        .ok_or_else(|| format!("model not found: {provider}/{model_id}"))?;
    if let Some(api) = case.get("apiOverride").and_then(Value::as_str) {
        model.api = api.to_owned();
    }
    model.base_url = "http://127.0.0.1:9".to_owned();

    let context: Context = serde_json::from_value(map_context(case)?)
        .map_err(|error| format!("context mapping: {error}"))?;

    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("fake-key".to_owned());
    if let Some(opts) = case.get("options") {
        if let Some(reasoning) = opts.get("reasoning") {
            options.reasoning = Some(
                serde_json::from_value(reasoning.clone())
                    .map_err(|error| format!("reasoning: {error}"))?,
            );
        }
        if let Some(temperature) = opts.get("temperature").and_then(Value::as_f64) {
            options.stream.temperature = Some(temperature);
        }
        if let Some(max_tokens) = opts.get("maxTokens").and_then(Value::as_u64) {
            options.stream.max_tokens = Some(max_tokens);
        }
        if let Some(session_id) = opts.get("sessionId").and_then(Value::as_str) {
            options.stream.session_id = Some(session_id.to_owned());
        }
    }

    let captured = Arc::new(Mutex::new(None));
    options.payload_hooks.push(Arc::new(CaptureThenAbortHook {
        captured: captured.clone(),
    }));

    let _ = complete_simple(&model, context, options).await;

    let payload = captured
        .lock()
        .map_err(|_| "capture lock poisoned".to_owned())?
        .take();
    payload.ok_or_else(|| "no payload captured".to_owned())
}

/// Builds the case context in the shared wire shape. This mirrors
/// scratchpad/diff/pi-dump.ts `mapContext` exactly — keep the two in sync.
fn map_context(case: &Value) -> Result<Value, String> {
    let ctx = &case["context"];
    let provider = case["provider"].as_str().expect("provider");
    let model_id = case["model"].as_str().expect("model");
    let api_override = case.get("apiOverride").and_then(Value::as_str);

    let mut messages = Vec::new();
    for message in ctx["messages"].as_array().expect("messages") {
        let role = message["role"].as_str().expect("role");
        let mapped = match role {
            "user" => {
                if let Some(blocks) = message.get("blocks").and_then(Value::as_array) {
                    let content = blocks
                        .iter()
                        .map(|block| match block["type"].as_str() {
                            Some("text") => json!({ "type": "text", "text": block["text"] }),
                            Some("image") => json!({
                                "type": "image",
                                "data": block["data"],
                                "mimeType": block["mimeType"],
                            }),
                            other => panic!("unknown block type {other:?}"),
                        })
                        .collect::<Vec<_>>();
                    json!({ "role": "user", "content": content, "timestamp": TS })
                } else {
                    json!({ "role": "user", "content": message["text"], "timestamp": TS })
                }
            }
            "assistant" => {
                let from = message.get("from");
                let (from_provider, from_model) = match from {
                    Some(from) => (
                        from["provider"].as_str().expect("from provider"),
                        from["model"].as_str().expect("from model"),
                    ),
                    None => (provider, model_id),
                };
                let base = get_model(from_provider, from_model)
                    .ok_or_else(|| format!("from model not found: {from_provider}/{from_model}"))?;
                let api = match from {
                    Some(from) => from["api"].as_str().expect("from api").to_owned(),
                    None => api_override.map(str::to_owned).unwrap_or(base.api),
                };
                let mut content = Vec::new();
                if let Some(thinking) = message.get("thinking") {
                    let mut block = json!({ "type": "thinking", "thinking": thinking });
                    if let Some(signature) = message.get("thinkingSignature") {
                        block["thinkingSignature"] = signature.clone();
                    }
                    content.push(block);
                }
                if let Some(text) = message.get("text") {
                    content.push(json!({ "type": "text", "text": text }));
                }
                let tool_calls = message
                    .get("toolCalls")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for tool_call in &tool_calls {
                    content.push(json!({
                        "type": "toolCall",
                        "id": tool_call["id"],
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }));
                }
                json!({
                    "role": "assistant",
                    "content": content,
                    "api": api,
                    "provider": from_provider,
                    "model": from_model,
                    "usage": {
                        "input": 0,
                        "output": 0,
                        "cacheRead": 0,
                        "cacheWrite": 0,
                        "totalTokens": 0,
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "total": 0,
                        },
                    },
                    "stopReason": if tool_calls.is_empty() { "stop" } else { "toolUse" },
                    "timestamp": TS,
                })
            }
            "toolResult" => json!({
                "role": "toolResult",
                "toolCallId": message["toolCallId"],
                "toolName": message["toolName"],
                "content": [{ "type": "text", "text": message["text"] }],
                "isError": message.get("isError").and_then(Value::as_bool).unwrap_or(false),
                "timestamp": TS,
            }),
            other => return Err(format!("unknown role {other}")),
        };
        messages.push(mapped);
    }

    let mut mapped = json!({ "messages": messages });
    if let Some(system) = ctx.get("system") {
        mapped["systemPrompt"] = system.clone();
    }
    if let Some(tools) = ctx.get("tools") {
        mapped["tools"] = tools.clone();
    }
    Ok(mapped)
}
