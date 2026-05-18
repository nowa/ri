use crate::{
    AssistantMessage, Context, Message, Model, ThinkingLevel, Tool, Usage,
    json_repair::parse_json_with_repair, openai_responses::parse_openai_responses_usage,
};
use chrono::{DateTime, Utc};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    io::ErrorKind,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
};

pub const DEFAULT_OPENAI_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
pub const OPENAI_CODEX_WEBSOCKET_BETA: &str = "responses_websockets=2026-02-06";
pub const OPENAI_CODEX_MAX_RETRIES: usize = 3;
pub const OPENAI_CODEX_BASE_RETRY_DELAY_MS: u64 = 1000;

const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const ACCOUNT_ID_ERROR: &str = "Failed to extract accountId from token";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAICodexResponsesPayloadOptions {
    pub session_id: Option<String>,
    pub temperature: Option<f64>,
    pub service_tier: Option<String>,
    pub text_verbosity: Option<String>,
    pub reasoning_effort: Option<ThinkingLevel>,
    pub reasoning_summary: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAICodexCachedWebSocketContinuation {
    pub last_request_body: Value,
    pub last_response_id: String,
    pub last_response_items: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAICodexCachedWebSocketRequestBody {
    pub body: Value,
    pub used_delta: bool,
    pub invalidated_continuation: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OpenAICodexWebSocketDebugStats {
    pub requests: u64,
    pub connections_created: u64,
    pub connections_reused: u64,
    pub cached_context_requests: u64,
    pub store_true_requests: u64,
    pub full_context_requests: u64,
    pub delta_requests: u64,
    pub last_input_items: usize,
    pub last_delta_input_items: Option<usize>,
    pub last_previous_response_id: Option<String>,
    pub websocket_failures: u64,
    pub sse_fallbacks: u64,
    pub websocket_fallback_active: Option<bool>,
    pub last_websocket_error: Option<String>,
}

pub fn build_openai_codex_responses_payload(
    model: &Model,
    context: &Context,
    options: OpenAICodexResponsesPayloadOptions,
) -> Value {
    let messages = crate::openai_responses::convert_openai_responses_messages(
        model,
        context,
        &["openai", "openai-codex", "opencode"],
        false,
    );
    let mut payload = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context.system_prompt.clone().unwrap_or_else(|| "You are a helpful assistant.".to_owned()),
        "input": messages,
        "text": { "verbosity": options.text_verbosity.unwrap_or_else(|| "low".to_owned()) },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if let Some(session_id) = options.session_id {
        payload["prompt_cache_key"] = Value::String(session_id);
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if let Some(service_tier) = options.service_tier {
        payload["service_tier"] = Value::String(service_tier);
    }
    if !context.tools.is_empty() {
        payload["tools"] =
            Value::Array(context.tools.iter().map(format_openai_codex_tool).collect());
    }
    if let Some(reasoning_effort) = options.reasoning_effort
        && let Some(effort) = openai_codex_reasoning_effort(model, reasoning_effort)
    {
        payload["reasoning"] = json!({
            "effort": effort,
            "summary": options.reasoning_summary.unwrap_or_else(|| "auto".to_owned()),
        });
    }

    payload
}

pub fn extract_openai_codex_account_id(token: &str) -> Result<String, String> {
    let mut parts = token.split('.');
    let (Some(_header), Some(payload), Some(_signature), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(ACCOUNT_ID_ERROR.to_owned());
    };

    let decoded = decode_base64_url(payload).map_err(|_| ACCOUNT_ID_ERROR.to_owned())?;
    let value: Value = serde_json::from_slice(&decoded).map_err(|_| ACCOUNT_ID_ERROR.to_owned())?;
    value
        .get(JWT_CLAIM_PATH)
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(Value::as_str)
        .filter(|account_id| !account_id.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| ACCOUNT_ID_ERROR.to_owned())
}

pub fn build_openai_codex_sse_headers(
    model_headers: &BTreeMap<String, String>,
    option_headers: &BTreeMap<String, String>,
    account_id: &str,
    token: &str,
    session_id: Option<&str>,
) -> BTreeMap<String, String> {
    let mut headers =
        build_openai_codex_base_headers(model_headers, option_headers, account_id, token);
    headers.insert(
        "OpenAI-Beta".to_owned(),
        "responses=experimental".to_owned(),
    );
    headers.insert("accept".to_owned(), "text/event-stream".to_owned());
    headers.insert("content-type".to_owned(), "application/json".to_owned());
    if let Some(session_id) = session_id {
        headers.insert("session_id".to_owned(), session_id.to_owned());
        headers.insert("x-client-request-id".to_owned(), session_id.to_owned());
    }
    headers
}

pub fn build_openai_codex_websocket_headers(
    model_headers: &BTreeMap<String, String>,
    option_headers: &BTreeMap<String, String>,
    account_id: &str,
    token: &str,
    request_id: &str,
) -> BTreeMap<String, String> {
    let mut headers =
        build_openai_codex_base_headers(model_headers, option_headers, account_id, token);
    headers.remove("accept");
    headers.remove("content-type");
    headers.remove("OpenAI-Beta");
    headers.remove("openai-beta");
    headers.insert(
        "OpenAI-Beta".to_owned(),
        OPENAI_CODEX_WEBSOCKET_BETA.to_owned(),
    );
    headers.insert("x-client-request-id".to_owned(), request_id.to_owned());
    headers.insert("session_id".to_owned(), request_id.to_owned());
    headers
}

pub fn resolve_openai_codex_url(base_url: Option<&str>) -> String {
    let raw = base_url
        .map(str::trim)
        .filter(|url| !url.is_empty())
        .unwrap_or(DEFAULT_OPENAI_CODEX_BASE_URL);
    let normalized = raw.trim_end_matches('/');
    if normalized.ends_with("/codex/responses") {
        normalized.to_owned()
    } else if normalized.ends_with("/codex") {
        format!("{normalized}/responses")
    } else {
        format!("{normalized}/codex/responses")
    }
}

pub fn resolve_openai_codex_websocket_url(base_url: Option<&str>) -> String {
    let url = resolve_openai_codex_url(base_url);
    if let Some(rest) = url.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = url.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        url
    }
}

pub fn resolve_openai_codex_service_tier(
    response_service_tier: Option<&str>,
    request_service_tier: Option<&str>,
) -> Option<String> {
    if response_service_tier == Some("default")
        && matches!(request_service_tier, Some("flex" | "priority"))
    {
        return request_service_tier.map(ToOwned::to_owned);
    }
    response_service_tier
        .or(request_service_tier)
        .map(ToOwned::to_owned)
}

pub fn parse_openai_codex_responses_usage(
    value: &Value,
    model: &Model,
    response_service_tier: Option<&str>,
    request_service_tier: Option<&str>,
) -> Usage {
    let service_tier =
        resolve_openai_codex_service_tier(response_service_tier, request_service_tier);
    parse_openai_responses_usage(value, model, service_tier.as_deref())
}

pub fn parse_openai_codex_sse_events(body: &str) -> Result<Vec<Value>, String> {
    let mut events = Vec::new();
    let mut data = Vec::new();
    let mut raw = Vec::new();

    for line in body.lines() {
        if line.is_empty() {
            flush_openai_codex_sse_event(&mut events, &mut data, &mut raw)?;
            continue;
        }

        raw.push(line.to_owned());
        if let Some(value) = line.strip_prefix("data:") {
            data.push(value.trim_start().to_owned());
        }
    }
    flush_openai_codex_sse_event(&mut events, &mut data, &mut raw)?;

    Ok(events)
}

trait AsyncReadWrite: AsyncRead + AsyncWrite + Unpin + Send {}

impl<T> AsyncReadWrite for T where T: AsyncRead + AsyncWrite + Unpin + Send {}

pub struct OpenAICodexWebSocket {
    stream: Box<dyn AsyncReadWrite>,
}

struct WebSocketFrame {
    fin: bool,
    opcode: u8,
    payload: Vec<u8>,
}

impl OpenAICodexWebSocket {
    pub async fn connect(url: &str, headers: &BTreeMap<String, String>) -> Result<Self, String> {
        let url = reqwest::Url::parse(url).map_err(|error| error.to_string())?;
        let scheme = url.scheme();
        if !matches!(scheme, "ws" | "wss") {
            return Err(format!("Unsupported WebSocket URL scheme: {scheme}"));
        }
        let host = url
            .host_str()
            .ok_or_else(|| "WebSocket URL is missing a host".to_owned())?;
        let port = url
            .port_or_known_default()
            .ok_or_else(|| "WebSocket URL is missing a port".to_owned())?;
        let tcp = TcpStream::connect((host, port))
            .await
            .map_err(|error| error.to_string())?;
        let stream: Box<dyn AsyncReadWrite> = if scheme == "wss" {
            let mut root_store = rustls::RootCertStore::empty();
            root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            let config = rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
            let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
            let server_name = rustls_pki_types::ServerName::try_from(host.to_owned())
                .map_err(|error| error.to_string())?;
            Box::new(
                connector
                    .connect(server_name, tcp)
                    .await
                    .map_err(|error| error.to_string())?,
            )
        } else {
            Box::new(tcp)
        };

        let mut socket = Self { stream };
        socket.handshake(&url, headers).await?;
        Ok(socket)
    }

    pub async fn send_json_text(&mut self, value: &Value) -> Result<(), String> {
        self.write_frame(0x1, value.to_string().as_bytes()).await
    }

    pub async fn read_json_text(&mut self) -> Result<Option<Value>, String> {
        let Some(text) = self.read_text().await? else {
            return Ok(None);
        };
        let value = parse_json_with_repair::<Value>(&text)
            .map_err(|error| format!("Invalid Codex WebSocket JSON: {error}; payload={text}"))?;
        Ok(Some(value))
    }

    pub async fn close(&mut self) -> Result<(), String> {
        self.write_frame(0x8, &[]).await
    }

    async fn handshake(
        &mut self,
        url: &reqwest::Url,
        headers: &BTreeMap<String, String>,
    ) -> Result<(), String> {
        let host = url.host_str().expect("validated host");
        let authority = match url.port() {
            Some(port) => format!("{host}:{port}"),
            None => host.to_owned(),
        };
        let path = if let Some(query) = url.query() {
            format!("{}?{}", url.path(), query)
        } else if url.path().is_empty() {
            "/".to_owned()
        } else {
            url.path().to_owned()
        };
        let key = websocket_key();
        let mut request = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: {authority}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Version: 13\r\n\
             Sec-WebSocket-Key: {key}\r\n"
        );
        for (name, value) in headers {
            let lower = name.to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "host"
                    | "upgrade"
                    | "connection"
                    | "sec-websocket-version"
                    | "sec-websocket-key"
                    | "openai-beta"
            ) {
                continue;
            }
            request.push_str(name);
            request.push_str(": ");
            request.push_str(value);
            request.push_str("\r\n");
        }
        request.push_str("\r\n");
        self.stream
            .write_all(request.as_bytes())
            .await
            .map_err(|error| error.to_string())?;

        let mut response = Vec::new();
        let mut byte = [0u8; 1];
        while !response.ends_with(b"\r\n\r\n") {
            self.stream
                .read_exact(&mut byte)
                .await
                .map_err(|error| error.to_string())?;
            response.push(byte[0]);
            if response.len() > 16 * 1024 {
                return Err("WebSocket handshake response exceeded 16 KiB".to_owned());
            }
        }
        let response = String::from_utf8_lossy(&response);
        if !response.starts_with("HTTP/1.1 101") && !response.starts_with("HTTP/1.0 101") {
            let status = response.lines().next().unwrap_or("HTTP response");
            return Err(format!("WebSocket handshake failed: {status}"));
        }
        Ok(())
    }

    async fn read_text(&mut self) -> Result<Option<String>, String> {
        let mut message = Vec::new();
        let mut reading_text = false;
        loop {
            let Some(frame) = self.read_frame().await? else {
                return if message.is_empty() {
                    Ok(None)
                } else {
                    Err("WebSocket closed during a fragmented text message".to_owned())
                };
            };
            match frame.opcode {
                0x0 if reading_text => {
                    message.extend(frame.payload);
                    if frame.fin {
                        return String::from_utf8(message)
                            .map(Some)
                            .map_err(|error| error.to_string());
                    }
                }
                0x1 => {
                    message.extend(frame.payload);
                    if frame.fin {
                        return String::from_utf8(message)
                            .map(Some)
                            .map_err(|error| error.to_string());
                    }
                    reading_text = true;
                }
                0x8 => return Ok(None),
                0x9 => self.write_frame(0xA, &frame.payload).await?,
                0xA => {}
                opcode => return Err(format!("Unsupported WebSocket opcode: {opcode}")),
            }
        }
    }

    async fn read_frame(&mut self) -> Result<Option<WebSocketFrame>, String> {
        let mut header = [0u8; 2];
        if let Err(error) = self.stream.read_exact(&mut header).await {
            if error.kind() == ErrorKind::UnexpectedEof {
                return Ok(None);
            }
            return Err(error.to_string());
        }
        let fin = header[0] & 0x80 != 0;
        let opcode = header[0] & 0x0f;
        let masked = header[1] & 0x80 != 0;
        let mut len = u64::from(header[1] & 0x7f);
        if len == 126 {
            let mut extended = [0u8; 2];
            self.stream
                .read_exact(&mut extended)
                .await
                .map_err(|error| error.to_string())?;
            len = u64::from(u16::from_be_bytes(extended));
        } else if len == 127 {
            let mut extended = [0u8; 8];
            self.stream
                .read_exact(&mut extended)
                .await
                .map_err(|error| error.to_string())?;
            len = u64::from_be_bytes(extended);
        }
        if len > 32 * 1024 * 1024 {
            return Err("WebSocket frame exceeded 32 MiB".to_owned());
        }
        let mask = if masked {
            let mut mask = [0u8; 4];
            self.stream
                .read_exact(&mut mask)
                .await
                .map_err(|error| error.to_string())?;
            Some(mask)
        } else {
            None
        };
        let mut payload = vec![0u8; len as usize];
        self.stream
            .read_exact(&mut payload)
            .await
            .map_err(|error| error.to_string())?;
        if let Some(mask) = mask {
            for (index, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[index % 4];
            }
        }
        Ok(Some(WebSocketFrame {
            fin,
            opcode,
            payload,
        }))
    }

    async fn write_frame(&mut self, opcode: u8, payload: &[u8]) -> Result<(), String> {
        let mut frame = Vec::with_capacity(payload.len() + 14);
        frame.push(0x80 | (opcode & 0x0f));
        if payload.len() <= 125 {
            frame.push(0x80 | payload.len() as u8);
        } else if payload.len() <= u16::MAX as usize {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        } else {
            frame.push(0x80 | 127);
            frame.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        }
        let mask = websocket_mask();
        frame.extend_from_slice(&mask);
        for (index, byte) in payload.iter().enumerate() {
            frame.push(byte ^ mask[index % 4]);
        }
        self.stream
            .write_all(&frame)
            .await
            .map_err(|error| error.to_string())?;
        self.stream.flush().await.map_err(|error| error.to_string())
    }
}

fn websocket_key() -> String {
    const KEY: &[u8; 16] = b"ri-codex-ws-key!";
    encode_base64(KEY)
}

fn websocket_mask() -> [u8; 4] {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    [
        nanos as u8,
        (nanos >> 8) as u8,
        (nanos >> 16) as u8,
        (nanos >> 24) as u8,
    ]
}

fn encode_base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut encoded = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        encoded.push(TABLE[(b0 >> 2) as usize] as char);
        encoded.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            encoded.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            encoded.push('=');
        }
        if chunk.len() > 2 {
            encoded.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            encoded.push('=');
        }
    }
    encoded
}

fn flush_openai_codex_sse_event(
    events: &mut Vec<Value>,
    data: &mut Vec<String>,
    raw: &mut Vec<String>,
) -> Result<(), String> {
    if data.is_empty() && raw.is_empty() {
        return Ok(());
    }

    let payload = data.join("\n");
    let raw_lines = std::mem::take(raw);
    data.clear();

    let trimmed = payload.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(());
    }

    let event = parse_json_with_repair::<Value>(&payload).map_err(|error| {
        format!(
            "Could not parse OpenAI Codex SSE event: {error}; data={payload}; raw={}",
            raw_lines.join("\\n")
        )
    })?;
    events.push(event);
    Ok(())
}

pub fn build_openai_codex_cached_websocket_continuation(
    model: &Model,
    last_request_body: Value,
    output: &AssistantMessage,
) -> Option<OpenAICodexCachedWebSocketContinuation> {
    let last_response_id = output.response_id.clone()?;
    Some(OpenAICodexCachedWebSocketContinuation {
        last_request_body,
        last_response_id,
        last_response_items: openai_codex_response_items_for_continuation(model, output),
    })
}

pub fn openai_codex_response_items_for_continuation(
    model: &Model,
    output: &AssistantMessage,
) -> Vec<Value> {
    crate::openai_responses::convert_openai_responses_messages(
        model,
        &Context {
            messages: vec![Message::Assistant(output.clone())],
            ..Default::default()
        },
        &["openai", "openai-codex", "opencode"],
        false,
    )
    .into_iter()
    .filter(|item| item.get("type").and_then(Value::as_str) != Some("function_call_output"))
    .collect()
}

pub fn openai_codex_cached_websocket_input_delta(
    body: &Value,
    continuation: &OpenAICodexCachedWebSocketContinuation,
) -> Option<Vec<Value>> {
    if request_body_without_cached_input(body)
        != request_body_without_cached_input(&continuation.last_request_body)
    {
        return None;
    }

    let current_input = request_body_input(body);
    let mut baseline = request_body_input(&continuation.last_request_body);
    baseline.extend(continuation.last_response_items.clone());
    if current_input.len() < baseline.len() {
        return None;
    }
    if current_input[..baseline.len()] != baseline[..] {
        return None;
    }
    Some(current_input[baseline.len()..].to_vec())
}

pub fn build_openai_codex_cached_websocket_request_body(
    body: &Value,
    continuation: Option<&OpenAICodexCachedWebSocketContinuation>,
) -> OpenAICodexCachedWebSocketRequestBody {
    let Some(continuation) = continuation else {
        return OpenAICodexCachedWebSocketRequestBody {
            body: body.clone(),
            used_delta: false,
            invalidated_continuation: false,
        };
    };
    let Some(delta) = openai_codex_cached_websocket_input_delta(body, continuation) else {
        return OpenAICodexCachedWebSocketRequestBody {
            body: body.clone(),
            used_delta: false,
            invalidated_continuation: true,
        };
    };
    if continuation.last_response_id.is_empty() {
        return OpenAICodexCachedWebSocketRequestBody {
            body: body.clone(),
            used_delta: false,
            invalidated_continuation: true,
        };
    }

    let mut request = body.clone();
    if let Some(object) = request.as_object_mut() {
        object.insert(
            "previous_response_id".to_owned(),
            Value::String(continuation.last_response_id.clone()),
        );
        object.insert("input".to_owned(), Value::Array(delta));
    }
    OpenAICodexCachedWebSocketRequestBody {
        body: request,
        used_delta: true,
        invalidated_continuation: false,
    }
}

pub fn record_openai_codex_websocket_request_stats(
    stats: &mut OpenAICodexWebSocketDebugStats,
    request_body: &Value,
    reused_connection: bool,
    use_cached_context: bool,
) {
    stats.requests += 1;
    if reused_connection {
        stats.connections_reused += 1;
    } else {
        stats.connections_created += 1;
    }
    if use_cached_context {
        stats.cached_context_requests += 1;
    }
    if request_body.get("store").and_then(Value::as_bool) == Some(true) {
        stats.store_true_requests += 1;
    }

    let input_items = request_body_input(request_body).len();
    stats.last_input_items = input_items;
    if let Some(previous_response_id) = request_body
        .get("previous_response_id")
        .and_then(Value::as_str)
    {
        stats.delta_requests += 1;
        stats.last_delta_input_items = Some(input_items);
        stats.last_previous_response_id = Some(previous_response_id.to_owned());
    } else {
        stats.full_context_requests += 1;
        stats.last_delta_input_items = None;
        stats.last_previous_response_id = None;
    }
}

pub fn record_openai_codex_websocket_failure(
    stats: &mut OpenAICodexWebSocketDebugStats,
    error: impl ToString,
) {
    stats.websocket_failures += 1;
    stats.last_websocket_error = Some(error.to_string());
    stats.websocket_fallback_active = Some(true);
}

pub fn record_openai_codex_websocket_sse_fallback(
    stats: &mut OpenAICodexWebSocketDebugStats,
    fallback_active: bool,
) {
    stats.sse_fallbacks += 1;
    stats.websocket_fallback_active = Some(fallback_active);
}

pub fn is_openai_codex_retryable_error(status: u16, error_text: &str) -> bool {
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    let normalized = error_text
        .chars()
        .flat_map(char::to_lowercase)
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    normalized.contains("ratelimit")
        || normalized.contains("overloaded")
        || normalized.contains("serviceunavailable")
        || normalized.contains("upstreamconnect")
        || normalized.contains("connectionrefused")
}

pub fn openai_codex_retry_delay_ms(
    status: u16,
    error_text: &str,
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    attempt: usize,
    now_ms: i64,
) -> Option<u64> {
    if attempt >= OPENAI_CODEX_MAX_RETRIES || !is_openai_codex_retryable_error(status, error_text) {
        return None;
    }

    let attempt = u32::try_from(attempt).unwrap_or(u32::MAX);
    let mut delay_ms =
        OPENAI_CODEX_BASE_RETRY_DELAY_MS.saturating_mul(2_u64.saturating_pow(attempt));
    if let Some(retry_after_ms) = retry_after_ms {
        if let Ok(millis) = retry_after_ms.parse::<f64>()
            && millis.is_finite()
        {
            delay_ms = millis.max(0.0) as u64;
        }
    } else if let Some(retry_after) = retry_after {
        if let Ok(seconds) = retry_after.parse::<f64>()
            && seconds.is_finite()
        {
            delay_ms = (seconds.max(0.0) * 1000.0) as u64;
        } else if let Ok(date) = DateTime::parse_from_rfc2822(retry_after) {
            delay_ms = date
                .with_timezone(&Utc)
                .timestamp_millis()
                .saturating_sub(now_ms)
                .max(0) as u64;
        }
    }

    Some(delay_ms)
}

fn build_openai_codex_base_headers(
    model_headers: &BTreeMap<String, String>,
    option_headers: &BTreeMap<String, String>,
    account_id: &str,
    token: &str,
) -> BTreeMap<String, String> {
    let mut headers = model_headers.clone();
    headers.extend(option_headers.clone());
    headers.insert("Authorization".to_owned(), format!("Bearer {token}"));
    headers.insert("chatgpt-account-id".to_owned(), account_id.to_owned());
    headers.insert("originator".to_owned(), "pi".to_owned());
    headers.insert("User-Agent".to_owned(), "pi (browser)".to_owned());
    headers
}

fn format_openai_codex_tool(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": Value::Null,
    })
}

fn openai_codex_reasoning_effort(model: &Model, level: ThinkingLevel) -> Option<String> {
    match model.thinking_level_map.get(&level) {
        Some(Some(effort)) => Some(effort.clone()),
        Some(None) => None,
        None => Some(
            match level {
                ThinkingLevel::Off => "none",
                ThinkingLevel::Minimal => "minimal",
                ThinkingLevel::Low => "low",
                ThinkingLevel::Medium => "medium",
                ThinkingLevel::High => "high",
                ThinkingLevel::XHigh => "xhigh",
            }
            .to_owned(),
        ),
    }
}

fn request_body_without_cached_input(body: &Value) -> Value {
    let Some(object) = body.as_object() else {
        return body.clone();
    };
    let mut object = object.clone();
    object.remove("input");
    object.remove("previous_response_id");
    Value::Object(object)
}

fn request_body_input(body: &Value) -> Vec<Value> {
    body.get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn decode_base64_url(input: &str) -> Result<Vec<u8>, ()> {
    let mut bytes = Vec::new();
    let mut buffer = 0u32;
    let mut bits = 0u8;

    for byte in input.bytes() {
        if byte == b'=' {
            break;
        }
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            _ => return Err(()),
        } as u32;

        buffer = (buffer << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            bytes.push(((buffer >> bits) & 0xff) as u8);
            buffer &= (1 << bits) - 1;
        }
    }

    Ok(bytes)
}
