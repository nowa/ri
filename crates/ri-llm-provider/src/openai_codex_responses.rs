use crate::{
    AssistantMessage, Context, Message, Model, ThinkingLevel, Tool, Usage,
    json_repair::parse_json_with_repair,
    node_http_proxy::resolve_http_proxy_url_for_websocket_target,
    openai_responses::parse_openai_responses_usage,
};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::ErrorKind,
    sync::{Arc, LazyLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
};

pub const DEFAULT_OPENAI_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
pub const OPENAI_CODEX_WEBSOCKET_BETA: &str = "responses_websockets=2026-02-06";
pub const OPENAI_CODEX_WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE: &str =
    "websocket_connection_limit_reached";
/// Codex closes websocket sessions server-side after about an hour; rotate
/// cached connections before that so a request never lands on a stale socket
/// (pi #6268).
pub const OPENAI_CODEX_SESSION_WEBSOCKET_MAX_AGE_MS: i64 = 55 * 60 * 1000;
/// Cached websocket connections idle longer than this are closed and a fresh
/// socket is dialed (pi SESSION_WEBSOCKET_CACHE_TTL_MS).
pub const OPENAI_CODEX_SESSION_WEBSOCKET_CACHE_TTL_MS: i64 = 5 * 60 * 1000;

pub fn openai_codex_websocket_session_expired(created_at_ms: i64, now_ms: i64) -> bool {
    now_ms - created_at_ms >= OPENAI_CODEX_SESSION_WEBSOCKET_MAX_AGE_MS
}

/// pi arms an idle timer when a cached socket is released; ri checks the idle
/// deadline lazily when the next request acquires the cached socket.
pub fn openai_codex_websocket_session_idle_expired(last_used_at_ms: i64, now_ms: i64) -> bool {
    now_ms - last_used_at_ms >= OPENAI_CODEX_SESSION_WEBSOCKET_CACHE_TTL_MS
}

/// The Codex backend accepts zstd-compressed request bodies on the SSE
/// responses endpoint (the same endpoint the official Codex client compresses
/// against); the websocket transport keeps uncompressed JSON frames.
const OPENAI_CODEX_REQUEST_COMPRESSION_ZSTD_LEVEL: i32 = 3;

/// zstd-compressed body bytes, or `None` when compression fails — callers
/// fall back to sending the uncompressed JSON.
pub fn compress_openai_codex_request_body(body_json: &str) -> Option<Vec<u8>> {
    zstd::encode_all(
        body_json.as_bytes(),
        OPENAI_CODEX_REQUEST_COMPRESSION_ZSTD_LEVEL,
    )
    .ok()
}

/// Read the error code of a Codex `error` event, preferring the top-level
/// `code` and falling back to the nested `error.code` (pi
/// `extractCodexEventError`).
pub fn openai_codex_error_event_code(event: &serde_json::Value) -> Option<&str> {
    event
        .get("code")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            event
                .pointer("/error/code")
                .and_then(serde_json::Value::as_str)
        })
}
/// No provider-internal retries unless the caller opts in via
/// `options.maxRetries` (pi DEFAULT_MAX_RETRIES = 0); the outer harness retry
/// policy owns the retry budget.
pub const OPENAI_CODEX_DEFAULT_MAX_RETRIES: usize = 0;
pub const OPENAI_CODEX_BASE_RETRY_DELAY_MS: u64 = 1000;
/// Default cap applied only to 429 retry-after delays (pi
/// DEFAULT_MAX_RETRY_DELAY_MS).
pub const OPENAI_CODEX_DEFAULT_MAX_RETRY_DELAY_MS: u64 = 60_000;
/// WebSocket close code 1009 means the frame exceeded the server's message
/// size limit; surface a readable reason when the close frame has none.
pub const OPENAI_CODEX_WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE: u16 = 1009;

const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const ACCOUNT_ID_ERROR: &str = "Failed to extract accountId from token";
static OPENAI_CODEX_WS_DEBUG_STATS: LazyLock<
    Mutex<BTreeMap<String, OpenAICodexWebSocketDebugStats>>,
> = LazyLock::new(|| Mutex::new(BTreeMap::new()));
static OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS: LazyLock<Mutex<BTreeSet<String>>> =
    LazyLock::new(|| Mutex::new(BTreeSet::new()));

#[derive(Debug, Clone, Default, PartialEq)]
pub struct OpenAICodexResponsesPayloadOptions {
    pub session_id: Option<String>,
    pub temperature: Option<f64>,
    pub service_tier: Option<String>,
    pub text_verbosity: Option<String>,
    pub reasoning_effort: Option<ThinkingLevel>,
    pub reasoning_summary: Option<String>,
    /// Forwarded `tool_choice` value ("auto"/"none"/"required").
    pub tool_choice: Option<String>,
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

pub fn get_openai_codex_websocket_debug_stats(
    session_id: &str,
) -> Option<OpenAICodexWebSocketDebugStats> {
    OPENAI_CODEX_WS_DEBUG_STATS.lock().get(session_id).cloned()
}

pub fn reset_openai_codex_websocket_debug_stats(session_id: Option<&str>) {
    if let Some(session_id) = session_id {
        OPENAI_CODEX_WS_DEBUG_STATS.lock().remove(session_id);
        OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS
            .lock()
            .remove(session_id);
    } else {
        OPENAI_CODEX_WS_DEBUG_STATS.lock().clear();
        OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS.lock().clear();
    }
}

pub fn openai_codex_websocket_sse_fallback_active(session_id: Option<&str>) -> bool {
    let Some(session_id) = session_id else {
        return false;
    };
    OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS
        .lock()
        .contains(session_id)
}

pub fn record_openai_codex_websocket_request_stats_for_session(
    session_id: Option<&str>,
    request: &Value,
    reused_connection: bool,
    use_cached_context: bool,
) {
    if let Some(session_id) = session_id {
        update_openai_codex_websocket_debug_stats(session_id, |stats| {
            record_openai_codex_websocket_request_stats(
                stats,
                request,
                reused_connection,
                use_cached_context,
            );
        });
    }
}

pub fn record_openai_codex_websocket_sse_fallback_for_session(session_id: Option<&str>) {
    if let Some(session_id) = session_id {
        let fallback_active = openai_codex_websocket_sse_fallback_active(Some(session_id));
        update_openai_codex_websocket_debug_stats(session_id, |stats| {
            record_openai_codex_websocket_sse_fallback(stats, fallback_active);
        });
    }
}

pub fn record_openai_codex_websocket_failure_for_session(
    session_id: Option<&str>,
    error: impl ToString,
) {
    if let Some(session_id) = session_id {
        OPENAI_CODEX_WS_SSE_FALLBACK_SESSIONS
            .lock()
            .insert(session_id.to_owned());
        update_openai_codex_websocket_debug_stats(session_id, |stats| {
            record_openai_codex_websocket_failure(stats, error.to_string());
        });
    }
}

fn update_openai_codex_websocket_debug_stats(
    session_id: &str,
    update: impl FnOnce(&mut OpenAICodexWebSocketDebugStats),
) {
    let mut all_stats = OPENAI_CODEX_WS_DEBUG_STATS.lock();
    let stats = all_stats.entry(session_id.to_owned()).or_default();
    update(stats);
}

/// OpenAI rejects prompt cache keys longer than 64 characters.
pub fn clamp_openai_prompt_cache_key(key: &str) -> String {
    key.chars().take(64).collect()
}

pub fn build_openai_codex_responses_payload(
    model: &Model,
    context: &Context,
    options: OpenAICodexResponsesPayloadOptions,
) -> Value {
    let placement = crate::deferred_tools::split_deferred_tools(
        context,
        model
            .compat
            .as_ref()
            .and_then(|compat| compat.get("supportsToolSearch"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        |name| name.to_owned(),
    );
    let messages = crate::openai_responses::convert_openai_responses_messages_with_deferred(
        model,
        context,
        &["openai", "openai-codex", "opencode"],
        false,
        &placement.deferred,
    );
    let mut payload = json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": context
            .system_prompt
            .clone()
            .filter(|prompt| !prompt.is_empty())
            .unwrap_or_else(|| "You are a helpful assistant.".to_owned()),
        "input": messages,
        "text": { "verbosity": options.text_verbosity.unwrap_or_else(|| "low".to_owned()) },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": options.tool_choice.clone().unwrap_or_else(|| "auto".to_owned()),
        "parallel_tool_calls": true,
    });

    if let Some(session_id) = options.session_id {
        payload["prompt_cache_key"] = Value::String(clamp_openai_prompt_cache_key(&session_id));
    }
    if let Some(temperature) = options.temperature {
        payload["temperature"] = json!(temperature);
    }
    if let Some(service_tier) = options.service_tier {
        payload["service_tier"] = Value::String(service_tier);
    }
    if !placement.immediate.is_empty() {
        payload["tools"] = Value::Array(
            placement
                .immediate
                .iter()
                .map(format_openai_codex_tool)
                .collect(),
        );
    }
    if let Some(reasoning_effort) = options.reasoning_effort {
        payload["reasoning"] = json!({
            "effort": openai_codex_reasoning_effort(model, reasoning_effort),
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
    set_header_case_insensitive(&mut headers, "OpenAI-Beta", "responses=experimental");
    set_header_case_insensitive(&mut headers, "accept", "text/event-stream");
    set_header_case_insensitive(&mut headers, "content-type", "application/json");
    if let Some(session_id) = session_id {
        // The Codex backend enforces the same 64-char cap as prompt_cache_key
        // on the session affinity headers (pi dcfe36c7).
        let session_id = clamp_openai_prompt_cache_key(session_id);
        set_header_case_insensitive(&mut headers, "session-id", &session_id);
        set_header_case_insensitive(&mut headers, "x-client-request-id", &session_id);
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
    remove_header_case_insensitive(&mut headers, "accept");
    remove_header_case_insensitive(&mut headers, "content-type");
    set_header_case_insensitive(&mut headers, "OpenAI-Beta", OPENAI_CODEX_WEBSOCKET_BETA);
    let request_id = clamp_openai_prompt_cache_key(request_id);
    set_header_case_insensitive(&mut headers, "x-client-request-id", &request_id);
    set_header_case_insensitive(&mut headers, "session-id", &request_id);
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

/// pi `extractWebSocketCloseError`: `"WebSocket closed <code> <reason>"`,
/// with close code 1009 gaining a synthetic " message too big" reason when
/// the server sent none.
pub fn openai_codex_websocket_close_error_message(code: Option<u16>, reason: &str) -> String {
    let code_text = code.map(|code| format!(" {code}")).unwrap_or_default();
    let reason_text = if !reason.is_empty() {
        format!(" {reason}")
    } else if code == Some(OPENAI_CODEX_WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE) {
        " message too big".to_owned()
    } else {
        String::new()
    };
    format!("WebSocket closed{code_text}{reason_text}")
        .trim()
        .to_owned()
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

        let stream =
            if let Some(proxy_url) = resolve_http_proxy_url_for_websocket_target(url.as_str())? {
                connect_websocket_proxy_stream(&url, proxy_url.as_str()).await?
            } else {
                let tcp = TcpStream::connect((host, port))
                    .await
                    .map_err(|error| error.to_string())?;
                maybe_tls_websocket_stream(scheme, host, Box::new(tcp)).await?
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
                "host" | "upgrade" | "connection" | "sec-websocket-version" | "sec-websocket-key"
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
                0x8 => {
                    // A coded close frame before completion surfaces as a
                    // pi-style "WebSocket closed <code> <reason>" error
                    // (pi extractWebSocketCloseError); a bare close keeps the
                    // legacy premature-close handling.
                    if frame.payload.len() >= 2 {
                        let code = u16::from_be_bytes([frame.payload[0], frame.payload[1]]);
                        let reason = String::from_utf8_lossy(&frame.payload[2..]).into_owned();
                        return Err(openai_codex_websocket_close_error_message(
                            Some(code),
                            &reason,
                        ));
                    }
                    return Ok(None);
                }
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

async fn connect_websocket_proxy_stream(
    target_url: &reqwest::Url,
    proxy_url: &str,
) -> Result<Box<dyn AsyncReadWrite>, String> {
    let proxy = reqwest::Url::parse(proxy_url)
        .map_err(|error| format!("Invalid proxy URL {proxy_url:?}: {error}"))?;
    let proxy_scheme = proxy.scheme();
    if !matches!(proxy_scheme, "http" | "https") {
        return Err(format!(
            "{} Got {proxy_scheme}:",
            crate::node_http_proxy::UNSUPPORTED_PROXY_PROTOCOL_MESSAGE
        ));
    }
    let proxy_host = proxy
        .host_str()
        .ok_or_else(|| "Proxy URL is missing a host".to_owned())?;
    let proxy_port = proxy
        .port_or_known_default()
        .ok_or_else(|| "Proxy URL is missing a port".to_owned())?;
    let tcp = TcpStream::connect((proxy_host, proxy_port))
        .await
        .map_err(|error| error.to_string())?;
    let mut stream = maybe_tls_websocket_stream(proxy_scheme, proxy_host, Box::new(tcp)).await?;

    let target_authority = websocket_connect_authority(target_url)?;
    let mut connect_request =
        format!("CONNECT {target_authority} HTTP/1.1\r\nHost: {target_authority}\r\n");
    if let Some(auth) = proxy_basic_auth(&proxy) {
        connect_request.push_str(&format!("Proxy-Authorization: Basic {auth}\r\n"));
    }
    connect_request.push_str("\r\n");
    stream
        .write_all(connect_request.as_bytes())
        .await
        .map_err(|error| error.to_string())?;
    stream.flush().await.map_err(|error| error.to_string())?;
    read_proxy_connect_response(&mut stream).await?;

    let target_host = target_url
        .host_str()
        .ok_or_else(|| "WebSocket URL is missing a host".to_owned())?;
    maybe_tls_websocket_stream(target_url.scheme(), target_host, stream).await
}

async fn maybe_tls_websocket_stream(
    scheme: &str,
    host: &str,
    stream: Box<dyn AsyncReadWrite>,
) -> Result<Box<dyn AsyncReadWrite>, String> {
    if !matches!(scheme, "https" | "wss") {
        return Ok(stream);
    }
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    let connector = tokio_rustls::TlsConnector::from(Arc::new(config));
    let server_name = rustls_pki_types::ServerName::try_from(host.to_owned())
        .map_err(|error| error.to_string())?;
    Ok(Box::new(
        connector
            .connect(server_name, stream)
            .await
            .map_err(|error| error.to_string())?,
    ))
}

fn websocket_connect_authority(url: &reqwest::Url) -> Result<String, String> {
    let host = url
        .host_str()
        .ok_or_else(|| "WebSocket URL is missing a host".to_owned())?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| "WebSocket URL is missing a port".to_owned())?;
    Ok(format!("{host}:{port}"))
}

fn proxy_basic_auth(proxy: &reqwest::Url) -> Option<String> {
    let username = proxy.username();
    if username.is_empty() {
        return None;
    }
    let credentials = match proxy.password() {
        Some(password) => format!("{username}:{password}"),
        None => format!("{username}:"),
    };
    Some(encode_base64(credentials.as_bytes()))
}

async fn read_proxy_connect_response(stream: &mut Box<dyn AsyncReadWrite>) -> Result<(), String> {
    let mut response = Vec::new();
    let mut buf = [0u8; 1024];
    loop {
        let n = stream
            .read(&mut buf)
            .await
            .map_err(|error| error.to_string())?;
        if n == 0 {
            break;
        }
        response.extend_from_slice(&buf[..n]);
        if response.windows(4).any(|window| window == b"\r\n\r\n") {
            break;
        }
        if response.len() > 16 * 1024 {
            return Err("Proxy CONNECT response exceeded 16 KiB".to_owned());
        }
    }
    let text = String::from_utf8_lossy(&response);
    let status = text
        .lines()
        .next()
        .ok_or_else(|| "Proxy CONNECT response was empty".to_owned())?;
    let status_code = status
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| format!("Invalid proxy CONNECT response: {status}"))?;
    if !(200..300).contains(&status_code) {
        return Err(format!("Proxy CONNECT failed: {status}"));
    }
    Ok(())
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

/// pi `isTerminalRateLimitError`: a 429 whose RAW body matches a terminal
/// account/subscription limit must not be retried.
pub fn is_openai_codex_terminal_rate_limit_error(error_text: &str) -> bool {
    crate::retry::is_non_retryable_provider_limit_error(error_text)
}

static OPENAI_CODEX_RETRYABLE_ERROR_TEXT_PATTERN: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(
        "(?i)rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
    )
    .expect("valid codex retryable error pattern")
});

/// pi `isRetryableError`: `error_text` is the RAW response body, not the
/// parsed friendly message.
pub fn is_openai_codex_retryable_error(status: u16, error_text: &str) -> bool {
    if status == 429 && is_openai_codex_terminal_rate_limit_error(error_text) {
        return false;
    }
    if matches!(status, 429 | 500 | 502 | 503 | 504) {
        return true;
    }
    OPENAI_CODEX_RETRYABLE_ERROR_TEXT_PATTERN.is_match(error_text)
}

/// JS `Number(text)` semantics for retry headers: trimmed empty input is 0,
/// hex/octal/binary literals parse, and unparseable text is NaN (`None`).
fn js_header_number(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some(0.0);
    }
    match trimmed {
        "Infinity" | "+Infinity" => return Some(f64::INFINITY),
        "-Infinity" => return Some(f64::NEG_INFINITY),
        _ => {}
    }
    if let Some(rest) = trimmed
        .strip_prefix("0x")
        .or_else(|| trimmed.strip_prefix("0X"))
    {
        return u128::from_str_radix(rest, 16)
            .ok()
            .map(|value| value as f64);
    }
    if let Some(rest) = trimmed
        .strip_prefix("0o")
        .or_else(|| trimmed.strip_prefix("0O"))
    {
        return u128::from_str_radix(rest, 8).ok().map(|value| value as f64);
    }
    if let Some(rest) = trimmed
        .strip_prefix("0b")
        .or_else(|| trimmed.strip_prefix("0B"))
    {
        return u128::from_str_radix(rest, 2).ok().map(|value| value as f64);
    }
    // Rust's f64 parser accepts "inf"/"nan" spellings JS Number() rejects.
    if trimmed
        .chars()
        .any(|ch| ch.is_ascii_alphabetic() && !matches!(ch, 'e' | 'E'))
    {
        return None;
    }
    trimmed.parse::<f64>().ok().filter(|value| !value.is_nan())
}

/// JS `Date.parse` accepts IMF-fixdate (RFC 2822) and ISO-8601 forms; naive
/// datetimes without a timezone are read as UTC here (JS uses local time, but
/// real retry-after dates always carry a zone).
fn parse_openai_codex_retry_after_date_ms(text: &str) -> Option<i64> {
    if let Ok(date) = DateTime::parse_from_rfc2822(text) {
        return Some(date.with_timezone(&Utc).timestamp_millis());
    }
    if let Ok(date) = DateTime::parse_from_rfc3339(text) {
        return Some(date.with_timezone(&Utc).timestamp_millis());
    }
    if let Ok(datetime) = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(datetime.and_utc().timestamp_millis());
    }
    if let Ok(date) = chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d") {
        return Some(date.and_hms_opt(0, 0, 0)?.and_utc().timestamp_millis());
    }
    None
}

/// pi `getRetryAfterDelayMs`: `retry-after-ms` wins only when it parses to a
/// finite number, otherwise the `retry-after` header is consulted (numeric
/// seconds first, then an HTTP date).
pub fn openai_codex_retry_after_delay_ms(
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    now_ms: i64,
) -> Option<u64> {
    if let Some(value) = retry_after_ms
        && let Some(millis) = js_header_number(value).filter(|millis| millis.is_finite())
    {
        return Some(millis.max(0.0) as u64);
    }

    // An empty retry-after header is falsy in pi and yields no delay.
    let retry_after = retry_after.filter(|value| !value.is_empty())?;
    if let Some(seconds) = js_header_number(retry_after).filter(|seconds| seconds.is_finite()) {
        return Some((seconds.max(0.0) * 1000.0) as u64);
    }
    let date_ms = parse_openai_codex_retry_after_date_ms(retry_after)?;
    Some(date_ms.saturating_sub(now_ms).max(0) as u64)
}

/// pi `capRetryDelayMs`: default cap 60s; a caller-provided cap of 0 disables
/// capping entirely (pi treats `maxRetryDelayMs <= 0` as "no cap").
pub fn cap_openai_codex_retry_delay_ms(delay_ms: u64, max_retry_delay_ms: Option<u64>) -> u64 {
    let cap = max_retry_delay_ms.unwrap_or(OPENAI_CODEX_DEFAULT_MAX_RETRY_DELAY_MS);
    if cap > 0 { delay_ms.min(cap) } else { delay_ms }
}

pub fn openai_codex_retry_delay_ms(
    status: u16,
    error_text: &str,
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    attempt: usize,
    now_ms: i64,
) -> Option<u64> {
    openai_codex_retry_delay_ms_with_limits(
        status,
        error_text,
        retry_after_ms,
        retry_after,
        attempt,
        now_ms,
        OPENAI_CODEX_DEFAULT_MAX_RETRIES,
        None,
    )
}

/// `error_text` must be the RAW response body (pi retries on the raw text and
/// only parses the friendly message for the final error).
pub fn openai_codex_retry_delay_ms_with_limits(
    status: u16,
    error_text: &str,
    retry_after_ms: Option<&str>,
    retry_after: Option<&str>,
    attempt: usize,
    now_ms: i64,
    max_retries: usize,
    max_retry_delay_ms: Option<u64>,
) -> Option<u64> {
    if attempt >= max_retries || !is_openai_codex_retryable_error(status, error_text) {
        return None;
    }

    match openai_codex_retry_after_delay_ms(retry_after_ms, retry_after, now_ms) {
        // The cap applies only to 429 retry-after delays; exponential backoff
        // and non-429 retry-after delays are never capped (pi :391-397).
        Some(delay_ms) if status == 429 => Some(cap_openai_codex_retry_delay_ms(
            delay_ms,
            max_retry_delay_ms,
        )),
        Some(delay_ms) => Some(delay_ms),
        None => {
            let attempt = u32::try_from(attempt).unwrap_or(u32::MAX);
            Some(OPENAI_CODEX_BASE_RETRY_DELAY_MS.saturating_mul(2_u64.saturating_pow(attempt)))
        }
    }
}

pub fn openai_codex_error_message_from_response(
    status: u16,
    status_text: &str,
    body: &str,
    now_ms: i64,
) -> String {
    // pi parseErrorResponse: `raw || response.statusText || "Request failed"`.
    let mut message = if !body.is_empty() {
        body.to_owned()
    } else if !status_text.is_empty() {
        status_text.to_owned()
    } else {
        "Request failed".to_owned()
    };
    let mut friendly_message = None;

    // Strict JSON.parse like pi; a malformed body keeps the raw text.
    if let Ok(value) = serde_json::from_str::<Value>(body)
        && let Some(error) = value.get("error").and_then(Value::as_object)
    {
        let code = error
            .get("code")
            .or_else(|| error.get("type"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if status == 429
            || code.contains("usage_limit_reached")
            || code.contains("usage_not_included")
            || code.contains("rate_limit_exceeded")
        {
            let plan = error
                .get("plan_type")
                .and_then(Value::as_str)
                .filter(|plan| !plan.is_empty())
                .map(|plan| format!(" ({} plan)", plan.to_ascii_lowercase()))
                .unwrap_or_default();
            let when = error
                .get("resets_at")
                .and_then(Value::as_f64)
                .filter(|resets_at| *resets_at > 0.0)
                .map(|resets_at| {
                    let millis = resets_at * 1000.0 - now_ms as f64;
                    let minutes = (millis / 60_000.0).round().max(0.0) as i64;
                    format!(" Try again in ~{minutes} min.")
                })
                .unwrap_or_default();
            friendly_message = Some(format!(
                "You have hit your ChatGPT usage limit{plan}.{when}"
            ));
        }
        // pi: `err.message || friendlyMessage || message` (empty is falsy).
        if let Some(error_message) = error
            .get("message")
            .and_then(Value::as_str)
            .filter(|error_message| !error_message.is_empty())
        {
            message = error_message.to_owned();
        } else if let Some(friendly_message) = &friendly_message {
            message = friendly_message.clone();
        }
    }

    friendly_message.unwrap_or(message)
}

fn build_openai_codex_base_headers(
    model_headers: &BTreeMap<String, String>,
    option_headers: &BTreeMap<String, String>,
    account_id: &str,
    token: &str,
) -> BTreeMap<String, String> {
    let mut headers = model_headers.clone();
    headers.extend(option_headers.clone());
    set_header_case_insensitive(&mut headers, "Authorization", &format!("Bearer {token}"));
    set_header_case_insensitive(&mut headers, "chatgpt-account-id", account_id);
    set_header_case_insensitive(&mut headers, "originator", "pi");
    set_header_case_insensitive(&mut headers, "User-Agent", "pi (browser)");
    headers
}

fn set_header_case_insensitive(headers: &mut BTreeMap<String, String>, name: &str, value: &str) {
    remove_header_case_insensitive(headers, name);
    headers.insert(name.to_owned(), value.to_owned());
}

fn remove_header_case_insensitive(headers: &mut BTreeMap<String, String>, name: &str) {
    let keys = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case(name))
        .cloned()
        .collect::<Vec<_>>();
    for key in keys {
        headers.remove(&key);
    }
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

fn openai_codex_reasoning_effort(model: &Model, level: ThinkingLevel) -> String {
    // pi: `thinkingLevelMap?.[level] ?? level` — a null map entry falls back
    // to the raw level name, so codex always sends a reasoning effort.
    match model.thinking_level_map.get(&level) {
        Some(Some(effort)) => effort.clone(),
        _ => match level {
            ThinkingLevel::Off => "none",
            ThinkingLevel::Minimal => "minimal",
            ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High => "high",
            ThinkingLevel::XHigh => "xhigh",
            ThinkingLevel::Max => "max",
        }
        .to_owned(),
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
