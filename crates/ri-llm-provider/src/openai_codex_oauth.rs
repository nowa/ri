use crate::{
    anthropic_oauth::{
        OAuthCallbackServer, OAuthCallbackServerOptions, OAuthCredentials, OAuthHttpRequest,
        OAuthLoginFlow, generate_pkce, noop_oauth_callback_server, parse_authorization_input,
        send_oauth_http_request, start_oauth_callback_server,
    },
    openai_codex_responses::extract_openai_codex_account_id,
    types::now_millis,
};
use ring::rand::SecureRandom;
use serde_json::Value;
use std::{collections::BTreeMap, net::SocketAddr};

pub const OPENAI_CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_OAUTH_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const OPENAI_CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const OPENAI_CODEX_OAUTH_CALLBACK_PORT: u16 = 1455;
pub const OPENAI_CODEX_OAUTH_CALLBACK_PATH: &str = "/auth/callback";
pub const OPENAI_CODEX_OAUTH_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const OPENAI_CODEX_OAUTH_SCOPE: &str = "openid profile email offline_access";

pub fn generate_openai_codex_oauth_state() -> Result<String, String> {
    let rng = ring::rand::SystemRandom::new();
    let mut bytes = [0_u8; 16];
    rng.fill(&mut bytes)
        .map_err(|_| "Failed to generate OpenAI Codex OAuth state".to_owned())?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

pub fn build_openai_codex_authorize_url(
    code_challenge: &str,
    state: &str,
    originator: Option<&str>,
) -> String {
    build_openai_codex_authorize_url_with_redirect_uri(
        code_challenge,
        state,
        originator,
        OPENAI_CODEX_OAUTH_REDIRECT_URI,
    )
}

pub fn build_openai_codex_authorize_url_with_redirect_uri(
    code_challenge: &str,
    state: &str,
    originator: Option<&str>,
    redirect_uri: &str,
) -> String {
    let params = [
        ("response_type", "code"),
        ("client_id", OPENAI_CODEX_OAUTH_CLIENT_ID),
        ("redirect_uri", redirect_uri),
        ("scope", OPENAI_CODEX_OAUTH_SCOPE),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("originator", originator.unwrap_or("pi")),
    ];
    let query = params
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{OPENAI_CODEX_OAUTH_AUTHORIZE_URL}?{query}")
}

pub async fn start_openai_codex_oauth_login_flow(
    state: &str,
    originator: Option<&str>,
) -> Result<OAuthLoginFlow, String> {
    let pkce = generate_pkce()?;
    start_openai_codex_oauth_login_flow_with_pkce(
        &pkce.verifier,
        &pkce.challenge,
        state,
        originator,
        OPENAI_CODEX_OAUTH_CALLBACK_PORT,
    )
    .await
}

pub async fn start_openai_codex_oauth_login_flow_with_pkce(
    verifier: &str,
    challenge: &str,
    state: &str,
    originator: Option<&str>,
    port: u16,
) -> Result<OAuthLoginFlow, String> {
    let callback_options = openai_codex_oauth_callback_server_options_with_port(state, port);
    let fallback_redirect_uri = callback_options.redirect_uri.clone();
    let fallback_addr = SocketAddr::from(([127, 0, 0, 1], port));
    let callback_server = start_oauth_callback_server(callback_options)
        .await
        .unwrap_or_else(|_| noop_oauth_callback_server(fallback_redirect_uri, fallback_addr));
    let redirect_uri = callback_server.redirect_uri.clone();
    let local_addr = callback_server.local_addr;
    Ok(OAuthLoginFlow {
        auth_url: build_openai_codex_authorize_url_with_redirect_uri(
            challenge,
            state,
            originator,
            &redirect_uri,
        ),
        instructions: Some("A browser window should open. Complete login to finish.".to_owned()),
        redirect_uri,
        verifier: verifier.to_owned(),
        state: state.to_owned(),
        local_addr,
        callback_server,
    })
}

pub async fn finish_openai_codex_oauth_login_from_callback_at(
    flow: OAuthLoginFlow,
    token_url: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    let OAuthLoginFlow {
        verifier,
        redirect_uri,
        callback_server,
        ..
    } = flow;
    let callback = callback_server
        .wait_for_code()
        .await?
        .ok_or_else(|| "Missing authorization code".to_owned())?;
    exchange_openai_codex_authorization_code_with_url_at(
        &callback.code,
        &verifier,
        Some(&redirect_uri),
        token_url,
        now_millis,
    )
    .await
}

pub async fn finish_openai_codex_oauth_login_from_manual_input_at(
    flow: OAuthLoginFlow,
    input: &str,
    token_url: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    let OAuthLoginFlow {
        verifier,
        state,
        redirect_uri,
        mut callback_server,
        ..
    } = flow;
    callback_server.cancel_wait();
    let _ = callback_server.wait_for_code().await?;
    let parsed = parse_authorization_input(input);
    if let Some(parsed_state) = parsed.state.as_deref()
        && parsed_state != state
    {
        return Err("State mismatch".to_owned());
    }
    let code = parsed
        .code
        .ok_or_else(|| "Missing authorization code".to_owned())?;
    exchange_openai_codex_authorization_code_with_url_at(
        &code,
        &verifier,
        Some(&redirect_uri),
        token_url,
        now_millis,
    )
    .await
}

pub fn openai_codex_oauth_callback_server_options(
    expected_state: &str,
) -> OAuthCallbackServerOptions {
    openai_codex_oauth_callback_server_options_with_port(
        expected_state,
        OPENAI_CODEX_OAUTH_CALLBACK_PORT,
    )
}

pub fn openai_codex_oauth_callback_server_options_with_port(
    expected_state: &str,
    port: u16,
) -> OAuthCallbackServerOptions {
    OAuthCallbackServerOptions {
        bind_host: oauth_callback_host(),
        port,
        callback_path: OPENAI_CODEX_OAUTH_CALLBACK_PATH.to_owned(),
        redirect_uri: format!("http://localhost:{port}{OPENAI_CODEX_OAUTH_CALLBACK_PATH}"),
        expected_state: expected_state.to_owned(),
        success_message: "OpenAI authentication completed. You can close this window.".to_owned(),
    }
}

pub async fn start_openai_codex_oauth_callback_server(
    expected_state: &str,
) -> Result<OAuthCallbackServer, String> {
    start_oauth_callback_server(openai_codex_oauth_callback_server_options(expected_state)).await
}

pub fn build_openai_codex_authorization_code_token_request(
    code: &str,
    verifier: &str,
    redirect_uri: Option<&str>,
) -> OAuthHttpRequest {
    build_openai_codex_authorization_code_token_request_with_url(
        code,
        verifier,
        redirect_uri,
        OPENAI_CODEX_OAUTH_TOKEN_URL,
    )
}

pub fn build_openai_codex_authorization_code_token_request_with_url(
    code: &str,
    verifier: &str,
    redirect_uri: Option<&str>,
    token_url: &str,
) -> OAuthHttpRequest {
    form_post_request(
        &[
            ("grant_type", "authorization_code"),
            ("client_id", OPENAI_CODEX_OAUTH_CLIENT_ID),
            ("code", code),
            ("code_verifier", verifier),
            (
                "redirect_uri",
                redirect_uri.unwrap_or(OPENAI_CODEX_OAUTH_REDIRECT_URI),
            ),
        ],
        token_url,
    )
}

pub async fn exchange_openai_codex_authorization_code_at(
    code: &str,
    verifier: &str,
    redirect_uri: Option<&str>,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    exchange_openai_codex_authorization_code_with_url_at(
        code,
        verifier,
        redirect_uri,
        OPENAI_CODEX_OAUTH_TOKEN_URL,
        now_millis,
    )
    .await
}

pub async fn exchange_openai_codex_authorization_code_with_url_at(
    code: &str,
    verifier: &str,
    redirect_uri: Option<&str>,
    token_url: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    let request = build_openai_codex_authorization_code_token_request_with_url(
        code,
        verifier,
        redirect_uri,
        token_url,
    );
    let response = send_oauth_http_request(&request).await?;
    if response.status / 100 != 2 {
        return Err(openai_codex_token_failure_message(
            "authorization_code",
            response.status,
            &response.status_text,
            Some(&response.body),
        ));
    }
    parse_openai_codex_oauth_token_response(&response.body, now_millis)
}

pub async fn exchange_openai_codex_authorization_code(
    code: &str,
    verifier: &str,
    redirect_uri: Option<&str>,
) -> Result<OAuthCredentials, String> {
    exchange_openai_codex_authorization_code_at(code, verifier, redirect_uri, now_millis() as i64)
        .await
}

pub fn build_openai_codex_refresh_token_request(refresh_token: &str) -> OAuthHttpRequest {
    build_openai_codex_refresh_token_request_with_url(refresh_token, OPENAI_CODEX_OAUTH_TOKEN_URL)
}

pub fn build_openai_codex_refresh_token_request_with_url(
    refresh_token: &str,
    token_url: &str,
) -> OAuthHttpRequest {
    form_post_request(
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", OPENAI_CODEX_OAUTH_CLIENT_ID),
        ],
        token_url,
    )
}

pub async fn refresh_openai_codex_token_at(
    refresh_token: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    refresh_openai_codex_token_with_url_at(refresh_token, OPENAI_CODEX_OAUTH_TOKEN_URL, now_millis)
        .await
}

pub async fn refresh_openai_codex_token_with_url_at(
    refresh_token: &str,
    token_url: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    let request = build_openai_codex_refresh_token_request_with_url(refresh_token, token_url);
    let response = send_oauth_http_request(&request).await?;
    if response.status / 100 != 2 {
        return Err(openai_codex_token_failure_message(
            "refresh",
            response.status,
            &response.status_text,
            Some(&response.body),
        ));
    }
    parse_openai_codex_oauth_token_response(&response.body, now_millis)
}

pub async fn refresh_openai_codex_token(refresh_token: &str) -> Result<OAuthCredentials, String> {
    refresh_openai_codex_token_at(refresh_token, now_millis() as i64).await
}

pub fn parse_openai_codex_oauth_token_response(
    response_body: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    let data: Value = serde_json::from_str(response_body)
        .map_err(|error| format!("OpenAI Codex token response was invalid JSON: {error}"))?;
    let access = data
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| "OpenAI Codex token response was missing access_token".to_owned())?;
    let refresh = data
        .get("refresh_token")
        .and_then(Value::as_str)
        .ok_or_else(|| "OpenAI Codex token response was missing refresh_token".to_owned())?;
    let expires_in = data
        .get("expires_in")
        .and_then(Value::as_i64)
        .ok_or_else(|| "OpenAI Codex token response was missing expires_in".to_owned())?;
    let account_id = extract_openai_codex_account_id(access)?;

    Ok(OAuthCredentials {
        refresh: refresh.to_owned(),
        access: access.to_owned(),
        expires: now_millis + expires_in * 1000,
        extra: BTreeMap::from([("accountId".to_owned(), Value::String(account_id))]),
    })
}

pub fn openai_codex_token_failure_message(
    operation: &str,
    status: u16,
    status_text: &str,
    response_body: Option<&str>,
) -> String {
    let details = response_body
        .filter(|body| !body.is_empty())
        .unwrap_or(status_text);
    format!("OpenAI Codex token {operation} failed ({status}): {details}")
}

fn form_post_request(params: &[(&str, &str)], token_url: &str) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: token_url.to_owned(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([(
            "Content-Type".to_owned(),
            "application/x-www-form-urlencoded".to_owned(),
        )]),
        body: form_body(params),
    }
}

fn form_body(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

fn form_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'*' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn oauth_callback_host() -> String {
    std::env::var("PI_OAUTH_CALLBACK_HOST").unwrap_or_else(|_| "127.0.0.1".to_owned())
}

pub const OPENAI_CODEX_DEVICE_USER_CODE_URL: &str =
    "https://auth.openai.com/api/accounts/deviceauth/usercode";
pub const OPENAI_CODEX_DEVICE_TOKEN_URL: &str =
    "https://auth.openai.com/api/accounts/deviceauth/token";
pub const OPENAI_CODEX_DEVICE_VERIFICATION_URI: &str = "https://auth.openai.com/codex/device";
pub const OPENAI_CODEX_DEVICE_REDIRECT_URI: &str = "https://auth.openai.com/deviceauth/callback";
pub const OPENAI_CODEX_DEVICE_CODE_TIMEOUT_SECONDS: u64 = 15 * 60;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAICodexDeviceAuth {
    pub device_auth_id: String,
    pub user_code: String,
    pub interval_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenAICodexDeviceAuthorization {
    pub authorization_code: String,
    pub code_verifier: String,
}

pub fn build_openai_codex_device_user_code_request() -> OAuthHttpRequest {
    build_openai_codex_device_user_code_request_with_url(OPENAI_CODEX_DEVICE_USER_CODE_URL)
}

pub fn build_openai_codex_device_user_code_request_with_url(url: &str) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: url.to_owned(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([("Content-Type".to_owned(), "application/json".to_owned())]),
        body: serde_json::json!({ "client_id": OPENAI_CODEX_OAUTH_CLIENT_ID }).to_string(),
    }
}

pub fn parse_openai_codex_device_user_code_response(
    status: u16,
    response_body: &str,
) -> Result<OpenAICodexDeviceAuth, String> {
    if status == 404 {
        return Err(
            "OpenAI Codex device code login is not enabled for this server. Use browser login or verify the server URL."
                .to_owned(),
        );
    }
    if status / 100 != 2 {
        let suffix = if response_body.is_empty() {
            String::new()
        } else {
            format!(": {response_body}")
        };
        return Err(format!(
            "OpenAI Codex device code request failed with status {status}{suffix}"
        ));
    }
    let data: Value = serde_json::from_str(response_body)
        .map_err(|_| format!("Invalid OpenAI Codex device code response: {response_body}"))?;
    let device_auth_id = data
        .get("device_auth_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    let user_code = data
        .get("user_code")
        .and_then(Value::as_str)
        .filter(|code| !code.is_empty());
    // `interval` may arrive as a number or a numeric string.
    let interval_seconds = match data.get("interval") {
        Some(Value::String(value)) => value.trim().parse::<f64>().ok(),
        Some(value) => value.as_f64(),
        None => None,
    }
    .filter(|interval| interval.is_finite() && *interval >= 0.0);
    match (device_auth_id, user_code, interval_seconds) {
        (Some(device_auth_id), Some(user_code), Some(interval_seconds)) => {
            Ok(OpenAICodexDeviceAuth {
                device_auth_id: device_auth_id.to_owned(),
                user_code: user_code.to_owned(),
                interval_seconds: interval_seconds as u64,
            })
        }
        _ => Err(format!("Invalid OpenAI Codex device code response: {data}")),
    }
}

pub async fn request_openai_codex_device_auth() -> Result<OpenAICodexDeviceAuth, String> {
    request_openai_codex_device_auth_with_url(OPENAI_CODEX_DEVICE_USER_CODE_URL).await
}

pub async fn request_openai_codex_device_auth_with_url(
    url: &str,
) -> Result<OpenAICodexDeviceAuth, String> {
    let request = build_openai_codex_device_user_code_request_with_url(url);
    let response = send_oauth_http_request(&request).await?;
    parse_openai_codex_device_user_code_response(response.status, &response.body)
}

pub fn build_openai_codex_device_token_poll_request(
    device: &OpenAICodexDeviceAuth,
) -> OAuthHttpRequest {
    build_openai_codex_device_token_poll_request_with_url(device, OPENAI_CODEX_DEVICE_TOKEN_URL)
}

pub fn build_openai_codex_device_token_poll_request_with_url(
    device: &OpenAICodexDeviceAuth,
    url: &str,
) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: url.to_owned(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([("Content-Type".to_owned(), "application/json".to_owned())]),
        body: serde_json::json!({
            "device_auth_id": device.device_auth_id,
            "user_code": device.user_code,
        })
        .to_string(),
    }
}

pub fn parse_openai_codex_device_token_poll_response(
    status: u16,
    response_body: &str,
) -> crate::device_code::DeviceCodePollResponse<OpenAICodexDeviceAuthorization> {
    use crate::device_code::DeviceCodePollResponse;
    if status / 100 == 2 {
        let parsed = serde_json::from_str::<Value>(response_body).ok();
        let authorization_code = parsed
            .as_ref()
            .and_then(|data| data.get("authorization_code"))
            .and_then(Value::as_str)
            .filter(|code| !code.is_empty());
        let code_verifier = parsed
            .as_ref()
            .and_then(|data| data.get("code_verifier"))
            .and_then(Value::as_str)
            .filter(|verifier| !verifier.is_empty());
        return match (authorization_code, code_verifier) {
            (Some(authorization_code), Some(code_verifier)) => {
                DeviceCodePollResponse::Complete(OpenAICodexDeviceAuthorization {
                    authorization_code: authorization_code.to_owned(),
                    code_verifier: code_verifier.to_owned(),
                })
            }
            _ => DeviceCodePollResponse::Failed {
                message: format!(
                    "Invalid OpenAI Codex device auth token response: {response_body}"
                ),
            },
        };
    }
    if status == 403 || status == 404 {
        return DeviceCodePollResponse::Pending;
    }
    let error_code = serde_json::from_str::<Value>(response_body)
        .ok()
        .and_then(|data| {
            let error = data.get("error")?.clone();
            match error {
                Value::String(code) => Some(code),
                Value::Object(_) => error
                    .get("code")
                    .and_then(Value::as_str)
                    .map(ToOwned::to_owned),
                _ => None,
            }
        });
    match error_code.as_deref() {
        Some("deviceauth_authorization_pending") => DeviceCodePollResponse::Pending,
        Some("slow_down") => DeviceCodePollResponse::SlowDown {
            interval_seconds: None,
        },
        _ => {
            let suffix = if response_body.is_empty() {
                String::new()
            } else {
                format!(": {response_body}")
            };
            DeviceCodePollResponse::Failed {
                message: format!("OpenAI Codex device auth failed with status {status}{suffix}"),
            }
        }
    }
}

pub async fn poll_openai_codex_device_auth_with_url_with_sleeper<F, Fut>(
    device: &OpenAICodexDeviceAuth,
    token_url: &str,
    start_ms: i64,
    sleep: F,
) -> Result<OpenAICodexDeviceAuthorization, String>
where
    F: FnMut(u64) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let config = crate::device_code::DeviceCodePollConfig {
        interval_seconds: Some(device.interval_seconds),
        expires_in_seconds: Some(OPENAI_CODEX_DEVICE_CODE_TIMEOUT_SECONDS),
        wait_before_first_poll: false,
    };
    let request = build_openai_codex_device_token_poll_request_with_url(device, token_url);
    crate::device_code::poll_device_code_flow_with_sleeper(
        &config,
        start_ms,
        || async {
            let response = send_oauth_http_request(&request).await?;
            Ok(parse_openai_codex_device_token_poll_response(
                response.status,
                &response.body,
            ))
        },
        sleep,
    )
    .await
}

pub async fn login_openai_codex_device_code_with_urls_with_sleeper<C, F, Fut>(
    user_code_url: &str,
    device_token_url: &str,
    exchange_token_url: &str,
    on_device_code: C,
    start_ms: i64,
    sleep: F,
) -> Result<OAuthCredentials, String>
where
    C: FnOnce(&OpenAICodexDeviceAuth),
    F: FnMut(u64) -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let device = request_openai_codex_device_auth_with_url(user_code_url).await?;
    on_device_code(&device);
    let authorization = poll_openai_codex_device_auth_with_url_with_sleeper(
        &device,
        device_token_url,
        start_ms,
        sleep,
    )
    .await?;
    exchange_openai_codex_authorization_code_with_url_at(
        &authorization.authorization_code,
        &authorization.code_verifier,
        Some(OPENAI_CODEX_DEVICE_REDIRECT_URI),
        exchange_token_url,
        start_ms,
    )
    .await
}

pub async fn login_openai_codex_device_code<C>(
    on_device_code: C,
) -> Result<OAuthCredentials, String>
where
    C: FnOnce(&OpenAICodexDeviceAuth),
{
    login_openai_codex_device_code_with_urls_with_sleeper(
        OPENAI_CODEX_DEVICE_USER_CODE_URL,
        OPENAI_CODEX_DEVICE_TOKEN_URL,
        OPENAI_CODEX_OAUTH_TOKEN_URL,
        on_device_code,
        now_millis() as i64,
        |delay_ms| async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        },
    )
    .await
}
