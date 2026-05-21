use crate::{
    anthropic_oauth::{
        OAuthCallbackServer, OAuthCallbackServerOptions, OAuthCredentials, OAuthHttpRequest,
        OAuthLoginFlow, generate_pkce, noop_oauth_callback_server, parse_authorization_input,
        send_oauth_http_request, start_oauth_callback_server,
    },
    openai_codex_responses::extract_openai_codex_account_id,
    types::now_millis,
};
use serde_json::Value;
use std::{collections::BTreeMap, net::SocketAddr};

pub const OPENAI_CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_OAUTH_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const OPENAI_CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const OPENAI_CODEX_OAUTH_CALLBACK_PORT: u16 = 1455;
pub const OPENAI_CODEX_OAUTH_CALLBACK_PATH: &str = "/auth/callback";
pub const OPENAI_CODEX_OAUTH_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const OPENAI_CODEX_OAUTH_SCOPE: &str = "openid profile email offline_access";

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
        expires: now_millis + expires_in * 1000 - 5 * 60 * 1000,
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
