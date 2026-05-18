use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const ANTHROPIC_OAUTH_CLIENT_ID: &str = "9d1c250a-e61b-44d9-88ed-5944d1962f5e";
pub const ANTHROPIC_OAUTH_AUTHORIZE_URL: &str = "https://claude.ai/oauth/authorize";
pub const ANTHROPIC_OAUTH_TOKEN_URL: &str = "https://platform.claude.com/v1/oauth/token";
pub const ANTHROPIC_OAUTH_CALLBACK_PORT: u16 = 53692;
pub const ANTHROPIC_OAUTH_CALLBACK_PATH: &str = "/callback";
pub const ANTHROPIC_OAUTH_REDIRECT_URI: &str = "http://localhost:53692/callback";
pub const ANTHROPIC_OAUTH_SCOPES: &str = "org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthHttpRequest {
    pub url: String,
    pub method: String,
    pub headers: BTreeMap<String, String>,
    pub body: String,
}

impl OAuthHttpRequest {
    pub fn json_body(&self) -> Result<Value, serde_json::Error> {
        serde_json::from_str(&self.body)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthCredentials {
    pub refresh: String,
    pub access: String,
    pub expires: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct AuthorizationInput {
    pub code: Option<String>,
    pub state: Option<String>,
}

pub fn build_anthropic_authorize_url(code_challenge: &str, state: &str) -> String {
    let params = [
        ("code", "true"),
        ("client_id", ANTHROPIC_OAUTH_CLIENT_ID),
        ("response_type", "code"),
        ("redirect_uri", ANTHROPIC_OAUTH_REDIRECT_URI),
        ("scope", ANTHROPIC_OAUTH_SCOPES),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("state", state),
    ];
    let query = params
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect::<Vec<_>>()
        .join("&");
    format!("{ANTHROPIC_OAUTH_AUTHORIZE_URL}?{query}")
}

pub fn parse_anthropic_authorization_input(input: &str) -> AuthorizationInput {
    let value = input.trim();
    if value.is_empty() {
        return AuthorizationInput::default();
    }

    if looks_like_url(value) {
        return parse_authorization_query(value);
    }

    if let Some((code, state)) = value.split_once('#') {
        return AuthorizationInput {
            code: non_empty(code),
            state: non_empty(state),
        };
    }

    if value.contains("code=") {
        return parse_query_string(value);
    }

    AuthorizationInput {
        code: Some(value.to_owned()),
        state: None,
    }
}

pub fn build_anthropic_authorization_code_token_request(
    code: &str,
    state: &str,
    verifier: &str,
    redirect_uri: &str,
) -> OAuthHttpRequest {
    build_json_post_request(json!({
        "grant_type": "authorization_code",
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        "code": code,
        "state": state,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
    }))
}

pub fn build_anthropic_refresh_token_request(refresh_token: &str) -> OAuthHttpRequest {
    build_json_post_request(json!({
        "grant_type": "refresh_token",
        "client_id": ANTHROPIC_OAUTH_CLIENT_ID,
        "refresh_token": refresh_token,
    }))
}

pub fn parse_anthropic_oauth_token_response(
    response_body: &str,
    now_millis: i64,
) -> Result<OAuthCredentials, String> {
    let data: Value = serde_json::from_str(response_body)
        .map_err(|error| format!("Anthropic OAuth token response was invalid JSON: {error}"))?;
    let access = data
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| "Anthropic OAuth token response was missing access_token".to_owned())?;
    let refresh = data
        .get("refresh_token")
        .and_then(Value::as_str)
        .ok_or_else(|| "Anthropic OAuth token response was missing refresh_token".to_owned())?;
    let expires_in = data
        .get("expires_in")
        .and_then(Value::as_i64)
        .ok_or_else(|| "Anthropic OAuth token response was missing expires_in".to_owned())?;

    Ok(OAuthCredentials {
        refresh: refresh.to_owned(),
        access: access.to_owned(),
        expires: now_millis + expires_in * 1000 - 5 * 60 * 1000,
    })
}

fn build_json_post_request(body: Value) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: ANTHROPIC_OAUTH_TOKEN_URL.to_owned(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            ("Content-Type".to_owned(), "application/json".to_owned()),
        ]),
        body: body.to_string(),
    }
}

fn parse_authorization_query(url: &str) -> AuthorizationInput {
    let Some((_, query_and_fragment)) = url.split_once('?') else {
        return AuthorizationInput::default();
    };
    let query = query_and_fragment.split('#').next().unwrap_or_default();
    parse_query_string(query)
}

fn parse_query_string(query: &str) -> AuthorizationInput {
    let mut result = AuthorizationInput::default();
    for pair in query.trim_start_matches('?').split('&') {
        let Some((key, value)) = pair.split_once('=') else {
            continue;
        };
        match key {
            "code" => result.code = non_empty(&form_decode(value)),
            "state" => result.state = non_empty(&form_decode(value)),
            _ => {}
        }
    }
    result
}

fn looks_like_url(value: &str) -> bool {
    value.starts_with("http://") || value.starts_with("https://")
}

fn non_empty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_owned())
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

fn form_decode(value: &str) -> String {
    let mut bytes = Vec::new();
    let mut index = 0;
    let source = value.as_bytes();
    while index < source.len() {
        match source[index] {
            b'+' => {
                bytes.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < source.len() => {
                let hex = &value[index + 1..index + 3];
                if let Ok(byte) = u8::from_str_radix(hex, 16) {
                    bytes.push(byte);
                    index += 3;
                } else {
                    bytes.push(source[index]);
                    index += 1;
                }
            }
            byte => {
                bytes.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
