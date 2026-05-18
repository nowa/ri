use crate::anthropic_oauth::OAuthHttpRequest;
use std::collections::BTreeMap;

pub const OPENAI_CODEX_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const OPENAI_CODEX_OAUTH_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";
pub const OPENAI_CODEX_OAUTH_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
pub const OPENAI_CODEX_OAUTH_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";
pub const OPENAI_CODEX_OAUTH_SCOPE: &str = "openid profile email offline_access";

pub fn build_openai_codex_authorize_url(
    code_challenge: &str,
    state: &str,
    originator: Option<&str>,
) -> String {
    let params = [
        ("response_type", "code"),
        ("client_id", OPENAI_CODEX_OAUTH_CLIENT_ID),
        ("redirect_uri", OPENAI_CODEX_OAUTH_REDIRECT_URI),
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

pub fn build_openai_codex_authorization_code_token_request(
    code: &str,
    verifier: &str,
    redirect_uri: Option<&str>,
) -> OAuthHttpRequest {
    form_post_request(&[
        ("grant_type", "authorization_code"),
        ("client_id", OPENAI_CODEX_OAUTH_CLIENT_ID),
        ("code", code),
        ("code_verifier", verifier),
        (
            "redirect_uri",
            redirect_uri.unwrap_or(OPENAI_CODEX_OAUTH_REDIRECT_URI),
        ),
    ])
}

pub fn build_openai_codex_refresh_token_request(refresh_token: &str) -> OAuthHttpRequest {
    form_post_request(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", OPENAI_CODEX_OAUTH_CLIENT_ID),
    ])
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

fn form_post_request(params: &[(&str, &str)]) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: OPENAI_CODEX_OAUTH_TOKEN_URL.to_owned(),
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
