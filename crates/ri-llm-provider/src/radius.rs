//! Radius gateway support (pi `providers/radius.ts`, `providers/radius-config.ts`,
//! `auth/oauth/radius.ts`).
//!
//! Radius is a pi-messages gateway. The model catalog comes from the gateway
//! (`/v1/config`) and persists through the `ModelsStore`; OAuth endpoints are
//! discovered from the gateway (`/v1/oauth`) and drive browser (local
//! callback) and device-code sign-in through [`AuthInteraction`].

use crate::anthropic_oauth::{OAuthCallbackServerOptions, start_oauth_callback_server};
use crate::auth::ModelAuth;
use crate::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthPromptOption, OAuthAuth, OAuthCredential,
};
use crate::device_code::{
    DeviceCodePollConfig, DeviceCodePollResponse, poll_device_code_flow_with_sleeper_and_abort,
};
use crate::types::{Model, now_millis};
use async_trait::async_trait;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

pub const DEFAULT_RADIUS_GATEWAY: &str = "https://radius.pi.dev";
const RADIUS_CALLBACK_HOST: &str = "127.0.0.1";
const RADIUS_CALLBACK_PORT: u16 = 1456;
const RADIUS_CALLBACK_PATH: &str = "/oauth/callback";
const TOKEN_EXPIRY_SKEW_MS: i64 = 60_000;
const LOGIN_METHOD_BROWSER: &str = "browser";
const LOGIN_METHOD_DEVICE_CODE: &str = "device-code";

// ============================================================================
// Gateway config / catalog
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct RadiusGatewayConfig {
    pub base_url: String,
    /// Raw model entries; invalid entries are filtered at conversion time.
    pub models: Vec<Value>,
}

pub fn normalize_radius_gateway_url(value: &str) -> String {
    let lowered = value.to_ascii_lowercase();
    // pi: /^https?:\/\//i — scheme detection is case-insensitive.
    let with_scheme = if lowered.starts_with("http://") || lowered.starts_with("https://") {
        value.to_owned()
    } else {
        format!("https://{value}")
    };
    with_scheme.trim_end_matches('/').to_owned()
}

pub fn sanitize_radius_gateway_config(config: &Value) -> Option<RadiusGatewayConfig> {
    let object = config.as_object()?;
    let base_url = object.get("baseUrl")?.as_str()?.to_owned();
    let models = object.get("models")?.as_array()?.clone();
    Some(RadiusGatewayConfig { base_url, models })
}

/// A `gatewayConfig` embedded in a stored Radius OAuth credential.
pub fn get_radius_credential_config(
    credential: Option<&OAuthCredential>,
) -> Option<RadiusGatewayConfig> {
    sanitize_radius_gateway_config(credential?.extra.get("gatewayConfig")?)
}

pub fn get_radius_models_from_config(
    provider_id: &str,
    config: &RadiusGatewayConfig,
) -> Vec<Model> {
    config
        .models
        .iter()
        .filter_map(|entry| {
            let mut entry = entry.as_object()?.clone();
            entry.insert("api".to_owned(), json!("pi-messages"));
            entry.insert("provider".to_owned(), json!(provider_id));
            entry.insert("baseUrl".to_owned(), json!(config.base_url));
            serde_json::from_value::<Model>(Value::Object(entry)).ok()
        })
        .collect()
}

pub fn get_radius_models(provider_id: &str, credential: Option<&OAuthCredential>) -> Vec<Model> {
    get_radius_credential_config(credential)
        .map(|config| get_radius_models_from_config(provider_id, &config))
        .unwrap_or_default()
}

fn truncate_http_body(body: &str) -> String {
    let trimmed = body.trim();
    let mut result: String = trimmed.chars().take(512).collect();
    if trimmed.chars().count() > 512 {
        result.push('…');
    }
    result
}

async fn radius_get_json(url: &str, api_key: Option<&str>) -> Result<(u16, String), String> {
    let mut headers = BTreeMap::from([("Accept".to_owned(), "application/json".to_owned())]);
    if let Some(api_key) = api_key {
        headers.insert("Authorization".to_owned(), format!("Bearer {api_key}"));
    }
    let response = crate::anthropic_oauth::send_oauth_http_request(&crate::OAuthHttpRequest {
        url: url.to_owned(),
        method: "GET".to_owned(),
        headers,
        body: String::new(),
    })
    .await?;
    Ok((response.status, response.body))
}

pub async fn load_radius_gateway_config(
    gateway: &str,
    api_key: Option<&str>,
) -> Result<RadiusGatewayConfig, String> {
    let gateway = normalize_radius_gateway_url(gateway);
    let (status, body) = radius_get_json(&format!("{gateway}/v1/config"), api_key).await?;
    if status / 100 != 2 {
        return Err(format!(
            "Could not load Radius config from {gateway}: {status}: {}",
            truncate_http_body(&body)
        ));
    }
    serde_json::from_str::<Value>(&body)
        .ok()
        .as_ref()
        .and_then(sanitize_radius_gateway_config)
        .ok_or_else(|| format!("Invalid Radius config from {gateway}"))
}

// ============================================================================
// Gateway OAuth
// ============================================================================

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RadiusOAuthConfig {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub device_authorization_endpoint: String,
    #[serde(default)]
    pub device_authorization_events_endpoint: String,
    pub verification_endpoint: String,
    pub client_id: String,
    pub scope: String,
    pub device_code_grant_type: String,
}

pub async fn load_radius_oauth_config(gateway: &str) -> Result<RadiusOAuthConfig, String> {
    let gateway = normalize_radius_gateway_url(gateway);
    let (status, body) = radius_get_json(&format!("{gateway}/v1/oauth"), None).await?;
    if status / 100 != 2 {
        return Err(format!(
            "Could not load Radius OAuth config from {gateway}: {status} {body}"
        ));
    }
    serde_json::from_str(&body)
        .map_err(|error| format!("Invalid Radius OAuth config from {gateway}: {error}"))
}

/// Token-endpoint failure with the OAuth error code preserved for device-flow
/// classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RadiusOAuthError {
    pub status: u16,
    pub oauth_error: Option<String>,
    pub message: String,
}

impl RadiusOAuthError {
    fn from_response(status: u16, body: &str, message: &str) -> Self {
        let parsed = serde_json::from_str::<Value>(body).ok();
        let oauth_error = parsed
            .as_ref()
            .and_then(|value| value.get("error"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let description = parsed
            .as_ref()
            .and_then(|value| value.get("error_description"))
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| (!body.is_empty()).then(|| body.to_owned()));
        let detail = match (&oauth_error, &description) {
            (Some(error), Some(description)) => format!("{error}: {description}"),
            (Some(error), None) => error.clone(),
            (None, Some(description)) => description.clone(),
            (None, None) => status.to_string(),
        };
        Self {
            status,
            oauth_error,
            message: format!("{message}: {detail}"),
        }
    }
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

fn form_body(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(key, value)| format!("{}={}", form_encode(key), form_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

async fn request_radius_oauth_token(
    config: &RadiusOAuthConfig,
    params: &[(&str, &str)],
) -> Result<OAuthCredential, RadiusOAuthError> {
    let response = crate::anthropic_oauth::send_oauth_http_request(&crate::OAuthHttpRequest {
        url: config.token_endpoint.clone(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            (
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
        ]),
        body: form_body(params),
    })
    .await
    .map_err(|error| RadiusOAuthError {
        status: 0,
        oauth_error: None,
        message: error,
    })?;
    if response.status / 100 != 2 {
        return Err(RadiusOAuthError::from_response(
            response.status,
            &response.body,
            "Radius OAuth token request failed",
        ));
    }
    parse_radius_token_response(&response.body, now_millis() as i64).map_err(|message| {
        RadiusOAuthError {
            status: response.status,
            oauth_error: None,
            message,
        }
    })
}

pub fn parse_radius_token_response(body: &str, now_ms: i64) -> Result<OAuthCredential, String> {
    let data: Value = serde_json::from_str(body)
        .map_err(|error| format!("Radius OAuth token response was invalid JSON: {error}"))?;
    let access = data
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or("Radius OAuth token response was missing access_token")?;
    let refresh = data
        .get("refresh_token")
        .and_then(Value::as_str)
        .ok_or("Radius OAuth token response was missing refresh_token")?;
    let expires_in = data
        .get("expires_in")
        .and_then(Value::as_i64)
        .ok_or("Radius OAuth token response was missing expires_in")?;
    let mut extra = Map::new();
    if let Some(scope) = data.get("scope").and_then(Value::as_str) {
        extra.insert("scope".to_owned(), json!(scope));
    }
    Ok(OAuthCredential {
        refresh: refresh.to_owned(),
        access: access.to_owned(),
        expires: now_ms + expires_in * 1000 - TOKEN_EXPIRY_SKEW_MS,
        extra,
    })
}

async fn login_radius_with_browser(
    config: &RadiusOAuthConfig,
    interaction: &dyn AuthInteraction,
) -> Result<OAuthCredential, String> {
    let pkce = crate::anthropic_oauth::generate_pkce()?;
    // pi: crypto.randomUUID() — a v4 UUID, not the codex hex state.
    let state = uuid::Uuid::new_v4().to_string();
    let redirect_uri =
        format!("http://{RADIUS_CALLBACK_HOST}:{RADIUS_CALLBACK_PORT}{RADIUS_CALLBACK_PATH}");
    let query = form_body(&[
        ("response_type", "code"),
        ("client_id", &config.client_id),
        ("redirect_uri", &redirect_uri),
        ("scope", &config.scope),
        ("code_challenge", &pkce.challenge),
        ("code_challenge_method", "S256"),
        ("handoff", "url"),
        ("state", &state),
    ]);
    let authorize_url = format!("{}?{query}", config.authorization_endpoint);

    let callback_server = start_oauth_callback_server(OAuthCallbackServerOptions {
        bind_host: RADIUS_CALLBACK_HOST.to_owned(),
        port: RADIUS_CALLBACK_PORT,
        callback_path: RADIUS_CALLBACK_PATH.to_owned(),
        redirect_uri: redirect_uri.clone(),
        expected_state: state,
        success_message: "Signed in to Radius. You may now close this page.".to_owned(),
    })
    .await?;
    interaction.notify(AuthEvent::Progress {
        message: format!("Listening for OAuth callback on {redirect_uri}"),
    });
    interaction.notify(AuthEvent::AuthUrl {
        url: authorize_url,
        instructions: Some("Continue in your browser.".to_owned()),
    });

    let callback = callback_server
        .wait_for_code()
        .await?
        .ok_or("OAuth callback did not complete.")?;
    request_radius_oauth_token(
        config,
        &[
            ("grant_type", "authorization_code"),
            ("client_id", &config.client_id),
            ("redirect_uri", &redirect_uri),
            ("code", &callback.code),
            ("code_verifier", &pkce.verifier),
        ],
    )
    .await
    .map_err(|error| error.message)
}

#[derive(Debug, Clone, PartialEq)]
pub struct RadiusDeviceAuthorization {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: Option<String>,
    pub verification_uri_complete: Option<String>,
    pub expires_in_seconds: u64,
    pub interval_seconds: Option<u64>,
}

async fn request_radius_device_authorization(
    config: &RadiusOAuthConfig,
) -> Result<RadiusDeviceAuthorization, String> {
    let response = crate::anthropic_oauth::send_oauth_http_request(&crate::OAuthHttpRequest {
        url: config.device_authorization_endpoint.clone(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            (
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
        ]),
        body: form_body(&[("client_id", &config.client_id), ("scope", &config.scope)]),
    })
    .await?;
    if response.status / 100 != 2 {
        return Err(RadiusOAuthError::from_response(
            response.status,
            &response.body,
            "Radius OAuth device authorization failed",
        )
        .message);
    }
    let data: Value = serde_json::from_str(&response.body)
        .map_err(|error| format!("Invalid Radius device authorization response: {error}"))?;
    let device_code = data.get("device_code").and_then(Value::as_str);
    let user_code = data.get("user_code").and_then(Value::as_str);
    let expires_in = data.get("expires_in").and_then(Value::as_u64);
    match (device_code, user_code, expires_in) {
        (Some(device_code), Some(user_code), Some(expires_in)) => Ok(RadiusDeviceAuthorization {
            device_code: device_code.to_owned(),
            user_code: user_code.to_owned(),
            verification_uri: data
                .get("verification_uri")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            verification_uri_complete: data
                .get("verification_uri_complete")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned),
            expires_in_seconds: expires_in,
            interval_seconds: data.get("interval").and_then(Value::as_u64),
        }),
        _ => {
            Err("Radius OAuth device authorization response is missing required fields".to_owned())
        }
    }
}

async fn login_radius_with_device_code(
    config: &RadiusOAuthConfig,
    interaction: &dyn AuthInteraction,
) -> Result<OAuthCredential, String> {
    let device = request_radius_device_authorization(config).await?;
    interaction.notify(AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device
            .verification_uri
            .clone()
            .unwrap_or_else(|| config.verification_endpoint.clone()),
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
    });
    let poll_config = DeviceCodePollConfig {
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
        wait_before_first_poll: false,
    };
    // pi passes `interaction.signal` here; ri's `AuthInteraction` does not
    // model an abort signal, so there is no flag to pass through yet.
    poll_device_code_flow_with_sleeper_and_abort(
        &poll_config,
        now_millis() as i64,
        || async {
            match request_radius_oauth_token(
                config,
                &[
                    ("grant_type", &config.device_code_grant_type),
                    ("client_id", &config.client_id),
                    ("device_code", &device.device_code),
                ],
            )
            .await
            {
                Ok(credential) => Ok(DeviceCodePollResponse::Complete(credential)),
                Err(error) => Ok(match error.oauth_error.as_deref() {
                    Some("authorization_pending") => DeviceCodePollResponse::Pending,
                    Some("slow_down") => DeviceCodePollResponse::SlowDown {
                        interval_seconds: None,
                    },
                    Some("expired_token") => DeviceCodePollResponse::Failed {
                        message: "Device authorization expired.".to_owned(),
                    },
                    Some("access_denied") => DeviceCodePollResponse::Failed {
                        message: "Device authorization was denied.".to_owned(),
                    },
                    _ => DeviceCodePollResponse::Failed {
                        message: error.message,
                    },
                }),
            }
        },
        |delay_ms| async move {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        },
        None,
    )
    .await
}

/// Radius gateway OAuth adapter.
pub struct RadiusOAuth {
    pub name: String,
    pub gateway: String,
}

impl RadiusOAuth {
    pub fn new(name: impl Into<String>, gateway: &str) -> Self {
        Self {
            name: name.into(),
            gateway: normalize_radius_gateway_url(gateway),
        }
    }
}

#[async_trait]
impl OAuthAuth for RadiusOAuth {
    fn name(&self) -> &str {
        &self.name
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, String> {
        let config = load_radius_oauth_config(&self.gateway).await?;
        let method = interaction
            .prompt(AuthPrompt::Select {
                message: format!("Sign in to {}:", self.name),
                options: vec![
                    AuthPromptOption {
                        id: LOGIN_METHOD_BROWSER.to_owned(),
                        label: "Sign in with browser (recommended)".to_owned(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: LOGIN_METHOD_DEVICE_CODE.to_owned(),
                        label: "Sign in with device code (when signing in from another device)"
                            .to_owned(),
                        description: None,
                    },
                ],
            })
            .await?;
        match method.as_str() {
            LOGIN_METHOD_DEVICE_CODE => login_radius_with_device_code(&config, interaction).await,
            LOGIN_METHOD_BROWSER => login_radius_with_browser(&config, interaction).await,
            other => Err(format!("Unknown {} sign-in method: {other}", self.name)),
        }
    }

    async fn refresh(&self, credential: &OAuthCredential) -> Result<OAuthCredential, String> {
        let config = load_radius_oauth_config(&self.gateway).await?;
        request_radius_oauth_token(
            &config,
            &[
                ("grant_type", "refresh_token"),
                ("client_id", &config.client_id),
                ("refresh_token", &credential.refresh),
            ],
        )
        .await
        .map_err(|error| error.message)
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, String> {
        Ok(ModelAuth {
            api_key: Some(credential.access.clone()),
            ..Default::default()
        })
    }
}
