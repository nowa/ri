//! xAI OAuth device-code flow (pi `auth/oauth/xai.ts`).
//!
//! Device authorization against `auth.x.ai`, token polling on the shared
//! RFC 8628 poller ([`crate::device_code`]), and refresh with rotation
//! semantics: xAI may omit `refresh_token` on refresh when the token is not
//! rotated, in which case the previous refresh token is kept. All endpoints
//! are URL-injectable for tests via [`XaiOAuthUrls`].

use crate::anthropic_oauth::{OAuthHttpRequest, send_oauth_http_request};
use crate::auth::{AuthEvent, AuthInteraction, ModelAuth, OAuthAuth, OAuthCredential};
use crate::device_code::{
    DeviceCodePollConfig, DeviceCodePollResponse, poll_device_code_flow_with_sleeper_and_abort,
};
use crate::types::now_millis;
use async_trait::async_trait;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicI64, Ordering},
};

pub const XAI_OAUTH_CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
pub const XAI_OAUTH_SCOPE: &str = "openid profile email offline_access grok-cli:access api:access";
pub const XAI_DEVICE_CODE_URL: &str = "https://auth.x.ai/oauth2/device/code";
pub const XAI_TOKEN_URL: &str = "https://auth.x.ai/oauth2/token";
pub const XAI_OAUTH_NAME: &str = "xAI (Grok/X subscription)";
pub const XAI_OAUTH_LOGIN_LABEL: &str = "Sign in with SuperGrok or X Premium";

/// Refresh slightly before the reported expiry to avoid using a token that
/// dies mid-request.
const REFRESH_SKEW_MS: i64 = 5 * 60 * 1000;
const DEFAULT_TOKEN_LIFETIME_SECONDS: f64 = 3600.0;

/// xAI OAuth endpoints, injectable for tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XaiOAuthUrls {
    pub device_code_url: String,
    pub token_url: String,
}

impl Default for XaiOAuthUrls {
    fn default() -> Self {
        Self {
            device_code_url: XAI_DEVICE_CODE_URL.to_owned(),
            token_url: XAI_TOKEN_URL.to_owned(),
        }
    }
}

/// Parsed device authorization response (pi `XaiDeviceCode`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XaiDeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub interval_seconds: Option<u64>,
    pub expires_in_seconds: u64,
}

fn is_ok_status(status: u16) -> bool {
    (200..300).contains(&status)
}

/// pi `postForm` body handling: invalid JSON is an error, non-object JSON is
/// treated as an empty object.
fn parse_json_object(status: u16, body: &str) -> Result<Map<String, Value>, String> {
    let parsed: Value = serde_json::from_str(body)
        .map_err(|_| format!("xAI OAuth returned invalid JSON (HTTP {status})"))?;
    Ok(match parsed {
        Value::Object(object) => object,
        _ => Map::new(),
    })
}

fn required_string(body: &Map<String, Value>, field: &str) -> Result<String, String> {
    body.get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("Invalid xAI OAuth response field: {field}"))
}

fn positive_number(body: &Map<String, Value>, field: &str) -> Result<f64, String> {
    body.get(field)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| format!("Invalid xAI OAuth response field: {field}"))
}

/// The verification URI is opened in the user's browser; force it to be an
/// https URL so a malicious response cannot make `open` launch something else.
fn validate_verification_uri(raw: &str) -> Result<String, String> {
    const UNTRUSTED_MESSAGE: &str = "Untrusted verification URI in xAI OAuth response";
    let url = reqwest::Url::parse(raw).map_err(|_| UNTRUSTED_MESSAGE.to_owned())?;
    if url.scheme() != "https" {
        return Err(UNTRUSTED_MESSAGE.to_owned());
    }
    Ok(url.into())
}

/// pi `requestFailure`: `xAI OAuth {action} failed (HTTP {status})` plus the
/// upstream `error`/`error_description` when present.
fn request_failure(action: &str, status: u16, body: &Map<String, Value>) -> String {
    let detail = ["error", "error_description"]
        .into_iter()
        .filter_map(|field| body.get(field).and_then(Value::as_str))
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>()
        .join(": ");
    if detail.is_empty() {
        format!("xAI OAuth {action} failed (HTTP {status})")
    } else {
        format!("xAI OAuth {action} failed (HTTP {status}): {detail}")
    }
}

fn credentials_from_token_response(
    body: &Map<String, Value>,
    previous_refresh_token: Option<&str>,
    now_ms: i64,
) -> Result<OAuthCredential, String> {
    let access = required_string(body, "access_token")?;
    // xAI may omit refresh_token on refresh when the token is not rotated.
    let refresh = match (body.get("refresh_token"), previous_refresh_token) {
        (None, Some(previous)) => previous.to_owned(),
        _ => required_string(body, "refresh_token")?,
    };
    let expires_in_seconds = if body.contains_key("expires_in") {
        positive_number(body, "expires_in")?
    } else {
        DEFAULT_TOKEN_LIFETIME_SECONDS
    };
    Ok(OAuthCredential {
        refresh,
        access,
        expires: now_ms + (expires_in_seconds * 1000.0) as i64 - REFRESH_SKEW_MS,
        extra: Map::new(),
    })
}

fn form_headers() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("Accept".to_owned(), "application/json".to_owned()),
        (
            "Content-Type".to_owned(),
            "application/x-www-form-urlencoded".to_owned(),
        ),
    ])
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

pub fn build_xai_device_code_request() -> OAuthHttpRequest {
    build_xai_device_code_request_with_url(XAI_DEVICE_CODE_URL)
}

pub fn build_xai_device_code_request_with_url(url: &str) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: url.to_owned(),
        method: "POST".to_owned(),
        headers: form_headers(),
        body: form_body(&[
            ("client_id", XAI_OAUTH_CLIENT_ID),
            ("scope", XAI_OAUTH_SCOPE),
            ("referrer", "pi"),
        ]),
    }
}

/// Parse the device authorization response (pi `parseDeviceCode` plus the
/// `requestFailure` guard from `requestDeviceCode`).
pub fn parse_xai_device_authorization_response(
    status: u16,
    body: &str,
) -> Result<XaiDeviceCode, String> {
    let object = parse_json_object(status, body)?;
    if !is_ok_status(status) {
        return Err(request_failure("device authorization", status, &object));
    }
    // RFC 8628 allows interval 0 (no minimum wait); fall back to the poller's
    // default instead of failing on non-positive or malformed values.
    let interval_seconds = object
        .get("interval")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| value as u64);
    let verification_uri_complete = match object
        .get("verification_uri_complete")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        Some(raw) => Some(validate_verification_uri(raw)?),
        None => None,
    };
    Ok(XaiDeviceCode {
        device_code: required_string(&object, "device_code")?,
        user_code: required_string(&object, "user_code")?,
        verification_uri: validate_verification_uri(&required_string(
            &object,
            "verification_uri",
        )?)?,
        verification_uri_complete,
        interval_seconds,
        expires_in_seconds: positive_number(&object, "expires_in")? as u64,
    })
}

pub async fn request_xai_device_authorization() -> Result<XaiDeviceCode, String> {
    request_xai_device_authorization_with_url(XAI_DEVICE_CODE_URL).await
}

pub async fn request_xai_device_authorization_with_url(url: &str) -> Result<XaiDeviceCode, String> {
    let request = build_xai_device_code_request_with_url(url);
    let response = send_oauth_http_request(&request).await?;
    parse_xai_device_authorization_response(response.status, &response.body)
}

pub fn build_xai_device_token_poll_request(device_code: &str) -> OAuthHttpRequest {
    build_xai_device_token_poll_request_with_url(device_code, XAI_TOKEN_URL)
}

pub fn build_xai_device_token_poll_request_with_url(
    device_code: &str,
    url: &str,
) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: url.to_owned(),
        method: "POST".to_owned(),
        headers: form_headers(),
        body: form_body(&[
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
            ("client_id", XAI_OAUTH_CLIENT_ID),
            ("device_code", device_code),
        ]),
    }
}

/// Classify one token-endpoint poll (pi `pollForTokens` poll callback).
pub fn parse_xai_device_token_poll_response(
    status: u16,
    body: &str,
    now_ms: i64,
) -> Result<DeviceCodePollResponse<OAuthCredential>, String> {
    let object = parse_json_object(status, body)?;
    if is_ok_status(status) {
        return Ok(DeviceCodePollResponse::Complete(
            credentials_from_token_response(&object, None, now_ms)?,
        ));
    }
    Ok(match object.get("error").and_then(Value::as_str) {
        Some("authorization_pending") => DeviceCodePollResponse::Pending,
        Some("slow_down") => DeviceCodePollResponse::SlowDown {
            interval_seconds: object
                .get("interval")
                .and_then(Value::as_f64)
                .filter(|value| value.is_finite() && *value > 0.0)
                .map(|value| value as u64),
        },
        Some("access_denied") | Some("authorization_denied") => DeviceCodePollResponse::Failed {
            message: "xAI device authorization was denied".to_owned(),
        },
        Some("expired_token") => DeviceCodePollResponse::Failed {
            message: "xAI device code expired".to_owned(),
        },
        _ => DeviceCodePollResponse::Failed {
            message: request_failure("device token polling", status, &object),
        },
    })
}

pub async fn login_xai_device_code_with_urls_with_sleeper<C, F, Fut>(
    urls: &XaiOAuthUrls,
    on_device_code: C,
    start_ms: i64,
    sleep: F,
) -> Result<OAuthCredential, String>
where
    C: FnOnce(&XaiDeviceCode),
    F: FnMut(u64) -> Fut,
    Fut: Future<Output = ()>,
{
    login_xai_device_code_with_urls_with_sleeper_and_abort(
        urls,
        on_device_code,
        start_ms,
        sleep,
        None,
    )
    .await
}

/// [`login_xai_device_code_with_urls_with_sleeper`] with a cooperative abort
/// flag (pi passes the login `signal` through to `pollOAuthDeviceCodeFlow`).
pub async fn login_xai_device_code_with_urls_with_sleeper_and_abort<C, F, Fut>(
    urls: &XaiOAuthUrls,
    on_device_code: C,
    start_ms: i64,
    mut sleep: F,
    abort_flag: Option<Arc<AtomicBool>>,
) -> Result<OAuthCredential, String>
where
    C: FnOnce(&XaiDeviceCode),
    F: FnMut(u64) -> Fut,
    Fut: Future<Output = ()>,
{
    let device = request_xai_device_authorization_with_url(&urls.device_code_url).await?;
    on_device_code(&device);
    let config = DeviceCodePollConfig {
        interval_seconds: device.interval_seconds,
        expires_in_seconds: Some(device.expires_in_seconds),
        wait_before_first_poll: true,
    };
    // The poller's time is virtual (it advances by exactly the delays it
    // sleeps), so track the same clock here: credential expiry then matches
    // pi's `Date.now()` at token-poll time.
    let elapsed_ms = AtomicI64::new(0);
    let request =
        build_xai_device_token_poll_request_with_url(&device.device_code, &urls.token_url);
    poll_device_code_flow_with_sleeper_and_abort(
        &config,
        start_ms,
        || async {
            let response = send_oauth_http_request(&request).await?;
            parse_xai_device_token_poll_response(
                response.status,
                &response.body,
                start_ms + elapsed_ms.load(Ordering::SeqCst),
            )
        },
        |delay_ms| {
            elapsed_ms.fetch_add(delay_ms as i64, Ordering::SeqCst);
            sleep(delay_ms)
        },
        abort_flag,
    )
    .await
}

pub fn build_xai_refresh_token_request(refresh_token: &str) -> OAuthHttpRequest {
    build_xai_refresh_token_request_with_url(refresh_token, XAI_TOKEN_URL)
}

pub fn build_xai_refresh_token_request_with_url(
    refresh_token: &str,
    url: &str,
) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: url.to_owned(),
        method: "POST".to_owned(),
        headers: form_headers(),
        body: form_body(&[
            ("grant_type", "refresh_token"),
            ("client_id", XAI_OAUTH_CLIENT_ID),
            ("refresh_token", refresh_token),
        ]),
    }
}

/// Parse the refresh response, keeping `previous_refresh_token` when the
/// server does not rotate it (pi `refreshXaiToken`).
pub fn parse_xai_refresh_token_response(
    status: u16,
    body: &str,
    previous_refresh_token: &str,
    now_ms: i64,
) -> Result<OAuthCredential, String> {
    let object = parse_json_object(status, body)?;
    if !is_ok_status(status) {
        return Err(request_failure("token refresh", status, &object));
    }
    credentials_from_token_response(&object, Some(previous_refresh_token), now_ms)
}

pub async fn refresh_xai_token(refresh_token: &str) -> Result<OAuthCredential, String> {
    refresh_xai_token_with_url_at(refresh_token, XAI_TOKEN_URL, now_millis()).await
}

pub async fn refresh_xai_token_with_url_at(
    refresh_token: &str,
    token_url: &str,
    now_ms: i64,
) -> Result<OAuthCredential, String> {
    let request = build_xai_refresh_token_request_with_url(refresh_token, token_url);
    let response = send_oauth_http_request(&request).await?;
    parse_xai_refresh_token_response(response.status, &response.body, refresh_token, now_ms)
}

/// xAI OAuth adapter (pi `xaiOAuth`).
#[derive(Debug, Clone)]
pub struct XaiOAuth {
    urls: XaiOAuthUrls,
}

impl XaiOAuth {
    pub fn new() -> Self {
        Self::with_urls(XaiOAuthUrls::default())
    }

    pub fn with_urls(urls: XaiOAuthUrls) -> Self {
        Self { urls }
    }
}

impl Default for XaiOAuth {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OAuthAuth for XaiOAuth {
    fn name(&self) -> &str {
        XAI_OAUTH_NAME
    }

    fn login_label(&self) -> Option<&str> {
        Some(XAI_OAUTH_LOGIN_LABEL)
    }

    async fn login(&self, interaction: &dyn AuthInteraction) -> Result<OAuthCredential, String> {
        // pi passes `interaction.signal` to the poller; ri's `AuthInteraction`
        // does not model an abort signal, so there is no flag to pass through
        // yet (same note as `radius.rs`).
        login_xai_device_code_with_urls_with_sleeper_and_abort(
            &self.urls,
            |device| {
                interaction.notify(AuthEvent::DeviceCode {
                    user_code: device.user_code.clone(),
                    verification_uri: device
                        .verification_uri_complete
                        .clone()
                        .unwrap_or_else(|| device.verification_uri.clone()),
                    interval_seconds: device.interval_seconds,
                    expires_in_seconds: Some(device.expires_in_seconds),
                });
            },
            now_millis(),
            |delay_ms| async move {
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            },
            None,
        )
        .await
    }

    async fn refresh(&self, credential: &OAuthCredential) -> Result<OAuthCredential, String> {
        refresh_xai_token_with_url_at(&credential.refresh, &self.urls.token_url, now_millis()).await
    }

    async fn to_auth(&self, credential: &OAuthCredential) -> Result<ModelAuth, String> {
        Ok(ModelAuth {
            api_key: Some(credential.access.clone()),
            ..Default::default()
        })
    }
}
