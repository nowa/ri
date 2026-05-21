use crate::{
    anthropic_oauth::{OAuthHttpRequest, send_oauth_http_request},
    models::get_models,
    types::now_millis,
};
use serde_json::Value;
use std::{collections::BTreeMap, future::Future, time::Duration};

pub const GITHUB_COPILOT_OAUTH_CLIENT_ID: &str = "Iv1.b507a08c87ecfe98";
pub const GITHUB_COPILOT_USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub const GITHUB_COPILOT_EDITOR_VERSION: &str = "vscode/1.107.0";
pub const GITHUB_COPILOT_EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub const GITHUB_COPILOT_INTEGRATION_ID: &str = "vscode-chat";

const INITIAL_POLL_INTERVAL_MULTIPLIER: f64 = 1.2;
const SLOW_DOWN_POLL_INTERVAL_MULTIPLIER: f64 = 1.4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubCopilotUrls {
    pub device_code_url: String,
    pub access_token_url: String,
    pub copilot_token_url: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubCopilotCredentials {
    pub refresh: String,
    pub access: String,
    pub expires: i64,
    pub enterprise_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubDeviceCode {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub verification_uri_complete: Option<String>,
    pub interval_seconds: u64,
    pub expires_in_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubCopilotLoginResult {
    pub device: GitHubDeviceCode,
    pub credentials: GitHubCopilotCredentials,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubCopilotModelPolicyOptions {
    pub enabled: bool,
    pub base_url: Option<String>,
    pub model_ids: Option<Vec<String>>,
}

impl Default for GitHubCopilotModelPolicyOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: None,
            model_ids: None,
        }
    }
}

impl GitHubCopilotModelPolicyOptions {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            base_url: None,
            model_ids: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitHubDeviceTokenResponse {
    AuthorizationPending,
    SlowDown {
        interval_seconds: Option<u64>,
    },
    Success {
        access_token: String,
    },
    Error {
        error: String,
        description: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitHubDevicePollOutcome {
    Poll { delay_ms: u64, poll_at_ms: i64 },
    Success { access_token: String },
    Failed { message: String },
    TimedOut { slow_down_seen: bool },
}

#[derive(Debug, Clone, PartialEq)]
pub struct GitHubDevicePollState {
    deadline_ms: i64,
    interval_ms: u64,
    interval_multiplier: f64,
    slow_down_responses: u32,
}

impl GitHubDevicePollState {
    pub fn new(now_ms: i64, interval_seconds: u64, expires_in_seconds: u64) -> Self {
        Self {
            deadline_ms: now_ms + expires_in_seconds as i64 * 1000,
            interval_ms: interval_seconds.saturating_mul(1000).max(1000),
            interval_multiplier: INITIAL_POLL_INTERVAL_MULTIPLIER,
            slow_down_responses: 0,
        }
    }

    pub fn next_poll(&self, now_ms: i64) -> GitHubDevicePollOutcome {
        if now_ms >= self.deadline_ms {
            return GitHubDevicePollOutcome::TimedOut {
                slow_down_seen: self.slow_down_responses > 0,
            };
        }
        let remaining_ms = (self.deadline_ms - now_ms).max(0) as u64;
        let adjusted_interval = (self.interval_ms as f64 * self.interval_multiplier).ceil() as u64;
        let delay_ms = adjusted_interval.min(remaining_ms);
        GitHubDevicePollOutcome::Poll {
            delay_ms,
            poll_at_ms: now_ms + delay_ms as i64,
        }
    }

    pub fn apply_response(
        &mut self,
        response: GitHubDeviceTokenResponse,
    ) -> GitHubDevicePollOutcome {
        match response {
            GitHubDeviceTokenResponse::AuthorizationPending => GitHubDevicePollOutcome::Poll {
                delay_ms: 0,
                poll_at_ms: 0,
            },
            GitHubDeviceTokenResponse::SlowDown { interval_seconds } => {
                self.slow_down_responses += 1;
                self.interval_ms = interval_seconds
                    .filter(|interval| *interval > 0)
                    .map(|interval| interval * 1000)
                    .unwrap_or_else(|| (self.interval_ms + 5000).max(1000));
                self.interval_multiplier = SLOW_DOWN_POLL_INTERVAL_MULTIPLIER;
                GitHubDevicePollOutcome::Poll {
                    delay_ms: 0,
                    poll_at_ms: 0,
                }
            }
            GitHubDeviceTokenResponse::Success { access_token } => {
                GitHubDevicePollOutcome::Success { access_token }
            }
            GitHubDeviceTokenResponse::Error { error, description } => {
                let suffix = description
                    .filter(|description| !description.is_empty())
                    .map(|description| format!(": {description}"))
                    .unwrap_or_default();
                GitHubDevicePollOutcome::Failed {
                    message: format!("Device flow failed: {error}{suffix}"),
                }
            }
        }
    }

    pub fn slow_down_seen(&self) -> bool {
        self.slow_down_responses > 0
    }
}

pub fn simulate_github_device_poll_times(
    start_ms: i64,
    interval_seconds: u64,
    expires_in_seconds: u64,
    responses: &[GitHubDeviceTokenResponse],
) -> (Vec<i64>, GitHubDevicePollOutcome) {
    let mut state = GitHubDevicePollState::new(start_ms, interval_seconds, expires_in_seconds);
    let mut now_ms = start_ms;
    let mut poll_times = Vec::new();

    for response in responses {
        match state.next_poll(now_ms) {
            GitHubDevicePollOutcome::Poll { poll_at_ms, .. } => {
                now_ms = poll_at_ms;
                poll_times.push(now_ms);
            }
            outcome => return (poll_times, outcome),
        }

        match state.apply_response(response.clone()) {
            GitHubDevicePollOutcome::Poll { .. } => {}
            outcome => return (poll_times, outcome),
        }
    }

    (poll_times, state.next_poll(now_ms))
}

pub fn normalize_github_domain(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let without_scheme = trimmed
        .find("://")
        .map(|scheme_end| &trimmed[scheme_end + 3..])
        .unwrap_or(trimmed);
    let host = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .split('@')
        .next_back()
        .unwrap_or_default()
        .split(':')
        .next()
        .unwrap_or_default()
        .trim()
        .trim_end_matches('.');
    if is_valid_domain(host) {
        Some(host.to_ascii_lowercase())
    } else {
        None
    }
}

pub fn github_copilot_urls(domain: &str) -> GitHubCopilotUrls {
    GitHubCopilotUrls {
        device_code_url: format!("https://{domain}/login/device/code"),
        access_token_url: format!("https://{domain}/login/oauth/access_token"),
        copilot_token_url: format!("https://api.{domain}/copilot_internal/v2/token"),
    }
}

pub fn build_github_copilot_device_code_request(domain: &str) -> OAuthHttpRequest {
    let urls = github_copilot_urls(domain);
    build_github_copilot_device_code_request_for_urls(&urls)
}

pub fn build_github_copilot_device_code_request_for_urls(
    urls: &GitHubCopilotUrls,
) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: urls.device_code_url.clone(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            (
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
            (
                "User-Agent".to_owned(),
                GITHUB_COPILOT_USER_AGENT.to_owned(),
            ),
        ]),
        body: form_body(&[
            ("client_id", GITHUB_COPILOT_OAUTH_CLIENT_ID),
            ("scope", "read:user"),
        ]),
    }
}

pub async fn request_github_copilot_device_code(domain: &str) -> Result<GitHubDeviceCode, String> {
    let urls = github_copilot_urls(domain);
    request_github_copilot_device_code_for_urls(&urls).await
}

pub async fn request_github_copilot_device_code_for_urls(
    urls: &GitHubCopilotUrls,
) -> Result<GitHubDeviceCode, String> {
    let request = build_github_copilot_device_code_request_for_urls(urls);
    let response = send_oauth_http_request(&request).await?;
    if response.status / 100 != 2 {
        return Err(format!(
            "GitHub Copilot device code request failed ({}): {}",
            response.status,
            if response.body.is_empty() {
                response.status_text
            } else {
                response.body
            }
        ));
    }
    parse_github_copilot_device_code_response(&response.body)
}

pub fn parse_github_copilot_device_code_response(
    response_body: &str,
) -> Result<GitHubDeviceCode, String> {
    let data: Value = serde_json::from_str(response_body).map_err(|error| {
        format!("GitHub Copilot device code response was invalid JSON: {error}")
    })?;
    Ok(GitHubDeviceCode {
        device_code: required_string(&data, "device_code", "GitHub Copilot device code")?,
        user_code: required_string(&data, "user_code", "GitHub Copilot device code")?,
        verification_uri: required_string(&data, "verification_uri", "GitHub Copilot device code")?,
        verification_uri_complete: data
            .get("verification_uri_complete")
            .and_then(Value::as_str)
            .map(str::to_owned),
        interval_seconds: required_u64(&data, "interval", "GitHub Copilot device code")?,
        expires_in_seconds: required_u64(&data, "expires_in", "GitHub Copilot device code")?,
    })
}

pub fn build_github_copilot_access_token_poll_request(
    domain: &str,
    device_code: &str,
) -> OAuthHttpRequest {
    let urls = github_copilot_urls(domain);
    build_github_copilot_access_token_poll_request_for_urls(&urls, device_code)
}

pub fn build_github_copilot_access_token_poll_request_for_urls(
    urls: &GitHubCopilotUrls,
    device_code: &str,
) -> OAuthHttpRequest {
    OAuthHttpRequest {
        url: urls.access_token_url.clone(),
        method: "POST".to_owned(),
        headers: BTreeMap::from([
            ("Accept".to_owned(), "application/json".to_owned()),
            (
                "Content-Type".to_owned(),
                "application/x-www-form-urlencoded".to_owned(),
            ),
            (
                "User-Agent".to_owned(),
                GITHUB_COPILOT_USER_AGENT.to_owned(),
            ),
        ]),
        body: form_body(&[
            ("client_id", GITHUB_COPILOT_OAUTH_CLIENT_ID),
            ("device_code", device_code),
            ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
        ]),
    }
}

pub async fn poll_github_copilot_access_token(
    domain: &str,
    device_code: &str,
) -> Result<GitHubDeviceTokenResponse, String> {
    let urls = github_copilot_urls(domain);
    poll_github_copilot_access_token_for_urls(&urls, device_code).await
}

pub async fn poll_github_copilot_access_token_for_urls(
    urls: &GitHubCopilotUrls,
    device_code: &str,
) -> Result<GitHubDeviceTokenResponse, String> {
    let request = build_github_copilot_access_token_poll_request_for_urls(urls, device_code);
    let response = send_oauth_http_request(&request).await?;
    if response.status / 100 != 2 {
        return Ok(GitHubDeviceTokenResponse::Error {
            error: format!("http_{}", response.status),
            description: Some(if response.body.is_empty() {
                response.status_text
            } else {
                response.body
            }),
        });
    }
    parse_github_copilot_device_token_response(&response.body)
}

pub async fn complete_github_copilot_device_flow_for_urls(
    urls: &GitHubCopilotUrls,
    device: &GitHubDeviceCode,
) -> Result<String, String> {
    complete_github_copilot_device_flow_for_urls_with_sleeper(
        urls,
        device,
        now_millis() as i64,
        |delay_ms| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        },
    )
    .await
}

pub async fn complete_github_copilot_device_flow_for_urls_with_sleeper<F, Fut>(
    urls: &GitHubCopilotUrls,
    device: &GitHubDeviceCode,
    start_ms: i64,
    mut sleep: F,
) -> Result<String, String>
where
    F: FnMut(u64) -> Fut,
    Fut: Future<Output = ()>,
{
    let mut state =
        GitHubDevicePollState::new(start_ms, device.interval_seconds, device.expires_in_seconds);
    let mut now_ms = start_ms;
    loop {
        match state.next_poll(now_ms) {
            GitHubDevicePollOutcome::Poll {
                delay_ms,
                poll_at_ms,
            } => {
                sleep(delay_ms).await;
                now_ms = poll_at_ms;
            }
            GitHubDevicePollOutcome::TimedOut { slow_down_seen } => {
                return Err(github_device_flow_timeout_message(slow_down_seen));
            }
            GitHubDevicePollOutcome::Failed { message } => return Err(message),
            GitHubDevicePollOutcome::Success { access_token } => return Ok(access_token),
        }

        let response = poll_github_copilot_access_token_for_urls(urls, &device.device_code).await?;
        match state.apply_response(response) {
            GitHubDevicePollOutcome::Poll { .. } => {}
            GitHubDevicePollOutcome::Success { access_token } => return Ok(access_token),
            GitHubDevicePollOutcome::Failed { message } => return Err(message),
            GitHubDevicePollOutcome::TimedOut { slow_down_seen } => {
                return Err(github_device_flow_timeout_message(slow_down_seen));
            }
        }
    }
}

pub async fn login_github_copilot_device_flow_for_urls<C>(
    urls: &GitHubCopilotUrls,
    enterprise_domain: Option<&str>,
    on_device_code: C,
) -> Result<GitHubCopilotLoginResult, String>
where
    C: FnOnce(&GitHubDeviceCode),
{
    login_github_copilot_device_flow_for_urls_with_sleeper(
        urls,
        enterprise_domain,
        on_device_code,
        now_millis() as i64,
        |delay_ms| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
        },
    )
    .await
}

pub async fn login_github_copilot_device_flow_for_urls_with_sleeper<C, F, Fut>(
    urls: &GitHubCopilotUrls,
    enterprise_domain: Option<&str>,
    on_device_code: C,
    start_ms: i64,
    sleep: F,
) -> Result<GitHubCopilotLoginResult, String>
where
    C: FnOnce(&GitHubDeviceCode),
    F: FnMut(u64) -> Fut,
    Fut: Future<Output = ()>,
{
    login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        urls,
        enterprise_domain,
        on_device_code,
        start_ms,
        sleep,
        &GitHubCopilotModelPolicyOptions::default(),
    )
    .await
}

pub async fn login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options<C, F, Fut>(
    urls: &GitHubCopilotUrls,
    enterprise_domain: Option<&str>,
    on_device_code: C,
    start_ms: i64,
    sleep: F,
    policy_options: &GitHubCopilotModelPolicyOptions,
) -> Result<GitHubCopilotLoginResult, String>
where
    C: FnOnce(&GitHubDeviceCode),
    F: FnMut(u64) -> Fut,
    Fut: Future<Output = ()>,
{
    let device = request_github_copilot_device_code_for_urls(urls).await?;
    on_device_code(&device);
    let refresh_token =
        complete_github_copilot_device_flow_for_urls_with_sleeper(urls, &device, start_ms, sleep)
            .await?;
    let credentials =
        refresh_github_copilot_token_for_urls_at(&refresh_token, urls, enterprise_domain, start_ms)
            .await?;
    let _ = enable_all_github_copilot_model_policies_with_options(
        &credentials.access,
        enterprise_domain,
        policy_options,
    )
    .await;
    Ok(GitHubCopilotLoginResult {
        device,
        credentials,
    })
}

pub fn parse_github_copilot_device_token_response(
    response_body: &str,
) -> Result<GitHubDeviceTokenResponse, String> {
    let data: Value = serde_json::from_str(response_body).map_err(|error| {
        format!("GitHub Copilot access token poll response was invalid JSON: {error}")
    })?;
    if let Some(access_token) = data.get("access_token").and_then(Value::as_str) {
        return Ok(GitHubDeviceTokenResponse::Success {
            access_token: access_token.to_owned(),
        });
    }
    let error = data
        .get("error")
        .and_then(Value::as_str)
        .unwrap_or("unknown_error");
    let description = data
        .get("error_description")
        .and_then(Value::as_str)
        .map(str::to_owned);
    Ok(match error {
        "authorization_pending" => GitHubDeviceTokenResponse::AuthorizationPending,
        "slow_down" => GitHubDeviceTokenResponse::SlowDown {
            interval_seconds: data.get("interval").and_then(Value::as_u64),
        },
        _ => GitHubDeviceTokenResponse::Error {
            error: error.to_owned(),
            description,
        },
    })
}

pub fn build_github_copilot_refresh_request(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> OAuthHttpRequest {
    let domain = enterprise_domain.unwrap_or("github.com");
    let urls = github_copilot_urls(domain);
    build_github_copilot_refresh_request_for_urls(refresh_token, &urls)
}

pub fn build_github_copilot_refresh_request_for_urls(
    refresh_token: &str,
    urls: &GitHubCopilotUrls,
) -> OAuthHttpRequest {
    let mut headers = copilot_headers();
    headers.insert("Accept".to_owned(), "application/json".to_owned());
    headers.insert(
        "Authorization".to_owned(),
        format!("Bearer {refresh_token}"),
    );
    OAuthHttpRequest {
        url: urls.copilot_token_url.clone(),
        method: "GET".to_owned(),
        headers,
        body: String::new(),
    }
}

pub async fn refresh_github_copilot_token_at(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
    now_millis: i64,
) -> Result<GitHubCopilotCredentials, String> {
    let domain = enterprise_domain.unwrap_or("github.com");
    let urls = github_copilot_urls(domain);
    refresh_github_copilot_token_for_urls_at(refresh_token, &urls, enterprise_domain, now_millis)
        .await
}

pub async fn refresh_github_copilot_token_for_urls_at(
    refresh_token: &str,
    urls: &GitHubCopilotUrls,
    enterprise_domain: Option<&str>,
    now_millis: i64,
) -> Result<GitHubCopilotCredentials, String> {
    let request = build_github_copilot_refresh_request_for_urls(refresh_token, urls);
    let response = send_oauth_http_request(&request).await?;
    if response.status / 100 != 2 {
        return Err(format!(
            "GitHub Copilot token refresh failed ({}): {}",
            response.status,
            if response.body.is_empty() {
                response.status_text
            } else {
                response.body
            }
        ));
    }
    parse_github_copilot_refresh_response(
        refresh_token,
        &response.body,
        enterprise_domain,
        now_millis,
    )
}

pub async fn refresh_github_copilot_token(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> Result<GitHubCopilotCredentials, String> {
    refresh_github_copilot_token_at(refresh_token, enterprise_domain, now_millis() as i64).await
}

pub fn build_github_copilot_model_policy_request(
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
) -> OAuthHttpRequest {
    build_github_copilot_model_policy_request_with_base_url(
        token,
        model_id,
        enterprise_domain,
        None,
    )
}

pub fn build_github_copilot_model_policy_request_with_base_url(
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
    base_url: Option<&str>,
) -> OAuthHttpRequest {
    let base_url = base_url
        .map(str::to_owned)
        .unwrap_or_else(|| github_copilot_base_url(Some(token), enterprise_domain));
    let mut headers = copilot_headers();
    headers.insert("Content-Type".to_owned(), "application/json".to_owned());
    headers.insert("Authorization".to_owned(), format!("Bearer {token}"));
    headers.insert("openai-intent".to_owned(), "chat-policy".to_owned());
    headers.insert("x-interaction-type".to_owned(), "chat-policy".to_owned());
    OAuthHttpRequest {
        url: format!(
            "{}/models/{model_id}/policy",
            base_url.trim_end_matches('/')
        ),
        method: "POST".to_owned(),
        headers,
        body: serde_json::json!({ "state": "enabled" }).to_string(),
    }
}

pub async fn enable_github_copilot_model_policy(
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
) -> bool {
    enable_github_copilot_model_policy_with_base_url(token, model_id, enterprise_domain, None).await
}

pub async fn enable_github_copilot_model_policy_with_base_url(
    token: &str,
    model_id: &str,
    enterprise_domain: Option<&str>,
    base_url: Option<&str>,
) -> bool {
    let request = build_github_copilot_model_policy_request_with_base_url(
        token,
        model_id,
        enterprise_domain,
        base_url,
    );
    send_oauth_http_request(&request)
        .await
        .map(|response| response.status / 100 == 2)
        .unwrap_or(false)
}

pub async fn enable_all_github_copilot_model_policies(
    token: &str,
    enterprise_domain: Option<&str>,
) -> Vec<(String, bool)> {
    enable_all_github_copilot_model_policies_with_options(
        token,
        enterprise_domain,
        &GitHubCopilotModelPolicyOptions::default(),
    )
    .await
}

pub async fn enable_all_github_copilot_model_policies_with_options(
    token: &str,
    enterprise_domain: Option<&str>,
    options: &GitHubCopilotModelPolicyOptions,
) -> Vec<(String, bool)> {
    if !options.enabled {
        return Vec::new();
    }
    let model_ids = options.model_ids.clone().unwrap_or_else(|| {
        get_models("github-copilot")
            .into_iter()
            .map(|model| model.id)
            .collect()
    });
    let mut results = Vec::with_capacity(model_ids.len());
    for model_id in model_ids {
        let success = enable_github_copilot_model_policy_with_base_url(
            token,
            &model_id,
            enterprise_domain,
            options.base_url.as_deref(),
        )
        .await;
        results.push((model_id, success));
    }
    results
}

pub fn parse_github_copilot_refresh_response(
    refresh_token: &str,
    response_body: &str,
    enterprise_domain: Option<&str>,
    _now_millis: i64,
) -> Result<GitHubCopilotCredentials, String> {
    let data: Value = serde_json::from_str(response_body)
        .map_err(|error| format!("GitHub Copilot token response was invalid JSON: {error}"))?;
    let token = data
        .get("token")
        .and_then(Value::as_str)
        .ok_or_else(|| "GitHub Copilot token response was missing token".to_owned())?;
    let expires_at_seconds = data
        .get("expires_at")
        .and_then(Value::as_i64)
        .ok_or_else(|| "GitHub Copilot token response was missing expires_at".to_owned())?;
    Ok(parse_github_copilot_token_response(
        refresh_token,
        token,
        expires_at_seconds,
        enterprise_domain,
    ))
}

pub fn parse_github_copilot_token_response(
    refresh_token: &str,
    token: &str,
    expires_at_seconds: i64,
    enterprise_domain: Option<&str>,
) -> GitHubCopilotCredentials {
    GitHubCopilotCredentials {
        refresh: refresh_token.to_owned(),
        access: token.to_owned(),
        expires: expires_at_seconds * 1000 - 5 * 60 * 1000,
        enterprise_url: enterprise_domain.map(str::to_owned),
    }
}

pub fn github_copilot_base_url(token: Option<&str>, enterprise_domain: Option<&str>) -> String {
    if let Some(token) = token
        && let Some(base_url) = github_copilot_base_url_from_token(token)
    {
        return base_url;
    }
    if let Some(domain) = enterprise_domain {
        return format!("https://copilot-api.{domain}");
    }
    "https://api.individual.githubcopilot.com".to_owned()
}

pub fn github_copilot_base_url_from_token(token: &str) -> Option<String> {
    let (_, tail) = token.split_once("proxy-ep=")?;
    let proxy_host = tail.split(';').next()?;
    if proxy_host.is_empty() {
        return None;
    }
    let api_host = proxy_host
        .strip_prefix("proxy.")
        .map(|host| format!("api.{host}"))
        .unwrap_or_else(|| proxy_host.to_owned());
    Some(format!("https://{api_host}"))
}

fn copilot_headers() -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "User-Agent".to_owned(),
            GITHUB_COPILOT_USER_AGENT.to_owned(),
        ),
        (
            "Editor-Version".to_owned(),
            GITHUB_COPILOT_EDITOR_VERSION.to_owned(),
        ),
        (
            "Editor-Plugin-Version".to_owned(),
            GITHUB_COPILOT_EDITOR_PLUGIN_VERSION.to_owned(),
        ),
        (
            "Copilot-Integration-Id".to_owned(),
            GITHUB_COPILOT_INTEGRATION_ID.to_owned(),
        ),
    ])
}

fn is_valid_domain(host: &str) -> bool {
    !host.is_empty()
        && host.split('.').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .all(|ch| ch.is_ascii_alphanumeric() || ch == '-')
        })
}

fn required_string(value: &Value, field: &str, label: &str) -> Result<String, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{label} response was missing {field}"))
}

fn required_u64(value: &Value, field: &str, label: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} response was missing {field}"))
}

fn github_device_flow_timeout_message(slow_down_seen: bool) -> String {
    if slow_down_seen {
        "Device flow timed out after one or more slow_down responses. This is often caused by clock drift in WSL or VM environments. Please sync or restart the VM clock and try again.".to_owned()
    } else {
        "Device flow timed out".to_owned()
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
