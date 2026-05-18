use crate::anthropic_oauth::OAuthHttpRequest;
use std::collections::BTreeMap;

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
        .strip_prefix("https://")
        .or_else(|| trimmed.strip_prefix("http://"))
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
        Some(host.to_owned())
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
    OAuthHttpRequest {
        url: urls.device_code_url,
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

pub fn build_github_copilot_access_token_poll_request(
    domain: &str,
    device_code: &str,
) -> OAuthHttpRequest {
    let urls = github_copilot_urls(domain);
    OAuthHttpRequest {
        url: urls.access_token_url,
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

pub fn build_github_copilot_refresh_request(
    refresh_token: &str,
    enterprise_domain: Option<&str>,
) -> OAuthHttpRequest {
    let domain = enterprise_domain.unwrap_or("github.com");
    let urls = github_copilot_urls(domain);
    let mut headers = copilot_headers();
    headers.insert("Accept".to_owned(), "application/json".to_owned());
    headers.insert(
        "Authorization".to_owned(),
        format!("Bearer {refresh_token}"),
    );
    OAuthHttpRequest {
        url: urls.copilot_token_url,
        method: "GET".to_owned(),
        headers,
        body: String::new(),
    }
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
