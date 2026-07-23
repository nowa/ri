//! Parity tests for the 2026-07 audit wave: Mistral prompt cache keys,
//! GitHub Copilot verification URI hardening, device-code poll cancellation,
//! OpenAI Codex select/cancel flows, and scoped proxy env overrides.
//!
//! Ports of pi `mistral-reasoning-mode.test.ts`, `github-copilot-oauth.test.ts`,
//! `oauth-device-code.test.ts`, `openai-codex-oauth.test.ts`, and
//! `node-http-proxy.test.ts` GAP rows.

use async_trait::async_trait;
use ri_llm_provider::auth::{
    AuthEvent, AuthInteraction, AuthPrompt, AuthPromptOption, login_builtin_oauth_provider,
};
use ri_llm_provider::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

// --- helpers (copied from tests/provider_core.rs, trimmed) ------------------

fn user_context(text: &str) -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(text))],
        ..Default::default()
    }
}

const CODEX_TEST_JWT_PAYLOAD: &str =
    "eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjX3Rlc3QifX0=";

fn codex_test_token() -> String {
    format!("aaa.{CODEX_TEST_JWT_PAYLOAD}.bbb")
}

fn codex_oauth_token_response(access_token: &str, refresh_token: &str, expires_in: i64) -> String {
    json!({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
    })
    .to_string()
}

fn request_is_complete(request: &[u8]) -> bool {
    let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n") else {
        return false;
    };
    let headers = String::from_utf8_lossy(&request[..header_end]);
    let content_length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|value| value.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    request.len() >= header_end + 4 + content_length
}

async fn mock_json_server(body: impl Into<String>) -> (String, tokio::task::JoinHandle<String>) {
    let (url, task) = mock_json_status_sequence_server_owned(vec![(200, "OK", body.into())]).await;
    let task = tokio::spawn(async move {
        task.await
            .expect("mock server task")
            .into_iter()
            .next()
            .expect("one request")
    });
    (url, task)
}

async fn mock_json_status_sequence_server(
    responses: Vec<(u16, &'static str, &'static str)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    mock_json_status_sequence_server_owned(
        responses
            .into_iter()
            .map(|(status, reason, body)| (status, reason, body.to_owned()))
            .collect(),
    )
    .await
}

async fn mock_json_status_sequence_server_owned(
    responses: Vec<(u16, &'static str, String)>,
) -> (String, tokio::task::JoinHandle<Vec<String>>) {
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock server");
    let addr = listener.local_addr().expect("local addr");
    let task = tokio::spawn(async move {
        let mut requests = Vec::new();
        for (status, reason, body) in responses {
            let (mut socket, _) = listener.accept().await.expect("accept request");
            let mut request = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                let n = socket.read(&mut buf).await.expect("read request");
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                if request_is_complete(&request) {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body,
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
            requests.push(String::from_utf8_lossy(&request).into_owned());
        }
        requests
    });
    (format!("http://{addr}"), task)
}

// --- mistral-reasoning-mode.test.ts: prompt cache key -----------------------

#[test]
fn mistral_simple_payload_uses_session_id_as_prompt_cache_key() {
    let model = get_model("mistral", "mistral-large-latest").expect("mistral model");
    let context = user_context("Hello");

    let mut options = SimpleStreamOptions::default();
    options.stream.session_id = Some("session-123".to_owned());
    let payload = build_mistral_simple_payload(&model, &context, options);

    assert_eq!(
        payload.get("promptCacheKey").and_then(Value::as_str),
        Some("session-123")
    );
}

#[test]
fn mistral_simple_payload_omits_prompt_cache_key_when_cache_retention_disabled() {
    let model = get_model("mistral", "mistral-large-latest").expect("mistral model");
    let context = user_context("Hello");

    let mut options = SimpleStreamOptions::default();
    options.stream.session_id = Some("session-123".to_owned());
    options.stream.cache_retention = Some(CacheRetention::None);
    let payload = build_mistral_simple_payload(&model, &context, options);
    assert!(payload.get("promptCacheKey").is_none());

    // No session id means no cache key either.
    let payload = build_mistral_simple_payload(&model, &context, SimpleStreamOptions::default());
    assert!(payload.get("promptCacheKey").is_none());
}

// --- github-copilot-oauth.test.ts: verification_uri hardening ---------------

#[tokio::test]
async fn copilot_rejects_non_http_verification_uri_before_device_code_callback() {
    // A malicious enterprise OAuth server could return a verification_uri that
    // the browser launcher would otherwise hand to the OS. Ensure such values
    // are rejected at the deserialization boundary.
    let (device_url, _device_task) = mock_json_server(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"$(id>/tmp/pwned)","interval":1,"expires_in":900}"#,
    )
    .await;
    let urls = GitHubCopilotUrls {
        device_code_url: device_url,
        access_token_url: "http://127.0.0.1:9/token".to_owned(),
        copilot_token_url: "http://127.0.0.1:9/copilot".to_owned(),
    };
    let on_device_code_called = Arc::new(AtomicBool::new(false));
    let on_device_code_called_ref = on_device_code_called.clone();

    let err = login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        &urls,
        None,
        move |_device| {
            on_device_code_called_ref.store(true, Ordering::SeqCst);
        },
        0,
        |_delay_ms| std::future::ready(()),
        &GitHubCopilotModelPolicyOptions::disabled(),
    )
    .await
    .expect_err("untrusted verification_uri must fail the login");

    assert!(
        err.contains("Untrusted verification_uri"),
        "unexpected error: {err}"
    );
    assert!(!on_device_code_called.load(Ordering::SeqCst));

    // Parsable non-http(s) schemes are equally untrusted.
    let err = parse_github_copilot_device_code_response(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"javascript:alert(1)","interval":1,"expires_in":900}"#,
    )
    .expect_err("non-http scheme must be rejected");
    assert_eq!(err, "Untrusted verification_uri in device code response");
}

#[tokio::test]
async fn copilot_normalizes_verification_uri_before_device_code_callback() {
    let raw_verification_uri = "https://github.com/login/\u{1b}]8;;evil";
    let normalized_verification_uri = "https://github.com/login/%1B]8;;evil";
    assert_ne!(normalized_verification_uri, raw_verification_uri);

    let (device_url, _device_task) = mock_json_server(
        r#"{"device_code":"device-code","user_code":"ABCD-EFGH","verification_uri":"https://github.com/login/\u001b]8;;evil","interval":1,"expires_in":900}"#,
    )
    .await;
    let (poll_url, _poll_task) = mock_json_status_sequence_server(vec![(
        200,
        "OK",
        r#"{"access_token":"ghu_refresh_token"}"#,
    )])
    .await;
    let (refresh_url, _refresh_task) = mock_json_server(
        r#"{"token":"tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;","expires_at":9999999999}"#,
    )
    .await;
    let urls = GitHubCopilotUrls {
        device_code_url: device_url,
        access_token_url: poll_url,
        copilot_token_url: refresh_url,
    };
    let seen_verification_uri = Arc::new(Mutex::new(None::<String>));
    let seen_verification_uri_ref = seen_verification_uri.clone();

    let result = login_github_copilot_device_flow_for_urls_with_sleeper_and_policy_options(
        &urls,
        None,
        move |device| {
            *seen_verification_uri_ref.lock().expect("seen uri") =
                Some(device.verification_uri.clone());
        },
        0,
        |_delay_ms| std::future::ready(()),
        &GitHubCopilotModelPolicyOptions::disabled(),
    )
    .await
    .expect("copilot login");

    assert_eq!(
        seen_verification_uri.lock().expect("seen uri").as_deref(),
        Some(normalized_verification_uri)
    );
    assert_eq!(result.device.verification_uri, normalized_verification_uri);
    assert_eq!(result.credentials.refresh, "ghu_refresh_token");
}

// --- oauth-device-code.test.ts: cancellation ---------------------------------

#[tokio::test]
async fn device_code_poll_flow_cancels_before_the_first_poll() {
    let abort_flag = Arc::new(AtomicBool::new(true));
    let config = DeviceCodePollConfig {
        interval_seconds: Some(5),
        expires_in_seconds: Some(30),
        wait_before_first_poll: true,
    };

    let err = poll_device_code_flow_with_sleeper_and_abort::<String, _, _, _, _>(
        &config,
        0,
        || async { panic!("poll must not run after cancellation") },
        |_delay_ms| -> std::future::Ready<()> {
            panic!("initial wait must not run after cancellation")
        },
        Some(abort_flag),
    )
    .await
    .expect_err("cancelled before the first poll");

    assert_eq!(err, "Login cancelled");
    assert_eq!(err, DEVICE_FLOW_CANCEL_MESSAGE);
}

#[tokio::test]
async fn device_code_poll_flow_cancels_an_in_flight_wait() {
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_ref = abort_flag.clone();
    let poll_count = Arc::new(AtomicUsize::new(0));
    let poll_count_ref = poll_count.clone();
    let config = DeviceCodePollConfig {
        interval_seconds: Some(5),
        expires_in_seconds: Some(30),
        wait_before_first_poll: false,
    };

    let err = poll_device_code_flow_with_sleeper_and_abort::<String, _, _, _, _>(
        &config,
        0,
        move || {
            let poll_count = poll_count_ref.clone();
            async move {
                poll_count.fetch_add(1, Ordering::SeqCst);
                Ok(DeviceCodePollResponse::Pending)
            }
        },
        move |_delay_ms| {
            // The user aborts while the flow waits between polls.
            abort_flag_ref.store(true, Ordering::SeqCst);
            std::future::ready(())
        },
        Some(abort_flag),
    )
    .await
    .expect_err("cancelled while waiting");

    assert_eq!(err, "Login cancelled");
    assert_eq!(poll_count.load(Ordering::SeqCst), 1);
}

// --- openai-codex-oauth.test.ts: select flow, cancellation -------------------

#[tokio::test]
async fn openai_codex_device_code_flow_reports_codex_verification_details() {
    let (user_code_url, _user_code_task) = mock_json_server(
        r#"{"device_auth_id":"device-auth-id","user_code":"WXYZ-7890","interval":"5"}"#,
    )
    .await;
    let (token_poll_url, _token_poll_task) = mock_json_status_sequence_server(vec![(
        200,
        "OK",
        r#"{"authorization_code":"oauth-code","code_challenge":"device-code-challenge","code_verifier":"device-code-verifier"}"#,
    )])
    .await;
    let (exchange_url, exchange_task) = mock_json_server(codex_oauth_token_response(
        &codex_test_token(),
        "refresh-token",
        3600,
    ))
    .await;
    let device_events = Arc::new(Mutex::new(Vec::<AuthEvent>::new()));
    let device_events_ref = device_events.clone();
    let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));
    let sleeps_ref = sleeps.clone();

    let credentials = login_openai_codex_device_code_with_urls_with_sleeper(
        &user_code_url,
        &token_poll_url,
        &exchange_url,
        // The interactive select flow notifies exactly this event for the
        // "device_code" choice (auth/interactive.rs).
        move |device| {
            device_events_ref
                .lock()
                .expect("device events")
                .push(AuthEvent::DeviceCode {
                    user_code: device.user_code.clone(),
                    verification_uri: OPENAI_CODEX_DEVICE_VERIFICATION_URI.to_owned(),
                    interval_seconds: Some(device.interval_seconds),
                    expires_in_seconds: Some(OPENAI_CODEX_DEVICE_CODE_TIMEOUT_SECONDS),
                });
        },
        0,
        move |delay_ms| {
            sleeps_ref.lock().expect("sleeps").push(delay_ms);
            std::future::ready(())
        },
    )
    .await
    .expect("device code login");

    assert_eq!(
        device_events.lock().expect("device events").as_slice(),
        &[AuthEvent::DeviceCode {
            user_code: "WXYZ-7890".to_owned(),
            verification_uri: "https://auth.openai.com/codex/device".to_owned(),
            interval_seconds: Some(5),
            expires_in_seconds: Some(900),
        }]
    );
    assert!(sleeps.lock().expect("sleeps").is_empty());

    let exchange_request = exchange_task.await.expect("exchange task");
    assert!(exchange_request.contains("grant_type=authorization_code"));
    assert!(exchange_request.contains("client_id=app_EMoamEEZ73f0CkXaXp7hrann"));
    assert!(exchange_request.contains("code=oauth-code"));
    assert!(exchange_request.contains("code_verifier=device-code-verifier"));
    assert!(
        exchange_request
            .contains("redirect_uri=https%3A%2F%2Fauth.openai.com%2Fdeviceauth%2Fcallback")
    );

    assert_eq!(credentials.access, codex_test_token());
    assert_eq!(credentials.refresh, "refresh-token");
    assert_eq!(credentials.expires, 3_600_000);
    assert_eq!(
        credentials.extra.get("accountId").and_then(Value::as_str),
        Some("acc_test")
    );
}

struct CancellingAuthInteraction {
    prompts: Mutex<Vec<AuthPrompt>>,
}

#[async_trait]
impl AuthInteraction for CancellingAuthInteraction {
    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, String> {
        self.prompts.lock().expect("prompts").push(prompt);
        Err("Login cancelled".to_owned())
    }

    fn notify(&self, _event: AuthEvent) {}
}

#[tokio::test]
async fn openai_codex_login_cancels_when_method_selection_is_cancelled() {
    let interaction = CancellingAuthInteraction {
        prompts: Mutex::new(Vec::new()),
    };

    let err = login_builtin_oauth_provider("openai-codex", &interaction)
        .await
        .expect_err("cancelled selection must propagate");
    assert_eq!(err, "Login cancelled");

    // The browser/device select prompt is offered before anything else.
    let prompts = interaction.prompts.lock().expect("prompts");
    assert_eq!(prompts.len(), 1);
    match &prompts[0] {
        AuthPrompt::Select { message, options } => {
            assert_eq!(message, "Select OpenAI Codex login method:");
            assert_eq!(
                options,
                &[
                    AuthPromptOption {
                        id: "browser".to_owned(),
                        label: "Browser login (default)".to_owned(),
                        description: None,
                    },
                    AuthPromptOption {
                        id: "device_code".to_owned(),
                        label: "Device code login (headless)".to_owned(),
                        description: None,
                    },
                ]
            );
        }
        other => panic!("expected a select prompt, got {other:?}"),
    }
}

#[tokio::test]
async fn openai_codex_device_code_flow_cancels_while_waiting() {
    let (user_code_url, _user_code_task) = mock_json_server(
        r#"{"device_auth_id":"device-auth-id","user_code":"ABCD-1234","interval":"5"}"#,
    )
    .await;
    let (token_poll_url, token_poll_task) = mock_json_status_sequence_server(vec![(
        403,
        "Forbidden",
        r#"{"error":{"message":"Device authorization is pending. Please try again.","type":"invalid_request_error","param":null,"code":"deviceauth_authorization_pending"}}"#,
    )])
    .await;
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_ref = abort_flag.clone();

    let err = login_openai_codex_device_code_with_urls_with_sleeper_and_abort(
        &user_code_url,
        &token_poll_url,
        "http://127.0.0.1:9/never-exchanged",
        |_device| {},
        0,
        move |_delay_ms| {
            // The user aborts while the flow waits for the next poll.
            abort_flag_ref.store(true, Ordering::SeqCst);
            std::future::ready(())
        },
        Some(abort_flag),
    )
    .await
    .expect_err("cancelled while waiting");

    assert_eq!(err, "Login cancelled");
    let poll_requests = token_poll_task.await.expect("token poll task");
    assert_eq!(poll_requests.len(), 1);
}

// --- node-http-proxy.test.ts: scoped env override ----------------------------

const PROXY_ENV_KEYS: &[&str] = &[
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
    "npm_config_http_proxy",
    "npm_config_https_proxy",
    "npm_config_proxy",
    "npm_config_no_proxy",
    "npm_config_all_proxy",
    "NPM_CONFIG_HTTP_PROXY",
    "NPM_CONFIG_HTTPS_PROXY",
    "NPM_CONFIG_PROXY",
    "NPM_CONFIG_NO_PROXY",
    "NPM_CONFIG_ALL_PROXY",
];

struct EnvGuard {
    values: Vec<(&'static str, Option<String>)>,
}

impl EnvGuard {
    fn clearing(keys: &[&'static str]) -> Self {
        let values = keys
            .iter()
            .map(|key| (*key, std::env::var(key).ok()))
            .collect::<Vec<_>>();
        for key in keys {
            remove_env(key);
        }
        Self { values }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        for (key, value) in &self.values {
            match value {
                Some(value) => set_env(key, value),
                None => remove_env(key),
            }
        }
    }
}

fn set_env(key: &str, value: &str) {
    // This test file mutates only proxy env keys no other test here reads.
    unsafe {
        std::env::set_var(key, value);
    }
}

fn remove_env(key: &str) {
    // This test file mutates only proxy env keys no other test here reads.
    unsafe {
        std::env::remove_var(key);
    }
}

#[test]
fn node_http_proxy_prefers_scoped_proxy_env_before_process_env() {
    let _guard = EnvGuard::clearing(PROXY_ENV_KEYS);
    set_env("https_proxy", "http://process-proxy.example:8080");
    let target = "https://bedrock-runtime.us-east-1.amazonaws.com";

    let scoped = BTreeMap::from([(
        "HTTPS_PROXY".to_owned(),
        "http://scoped-proxy.example:8080".to_owned(),
    )]);
    let proxy = resolve_http_proxy_url_for_target_with_env(target, &scoped)
        .expect("resolve proxy")
        .expect("proxy configured");
    assert_eq!(proxy.as_str(), "http://scoped-proxy.example:8080/");

    // Without scoped overrides the process env still applies.
    let proxy = resolve_http_proxy_url_for_target_with_env(target, &BTreeMap::new())
        .expect("resolve proxy")
        .expect("proxy configured");
    assert_eq!(proxy.as_str(), "http://process-proxy.example:8080/");

    // Scoped no_proxy exclusions win over scoped proxy URLs too.
    let scoped = BTreeMap::from([
        (
            "HTTPS_PROXY".to_owned(),
            "http://scoped-proxy.example:8080".to_owned(),
        ),
        (
            "no_proxy".to_owned(),
            "bedrock-runtime.us-east-1.amazonaws.com".to_owned(),
        ),
    ]);
    assert_eq!(
        resolve_http_proxy_url_for_target_with_env(target, &scoped),
        Ok(None)
    );
}
