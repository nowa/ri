//! Behavior-parity feature gaps closed after the 2026-07-25 audit: GitHub
//! Copilot entitlement filtering, Bedrock/Vertex interactive logins, and
//! provider-SDK retry semantics on the non-codex chat paths.

use async_trait::async_trait;
use ri_llm_provider::auth::{
    ApiKeyAuth, ApiKeyCredential, AuthContext, AuthEvent, AuthInteraction, AuthPrompt, AuthResult,
    Credential, OAuthCredential,
};
use ri_llm_provider::*;
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::Mutex,
};

// ---------------------------------------------------------------------------
// GitHub Copilot entitlement listing (pi providers/github-copilot.ts)
// ---------------------------------------------------------------------------

#[test]
fn copilot_entitlement_listing_keeps_only_selectable_models() {
    let body = json!({
        "data": [
            { "id": "picker-on", "model_picker_enabled": true },
            { "id": "picker-off", "model_picker_enabled": false },
            { "id": "picker-missing" },
            {
                "id": "policy-disabled",
                "model_picker_enabled": true,
                "policy": { "state": "disabled" },
            },
            {
                "id": "policy-enabled",
                "model_picker_enabled": true,
                "policy": { "state": "enabled" },
            },
            {
                "id": "tools-unsupported",
                "model_picker_enabled": true,
                "capabilities": { "supports": { "tool_calls": false } },
            },
            {
                "id": "tools-supported",
                "model_picker_enabled": true,
                "capabilities": { "supports": { "tool_calls": true } },
            },
        ]
    })
    .to_string();

    assert_eq!(
        parse_available_github_copilot_model_ids(&body).expect("listing"),
        vec!["picker-on", "policy-enabled", "tools-supported"]
    );
}

#[test]
fn copilot_entitlement_listing_rejects_malformed_payloads() {
    for body in ["{}", r#"{"data":{}}"#, "not json"] {
        assert_eq!(
            parse_available_github_copilot_model_ids(body).expect_err("invalid"),
            "Invalid Copilot models response"
        );
    }
}

#[test]
fn copilot_entitlement_request_carries_api_version_and_auth() {
    let request = build_github_copilot_models_request(
        "tid=test;proxy-ep=proxy.individual.githubcopilot.com;",
        None,
        None,
    );
    assert_eq!(request.method, "GET");
    assert_eq!(
        request.url,
        "https://api.individual.githubcopilot.com/models"
    );
    assert_eq!(
        request
            .headers
            .get("X-GitHub-Api-Version")
            .map(String::as_str),
        Some("2026-06-01")
    );
    assert_eq!(
        request.headers.get("Accept").map(String::as_str),
        Some("application/json")
    );
    assert!(
        request
            .headers
            .get("Authorization")
            .is_some_and(|value| value.starts_with("Bearer tid=test"))
    );
    // The Copilot editor identity headers ride along.
    assert_eq!(
        request
            .headers
            .get("Copilot-Integration-Id")
            .map(String::as_str),
        Some("vscode-chat")
    );
}

#[test]
fn copilot_provider_filters_models_by_stored_entitlements() {
    let provider = builtin_provider("github-copilot").expect("copilot provider");
    let all_models = provider.get_models();
    assert!(all_models.len() > 1, "catalog should list several models");

    // No credential, api-key credentials, and malformed listings pass through.
    assert_eq!(
        provider.filter_models(all_models.clone(), None).len(),
        all_models.len()
    );
    let api_key = Credential::ApiKey(ApiKeyCredential {
        key: Some("token".to_owned()),
        env: Default::default(),
    });
    assert_eq!(
        provider
            .filter_models(all_models.clone(), Some(&api_key))
            .len(),
        all_models.len()
    );

    let oauth = |available: serde_json::Value| {
        Credential::OAuth(OAuthCredential {
            refresh: "refresh".to_owned(),
            access: "access".to_owned(),
            expires: 0,
            extra: [("availableModelIds".to_owned(), available)]
                .into_iter()
                .collect(),
        })
    };
    assert_eq!(
        provider
            .filter_models(all_models.clone(), Some(&oauth(json!("not-an-array"))))
            .len(),
        all_models.len()
    );
    assert_eq!(
        provider
            .filter_models(all_models.clone(), Some(&oauth(json!(["ok", 7]))))
            .len(),
        all_models.len()
    );

    let entitled = all_models[0].id.clone();
    let filtered = provider.filter_models(all_models, Some(&oauth(json!([entitled.clone()]))));
    assert_eq!(
        filtered
            .iter()
            .map(|model| model.id.clone())
            .collect::<Vec<_>>(),
        vec![entitled]
    );
}

// ---------------------------------------------------------------------------
// Bedrock / Vertex interactive logins (pi providers/{amazon-bedrock,
// google-vertex}.ts)
// ---------------------------------------------------------------------------

struct ScriptedInteraction {
    answers: Mutex<VecDeque<String>>,
    prompts: Mutex<Vec<AuthPrompt>>,
    events: Mutex<Vec<AuthEvent>>,
}

impl ScriptedInteraction {
    fn new(answers: &[&str]) -> Self {
        Self {
            answers: Mutex::new(answers.iter().map(|answer| (*answer).to_owned()).collect()),
            prompts: Mutex::new(Vec::new()),
            events: Mutex::new(Vec::new()),
        }
    }

    fn prompt_messages(&self) -> Vec<String> {
        self.prompts
            .lock()
            .expect("prompts")
            .iter()
            .map(|prompt| match prompt {
                AuthPrompt::Text { message, .. }
                | AuthPrompt::Secret { message, .. }
                | AuthPrompt::Select { message, .. }
                | AuthPrompt::ManualCode { message, .. } => message.clone(),
            })
            .collect()
    }
}

#[async_trait]
impl AuthInteraction for ScriptedInteraction {
    async fn prompt(&self, prompt: AuthPrompt) -> Result<String, String> {
        self.prompts.lock().expect("prompts").push(prompt);
        Ok(self
            .answers
            .lock()
            .expect("answers")
            .pop_front()
            .unwrap_or_default())
    }

    fn notify(&self, event: AuthEvent) {
        self.events.lock().expect("events").push(event);
    }
}

struct StubAuthContext {
    env: BTreeMap<String, String>,
    files: Vec<String>,
}

impl StubAuthContext {
    fn new(env: &[(&str, &str)], files: &[&str]) -> Self {
        Self {
            env: env
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
            files: files.iter().map(|path| (*path).to_owned()).collect(),
        }
    }
}

#[async_trait]
impl AuthContext for StubAuthContext {
    async fn env(&self, name: &str) -> Option<String> {
        self.env.get(name).cloned()
    }

    async fn file_exists(&self, path: &str) -> bool {
        self.files.iter().any(|known| known == path)
    }
}

fn provider_api_key_auth(provider_id: &str) -> std::sync::Arc<dyn ApiKeyAuth> {
    builtin_provider(provider_id)
        .expect("provider")
        .auth()
        .api_key
        .clone()
        .expect("api key auth")
}

#[tokio::test]
async fn bedrock_login_supports_token_profile_and_credential_chain() {
    let auth = provider_api_key_auth("amazon-bedrock");
    assert!(auth.supports_login());
    assert_eq!(auth.name(), "AWS credentials or bearer token");

    let interaction = ScriptedInteraction::new(&["bearer-token", "secret-token"]);
    let credential = auth.login(&interaction).await.expect("bearer login");
    assert_eq!(credential.key.as_deref(), Some("secret-token"));
    assert!(credential.env.is_empty());
    assert_eq!(
        interaction.prompt_messages(),
        vec![
            "Select Amazon Bedrock authentication method:",
            "Enter Amazon Bedrock bearer token"
        ]
    );

    let interaction = ScriptedInteraction::new(&["aws-profile", "prod"]);
    let credential = auth.login(&interaction).await.expect("profile login");
    assert_eq!(credential.key, None);
    assert_eq!(
        credential.env.get("AWS_PROFILE").map(String::as_str),
        Some("prod")
    );
    // pi notifies about the credential provider chain before prompting.
    assert!(matches!(
        interaction.events.lock().expect("events").first(),
        Some(AuthEvent::Info { links, .. }) if links.len() == 1
    ));

    let interaction = ScriptedInteraction::new(&["credential-chain", ""]);
    let credential = auth.login(&interaction).await.expect("chain login");
    assert_eq!(credential, ApiKeyCredential::default());

    let interaction = ScriptedInteraction::new(&["bogus"]);
    assert_eq!(
        auth.login(&interaction).await.expect_err("unknown method"),
        "Unknown Amazon Bedrock auth method: bogus"
    );
}

#[tokio::test]
async fn bedrock_resolve_reports_pi_credential_sources() {
    let auth = provider_api_key_auth("amazon-bedrock");
    let source =
        |result: Option<AuthResult>| result.and_then(|result| result.source).unwrap_or_default();

    let stored = ApiKeyCredential {
        key: Some("token".to_owned()),
        env: Default::default(),
    };
    let ctx = StubAuthContext::new(&[], &[]);
    assert_eq!(
        source(auth.resolve(&ctx, Some(&stored)).await.expect("resolve")),
        "stored credential"
    );

    let ctx = StubAuthContext::new(&[("AWS_BEARER_TOKEN_BEDROCK", "token")], &[]);
    assert_eq!(
        source(auth.resolve(&ctx, None).await.expect("resolve")),
        "AWS_BEARER_TOKEN_BEDROCK"
    );

    let profile_credential = ApiKeyCredential {
        key: None,
        env: [("AWS_PROFILE".to_owned(), "prod".to_owned())]
            .into_iter()
            .collect(),
    };
    let ctx = StubAuthContext::new(&[], &[]);
    let resolved = auth
        .resolve(&ctx, Some(&profile_credential))
        .await
        .expect("resolve")
        .expect("configured");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));
    assert_eq!(
        resolved.env.get("AWS_PROFILE").map(String::as_str),
        Some("prod")
    );
    assert_eq!(resolved.auth.api_key, None);

    let ctx = StubAuthContext::new(&[("AWS_PROFILE", "prod")], &[]);
    assert_eq!(
        source(auth.resolve(&ctx, None).await.expect("resolve")),
        "AWS_PROFILE"
    );

    let ctx = StubAuthContext::new(
        &[
            ("AWS_ACCESS_KEY_ID", "id"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
        ],
        &[],
    );
    assert_eq!(
        source(auth.resolve(&ctx, None).await.expect("resolve")),
        "AWS access keys"
    );

    let ctx = StubAuthContext::new(
        &[("AWS_CONTAINER_CREDENTIALS_FULL_URI", "http://169.254.170.2")],
        &[],
    );
    assert_eq!(
        source(auth.resolve(&ctx, None).await.expect("resolve")),
        "ECS task role"
    );

    // pi treats the web-identity token file alone as ambient credentials.
    let ctx = StubAuthContext::new(&[("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/token")], &[]);
    assert_eq!(
        source(auth.resolve(&ctx, None).await.expect("resolve")),
        "web identity token"
    );

    let ctx = StubAuthContext::new(&[], &[]);
    assert!(auth.resolve(&ctx, None).await.expect("resolve").is_none());
}

#[tokio::test]
async fn vertex_login_supports_api_key_adc_and_service_account() {
    let auth = provider_api_key_auth("google-vertex");
    assert!(auth.supports_login());
    assert_eq!(auth.name(), "Google Cloud credentials");

    let interaction = ScriptedInteraction::new(&["api-key", "key-value"]);
    let credential = auth.login(&interaction).await.expect("api key login");
    assert_eq!(credential.key.as_deref(), Some("key-value"));

    let interaction = ScriptedInteraction::new(&["adc", "my-project", "us-central1"]);
    let credential = auth.login(&interaction).await.expect("adc login");
    assert_eq!(credential.key, None);
    assert_eq!(
        credential
            .env
            .get("GOOGLE_CLOUD_PROJECT")
            .map(String::as_str),
        Some("my-project")
    );
    assert_eq!(
        credential
            .env
            .get("GOOGLE_CLOUD_LOCATION")
            .map(String::as_str),
        Some("us-central1")
    );
    assert!(
        !credential
            .env
            .contains_key("GOOGLE_APPLICATION_CREDENTIALS")
    );
    assert_eq!(
        interaction.prompt_messages(),
        vec![
            "Select Google Vertex AI authentication method:",
            "Enter Google Cloud project ID",
            "Enter Google Cloud location"
        ]
    );

    let interaction = ScriptedInteraction::new(&[
        "service-account",
        "my-project",
        "us-central1",
        "/tmp/sa.json",
    ]);
    let credential = auth
        .login(&interaction)
        .await
        .expect("service account login");
    assert_eq!(
        credential
            .env
            .get("GOOGLE_APPLICATION_CREDENTIALS")
            .map(String::as_str),
        Some("/tmp/sa.json")
    );

    let interaction = ScriptedInteraction::new(&["bogus"]);
    assert_eq!(
        auth.login(&interaction).await.expect_err("unknown method"),
        "Unknown Google Vertex AI auth method: bogus"
    );
}

#[tokio::test]
async fn vertex_resolve_prefers_keys_then_application_default_credentials() {
    let auth = provider_api_key_auth("google-vertex");

    let ctx = StubAuthContext::new(&[("GOOGLE_CLOUD_API_KEY", "env-key")], &[]);
    let resolved = auth
        .resolve(&ctx, None)
        .await
        .expect("resolve")
        .expect("configured");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("env-key"));
    assert_eq!(resolved.source.as_deref(), Some("GOOGLE_CLOUD_API_KEY"));

    let stored_key = ApiKeyCredential {
        key: Some("stored-key".to_owned()),
        env: Default::default(),
    };
    let resolved = auth
        .resolve(&ctx, Some(&stored_key))
        .await
        .expect("resolve")
        .expect("configured");
    assert_eq!(resolved.auth.api_key.as_deref(), Some("stored-key"));
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));

    // Ambient ADC needs the default credentials file plus project and location.
    let ctx = StubAuthContext::new(
        &[
            ("GOOGLE_CLOUD_PROJECT", "proj"),
            ("GOOGLE_CLOUD_LOCATION", "us-central1"),
        ],
        &["~/.config/gcloud/application_default_credentials.json"],
    );
    assert_eq!(
        auth.resolve(&ctx, None)
            .await
            .expect("resolve")
            .and_then(|result| result.source),
        Some("gcloud application default credentials".to_owned())
    );

    // GCLOUD_PROJECT is the documented fallback for the project id.
    let ctx = StubAuthContext::new(
        &[
            ("GCLOUD_PROJECT", "proj"),
            ("GOOGLE_CLOUD_LOCATION", "us-central1"),
        ],
        &["~/.config/gcloud/application_default_credentials.json"],
    );
    assert!(auth.resolve(&ctx, None).await.expect("resolve").is_some());

    // A stored service-account credential carries its own file and config.
    let stored_env = ApiKeyCredential {
        key: None,
        env: [
            (
                "GOOGLE_APPLICATION_CREDENTIALS".to_owned(),
                "/tmp/sa.json".to_owned(),
            ),
            ("GOOGLE_CLOUD_PROJECT".to_owned(), "proj".to_owned()),
            ("GOOGLE_CLOUD_LOCATION".to_owned(), "us-central1".to_owned()),
        ]
        .into_iter()
        .collect(),
    };
    let ctx = StubAuthContext::new(&[], &["/tmp/sa.json"]);
    let resolved = auth
        .resolve(&ctx, Some(&stored_env))
        .await
        .expect("resolve")
        .expect("configured");
    assert_eq!(resolved.source.as_deref(), Some("stored credential"));
    assert_eq!(
        resolved.env.get("GOOGLE_CLOUD_PROJECT").map(String::as_str),
        Some("proj")
    );

    // Missing location leaves the provider unconfigured.
    let ctx = StubAuthContext::new(
        &[("GOOGLE_CLOUD_PROJECT", "proj")],
        &["~/.config/gcloud/application_default_credentials.json"],
    );
    assert!(auth.resolve(&ctx, None).await.expect("resolve").is_none());
}

// ---------------------------------------------------------------------------
// Provider-SDK retry semantics (openai-node / @anthropic-ai/sdk)
// ---------------------------------------------------------------------------

#[test]
fn sdk_retry_status_decisions_match_the_provider_sdks() {
    for status in [408, 409, 429, 500, 503, 599] {
        assert!(sdk_should_retry_status(status, None), "{status}");
    }
    for status in [200, 400, 401, 403, 404, 422] {
        assert!(!sdk_should_retry_status(status, None), "{status}");
    }
    // `x-should-retry` overrides the status in both directions.
    assert!(sdk_should_retry_status(400, Some("true")));
    assert!(!sdk_should_retry_status(429, Some("false")));
    // Any other value falls through to the status decision.
    assert!(sdk_should_retry_status(429, Some("maybe")));
}

#[test]
fn sdk_default_retry_delay_follows_the_jittered_exponential_ladder() {
    // min(0.5s * 2^n, 8s), with jitter shaving off up to 25%.
    assert_eq!(sdk_default_retry_delay_ms(0, 0.0), 500);
    assert_eq!(sdk_default_retry_delay_ms(1, 0.0), 1_000);
    assert_eq!(sdk_default_retry_delay_ms(2, 0.0), 2_000);
    assert_eq!(sdk_default_retry_delay_ms(4, 0.0), 8_000);
    assert_eq!(sdk_default_retry_delay_ms(40, 0.0), 8_000);
    assert_eq!(sdk_default_retry_delay_ms(0, 0.25), 375);
    assert_eq!(sdk_default_retry_delay_ms(1, 0.1), 900);
    // Samples outside the SDK's range are clamped, never amplified.
    assert_eq!(sdk_default_retry_delay_ms(0, 1.0), 375);
    assert_eq!(sdk_default_retry_delay_ms(0, -1.0), 500);

    let sample = sdk_retry_jitter_sample();
    assert!((0.0..0.25).contains(&sample), "{sample}");
}

#[test]
fn sdk_retry_after_headers_win_over_the_default_ladder() {
    // `retry-after-ms` wins, parsed with JS parseFloat semantics.
    assert_eq!(
        sdk_retry_delay_ms(0, Some("1500"), Some("60"), 0, 0.0),
        1_500
    );
    assert_eq!(
        sdk_retry_delay_ms(0, Some("1500.75abc"), None, 0, 0.0),
        1_500
    );
    // Unparseable `retry-after-ms` falls through to `retry-after`.
    assert_eq!(
        sdk_retry_delay_ms(0, Some("soon"), Some("2.5"), 0, 0.0),
        2_500
    );
    // An HTTP date is honored relative to now, and past dates collapse to 0.
    assert_eq!(
        sdk_retry_delay_ms(
            0,
            None,
            Some("Wed, 21 Oct 2026 07:28:00 GMT"),
            1_792_567_620_000,
            0.0
        ),
        60_000
    );
    assert_eq!(
        sdk_retry_delay_ms(
            0,
            None,
            Some("Wed, 21 Oct 2026 07:28:00 GMT"),
            1_792_567_860_000,
            0.0
        ),
        0
    );
    // The SDKs run `parseFloat` first, so an ISO-8601 retry-after is read as
    // seconds (`parseFloat("2026-10-21T…") === 2026`) and never reaches
    // `Date.parse` — unlike pi's own codex path, which uses `Number()` and
    // does fall through to date parsing. Verified against node.
    assert_eq!(
        sdk_retry_delay_ms(
            0,
            None,
            Some("2026-10-21T07:28:00Z"),
            1_792_567_620_000,
            0.0
        ),
        2_026_000
    );
    // Empty headers are falsy and fall back to the ladder.
    assert_eq!(sdk_retry_delay_ms(1, Some(""), Some(""), 0, 0.0), 1_000);
    assert_eq!(sdk_retry_delay_ms(1, None, None, 0, 0.0), 1_000);
    // The SDKs apply no cap to header-provided delays.
    assert_eq!(sdk_retry_delay_ms(0, None, Some("600"), 0, 0.0), 600_000);
}

#[tokio::test]
async fn completions_stream_retries_retryable_statuses_when_opted_in() {
    let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = requests.clone();
    let server = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = server.local_addr().expect("addr");
    let task = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut socket, _) = server.accept().await.expect("accept");
            let mut buffer = [0u8; 2048];
            let _ = socket.read(&mut buffer).await;
            let attempt = seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let response = if attempt == 0 {
                // Retryable, with a short retry-after so the test stays fast.
                "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\nretry-after-ms: 1\r\ncontent-length: 2\r\n\r\n{}".to_owned()
            } else {
                let body = "data: {\"id\":\"chatcmpl-retry\",\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
                format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\n\r\n{body}",
                    body.len()
                )
            };
            let _ = socket.write_all(response.as_bytes()).await;
            let _ = socket.flush().await;
        }
    });

    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    model.base_url = format!("http://{addr}/v1");
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Hi"))],
        ..Default::default()
    };
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.max_retries = Some(2);

    let message = complete_simple(&model, context, options)
        .await
        .expect("retried request succeeds");
    assert_eq!(message.stop_reason, StopReason::Stop);
    assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 2);
    task.abort();
}

#[tokio::test]
async fn completions_stream_makes_one_attempt_by_default() {
    let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = requests.clone();
    let server = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = server.local_addr().expect("addr");
    let task = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut socket, _) = server.accept().await.expect("accept");
            let mut buffer = [0u8; 2048];
            let _ = socket.read(&mut buffer).await;
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = socket
                .write_all(
                    b"HTTP/1.1 503 Service Unavailable\r\ncontent-type: application/json\r\ncontent-length: 2\r\n\r\n{}",
                )
                .await;
            let _ = socket.flush().await;
        }
    });

    let mut model = get_model("openai", "gpt-4o-mini").expect("model");
    model.api = "openai-completions".to_owned();
    model.base_url = format!("http://{addr}/v1");
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Hi"))],
        ..Default::default()
    };
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());

    // pi defaults maxRetries to 0 on every SDK-backed path.
    let message = complete_simple(&model, context, options)
        .await
        .expect("provider failures surface as an error-stopped message");
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(
        message.error_message.as_deref(),
        Some("503 status code (no body)")
    );
    assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 1);
    task.abort();
}

#[tokio::test]
async fn non_retryable_statuses_are_not_retried_even_when_opted_in() {
    let requests = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let seen = requests.clone();
    let server = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = server.local_addr().expect("addr");
    let task = tokio::spawn(async move {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        loop {
            let (mut socket, _) = server.accept().await.expect("accept");
            let mut buffer = [0u8; 2048];
            let _ = socket.read(&mut buffer).await;
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let _ = socket
                .write_all(
                    b"HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: 2\r\n\r\n{}",
                )
                .await;
            let _ = socket.flush().await;
        }
    });

    let model = {
        let mut model = get_model("anthropic", "claude-sonnet-4-5").expect("model");
        model.base_url = format!("http://{addr}");
        model
    };
    let context = Context {
        messages: vec![Message::User(UserMessage::text("Hi"))],
        ..Default::default()
    };
    let mut options = SimpleStreamOptions::default();
    options.stream.api_key = Some("test-key".to_owned());
    options.stream.max_retries = Some(3);

    let message = complete_simple(&model, context, options)
        .await
        .expect("provider failures surface as an error-stopped message");
    assert_eq!(message.stop_reason, StopReason::Error);
    assert_eq!(message.error_message.as_deref(), Some("400 {}"));
    assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 1);
    task.abort();
}
