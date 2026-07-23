//! Parity tests for the xAI OAuth device-code flow.
//!
//! Ports pi `xai-oauth.test.ts` (13 cases) and the xAI row of
//! `oauth-auth.test.ts` ("xAI toAuth derives the api key from the access
//! token"). pi drives the flow with vitest fake timers; here the shared
//! poller's virtual time is driven by recording sleepers, so poll cadence is
//! asserted through the recorded delays instead of `Date.now()` samples.

use async_trait::async_trait;
use ri_llm_provider::auth::{AuthEvent, AuthInteraction, AuthPrompt, OAuthCredential};
use ri_llm_provider::*;
use serde_json::{Map, Value, json};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

const CLIENT_ID: &str = "b1a00492-073a-47ea-816f-4c329264a828";
/// Arbitrary fixed login start (pi pins `2026-07-09T20:00:00Z`).
const START_MS: i64 = 1_752_091_200_000;

// --- helpers (copied from tests/provider_core.rs, trimmed) ------------------

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
    let (url, task) = mock_json_status_sequence_server(vec![(200, "OK", body.into())]).await;
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

// --- xai-oauth.test.ts fixtures ----------------------------------------------

/// pi `deviceCodeResponse(overrides)`: `Value::Null` removes a field, the way
/// spreading `undefined` drops it from the JSON body.
fn device_code_response(overrides: &[(&str, Value)]) -> String {
    json_fixture(
        json!({
            "device_code": "device-code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://accounts.x.ai/oauth2/device",
            "expires_in": 900,
            "interval": 5,
        }),
        overrides,
    )
}

/// pi `tokenResponse(overrides)`.
fn token_response(overrides: &[(&str, Value)]) -> String {
    json_fixture(
        json!({
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_in": 21_600,
            "token_type": "Bearer",
        }),
        overrides,
    )
}

fn json_fixture(base: Value, overrides: &[(&str, Value)]) -> String {
    let mut object = match base {
        Value::Object(object) => object,
        _ => Map::new(),
    };
    for (key, value) in overrides {
        if value.is_null() {
            object.remove(*key);
        } else {
            object.insert((*key).to_owned(), value.clone());
        }
    }
    Value::Object(object).to_string()
}

fn recording_sleeper(sleeps: Arc<Mutex<Vec<u64>>>) -> impl FnMut(u64) -> std::future::Ready<()> {
    move |delay_ms| {
        sleeps.lock().expect("sleeps").push(delay_ms);
        std::future::ready(())
    }
}

/// pi `refreshXaiForTest`: refresh through the `OAuthAuth` adapter.
async fn refresh_xai_for_test(
    token_url: &str,
    refresh_token: &str,
) -> Result<OAuthCredential, String> {
    use ri_llm_provider::auth::OAuthAuth as _;
    let oauth = XaiOAuth::with_urls(XaiOAuthUrls {
        device_code_url: "http://127.0.0.1:9/device".to_owned(),
        token_url: token_url.to_owned(),
    });
    oauth
        .refresh(&OAuthCredential {
            refresh: refresh_token.to_owned(),
            access: "old-access".to_owned(),
            expires: 0,
            extra: Map::new(),
        })
        .await
}

struct RecordingAuthInteraction {
    events: Mutex<Vec<AuthEvent>>,
}

#[async_trait]
impl AuthInteraction for RecordingAuthInteraction {
    async fn prompt(&self, _prompt: AuthPrompt) -> Result<String, String> {
        Err("Unexpected prompt".to_owned())
    }

    fn notify(&self, event: AuthEvent) {
        self.events.lock().expect("events").push(event);
    }
}

// --- xai-oauth.test.ts --------------------------------------------------------

#[tokio::test]
async fn xai_uses_the_device_grant_delays_polling_and_handles_pending_and_slow_down() {
    let (device_url, device_task) = mock_json_server(device_code_response(&[])).await;
    let (token_url, token_task) = mock_json_status_sequence_server(vec![
        (
            400,
            "Bad Request",
            json!({"error": "authorization_pending"}).to_string(),
        ),
        (
            400,
            "Bad Request",
            json!({"error": "slow_down", "interval": 10}).to_string(),
        ),
        (200, "OK", token_response(&[])),
    ])
    .await;
    let urls = XaiOAuthUrls {
        device_code_url: device_url,
        token_url,
    };
    let device_codes = Arc::new(Mutex::new(Vec::<XaiDeviceCode>::new()));
    let device_codes_ref = device_codes.clone();
    let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));

    let credential = login_xai_device_code_with_urls_with_sleeper(
        &urls,
        move |device| {
            device_codes_ref
                .lock()
                .expect("device codes")
                .push(device.clone());
        },
        START_MS,
        recording_sleeper(sleeps.clone()),
    )
    .await
    .expect("xai login");

    let device_request = device_task.await.expect("device task");
    assert!(device_request.contains(&format!("client_id={CLIENT_ID}")));
    assert!(
        device_request
            .contains("scope=openid+profile+email+offline_access+grok-cli%3Aaccess+api%3Aaccess")
    );
    assert!(device_request.contains("referrer=pi"));

    assert_eq!(
        device_codes.lock().expect("device codes").as_slice(),
        &[XaiDeviceCode {
            device_code: "device-code".to_owned(),
            user_code: "ABCD-1234".to_owned(),
            verification_uri: "https://accounts.x.ai/oauth2/device".to_owned(),
            verification_uri_complete: None,
            interval_seconds: Some(5),
            expires_in_seconds: 900,
        }]
    );

    // pi asserts pollTimes [start+5000, start+10_000, start+20_000]: one
    // interval before the first poll, one after authorization_pending, and
    // the raised 10s interval after slow_down.
    assert_eq!(
        sleeps.lock().expect("sleeps").as_slice(),
        &[5000, 5000, 10_000]
    );

    let token_requests = token_task.await.expect("token task");
    assert_eq!(token_requests.len(), 3);
    for token_request in &token_requests {
        assert!(
            token_request
                .contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Adevice_code")
        );
        assert!(token_request.contains(&format!("client_id={CLIENT_ID}")));
        assert!(token_request.contains("device_code=device-code"));
    }

    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
    assert_eq!(credential.expires, START_MS + 20_000 + 21_600_000 - 300_000);
    assert!(credential.extra.is_empty());
}

#[tokio::test]
async fn xai_falls_back_to_the_default_poll_interval_when_the_response_reports_interval_0() {
    let (device_url, _device_task) =
        mock_json_server(device_code_response(&[("interval", json!(0))])).await;
    let (token_url, token_task) =
        mock_json_status_sequence_server(vec![(200, "OK", token_response(&[]))]).await;
    let urls = XaiOAuthUrls {
        device_code_url: device_url,
        token_url,
    };
    let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));

    login_xai_device_code_with_urls_with_sleeper(
        &urls,
        |_device| {},
        START_MS,
        recording_sleeper(sleeps.clone()),
    )
    .await
    .expect("xai login");

    // RFC 8628 default interval is 5 seconds when the server does not require
    // a wait.
    assert_eq!(sleeps.lock().expect("sleeps").as_slice(), &[5000]);
    assert_eq!(token_task.await.expect("token task").len(), 1);
}

#[tokio::test]
async fn xai_prefers_verification_uri_complete_when_the_server_provides_it() {
    // Runs the full `OAuthAuth::login` adapter (real sleeper), so the device
    // response reports interval 1 instead of pi's fake-timer interval 5.
    let (device_url, _device_task) = mock_json_server(device_code_response(&[
        (
            "verification_uri_complete",
            json!("https://accounts.x.ai/oauth2/device?user_code=ABCD-1234"),
        ),
        ("interval", json!(1)),
    ]))
    .await;
    let (token_url, _token_task) =
        mock_json_status_sequence_server(vec![(200, "OK", token_response(&[]))]).await;
    let oauth = XaiOAuth::with_urls(XaiOAuthUrls {
        device_code_url: device_url,
        token_url,
    });
    let interaction = RecordingAuthInteraction {
        events: Mutex::new(Vec::new()),
    };

    use ri_llm_provider::auth::OAuthAuth as _;
    let credential = oauth.login(&interaction).await.expect("xai login");

    assert_eq!(
        interaction.events.lock().expect("events").as_slice(),
        &[AuthEvent::DeviceCode {
            user_code: "ABCD-1234".to_owned(),
            verification_uri: "https://accounts.x.ai/oauth2/device?user_code=ABCD-1234".to_owned(),
            interval_seconds: Some(1),
            expires_in_seconds: Some(900),
        }]
    );
    assert_eq!(credential.access, "access-token");
    assert_eq!(credential.refresh, "refresh-token");
}

#[tokio::test]
async fn xai_rejects_a_non_https_verification_uri_complete() {
    let (device_url, _device_task) = mock_json_server(device_code_response(&[(
        "verification_uri_complete",
        json!("http://accounts.x.ai/oauth2/device?user_code=ABCD-1234"),
    )]))
    .await;
    let urls = XaiOAuthUrls {
        device_code_url: device_url,
        token_url: "http://127.0.0.1:9/token".to_owned(),
    };
    let on_device_code_called = Arc::new(AtomicBool::new(false));
    let on_device_code_called_ref = on_device_code_called.clone();

    let err = login_xai_device_code_with_urls_with_sleeper(
        &urls,
        move |_device| on_device_code_called_ref.store(true, Ordering::SeqCst),
        START_MS,
        |_delay_ms| -> std::future::Ready<()> {
            panic!("must not poll after an untrusted verification URI")
        },
    )
    .await
    .expect_err("untrusted verification_uri_complete must fail the login");

    assert_eq!(err, "Untrusted verification URI in xAI OAuth response");
    assert!(!on_device_code_called.load(Ordering::SeqCst));
}

#[tokio::test]
async fn xai_rejects_a_non_https_verification_uri() {
    // pi runs this as it.each over three untrusted URIs.
    for verification_uri in [
        "http://accounts.x.ai/oauth2/device",
        "file:///etc/passwd",
        "not a url",
    ] {
        let (device_url, _device_task) = mock_json_server(device_code_response(&[(
            "verification_uri",
            json!(verification_uri),
        )]))
        .await;
        let urls = XaiOAuthUrls {
            device_code_url: device_url,
            token_url: "http://127.0.0.1:9/token".to_owned(),
        };

        let err = login_xai_device_code_with_urls_with_sleeper(
            &urls,
            |_device| {},
            START_MS,
            |_delay_ms| -> std::future::Ready<()> {
                panic!("must not poll after an untrusted verification URI")
            },
        )
        .await
        .expect_err("untrusted verification_uri must fail the login");

        assert_eq!(
            err, "Untrusted verification URI in xAI OAuth response",
            "verification_uri: {verification_uri}"
        );
    }
}

#[tokio::test]
async fn xai_fails_when_device_authorization_is_denied() {
    // pi runs this as it.each over both denial error codes.
    for denial_error in ["access_denied", "authorization_denied"] {
        let (device_url, _device_task) =
            mock_json_server(device_code_response(&[("interval", json!(1))])).await;
        let (token_url, token_task) = mock_json_status_sequence_server(vec![(
            400,
            "Bad Request",
            json!({ "error": denial_error }).to_string(),
        )])
        .await;
        let urls = XaiOAuthUrls {
            device_code_url: device_url,
            token_url,
        };
        let sleeps = Arc::new(Mutex::new(Vec::<u64>::new()));

        let err = login_xai_device_code_with_urls_with_sleeper(
            &urls,
            |_device| {},
            START_MS,
            recording_sleeper(sleeps.clone()),
        )
        .await
        .expect_err("denied device authorization must fail the login");

        assert_eq!(
            err, "xAI device authorization was denied",
            "error: {denial_error}"
        );
        assert_eq!(sleeps.lock().expect("sleeps").as_slice(), &[1000]);
        assert_eq!(token_task.await.expect("token task").len(), 1);
    }
}

#[tokio::test]
async fn xai_cancels_while_waiting_for_the_first_token_poll() {
    let (device_url, device_task) = mock_json_server(device_code_response(&[])).await;
    let urls = XaiOAuthUrls {
        device_code_url: device_url,
        // A poll would fail with a transport error, not "Login cancelled".
        token_url: "http://127.0.0.1:9/token".to_owned(),
    };
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_flag_ref = abort_flag.clone();

    // pi aborts from inside the device_code notification, before the first
    // poll delay elapses.
    let err = login_xai_device_code_with_urls_with_sleeper_and_abort(
        &urls,
        move |_device| abort_flag_ref.store(true, Ordering::SeqCst),
        START_MS,
        |_delay_ms| -> std::future::Ready<()> {
            panic!("initial wait must not run after cancellation")
        },
        Some(abort_flag),
    )
    .await
    .expect_err("cancelled before the first poll");

    assert_eq!(err, "Login cancelled");
    // Only the device authorization request was made (pi: fetch called once).
    device_task.await.expect("device task");
}

#[tokio::test]
async fn xai_refreshes_tokens_and_preserves_an_unrotated_refresh_token() {
    use ri_llm_provider::auth::{ModelAuth, OAuthAuth as _};
    let (token_url, token_task) = mock_json_status_sequence_server(vec![
        (
            200,
            "OK",
            token_response(&[
                ("access_token", json!("new-access")),
                ("refresh_token", json!("new-refresh")),
            ]),
        ),
        (
            200,
            "OK",
            token_response(&[
                ("access_token", json!("newer-access")),
                ("refresh_token", Value::Null),
            ]),
        ),
    ])
    .await;

    let rotated = refresh_xai_for_test(&token_url, "old-refresh")
        .await
        .expect("rotated refresh");
    let preserved = refresh_xai_for_test(&token_url, "keep-refresh")
        .await
        .expect("preserved refresh");

    let token_requests = token_task.await.expect("token task");
    assert_eq!(token_requests.len(), 2);
    for token_request in &token_requests {
        assert!(token_request.contains("grant_type=refresh_token"));
        assert!(token_request.contains(&format!("client_id={CLIENT_ID}")));
    }
    assert!(token_requests[0].contains("refresh_token=old-refresh"));
    assert!(token_requests[1].contains("refresh_token=keep-refresh"));

    assert_eq!(rotated.refresh, "new-refresh");
    assert_eq!(rotated.access, "new-access");
    assert_eq!(preserved.refresh, "keep-refresh");
    assert_eq!(preserved.access, "newer-access");

    let oauth = XaiOAuth::new();
    assert_eq!(oauth.name(), "xAI (Grok/X subscription)");
    assert_eq!(
        oauth.to_auth(&preserved).await,
        Ok(ModelAuth {
            api_key: Some("newer-access".to_owned()),
            ..Default::default()
        })
    );
}

#[tokio::test]
async fn xai_assumes_a_one_hour_lifetime_when_expires_in_is_missing() {
    let (token_url, _token_task) = mock_json_status_sequence_server(vec![(
        200,
        "OK",
        token_response(&[("expires_in", Value::Null)]),
    )])
    .await;

    let credential = refresh_xai_token_with_url_at("old-refresh", &token_url, START_MS)
        .await
        .expect("refresh");

    assert_eq!(credential.expires, START_MS + 3_600_000 - 300_000);
}

#[tokio::test]
async fn xai_rejects_token_responses_with_missing_fields() {
    let (token_url, _token_task) = mock_json_status_sequence_server(vec![(
        200,
        "OK",
        token_response(&[("access_token", Value::Null)]),
    )])
    .await;

    let err = refresh_xai_for_test(&token_url, "old-refresh")
        .await
        .expect_err("missing access_token must fail the refresh");

    assert_eq!(err, "Invalid xAI OAuth response field: access_token");
}

#[tokio::test]
async fn xai_surfaces_the_upstream_error_code_and_description_on_refresh_failure() {
    let (token_url, _token_task) = mock_json_status_sequence_server(vec![(
        400,
        "Bad Request",
        json!({
            "error": "invalid_grant",
            "error_description": "refresh token revoked",
        })
        .to_string(),
    )])
    .await;

    let err = refresh_xai_for_test(&token_url, "old-refresh")
        .await
        .expect_err("upstream error must fail the refresh");

    assert_eq!(
        err,
        "xAI OAuth token refresh failed (HTTP 400): invalid_grant: refresh token revoked"
    );
}

// --- oauth-auth.test.ts: xAI row ----------------------------------------------

#[tokio::test]
async fn xai_to_auth_derives_the_api_key_from_the_access_token() {
    use ri_llm_provider::auth::{ModelAuth, OAuthAuth as _};
    let auth = XaiOAuth::new()
        .to_auth(&OAuthCredential {
            refresh: "r".to_owned(),
            access: "token".to_owned(),
            expires: 0,
            extra: Map::new(),
        })
        .await
        .expect("to_auth");
    assert_eq!(
        auth,
        ModelAuth {
            api_key: Some("token".to_owned()),
            ..Default::default()
        }
    );
}

// --- providers/all wiring -------------------------------------------------------

#[test]
fn xai_builtin_provider_carries_the_xai_oauth() {
    let provider = builtin_provider("xai").expect("xai provider");
    let oauth = provider.auth().oauth.as_ref().expect("xai oauth arm");
    assert_eq!(oauth.name(), "xAI (Grok/X subscription)");
    assert_eq!(
        oauth.login_label(),
        Some("Sign in with SuperGrok or X Premium")
    );
}
