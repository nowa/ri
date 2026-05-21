use crate::{Model, node_http_proxy::reqwest_client_for_target};
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD as BASE64_STANDARD, URL_SAFE_NO_PAD},
};
use ring::{rand, signature};
use serde_json::{Value, json};
use std::{collections::BTreeMap, env, fs, path::PathBuf};

pub const GOOGLE_VERTEX_API_VERSION: &str = "v1";
pub const GCP_VERTEX_CREDENTIALS_MARKER: &str = "gcp-vertex-credentials";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GoogleVertexOptions {
    pub api_key: Option<String>,
    pub project: Option<String>,
    pub location: Option<String>,
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleVertexHttpOptions {
    pub base_url: String,
    pub base_url_resource_scope: String,
    pub api_version: Option<String>,
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoogleVertexClientConfig {
    pub vertexai: bool,
    pub api_key: Option<String>,
    pub project: Option<String>,
    pub location: Option<String>,
    pub api_version: String,
    pub http_options: Option<GoogleVertexHttpOptions>,
}

pub fn resolve_google_vertex_client_config(
    model: &Model,
    options: GoogleVertexOptions,
) -> Result<GoogleVertexClientConfig, String> {
    let api_key = resolve_google_vertex_api_key(options.api_key.as_deref());
    let http_options = build_google_vertex_http_options(model, &options.headers);

    if let Some(api_key) = api_key {
        return Ok(GoogleVertexClientConfig {
            vertexai: true,
            api_key: Some(api_key),
            project: None,
            location: None,
            api_version: GOOGLE_VERTEX_API_VERSION.to_owned(),
            http_options,
        });
    }

    Ok(GoogleVertexClientConfig {
        vertexai: true,
        api_key: None,
        project: Some(resolve_google_vertex_project(options.project.as_deref())?),
        location: Some(resolve_google_vertex_location(options.location.as_deref())?),
        api_version: GOOGLE_VERTEX_API_VERSION.to_owned(),
        http_options,
    })
}

pub fn resolve_google_vertex_api_key(option_api_key: Option<&str>) -> Option<String> {
    let api_key = option_api_key
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| {
            std::env::var("GOOGLE_CLOUD_API_KEY")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })?;

    if api_key == GCP_VERTEX_CREDENTIALS_MARKER || is_placeholder_api_key(&api_key) {
        None
    } else {
        Some(api_key)
    }
}

pub fn resolve_custom_google_vertex_base_url(base_url: &str) -> Option<String> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() || trimmed.contains("{location}") {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

pub async fn resolve_google_vertex_adc_access_token() -> Result<Option<String>, String> {
    if let Some(token) = env::var("GOOGLE_OAUTH_ACCESS_TOKEN")
        .ok()
        .or_else(|| env::var("GOOGLE_ACCESS_TOKEN").ok())
        .or_else(|| env::var("CLOUDSDK_AUTH_ACCESS_TOKEN").ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
    {
        return Ok(Some(token));
    }

    let Some(path) = google_adc_credentials_path() else {
        return Ok(None);
    };
    let text = fs::read_to_string(&path).map_err(|error| {
        format!(
            "Failed to read Google ADC credentials at {:?}: {error}",
            path
        )
    })?;
    let credentials: Value = serde_json::from_str(&text).map_err(|error| {
        format!(
            "Failed to parse Google ADC credentials at {:?}: {error}",
            path
        )
    })?;

    if let Some(token) = credentials
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        return Ok(Some(token.to_owned()));
    }

    match credentials.get("type").and_then(Value::as_str) {
        Some("authorized_user") => refresh_google_authorized_user_token(&credentials)
            .await
            .map(Some),
        Some("service_account") => refresh_google_service_account_token(&credentials)
            .await
            .map(Some),
        Some(kind) => Err(format!("Unsupported Google ADC credentials type: {kind}")),
        None => Err("Google ADC credentials are missing a type field".to_owned()),
    }
}

fn google_adc_credentials_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("GOOGLE_APPLICATION_CREDENTIALS")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        return Some(path);
    }

    if let Some(config_dir) = env::var_os("APPDATA").map(PathBuf::from) {
        let path = config_dir
            .join("gcloud")
            .join("application_default_credentials.json");
        if path.exists() {
            return Some(path);
        }
    }

    let path = env::var_os("HOME").map(PathBuf::from).map(|home| {
        home.join(".config")
            .join("gcloud")
            .join("application_default_credentials.json")
    })?;
    path.exists().then_some(path)
}

async fn refresh_google_authorized_user_token(credentials: &Value) -> Result<String, String> {
    let refresh_token = required_google_adc_string(credentials, "refresh_token")?;
    let client_id = required_google_adc_string(credentials, "client_id")?;
    let client_secret = required_google_adc_string(credentials, "client_secret")?;
    let token_uri = google_adc_token_uri(credentials);
    let body = google_form_urlencoded(&[
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", client_id),
        ("client_secret", client_secret),
    ]);
    request_google_adc_token(&token_uri, body).await
}

async fn refresh_google_service_account_token(credentials: &Value) -> Result<String, String> {
    let client_email = required_google_adc_string(credentials, "client_email")?;
    let private_key = required_google_adc_string(credentials, "private_key")?;
    let token_uri = google_adc_token_uri(credentials);
    let now = chrono::Utc::now().timestamp();
    let header = json!({ "alg": "RS256", "typ": "JWT" });
    let claims = json!({
        "iss": client_email,
        "scope": "https://www.googleapis.com/auth/cloud-platform",
        "aud": token_uri,
        "iat": now,
        "exp": now + 3600,
    });
    let signing_input = format!(
        "{}.{}",
        base64_url_json(&header)?,
        base64_url_json(&claims)?
    );
    let signature = sign_google_service_account_jwt(private_key, signing_input.as_bytes())?;
    let assertion = format!(
        "{}.{}",
        signing_input,
        URL_SAFE_NO_PAD.encode(signature.as_slice())
    );
    let body = google_form_urlencoded(&[
        ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
        ("assertion", &assertion),
    ]);
    request_google_adc_token(&token_uri, body).await
}

async fn request_google_adc_token(token_uri: &str, body: String) -> Result<String, String> {
    let response = reqwest_client_for_target(token_uri)?
        .post(token_uri)
        .header("content-type", "application/x-www-form-urlencoded")
        .body(body)
        .send()
        .await
        .map_err(|error| error.to_string())?;
    let status = response.status();
    let text = response.text().await.map_err(|error| error.to_string())?;
    if !status.is_success() {
        return Err(format!(
            "Google ADC token request failed with status {}: {}",
            status.as_u16(),
            text
        ));
    }
    let value: Value = serde_json::from_str(&text)
        .map_err(|error| format!("Failed to parse Google ADC token response: {error}"))?;
    value
        .get("access_token")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| "Google ADC token response did not include access_token".to_owned())
}

fn required_google_adc_string<'a>(credentials: &'a Value, key: &str) -> Result<&'a str, String> {
    credentials
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("Google ADC credentials are missing {key}"))
}

fn google_adc_token_uri(credentials: &Value) -> String {
    credentials
        .get("token_uri")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("https://oauth2.googleapis.com/token")
        .to_owned()
}

fn base64_url_json(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| URL_SAFE_NO_PAD.encode(bytes))
        .map_err(|error| error.to_string())
}

fn sign_google_service_account_jwt(private_key_pem: &str, bytes: &[u8]) -> Result<Vec<u8>, String> {
    let der = decode_google_private_key_pem(private_key_pem)?;
    let key_pair = signature::RsaKeyPair::from_pkcs8(&der)
        .map_err(|_| "Failed to parse Google service account private key".to_owned())?;
    let rng = rand::SystemRandom::new();
    let mut signature = vec![0_u8; key_pair.public().modulus_len()];
    key_pair
        .sign(&signature::RSA_PKCS1_SHA256, &rng, bytes, &mut signature)
        .map_err(|_| "Failed to sign Google service account JWT".to_owned())?;
    Ok(signature)
}

fn decode_google_private_key_pem(private_key_pem: &str) -> Result<Vec<u8>, String> {
    let mut encoded = String::new();
    for line in private_key_pem.lines() {
        let line = line.trim();
        if line.starts_with("-----") || line.is_empty() {
            continue;
        }
        encoded.push_str(line);
    }
    BASE64_STANDARD
        .decode(encoded.as_bytes())
        .map_err(|error| format!("Failed to decode Google service account private key: {error}"))
}

fn google_form_urlencoded(fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .map(|(key, value)| format!("{}={}", google_form_encode(key), google_form_encode(value)))
        .collect::<Vec<_>>()
        .join("&")
}

fn google_form_encode(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        let allowed = byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~');
        if allowed {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

pub fn google_vertex_base_url_includes_api_version(base_url: &str) -> bool {
    let path = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url)
        .split_once('/')
        .map(|(_, path)| path)
        .unwrap_or_default();
    path.split('/')
        .any(|part| is_google_vertex_api_version_segment(part))
}

fn build_google_vertex_http_options(
    model: &Model,
    option_headers: &BTreeMap<String, String>,
) -> Option<GoogleVertexHttpOptions> {
    let base_url = resolve_custom_google_vertex_base_url(&model.base_url);
    let mut headers = model.headers.clone();
    headers.extend(option_headers.clone());

    if base_url.is_none() && headers.is_empty() {
        return None;
    }

    let base_url = base_url.unwrap_or_default();
    let api_version =
        if !base_url.is_empty() && google_vertex_base_url_includes_api_version(&base_url) {
            Some(String::new())
        } else {
            None
        };

    Some(GoogleVertexHttpOptions {
        base_url,
        base_url_resource_scope: "COLLECTION".to_owned(),
        api_version,
        headers,
    })
}

fn resolve_google_vertex_project(project: Option<&str>) -> Result<String, String> {
    project
        .filter(|project| !project.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_var_nonempty("GOOGLE_CLOUD_PROJECT"))
        .or_else(|| env_var_nonempty("GCLOUD_PROJECT"))
        .ok_or_else(|| {
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options.".to_owned()
        })
}

fn resolve_google_vertex_location(location: Option<&str>) -> Result<String, String> {
    location
        .filter(|location| !location.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| env_var_nonempty("GOOGLE_CLOUD_LOCATION"))
        .ok_or_else(|| {
            "Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options."
                .to_owned()
        })
}

fn is_placeholder_api_key(api_key: &str) -> bool {
    api_key
        .strip_prefix('<')
        .and_then(|inner| inner.strip_suffix('>'))
        .is_some_and(|inner| !inner.is_empty() && !inner.contains('>'))
}

fn env_var_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

fn is_google_vertex_api_version_segment(part: &str) -> bool {
    let Some(rest) = part.strip_prefix('v') else {
        return false;
    };
    let digits = rest
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return false;
    }
    let suffix = &rest[digits.len()..];
    suffix.is_empty()
        || (suffix.starts_with("beta")
            && suffix["beta".len()..].chars().all(|ch| ch.is_ascii_digit()))
}
