use crate::{
    AssistantContent, AssistantMessage, AssistantMessageEvent, AssistantMessageEventSender,
    CacheRetention, Context, ImageContent, Message, Model, StopReason, TextContent,
    ThinkingBudgets, ThinkingContent, ThinkingLevel, Tool, ToolCall, ToolResultContent, Usage,
    UserContent, UserContentValue, json_repair::parse_streaming_json,
    message_transform::transform_messages, models::calculate_cost,
    node_http_proxy::reqwest_client_for_target,
};
use chrono::{DateTime, Utc};
use ring::{digest, hmac};
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{collections::BTreeMap, env, fs, path::PathBuf, time::Duration};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BedrockClientOptions {
    pub region: Option<String>,
    pub profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BedrockClientConfig {
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BedrockToolChoice {
    Auto,
    Any,
    None,
    Tool { name: String },
}

pub fn resolve_bedrock_client_config(
    model: &Model,
    options: BedrockClientOptions,
) -> BedrockClientConfig {
    let configured_region = configured_bedrock_region(&options);
    let has_configured_profile = std::env::var_os("AWS_PROFILE").is_some();
    let endpoint_region = standard_bedrock_endpoint_region(&model.base_url);
    let use_explicit_endpoint = should_use_explicit_bedrock_endpoint(
        &model.base_url,
        configured_region.as_deref(),
        has_configured_profile,
    );

    let endpoint = use_explicit_endpoint.then(|| model.base_url.clone());
    let region = configured_region.or_else(|| {
        if use_explicit_endpoint {
            endpoint_region
        } else if !has_configured_profile {
            Some("us-east-1".to_owned())
        } else {
            None
        }
    });

    BedrockClientConfig {
        region,
        endpoint,
        profile: options.profile,
    }
}

pub fn bedrock_base_url_for_model(model_id: &str) -> &'static str {
    if model_id.starts_with("eu.") {
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
    } else {
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    }
}

pub fn standard_bedrock_endpoint_region(base_url: &str) -> Option<String> {
    let host = base_url
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(base_url)
        .split('/')
        .next()
        .unwrap_or_default()
        .to_ascii_lowercase();

    let host = host
        .strip_suffix(".amazonaws.com.cn")
        .or_else(|| host.strip_suffix(".amazonaws.com"))?;
    host.strip_prefix("bedrock-runtime.")
        .or_else(|| host.strip_prefix("bedrock-runtime-fips."))
        .filter(|region| !region.is_empty())
        .map(ToOwned::to_owned)
}

pub fn should_use_explicit_bedrock_endpoint(
    base_url: &str,
    configured_region: Option<&str>,
    has_configured_profile: bool,
) -> bool {
    if standard_bedrock_endpoint_region(base_url).is_none() {
        return true;
    }
    configured_region.is_none() && !has_configured_profile
}

fn configured_bedrock_region(options: &BedrockClientOptions) -> Option<String> {
    options
        .region
        .clone()
        .or_else(|| std::env::var("AWS_REGION").ok())
        .or_else(|| std::env::var("AWS_DEFAULT_REGION").ok())
        .filter(|region| !region.is_empty())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

pub fn resolve_aws_credentials(profile: Option<&str>) -> Option<AwsCredentials> {
    if env::var("AWS_BEDROCK_SKIP_AUTH").ok().as_deref() == Some("1") {
        return Some(AwsCredentials {
            access_key_id: "dummy-access-key".to_owned(),
            secret_access_key: "dummy-secret-key".to_owned(),
            session_token: None,
        });
    }

    if let (Ok(access_key_id), Ok(secret_access_key)) = (
        env::var("AWS_ACCESS_KEY_ID"),
        env::var("AWS_SECRET_ACCESS_KEY"),
    ) && !access_key_id.is_empty()
        && !secret_access_key.is_empty()
    {
        return Some(AwsCredentials {
            access_key_id,
            secret_access_key,
            session_token: env::var("AWS_SESSION_TOKEN")
                .ok()
                .or_else(|| env::var("AWS_SECURITY_TOKEN").ok())
                .filter(|token| !token.is_empty()),
        });
    }

    let profile_name = aws_profile_name(profile);
    let properties = load_aws_profile_properties(&profile_name)?;
    let access_key_id = properties.get("aws_access_key_id")?.to_owned();
    let secret_access_key = properties.get("aws_secret_access_key")?.to_owned();
    if access_key_id.is_empty() || secret_access_key.is_empty() {
        return None;
    }

    Some(AwsCredentials {
        access_key_id,
        secret_access_key,
        session_token: properties
            .get("aws_session_token")
            .or_else(|| properties.get("aws_security_token"))
            .filter(|token| !token.is_empty())
            .cloned(),
    })
}

pub async fn resolve_aws_credentials_with_container(
    profile: Option<&str>,
) -> Result<Option<AwsCredentials>, String> {
    resolve_aws_credentials_with_runtime(profile).await
}

pub async fn resolve_aws_credentials_with_runtime(
    profile: Option<&str>,
) -> Result<Option<AwsCredentials>, String> {
    if let Some(credentials) = resolve_aws_credentials(profile) {
        return Ok(Some(credentials));
    }
    if let Some(credentials) = resolve_aws_web_identity_credentials().await? {
        return Ok(Some(credentials));
    }
    resolve_aws_container_credentials().await
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
struct AwsContainerCredentialsResponse {
    access_key_id: String,
    secret_access_key: String,
    token: Option<String>,
}

async fn resolve_aws_web_identity_credentials() -> Result<Option<AwsCredentials>, String> {
    let Some(token_path) = env_var_nonempty("AWS_WEB_IDENTITY_TOKEN_FILE") else {
        return Ok(None);
    };
    let Some(role_arn) = env_var_nonempty("AWS_ROLE_ARN") else {
        return Ok(None);
    };
    let token = fs::read_to_string(&token_path)
        .map_err(|error| format!("Failed to read AWS web identity token: {error}"))?;
    let token = token.trim();
    if token.is_empty() {
        return Ok(None);
    }
    let session_name = env_var_nonempty("AWS_ROLE_SESSION_NAME")
        .unwrap_or_else(|| "ri-bedrock-web-identity".to_owned());
    let url = aws_sts_endpoint_url();
    let body = form_url_encode(&[
        ("Action", "AssumeRoleWithWebIdentity"),
        ("Version", "2011-06-15"),
        ("RoleArn", &role_arn),
        ("RoleSessionName", &session_name),
        ("WebIdentityToken", token),
    ]);
    let client = reqwest_client_for_target(&url)?;
    let response = client
        .post(&url)
        .header("content-type", "application/x-www-form-urlencoded")
        .timeout(Duration::from_secs(5))
        .body(body)
        .send()
        .await
        .map_err(|error| format!("Failed to assume AWS web identity role: {error}"))?;
    let status = response.status();
    let response_body = response
        .text()
        .await
        .map_err(|error| format!("Failed to read AWS web identity response: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "AWS web identity role assumption returned HTTP {}: {response_body}",
            status.as_u16()
        ));
    }
    parse_aws_web_identity_credentials(&response_body)
        .map(Some)
        .map_err(|error| format!("Invalid AWS web identity response: {error}"))
}

fn aws_sts_endpoint_url() -> String {
    env_var_nonempty("AWS_ENDPOINT_URL_STS")
        .or_else(|| env_var_nonempty("AWS_ENDPOINT_URL"))
        .unwrap_or_else(|| {
            configured_bedrock_region(&BedrockClientOptions::default()).map_or_else(
                || "https://sts.amazonaws.com/".to_owned(),
                |region| {
                    if region.starts_with("cn-") {
                        format!("https://sts.{region}.amazonaws.com.cn/")
                    } else {
                        format!("https://sts.{region}.amazonaws.com/")
                    }
                },
            )
        })
}

fn parse_aws_web_identity_credentials(body: &str) -> Result<AwsCredentials, String> {
    let access_key_id = xml_tag_text(body, "AccessKeyId")
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "missing AccessKeyId".to_owned())?;
    let secret_access_key = xml_tag_text(body, "SecretAccessKey")
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "missing SecretAccessKey".to_owned())?;
    let session_token = xml_tag_text(body, "SessionToken").filter(|value| !value.trim().is_empty());
    Ok(AwsCredentials {
        access_key_id,
        secret_access_key,
        session_token,
    })
}

fn xml_tag_text(body: &str, tag: &str) -> Option<String> {
    let start = format!("<{tag}>");
    let end = format!("</{tag}>");
    let (_, rest) = body.split_once(&start)?;
    let (value, _) = rest.split_once(&end)?;
    Some(xml_unescape(value.trim()))
}

fn xml_unescape(value: &str) -> String {
    value
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

fn form_url_encode(pairs: &[(&str, &str)]) -> String {
    pairs
        .iter()
        .map(|(name, value)| {
            format!(
                "{}={}",
                form_url_encode_component(name),
                form_url_encode_component(value)
            )
        })
        .collect::<Vec<_>>()
        .join("&")
}

fn form_url_encode_component(value: &str) -> String {
    let mut encoded = String::new();
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            b' ' => encoded.push('+'),
            _ => encoded.push_str(&format!("%{byte:02X}")),
        }
    }
    encoded
}

fn env_var_nonempty(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.trim().is_empty())
}

async fn resolve_aws_container_credentials() -> Result<Option<AwsCredentials>, String> {
    let Some(url) = aws_container_credentials_url() else {
        return Ok(None);
    };
    let client = reqwest_client_for_target(&url)?;
    let mut request = client.get(&url).timeout(Duration::from_secs(2));
    if let Some(token) = aws_container_authorization_token()? {
        request = request.header("authorization", token);
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("Failed to fetch AWS container credentials: {error}"))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| format!("Failed to read AWS container credentials: {error}"))?;
    if !status.is_success() {
        return Err(format!(
            "AWS container credentials endpoint returned HTTP {}: {body}",
            status.as_u16()
        ));
    }
    let credentials: AwsContainerCredentialsResponse = serde_json::from_str(&body)
        .map_err(|error| format!("Invalid AWS container credentials response: {error}"))?;
    if credentials.access_key_id.trim().is_empty()
        || credentials.secret_access_key.trim().is_empty()
    {
        return Ok(None);
    }
    Ok(Some(AwsCredentials {
        access_key_id: credentials.access_key_id,
        secret_access_key: credentials.secret_access_key,
        session_token: credentials.token.filter(|token| !token.trim().is_empty()),
    }))
}

fn aws_container_credentials_url() -> Option<String> {
    env::var("AWS_CONTAINER_CREDENTIALS_FULL_URI")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|path| {
                    if path.starts_with('/') {
                        format!("http://169.254.170.2{path}")
                    } else {
                        format!("http://169.254.170.2/{path}")
                    }
                })
        })
}

fn aws_container_authorization_token() -> Result<Option<String>, String> {
    if let Ok(path) = env::var("AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE")
        && !path.trim().is_empty()
    {
        return fs::read_to_string(&path)
            .map(|token| Some(token.trim().to_owned()))
            .map_err(|error| format!("Failed to read AWS container authorization token: {error}"));
    }
    Ok(env::var("AWS_CONTAINER_AUTHORIZATION_TOKEN")
        .ok()
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty()))
}

pub fn resolve_aws_profile_region(profile: Option<&str>) -> Option<String> {
    let profile_name = aws_profile_name(profile);
    load_aws_profile_properties(&profile_name)?
        .get("region")
        .filter(|region| !region.is_empty())
        .cloned()
}

pub fn sign_aws_sigv4_headers(
    method: &str,
    url: &str,
    service: &str,
    region: &str,
    headers: &mut BTreeMap<String, String>,
    body: &[u8],
    credentials: &AwsCredentials,
    now: DateTime<Utc>,
) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url).map_err(|error| error.to_string())?;
    let host = aws_sigv4_host(&parsed)?;
    let payload_hash = sha256_hex(body);
    let amz_date = now.format("%Y%m%dT%H%M%SZ").to_string();
    let short_date = now.format("%Y%m%d").to_string();

    remove_header_case_insensitive(headers, "authorization");
    set_header_case_insensitive(headers, "host", host);
    set_header_case_insensitive(headers, "x-amz-date", amz_date.clone());
    set_header_case_insensitive(headers, "x-amz-content-sha256", payload_hash.clone());
    if let Some(session_token) = &credentials.session_token {
        set_header_case_insensitive(headers, "x-amz-security-token", session_token.clone());
    }

    let (canonical_headers, signed_headers) = canonical_aws_headers(headers);
    let canonical_request = format!(
        "{}\n{}\n{}\n{}\n{}\n{}",
        method.to_ascii_uppercase(),
        canonical_aws_path(parsed.path()),
        canonical_aws_query(parsed.query().unwrap_or_default()),
        canonical_headers,
        signed_headers,
        payload_hash
    );
    let credential_scope = format!("{short_date}/{region}/{service}/aws4_request");
    let string_to_sign = format!(
        "AWS4-HMAC-SHA256\n{}\n{}\n{}",
        amz_date,
        credential_scope,
        sha256_hex(canonical_request.as_bytes())
    );
    let signing_key =
        aws_sigv4_signing_key(&credentials.secret_access_key, &short_date, region, service);
    let signature = hex_lower(&hmac_sha256(&signing_key, string_to_sign.as_bytes()));
    let authorization = format!(
        "AWS4-HMAC-SHA256 Credential={}/{}, SignedHeaders={}, Signature={}",
        credentials.access_key_id, credential_scope, signed_headers, signature
    );
    set_header_case_insensitive(headers, "authorization", authorization);
    Ok(())
}

fn aws_profile_name(profile: Option<&str>) -> String {
    profile
        .filter(|profile| !profile.is_empty())
        .map(str::to_owned)
        .or_else(|| env::var("AWS_PROFILE").ok())
        .filter(|profile| !profile.is_empty())
        .unwrap_or_else(|| "default".to_owned())
}

fn load_aws_profile_properties(profile: &str) -> Option<BTreeMap<String, String>> {
    let mut properties = BTreeMap::new();
    if let Some(credentials) = read_aws_ini("AWS_SHARED_CREDENTIALS_FILE", ".aws/credentials")
        && let Some(section) = credentials.get(profile)
    {
        properties.extend(section.clone());
    }

    if let Some(config) = read_aws_ini("AWS_CONFIG_FILE", ".aws/config") {
        let config_section = if profile == "default" {
            "default".to_owned()
        } else {
            format!("profile {profile}")
        };
        if let Some(section) = config.get(&config_section) {
            for (key, value) in section {
                properties
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        }
        if let Some(section) = config.get(profile) {
            for (key, value) in section {
                properties
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        }
    }

    (!properties.is_empty()).then_some(properties)
}

fn read_aws_ini(
    env_key: &str,
    default_suffix: &str,
) -> Option<BTreeMap<String, BTreeMap<String, String>>> {
    let path = env::var_os(env_key)
        .map(PathBuf::from)
        .or_else(|| aws_home_dir().map(|home| home.join(default_suffix)))?;
    let text = fs::read_to_string(path).ok()?;
    Some(parse_aws_ini(&text))
}

fn aws_home_dir() -> Option<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
}

fn parse_aws_ini(text: &str) -> BTreeMap<String, BTreeMap<String, String>> {
    let mut sections = BTreeMap::new();
    let mut current_section: Option<String> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(section) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            let section = section.trim().to_owned();
            sections
                .entry(section.clone())
                .or_insert_with(BTreeMap::new);
            current_section = Some(section);
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if let Some(section) = current_section.as_deref() {
            sections
                .entry(section.to_owned())
                .or_insert_with(BTreeMap::new)
                .insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    sections
}

fn aws_sigv4_host(url: &reqwest::Url) -> Result<String, String> {
    let host = url
        .host_str()
        .ok_or_else(|| "AWS SigV4 URL is missing host".to_owned())?;
    let include_port = match (url.scheme(), url.port()) {
        ("http", Some(80)) | ("https", Some(443)) | (_, None) => false,
        (_, Some(_)) => true,
    };
    Ok(if include_port {
        format!("{}:{}", host, url.port().expect("checked port"))
    } else {
        host.to_owned()
    })
}

fn canonical_aws_path(path: &str) -> String {
    if path.is_empty() {
        return "/".to_owned();
    }
    aws_percent_encode(path.as_bytes(), true)
}

fn canonical_aws_query(query: &str) -> String {
    if query.is_empty() {
        return String::new();
    }
    let mut pairs = query
        .split('&')
        .map(|part| {
            let (name, value) = part.split_once('=').unwrap_or((part, ""));
            (
                aws_percent_encode(name.as_bytes(), false),
                aws_percent_encode(value.as_bytes(), false),
            )
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
        .into_iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn canonical_aws_headers(headers: &BTreeMap<String, String>) -> (String, String) {
    let mut normalized = BTreeMap::<String, String>::new();
    for (name, value) in headers {
        if name.eq_ignore_ascii_case("authorization") {
            continue;
        }
        let normalized_name = name.to_ascii_lowercase();
        let normalized_value = normalize_aws_header_value(value);
        normalized
            .entry(normalized_name)
            .and_modify(|existing| {
                existing.push(',');
                existing.push_str(&normalized_value);
            })
            .or_insert(normalized_value);
    }
    let canonical_headers = normalized
        .iter()
        .map(|(name, value)| format!("{name}:{value}\n"))
        .collect::<String>();
    let signed_headers = normalized.keys().cloned().collect::<Vec<_>>().join(";");
    (canonical_headers, signed_headers)
}

fn normalize_aws_header_value(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn set_header_case_insensitive(headers: &mut BTreeMap<String, String>, name: &str, value: String) {
    remove_header_case_insensitive(headers, name);
    headers.insert(name.to_owned(), value);
}

fn remove_header_case_insensitive(headers: &mut BTreeMap<String, String>, name: &str) {
    let keys = headers
        .keys()
        .filter(|key| key.eq_ignore_ascii_case(name))
        .cloned()
        .collect::<Vec<_>>();
    for key in keys {
        headers.remove(&key);
    }
}

fn aws_percent_encode(bytes: &[u8], preserve_slash: bool) -> String {
    let mut result = String::new();
    for byte in bytes {
        let allowed = byte.is_ascii_alphanumeric() || matches!(*byte, b'-' | b'.' | b'_' | b'~');
        if allowed || (preserve_slash && *byte == b'/') {
            result.push(*byte as char);
        } else {
            result.push_str(&format!("%{byte:02X}"));
        }
    }
    result
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(digest::digest(&digest::SHA256, bytes).as_ref())
}

fn aws_sigv4_signing_key(secret: &str, date: &str, region: &str, service: &str) -> Vec<u8> {
    let k_date = hmac_sha256(format!("AWS4{secret}").as_bytes(), date.as_bytes());
    let k_region = hmac_sha256(&k_date, region.as_bytes());
    let k_service = hmac_sha256(&k_region, service.as_bytes());
    hmac_sha256(&k_service, b"aws4_request")
}

fn hmac_sha256(key: &[u8], bytes: &[u8]) -> Vec<u8> {
    hmac::sign(&hmac::Key::new(hmac::HMAC_SHA256, key), bytes)
        .as_ref()
        .to_vec()
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[(byte >> 4) as usize] as char);
        result.push(HEX[(byte & 0x0f) as usize] as char);
    }
    result
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BedrockPayloadOptions {
    pub cache_retention: Option<CacheRetention>,
    pub max_tokens: Option<u64>,
    pub temperature: Option<f64>,
    pub tool_choice: Option<BedrockToolChoice>,
    pub reasoning: Option<ThinkingLevel>,
    pub region: Option<String>,
    pub thinking_budgets: Option<ThinkingBudgets>,
    pub interleaved_thinking: Option<bool>,
    pub thinking_display: Option<String>,
    pub request_metadata: Option<Value>,
}

pub fn build_bedrock_payload(
    model: &Model,
    context: &Context,
    options: BedrockPayloadOptions,
) -> Value {
    let cache_retention = resolve_bedrock_cache_retention(options.cache_retention);
    let mut payload = json!({
        "modelId": model.id,
        "messages": convert_bedrock_messages(context, model, cache_retention),
    });
    if let Some(system) =
        build_bedrock_system_prompt(context.system_prompt.as_deref(), model, cache_retention)
    {
        payload["system"] = system;
    }
    if options.max_tokens.is_some() || options.temperature.is_some() {
        let mut inference_config = Map::new();
        if let Some(max_tokens) = options.max_tokens {
            inference_config.insert("maxTokens".to_owned(), json!(max_tokens));
        }
        if let Some(temperature) = options.temperature {
            inference_config.insert("temperature".to_owned(), json!(temperature));
        }
        payload["inferenceConfig"] = Value::Object(inference_config);
    }
    if let Some(fields) = build_bedrock_additional_model_request_fields(model, &options) {
        payload["additionalModelRequestFields"] = fields;
    }
    if let Some(tool_config) = build_bedrock_tool_config(&context.tools, options.tool_choice) {
        payload["toolConfig"] = tool_config;
    }
    if let Some(request_metadata) = options.request_metadata {
        payload["requestMetadata"] = request_metadata;
    }
    payload
}

pub fn parse_bedrock_tool_choice(value: &Value) -> Option<BedrockToolChoice> {
    match value.as_str() {
        Some("auto") => Some(BedrockToolChoice::Auto),
        Some("any") => Some(BedrockToolChoice::Any),
        Some("none") => Some(BedrockToolChoice::None),
        Some(_) => None,
        None => {
            let object = value.as_object()?;
            if object.get("type").and_then(Value::as_str) != Some("tool") {
                return None;
            }
            let name = object.get("name")?.as_str().map(str::to_owned)?;
            Some(BedrockToolChoice::Tool { name })
        }
    }
}

fn build_bedrock_tool_config(
    tools: &[Tool],
    tool_choice: Option<BedrockToolChoice>,
) -> Option<Value> {
    if tools.is_empty() || matches!(tool_choice, Some(BedrockToolChoice::None)) {
        return None;
    }

    let mut config = Map::new();
    config.insert(
        "tools".to_owned(),
        Value::Array(tools.iter().map(format_bedrock_tool).collect()),
    );
    if let Some(tool_choice) = tool_choice.and_then(format_bedrock_tool_choice) {
        config.insert("toolChoice".to_owned(), tool_choice);
    }
    Some(Value::Object(config))
}

fn format_bedrock_tool(tool: &Tool) -> Value {
    json!({
        "toolSpec": {
            "name": &tool.name,
            "description": &tool.description,
            "inputSchema": { "json": &tool.parameters },
        },
    })
}

fn format_bedrock_tool_choice(choice: BedrockToolChoice) -> Option<Value> {
    match choice {
        BedrockToolChoice::Auto => Some(json!({ "auto": {} })),
        BedrockToolChoice::Any => Some(json!({ "any": {} })),
        BedrockToolChoice::Tool { name } => Some(json!({ "tool": { "name": name } })),
        BedrockToolChoice::None => None,
    }
}

pub fn process_bedrock_converse_stream_events<I>(
    events: I,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    model: &Model,
) -> Result<(), String>
where
    I: IntoIterator<Item = Value>,
{
    let mut processor = BedrockConverseStreamProcessor::new();
    for event in events {
        processor.process_event(event, output, sender, model)?;
    }
    processor.finish(output, sender)
}

#[derive(Debug, Default)]
pub struct BedrockConverseStreamProcessor {
    content_indexes: BTreeMap<u64, usize>,
    partial_tool_json: BTreeMap<usize, String>,
}

impl BedrockConverseStreamProcessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_event(
        &mut self,
        event: Value,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
        model: &Model,
    ) -> Result<(), String> {
        if !event.is_object() {
            return Ok(());
        }

        if let Some(message) = bedrock_stream_exception_message(&event) {
            return finish_bedrock_stream_with_error(output, sender, message);
        }

        if let Some(message_start) = event.get("messageStart") {
            let role = message_start
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or_default();
            if role != "assistant" {
                return finish_bedrock_stream_with_error(
                    output,
                    sender,
                    "Unexpected assistant message start but got user message start instead"
                        .to_owned(),
                );
            }
            sender.push(AssistantMessageEvent::Start {
                partial: output.clone(),
            });
            return Ok(());
        }

        if let Some(content_block_start) = event.get("contentBlockStart") {
            handle_bedrock_content_block_start(
                content_block_start,
                output,
                sender,
                &mut self.content_indexes,
                &mut self.partial_tool_json,
            );
            return Ok(());
        }

        if let Some(content_block_delta) = event.get("contentBlockDelta") {
            handle_bedrock_content_block_delta(
                content_block_delta,
                output,
                sender,
                &mut self.content_indexes,
                &mut self.partial_tool_json,
            );
            return Ok(());
        }

        if let Some(content_block_stop) = event.get("contentBlockStop") {
            handle_bedrock_content_block_stop(
                content_block_stop,
                output,
                sender,
                &mut self.content_indexes,
                &mut self.partial_tool_json,
            );
            return Ok(());
        }

        if let Some(message_stop) = event.get("messageStop") {
            output.stop_reason = map_bedrock_stop_reason(
                message_stop
                    .get("stopReason")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            return Ok(());
        }

        if let Some(metadata) = event.get("metadata") {
            apply_bedrock_stream_metadata(output, model, metadata);
        }

        Ok(())
    }

    pub fn finish(
        self,
        output: &mut AssistantMessage,
        sender: &AssistantMessageEventSender,
    ) -> Result<(), String> {
        if output.stop_reason == StopReason::Error {
            return finish_bedrock_stream_with_error(
                output,
                sender,
                "An unknown error occurred".to_owned(),
            );
        }

        sender.push(AssistantMessageEvent::Done {
            reason: output.stop_reason,
            message: output.clone(),
        });
        Ok(())
    }
}

fn handle_bedrock_content_block_start(
    event: &Value,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_indexes: &mut BTreeMap<u64, usize>,
    partial_tool_json: &mut BTreeMap<usize, String>,
) {
    let Some(index) = event.get("contentBlockIndex").and_then(Value::as_u64) else {
        return;
    };
    let Some(tool_use) = event.pointer("/start/toolUse") else {
        return;
    };
    let content_index = output.content.len();
    output.content.push(AssistantContent::ToolCall(ToolCall {
        id: tool_use
            .get("toolUseId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        name: tool_use
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        arguments: Map::new(),
        thought_signature: None,
    }));
    content_indexes.insert(index, content_index);
    partial_tool_json.insert(content_index, String::new());
    sender.push(AssistantMessageEvent::ToolcallStart {
        content_index,
        partial: output.clone(),
    });
}

fn handle_bedrock_content_block_delta(
    event: &Value,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_indexes: &mut BTreeMap<u64, usize>,
    partial_tool_json: &mut BTreeMap<usize, String>,
) {
    let Some(index) = event.get("contentBlockIndex").and_then(Value::as_u64) else {
        return;
    };
    let Some(delta) = event.get("delta") else {
        return;
    };

    if let Some(text_delta) = delta.get("text").and_then(Value::as_str) {
        let content_index = ensure_bedrock_text_block(index, output, sender, content_indexes);
        if let Some(AssistantContent::Text(block)) = output.content.get_mut(content_index) {
            block.text.push_str(text_delta);
            sender.push(AssistantMessageEvent::TextDelta {
                content_index,
                delta: text_delta.to_owned(),
                partial: output.clone(),
            });
        }
        return;
    }

    if let Some(tool_use) = delta.get("toolUse") {
        let Some(content_index) = content_indexes.get(&index).copied() else {
            return;
        };
        if !matches!(
            output.content.get(content_index),
            Some(AssistantContent::ToolCall(_))
        ) {
            return;
        }
        let input_delta = tool_use
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let partial = partial_tool_json.entry(content_index).or_default();
        partial.push_str(input_delta);
        if let Some(AssistantContent::ToolCall(block)) = output.content.get_mut(content_index) {
            block.arguments = parse_bedrock_tool_arguments(partial);
        }
        sender.push(AssistantMessageEvent::ToolcallDelta {
            content_index,
            delta: input_delta.to_owned(),
            partial: output.clone(),
        });
        return;
    }

    if let Some(reasoning_content) = delta.get("reasoningContent") {
        let content_index = ensure_bedrock_thinking_block(index, output, sender, content_indexes);
        let mut thinking_delta_event = None;
        if let Some(AssistantContent::Thinking(block)) = output.content.get_mut(content_index) {
            if let Some(thinking_delta) = reasoning_content.get("text").and_then(Value::as_str)
                && !thinking_delta.is_empty()
            {
                block.thinking.push_str(thinking_delta);
                thinking_delta_event = Some(thinking_delta.to_owned());
            }
            if let Some(signature_delta) =
                reasoning_content.get("signature").and_then(Value::as_str)
                && !signature_delta.is_empty()
            {
                let signature = block.thinking_signature.get_or_insert_with(String::new);
                signature.push_str(signature_delta);
            }
        }
        if let Some(delta) = thinking_delta_event {
            sender.push(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                partial: output.clone(),
            });
        }
    }
}

fn handle_bedrock_content_block_stop(
    event: &Value,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_indexes: &mut BTreeMap<u64, usize>,
    partial_tool_json: &mut BTreeMap<usize, String>,
) {
    let Some(index) = event.get("contentBlockIndex").and_then(Value::as_u64) else {
        return;
    };
    let Some(content_index) = content_indexes.remove(&index) else {
        return;
    };

    match output.content.get_mut(content_index) {
        Some(AssistantContent::Text(block)) => {
            sender.push(AssistantMessageEvent::TextEnd {
                content_index,
                content: block.text.clone(),
                partial: output.clone(),
            });
        }
        Some(AssistantContent::Thinking(block)) => {
            sender.push(AssistantMessageEvent::ThinkingEnd {
                content_index,
                content: block.thinking.clone(),
                partial: output.clone(),
            });
        }
        Some(AssistantContent::ToolCall(block)) => {
            if let Some(partial) = partial_tool_json.remove(&content_index) {
                block.arguments = parse_bedrock_tool_arguments(&partial);
            }
            let tool_call = block.clone();
            sender.push(AssistantMessageEvent::ToolcallEnd {
                content_index,
                tool_call,
                partial: output.clone(),
            });
        }
        _ => {}
    }
}

fn ensure_bedrock_text_block(
    index: u64,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_indexes: &mut BTreeMap<u64, usize>,
) -> usize {
    if let Some(content_index) = content_indexes.get(&index).copied() {
        return content_index;
    }
    let content_index = output.content.len();
    output
        .content
        .push(AssistantContent::Text(TextContent::new("")));
    content_indexes.insert(index, content_index);
    sender.push(AssistantMessageEvent::TextStart {
        content_index,
        partial: output.clone(),
    });
    content_index
}

fn ensure_bedrock_thinking_block(
    index: u64,
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    content_indexes: &mut BTreeMap<u64, usize>,
) -> usize {
    if let Some(content_index) = content_indexes.get(&index).copied() {
        return content_index;
    }
    let content_index = output.content.len();
    output
        .content
        .push(AssistantContent::Thinking(ThinkingContent::new("")));
    content_indexes.insert(index, content_index);
    sender.push(AssistantMessageEvent::ThinkingStart {
        content_index,
        partial: output.clone(),
    });
    content_index
}

fn apply_bedrock_stream_metadata(output: &mut AssistantMessage, model: &Model, event: &Value) {
    let Some(usage) = event.get("usage") else {
        return;
    };
    let input = usage
        .get("inputTokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let output_tokens = usage
        .get("outputTokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let cache_read = usage
        .get("cacheReadInputTokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let cache_write = usage
        .get("cacheWriteInputTokens")
        .and_then(Value::as_u64)
        .unwrap_or_default();
    let total_tokens = usage
        .get("totalTokens")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            input
                .saturating_add(output_tokens)
                .saturating_add(cache_read)
                .saturating_add(cache_write)
        });
    output.usage = Usage {
        input,
        output: output_tokens,
        cache_read,
        cache_write,
        total_tokens,
        cost: Default::default(),
    };
    calculate_cost(model, &mut output.usage);
}

fn map_bedrock_stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" | "model_context_window_exceeded" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Error,
    }
}

fn parse_bedrock_tool_arguments(arguments: &str) -> Map<String, Value> {
    parse_streaming_json(Some(arguments))
        .as_object()
        .cloned()
        .unwrap_or_default()
}

fn finish_bedrock_stream_with_error(
    output: &mut AssistantMessage,
    sender: &AssistantMessageEventSender,
    message: String,
) -> Result<(), String> {
    output.stop_reason = StopReason::Error;
    output.error_message = Some(message.clone());
    sender.push(AssistantMessageEvent::Error {
        reason: StopReason::Error,
        error: output.clone(),
    });
    Err(message)
}

fn bedrock_stream_exception_message(event: &Value) -> Option<String> {
    const EXCEPTIONS: [(&str, &str); 5] = [
        ("internalServerException", "Internal server error"),
        ("modelStreamErrorException", "Model stream error"),
        ("validationException", "Validation error"),
        ("throttlingException", "Throttling error"),
        ("serviceUnavailableException", "Service unavailable"),
    ];

    for (key, prefix) in EXCEPTIONS {
        if let Some(error) = event.get(key) {
            let message = error
                .get("message")
                .or_else(|| error.get("Message"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| error.to_string());
            return Some(format!("{prefix}: {message}"));
        }
    }

    None
}

pub fn convert_bedrock_messages(
    context: &Context,
    model: &Model,
    cache_retention: CacheRetention,
) -> Vec<Value> {
    let transformed_messages = transform_messages(
        &context.messages,
        model,
        Some(&|id, _model, _source| normalize_bedrock_tool_call_id(id)),
    );
    let mut result = Vec::new();
    let mut index = 0;
    while index < transformed_messages.len() {
        match &transformed_messages[index] {
            Message::User(user) => {
                let content = match &user.content {
                    UserContentValue::Plain(text) => vec![json!({ "text": text })],
                    UserContentValue::Blocks(blocks) => blocks
                        .iter()
                        .map(|block| match block {
                            UserContent::Text(text) => json!({ "text": text.text }),
                            UserContent::Image(image) => {
                                json!({ "image": bedrock_image_block(image) })
                            }
                        })
                        .collect(),
                };
                if !content.is_empty() {
                    result.push(json!({ "role": "user", "content": content }));
                }
            }
            Message::Assistant(assistant) => {
                let content = assistant
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
                            Some(json!({ "text": text.text }))
                        }
                        AssistantContent::ToolCall(tool_call) => Some(json!({
                            "toolUse": {
                                "toolUseId": tool_call.id,
                                "name": tool_call.name,
                                "input": tool_call.arguments,
                            },
                        })),
                        AssistantContent::Thinking(thinking)
                            if !thinking.thinking.trim().is_empty() =>
                        {
                            if supports_bedrock_thinking_signature(model) {
                                if let Some(signature) = thinking
                                    .thinking_signature
                                    .as_deref()
                                    .filter(|signature| !signature.trim().is_empty())
                                {
                                    Some(json!({
                                        "reasoningContent": {
                                            "reasoningText": {
                                                "text": thinking.thinking,
                                                "signature": signature,
                                            },
                                        },
                                    }))
                                } else {
                                    Some(json!({ "text": thinking.thinking }))
                                }
                            } else {
                                Some(json!({
                                    "reasoningContent": {
                                        "reasoningText": { "text": thinking.thinking },
                                    },
                                }))
                            }
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !content.is_empty() {
                    result.push(json!({ "role": "assistant", "content": content }));
                }
            }
            Message::ToolResult(_) => {
                let mut tool_results = Vec::new();
                let mut next = index;
                while next < transformed_messages.len() {
                    let Message::ToolResult(tool_result) = &transformed_messages[next] else {
                        break;
                    };
                    let content = tool_result
                        .content
                        .iter()
                        .map(|content| match content {
                            ToolResultContent::Text(text) => json!({ "text": text.text }),
                            ToolResultContent::Image(image) => {
                                json!({ "image": bedrock_image_block(image) })
                            }
                        })
                        .collect::<Vec<_>>();
                    tool_results.push(json!({
                        "toolResult": {
                            "toolUseId": tool_result.tool_call_id,
                            "content": content,
                            "status": if tool_result.is_error { "error" } else { "success" },
                        },
                    }));
                    next += 1;
                }
                index = next - 1;
                result.push(json!({ "role": "user", "content": tool_results }));
            }
        }
        index += 1;
    }

    append_bedrock_cache_point_to_last_user(&mut result, model, cache_retention);
    result
}

pub fn convert_bedrock_raw_messages(
    messages: &[Value],
    model: &Model,
    cache_retention: CacheRetention,
) -> Vec<Value> {
    let mut result = Vec::new();
    for message in messages {
        let Some(role) = message.get("role").and_then(Value::as_str) else {
            continue;
        };
        match role {
            "user" => {
                let content = convert_bedrock_raw_content(message.get("content"), true);
                if !content.is_empty() {
                    result.push(json!({ "role": "user", "content": content }));
                }
            }
            "assistant" => {
                let content = convert_bedrock_raw_content(message.get("content"), false);
                if !content.is_empty() {
                    result.push(json!({ "role": "assistant", "content": content }));
                }
            }
            _ => {}
        }
    }
    append_bedrock_cache_point_to_last_user(&mut result, model, cache_retention);
    result
}

fn convert_bedrock_raw_content(content: Option<&Value>, allow_images: bool) -> Vec<Value> {
    match content {
        Some(Value::String(text)) => vec![json!({ "text": text })],
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|block| match block.get("type").and_then(Value::as_str) {
                Some("text") => block.get("text").map(|text| json!({ "text": text })),
                Some("image") if allow_images => {
                    let mime_type = block
                        .get("mimeType")
                        .and_then(Value::as_str)
                        .unwrap_or("image/png");
                    let data = block
                        .get("data")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    Some(json!({ "image": bedrock_image_block_parts(mime_type, data) }))
                }
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

pub fn build_bedrock_system_prompt(
    system_prompt: Option<&str>,
    model: &Model,
    cache_retention: CacheRetention,
) -> Option<Value> {
    let system_prompt = system_prompt?;
    let mut blocks = vec![json!({ "text": system_prompt })];
    if cache_retention != CacheRetention::None && supports_bedrock_prompt_caching(model) {
        blocks.push(bedrock_cache_point(cache_retention));
    }
    Some(Value::Array(blocks))
}

pub fn build_bedrock_additional_model_request_fields(
    model: &Model,
    options: &BedrockPayloadOptions,
) -> Option<Value> {
    let reasoning = options.reasoning?;
    if !model.reasoning || !is_bedrock_anthropic_claude_model(model) {
        return None;
    }

    let display = if is_gov_cloud_bedrock_target(model, options.region.as_deref()) {
        None
    } else {
        Some(
            options
                .thinking_display
                .clone()
                .unwrap_or_else(|| "summarized".to_owned()),
        )
    };

    let mut result = if supports_bedrock_adaptive_thinking(model) {
        let mut thinking = json!({ "type": "adaptive" });
        if let Some(display) = display {
            thinking["display"] = Value::String(display);
        }
        json!({
            "thinking": thinking,
            "output_config": {
                "effort": map_bedrock_thinking_level_to_effort(model, reasoning),
            },
        })
    } else {
        let mut thinking = json!({
            "type": "enabled",
            "budget_tokens": bedrock_thinking_budget(options, reasoning),
        });
        if let Some(display) = display {
            thinking["display"] = Value::String(display);
        }
        json!({ "thinking": thinking })
    };

    if !supports_bedrock_adaptive_thinking(model) && options.interleaved_thinking != Some(false) {
        result["anthropic_beta"] = json!(["interleaved-thinking-2025-05-14"]);
    }

    Some(result)
}

pub fn supports_bedrock_prompt_caching(model: &Model) -> bool {
    let candidates = bedrock_model_match_candidates(model);
    let has_claude_ref = candidates
        .iter()
        .any(|candidate| candidate.contains("claude"));
    if !has_claude_ref {
        return std::env::var("AWS_BEDROCK_FORCE_CACHE").ok().as_deref() == Some("1");
    }
    candidates.iter().any(|candidate| {
        candidate.contains("-4-")
            || candidate.contains("claude-3-7-sonnet")
            || candidate.contains("claude-3-5-haiku")
    })
}

fn append_bedrock_cache_point_to_last_user(
    messages: &mut [Value],
    model: &Model,
    cache_retention: CacheRetention,
) {
    if cache_retention == CacheRetention::None || !supports_bedrock_prompt_caching(model) {
        return;
    }
    let Some(last_message) = messages.last_mut() else {
        return;
    };
    if last_message.get("role").and_then(Value::as_str) != Some("user") {
        return;
    }
    if let Some(content) = last_message
        .get_mut("content")
        .and_then(Value::as_array_mut)
    {
        content.push(bedrock_cache_point(cache_retention));
    }
}

fn resolve_bedrock_cache_retention(cache_retention: Option<CacheRetention>) -> CacheRetention {
    if let Some(cache_retention) = cache_retention {
        return cache_retention;
    }
    if std::env::var("PI_CACHE_RETENTION").ok().as_deref() == Some("long") {
        CacheRetention::Long
    } else {
        CacheRetention::Short
    }
}

fn bedrock_cache_point(cache_retention: CacheRetention) -> Value {
    let mut cache_point = json!({ "type": "default" });
    if cache_retention == CacheRetention::Long {
        cache_point["ttl"] = Value::String("ONE_HOUR".to_owned());
    }
    json!({ "cachePoint": cache_point })
}

fn bedrock_image_block(image: &ImageContent) -> Value {
    bedrock_image_block_parts(&image.mime_type, &image.data)
}

fn bedrock_image_block_parts(mime_type: &str, data: &str) -> Value {
    let format = match mime_type.to_ascii_lowercase().as_str() {
        "image/jpeg" | "image/jpg" => "jpeg".to_owned(),
        "image/png" => "png".to_owned(),
        "image/gif" => "gif".to_owned(),
        "image/webp" => "webp".to_owned(),
        other => other.strip_prefix("image/").unwrap_or(other).to_owned(),
    };
    json!({
        "format": format,
        "source": { "bytes": data },
    })
}

fn normalize_bedrock_tool_call_id(id: &str) -> String {
    id.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

fn is_bedrock_anthropic_claude_model(model: &Model) -> bool {
    bedrock_model_match_candidates(model)
        .iter()
        .any(|candidate| {
            candidate.contains("anthropic-claude")
                || candidate.contains("anthropic/claude")
                || candidate.contains("claude")
        })
}

fn supports_bedrock_thinking_signature(model: &Model) -> bool {
    is_bedrock_anthropic_claude_model(model)
}

fn supports_bedrock_adaptive_thinking(model: &Model) -> bool {
    bedrock_model_match_candidates(model)
        .iter()
        .any(|candidate| {
            candidate.contains("opus-4-6")
                || candidate.contains("opus-4-7")
                || candidate.contains("sonnet-4-6")
        })
}

fn supports_bedrock_native_xhigh_effort(model: &Model) -> bool {
    bedrock_model_match_candidates(model)
        .iter()
        .any(|candidate| candidate.contains("opus-4-7"))
}

fn map_bedrock_thinking_level_to_effort(model: &Model, level: ThinkingLevel) -> &'static str {
    if level == ThinkingLevel::XHigh && supports_bedrock_native_xhigh_effort(model) {
        return "xhigh";
    }
    if let Some(Some(mapped)) = model.thinking_level_map.get(&level) {
        return match mapped.as_str() {
            "low" => "low",
            "medium" => "medium",
            "high" => "high",
            "xhigh" => "xhigh",
            "max" => "max",
            _ => "high",
        };
    }
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High | ThinkingLevel::XHigh | ThinkingLevel::Off => "high",
    }
}

fn bedrock_thinking_budget(options: &BedrockPayloadOptions, level: ThinkingLevel) -> u64 {
    let budgets = options.thinking_budgets.as_ref();
    match level {
        ThinkingLevel::Minimal => budgets.and_then(|budget| budget.minimal).unwrap_or(1_024),
        ThinkingLevel::Low => budgets.and_then(|budget| budget.low).unwrap_or(2_048),
        ThinkingLevel::Medium => budgets.and_then(|budget| budget.medium).unwrap_or(8_192),
        ThinkingLevel::High => budgets.and_then(|budget| budget.high).unwrap_or(16_384),
        ThinkingLevel::XHigh => budgets.and_then(|budget| budget.high).unwrap_or(16_384),
        ThinkingLevel::Off => 16_384,
    }
}

fn is_gov_cloud_bedrock_target(model: &Model, region: Option<&str>) -> bool {
    if region
        .map(|region| region.to_ascii_lowercase().starts_with("us-gov-"))
        .unwrap_or(false)
    {
        return true;
    }
    let model_id = model.id.to_ascii_lowercase();
    model_id.starts_with("us-gov.") || model_id.starts_with("arn:aws-us-gov:")
}

fn bedrock_model_match_candidates(model: &Model) -> Vec<String> {
    [model.id.as_str(), model.name.as_str()]
        .into_iter()
        .flat_map(|value| {
            let lower = value.to_ascii_lowercase();
            let normalized = lower
                .chars()
                .map(|ch| {
                    if matches!(ch, ' ' | '_' | '.' | ':') {
                        '-'
                    } else {
                        ch
                    }
                })
                .collect::<String>();
            [lower, normalized]
        })
        .collect()
}
