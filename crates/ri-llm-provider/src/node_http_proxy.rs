use std::collections::HashMap;
use std::env;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub const UNSUPPORTED_PROXY_PROTOCOL_MESSAGE: &str = "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported; use an HTTP or HTTPS proxy URL.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyUrl {
    raw: String,
}

impl ProxyUrl {
    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

impl std::fmt::Display for ProxyUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.raw)
    }
}

pub fn resolve_http_proxy_url_for_target(target_url: &str) -> Result<Option<ProxyUrl>, String> {
    let Some(target) = ParsedUrl::parse(target_url) else {
        return Ok(None);
    };
    if !should_proxy_hostname(&target.hostname, target.port) {
        return Ok(None);
    }
    let mut proxy = proxy_env_with_npm_fallback(&format!("{}_proxy", target.protocol))
        .or_else(|| proxy_env("npm_config_proxy"))
        .or_else(|| proxy_env_with_npm_fallback("all_proxy"))
        .unwrap_or_default();
    if proxy.is_empty() {
        return Ok(None);
    }
    if !proxy.contains("://") {
        proxy = format!("{}://{proxy}", target.protocol);
    }
    let proxy_url = reqwest::Url::parse(&proxy)
        .map_err(|error| format!("Invalid proxy URL {proxy:?}: {error}"))?;
    let protocol = proxy_url.scheme();
    if protocol != "http" && protocol != "https" {
        return Err(format!(
            "{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {protocol}:"
        ));
    }
    Ok(Some(ProxyUrl {
        raw: normalize_proxy_url(proxy_url.as_str()),
    }))
}

pub fn resolve_http_proxy_url_for_websocket_target(
    target_url: &str,
) -> Result<Option<ProxyUrl>, String> {
    let http_target = if let Some(rest) = target_url.strip_prefix("wss://") {
        format!("https://{rest}")
    } else if let Some(rest) = target_url.strip_prefix("ws://") {
        format!("http://{rest}")
    } else {
        target_url.to_owned()
    };
    resolve_http_proxy_url_for_target(&http_target)
}

// Gateways commonly drop idle connections after ~60s; keeping our pool idle
// timeout below that means we never try to reuse a connection the gateway has
// already silently closed.
const POOL_IDLE_TIMEOUT: Duration = Duration::from_secs(50);

// One client per resolved proxy configuration. The proxy URL is the only
// per-target variable in client construction, so caching on it preserves the
// exact configuration each target would have received, while letting every
// request to the same proxy configuration share one connection pool.
static CLIENT_CACHE: OnceLock<Mutex<HashMap<Option<String>, reqwest::Client>>> = OnceLock::new();

pub fn reqwest_client_for_target(target_url: &str) -> Result<reqwest::Client, String> {
    let proxy_url = resolve_http_proxy_url_for_target(target_url)?;
    let (client, _cached) = cached_or_build_client(proxy_url)?;
    Ok(client)
}

fn cached_or_build_client(proxy_url: Option<ProxyUrl>) -> Result<(reqwest::Client, bool), String> {
    let key = proxy_url.as_ref().map(|proxy| proxy.as_str().to_owned());
    let cache = CLIENT_CACHE.get_or_init(Default::default);
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(client) = cache.get(&key) {
        return Ok((client.clone(), true));
    }
    let mut builder = reqwest::Client::builder()
        .no_proxy()
        .pool_idle_timeout(POOL_IDLE_TIMEOUT);
    if let Some(proxy_url) = &proxy_url {
        let proxy = reqwest::Proxy::all(proxy_url.as_str())
            .map_err(|error| format!("Invalid proxy URL {:?}: {error}", proxy_url.as_str()))?;
        builder = builder.proxy(proxy);
    }
    let client = builder
        .build()
        .map_err(|error| format!("Could not build HTTP client: {error}"))?;
    cache.insert(key, client.clone());
    Ok((client, false))
}

fn proxy_env(key: &str) -> Option<String> {
    env::var(key.to_ascii_lowercase())
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| env::var(key.to_ascii_uppercase()).ok())
        .filter(|value| !value.is_empty())
}

fn proxy_env_with_npm_fallback(key: &str) -> Option<String> {
    proxy_env(key).or_else(|| proxy_env(&format!("npm_config_{key}")))
}

fn should_proxy_hostname(hostname: &str, port: u16) -> bool {
    let no_proxy = proxy_env_with_npm_fallback("no_proxy")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if no_proxy.is_empty() {
        return true;
    }
    if no_proxy == "*" {
        return false;
    }
    no_proxy
        .split(|character: char| character == ',' || character.is_whitespace())
        .all(|entry| {
            if entry.is_empty() {
                return true;
            }
            let (mut proxy_hostname, proxy_port) = split_host_port(entry);
            if let Some(proxy_port) = proxy_port {
                if proxy_port != port {
                    return true;
                }
            }
            if !proxy_hostname.starts_with(['.', '*']) {
                return hostname != proxy_hostname;
            }
            if proxy_hostname.starts_with('*') {
                proxy_hostname = &proxy_hostname[1..];
            }
            !hostname.ends_with(proxy_hostname)
        })
}

fn split_host_port(value: &str) -> (&str, Option<u16>) {
    let Some((host, port)) = value.rsplit_once(':') else {
        return (value, None);
    };
    match port.parse::<u16>() {
        Ok(port) => (host, Some(port)),
        Err(_) => (value, None),
    }
}

fn normalize_proxy_url(proxy: &str) -> String {
    if proxy.ends_with('/') {
        proxy.to_owned()
    } else {
        format!("{proxy}/")
    }
}

struct ParsedUrl {
    protocol: String,
    hostname: String,
    port: u16,
}

impl ParsedUrl {
    fn parse(value: &str) -> Option<Self> {
        let (protocol, rest) = value.split_once("://")?;
        if protocol.is_empty() {
            return None;
        }
        let authority = rest.split('/').next().unwrap_or(rest);
        let (hostname, explicit_port) = split_host_port(authority);
        if hostname.is_empty() {
            return None;
        }
        let port = explicit_port.unwrap_or_else(|| default_port(protocol));
        Some(Self {
            protocol: protocol.to_ascii_lowercase(),
            hostname: hostname.to_ascii_lowercase(),
            port,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proxy(raw: &str) -> ProxyUrl {
        ProxyUrl {
            raw: raw.to_owned(),
        }
    }

    #[test]
    fn same_proxy_configuration_reuses_cached_client() {
        let key = proxy("http://reuse-test-proxy.example:18080/");
        let (_, cached) = cached_or_build_client(Some(key.clone())).expect("first build");
        assert!(!cached, "first call must build a fresh client");
        let (_, cached) = cached_or_build_client(Some(key)).expect("second build");
        assert!(cached, "second call with the same proxy must hit the cache");
    }

    #[test]
    fn different_proxy_configurations_get_distinct_clients() {
        let (_, cached) = cached_or_build_client(Some(proxy("http://distinct-a.example:18081/")))
            .expect("first build");
        assert!(!cached);
        let (_, cached) = cached_or_build_client(Some(proxy("http://distinct-b.example:18082/")))
            .expect("second build");
        assert!(
            !cached,
            "a different proxy configuration must not share a client"
        );
    }

    #[test]
    fn no_proxy_targets_share_one_client() {
        let (_, _) = cached_or_build_client(None).expect("first build");
        let (_, cached) = cached_or_build_client(None).expect("second build");
        assert!(
            cached,
            "all direct (no-proxy) targets must share one client"
        );
    }
}

fn default_port(protocol: &str) -> u16 {
    match protocol {
        "ftp" => 21,
        "gopher" => 70,
        "http" | "ws" => 80,
        "https" | "wss" => 443,
        _ => 0,
    }
}
