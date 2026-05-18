use std::env;

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
    let mut proxy = proxy_env(&format!("{}_proxy", target.protocol))
        .or_else(|| proxy_env("all_proxy"))
        .unwrap_or_default();
    if proxy.is_empty() {
        return Ok(None);
    }
    if !proxy.contains("://") {
        proxy = format!("{}://{proxy}", target.protocol);
    }
    let Some(protocol) = proxy.split_once("://").map(|(protocol, _)| protocol) else {
        return Err(format!("Invalid proxy URL {proxy:?}"));
    };
    if protocol != "http" && protocol != "https" {
        return Err(format!(
            "{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {protocol}:"
        ));
    }
    Ok(Some(ProxyUrl {
        raw: normalize_proxy_url(&proxy),
    }))
}

fn proxy_env(key: &str) -> Option<String> {
    env::var(key.to_ascii_lowercase())
        .ok()
        .or_else(|| env::var(key.to_ascii_uppercase()).ok())
        .filter(|value| !value.is_empty())
}

fn should_proxy_hostname(hostname: &str, port: u16) -> bool {
    let no_proxy = proxy_env("no_proxy")
        .unwrap_or_default()
        .to_ascii_lowercase();
    if no_proxy.is_empty() {
        return true;
    }
    if no_proxy == "*" {
        return false;
    }
    no_proxy.split([',', ' ', '\t']).all(|entry| {
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
        let authority = rest.split('/').next().unwrap_or(rest);
        let (hostname, explicit_port) = split_host_port(authority);
        let port = explicit_port.unwrap_or_else(|| default_port(protocol));
        Some(Self {
            protocol: protocol.to_ascii_lowercase(),
            hostname: hostname.to_ascii_lowercase(),
            port,
        })
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
