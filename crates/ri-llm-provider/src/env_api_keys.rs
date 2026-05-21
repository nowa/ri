use std::{env, path::PathBuf};

fn api_key_env_vars(provider: &str) -> Option<&'static [&'static str]> {
    match provider {
        "github-copilot" => Some(&["COPILOT_GITHUB_TOKEN"]),
        "anthropic" => Some(&["ANTHROPIC_OAUTH_TOKEN", "ANTHROPIC_API_KEY"]),
        "openai" => Some(&["OPENAI_API_KEY"]),
        "azure-openai-responses" => Some(&["AZURE_OPENAI_API_KEY"]),
        "deepseek" => Some(&["DEEPSEEK_API_KEY"]),
        "google" => Some(&["GEMINI_API_KEY"]),
        "google-vertex" => Some(&["GOOGLE_CLOUD_API_KEY"]),
        "groq" => Some(&["GROQ_API_KEY"]),
        "cerebras" => Some(&["CEREBRAS_API_KEY"]),
        "xai" => Some(&["XAI_API_KEY"]),
        "openrouter" => Some(&["OPENROUTER_API_KEY"]),
        "vercel-ai-gateway" => Some(&["AI_GATEWAY_API_KEY"]),
        "zai" => Some(&["ZAI_API_KEY"]),
        "mistral" => Some(&["MISTRAL_API_KEY"]),
        "minimax" => Some(&["MINIMAX_API_KEY"]),
        "minimax-cn" => Some(&["MINIMAX_CN_API_KEY"]),
        "moonshotai" | "moonshotai-cn" => Some(&["MOONSHOT_API_KEY"]),
        "huggingface" => Some(&["HF_TOKEN"]),
        "fireworks" => Some(&["FIREWORKS_API_KEY"]),
        "together" => Some(&["TOGETHER_API_KEY"]),
        "opencode" | "opencode-go" => Some(&["OPENCODE_API_KEY"]),
        "kimi-coding" => Some(&["KIMI_API_KEY"]),
        "cloudflare-workers-ai" | "cloudflare-ai-gateway" => Some(&["CLOUDFLARE_API_KEY"]),
        "xiaomi" => Some(&["XIAOMI_API_KEY"]),
        "xiaomi-token-plan-cn" => Some(&["XIAOMI_TOKEN_PLAN_CN_API_KEY"]),
        "xiaomi-token-plan-ams" => Some(&["XIAOMI_TOKEN_PLAN_AMS_API_KEY"]),
        "xiaomi-token-plan-sgp" => Some(&["XIAOMI_TOKEN_PLAN_SGP_API_KEY"]),
        _ => None,
    }
}

pub fn api_key_env_var_names(provider: &str) -> Option<&'static [&'static str]> {
    api_key_env_vars(provider)
}

pub fn find_env_keys(provider: &str) -> Option<Vec<String>> {
    let keys: Vec<String> = api_key_env_vars(provider)?
        .iter()
        .filter(|name| env_string_nonempty(name).is_some())
        .map(|name| (*name).to_owned())
        .collect();
    (!keys.is_empty()).then_some(keys)
}

pub fn get_env_api_key(provider: &str) -> Option<String> {
    if let Some(first_key) = find_env_keys(provider).and_then(|keys| keys.into_iter().next()) {
        return env_string_nonempty(&first_key);
    }

    if provider == "amazon-bedrock" && has_bedrock_credentials() {
        return Some("<authenticated>".to_owned());
    }

    if provider == "google-vertex" && has_vertex_credentials() {
        return Some("<authenticated>".to_owned());
    }

    None
}

fn env_string_nonempty(key: &str) -> Option<String> {
    env::var(key).ok().filter(|value| !value.is_empty())
}

fn env_os_nonempty(key: &str) -> bool {
    env::var_os(key).is_some_and(|value| !value.as_os_str().is_empty())
}

fn env_path_nonempty(key: &str) -> Option<PathBuf> {
    env::var_os(key)
        .filter(|value| !value.as_os_str().is_empty())
        .map(PathBuf::from)
}

fn has_bedrock_credentials() -> bool {
    env_os_nonempty("AWS_PROFILE")
        || (env_os_nonempty("AWS_ACCESS_KEY_ID") && env_os_nonempty("AWS_SECRET_ACCESS_KEY"))
        || env_os_nonempty("AWS_BEARER_TOKEN_BEDROCK")
        || (env_os_nonempty("AWS_WEB_IDENTITY_TOKEN_FILE") && env_os_nonempty("AWS_ROLE_ARN"))
        || env_os_nonempty("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI")
        || env_os_nonempty("AWS_CONTAINER_CREDENTIALS_FULL_URI")
}

fn has_vertex_credentials() -> bool {
    let has_project = env_string_nonempty("GOOGLE_CLOUD_PROJECT").is_some()
        || env_string_nonempty("GCLOUD_PROJECT").is_some();
    let has_location = env_string_nonempty("GOOGLE_CLOUD_LOCATION").is_some();
    google_adc_credentials_exists() && has_project && has_location
}

fn google_adc_credentials_exists() -> bool {
    if let Some(path) = env_path_nonempty("GOOGLE_APPLICATION_CREDENTIALS") {
        return path.exists();
    }

    default_google_adc_paths()
        .into_iter()
        .any(|path| path.exists())
}

fn default_google_adc_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(config_dir) = env_path_nonempty("APPDATA") {
        paths.push(
            config_dir
                .join("gcloud")
                .join("application_default_credentials.json"),
        );
    }
    if let Some(home) = env_path_nonempty("HOME").or_else(|| env_path_nonempty("USERPROFILE")) {
        paths.push(
            home.join(".config")
                .join("gcloud")
                .join("application_default_credentials.json"),
        );
    }
    paths
}
