use std::env;

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
        .filter(|name| env::var_os(name).is_some())
        .map(|name| (*name).to_owned())
        .collect();
    (!keys.is_empty()).then_some(keys)
}

pub fn get_env_api_key(provider: &str) -> Option<String> {
    if let Some(first_key) = find_env_keys(provider).and_then(|keys| keys.into_iter().next()) {
        return env::var(first_key).ok();
    }

    if provider == "amazon-bedrock" && has_bedrock_credentials() {
        return Some("<authenticated>".to_owned());
    }

    if provider == "google-vertex" && has_vertex_credentials() {
        return Some("<authenticated>".to_owned());
    }

    None
}

fn has_bedrock_credentials() -> bool {
    env::var_os("AWS_PROFILE").is_some()
        || (env::var_os("AWS_ACCESS_KEY_ID").is_some()
            && env::var_os("AWS_SECRET_ACCESS_KEY").is_some())
        || env::var_os("AWS_BEARER_TOKEN_BEDROCK").is_some()
        || (env::var_os("AWS_WEB_IDENTITY_TOKEN_FILE").is_some()
            && env::var_os("AWS_ROLE_ARN").is_some())
        || env::var_os("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
        || env::var_os("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some()
}

fn has_vertex_credentials() -> bool {
    let has_project =
        env::var_os("GOOGLE_CLOUD_PROJECT").is_some() || env::var_os("GCLOUD_PROJECT").is_some();
    let has_location = env::var_os("GOOGLE_CLOUD_LOCATION").is_some();
    let has_key = env::var_os("GOOGLE_CLOUD_API_KEY").is_some();
    let has_adc_path = env::var_os("GOOGLE_APPLICATION_CREDENTIALS").is_some();
    (has_key || has_adc_path) && has_project && has_location
}
