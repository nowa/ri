use crate::{
    anthropic_oauth::{
        ANTHROPIC_OAUTH_TOKEN_URL, OAuthCallback, OAuthCredentials, OAuthLoginFlow,
        exchange_anthropic_authorization_code_with_url_at, parse_authorization_input,
        start_anthropic_oauth_login_flow,
    },
    github_copilot_oauth::{
        GitHubCopilotCredentials, GitHubDeviceCode, github_copilot_urls,
        login_github_copilot_device_flow_for_urls, normalize_github_domain,
    },
    oauth_auth_storage::{
        AuthCredential, AuthStorage, OAuthProviderInfo, StoredOAuthCredentials, get_oauth_provider,
        get_oauth_providers, load_auth_storage_from_path, save_auth_storage_to_path,
    },
    openai_codex_oauth::{
        OPENAI_CODEX_OAUTH_TOKEN_URL, exchange_openai_codex_authorization_code_with_url_at,
        start_openai_codex_oauth_login_flow,
    },
    types::now_millis,
};
use std::{
    io::{self, Write},
    path::Path,
    sync::mpsc,
};

pub const CLI_AUTH_FILE: &str = "auth.json";

pub fn render_cli_help(binary_name: &str, providers: &[OAuthProviderInfo]) -> String {
    let provider_list = providers
        .iter()
        .map(format_provider_line)
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        concat!(
            "Usage: {binary_name} <command> [provider]\n\n",
            "Commands:\n",
            "  login [provider]  Login to an OAuth provider\n",
            "  list              List available providers\n\n",
            "Providers:\n",
            "{provider_list}\n\n",
            "Examples:\n",
            "  {binary_name} login              # interactive provider selection\n",
            "  {binary_name} login anthropic    # login to specific provider\n",
            "  {binary_name} list               # list providers\n",
        ),
        binary_name = binary_name,
        provider_list = provider_list
    )
}

pub fn render_cli_provider_list(providers: &[OAuthProviderInfo]) -> String {
    let provider_list = providers
        .iter()
        .map(format_provider_line)
        .collect::<Vec<_>>()
        .join("\n");
    format!("Available OAuth providers:\n\n{provider_list}\n")
}

pub fn parse_cli_provider_selection(
    input: &str,
    providers: &[OAuthProviderInfo],
) -> Result<String, String> {
    let index = input
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "Invalid selection".to_owned())?;
    providers
        .get(index)
        .map(|provider| provider.id.clone())
        .ok_or_else(|| "Invalid selection".to_owned())
}

pub fn save_cli_oauth_credentials(
    path: impl AsRef<Path>,
    provider_id: &str,
    credentials: StoredOAuthCredentials,
) -> Result<(), String> {
    let mut auth = load_auth_storage_from_path(path.as_ref())?;
    auth.insert(
        provider_id.to_owned(),
        AuthCredential::OAuth { credentials },
    );
    save_auth_storage_to_path(path, &auth)
}

pub fn load_cli_auth(path: impl AsRef<Path>) -> Result<AuthStorage, String> {
    load_auth_storage_from_path(path)
}

fn format_provider_line(provider: &OAuthProviderInfo) -> String {
    format!("  {:<20} {}", provider.id, provider.name)
}

pub async fn run_cli_with_env_args() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    run_cli(args).await
}

pub async fn run_cli(args: Vec<String>) -> Result<(), String> {
    let providers = get_oauth_providers();
    let command = args.first().map(String::as_str);
    match command {
        None | Some("help" | "--help" | "-h") => {
            print!("{}", render_cli_help("ri-ai", &providers));
            Ok(())
        }
        Some("list") => {
            print!("{}", render_cli_provider_list(&providers));
            Ok(())
        }
        Some("login") => {
            let provider_id = match args.get(1) {
                Some(provider_id) => provider_id.clone(),
                None => prompt_provider_selection(&providers)?,
            };
            if !providers.iter().any(|provider| provider.id == provider_id) {
                return Err(format!(
                    "Unknown provider: {provider_id}\nUse 'ri-ai list' to see available providers"
                ));
            }
            println!("Logging in to {provider_id}...");
            login_provider_to_cli_auth(&provider_id, CLI_AUTH_FILE).await?;
            println!("\nCredentials saved to {CLI_AUTH_FILE}");
            Ok(())
        }
        Some(other) => Err(format!(
            "Unknown command: {other}\nUse 'ri-ai --help' for usage"
        )),
    }
}

fn prompt_provider_selection(providers: &[OAuthProviderInfo]) -> Result<String, String> {
    println!("Select a provider:\n");
    for (index, provider) in providers.iter().enumerate() {
        println!("  {}. {}", index + 1, provider.name);
    }
    println!();
    let choice = prompt_line(&format!("Enter number (1-{}): ", providers.len()))?;
    parse_cli_provider_selection(&choice, providers)
}

async fn login_provider_to_cli_auth(
    provider_id: &str,
    auth_path: impl AsRef<Path>,
) -> Result<(), String> {
    let provider = get_oauth_provider(provider_id)
        .ok_or_else(|| format!("Unknown provider: {provider_id}"))?;
    let credentials = match provider.id.as_str() {
        "anthropic" => StoredOAuthCredentials::from(login_anthropic_cli().await?),
        "github-copilot" => StoredOAuthCredentials::from(login_github_copilot_cli().await?),
        "openai-codex" => StoredOAuthCredentials::from(login_openai_codex_cli().await?),
        other => {
            return Err(format!(
                "OAuth login is not implemented for provider: {other}"
            ));
        }
    };
    save_cli_oauth_credentials(auth_path, &provider.id, credentials)
}

async fn login_anthropic_cli() -> Result<OAuthCredentials, String> {
    let flow = start_anthropic_oauth_login_flow().await?;
    print_auth_info(&flow.auth_url, flow.instructions.as_deref());
    let callback_or_manual = wait_for_callback_or_manual_code(
        flow,
        "Paste the authorization code or full redirect URL, or press Enter to wait for browser callback: ",
        "OAuth state mismatch",
    )
    .await?;
    let now = now_millis() as i64;
    exchange_anthropic_authorization_code_with_url_at(
        &callback_or_manual.code,
        &callback_or_manual.state,
        &callback_or_manual.verifier,
        &callback_or_manual.redirect_uri,
        ANTHROPIC_OAUTH_TOKEN_URL,
        now,
    )
    .await
}

async fn login_openai_codex_cli() -> Result<OAuthCredentials, String> {
    let state = generate_cli_state()?;
    let flow = start_openai_codex_oauth_login_flow(&state, Some("pi")).await?;
    print_auth_info(&flow.auth_url, flow.instructions.as_deref());
    let callback_or_manual = wait_for_callback_or_manual_code(
        flow,
        "Paste the authorization code or full redirect URL, or press Enter to wait for browser callback: ",
        "State mismatch",
    )
    .await?;
    let now = now_millis() as i64;
    exchange_openai_codex_authorization_code_with_url_at(
        &callback_or_manual.code,
        &callback_or_manual.verifier,
        Some(&callback_or_manual.redirect_uri),
        OPENAI_CODEX_OAUTH_TOKEN_URL,
        now,
    )
    .await
}

async fn login_github_copilot_cli() -> Result<GitHubCopilotCredentials, String> {
    let enterprise_input = prompt_line("GitHub Enterprise URL/domain (blank for github.com): ")?;
    let enterprise_domain = if enterprise_input.trim().is_empty() {
        None
    } else {
        Some(
            normalize_github_domain(&enterprise_input)
                .ok_or_else(|| "Invalid GitHub Enterprise URL/domain".to_owned())?,
        )
    };
    let domain = enterprise_domain.as_deref().unwrap_or("github.com");
    let urls = github_copilot_urls(domain);
    let result = login_github_copilot_device_flow_for_urls(
        &urls,
        enterprise_domain.as_deref(),
        print_github_device_code,
    )
    .await?;
    Ok(result.credentials)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CallbackOrManualCode {
    code: String,
    state: String,
    verifier: String,
    redirect_uri: String,
}

async fn wait_for_callback_or_manual_code(
    flow: OAuthLoginFlow,
    prompt: &'static str,
    state_mismatch_message: &'static str,
) -> Result<CallbackOrManualCode, String> {
    let OAuthLoginFlow {
        verifier,
        state,
        redirect_uri,
        callback_server,
        ..
    } = flow;
    let manual = spawn_prompt_reader(prompt);
    let callback = callback_server.wait_for_code();
    tokio::pin!(callback);

    let (code, state) = tokio::select! {
        callback = &mut callback => callback_to_code(callback?, &state)?,
        manual = recv_manual_code(manual) => {
            let manual = manual?;
            if manual.trim().is_empty() {
                callback_to_code(callback.await?, &state)?
            } else {
                manual_to_code(&manual, &state, state_mismatch_message)?
            }
        }
    };

    Ok(CallbackOrManualCode {
        code,
        state,
        verifier,
        redirect_uri,
    })
}

fn callback_to_code(
    callback: Option<OAuthCallback>,
    fallback_state: &str,
) -> Result<(String, String), String> {
    let callback = callback.ok_or_else(|| "Missing authorization code".to_owned())?;
    Ok((callback.code, callback.state.if_empty(fallback_state)))
}

fn manual_to_code(
    input: &str,
    expected_state: &str,
    state_mismatch_message: &str,
) -> Result<(String, String), String> {
    let parsed = parse_authorization_input(input);
    if let Some(parsed_state) = parsed.state.as_deref()
        && parsed_state != expected_state
    {
        return Err(state_mismatch_message.to_owned());
    }
    let code = parsed
        .code
        .ok_or_else(|| "Missing authorization code".to_owned())?;
    Ok((
        code,
        parsed.state.unwrap_or_else(|| expected_state.to_owned()),
    ))
}

fn spawn_prompt_reader(prompt: &'static str) -> mpsc::Receiver<Result<String, String>> {
    let (sender, receiver) = mpsc::channel();
    std::thread::spawn(move || {
        let result = prompt_line(prompt);
        let _ = sender.send(result);
    });
    receiver
}

async fn recv_manual_code(
    receiver: mpsc::Receiver<Result<String, String>>,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        receiver
            .recv()
            .unwrap_or_else(|_| Err("Manual input reader stopped".to_owned()))
    })
    .await
    .map_err(|error| format!("Manual input reader failed: {error}"))?
}

fn print_auth_info(url: &str, instructions: Option<&str>) {
    println!("\nOpen this URL in your browser:\n{url}");
    if let Some(instructions) = instructions {
        println!("{instructions}");
    }
    println!();
}

fn print_github_device_code(device: &GitHubDeviceCode) {
    let url = device
        .verification_uri_complete
        .as_deref()
        .unwrap_or(&device.verification_uri);
    println!("\nOpen this URL in your browser:\n{url}");
    println!("Enter code: {}", device.user_code);
    println!();
}

fn prompt_line(prompt: &str) -> Result<String, String> {
    print!("{prompt}");
    io::stdout()
        .flush()
        .map_err(|error| format!("Failed to flush stdout: {error}"))?;
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .map_err(|error| format!("Failed to read input: {error}"))?;
    Ok(line.trim_end_matches(['\r', '\n']).to_owned())
}

fn generate_cli_state() -> Result<String, String> {
    let pkce = crate::anthropic_oauth::generate_pkce()?;
    Ok(pkce.verifier.chars().take(32).collect())
}

trait EmptyStringFallback {
    fn if_empty(self, fallback: &str) -> String;
}

impl EmptyStringFallback for String {
    fn if_empty(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_owned()
        } else {
            self
        }
    }
}
