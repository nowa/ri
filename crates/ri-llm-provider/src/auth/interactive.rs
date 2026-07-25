//! Interactive OAuth login flows for the built-in providers, driven through
//! [`AuthInteraction`] (pi `auth/oauth/{anthropic,github-copilot,openai-codex}.ts`).
//!
//! The flows reuse the standalone OAuth primitives (authorize URLs, callback
//! servers, device-code polling, token exchange) and add only the interaction
//! protocol: `notify` for auth URLs / device codes / progress, `prompt` for
//! manual codes and choices. Browser flows race the local callback server
//! against a manual paste prompt, exactly like the `ri-ai` CLI.

use super::types::{AuthEvent, AuthInteraction, AuthPrompt, AuthPromptOption, OAuthCredential};
use crate::{
    anthropic_oauth::{
        ANTHROPIC_OAUTH_TOKEN_URL, OAuthCallback, OAuthCredentials, OAuthLoginFlow,
        exchange_anthropic_authorization_code_with_url_at, parse_authorization_input,
        start_anthropic_oauth_login_flow,
    },
    github_copilot_oauth::{
        GitHubCopilotCredentials, GitHubCopilotModelPolicyOptions, GitHubCopilotUrls,
        complete_github_copilot_device_flow_for_urls,
        enable_all_github_copilot_model_policies_with_options,
        fetch_available_github_copilot_model_ids_with_base_url, github_copilot_urls,
        normalize_github_domain, refresh_github_copilot_token_for_urls_at,
        request_github_copilot_device_code_for_urls,
    },
    oauth_auth_storage::StoredOAuthCredentials,
    openai_codex_oauth::{
        OPENAI_CODEX_DEVICE_CODE_TIMEOUT_SECONDS, OPENAI_CODEX_DEVICE_VERIFICATION_URI,
        OPENAI_CODEX_OAUTH_TOKEN_URL, exchange_openai_codex_authorization_code_with_url_at,
        generate_openai_codex_oauth_state, login_openai_codex_device_code,
        start_openai_codex_oauth_login_flow,
    },
    types::now_millis,
};

const OPENAI_CODEX_BROWSER_LOGIN_METHOD: &str = "browser";
const OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD: &str = "device_code";
const MANUAL_CODE_PROMPT: &str =
    "Complete login in your browser, or paste the authorization code / redirect URL here:";

/// Interactive OAuth login for one of the built-in OAuth providers.
pub async fn login_builtin_oauth_provider(
    provider_id: &str,
    interaction: &dyn AuthInteraction,
) -> Result<OAuthCredential, String> {
    match provider_id {
        "anthropic" => login_anthropic_interactive(interaction).await,
        "github-copilot" => login_github_copilot_interactive(interaction).await,
        "openai-codex" => login_openai_codex_interactive(interaction).await,
        other => Err(format!(
            "OAuth login is not implemented for provider: {other}"
        )),
    }
}

async fn login_anthropic_interactive(
    interaction: &dyn AuthInteraction,
) -> Result<OAuthCredential, String> {
    let flow = start_anthropic_oauth_login_flow().await?;
    let authorization =
        wait_for_callback_or_manual_code(flow, interaction, "OAuth state mismatch").await?;
    interaction.notify(AuthEvent::Progress {
        message: "Exchanging authorization code for tokens...".to_owned(),
    });
    exchange_anthropic_authorization_code_with_url_at(
        &authorization.code,
        &authorization.state,
        &authorization.verifier,
        &authorization.redirect_uri,
        ANTHROPIC_OAUTH_TOKEN_URL,
        now_millis() as i64,
    )
    .await
    .map(oauth_credentials_to_credential)
}

async fn login_github_copilot_interactive(
    interaction: &dyn AuthInteraction,
) -> Result<OAuthCredential, String> {
    let input = interaction
        .prompt(AuthPrompt::Text {
            message: "GitHub Enterprise URL/domain (blank for github.com)".to_owned(),
            placeholder: Some("company.ghe.com".to_owned()),
        })
        .await?;
    let trimmed = input.trim();
    let enterprise_domain = if trimmed.is_empty() {
        None
    } else {
        Some(
            normalize_github_domain(trimmed)
                .ok_or_else(|| "Invalid GitHub Enterprise URL/domain".to_owned())?,
        )
    };
    let domain = enterprise_domain.as_deref().unwrap_or("github.com");
    let urls = github_copilot_urls(domain);
    login_github_copilot_with_urls(
        &urls,
        enterprise_domain.as_deref(),
        interaction,
        &GitHubCopilotModelPolicyOptions::default(),
    )
    .await
}

/// Copilot device-flow login against explicit endpoints. Test-injectable.
pub async fn login_github_copilot_with_urls(
    urls: &GitHubCopilotUrls,
    enterprise_domain: Option<&str>,
    interaction: &dyn AuthInteraction,
    policy_options: &GitHubCopilotModelPolicyOptions,
) -> Result<OAuthCredential, String> {
    let device = request_github_copilot_device_code_for_urls(urls).await?;
    interaction.notify(AuthEvent::DeviceCode {
        user_code: device.user_code.clone(),
        verification_uri: device.verification_uri.clone(),
        interval_seconds: Some(device.interval_seconds),
        expires_in_seconds: Some(device.expires_in_seconds),
    });
    let refresh_token = complete_github_copilot_device_flow_for_urls(urls, &device).await?;
    let credentials = refresh_github_copilot_token_for_urls_at(
        &refresh_token,
        urls,
        enterprise_domain,
        now_millis() as i64,
    )
    .await?;
    interaction.notify(AuthEvent::Progress {
        message: "Enabling models...".to_owned(),
    });
    let _ = enable_all_github_copilot_model_policies_with_options(
        &credentials.access,
        enterprise_domain,
        policy_options,
    )
    .await;
    // pi stores the entitlement listing on the credential after enabling
    // policies, and a listing failure fails the login.
    let mut credentials = credentials;
    credentials.available_model_ids = Some(
        fetch_available_github_copilot_model_ids_with_base_url(
            &credentials.access,
            enterprise_domain,
            policy_options.base_url.as_deref(),
        )
        .await?,
    );
    Ok(copilot_credentials_to_credential(credentials))
}

async fn login_openai_codex_interactive(
    interaction: &dyn AuthInteraction,
) -> Result<OAuthCredential, String> {
    let method = interaction
        .prompt(AuthPrompt::Select {
            message: "Select OpenAI Codex login method:".to_owned(),
            options: vec![
                AuthPromptOption {
                    id: OPENAI_CODEX_BROWSER_LOGIN_METHOD.to_owned(),
                    label: "Browser login (default)".to_owned(),
                    description: None,
                },
                AuthPromptOption {
                    id: OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD.to_owned(),
                    label: "Device code login (headless)".to_owned(),
                    description: None,
                },
            ],
        })
        .await?;

    if method == OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD {
        return login_openai_codex_device_code(|device| {
            interaction.notify(AuthEvent::DeviceCode {
                user_code: device.user_code.clone(),
                verification_uri: OPENAI_CODEX_DEVICE_VERIFICATION_URI.to_owned(),
                interval_seconds: Some(device.interval_seconds),
                expires_in_seconds: Some(OPENAI_CODEX_DEVICE_CODE_TIMEOUT_SECONDS),
            });
        })
        .await
        .map(oauth_credentials_to_credential);
    }
    if method != OPENAI_CODEX_BROWSER_LOGIN_METHOD {
        return Err(format!("Unknown OpenAI Codex login method: {method}"));
    }

    let state = generate_openai_codex_oauth_state()?;
    let flow = start_openai_codex_oauth_login_flow(&state, Some("pi")).await?;
    let authorization =
        wait_for_callback_or_manual_code(flow, interaction, "State mismatch").await?;
    interaction.notify(AuthEvent::Progress {
        message: "Exchanging authorization code for tokens...".to_owned(),
    });
    exchange_openai_codex_authorization_code_with_url_at(
        &authorization.code,
        &authorization.verifier,
        Some(&authorization.redirect_uri),
        OPENAI_CODEX_OAUTH_TOKEN_URL,
        now_millis() as i64,
    )
    .await
    .map(oauth_credentials_to_credential)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InteractiveAuthorization {
    pub code: String,
    pub state: String,
    pub verifier: String,
    pub redirect_uri: String,
}

/// Race the local callback server against a manual paste prompt. An empty
/// manual input falls back to waiting for the browser callback.
pub async fn wait_for_callback_or_manual_code(
    flow: OAuthLoginFlow,
    interaction: &dyn AuthInteraction,
    state_mismatch_message: &str,
) -> Result<InteractiveAuthorization, String> {
    let OAuthLoginFlow {
        auth_url,
        instructions,
        verifier,
        state,
        redirect_uri,
        callback_server,
        ..
    } = flow;
    interaction.notify(AuthEvent::AuthUrl {
        url: auth_url,
        instructions,
    });

    let manual = interaction.prompt(AuthPrompt::ManualCode {
        message: MANUAL_CODE_PROMPT.to_owned(),
        placeholder: Some(redirect_uri.clone()),
    });
    tokio::pin!(manual);
    let callback = callback_server.wait_for_code();
    tokio::pin!(callback);

    let (code, resolved_state) = tokio::select! {
        callback = &mut callback => callback_to_code(callback?, &state)?,
        manual = &mut manual => {
            let manual = manual?;
            if manual.trim().is_empty() {
                callback_to_code(callback.await?, &state)?
            } else {
                manual_to_code(&manual, &state, state_mismatch_message)?
            }
        }
    };

    Ok(InteractiveAuthorization {
        code,
        state: resolved_state,
        verifier,
        redirect_uri,
    })
}

fn callback_to_code(
    callback: Option<OAuthCallback>,
    fallback_state: &str,
) -> Result<(String, String), String> {
    let callback = callback.ok_or_else(|| "Missing authorization code".to_owned())?;
    let state = if callback.state.is_empty() {
        fallback_state.to_owned()
    } else {
        callback.state
    };
    Ok((callback.code, state))
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

fn oauth_credentials_to_credential(credentials: OAuthCredentials) -> OAuthCredential {
    OAuthCredential {
        refresh: credentials.refresh,
        access: credentials.access,
        expires: credentials.expires,
        extra: credentials.extra.into_iter().collect(),
    }
}

fn copilot_credentials_to_credential(credentials: GitHubCopilotCredentials) -> OAuthCredential {
    let stored = StoredOAuthCredentials::from(credentials);
    OAuthCredential {
        refresh: stored.refresh,
        access: stored.access,
        expires: stored.expires,
        extra: stored.extra.into_iter().collect(),
    }
}
