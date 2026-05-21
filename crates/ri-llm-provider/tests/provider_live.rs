use futures::StreamExt;
use ri_llm_provider::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    future::Future,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const LIVE_GATE_ENV: &str = "RI_LIVE_PROVIDER_TESTS";
const LIVE_STRICT_ENV: &str = "RI_LIVE_PROVIDER_STRICT";
const LIVE_OAUTH_INTERACTIVE_ENV: &str = "RI_LIVE_OAUTH_INTERACTIVE_TESTS";
const LIVE_BEDROCK_EXTENSIVE_ENV: &str = "BEDROCK_EXTENSIVE_MODEL_TEST";
const LIVE_OAUTH_INTERACTIVE_TIMEOUT_SECS: u64 = 15 * 60;
const LIVE_PROMPT: &str =
    "Reply with a short plain-text confirmation containing the token ri-live-ok.";
const LIVE_RESPONSE_ID_PROMPT: &str = "Reply with exactly: response id test";
const LIVE_ABORT_PROMPT: &str = "Write a long poem with 20 stanzas about the beauty of nature.";
const LIVE_ABORT_MIN_DELTA_CHARS: usize = 1000;
const LIVE_MULTITURN_FIRST: &str = "first turn anchor";
const LIVE_MULTITURN_SECOND: &str = "second turn ok";
const LIVE_BOGUS_API_KEY: &str = "ri-live-invalid-api-key";
const LIVE_RED_PIXEL_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z8BQDwAFgwJ/l6p9qAAAAABJRU5ErkJggg==";
const LIVE_RED_CIRCLE_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAABmJLR0QA/wD/AP+gvaeTAAAJuklEQVR4nO3df2hV9R/H8deNLSl1a6Zs17S1pmsTolaUJdoPHJaU/RQ00DQK/4hIooIiI6J/FPpF/SdJ5ciMMDD7gThRyyChH5DR5kQXarU5N62c2lzs+8eHfZGwN3Ode9/3fs7zAUP/iPm+Oz33Ofecc8/JDA4ODgrAWZ3nPQBQyAgEMBAIYCAQwEAggIFAAAOBAAYCAQwEAhgIBDAQCGAgEMBAIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGAgEMBAIYCAQwEAggIFAAAOBAAYCAQwEAhgIBDAQCGAgEMBAIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGEq8B0iNnh5pzx6prU1qbw9/7+yU+vrC19Gj4U9JGj1aqqgIf44eLWWzUl1d+Kqvl664Qrr4Yt/XkxKZwcHBQe8hotTZKX35pdTSIm3ZInV0JPv9s1lp5kypqUm67TapujrZ7w9JBJKs1lZp7VppwwZp7978/tt1ddL8+dLixWGVQSII5L86ckRat05qbpa++cZ7muD666VFi6QHHpDGj/eepqgRyEgdOCC98or01lvSiRPe05zdqFHSkiXSihXS5Mne0xQlAjlXHR3S669Lq1dLp055TzM8paXSwoXSc8+FN/gYNgIZrr4+6aWXwqoxMOA9zciUlEiPPhpeR1mZ9zRFgUCGY9Om8D/WoUPekyQjm5VWrgxv6DMZ72kKGoFYOjqkhx+Wtm3zniQ3Zs+W1qzhELGBM+n/ZuNG6dpr441DkrZula66SvrwQ+9JChaB/NPp09Izz0j33hvObsfu99+lBQuk5cul/n7vaQoOu1hn+u036a67Cud8Rr5Nnx5WzspK70kKBoEM6egIl2zk+wx4oampkTZvlqZO9Z6kILCLJUm7d4frmtIehxR+UcyaJX3/vfckBYFAduwIcfz6q/ckhaOrS7rllnCxZcqlexfrhx+km2+Wjh3znqQwlZWFo3jXXOM9iZv0BrJvX1g5Oju9JylsEyZIO3eGq4VTKJ27WIcPS3PnEsdwdHen+meVvkD++ku64w7ekJ+L/fule+5J5XmS9AXy9NPpPc/xX+zaJT37rPcUeZeu9yCffBJOBKboJScqk5E++iisJimRnkAOHJAaG6XeXu9JiltFhfTtt+GEYgqkYxdrcFBaupQ4knD0qLRsmfcUeZOOQNati/uq3HxraZE++MB7iryIfxfrjz+khgbOlCetqirc46u83HuSnIp/BVmxgjhyobNTevFF7ylyLu4VZO/esHr8/bf3JHEqKQmrSG2t9yQ5E/cKsnIlceTSwIC0apX3FDkV7wpy8KA0ZUoqz/7mVWlpWKkj/Vx7vCvIqlXEkQ+nT0uvvuo9Rc7EuYJ0d4ffaCdPek+SDhdcEFbsCO84H+cK8v77xJFPJ09K69d7T5ETcQbS3Ow9QfpE+jOPbxertVWaNs17inRqa4vu3r/xrSBr13pPkF7vvec9QeLiW0Hq6vgwlJeGBumnn7ynSFRcgfzyizRpkvcU6XbokHTJJd5TJCauXaytW70nwPbt3hMkKq5AuKTdX2TbIK5drOrq8MlB+KmpCTd5iEQ8gRw5Eu7hBH89PdK4cd5TJCKeXaw9e7wnwJD2du8JEkMgSF5E24JAkLyItkU8gUS0rBc9AilAKb13bEHq6vKeIDHxBHL8uPcEGPLnn94TJCaeQCLaKEUvom1BIEheRNsinkDYxSocBAKkQzyBjBnjPQGGjB3rPUFi4gkkoo1S9CLaFgSC5EW0LeIJhF2swkEgBaiqynsCDKms9J4gMfEEEtntZopafb33BIkhECQvom1BIEheRNsino/c9vZGefPkosRHbgvQuHHRPqOiqNTURBOHFFMgknTrrd4TYPZs7wkSRSBIVmTbIJ73IFJ4mm1Et70sOplMuP1rNus9SWLiWkEmTozqCErRaWiIKg4ptkAk6b77vCdIr/nzvSdIXFy7WFK4uwmriA8eoFME6uqk667zniJ9brwxujikGAORpMWLvSdIn0h/5vHtYknhRtaXXsqTbvPlwgvDXfUjvJIhzhVk/HjpkUe8p0iPZcuijEOKdQWRwqPAamul/n7vSeI2apS0b1+055/iXEGk8KzCBx/0niJ+Dz0UbRxSzCuIFH6z1ddLAwPek8SptDTcqLqmxnuSnIl3BZHCLtZjj3lPEa/ly6OOQ4p9BZHCXf4aGsI1QkhONhtODJaVeU+SU3GvIFK4w8bLL3tPEZ833og+DikNK8iQpiaeo56UOXOkzZu9p8iL9ARy8KDU2Bg+DoqRq6iQvvtOuuwy70nyIv5drCGTJ0vvvBM+s4CRyWSkt99OTRxSmgKRpDvvDEdeMDJPPSXdfbf3FHmVnl2sIf390k03Sbt2eU9SXGbMkLZvD+c+UiR9gUhSd7c0a1ZUT2PNqdpa6auvorql6HClaxdryIQJ0mefRffx0JzIZqUtW1IZh5TWQCTp8svDocqLLvKepHCVlUmffhr92XJLegORpCuvlDZulMrLvScpPOXlIY7GRu9JXKXzPcg//fijdPvtXI4ypKpK+vxz6eqrvSdxRyBDOjpCJO3t3pP4Gtr1nDLFe5KCkO5drDPV1EhffCFNn+49iZ8ZM6SvvyaOMxDImSorpZ07pRdekM5L0Y8mk5Eef1zati0c4cP/sYv1bzZtkpYuDY9ViFl5ubRmjXT//d6TFKQU/Zo8R/PmhYvympq8J8mdOXOk3buJw0AglurqcJLs44/DxY6xmDhRevfd8GY8pteVAwQyHPPmhUPBTzwhlZR4TzNypaXSk0+GS2y4ocWw8B7kXP38s/Taa9Lq1dKpU97TDM/550sLFkjPPy9Nneo9TVEhkJHq6gqhvPmmdOKE9zRnN2qUtGRJCGPSJO9pihKB/Fc9PdL69VJzc+FcQn/DDeFeuQsXRvW8QA8EkqT29hDKhg1Sa2t+/+1p08LzORYtYjcqQQSSK4cPSzt2SC0t4Wv//mS/fzYrzZwZDkPPncvRqBwhkHzp7Q0rTFtbOIrU3i51dkp9feHeXceOScePh/92zJhwGf7YseHvlZXh2RtnflVU+L6elCAQwMB5EMBAIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGAgEMBAIYCAQwEAggIFAAAOBAAYCAQwEAhgIBDAQCGAgEMBAIICBQAADgQAGAgEMBAIYCAQwEAhgIBDAQCCAgUAAA4EABgIBDAQCGAgEMBAIYCAQwEAggIFAAAOBAAYCAQwEAhgIBDAQCGAgEMDwP4yLqLwXdlfVAAAAAElFTkSuQmCC";
const LIVE_RESPONSES_TOOL_IMAGE_TEXT: &str = "A red circle with a diameter of 100 pixels.";
const LIVE_LONG_SYSTEM_PROMPT_SEGMENT: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.";
const LIVE_CONTEXT_OVERFLOW_LOREM: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. ";
const LIVE_LONG_PIPE_TOOL_CALL_ID: &str = "call_pAYbIr76hXIjncD9UE4eGfnS|t5nnb2qYMFWGSsr13fhCd1CaCu3t3qONEPuOudu4HSVEtA8YJSL6FAZUxvoOoD792VIJWl91g87EdqsCWp9krVsdBysQoDaf9lMCLb8BS4EYi4gQd5kBQBYLlgD71PYwvf+TbMD9J9/5OMD42oxSRj8H+vRf78/l2Xla33LWz4nOgsddBlbvabICRs8GHt5C9PK5keFtzyi3lsyVKNlfduK3iphsZqs4MLv4zyGJnvZo/+QzShyk5xnMSQX/f98+aEoNflEApCdEOXipipgeiNWnpFSHbcwmMkZoJhURNu+JEz3xCh1mrXeYoN5o+trLL3IXJacSsLYXDrYTipZZbJFRPAucgbnjYBC+/ZzJOfkwCs+Gkw7EoZR7ZQgJ8ma+9586n4tT4cI8DEhBSZsWMjrCt8dxKg==";

static LIVE_TEST_ENV_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Copy)]
enum LiveAbortUsageExpectation {
    ZeroInputOutput,
    PositiveInputOutput,
    PositiveInputZeroOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveEmptyCase {
    EmptyContentArray,
    EmptyString,
    WhitespaceOnly,
    EmptyAssistant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveUnicodeToolResultCase {
    Emoji,
    LinkedIn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveImageToolResultCase {
    ImageOnly,
    TextAndImage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiveContextOverflowExpectation {
    Error,
    LengthZeroOutput,
    ZaiInconsistent,
    SilentTruncationOrError,
}

#[derive(Debug)]
struct CaptureProviderPayloadHook {
    captured: Arc<Mutex<Option<Value>>>,
}

impl ProviderPayloadHook for CaptureProviderPayloadHook {
    fn on_payload(&self, _model: &Model, payload: Value) -> Result<Value, String> {
        let mut captured = self
            .captured
            .lock()
            .map_err(|_| "payload capture lock poisoned".to_owned())?;
        *captured = Some(payload.clone());
        Ok(payload)
    }
}

#[derive(Debug)]
struct OpenRouterCacheControlPayloadHook;

impl ProviderPayloadHook for OpenRouterCacheControlPayloadHook {
    fn on_payload(&self, _model: &Model, mut payload: Value) -> Result<Value, String> {
        let Some(messages) = payload.get_mut("messages").and_then(Value::as_array_mut) else {
            return Ok(payload);
        };
        for message in messages.iter_mut().rev() {
            if message.get("role").and_then(Value::as_str) != Some("user") {
                continue;
            }
            if let Some(text) = message
                .get("content")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
            {
                message["content"] = json!([{
                    "type": "text",
                    "text": text,
                    "cache_control": { "type": "ephemeral" },
                }]);
                break;
            }
            let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
                break;
            };
            for part in content.iter_mut().rev() {
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    part["cache_control"] = json!({ "type": "ephemeral" });
                    break;
                }
            }
            break;
        }
        Ok(payload)
    }
}

#[derive(Debug)]
struct LiveEnvVarGuard {
    name: &'static str,
    previous: Option<std::ffi::OsString>,
}

impl LiveEnvVarGuard {
    fn set(name: &'static str, value: &std::ffi::OsStr) -> Self {
        let previous = env::var_os(name);
        unsafe {
            env::set_var(name, value);
        }
        Self { name, previous }
    }

    fn remove(name: &'static str) -> Self {
        let previous = env::var_os(name);
        unsafe {
            env::remove_var(name);
        }
        Self { name, previous }
    }
}

impl Drop for LiveEnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(value) => env::set_var(self.name, value),
                None => env::remove_var(self.name),
            }
        }
    }
}

#[derive(Debug)]
struct LiveTempDir {
    path: std::path::PathBuf,
}

impl LiveTempDir {
    fn new(label: &str) -> Self {
        let mut path = env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after Unix epoch")
            .as_nanos();
        path.push(format!(
            "ri-provider-live-{label}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create live provider temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for LiveTempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[derive(Debug, Clone)]
struct LiveAnthropicMessagesE2ECase {
    name: String,
    provider: String,
    model: Model,
}

#[derive(Debug, Clone, Copy)]
struct LiveCrossProviderHandoffPair {
    provider: &'static str,
    model_id: &'static str,
    label: &'static str,
    api_override: Option<&'static str>,
    upstream_api_key_env: Option<&'static str>,
}

#[derive(Debug, Clone)]
struct LiveCrossProviderHandoffFixture {
    label: &'static str,
    model: Model,
    options: SimpleStreamOptions,
    messages: Vec<Message>,
}

fn live_tests_enabled() -> bool {
    live_gate_value_enabled(env::var(LIVE_GATE_ENV).ok().as_deref())
}

fn live_gate_value_enabled(value: Option<&str>) -> bool {
    value
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct LiveExternalRequirement {
    category: &'static str,
    name: &'static str,
    detail: &'static str,
}

fn live_external_requirements() -> &'static [LiveExternalRequirement] {
    &[
        LiveExternalRequirement {
            category: "gate",
            name: LIVE_GATE_ENV,
            detail: "set to a truthy value to enable live provider tests",
        },
        LiveExternalRequirement {
            category: "gate",
            name: LIVE_STRICT_ENV,
            detail: "set to a truthy value to turn credential/service skips into failures",
        },
        LiveExternalRequirement {
            category: "gate",
            name: LIVE_OAUTH_INTERACTIVE_ENV,
            detail: "set with RI_LIVE_PROVIDER_TESTS=1 to run manual browser/device OAuth login-to-auth.json live tests",
        },
        LiveExternalRequirement {
            category: "gate",
            name: LIVE_BEDROCK_EXTENSIVE_ENV,
            detail: "set with RI_LIVE_PROVIDER_TESTS=1 and AWS Bedrock credentials to run per-model Bedrock live requests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "OPENAI_API_KEY",
            detail: "OpenAI Responses, Completions, Cloudflare BYOK, and related live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "ANTHROPIC_API_KEY",
            detail: "Anthropic Messages, Claude, Cloudflare BYOK, and related live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "OPENROUTER_API_KEY",
            detail: "OpenRouter text, cache-write, image, and related live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "AZURE_OPENAI_API_KEY",
            detail: "Azure OpenAI Responses live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "GOOGLE_CLOUD_API_KEY",
            detail: "Google/Gemini and Vertex API-key live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "GEMINI_API_KEY",
            detail: "Google Generative AI API-key live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "CLOUDFLARE_API_KEY",
            detail: "Cloudflare Workers AI and AI Gateway live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "TOGETHER_API_KEY",
            detail: "Together live tests and OpenAI-compatible provider matrix coverage",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "XAI_API_KEY",
            detail: "xAI OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "GROQ_API_KEY",
            detail: "Groq OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "CEREBRAS_API_KEY",
            detail: "Cerebras OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "DEEPSEEK_API_KEY",
            detail: "DeepSeek OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "HF_TOKEN",
            detail: "Hugging Face OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "FIREWORKS_API_KEY",
            detail: "Fireworks Anthropic-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "ZAI_API_KEY",
            detail: "Zai OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "MISTRAL_API_KEY",
            detail: "Mistral Conversations and OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "MINIMAX_API_KEY",
            detail: "MiniMax OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "MINIMAX_CN_API_KEY",
            detail: "MiniMax CN Anthropic-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "XIAOMI_API_KEY",
            detail: "Xiaomi OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "XIAOMI_TOKEN_PLAN_CN_API_KEY",
            detail: "Xiaomi token-plan CN live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
            detail: "Xiaomi token-plan AMS live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
            detail: "Xiaomi token-plan SGP live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "KIMI_API_KEY",
            detail: "Kimi Coding OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "AI_GATEWAY_API_KEY",
            detail: "Vercel AI Gateway OpenAI-compatible live tests",
        },
        LiveExternalRequirement {
            category: "api_key",
            name: "OPENCODE_API_KEY",
            detail: "OpenCode Zen and OpenCode Go live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AZURE_OPENAI_BASE_URL",
            detail: "Azure OpenAI endpoint resolution for live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AZURE_OPENAI_RESOURCE_NAME",
            detail: "Azure OpenAI resource-name endpoint resolution for live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AZURE_OPENAI_DEPLOYMENT_NAME",
            detail: "Azure OpenAI deployment name for live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "GOOGLE_CLOUD_PROJECT",
            detail: "Google Vertex project id for ADC/API-key live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "GCLOUD_PROJECT",
            detail: "Google Vertex fallback project id for ADC/API-key live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "GOOGLE_CLOUD_LOCATION",
            detail: "Google Vertex location for ADC/API-key live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "GOOGLE_APPLICATION_CREDENTIALS",
            detail: "Google Vertex ADC/token live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "GOOGLE_OAUTH_ACCESS_TOKEN",
            detail: "Google Vertex OAuth access token live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "GOOGLE_ACCESS_TOKEN",
            detail: "Google Vertex access token live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "CLOUDSDK_AUTH_ACCESS_TOKEN",
            detail: "Google Vertex Cloud SDK access token live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_REGION",
            detail: "Bedrock Converse live test region",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_DEFAULT_REGION",
            detail: "Bedrock Converse fallback live test region",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_PROFILE",
            detail: "Bedrock Converse shared-profile live test credentials",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_ACCESS_KEY_ID",
            detail: "Bedrock Converse access-key live test credentials",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_SECRET_ACCESS_KEY",
            detail: "Bedrock Converse secret-key live test credentials",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_BEARER_TOKEN_BEDROCK",
            detail: "Bedrock Converse bearer-token live test credentials",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_WEB_IDENTITY_TOKEN_FILE",
            detail: "Bedrock Converse web-identity live test token file",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_ROLE_ARN",
            detail: "Bedrock Converse web-identity live test role ARN",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_ROLE_SESSION_NAME",
            detail: "optional Bedrock Converse web-identity role session name",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            detail: "Bedrock Converse ECS task-role live test credentials",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            detail: "Bedrock Converse ECS task-role live test credentials",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_CONTAINER_AUTHORIZATION_TOKEN",
            detail: "optional Bedrock ECS task-role metadata authorization token",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
            detail: "optional Bedrock ECS task-role metadata authorization token file",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "CLOUDFLARE_ACCOUNT_ID",
            detail: "Cloudflare Workers AI and AI Gateway account live tests",
        },
        LiveExternalRequirement {
            category: "provider_env",
            name: "CLOUDFLARE_GATEWAY_ID",
            detail: "Cloudflare AI Gateway route live tests",
        },
        LiveExternalRequirement {
            category: "oauth_auth_storage",
            name: "anthropic",
            detail: "~/.pi/agent/auth.json OAuth credential for Anthropic OAuth live tests",
        },
        LiveExternalRequirement {
            category: "oauth_auth_storage",
            name: "github-copilot",
            detail: "~/.pi/agent/auth.json OAuth credential for GitHub Copilot live tests",
        },
        LiveExternalRequirement {
            category: "oauth_auth_storage",
            name: "openai-codex",
            detail: "~/.pi/agent/auth.json OAuth credential for OpenAI Codex live tests",
        },
        LiveExternalRequirement {
            category: "runtime_env",
            name: "HOME",
            detail: "auth storage, Google ADC, and Bedrock shared config lookup",
        },
        LiveExternalRequirement {
            category: "runtime_env",
            name: "APPDATA",
            detail: "Windows Google ADC lookup",
        },
        LiveExternalRequirement {
            category: "local_service",
            name: "Ollama gpt-oss:20b on localhost:11434",
            detail: "local OpenAI-compatible stream/context-overflow live tests",
        },
        LiveExternalRequirement {
            category: "local_service",
            name: "LM Studio OpenAI-compatible server on localhost:1234",
            detail: "local OpenAI-compatible context-overflow live tests",
        },
        LiveExternalRequirement {
            category: "local_service",
            name: "llama.cpp OpenAI-compatible server on localhost:8081",
            detail: "local OpenAI-compatible context-overflow live tests",
        },
        LiveExternalRequirement {
            category: "skip_control",
            name: "PI_NO_LOCAL_LLM",
            detail: "must be unset for local Ollama live tests in strict mode",
        },
    ]
}

fn has_live_external_requirement(category: &str, name: &str) -> bool {
    live_external_requirements()
        .iter()
        .any(|requirement| requirement.category == category && requirement.name == name)
}

fn live_strict_readiness_provider_env_names() -> BTreeSet<&'static str> {
    BTreeSet::from([
        "AZURE_OPENAI_BASE_URL",
        "AZURE_OPENAI_RESOURCE_NAME",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "GOOGLE_ACCESS_TOKEN",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "AWS_ROLE_ARN",
        "AWS_ROLE_SESSION_NAME",
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN",
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
        "CLOUDFLARE_ACCOUNT_ID",
        "CLOUDFLARE_GATEWAY_ID",
    ])
}

fn push_missing_env(missing: &mut Vec<String>, name: &str, detail: &str) {
    if live_env(name).is_none() {
        missing.push(format!("{name} is not set ({detail})"));
    }
}

fn push_missing_truthy_gate(missing: &mut Vec<String>, name: &str, detail: &str) {
    if !live_gate_value_enabled(env::var(name).ok().as_deref()) {
        missing.push(format!("{name}=1 is not set ({detail})"));
    }
}

async fn live_strict_readiness_add_oauth_storage_missing(missing: &mut Vec<String>) {
    let Some(path) = default_auth_storage_path() else {
        missing.push("HOME is not set; cannot read ~/.pi/agent/auth.json".to_owned());
        return;
    };
    if !path.exists() {
        missing.push("~/.pi/agent/auth.json is missing (OAuth credential storage)".to_owned());
        return;
    }

    let storage = match load_auth_storage_from_path(&path) {
        Ok(storage) => storage,
        Err(error) => {
            missing.push(format!("failed to read ~/.pi/agent/auth.json: {error}"));
            return;
        }
    };

    for requirement in live_external_requirements()
        .iter()
        .filter(|requirement| requirement.category == "oauth_auth_storage")
    {
        match storage.get(requirement.name) {
            Some(AuthCredential::OAuth { credentials })
                if !credentials.access.trim().is_empty()
                    && !credentials.refresh.trim().is_empty()
                    && credentials.expires > 0 =>
            {
                match resolve_auth_storage_api_key(requirement.name).await {
                    Ok(Some(resolution))
                        if resolution.credentials.is_some()
                            && !resolution.api_key.trim().is_empty() => {}
                    Ok(Some(_)) => missing.push(format!(
                        "~/.pi/agent/auth.json credential for {} resolved to a non-OAuth credential ({})",
                        requirement.name, requirement.detail
                    )),
                    Ok(None) => missing.push(format!(
                        "~/.pi/agent/auth.json has no {} credential ({})",
                        requirement.name, requirement.detail
                    )),
                    Err(error) => missing.push(format!(
                        "failed to resolve or refresh ~/.pi/agent/auth.json credential for {}: {error}",
                        requirement.name
                    )),
                }
            }
            Some(_) => missing.push(format!(
                "~/.pi/agent/auth.json credential for {} is incomplete ({})",
                requirement.name, requirement.detail
            )),
            None => missing.push(format!(
                "~/.pi/agent/auth.json has no {} credential ({})",
                requirement.name, requirement.detail
            )),
        }
    }
}

fn live_strict_readiness_add_google_vertex_missing(missing: &mut Vec<String>) {
    if live_env("GOOGLE_CLOUD_PROJECT")
        .or_else(|| live_env("GCLOUD_PROJECT"))
        .is_none()
    {
        missing.push(
            "GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT is not set (Google Vertex ADC live tests)"
                .to_owned(),
        );
    }
    push_missing_env(
        missing,
        "GOOGLE_CLOUD_LOCATION",
        "Google Vertex ADC live tests",
    );

    if live_env("GOOGLE_OAUTH_ACCESS_TOKEN").is_some()
        || live_env("GOOGLE_ACCESS_TOKEN").is_some()
        || live_env("CLOUDSDK_AUTH_ACCESS_TOKEN").is_some()
    {
        return;
    }

    if let Some(path) = live_env("GOOGLE_APPLICATION_CREDENTIALS") {
        if std::path::Path::new(&path).exists() {
            return;
        }
        missing.push("GOOGLE_APPLICATION_CREDENTIALS points to a missing file".to_owned());
        return;
    }

    let appdata_adc = env::var_os("APPDATA")
        .map(std::path::PathBuf::from)
        .map(|path| {
            path.join("gcloud")
                .join("application_default_credentials.json")
        })
        .is_some_and(|path| path.exists());
    let home_adc = env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|path| {
            path.join(".config")
                .join("gcloud")
                .join("application_default_credentials.json")
        })
        .is_some_and(|path| path.exists());
    if !appdata_adc && !home_adc {
        missing
            .push("Google ADC credentials are not configured for Vertex AI live tests".to_owned());
    }
}

fn live_strict_readiness_add_azure_missing(missing: &mut Vec<String>) {
    if live_env("AZURE_OPENAI_BASE_URL").is_none()
        && live_env("AZURE_OPENAI_RESOURCE_NAME").is_none()
    {
        missing.push(
            "AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME is not set (Azure OpenAI live tests)"
                .to_owned(),
        );
    }
}

fn live_strict_readiness_add_cloudflare_missing(missing: &mut Vec<String>) {
    push_missing_env(
        missing,
        "CLOUDFLARE_ACCOUNT_ID",
        "Cloudflare Workers AI and AI Gateway live tests",
    );
    push_missing_env(
        missing,
        "CLOUDFLARE_GATEWAY_ID",
        "Cloudflare AI Gateway route live tests",
    );
}

fn live_strict_readiness_add_bedrock_missing(missing: &mut Vec<String>) {
    if live_env("AWS_REGION")
        .or_else(|| live_env("AWS_DEFAULT_REGION"))
        .is_none()
    {
        missing.push("AWS_REGION/AWS_DEFAULT_REGION is not set (Bedrock live tests)".to_owned());
    }

    let has_access_key_pair =
        live_env("AWS_ACCESS_KEY_ID").is_some() && live_env("AWS_SECRET_ACCESS_KEY").is_some();
    let has_web_identity = live_env("AWS_WEB_IDENTITY_TOKEN_FILE")
        .filter(|path| std::path::Path::new(path).exists())
        .is_some()
        && live_env("AWS_ROLE_ARN").is_some();
    let has_auth = live_env("AWS_PROFILE").is_some()
        || has_access_key_pair
        || live_env("AWS_BEARER_TOKEN_BEDROCK").is_some()
        || has_web_identity
        || live_env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
        || live_env("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some();
    if !has_auth {
        missing.push(
            "Bedrock live tests need AWS auth: AWS_PROFILE, AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY, AWS_BEARER_TOKEN_BEDROCK, AWS_WEB_IDENTITY_TOKEN_FILE+AWS_ROLE_ARN, or ECS container credentials".to_owned(),
        );
    }
}

async fn live_strict_readiness_probe_local_service(
    url: &str,
    label: &str,
) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1_000))
        .build()
        .map_err(|error| format!("failed to build local HTTP client: {error}"))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|error| format!("{label} is not reachable at {url}: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "{label} readiness probe returned {}",
            response.status()
        ));
    }
    response
        .text()
        .await
        .map_err(|error| format!("failed to read {label} readiness response: {error}"))
}

async fn live_strict_readiness_add_local_service_missing(missing: &mut Vec<String>) {
    if live_env("PI_NO_LOCAL_LLM").is_some() {
        missing.push("PI_NO_LOCAL_LLM is set; local provider live tests would skip".to_owned());
        return;
    }

    match live_strict_readiness_probe_local_service("http://localhost:11434/api/tags", "Ollama")
        .await
    {
        Ok(tags) if tags.contains("gpt-oss:20b") => {}
        Ok(_) => {
            missing.push("Ollama is reachable but model gpt-oss:20b is not installed".to_owned())
        }
        Err(error) => missing.push(error),
    }
    if let Err(error) =
        live_strict_readiness_probe_local_service("http://localhost:1234/v1/models", "LM Studio")
            .await
    {
        missing.push(error);
    }
    if let Err(error) =
        live_strict_readiness_probe_local_service("http://localhost:8081/health", "llama.cpp").await
    {
        missing.push(error);
    }
}

async fn live_strict_readiness_missing_requirements() -> Vec<String> {
    let mut missing = Vec::new();

    for requirement in live_external_requirements() {
        match requirement.category {
            "gate" => {
                push_missing_truthy_gate(&mut missing, requirement.name, requirement.detail);
            }
            "api_key" => push_missing_env(&mut missing, requirement.name, requirement.detail),
            "runtime_env" => {
                if requirement.name != "APPDATA" || cfg!(windows) {
                    push_missing_env(&mut missing, requirement.name, requirement.detail);
                }
            }
            "skip_control" => {
                if live_env(requirement.name).is_some() {
                    missing.push(format!(
                        "{} must be unset ({})",
                        requirement.name, requirement.detail
                    ));
                }
            }
            "provider_env" | "oauth_auth_storage" | "local_service" => {}
            category => missing.push(format!(
                "unknown live external requirement category {category:?} for {}",
                requirement.name
            )),
        }
    }

    live_strict_readiness_add_azure_missing(&mut missing);
    live_strict_readiness_add_google_vertex_missing(&mut missing);
    live_strict_readiness_add_cloudflare_missing(&mut missing);
    live_strict_readiness_add_bedrock_missing(&mut missing);
    live_strict_readiness_add_oauth_storage_missing(&mut missing).await;
    live_strict_readiness_add_local_service_missing(&mut missing).await;

    missing
}

#[test]
fn live_provider_gate_defaults_off_and_accepts_explicit_truthy_values() {
    assert!(!live_gate_value_enabled(None));
    for value in ["", "0", "false", "no", "off", "enabled"] {
        assert!(!live_gate_value_enabled(Some(value)));
    }
    for value in ["1", "true", "TRUE", " yes ", "on"] {
        assert!(live_gate_value_enabled(Some(value)));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn interactive_oauth_auth_storage_helper_persists_and_resolves_credentials_locally() {
    let _env_lock = LIVE_TEST_ENV_LOCK
        .lock()
        .expect("live provider env lock should not be poisoned");
    let temp_home =
        LiveTempDir::new("interactive_oauth_auth_storage_helper_persists_and_resolves_credentials");
    let _home_guard = LiveEnvVarGuard::set("HOME", temp_home.path().as_os_str());
    let provider_id = "anthropic";
    let access_token = "ri-local-oauth-access-token";
    let refresh_token = "ri-local-oauth-refresh-token";
    let credentials = StoredOAuthCredentials {
        refresh: refresh_token.to_owned(),
        access: access_token.to_owned(),
        expires: live_now_millis_i64() + 60_000,
        extra: BTreeMap::new(),
    };

    save_live_oauth_credentials_to_auth_storage(
        "interactive_oauth_auth_storage_helper_persists_and_resolves_credentials_locally",
        provider_id,
        credentials.clone(),
    )
    .await
    .expect("save live OAuth credentials into temp auth storage");

    let path = default_auth_storage_path().expect("temp HOME should resolve auth storage path");
    assert_eq!(
        path,
        temp_home.path().join(".pi").join("agent").join("auth.json")
    );
    assert!(path.exists(), "auth storage file should be written");

    let storage = load_auth_storage_from_path(&path).expect("load temp auth storage");
    match storage
        .get(provider_id)
        .expect("stored provider OAuth credential should exist")
    {
        AuthCredential::OAuth {
            credentials: stored,
        } => {
            assert_eq!(stored.access, access_token);
            assert_eq!(stored.refresh, refresh_token);
            assert_eq!(stored.expires, credentials.expires);
        }
        AuthCredential::ApiKey { .. } => {
            panic!("stored provider credential should be OAuth, not API key")
        }
    }

    let resolution = resolve_auth_storage_api_key(provider_id)
        .await
        .expect("resolve temp auth storage credential")
        .expect("stored credential should resolve");
    assert_eq!(resolution.api_key, access_token);
    assert_eq!(resolution.credentials, Some(credentials));
    assert!(!resolution.refreshed);
}

fn live_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn skip_live(test: &str, reason: impl AsRef<str>) {
    let reason = reason.as_ref();
    skip_live_with_strict_value(test, reason, env::var(LIVE_STRICT_ENV).ok().as_deref());
}

fn skip_live_with_strict_value(test: &str, reason: &str, strict_value: Option<&str>) {
    if live_gate_value_enabled(strict_value) {
        panic!("strict live test {test} would skip: {reason}");
    }
    eprintln!("skipping {test}: {reason}");
}

#[test]
fn live_strict_gate_turns_skips_into_failures() {
    skip_live_with_strict_value("non_strict_live_test", "missing credential", None);
    skip_live_with_strict_value("non_strict_live_test", "missing credential", Some("0"));

    let panic = std::panic::catch_unwind(|| {
        skip_live_with_strict_value("strict_live_test", "missing credential", Some("1"));
    })
    .expect_err("strict skip should panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&'static str>().copied())
        .unwrap_or_default();
    assert!(message.contains("strict live test strict_live_test would skip"));
    assert!(message.contains("missing credential"));
}

#[tokio::test(flavor = "current_thread")]
async fn strict_live_readiness_missing_requirements_reports_core_external_proof_gates() {
    let _env_lock = LIVE_TEST_ENV_LOCK
        .lock()
        .expect("live provider env lock should not be poisoned");
    let mut guards = Vec::new();
    let mut env_names = BTreeSet::new();
    for requirement in live_external_requirements() {
        if requirement.category != "local_service" {
            env_names.insert(requirement.name);
        }
    }
    env_names.extend(live_strict_readiness_provider_env_names());
    for env_name in env_names {
        guards.push(LiveEnvVarGuard::remove(env_name));
    }
    let temp_home = LiveTempDir::new("strict_live_readiness_missing_requirements");
    let _home_guard = LiveEnvVarGuard::set("HOME", temp_home.path().as_os_str());

    let missing = live_strict_readiness_missing_requirements().await;
    for requirement in live_external_requirements() {
        match requirement.category {
            "gate" => {
                let expected = format!("{}=1 is not set", requirement.name);
                assert!(
                    missing.iter().any(|item| item.contains(&expected)),
                    "strict readiness missing list must include gate requirement {expected:?}; got:\n{}",
                    missing.join("\n")
                );
            }
            "api_key" => {
                let expected = format!("{} is not set", requirement.name);
                assert!(
                    missing.iter().any(|item| item.contains(&expected)),
                    "strict readiness missing list must include API-key requirement {expected:?}; got:\n{}",
                    missing.join("\n")
                );
            }
            _ => {}
        }
    }
    for expected in [
        "RI_LIVE_PROVIDER_TESTS=1 is not set",
        "RI_LIVE_PROVIDER_STRICT=1 is not set",
        "RI_LIVE_OAUTH_INTERACTIVE_TESTS=1 is not set",
        "BEDROCK_EXTENSIVE_MODEL_TEST=1 is not set",
        "OPENAI_API_KEY is not set",
        "ANTHROPIC_API_KEY is not set",
        "OPENROUTER_API_KEY is not set",
        "AWS_REGION/AWS_DEFAULT_REGION is not set",
        "Bedrock live tests need AWS auth",
        "~/.pi/agent/auth.json is missing",
    ] {
        assert!(
            missing.iter().any(|item| item.contains(expected)),
            "strict readiness missing list must include {expected:?}; got:\n{}",
            missing.join("\n")
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn strict_live_readiness_reports_per_provider_oauth_auth_storage_gaps() {
    let _env_lock = LIVE_TEST_ENV_LOCK
        .lock()
        .expect("live provider env lock should not be poisoned");
    let temp_home = LiveTempDir::new("strict_live_readiness_oauth_auth_storage_gaps");
    let _home_guard = LiveEnvVarGuard::set("HOME", temp_home.path().as_os_str());
    let path = default_auth_storage_path().expect("temp HOME should resolve auth storage path");
    let mut storage = AuthStorage::new();
    storage.insert(
        "anthropic".to_owned(),
        AuthCredential::OAuth {
            credentials: StoredOAuthCredentials {
                refresh: "anthropic-refresh".to_owned(),
                access: "anthropic-access".to_owned(),
                expires: live_now_millis_i64() + 60_000,
                extra: BTreeMap::new(),
            },
        },
    );
    storage.insert(
        "github-copilot".to_owned(),
        AuthCredential::OAuth {
            credentials: StoredOAuthCredentials {
                refresh: "github-refresh".to_owned(),
                access: String::new(),
                expires: live_now_millis_i64() + 60_000,
                extra: BTreeMap::new(),
            },
        },
    );
    save_auth_storage_to_path(&path, &storage).expect("save temp auth storage");

    let missing = live_strict_readiness_missing_requirements().await;
    assert!(
        !missing
            .iter()
            .any(|item| item.contains("credential for anthropic")),
        "valid unexpired anthropic OAuth credential should not be reported missing:\n{}",
        missing.join("\n")
    );
    for expected in [
        "~/.pi/agent/auth.json credential for github-copilot is incomplete",
        "~/.pi/agent/auth.json has no openai-codex credential",
    ] {
        assert!(
            missing.iter().any(|item| item.contains(expected)),
            "strict readiness missing list must include {expected:?}; got:\n{}",
            missing.join("\n")
        );
    }
}

#[tokio::test]
async fn strict_live_readiness_reports_all_missing_external_requirements() {
    if !live_gate_value_enabled(env::var(LIVE_STRICT_ENV).ok().as_deref()) {
        return;
    }

    let missing = live_strict_readiness_missing_requirements().await;
    assert!(
        missing.is_empty(),
        "strict live provider matrix is not ready; missing requirements:\n- {}",
        missing.join("\n- ")
    );
}

fn live_api_key(test: &str, env_name: &str) -> Option<String> {
    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return None;
    }
    live_env(env_name).or_else(|| {
        skip_live(test, format!("{env_name} is not set"));
        None
    })
}

async fn live_oauth_resolution(test: &str, provider_id: &str) -> Option<OAuthApiKeyResolution> {
    assert!(
        has_live_external_requirement("oauth_auth_storage", provider_id),
        "missing live OAuth auth-storage external requirement for provider {provider_id:?}"
    );
    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return None;
    }
    let Some(path) = default_auth_storage_path() else {
        skip_live(test, "HOME is not set; cannot read ~/.pi/agent/auth.json");
        return None;
    };
    match resolve_auth_storage_api_key(provider_id).await {
        Ok(Some(resolution)) => Some(resolution),
        Ok(None) => {
            skip_live(
                test,
                format!("{} has no {provider_id} credential", path.display()),
            );
            None
        }
        Err(error) => {
            skip_live(
                test,
                format!("failed to resolve {provider_id} OAuth credential: {error}"),
            );
            None
        }
    }
}

fn live_oauth_interactive_enabled(test: &str) -> bool {
    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return false;
    }
    if !live_gate_value_enabled(env::var(LIVE_OAUTH_INTERACTIVE_ENV).ok().as_deref()) {
        eprintln!(
            "skipping {test}: {LIVE_OAUTH_INTERACTIVE_ENV}=1 is not set; interactive OAuth login would wait for browser/device-code confirmation and write ~/.pi/agent/auth.json"
        );
        return false;
    }
    true
}

fn live_oauth_interactive_timeout() -> Duration {
    Duration::from_secs(LIVE_OAUTH_INTERACTIVE_TIMEOUT_SECS)
}

fn live_now_millis_i64() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after Unix epoch")
        .as_millis() as i64
}

async fn live_oauth_interactive_timeout_result<T>(
    test: &str,
    label: &str,
    future: impl Future<Output = Result<T, String>>,
) -> Result<T, Box<dyn Error>> {
    match tokio::time::timeout(live_oauth_interactive_timeout(), future).await {
        Ok(result) => result.map_err(Into::into),
        Err(_) => Err(format!(
            "{test} timed out waiting for {label} after {LIVE_OAUTH_INTERACTIVE_TIMEOUT_SECS} seconds"
        )
        .into()),
    }
}

fn live_print_oauth_callback_instructions(test: &str, provider: &str, flow: &OAuthLoginFlow) {
    eprintln!("{test}: complete {provider} OAuth login in a browser:");
    eprintln!("{}", flow.auth_url);
    eprintln!(
        "{test}: waiting for localhost callback at {} and will store the credential in ~/.pi/agent/auth.json",
        flow.local_addr
    );
    if let Some(instructions) = flow.instructions.as_deref() {
        eprintln!("{test}: {instructions}");
    }
}

async fn save_live_oauth_credentials_to_auth_storage(
    test: &str,
    provider_id: &str,
    credentials: StoredOAuthCredentials,
) -> Result<(), Box<dyn Error>> {
    assert!(
        has_live_external_requirement("oauth_auth_storage", provider_id),
        "missing live OAuth auth-storage external requirement for provider {provider_id:?}"
    );
    let Some(path) = default_auth_storage_path() else {
        return Err(format!("{test}: HOME is not set; cannot write ~/.pi/agent/auth.json").into());
    };

    let mut storage = load_auth_storage_from_path(&path)?;
    storage.insert(
        provider_id.to_owned(),
        AuthCredential::OAuth { credentials },
    );
    save_auth_storage_to_path(&path, &storage)?;

    let resolution = resolve_auth_storage_api_key(provider_id)
        .await?
        .ok_or_else(|| format!("{test}: stored {provider_id} OAuth credential was not found"))?;
    assert!(
        resolution.credentials.is_some(),
        "{test}: stored {provider_id} credential resolved as a non-OAuth credential"
    );
    assert!(
        !resolution.api_key.trim().is_empty(),
        "{test}: stored {provider_id} OAuth credential resolved to an empty access token"
    );
    eprintln!(
        "{test}: stored and resolved {provider_id} OAuth credential in {}",
        path.display()
    );
    Ok(())
}

fn live_network_enabled(test: &str) -> bool {
    if live_tests_enabled() {
        true
    } else {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        false
    }
}

fn live_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text(LIVE_PROMPT))],
        ..Default::default()
    }
}

fn live_image_input_context() -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage {
            content: UserContentValue::Blocks(vec![
                UserContent::Text(TextContent::new(
                    "What do you see in this image? Describe the shape and color. You MUST reply in English.",
                )),
                UserContent::Image(ImageContent {
                    data: LIVE_RED_CIRCLE_PNG_BASE64.to_owned(),
                    mime_type: "image/png".to_owned(),
                }),
            ]),
            timestamp: now_millis(),
        })],
        ..Default::default()
    }
}

fn live_reasoning_context() -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text(
            "Think step by step about 42 + 27, then output the result.",
        ))],
        ..Default::default()
    }
}

fn live_response_id_context() -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant. Be concise.".to_owned()),
        messages: vec![Message::User(UserMessage::text(LIVE_RESPONSE_ID_PROMPT))],
        ..Default::default()
    }
}

fn live_abort_context() -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text(LIVE_ABORT_PROMPT))],
        ..Default::default()
    }
}

fn live_immediate_abort_context() -> Context {
    Context {
        messages: vec![Message::User(UserMessage::text("Hello"))],
        ..Default::default()
    }
}

fn live_abort_then_new_message_context(aborted: AssistantMessage) -> Context {
    Context {
        messages: vec![
            Message::User(UserMessage::text("Hello, how are you?")),
            Message::Assistant(aborted),
            Message::User(UserMessage::text("What is 2 + 2?")),
        ],
        ..Default::default()
    }
}

fn live_midstream_abort_then_new_message_context(aborted: AssistantMessage) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![
            Message::User(UserMessage::text(LIVE_ABORT_PROMPT)),
            Message::Assistant(aborted),
            Message::User(UserMessage::text(
                "Please continue, but keep the answer to one short sentence.",
            )),
        ],
        ..Default::default()
    }
}

fn live_multiturn_context(model: &Model) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant. Be concise.".to_owned()),
        messages: vec![
            Message::User(UserMessage::text(format!(
                "Reply with exactly: {LIVE_MULTITURN_FIRST}"
            ))),
            Message::Assistant(live_assistant_message(model, LIVE_MULTITURN_FIRST)),
            Message::User(UserMessage::text(format!(
                "Now reply with exactly: {LIVE_MULTITURN_SECOND}"
            ))),
        ],
        ..Default::default()
    }
}

fn live_long_system_prompt() -> String {
    let mut prompt = String::from(
        "You are a helpful assistant. Be concise in your responses.\n\n\
         Here is some additional context that makes this system prompt long enough to exercise provider usage accounting:\n\n",
    );
    for _ in 0..50 {
        prompt.push_str(LIVE_LONG_SYSTEM_PROMPT_SEGMENT);
        prompt.push_str("\n\n");
    }
    prompt.push_str("Remember: Always be helpful and concise.");
    prompt
}

fn live_openrouter_cache_write_system_prompt() -> String {
    let mut prompt = format!(
        "You are a concise assistant.\nCache nonce: {}\n\n",
        now_millis()
    );
    for _ in 0..80 {
        prompt.push_str(
            "Prompt-caching probe content. Keep this exact text stable across requests so the provider can reuse prefix tokens and report cache read and cache write usage.",
        );
        prompt.push_str("\n\n");
    }
    prompt
}

fn live_openrouter_cache_write_context() -> Context {
    Context {
        system_prompt: Some(live_openrouter_cache_write_system_prompt()),
        messages: vec![Message::User(UserMessage::text("Reply with exactly: OK"))],
        ..Default::default()
    }
}

fn live_anthropic_opus_47_reasoning_context() -> Context {
    Context {
        system_prompt: Some(
            "You are a precise assistant. Follow the user's instructions exactly.".to_owned(),
        ),
        messages: vec![Message::User(UserMessage::text(
            "Compute 48291 * 7317 and 90844 - 17729, add the results, and determine whether the sum is divisible by 11. Reply with exactly this format and nothing else: sum=<sum>; divisibleBy11=<yes|no>",
        ))],
        ..Default::default()
    }
}

fn live_thinking_disable_context() -> Context {
    Context {
        system_prompt: Some(
            "You are a precise assistant. Follow the requested output format exactly.".to_owned(),
        ),
        messages: vec![Message::User(UserMessage::text(
            "Before replying, carefully solve 36863 * 5279 internally. Then reply with the word pong repeated exactly 40 times, separated by single spaces. Do not add any other text.",
        ))],
        ..Default::default()
    }
}

fn live_context_overflow_content(context_window: u64) -> String {
    let target_tokens = context_window.saturating_add(10_000);
    let target_chars = (target_tokens as f64 * 4.0 * 1.5).ceil() as usize;
    let repetitions = target_chars.div_ceil(LIVE_CONTEXT_OVERFLOW_LOREM.len());
    LIVE_CONTEXT_OVERFLOW_LOREM.repeat(repetitions)
}

fn live_context_overflow_context(model: &Model) -> Context {
    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text(
            live_context_overflow_content(model.context_window),
        ))],
        ..Default::default()
    }
}

async fn live_local_http_get_text(test: &str, url: &str, label: &str) -> Option<String> {
    if !live_network_enabled(test) {
        return None;
    }
    if live_env("PI_NO_LOCAL_LLM").is_some() {
        skip_live(test, "PI_NO_LOCAL_LLM is set");
        return None;
    }
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(1_000))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            skip_live(test, format!("failed to build local HTTP client: {error}"));
            return None;
        }
    };
    let response = match client.get(url).send().await {
        Ok(response) => response,
        Err(error) => {
            skip_live(test, format!("{label} is not reachable at {url}: {error}"));
            return None;
        }
    };
    if !response.status().is_success() {
        skip_live(
            test,
            format!("{label} readiness probe returned {}", response.status()),
        );
        return None;
    }
    match response.text().await {
        Ok(text) => Some(text),
        Err(error) => {
            skip_live(
                test,
                format!("failed to read {label} readiness response: {error}"),
            );
            None
        }
    }
}

fn live_local_openai_completions_model(
    provider: &str,
    id: &str,
    base_url: &str,
    reasoning: bool,
    context_window: u64,
    max_tokens: u64,
    name: &str,
) -> Model {
    Model {
        id: id.to_owned(),
        name: name.to_owned(),
        api: "openai-completions".to_owned(),
        provider: provider.to_owned(),
        base_url: base_url.to_owned(),
        reasoning,
        thinking_level_map: BTreeMap::new(),
        input: vec![InputKind::Text],
        cost: ModelCost::default(),
        context_window,
        max_tokens,
        headers: BTreeMap::new(),
        compat: None,
    }
}

async fn live_ollama_model_options(
    test: &str,
) -> Result<Option<(Model, SimpleStreamOptions)>, Box<dyn Error>> {
    let Some(tags) =
        live_local_http_get_text(test, "http://localhost:11434/api/tags", "Ollama").await
    else {
        return Ok(None);
    };
    if !tags.contains("gpt-oss:20b") {
        skip_live(test, "Ollama model gpt-oss:20b is not installed");
        return Ok(None);
    }
    let model = live_local_openai_completions_model(
        "ollama",
        "gpt-oss:20b",
        "http://localhost:11434/v1",
        true,
        128_000,
        16_000,
        "Ollama GPT-OSS 20B",
    );
    Ok(Some((model, live_text_options(Some("ollama".to_owned())))))
}

fn live_usage_first_context() -> Context {
    Context {
        system_prompt: Some(live_long_system_prompt()),
        messages: vec![Message::User(UserMessage::text(
            "What is 2 + 2? Reply with just the number.",
        ))],
        ..Default::default()
    }
}

fn live_usage_second_context(first: AssistantMessage) -> Context {
    Context {
        system_prompt: Some(live_long_system_prompt()),
        messages: vec![
            Message::User(UserMessage::text(
                "What is 2 + 2? Reply with just the number.",
            )),
            Message::Assistant(first),
            Message::User(UserMessage::text(
                "What is 3 + 3? Reply with just the number.",
            )),
        ],
        ..Default::default()
    }
}

fn live_empty_case_context(model: &Model, case: LiveEmptyCase) -> Context {
    match case {
        LiveEmptyCase::EmptyContentArray => Context {
            messages: vec![Message::User(UserMessage {
                content: UserContentValue::Blocks(Vec::new()),
                timestamp: now_millis(),
            })],
            ..Default::default()
        },
        LiveEmptyCase::EmptyString => Context {
            messages: vec![Message::User(UserMessage::text(""))],
            ..Default::default()
        },
        LiveEmptyCase::WhitespaceOnly => Context {
            messages: vec![Message::User(UserMessage::text("   \n\t  "))],
            ..Default::default()
        },
        LiveEmptyCase::EmptyAssistant => {
            let mut assistant = live_assistant_message(model, "");
            assistant.content.clear();
            assistant.usage.input = 10;
            assistant.usage.total_tokens = 10;
            Context {
                messages: vec![
                    Message::User(UserMessage::text("Hello, how are you?")),
                    Message::Assistant(assistant),
                    Message::User(UserMessage::text("Please respond this time.")),
                ],
                ..Default::default()
            }
        }
    }
}

fn live_tool_call_context() -> Context {
    Context {
        system_prompt: Some(
            "You are a helpful assistant. You must use available tools when asked.".to_owned(),
        ),
        messages: vec![Message::User(UserMessage::text(
            "Calculate 15 + 27 using the math_operation tool. Do not answer directly.",
        ))],
        tools: vec![live_calculator_tool()],
    }
}

fn live_interleaved_thinking_system_prompt() -> String {
    [
        "You are a helpful assistant that must use tools for arithmetic.",
        "Always think before every tool call, not just the first one.",
        "Do not answer with plain text when a tool call is required.",
    ]
    .join(" ")
}

fn live_interleaved_thinking_user_prompt() -> String {
    [
        "Use calculator to calculate 328 * 29.",
        "You must call the calculator tool exactly once.",
        "Provide the final answer based on the best guess given the tool result, even if it seems unreliable.",
        "Start by thinking about the steps you will take to solve the problem.",
    ]
    .join(" ")
}

fn live_interleaved_calculator_tool() -> Tool {
    let mut tool = live_calculator_tool();
    tool.name = "calculator".to_owned();
    tool
}

fn live_interleaved_thinking_first_context() -> Context {
    Context {
        system_prompt: Some(live_interleaved_thinking_system_prompt()),
        messages: vec![Message::User(UserMessage::text(
            live_interleaved_thinking_user_prompt(),
        ))],
        tools: vec![live_interleaved_calculator_tool()],
    }
}

fn live_interleaved_thinking_second_context(
    first: AssistantMessage,
    tool_call: ToolCall,
) -> Result<Context, Box<dyn Error>> {
    let answer = evaluate_live_interleaved_calculator_call(&tool_call)?;
    let answer_text = if answer.fract().abs() < f64::EPSILON {
        format!("{}", answer as i64)
    } else {
        answer.to_string()
    };
    Ok(Context {
        system_prompt: Some(live_interleaved_thinking_system_prompt()),
        messages: vec![
            Message::User(UserMessage::text(live_interleaved_thinking_user_prompt())),
            Message::Assistant(first),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                content: vec![ToolResultContent::text(format!(
                    "The answer is {answer_text} or {}.",
                    answer * 2.0
                ))],
                details: None,
                is_error: false,
                timestamp: now_millis(),
            }),
        ],
        tools: vec![live_interleaved_calculator_tool()],
    })
}

fn live_calculator_tool() -> Tool {
    Tool {
        name: "math_operation".to_owned(),
        description: "Perform basic arithmetic operations".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number",
                    "description": "Second number"
                },
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                }
            },
            "required": ["a", "b", "operation"]
        }),
    }
}

fn live_calculate_expression_tool() -> Tool {
    Tool {
        name: "calculate".to_owned(),
        description: "Evaluate mathematical expressions".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
    }
}

fn live_echo_tool() -> Tool {
    Tool {
        name: "echo".to_owned(),
        description: "Echoes the message back".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Message to echo back"
                }
            },
            "required": ["message"]
        }),
    }
}

fn live_empty_schema_tool(name: &str, description: &str) -> Tool {
    Tool {
        name: name.to_owned(),
        description: description.to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {}
        }),
    }
}

fn live_string_arg_tool(
    name: &str,
    description: &str,
    arg_name: &str,
    arg_description: &str,
) -> Tool {
    let mut properties = serde_json::Map::new();
    properties.insert(
        arg_name.to_owned(),
        json!({
            "type": "string",
            "description": arg_description
        }),
    );
    Tool {
        name: name.to_owned(),
        description: description.to_owned(),
        parameters: json!({
            "type": "object",
            "properties": properties,
            "required": [arg_name]
        }),
    }
}

fn live_unicode_tool_result_context(model: &Model, case: LiveUnicodeToolResultCase) -> Context {
    let is_mistral = model.provider == "mistral";
    let (tool_name, tool_description, tool_call_id, prompt, result_text, follow_up) = match case {
        LiveUnicodeToolResultCase::Emoji => (
            "test_tool",
            "A test tool",
            if is_mistral { "testtool1" } else { "test_1" },
            "Use the test tool",
            "Test with emoji 🙈 and other characters:\n\
             - Monkey emoji: 🙈\n\
             - Thumbs up: 👍\n\
             - Heart: ❤️\n\
             - Thinking face: 🤔\n\
             - Rocket: 🚀\n\
             - Mixed text: Mario Zechner wann? Wo? Bin grad äußersr eventuninformiert 🙈\n\
             - Japanese: こんにちは\n\
             - Chinese: 你好\n\
             - Mathematical symbols: ∑∫∂√\n\
             - Special quotes: \"curly\" 'quotes'",
            "Summarize the tool result briefly.",
        ),
        LiveUnicodeToolResultCase::LinkedIn => (
            "linkedin_skill",
            "Get LinkedIn comments",
            if is_mistral {
                "linkedin1"
            } else {
                "linkedin_1"
            },
            "Use the linkedin tool to get comments",
            "Post: Hab einen \"Generative KI für Nicht-Techniker\" Workshop gebaut.\n\
             Unanswered Comments: 2\n\n\
             => {\n\
             \"comments\": [\n\
             {\n\
             \"author\": \"Matthias Neumayer's  graphic link\",\n\
             \"text\": \"Leider nehmen das viel zu wenige Leute ernst\"\n\
             },\n\
             {\n\
             \"author\": \"Matthias Neumayer's  graphic link\",\n\
             \"text\": \"Mario Zechner wann? Wo? Bin grad äußersr eventuninformiert 🙈\"\n\
             }\n\
             ]\n\
             }",
            "How many comments are there?",
        ),
    };

    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: tool_call_id.to_owned(),
            name: tool_name.to_owned(),
            arguments: serde_json::Map::new(),
            thought_signature: None,
        })],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: now_millis(),
    };

    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![
            Message::User(UserMessage::text(prompt)),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: tool_call_id.to_owned(),
                tool_name: tool_name.to_owned(),
                content: vec![ToolResultContent::text(result_text)],
                details: None,
                is_error: false,
                timestamp: now_millis(),
            }),
            Message::User(UserMessage::text(follow_up)),
        ],
        tools: vec![live_empty_schema_tool(tool_name, tool_description)],
    }
}

fn live_image_tool_result_tool_spec(
    case: LiveImageToolResultCase,
) -> (&'static str, &'static str, &'static str) {
    match case {
        LiveImageToolResultCase::ImageOnly => (
            "get_circle",
            "Returns a circle image for visualization",
            "Call the get_circle tool to get an image, and describe what you see, shapes, colors, etc.",
        ),
        LiveImageToolResultCase::TextAndImage => (
            "get_circle_with_description",
            "Returns a circle image with a text description",
            "Use the get_circle_with_description tool and tell me what you learned. Also say what color the shape is.",
        ),
    }
}

fn live_image_tool_result_first_context(case: LiveImageToolResultCase) -> Context {
    let (tool_name, tool_description, prompt) = live_image_tool_result_tool_spec(case);
    Context {
        system_prompt: Some("You are a helpful assistant that uses tools when asked.".to_owned()),
        messages: vec![Message::User(UserMessage::text(prompt))],
        tools: vec![live_empty_schema_tool(tool_name, tool_description)],
    }
}

fn live_image_tool_result_second_context(
    first: AssistantMessage,
    tool_call: ToolCall,
    case: LiveImageToolResultCase,
) -> Context {
    let (_, _, prompt) = live_image_tool_result_tool_spec(case);
    let mut content = Vec::new();
    if case == LiveImageToolResultCase::TextAndImage {
        content.push(ToolResultContent::text(
            "This is a geometric shape with specific properties: it has a diameter of 100 pixels.",
        ));
    }
    content.push(ToolResultContent::Image(ImageContent {
        data: LIVE_RED_CIRCLE_PNG_BASE64.to_owned(),
        mime_type: "image/png".to_owned(),
    }));

    Context {
        system_prompt: Some("You are a helpful assistant that uses tools when asked.".to_owned()),
        messages: vec![
            Message::User(UserMessage::text(prompt)),
            Message::Assistant(first),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                content,
                details: None,
                is_error: false,
                timestamp: now_millis(),
            }),
        ],
        tools: vec![live_empty_schema_tool(
            live_image_tool_result_tool_spec(case).0,
            live_image_tool_result_tool_spec(case).1,
        )],
    }
}

fn live_responses_tool_result_images_first_context() -> Context {
    Context {
        system_prompt: Some(
            "You are a helpful assistant that always uses the provided tool when asked.".to_owned(),
        ),
        messages: vec![Message::User(UserMessage::text(
            "Call get_circle_with_description, then describe both the tool text and the image. Mention the color and shape.",
        ))],
        tools: vec![live_empty_schema_tool(
            "get_circle_with_description",
            "Returns a red circle image with a short text description.",
        )],
    }
}

fn live_responses_tool_result_images_second_context(
    first: AssistantMessage,
    tool_call: ToolCall,
) -> Context {
    Context {
        system_prompt: Some(
            "You are a helpful assistant that always uses the provided tool when asked.".to_owned(),
        ),
        messages: vec![
            Message::User(UserMessage::text(
                "Call get_circle_with_description, then describe both the tool text and the image. Mention the color and shape.",
            )),
            Message::Assistant(first),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: tool_call.id,
                tool_name: tool_call.name,
                content: vec![
                    ToolResultContent::text(LIVE_RESPONSES_TOOL_IMAGE_TEXT),
                    ToolResultContent::Image(ImageContent {
                        data: LIVE_RED_CIRCLE_PNG_BASE64.to_owned(),
                        mime_type: "image/png".to_owned(),
                    }),
                ],
                details: None,
                is_error: false,
                timestamp: now_millis(),
            }),
        ],
        tools: vec![live_empty_schema_tool(
            "get_circle_with_description",
            "Returns a red circle image with a short text description.",
        )],
    }
}

fn live_tool_call_without_result_first_context() -> Context {
    Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the calculate tool when asked to perform calculations."
                .to_owned(),
        ),
        messages: vec![Message::User(UserMessage::text(
            "Please calculate 25 * 18 using the calculate tool.",
        ))],
        tools: vec![live_calculate_expression_tool()],
    }
}

fn live_tool_call_without_result_second_context(first: AssistantMessage) -> Context {
    Context {
        system_prompt: Some(
            "You are a helpful assistant. Use the calculate tool when asked to perform calculations."
                .to_owned(),
        ),
        messages: vec![
            Message::User(UserMessage::text(
                "Please calculate 25 * 18 using the calculate tool.",
            )),
            Message::Assistant(first),
            Message::User(UserMessage::text("Never mind, just tell me what is 2+2?")),
        ],
        tools: vec![live_calculate_expression_tool()],
    }
}

fn live_tool_name_normalization_context(
    tool: Tool,
    system_prompt: &str,
    user_prompt: &str,
) -> Context {
    Context {
        system_prompt: Some(system_prompt.to_owned()),
        messages: vec![Message::User(UserMessage::text(user_prompt))],
        tools: vec![tool],
    }
}

fn live_assistant_message(model: &Model, text: &str) -> AssistantMessage {
    AssistantMessage {
        content: vec![AssistantContent::Text(TextContent::new(text))],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: now_millis(),
    }
}

fn live_text_options(api_key: Option<String>) -> SimpleStreamOptions {
    SimpleStreamOptions {
        stream: StreamOptions {
            api_key,
            max_tokens: Some(32),
            timeout_ms: Some(60_000),
            max_retries: Some(1),
            max_retry_delay_ms: Some(1_000),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn live_context_overflow_options(api_key: Option<String>) -> SimpleStreamOptions {
    let mut options = live_text_options(api_key);
    options.stream.timeout_ms = Some(120_000);
    options
}

fn live_abort_options(api_key: Option<String>, abort_flag: Arc<AtomicBool>) -> SimpleStreamOptions {
    let mut options = live_text_options(api_key);
    options.stream.max_tokens = Some(2_048);
    options.stream.abort_flag = Some(abort_flag);
    options
}

fn assert_live_text_response(test: &str, message: &AssistantMessage) {
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{test} returned provider error: {:?}",
        message.error_message
    );
    assert_ne!(
        message.stop_reason,
        StopReason::Aborted,
        "{test} was aborted: {:?}",
        message.error_message
    );
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !text.trim().is_empty(),
        "{test} returned no text content: {:?}",
        message.content
    );
}

fn assert_live_response_id(test: &str, message: &AssistantMessage) {
    assert_live_text_response(test, message);
    assert!(
        message
            .response_id
            .as_deref()
            .map(str::trim)
            .is_some_and(|response_id| !response_id.is_empty()),
        "{test} returned no response_id: {message:?}"
    );
}

fn assert_live_abort_usage(
    test: &str,
    message: &AssistantMessage,
    expectation: LiveAbortUsageExpectation,
) {
    assert_eq!(
        message.stop_reason,
        StopReason::Aborted,
        "{test} expected an aborted stream, got {:?}: {:?}",
        message.stop_reason,
        message.error_message
    );
    match expectation {
        LiveAbortUsageExpectation::ZeroInputOutput => {
            assert_eq!(message.usage.input, 0, "{test} input usage");
            assert_eq!(message.usage.output, 0, "{test} output usage");
        }
        LiveAbortUsageExpectation::PositiveInputOutput => {
            assert!(
                message.usage.input > 0,
                "{test} expected positive input usage: {:?}",
                message.usage
            );
            assert!(
                message.usage.output > 0,
                "{test} expected positive output usage: {:?}",
                message.usage
            );
        }
        LiveAbortUsageExpectation::PositiveInputZeroOutput => {
            assert!(
                message.usage.input > 0,
                "{test} expected positive input usage: {:?}",
                message.usage
            );
            assert_eq!(message.usage.output, 0, "{test} output usage");
        }
    }
}

fn assert_live_usage_total_matches_components(test: &str, usage: &Usage) {
    assert_eq!(
        usage.total_tokens,
        usage.component_total(),
        "{test} returned inconsistent usage totals: {usage:?}"
    );
    assert!(
        usage.total_tokens_match_components(),
        "{test} usage total did not match components: {usage:?}"
    );
    assert!(
        usage.component_total() > 0,
        "{test} returned empty usage for a completed request: {usage:?}"
    );
}

fn assert_live_anthropic_opus_47_payload(test: &str, payload: &Value) {
    assert_eq!(
        payload
            .get("thinking")
            .and_then(|thinking| thinking.get("type"))
            .and_then(Value::as_str),
        Some("adaptive"),
        "{test} expected adaptive thinking payload: {payload:?}"
    );
    assert_eq!(
        payload
            .get("output_config")
            .and_then(|config| config.get("effort"))
            .and_then(Value::as_str),
        Some("high"),
        "{test} expected output_config.effort=high: {payload:?}"
    );
}

fn assert_live_anthropic_opus_47_response(
    test: &str,
    message: &AssistantMessage,
    saw_thinking_event: bool,
) {
    assert_eq!(
        message.stop_reason,
        StopReason::Stop,
        "{test} expected stop, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
    assert!(
        saw_thinking_event,
        "{test} completed without streamed thinking events: {message:?}"
    );
    let thinking = message
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::Thinking(thinking) => Some(thinking),
            _ => None,
        })
        .unwrap_or_else(|| panic!("{test} returned no thinking block: {message:?}"));
    assert!(
        thinking
            .thinking_signature
            .as_deref()
            .is_some_and(|signature| !signature.trim().is_empty()),
        "{test} returned no thinking signature: {thinking:?}"
    );
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_owned();
    assert_eq!(
        text, "sum=353418362; divisibleBy11=yes",
        "{test} returned unexpected final answer: {message:?}"
    );
}

fn live_count_pongs(text: &str) -> usize {
    text.split(|ch: char| !ch.is_ascii_alphabetic())
        .filter(|word| word.eq_ignore_ascii_case("pong"))
        .count()
}

fn assert_live_thinking_disabled_response(
    test: &str,
    message: &AssistantMessage,
    saw_thinking_event: bool,
    thinking_char_count: usize,
    min_pongs: usize,
    max_output_tokens: Option<u64>,
) {
    assert_eq!(
        message.stop_reason,
        StopReason::Stop,
        "{test} expected stop, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
    assert!(
        !saw_thinking_event,
        "{test} emitted thinking events despite reasoning being disabled: {message:?}"
    );
    assert_eq!(
        thinking_char_count, 0,
        "{test} emitted thinking deltas despite reasoning being disabled"
    );
    assert!(
        !message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Thinking(_))),
        "{test} returned a thinking content block despite reasoning being disabled: {message:?}"
    );
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_owned();
    let pong_count = live_count_pongs(&text);
    assert!(
        pong_count >= min_pongs,
        "{test} returned too few pong tokens ({pong_count} < {min_pongs}): {text:?}"
    );
    if let Some(max_output_tokens) = max_output_tokens {
        assert!(
            message.usage.output < max_output_tokens,
            "{test} expected output usage below {max_output_tokens}, got {:?}",
            message.usage
        );
    }
}

fn assert_live_context_overflow_response(
    test: &str,
    model: &Model,
    message: &AssistantMessage,
    expectation: LiveContextOverflowExpectation,
) {
    match expectation {
        LiveContextOverflowExpectation::Error => {
            assert_eq!(
                message.stop_reason,
                StopReason::Error,
                "{test} expected context overflow error, got {:?}: {message:?}",
                message.stop_reason
            );
            assert!(
                is_context_overflow(message, Some(model.context_window)),
                "{test} response was not recognized as context overflow: {message:?}"
            );
        }
        LiveContextOverflowExpectation::LengthZeroOutput => {
            assert_eq!(
                message.stop_reason,
                StopReason::Length,
                "{test} expected length-stop overflow, got {:?}: {message:?}",
                message.stop_reason
            );
            assert_eq!(
                message.usage.output, 0,
                "{test} expected zero output tokens for length-stop overflow: {:?}",
                message.usage
            );
            assert!(
                is_context_overflow(message, Some(model.context_window)),
                "{test} length-stop response was not recognized as context overflow: {message:?}"
            );
        }
        LiveContextOverflowExpectation::ZaiInconsistent => {
            if message.stop_reason == StopReason::Error {
                if is_context_overflow(message, Some(model.context_window)) {
                    return;
                } else {
                    skip_live(
                        test,
                        "z.ai returned a non-overflow error, likely a transient rate limit",
                    );
                }
            } else if message.stop_reason == StopReason::Stop
                && message.usage.input + message.usage.cache_read > model.context_window
            {
                assert!(
                    is_context_overflow(message, Some(model.context_window)),
                    "{test} z.ai accepted overflow but was not recognized: {message:?}"
                );
            } else {
                skip_live(
                    test,
                    format!("z.ai did not expose an overflow signal: {message:?}"),
                );
            }
        }
        LiveContextOverflowExpectation::SilentTruncationOrError => {
            if message.stop_reason == StopReason::Error {
                assert!(
                    is_context_overflow(message, Some(model.context_window)),
                    "{test} local provider error was not recognized as context overflow: {message:?}"
                );
            } else if message.stop_reason == StopReason::Stop
                && message.usage.input + message.usage.cache_read > 0
            {
                skip_live(
                    test,
                    format!(
                        "local provider silently truncated or accepted oversized input: {message:?}"
                    ),
                );
            } else {
                skip_live(
                    test,
                    format!("local provider did not expose an overflow signal: {message:?}"),
                );
            }
        }
    }
}

fn assert_live_interleaved_has_thinking(test: &str, message: &AssistantMessage, turn: &str) {
    assert!(
        message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Thinking(_))),
        "{test} {turn} turn returned no thinking block: {message:?}"
    );
}

fn assert_live_interleaved_final_response(test: &str, message: &AssistantMessage) {
    assert_live_anthropic_e2e_accepted(test, message);
    assert_eq!(
        message.stop_reason,
        StopReason::Stop,
        "{test} expected final stop, got {:?}: {message:?}",
        message.stop_reason
    );
    assert_live_interleaved_has_thinking(test, message, "second");
    assert!(
        message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Text(text) if !text.text.trim().is_empty())),
        "{test} final response returned no text block: {message:?}"
    );
}

fn assert_live_empty_response(test: &str, message: &AssistantMessage, case: LiveEmptyCase) {
    assert_ne!(
        message.stop_reason,
        StopReason::Aborted,
        "{test} unexpectedly aborted: {message:?}"
    );
    if message.stop_reason == StopReason::Error {
        assert!(
            message
                .error_message
                .as_deref()
                .map(str::trim)
                .is_some_and(|error| !error.is_empty()),
            "{test} returned error without message: {message:?}"
        );
    } else if case == LiveEmptyCase::EmptyAssistant {
        assert!(
            !message.content.is_empty(),
            "{test} returned no assistant content after empty assistant history: {message:?}"
        );
    }
}

fn assert_live_provider_error(test: &str, message: &AssistantMessage) {
    assert_eq!(
        message.stop_reason,
        StopReason::Error,
        "{test} expected provider error, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        message
            .error_message
            .as_deref()
            .map(str::trim)
            .is_some_and(|error| !error.is_empty()),
        "{test} returned no provider error message: {message:?}"
    );
}

fn assert_live_tool_call(test: &str, message: &AssistantMessage) {
    assert_eq!(
        message.stop_reason,
        StopReason::ToolUse,
        "{test} expected tool use, got {:?}: {message:?}",
        message.stop_reason
    );
    let Some(tool_call) = message.content.iter().find_map(|content| match content {
        AssistantContent::ToolCall(tool_call) => Some(tool_call),
        _ => None,
    }) else {
        panic!("{test} returned no tool call: {message:?}");
    };
    assert_valid_live_math_tool_call(test, tool_call);
}

fn live_interleaved_tool_call_from_response(
    test: &str,
    message: &AssistantMessage,
) -> Result<ToolCall, Box<dyn Error>> {
    assert_live_anthropic_e2e_accepted(test, message);
    assert_eq!(
        message.stop_reason,
        StopReason::ToolUse,
        "{test} expected first turn tool use, got {:?}: {message:?}",
        message.stop_reason
    );
    assert_live_interleaved_has_thinking(test, message, "first");
    message
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .ok_or_else(|| format!("{test} first turn returned no tool call: {message:?}").into())
}

fn evaluate_live_interleaved_calculator_call(tool_call: &ToolCall) -> Result<f64, Box<dyn Error>> {
    let a = tool_call
        .arguments
        .get("a")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("calculator argument `a` must be numeric: {tool_call:?}"))?;
    let b = tool_call
        .arguments
        .get("b")
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("calculator argument `b` must be numeric: {tool_call:?}"))?;
    let operation = tool_call
        .arguments
        .get("operation")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            format!("calculator argument `operation` must be a string: {tool_call:?}")
        })?;
    match operation {
        "add" => Ok(a + b),
        "subtract" => Ok(a - b),
        "multiply" => Ok(a * b),
        "divide" => Ok(a / b),
        other => Err(format!("unsupported calculator operation `{other}`: {tool_call:?}").into()),
    }
}

fn assert_live_named_tool_call(test: &str, message: &AssistantMessage, expected_name: &str) {
    assert_eq!(
        message.stop_reason,
        StopReason::ToolUse,
        "{test} expected tool use, got {:?}: {message:?}",
        message.stop_reason
    );
    let Some(tool_call) = message.content.iter().find_map(|content| match content {
        AssistantContent::ToolCall(tool_call) => Some(tool_call),
        _ => None,
    }) else {
        panic!("{test} returned no tool call: {message:?}");
    };
    assert_eq!(
        tool_call.name, expected_name,
        "{test} returned unexpected tool name: {tool_call:?}"
    );
}

fn assert_valid_live_math_tool_call(test: &str, tool_call: &ToolCall) {
    assert_eq!(tool_call.name, "math_operation", "{test} tool name");
    assert!(
        !tool_call.id.trim().is_empty(),
        "{test} returned an empty tool call id: {tool_call:?}"
    );
    assert_number_arg(test, tool_call, "a", 15.0);
    assert_number_arg(test, tool_call, "b", 27.0);
    assert!(
        matches!(
            tool_call.arguments.get("operation").and_then(Value::as_str),
            Some("add" | "subtract" | "multiply" | "divide")
        ),
        "{test} returned invalid operation argument: {:?}",
        tool_call.arguments
    );
}

fn assert_live_any_tool_call(test: &str, message: &AssistantMessage) {
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{test} returned provider error before tool call: {:?}",
        message.error_message
    );
    assert!(
        message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(_))),
        "{test} expected first response to contain a tool call: {message:?}"
    );
}

fn assert_live_tool_call_without_result_response(test: &str, message: &AssistantMessage) {
    assert!(
        matches!(message.stop_reason, StopReason::Stop | StopReason::ToolUse),
        "{test} expected stop/toolUse after orphaned tool call, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        !message.content.is_empty(),
        "{test} returned no content after orphaned tool call: {message:?}"
    );
    let has_text = message.content.iter().any(
        |content| matches!(content, AssistantContent::Text(text) if !text.text.trim().is_empty()),
    );
    let has_tool_call = message
        .content
        .iter()
        .any(|content| matches!(content, AssistantContent::ToolCall(_)));
    assert!(
        has_text || has_tool_call,
        "{test} returned neither text nor tool call after orphaned tool call: {message:?}"
    );
}

fn assert_live_unicode_tool_result_response(
    test: &str,
    message: &AssistantMessage,
    case: LiveUnicodeToolResultCase,
) {
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{test} returned provider error for Unicode tool result: {:?}",
        message.error_message
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
    assert!(
        !message.content.is_empty(),
        "{test} returned no content for Unicode tool result: {message:?}"
    );
    if case == LiveUnicodeToolResultCase::LinkedIn {
        assert!(
            message
                .content
                .iter()
                .any(|content| matches!(content, AssistantContent::Text(_))),
            "{test} returned no text content for LinkedIn Unicode data: {message:?}"
        );
    }
}

fn assert_live_image_tool_result_response(
    test: &str,
    message: &AssistantMessage,
    case: LiveImageToolResultCase,
) {
    assert_eq!(
        message.stop_reason,
        StopReason::Stop,
        "{test} expected stop after image tool result, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_ascii_lowercase();
    assert!(
        !text.trim().is_empty(),
        "{test} returned no text content for image tool result: {message:?}"
    );
    assert!(
        text.contains("red"),
        "{test} did not describe the image color as red: {text:?}"
    );
    assert!(
        text.contains("circle"),
        "{test} did not describe the image shape as a circle: {text:?}"
    );
    if case == LiveImageToolResultCase::TextAndImage {
        assert!(
            text.contains("diameter") || text.contains("100") || text.contains("pixel"),
            "{test} did not mention text-derived diameter/pixel details: {text:?}"
        );
    }
}

fn assert_live_image_input_response(test: &str, message: &AssistantMessage) {
    assert_eq!(
        message.stop_reason,
        StopReason::Stop,
        "{test} expected stop after image input, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_ascii_lowercase();
    assert!(
        !text.trim().is_empty(),
        "{test} returned no text content for image input: {message:?}"
    );
    assert!(
        text.contains("red"),
        "{test} did not describe the image color as red: {text:?}"
    );
    assert!(
        text.contains("circle"),
        "{test} did not describe the image shape as a circle: {text:?}"
    );
}

fn assert_live_responses_tool_result_images_payload(test: &str, payload: &Value) {
    let input = payload
        .get("input")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{test} expected Responses payload input array: {payload:?}"));
    let output_index = input
        .iter()
        .position(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .unwrap_or_else(|| {
            panic!("{test} expected function_call_output item in payload: {payload:?}")
        });
    let output = input[output_index]
        .get("output")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!("{test} expected function_call_output output content array: {payload:?}")
        });
    let text_item = output.iter().find(|item| {
        item.get("type").and_then(Value::as_str) == Some("input_text")
            && item
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains(LIVE_RESPONSES_TOOL_IMAGE_TEXT))
    });
    assert!(
        text_item.is_some(),
        "{test} expected function_call_output input_text containing tool text: {payload:?}"
    );
    let image_item = output.iter().find(|item| {
        item.get("type").and_then(Value::as_str) == Some("input_image")
            && item
                .get("image_url")
                .and_then(Value::as_str)
                .is_some_and(|url| url.starts_with("data:image/png;base64,"))
    });
    assert!(
        image_item.is_some(),
        "{test} expected function_call_output input_image data URL: {payload:?}"
    );
    let later_user_messages = input[output_index + 1..]
        .iter()
        .filter(|item| item.get("role").and_then(Value::as_str) == Some("user"))
        .count();
    assert_eq!(
        later_user_messages, 0,
        "{test} moved tool-result image blocks into later user messages: {payload:?}"
    );
}

fn assert_number_arg(test: &str, tool_call: &ToolCall, name: &str, expected: f64) {
    let actual = tool_call
        .arguments
        .get(name)
        .and_then(Value::as_f64)
        .unwrap_or_else(|| {
            panic!(
                "{test} returned missing/non-numeric {name} argument: {:?}",
                tool_call.arguments
            )
        });
    assert!(
        (actual - expected).abs() < f64::EPSILON,
        "{test} expected {name}={expected}, got {actual}: {:?}",
        tool_call.arguments
    );
}

async fn run_live_text_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let message = complete_simple(&model, live_context(), live_text_options(Some(api_key))).await?;
    assert_live_text_response(test, &message);
    Ok(())
}

async fn run_live_opencode_zen_catalog_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "OPENCODE_API_KEY") else {
        return Ok(());
    };

    for provider in ["opencode", "opencode-go"] {
        let models = get_models(provider);
        assert!(!models.is_empty(), "{test} found no models for {provider}");

        for model in models {
            let case = format!("{test} {provider}/{}", model.id);
            let mut options = live_text_options(Some(api_key.clone()));
            options.stream.max_tokens = Some(64);
            let message = complete_simple(
                &model,
                Context {
                    messages: vec![Message::User(UserMessage::text("Say hello."))],
                    ..Default::default()
                },
                options,
            )
            .await?;

            assert_live_text_response(&case, &message);
            assert_eq!(
                message.stop_reason,
                StopReason::Stop,
                "{case} expected stop after OpenCode Zen catalog smoke: {message:?}"
            );
        }
    }

    Ok(())
}

async fn run_live_basic_text_generation_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let mut context = Context {
        system_prompt: Some("You are a helpful assistant. Be concise.".to_owned()),
        messages: vec![Message::User(UserMessage::text(
            "Reply with exactly: 'Hello test successful'",
        ))],
        ..Default::default()
    };
    let first = complete_simple(model, context.clone(), options.clone()).await?;
    assert_live_contains_text(test, &first, "Hello test successful");

    context.messages.push(Message::Assistant(first));
    context.messages.push(Message::User(UserMessage::text(
        "Now say 'Goodbye test successful'",
    )));
    let second = complete_simple(model, context, options).await?;
    assert_live_contains_text(test, &second, "Goodbye test successful");
    Ok(())
}

async fn run_live_basic_text_generation_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_basic_text_generation_with_options(test, &model, live_text_options(Some(api_key)))
        .await
}

async fn run_live_anthropic_opus_47_reasoning_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    let model = get_model("anthropic", "claude-opus-4-7")
        .ok_or_else(|| "missing model registry entry: anthropic/claude-opus-4-7")?;
    let captured = Arc::new(Mutex::new(None));
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    options.stream.max_tokens = Some(1_024);
    options
        .payload_hooks
        .push(Arc::new(CaptureProviderPayloadHook {
            captured: Arc::clone(&captured),
        }));

    let mut stream = stream_simple(&model, live_anthropic_opus_47_reasoning_context(), options)?;
    let mut saw_thinking_event = false;
    while let Some(event) = stream.next().await {
        if matches!(
            event,
            AssistantMessageEvent::ThinkingStart { .. }
                | AssistantMessageEvent::ThinkingDelta { .. }
                | AssistantMessageEvent::ThinkingEnd { .. }
        ) {
            saw_thinking_event = true;
        }
    }
    let message = stream.result().await;
    let payload = captured
        .lock()
        .map_err(|_| "payload capture lock poisoned".to_owned())?
        .clone()
        .ok_or_else(|| format!("{test} did not capture a provider payload"))?;

    assert_live_anthropic_opus_47_payload(test, &payload);
    assert_live_anthropic_opus_47_response(test, &message, saw_thinking_event);
    Ok(())
}

async fn run_live_thinking_disable_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
    max_tokens: u64,
    temperature: Option<f64>,
    min_pongs: usize,
    max_output_tokens: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    options.stream.max_tokens = Some(max_tokens);
    options.stream.temperature = temperature;
    let mut stream = stream_simple(model, live_thinking_disable_context(), options)?;
    let mut saw_thinking_event = false;
    let mut thinking_char_count = 0usize;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ThinkingStart { .. }
            | AssistantMessageEvent::ThinkingEnd { .. } => {
                saw_thinking_event = true;
            }
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                saw_thinking_event = true;
                thinking_char_count += delta.chars().count();
            }
            _ => {}
        }
    }
    let message = stream.result().await;
    assert_live_thinking_disabled_response(
        test,
        &message,
        saw_thinking_event,
        thinking_char_count,
        min_pongs,
        max_output_tokens,
    );
    Ok(())
}

async fn run_live_thinking_disable_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    max_tokens: u64,
    temperature: Option<f64>,
    min_pongs: usize,
    max_output_tokens: Option<u64>,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_thinking_disable_with_options(
        test,
        &model,
        live_text_options(Some(api_key)),
        max_tokens,
        temperature,
        min_pongs,
        max_output_tokens,
    )
    .await
}

fn live_oauth_model(
    provider: &str,
    model_id: &str,
    resolution: &OAuthApiKeyResolution,
) -> Result<Model, Box<dyn Error>> {
    let mut model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    if provider == "github-copilot" {
        let enterprise_domain = resolution
            .credentials
            .as_ref()
            .and_then(StoredOAuthCredentials::enterprise_domain)
            .map(str::to_owned);
        model.base_url =
            github_copilot_base_url(Some(&resolution.api_key), enterprise_domain.as_deref());
    }
    Ok(model)
}

fn live_anthropic_messages_models(provider: &str) -> Vec<Model> {
    get_models(provider)
        .into_iter()
        .filter(|model| model.api == "anthropic-messages")
        .collect()
}

fn live_anthropic_messages_cases() -> Vec<LiveAnthropicMessagesE2ECase> {
    let mut cases = get_providers()
        .into_iter()
        .flat_map(|provider| {
            live_anthropic_messages_models(&provider)
                .into_iter()
                .map(move |model| LiveAnthropicMessagesE2ECase {
                    name: format!("{provider}/{}", model.id),
                    provider: provider.clone(),
                    model,
                })
        })
        .collect::<Vec<_>>();
    cases.sort_by(|a, b| a.name.cmp(&b.name));
    cases
}

fn live_anthropic_messages_probe_priority(model: &Model) -> f64 {
    let model_id = model.id.to_ascii_lowercase();
    let mut priority = model.cost.input + model.cost.output;

    if model_id.contains("haiku") && (model_id.contains("4-5") || model_id.contains("4.5")) {
        priority -= 1000.0;
    } else if model_id.contains("sonnet") && (model_id.contains("4-") || model_id.contains("4.")) {
        priority -= 750.0;
    } else if model_id.contains("claude") && (model_id.contains("4-") || model_id.contains("4.")) {
        priority -= 500.0;
    }

    priority
}

fn live_select_one_anthropic_messages_case_per_provider(
    cases: Vec<LiveAnthropicMessagesE2ECase>,
) -> Vec<LiveAnthropicMessagesE2ECase> {
    let mut by_provider: BTreeMap<String, Vec<LiveAnthropicMessagesE2ECase>> = BTreeMap::new();
    for case in cases {
        by_provider
            .entry(case.provider.clone())
            .or_default()
            .push(case);
    }

    by_provider
        .into_values()
        .filter_map(|mut provider_cases| {
            provider_cases.sort_by(|a, b| {
                live_anthropic_messages_probe_priority(&a.model)
                    .partial_cmp(&live_anthropic_messages_probe_priority(&b.model))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.model.id.cmp(&b.model.id))
            });
            provider_cases.into_iter().next()
        })
        .collect()
}

fn live_model_compat_bool(model: &Model, key: &str) -> Option<bool> {
    model
        .compat
        .as_ref()
        .and_then(|compat| compat.get(key))
        .and_then(Value::as_bool)
}

fn live_with_model_compat_bool(model: &Model, key: &str, value: bool) -> Model {
    let mut model = model.clone();
    let mut compat = model.compat.clone().unwrap_or_else(|| json!({}));
    if !compat.is_object() {
        compat = json!({});
    }
    compat
        .as_object_mut()
        .expect("compat object")
        .insert(key.to_owned(), Value::Bool(value));
    model.compat = Some(compat);
    model
}

fn live_selected_anthropic_messages_case(
    provider: &str,
    exclude_eager_disabled: bool,
) -> Option<LiveAnthropicMessagesE2ECase> {
    let cases = live_anthropic_messages_cases()
        .into_iter()
        .filter(|case| case.provider == provider)
        .filter(|case| {
            !exclude_eager_disabled
                || live_model_compat_bool(&case.model, "supportsEagerToolInputStreaming")
                    != Some(false)
        })
        .collect::<Vec<_>>();
    live_select_one_anthropic_messages_case_per_provider(cases)
        .into_iter()
        .next()
}

async fn live_anthropic_messages_case_options(
    test: &str,
    model: &Model,
) -> Result<Option<(Model, SimpleStreamOptions)>, Box<dyn Error>> {
    if model.provider == "github-copilot" {
        let Some(resolution) = live_oauth_resolution(test, "github-copilot").await else {
            return Ok(None);
        };
        let model = live_oauth_model("github-copilot", &model.id, &resolution)?;
        return Ok(Some((model, live_text_options(Some(resolution.api_key)))));
    }

    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return Ok(None);
    }

    if model.provider == "anthropic" {
        let Some(api_key) = live_api_key(test, "ANTHROPIC_API_KEY") else {
            return Ok(None);
        };
        return Ok(Some((model.clone(), live_text_options(Some(api_key)))));
    }

    if model.provider == "cloudflare-ai-gateway" {
        let Some(api_key) = live_env("CLOUDFLARE_API_KEY") else {
            skip_live(test, "CLOUDFLARE_API_KEY is not set");
            return Ok(None);
        };
        let Some(options) = live_cloudflare_options(test, api_key, true) else {
            return Ok(None);
        };
        return Ok(Some((model.clone(), options)));
    }

    let Some(api_key) = get_env_api_key(&model.provider) else {
        skip_live(
            test,
            format!("no API key is configured for provider {}", model.provider),
        );
        return Ok(None);
    };

    Ok(Some((model.clone(), live_text_options(Some(api_key)))))
}

fn live_anthropic_eager_tool_context() -> Context {
    Context {
        system_prompt: Some("You are a concise assistant. Use tools when useful.".to_owned()),
        messages: vec![Message::User(UserMessage::text(
            "Call echo_value with value set to eager-input-streaming-compat.",
        ))],
        tools: vec![live_string_arg_tool(
            "echo_value",
            "Echo a string value",
            "value",
            "The value to echo",
        )],
    }
}

fn live_anthropic_long_cache_retention_context() -> Context {
    Context {
        system_prompt: Some("You are a concise assistant.".to_owned()),
        messages: vec![Message::User(UserMessage::text(
            "Reply with exactly: long cache retention accepted",
        ))],
        ..Default::default()
    }
}

fn assert_live_anthropic_e2e_accepted(test: &str, message: &AssistantMessage) {
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{test} returned provider error: {:?}",
        message.error_message
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
}

async fn run_live_anthropic_messages_eager_tool_input_probe(
    test: &str,
    provider: &str,
    force_eager_input_streaming: bool,
) -> Result<(), Box<dyn Error>> {
    let Some(case) = live_selected_anthropic_messages_case(provider, force_eager_input_streaming)
    else {
        skip_live(test, format!("no anthropic-messages model for {provider}"));
        return Ok(());
    };
    let Some((mut model, mut options)) =
        live_anthropic_messages_case_options(test, &case.model).await?
    else {
        return Ok(());
    };
    if force_eager_input_streaming {
        model = live_with_model_compat_bool(&model, "supportsEagerToolInputStreaming", true);
    }
    options.stream.max_tokens = Some(128);
    options.reasoning = Some(ThinkingLevel::Off);

    let message = complete_simple(&model, live_anthropic_eager_tool_context(), options).await?;
    assert_live_anthropic_e2e_accepted(test, &message);
    Ok(())
}

async fn run_live_anthropic_messages_long_cache_retention_probe(
    test: &str,
    provider: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(case) = live_selected_anthropic_messages_case(provider, false) else {
        skip_live(test, format!("no anthropic-messages model for {provider}"));
        return Ok(());
    };
    let Some((model, mut options)) =
        live_anthropic_messages_case_options(test, &case.model).await?
    else {
        return Ok(());
    };
    let model = live_with_model_compat_bool(&model, "supportsLongCacheRetention", true);
    options.stream.cache_retention = Some(CacheRetention::Long);
    options.stream.max_tokens = Some(128);
    options.reasoning = Some(ThinkingLevel::Off);

    let message = complete_simple(
        &model,
        live_anthropic_long_cache_retention_context(),
        options,
    )
    .await?;
    assert_live_anthropic_e2e_accepted(test, &message);
    Ok(())
}

fn live_double_number_tool() -> Tool {
    Tool {
        name: "double_number".to_owned(),
        description: "Doubles a number and returns the result".to_owned(),
        parameters: json!({
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "A number to double"
                }
            },
            "required": ["value"]
        }),
    }
}

fn live_double_number_user_message() -> Message {
    Message::User(UserMessage::text(
        "Use the double_number tool to double 21.",
    ))
}

fn live_openai_responses_high_reasoning_options(api_key: String) -> SimpleStreamOptions {
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    options.stream.max_tokens = Some(1_024);
    options
}

fn live_anthropic_high_reasoning_options(api_key: String) -> SimpleStreamOptions {
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    options.thinking_budgets = Some(ThinkingBudgets {
        high: Some(5_000),
        ..Default::default()
    });
    options.stream.max_tokens = Some(6_000);
    options
}

fn live_high_interleaved_thinking_options(api_key: Option<String>) -> SimpleStreamOptions {
    let mut options = live_text_options(api_key);
    options.reasoning = Some(ThinkingLevel::High);
    options.thinking_budgets = Some(ThinkingBudgets {
        high: Some(5_000),
        ..Default::default()
    });
    options.stream.max_tokens = Some(6_000);
    options
}

fn live_openai_reasoning_replay_first_context(system_prompt: &str, user: Message) -> Context {
    Context {
        system_prompt: Some(system_prompt.to_owned()),
        messages: vec![user],
        tools: vec![live_double_number_tool()],
    }
}

fn live_openai_reasoning_replay_followup_context(
    system_prompt: &str,
    messages: Vec<Message>,
) -> Context {
    Context {
        system_prompt: Some(system_prompt.to_owned()),
        messages,
        tools: vec![live_double_number_tool()],
    }
}

fn live_tool_call_from_response(
    test: &str,
    message: &AssistantMessage,
    provider_label: &str,
) -> Result<ToolCall, Box<dyn Error>> {
    assert_live_anthropic_e2e_accepted(test, message);
    message
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .ok_or_else(|| {
            format!("{test} missing tool call from {provider_label}: {message:?}").into()
        })
}

fn live_tool_result_for_call(tool_call: &ToolCall) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        content: vec![ToolResultContent::text("42")],
        details: None,
        is_error: false,
        timestamp: now_millis(),
    })
}

fn live_prefilled_long_pipe_tool_call_context() -> Context {
    let mut arguments = serde_json::Map::new();
    arguments.insert("message".to_owned(), json!("hello"));
    let assistant = AssistantMessage {
        content: vec![AssistantContent::ToolCall(ToolCall {
            id: LIVE_LONG_PIPE_TOOL_CALL_ID.to_owned(),
            name: "echo".to_owned(),
            arguments,
            thought_signature: None,
        })],
        api: "openai-responses".to_owned(),
        provider: "github-copilot".to_owned(),
        model: "gpt-5.2-codex".to_owned(),
        response_model: None,
        response_id: None,
        diagnostics: Vec::new(),
        usage: Usage::zero(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: now_millis(),
    };

    Context {
        system_prompt: Some("You are a helpful assistant.".to_owned()),
        messages: vec![
            Message::User(UserMessage::text("Use the echo tool to echo 'hello'.")),
            Message::Assistant(assistant),
            Message::ToolResult(ToolResultMessage {
                tool_call_id: LIVE_LONG_PIPE_TOOL_CALL_ID.to_owned(),
                tool_name: "echo".to_owned(),
                content: vec![ToolResultContent::text("hello")],
                details: None,
                is_error: false,
                timestamp: now_millis(),
            }),
            Message::User(UserMessage::text("Say hi.")),
        ],
        tools: vec![live_echo_tool()],
    }
}

async fn run_live_prefilled_long_pipe_tool_call_id_smoke(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let message =
        complete_simple(model, live_prefilled_long_pipe_tool_call_context(), options).await?;
    assert_live_non_error_with_content(test, &message);
    let error_message = message.error_message.as_deref().unwrap_or("");
    assert!(
        !error_message.contains("call_id")
            && !error_message.contains("too long")
            && !error_message.contains("additional characters"),
        "{test} returned tool-call-id normalization error: {message:?}"
    );
    Ok(())
}

async fn live_github_copilot_pipe_tool_call_fixture(
    test: &str,
    echoed: &str,
) -> Result<Option<(Message, AssistantMessage, Message)>, Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, "github-copilot").await else {
        return Ok(None);
    };
    let model = live_oauth_model("github-copilot", "gpt-5.2-codex", &resolution)?;
    let user = Message::User(UserMessage::text(format!(
        "Use the echo tool to echo '{echoed}'"
    )));
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.max_tokens = Some(1_024);
    let assistant = complete_simple(
        &model,
        Context {
            system_prompt: Some(
                "You are a helpful assistant. Use the echo tool when asked.".to_owned(),
            ),
            messages: vec![user.clone()],
            tools: vec![live_echo_tool()],
        },
        options,
    )
    .await?;
    assert_eq!(
        assistant.stop_reason,
        StopReason::ToolUse,
        "{test} expected GitHub Copilot to return a tool call: {assistant:?}"
    );
    let tool_call = live_tool_call_from_response(test, &assistant, "GitHub Copilot")?;
    assert!(
        tool_call.id.contains('|'),
        "{test} expected GitHub Copilot Responses tool-call id to contain a pipe: {:?}",
        tool_call.id
    );
    let tool_result = Message::ToolResult(ToolResultMessage {
        tool_call_id: tool_call.id,
        tool_name: tool_call.name,
        content: vec![ToolResultContent::text(echoed)],
        details: None,
        is_error: false,
        timestamp: now_millis(),
    });
    Ok(Some((user, assistant, tool_result)))
}

async fn run_live_generated_pipe_tool_call_id_handoff_smoke(
    test: &str,
    target_model: &Model,
    target_options: SimpleStreamOptions,
    echoed: &str,
) -> Result<(), Box<dyn Error>> {
    let Some((user, assistant, tool_result)) =
        live_github_copilot_pipe_tool_call_fixture(test, echoed).await?
    else {
        return Ok(());
    };
    let message = complete_simple(
        target_model,
        Context {
            system_prompt: Some("You are a helpful assistant.".to_owned()),
            messages: vec![
                user,
                Message::Assistant(assistant),
                tool_result,
                Message::User(UserMessage::text("Say hi.")),
            ],
            tools: vec![live_echo_tool()],
        },
        target_options,
    )
    .await?;
    assert_live_non_error_with_content(test, &message);
    let error_message = message.error_message.as_deref().unwrap_or("");
    assert!(
        !error_message.contains("call_id")
            && !error_message.contains("too long")
            && !error_message.contains("id")
            && !error_message.contains("additional characters"),
        "{test} returned tool-call-id normalization error: {message:?}"
    );
    Ok(())
}

fn assert_live_non_error_with_content(test: &str, message: &AssistantMessage) {
    assert_live_anthropic_e2e_accepted(test, message);
    assert!(
        !message.content.is_empty(),
        "{test} returned no content: {message:?}"
    );
}

fn live_capture_payload_hook() -> (Arc<Mutex<Option<Value>>>, Arc<dyn ProviderPayloadHook>) {
    let captured = Arc::new(Mutex::new(None));
    (
        Arc::clone(&captured),
        Arc::new(CaptureProviderPayloadHook {
            captured: Arc::clone(&captured),
        }),
    )
}

fn live_captured_payload(
    test: &str,
    captured: &Arc<Mutex<Option<Value>>>,
) -> Result<Value, Box<dyn Error>> {
    captured
        .lock()
        .map_err(|_| "payload capture lock poisoned".to_owned())?
        .clone()
        .ok_or_else(|| format!("{test} did not capture a provider payload").into())
}

fn assert_live_openai_responses_payload_omits_reasoning(test: &str, payload: &Value) {
    let input = payload
        .get("input")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{test} expected Responses payload input array: {payload:?}"));
    assert!(
        input
            .iter()
            .all(|item| item.get("type").and_then(Value::as_str) != Some("reasoning")),
        "{test} replayed a reasoning item: {payload:?}"
    );
}

fn assert_live_openai_responses_handoff_payload(test: &str, payload: &Value) {
    assert_live_openai_responses_payload_omits_reasoning(test, payload);
    let input = payload
        .get("input")
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("{test} expected Responses payload input array: {payload:?}"));
    let function_call = input
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
        .unwrap_or_else(|| panic!("{test} expected function_call item: {payload:?}"));
    assert!(
        function_call.get("id").is_none() || function_call.get("id") == Some(&Value::Null),
        "{test} should omit stale function_call item id: {payload:?}"
    );
    let tool_result = input
        .iter()
        .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
        .unwrap_or_else(|| panic!("{test} expected function_call_output item: {payload:?}"));
    assert_eq!(
        tool_result.get("output").and_then(Value::as_str),
        Some("42"),
        "{test} did not preserve tool result output: {payload:?}"
    );
}

async fn run_live_oauth_text_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    let message = complete_simple(
        &model,
        live_context(),
        live_text_options(Some(resolution.api_key)),
    )
    .await?;
    assert_live_text_response(test, &message);
    Ok(())
}

async fn live_oauth_model_options(
    test: &str,
    provider: &str,
    model_id: &str,
    transport: Option<Transport>,
) -> Result<Option<(Model, SimpleStreamOptions)>, Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(None);
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.transport = transport;
    Ok(Some((model, options)))
}

async fn run_live_oauth_basic_text_generation_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some((model, options)) =
        live_oauth_model_options(test, provider, model_id, transport).await?
    else {
        return Ok(());
    };
    run_live_basic_text_generation_with_options(test, &model, options).await
}

async fn run_live_oauth_tool_call_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some((model, options)) =
        live_oauth_model_options(test, provider, model_id, transport).await?
    else {
        return Ok(());
    };
    run_live_tool_call_with_options(test, &model, options).await
}

async fn run_live_oauth_streaming_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some((model, options)) =
        live_oauth_model_options(test, provider, model_id, transport).await?
    else {
        return Ok(());
    };
    run_live_streaming_with_options(test, &model, options).await
}

async fn run_live_oauth_reasoning_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    reasoning: ThinkingLevel,
    max_tokens: Option<u64>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some((model, mut options)) =
        live_oauth_model_options(test, provider, model_id, transport).await?
    else {
        return Ok(());
    };
    options.thinking_budgets = thinking_budgets;
    if let Some(max_tokens) = max_tokens {
        options.stream.max_tokens = Some(max_tokens);
    }
    run_live_reasoning_stream_with_options(test, &model, options, reasoning).await
}

async fn run_live_oauth_tool_followup_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    reasoning: Option<ThinkingLevel>,
    max_tokens: Option<u64>,
    thinking_budgets: Option<ThinkingBudgets>,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some((model, mut options)) =
        live_oauth_model_options(test, provider, model_id, transport).await?
    else {
        return Ok(());
    };
    options.reasoning = reasoning;
    options.thinking_budgets = thinking_budgets;
    if let Some(max_tokens) = max_tokens {
        options.stream.max_tokens = Some(max_tokens);
    }
    run_live_tool_followup_with_options(test, &model, options).await
}

async fn run_live_anthropic_oauth_tool_name_normalization_smoke(
    test: &str,
    tool: Tool,
    system_prompt: &str,
    user_prompt: &str,
    expected_tool_name: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, "anthropic").await else {
        return Ok(());
    };
    let model = live_oauth_model("anthropic", "claude-sonnet-4-6", &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.max_tokens = Some(128);
    let message = complete_simple(
        &model,
        live_tool_name_normalization_context(tool, system_prompt, user_prompt),
        options,
    )
    .await?;
    assert_live_named_tool_call(test, &message, expected_tool_name);
    Ok(())
}

fn assert_live_contains_text(test: &str, message: &AssistantMessage, expected_text: &str) {
    assert_ne!(
        message.stop_reason,
        StopReason::Error,
        "{test} returned provider error: {:?}",
        message.error_message
    );
    assert!(
        message
            .error_message
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty(),
        "{test} returned unexpected error message: {message:?}"
    );
    let text = message
        .content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert!(
        text.contains(expected_text),
        "{test} expected response text to contain {expected_text:?}, got {text:?}: {message:?}"
    );
}

async fn run_live_cache_affinity_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    session_id: &str,
    prompt: &str,
    expected_text: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let mut options = live_text_options(Some(api_key));
    options.stream.session_id = Some(session_id.to_owned());
    options.stream.max_tokens = Some(64);
    let message = complete_simple(
        &model,
        Context {
            system_prompt: Some(
                "You are a helpful assistant. Reply exactly as requested.".to_owned(),
            ),
            messages: vec![Message::User(UserMessage::text(prompt))],
            ..Default::default()
        },
        options,
    )
    .await?;
    assert_live_contains_text(test, &message, expected_text);
    Ok(())
}

async fn run_live_oauth_cache_affinity_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    session_id: &str,
    prompt: &str,
    expected_text: &str,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.session_id = Some(session_id.to_owned());
    options.stream.transport = transport;
    options.stream.max_tokens = Some(64);
    let message = complete_simple(
        &model,
        Context {
            system_prompt: Some(
                "You are a helpful assistant. Reply exactly as requested.".to_owned(),
            ),
            messages: vec![Message::User(UserMessage::text(prompt))],
            ..Default::default()
        },
        options,
    )
    .await?;
    assert_live_contains_text(test, &message, expected_text);
    Ok(())
}

async fn run_live_tool_call_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let mut stream = stream_simple(model, live_tool_call_context(), options)?;
    let mut saw_start = false;
    let mut saw_delta = false;
    let mut saw_end = false;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ToolcallStart { .. } => {
                saw_start = true;
            }
            AssistantMessageEvent::ToolcallDelta { .. } => {
                saw_delta = true;
            }
            AssistantMessageEvent::ToolcallEnd { tool_call, .. } => {
                saw_end = true;
                assert_valid_live_math_tool_call(test, &tool_call);
            }
            _ => {}
        }
    }
    let message = stream.result().await;
    assert_live_tool_call(test, &message);
    assert!(saw_start, "{test} completed without toolcall_start");
    assert!(saw_delta, "{test} completed without toolcall_delta");
    assert!(saw_end, "{test} completed without toolcall_end");
    Ok(())
}

async fn run_live_tool_followup_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let mut context = live_tool_call_context();
    let first = complete_simple(model, context.clone(), options.clone()).await?;
    assert_live_tool_call(test, &first);
    let tool_call = first
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .ok_or_else(|| format!("{test} first turn returned no tool call: {first:?}"))?;
    let result = evaluate_live_interleaved_calculator_call(&tool_call)?;
    let result_text = if result.fract().abs() < f64::EPSILON {
        format!("{}", result as i64)
    } else {
        result.to_string()
    };

    context.messages.push(Message::Assistant(first));
    context
        .messages
        .push(Message::ToolResult(ToolResultMessage {
            tool_call_id: tool_call.id,
            tool_name: tool_call.name,
            content: vec![ToolResultContent::text(result_text.clone())],
            details: None,
            is_error: false,
            timestamp: now_millis(),
        }));
    context
        .messages
        .push(Message::User(UserMessage::text(format!(
            "Use the tool result to answer. Include {result_text} in the final answer."
        ))));

    let second = complete_simple(model, context, options).await?;
    assert_live_contains_text(test, &second, &result_text);
    Ok(())
}

async fn run_live_tool_followup_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    reasoning: Option<ThinkingLevel>,
    max_tokens: Option<u64>,
    thinking_budgets: Option<ThinkingBudgets>,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = reasoning;
    options.thinking_budgets = thinking_budgets;
    if let Some(max_tokens) = max_tokens {
        options.stream.max_tokens = Some(max_tokens);
    }
    run_live_tool_followup_with_options(test, &model, options).await
}

async fn run_live_tool_call_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_tool_call_with_options(test, &model, live_text_options(Some(api_key))).await
}

async fn run_live_interleaved_thinking_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let first = complete_simple(
        model,
        live_interleaved_thinking_first_context(),
        options.clone(),
    )
    .await?;
    let tool_call = live_interleaved_tool_call_from_response(test, &first)?;
    let second_context = live_interleaved_thinking_second_context(first, tool_call)?;
    let second = complete_simple(model, second_context, options).await?;
    assert_live_interleaved_final_response(test, &second);
    Ok(())
}

async fn run_live_interleaved_thinking_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_interleaved_thinking_with_options(
        test,
        &model,
        live_high_interleaved_thinking_options(Some(api_key)),
    )
    .await
}

async fn run_live_interleaved_thinking_bedrock_smoke(
    test: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    if !live_network_enabled(test) {
        return Ok(());
    }
    let model = get_model("amazon-bedrock", model_id)
        .ok_or_else(|| format!("missing model registry entry: amazon-bedrock/{model_id}"))?;
    run_live_interleaved_thinking_with_options(
        test,
        &model,
        live_high_interleaved_thinking_options(None),
    )
    .await
}

async fn run_live_context_overflow_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
    expectation: LiveContextOverflowExpectation,
) -> Result<(), Box<dyn Error>> {
    let message = complete_simple(model, live_context_overflow_context(model), options).await?;
    assert_live_context_overflow_response(test, model, &message, expectation);
    Ok(())
}

async fn run_live_context_overflow_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    expectation: LiveContextOverflowExpectation,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(Some(api_key)),
        expectation,
    )
    .await
}

async fn run_live_context_overflow_oauth_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    expectation: LiveContextOverflowExpectation,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(Some(resolution.api_key)),
        expectation,
    )
    .await
}

async fn run_live_context_overflow_openai_responses_smoke(
    test: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let mut model = get_model("openai", "gpt-4o").ok_or_else(|| "missing openai/gpt-4o")?;
    model.api = "openai-responses".to_owned();
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(Some(api_key)),
        LiveContextOverflowExpectation::Error,
    )
    .await
}

async fn run_live_context_overflow_azure_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_context_overflow_with_options(
        test,
        &model,
        {
            let mut options = options;
            options.stream.timeout_ms = Some(120_000);
            options
        },
        LiveContextOverflowExpectation::Error,
    )
    .await
}

async fn run_live_context_overflow_bedrock_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(None),
        LiveContextOverflowExpectation::Error,
    )
    .await
}

async fn run_live_tool_call_without_result_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let first = complete_simple(
        model,
        live_tool_call_without_result_first_context(),
        options.clone(),
    )
    .await?;
    assert_live_any_tool_call(test, &first);

    let second = complete_simple(
        model,
        live_tool_call_without_result_second_context(first),
        options,
    )
    .await?;
    assert_live_tool_call_without_result_response(test, &second);
    Ok(())
}

async fn run_live_tool_call_without_result_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_tool_call_without_result_with_options(test, &model, live_text_options(Some(api_key)))
        .await
}

async fn run_live_unicode_tool_result_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
    case: LiveUnicodeToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let message = complete_simple(
        model,
        live_unicode_tool_result_context(model, case),
        options,
    )
    .await?;
    assert_live_unicode_tool_result_response(test, &message, case);
    Ok(())
}

async fn run_live_unicode_tool_result_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    case: LiveUnicodeToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_unicode_tool_result_with_options(test, &model, live_text_options(Some(api_key)), case)
        .await
}

async fn run_live_image_tool_result_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
    case: LiveImageToolResultCase,
) -> Result<(), Box<dyn Error>> {
    if !model.input.contains(&InputKind::Image) {
        skip_live(
            test,
            format!(
                "model {}/{} does not advertise image input support",
                model.provider, model.id
            ),
        );
        return Ok(());
    }

    options.stream.max_tokens = Some(96);
    let (expected_tool_name, _, _) = live_image_tool_result_tool_spec(case);
    let first = complete_simple(
        model,
        live_image_tool_result_first_context(case),
        options.clone(),
    )
    .await?;
    assert_eq!(
        first.stop_reason,
        StopReason::ToolUse,
        "{test} expected first response to call a tool, got {:?}: {first:?}",
        first.stop_reason
    );
    let tool_call = first
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .ok_or_else(|| format!("{test} returned no tool call: {first:?}"))?;
    assert_eq!(
        tool_call.name, expected_tool_name,
        "{test} returned unexpected tool call: {tool_call:?}"
    );

    let second = complete_simple(
        model,
        live_image_tool_result_second_context(first, tool_call, case),
        options,
    )
    .await?;
    assert_live_image_tool_result_response(test, &second, case);
    Ok(())
}

async fn run_live_image_tool_result_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    case: LiveImageToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_image_tool_result_with_options(test, &model, live_text_options(Some(api_key)), case)
        .await
}

async fn run_live_image_input_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    if !model.input.contains(&InputKind::Image) {
        skip_live(
            test,
            format!(
                "model {}/{} does not advertise image input support",
                model.provider, model.id
            ),
        );
        return Ok(());
    }

    options.stream.max_tokens = Some(options.stream.max_tokens.unwrap_or(96).max(96));
    let message = complete_simple(model, live_image_input_context(), options).await?;
    assert_live_image_input_response(test, &message);
    Ok(())
}

async fn run_live_image_input_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_image_input_with_options(test, &model, live_text_options(Some(api_key))).await
}

async fn run_live_responses_tool_result_images_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    if !model.input.contains(&InputKind::Image) {
        skip_live(
            test,
            format!(
                "model {}/{} does not advertise image input support",
                model.provider, model.id
            ),
        );
        return Ok(());
    }

    options.stream.max_tokens = Some(96);
    let first = complete_simple(
        model,
        live_responses_tool_result_images_first_context(),
        options.clone(),
    )
    .await?;
    assert_eq!(
        first.stop_reason,
        StopReason::ToolUse,
        "{test} expected first response to call a tool, got {:?}: {first:?}",
        first.stop_reason
    );
    let tool_call = first
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .ok_or_else(|| format!("{test} returned no tool call: {first:?}"))?;
    assert_eq!(
        tool_call.name, "get_circle_with_description",
        "{test} returned unexpected tool call: {tool_call:?}"
    );

    let captured = Arc::new(Mutex::new(None));
    options
        .payload_hooks
        .push(Arc::new(CaptureProviderPayloadHook {
            captured: Arc::clone(&captured),
        }));
    let second = complete_simple(
        model,
        live_responses_tool_result_images_second_context(first, tool_call),
        options,
    )
    .await?;
    assert_live_image_tool_result_response(test, &second, LiveImageToolResultCase::TextAndImage);

    let payload = captured
        .lock()
        .map_err(|_| "payload capture lock poisoned".to_owned())?
        .clone()
        .ok_or_else(|| format!("{test} did not capture a provider payload"))?;
    assert_live_responses_tool_result_images_payload(test, &payload);
    Ok(())
}

async fn run_live_responses_tool_result_images_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    reasoning: Option<ThinkingLevel>,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = reasoning;
    run_live_responses_tool_result_images_with_options(test, &model, options).await
}

async fn run_live_provider_error_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    if !live_network_enabled(test) {
        return Ok(());
    }
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let message = complete_simple(
        &model,
        live_context(),
        live_text_options(Some(LIVE_BOGUS_API_KEY.to_owned())),
    )
    .await?;
    assert_live_provider_error(test, &message);
    Ok(())
}

async fn run_live_immediate_abort_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
) -> Result<AssistantMessage, Box<dyn Error>> {
    options.stream.abort_flag = Some(Arc::new(AtomicBool::new(true)));
    let message = complete_simple(model, live_immediate_abort_context(), options).await?;
    assert_eq!(
        message.stop_reason,
        StopReason::Aborted,
        "{test} expected immediate abort, got {:?}: {message:?}",
        message.stop_reason
    );
    assert!(
        message.content.is_empty(),
        "{test} immediate abort should not preserve assistant content: {message:?}"
    );
    Ok(message)
}

async fn run_live_immediate_abort_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_immediate_abort_with_options(test, &model, live_text_options(Some(api_key)))
        .await
        .map(|_| ())
}

async fn run_live_immediate_abort_oauth_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_immediate_abort_with_options(test, &model, live_text_options(Some(resolution.api_key)))
        .await
        .map(|_| ())
}

async fn run_live_immediate_abort_azure_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_immediate_abort_with_options(test, &model, options)
        .await
        .map(|_| ())
}

async fn run_live_immediate_abort_bedrock_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_immediate_abort_with_options(test, &model, live_text_options(None))
        .await
        .map(|_| ())
}

async fn run_live_abort_then_new_message_bedrock_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    let aborted =
        run_live_immediate_abort_with_options(test, &model, live_text_options(None)).await?;
    let follow_up = complete_simple(
        &model,
        live_abort_then_new_message_context(aborted),
        live_text_options(None),
    )
    .await?;
    assert_live_text_response(test, &follow_up);
    Ok(())
}

async fn run_live_midstream_abort_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
) -> Result<AssistantMessage, Box<dyn Error>> {
    let abort_flag = options
        .stream
        .abort_flag
        .clone()
        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
    options.stream.abort_flag = Some(abort_flag.clone());
    let mut stream = stream_simple(model, live_abort_context(), options)?;
    let mut abort_fired = false;
    let mut delta_chars = 0usize;
    while let Some(event) = stream.next().await {
        if !abort_fired {
            match event {
                AssistantMessageEvent::TextDelta { delta, .. }
                | AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                    delta_chars += delta.chars().count();
                    if delta_chars >= LIVE_ABORT_MIN_DELTA_CHARS {
                        abort_fired = true;
                        abort_flag.store(true, Ordering::SeqCst);
                    }
                }
                _ => {}
            }
        }
    }
    let message = stream.result().await;
    assert!(
        abort_fired,
        "{test} completed without text/thinking deltas to abort: {message:?}"
    );
    assert!(
        !message.content.is_empty(),
        "{test} aborted stream should preserve partial assistant content: {message:?}"
    );
    assert_eq!(
        message.stop_reason,
        StopReason::Aborted,
        "{test} expected an aborted stream, got {:?}: {:?}",
        message.stop_reason,
        message.error_message
    );
    Ok(message)
}

async fn run_live_abort_tokens_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
    expectation: LiveAbortUsageExpectation,
) -> Result<AssistantMessage, Box<dyn Error>> {
    let message = run_live_midstream_abort_with_options(test, model, options).await?;
    assert_live_abort_usage(test, &message, expectation);
    Ok(message)
}

async fn run_live_abort_tokens_then_new_message_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
    expectation: LiveAbortUsageExpectation,
) -> Result<(), Box<dyn Error>> {
    let mut follow_up_options = options.clone();
    follow_up_options.stream.abort_flag = None;
    let aborted = run_live_abort_tokens_with_options(test, model, options, expectation).await?;
    let follow_up = complete_simple(
        model,
        live_midstream_abort_then_new_message_context(aborted),
        follow_up_options,
    )
    .await?;
    assert_live_text_response(test, &follow_up);
    Ok(())
}

async fn run_live_midstream_abort_then_new_message_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let mut follow_up_options = options.clone();
    follow_up_options.stream.abort_flag = None;
    let aborted = run_live_midstream_abort_with_options(test, model, options).await?;
    let follow_up = complete_simple(
        model,
        live_midstream_abort_then_new_message_context(aborted),
        follow_up_options,
    )
    .await?;
    assert_live_text_response(test, &follow_up);
    Ok(())
}

async fn run_live_streaming_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let mut stream = stream_simple(model, live_context(), options)?;
    let mut saw_delta = false;
    while let Some(event) = stream.next().await {
        if matches!(
            event,
            AssistantMessageEvent::TextDelta { .. } | AssistantMessageEvent::ThinkingDelta { .. }
        ) {
            saw_delta = true;
        }
    }
    let message = stream.result().await;
    assert_live_text_response(test, &message);
    assert!(saw_delta, "{test} completed without streamed deltas");
    Ok(())
}

async fn run_live_reasoning_stream_with_options(
    test: &str,
    model: &Model,
    mut options: SimpleStreamOptions,
    reasoning: ThinkingLevel,
) -> Result<(), Box<dyn Error>> {
    options.reasoning = Some(reasoning);
    options.stream.max_tokens = Some(options.stream.max_tokens.unwrap_or(512).max(512));
    let mut stream = stream_simple(model, live_reasoning_context(), options)?;
    let mut saw_start = false;
    let mut saw_delta = false;
    let mut saw_end = false;
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::ThinkingStart { .. } => saw_start = true,
            AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                if !delta.trim().is_empty() {
                    saw_delta = true;
                }
            }
            AssistantMessageEvent::ThinkingEnd { .. } => saw_end = true,
            _ => {}
        }
    }
    let message = stream.result().await;
    assert_live_text_response(test, &message);
    assert!(saw_start, "{test} completed without thinking_start");
    assert!(saw_delta, "{test} completed without thinking_delta");
    assert!(saw_end, "{test} completed without thinking_end");
    assert!(
        message
            .content
            .iter()
            .any(|content| matches!(content, AssistantContent::Thinking(_))),
        "{test} returned no thinking content block: {message:?}"
    );
    Ok(())
}

async fn run_live_reasoning_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    reasoning: ThinkingLevel,
    max_tokens: Option<u64>,
    thinking_budgets: Option<ThinkingBudgets>,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let mut options = live_text_options(Some(api_key));
    options.thinking_budgets = thinking_budgets;
    if let Some(max_tokens) = max_tokens {
        options.stream.max_tokens = Some(max_tokens);
    }
    run_live_reasoning_stream_with_options(test, &model, options, reasoning).await
}

async fn run_live_streaming_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_streaming_with_options(test, &model, live_text_options(Some(api_key))).await
}

async fn run_live_multiturn_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let message = complete_simple(model, live_multiturn_context(model), options).await?;
    assert_live_text_response(test, &message);
    Ok(())
}

async fn run_live_multiturn_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_multiturn_with_options(test, &model, live_text_options(Some(api_key))).await
}

async fn run_live_total_usage_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let first = complete_simple(model, live_usage_first_context(), options.clone()).await?;
    assert_live_text_response(test, &first);
    assert_live_usage_total_matches_components(test, &first.usage);

    let second = complete_simple(model, live_usage_second_context(first), options).await?;
    assert_live_text_response(test, &second);
    assert_live_usage_total_matches_components(test, &second.usage);
    Ok(())
}

async fn run_live_total_usage_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_total_usage_with_options(test, &model, live_text_options(Some(api_key))).await
}

async fn run_live_empty_with_options(
    test: &str,
    model: &Model,
    options: SimpleStreamOptions,
    case: LiveEmptyCase,
) -> Result<(), Box<dyn Error>> {
    let message = complete_simple(model, live_empty_case_context(model, case), options).await?;
    assert_live_empty_response(test, &message, case);
    Ok(())
}

async fn run_live_empty_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    case: LiveEmptyCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_empty_with_options(test, &model, live_text_options(Some(api_key)), case).await
}

async fn run_live_abort_tokens_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    expectation: LiveAbortUsageExpectation,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_abort_tokens_with_options(
        test,
        &model,
        live_abort_options(Some(api_key), Arc::new(AtomicBool::new(false))),
        expectation,
    )
    .await
    .map(|_| ())
}

async fn run_live_abort_tokens_then_new_message_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
    expectation: LiveAbortUsageExpectation,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        live_abort_options(Some(api_key), Arc::new(AtomicBool::new(false))),
        expectation,
    )
    .await
}

async fn run_live_midstream_abort_then_new_message_api_key_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_midstream_abort_then_new_message_with_options(
        test,
        &model,
        live_abort_options(Some(api_key), Arc::new(AtomicBool::new(false))),
    )
    .await
}

async fn run_live_response_id_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    api_key_env: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, api_key_env) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    let message = complete_simple(
        &model,
        live_response_id_context(),
        live_text_options(Some(api_key)),
    )
    .await?;
    assert_live_response_id(test, &message);
    Ok(())
}

async fn run_live_oauth_response_id_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    let message = complete_simple(
        &model,
        live_response_id_context(),
        live_text_options(Some(resolution.api_key)),
    )
    .await?;
    assert_live_response_id(test, &message);
    Ok(())
}

async fn run_live_oauth_empty_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    case: LiveEmptyCase,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_empty_with_options(
        test,
        &model,
        live_text_options(Some(resolution.api_key)),
        case,
    )
    .await
}

async fn run_live_oauth_tool_call_without_result_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_tool_call_without_result_with_options(
        test,
        &model,
        live_text_options(Some(resolution.api_key)),
    )
    .await
}

async fn run_live_oauth_unicode_tool_result_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    case: LiveUnicodeToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_unicode_tool_result_with_options(
        test,
        &model,
        live_text_options(Some(resolution.api_key)),
        case,
    )
    .await
}

async fn run_live_oauth_image_tool_result_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    case: LiveImageToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_image_tool_result_with_options(
        test,
        &model,
        live_text_options(Some(resolution.api_key)),
        case,
    )
    .await
}

async fn run_live_oauth_image_input_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    transport: Option<Transport>,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.transport = transport;
    run_live_image_input_with_options(test, &model, options).await
}

async fn run_live_oauth_responses_tool_result_images_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    reasoning: Option<ThinkingLevel>,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.reasoning = reasoning;
    run_live_responses_tool_result_images_with_options(test, &model, options).await
}

async fn run_live_oauth_abort_tokens_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    expectation: LiveAbortUsageExpectation,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_abort_tokens_with_options(
        test,
        &model,
        live_abort_options(Some(resolution.api_key), Arc::new(AtomicBool::new(false))),
        expectation,
    )
    .await
    .map(|_| ())
}

async fn run_live_oauth_abort_tokens_then_new_message_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    expectation: LiveAbortUsageExpectation,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        live_abort_options(Some(resolution.api_key), Arc::new(AtomicBool::new(false))),
        expectation,
    )
    .await
}

async fn run_live_oauth_total_usage_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(resolution) = live_oauth_resolution(test, provider).await else {
        return Ok(());
    };
    let model = live_oauth_model(provider, model_id, &resolution)?;
    run_live_total_usage_with_options(test, &model, live_text_options(Some(resolution.api_key)))
        .await
}

fn live_azure_options(test: &str, api_key: String) -> Option<SimpleStreamOptions> {
    if live_env("AZURE_OPENAI_BASE_URL").is_none()
        && live_env("AZURE_OPENAI_RESOURCE_NAME").is_none()
    {
        skip_live(
            test,
            "AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME is not set",
        );
        return None;
    }

    let mut options = live_text_options(Some(api_key));
    if let Some(deployment_name) = live_env("AZURE_OPENAI_DEPLOYMENT_NAME") {
        options
            .stream
            .extra
            .insert("azureDeploymentName".to_owned(), json!(deployment_name));
    }
    Some(options)
}

fn live_google_vertex_options(test: &str) -> Option<SimpleStreamOptions> {
    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return None;
    }
    if let Some(api_key) = live_env("GOOGLE_CLOUD_API_KEY") {
        return Some(live_text_options(Some(api_key)));
    }
    let project = live_env("GOOGLE_CLOUD_PROJECT").or_else(|| live_env("GCLOUD_PROJECT"));
    let location = live_env("GOOGLE_CLOUD_LOCATION");
    let Some(project) = project else {
        skip_live(
            test,
            "GOOGLE_CLOUD_API_KEY or GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT is not set",
        );
        return None;
    };
    let Some(location) = location else {
        skip_live(
            test,
            "GOOGLE_CLOUD_API_KEY or GOOGLE_CLOUD_LOCATION is not set",
        );
        return None;
    };
    let mut options = live_text_options(None);
    options
        .stream
        .extra
        .insert("project".to_owned(), json!(project));
    options
        .stream
        .extra
        .insert("location".to_owned(), json!(location));
    Some(options)
}

fn live_google_vertex_model() -> Result<Model, Box<dyn Error>> {
    get_model("google-vertex", "gemini-3-flash-preview")
        .ok_or_else(|| "missing model registry entry: google-vertex/gemini-3-flash-preview".into())
}

fn live_google_vertex_api_key_options(test: &str) -> Option<SimpleStreamOptions> {
    live_api_key(test, "GOOGLE_CLOUD_API_KEY").map(|api_key| live_text_options(Some(api_key)))
}

fn live_google_vertex_adc_credentials_configured() -> bool {
    if live_env("GOOGLE_OAUTH_ACCESS_TOKEN").is_some()
        || live_env("GOOGLE_ACCESS_TOKEN").is_some()
        || live_env("CLOUDSDK_AUTH_ACCESS_TOKEN").is_some()
        || live_env("GOOGLE_APPLICATION_CREDENTIALS").is_some()
    {
        return true;
    }

    let appdata_adc = env::var_os("APPDATA")
        .map(std::path::PathBuf::from)
        .map(|path| {
            path.join("gcloud")
                .join("application_default_credentials.json")
        })
        .is_some_and(|path| path.exists());
    let home_adc = env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|path| {
            path.join(".config")
                .join("gcloud")
                .join("application_default_credentials.json")
        })
        .is_some_and(|path| path.exists());
    appdata_adc || home_adc
}

fn live_google_vertex_adc_options(test: &str) -> Option<SimpleStreamOptions> {
    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return None;
    }
    let project = live_env("GOOGLE_CLOUD_PROJECT").or_else(|| live_env("GCLOUD_PROJECT"));
    let location = live_env("GOOGLE_CLOUD_LOCATION");
    let Some(project) = project else {
        skip_live(test, "GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT is not set");
        return None;
    };
    let Some(location) = location else {
        skip_live(test, "GOOGLE_CLOUD_LOCATION is not set");
        return None;
    };
    if !live_google_vertex_adc_credentials_configured() {
        skip_live(
            test,
            "Google ADC credentials are not configured for Vertex AI",
        );
        return None;
    }

    let mut options = live_text_options(Some(GCP_VERTEX_CREDENTIALS_MARKER.to_owned()));
    options
        .stream
        .extra
        .insert("project".to_owned(), json!(project));
    options
        .stream
        .extra
        .insert("location".to_owned(), json!(location));
    Some(options)
}

async fn run_live_google_vertex_text_with_options(
    test: &str,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let model = live_google_vertex_model()?;
    let message = complete_simple(&model, live_context(), options).await?;
    assert_live_text_response(test, &message);
    Ok(())
}

async fn run_live_google_vertex_response_id_with_options(
    test: &str,
    options: SimpleStreamOptions,
) -> Result<(), Box<dyn Error>> {
    let model = live_google_vertex_model()?;
    let message = complete_simple(&model, live_response_id_context(), options).await?;
    assert_live_response_id(test, &message);
    Ok(())
}

fn live_cloudflare_options(
    test: &str,
    api_key: String,
    requires_gateway: bool,
) -> Option<SimpleStreamOptions> {
    if live_env("CLOUDFLARE_ACCOUNT_ID").is_none() {
        skip_live(test, "CLOUDFLARE_ACCOUNT_ID is not set");
        return None;
    }
    if requires_gateway && live_env("CLOUDFLARE_GATEWAY_ID").is_none() {
        skip_live(test, "CLOUDFLARE_GATEWAY_ID is not set");
        return None;
    }
    Some(live_text_options(Some(api_key)))
}

fn live_cloudflare_gateway_byok_options(
    test: &str,
    upstream_api_key_env: &str,
) -> Option<SimpleStreamOptions> {
    let api_key = live_api_key(test, "CLOUDFLARE_API_KEY")?;
    let mut options = live_cloudflare_options(test, api_key, true)?;
    let Some(upstream_api_key) = live_env(upstream_api_key_env) else {
        skip_live(test, format!("{upstream_api_key_env} is not set"));
        return None;
    };
    options.stream.headers.insert(
        "Authorization".to_owned(),
        format!("Bearer {upstream_api_key}"),
    );
    Some(options)
}

fn live_cloudflare_gateway_model(model_id: &str) -> Result<Model, Box<dyn Error>> {
    get_model("cloudflare-ai-gateway", model_id).ok_or_else(|| {
        format!("missing model registry entry: cloudflare-ai-gateway/{model_id}").into()
    })
}

fn live_cloudflare_model_options(
    test: &str,
    provider: &str,
    model_id: &str,
    requires_gateway: bool,
) -> Result<Option<(Model, SimpleStreamOptions)>, Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(None);
    };
    let Some(options) = live_cloudflare_options(test, api_key, requires_gateway) else {
        return Ok(None);
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    Ok(Some((model, options)))
}

fn live_cross_provider_handoff_pairs() -> &'static [LiveCrossProviderHandoffPair] {
    &[
        LiveCrossProviderHandoffPair {
            provider: "anthropic",
            model_id: "claude-sonnet-4-5",
            label: "anthropic-claude-sonnet-4-5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "google",
            model_id: "gemini-3-flash-preview",
            label: "google-gemini-3-flash-preview",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "openai",
            model_id: "gpt-4o-mini",
            label: "openai-completions-gpt-4o-mini",
            api_override: Some("openai-completions"),
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "openai",
            model_id: "gpt-5-mini",
            label: "openai-responses-gpt-5-mini",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "azure-openai-responses",
            model_id: "gpt-4o-mini",
            label: "azure-openai-responses-gpt-4o-mini",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "openai-codex",
            model_id: "gpt-5.5",
            label: "openai-codex-gpt-5.5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "github-copilot",
            model_id: "claude-sonnet-4.5",
            label: "copilot-claude-sonnet-4.5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "github-copilot",
            model_id: "gpt-5.1-codex",
            label: "copilot-gpt-5.1-codex",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "github-copilot",
            model_id: "gemini-3-flash-preview",
            label: "copilot-gemini-3-flash-preview",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "github-copilot",
            model_id: "grok-code-fast-1",
            label: "copilot-grok-code-fast-1",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "amazon-bedrock",
            model_id: "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            label: "bedrock-claude-sonnet-4-5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "xai",
            model_id: "grok-code-fast-1",
            label: "xai-grok-code-fast-1",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "cerebras",
            model_id: "zai-glm-4.7",
            label: "cerebras-zai-glm-4.7",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "cloudflare-workers-ai",
            model_id: "@cf/moonshotai/kimi-k2.6",
            label: "cloudflare-kimi-k2.6",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "cloudflare-ai-gateway",
            model_id: "workers-ai/@cf/moonshotai/kimi-k2.6",
            label: "cloudflare-gateway-kimi-k2.6",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "cloudflare-ai-gateway",
            model_id: "claude-sonnet-4-5",
            label: "cloudflare-gateway-claude-sonnet-4-5",
            api_override: None,
            upstream_api_key_env: Some("ANTHROPIC_API_KEY"),
        },
        LiveCrossProviderHandoffPair {
            provider: "cloudflare-ai-gateway",
            model_id: "gpt-5.1",
            label: "cloudflare-gateway-gpt-5.1",
            api_override: None,
            upstream_api_key_env: Some("OPENAI_API_KEY"),
        },
        LiveCrossProviderHandoffPair {
            provider: "groq",
            model_id: "openai/gpt-oss-120b",
            label: "groq-gpt-oss-120b",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "huggingface",
            model_id: "moonshotai/Kimi-K2.5",
            label: "huggingface-kimi-k2.5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "together",
            model_id: "moonshotai/Kimi-K2.6",
            label: "together-kimi-k2.6",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "kimi-coding",
            model_id: "kimi-k2-thinking",
            label: "kimi-coding-k2-thinking",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "mistral",
            model_id: "devstral-medium-latest",
            label: "mistral-devstral-medium",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "minimax",
            model_id: "MiniMax-M2.7",
            label: "minimax-m2.7",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "minimax-cn",
            model_id: "MiniMax-M2.7",
            label: "minimax-m2.7",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode",
            model_id: "big-pickle",
            label: "zen-big-pickle",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode",
            model_id: "claude-sonnet-4-5",
            label: "zen-claude-sonnet-4-5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode",
            model_id: "gemini-3-flash",
            label: "zen-gemini-3-flash",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode",
            model_id: "glm-4.7-free",
            label: "zen-glm-4.7-free",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode",
            model_id: "gpt-5.2-codex",
            label: "zen-gpt-5.2-codex",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode",
            model_id: "minimax-m2.1-free",
            label: "zen-minimax-m2.1-free",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode-go",
            model_id: "kimi-k2.5",
            label: "go-kimi-k2.5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "opencode-go",
            model_id: "minimax-m2.5",
            label: "go-minimax-m2.5",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "xiaomi",
            model_id: "mimo-v2.5-pro",
            label: "xiaomi-mimo-v2.5-pro",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "xiaomi-token-plan-cn",
            model_id: "mimo-v2.5-pro",
            label: "xiaomi-token-plan-cn-mimo-v2.5-pro",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "xiaomi-token-plan-ams",
            model_id: "mimo-v2.5-pro",
            label: "xiaomi-token-plan-ams-mimo-v2.5-pro",
            api_override: None,
            upstream_api_key_env: None,
        },
        LiveCrossProviderHandoffPair {
            provider: "xiaomi-token-plan-sgp",
            model_id: "mimo-v2.5-pro",
            label: "xiaomi-token-plan-sgp-mimo-v2.5-pro",
            api_override: None,
            upstream_api_key_env: None,
        },
    ]
}

fn live_cross_provider_handoff_source_missing_model(pair: &LiveCrossProviderHandoffPair) -> bool {
    matches!(
        (pair.provider, pair.model_id),
        ("github-copilot", "gpt-5.1-codex")
            | ("opencode", "glm-4.7-free")
            | ("opencode", "minimax-m2.1-free")
    )
}

fn live_handoff_options_for_model(
    model: &Model,
    mut options: SimpleStreamOptions,
) -> SimpleStreamOptions {
    options.stream.max_tokens = Some(1_024);
    if model.reasoning {
        options.reasoning = Some(ThinkingLevel::High);
        if model.api == "anthropic-messages" {
            options.stream.max_tokens = Some(6_000);
            options.thinking_budgets = Some(ThinkingBudgets {
                high: Some(5_000),
                ..Default::default()
            });
        }
    }
    options
}

async fn live_cross_provider_handoff_model_options(
    test: &str,
    pair: &LiveCrossProviderHandoffPair,
) -> Result<Option<(Model, SimpleStreamOptions)>, Box<dyn Error>> {
    let Some(mut model) = get_model(pair.provider, pair.model_id) else {
        if live_cross_provider_handoff_source_missing_model(pair) {
            eprintln!(
                "skipping {test}: source pair {} has no generated model entry",
                pair.label
            );
            return Ok(None);
        }
        skip_live(
            test,
            format!(
                "missing model registry entry for source handoff pair {}/{}",
                pair.provider, pair.model_id
            ),
        );
        return Ok(None);
    };
    if let Some(api_override) = pair.api_override {
        model.api = api_override.to_owned();
    }

    let options = match pair.provider {
        "openai-codex" => {
            let Some(resolution) = live_oauth_resolution(test, "openai-codex").await else {
                return Ok(None);
            };
            model = live_oauth_model("openai-codex", pair.model_id, &resolution)?;
            live_text_options(Some(resolution.api_key))
        }
        "github-copilot" => {
            let Some(resolution) = live_oauth_resolution(test, "github-copilot").await else {
                return Ok(None);
            };
            model = live_oauth_model("github-copilot", pair.model_id, &resolution)?;
            live_text_options(Some(resolution.api_key))
        }
        "anthropic" => {
            let Some(api_key) = live_api_key(test, "ANTHROPIC_API_KEY") else {
                return Ok(None);
            };
            live_text_options(Some(api_key))
        }
        "azure-openai-responses" => {
            let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
                return Ok(None);
            };
            let Some(options) = live_azure_options(test, api_key) else {
                return Ok(None);
            };
            options
        }
        "amazon-bedrock" => {
            if !live_bedrock_ready(test) {
                return Ok(None);
            }
            live_text_options(None)
        }
        "cloudflare-workers-ai" => {
            let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
                return Ok(None);
            };
            let Some(options) = live_cloudflare_options(test, api_key, false) else {
                return Ok(None);
            };
            options
        }
        "cloudflare-ai-gateway" => {
            if let Some(upstream_api_key_env) = pair.upstream_api_key_env {
                let Some(options) =
                    live_cloudflare_gateway_byok_options(test, upstream_api_key_env)
                else {
                    return Ok(None);
                };
                options
            } else {
                let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
                    return Ok(None);
                };
                let Some(options) = live_cloudflare_options(test, api_key, true) else {
                    return Ok(None);
                };
                options
            }
        }
        provider => {
            if !live_tests_enabled() {
                skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
                return Ok(None);
            }
            let Some(api_key) = get_env_api_key(provider) else {
                skip_live(
                    test,
                    format!("no API key is configured for provider {provider}"),
                );
                return Ok(None);
            };
            live_text_options(Some(api_key))
        }
    };

    Ok(Some((
        model.clone(),
        live_handoff_options_for_model(&model, options),
    )))
}

async fn live_generate_cross_provider_handoff_fixture(
    test: &str,
    pair: &LiveCrossProviderHandoffPair,
    model: &Model,
    options: &SimpleStreamOptions,
) -> Result<LiveCrossProviderHandoffFixture, Box<dyn Error>> {
    let user = live_double_number_user_message();
    let first = complete_simple(
        model,
        live_openai_reasoning_replay_first_context(
            "You are a helpful assistant. Use the provided tool to complete the task.",
            user.clone(),
        ),
        options.clone(),
    )
    .await?;
    let tool_call = live_tool_call_from_response(test, &first, pair.label)?;
    let tool_result = live_tool_result_for_call(&tool_call);
    let final_response = complete_simple(
        model,
        live_openai_reasoning_replay_followup_context(
            "You are a helpful assistant.",
            vec![
                user.clone(),
                Message::Assistant(first.clone()),
                tool_result.clone(),
            ],
        ),
        options.clone(),
    )
    .await?;
    assert_live_non_error_with_content(test, &final_response);

    Ok(LiveCrossProviderHandoffFixture {
        label: pair.label,
        model: model.clone(),
        options: options.clone(),
        messages: vec![
            user,
            Message::Assistant(first),
            tool_result,
            Message::Assistant(final_response),
        ],
    })
}

async fn run_live_cross_provider_handoff_matrix(test: &str) -> Result<(), Box<dyn Error>> {
    if !live_network_enabled(test) {
        return Ok(());
    }

    let mut fixtures = Vec::new();
    for pair in live_cross_provider_handoff_pairs() {
        let Some((model, options)) = live_cross_provider_handoff_model_options(test, pair).await?
        else {
            continue;
        };
        let fixture =
            live_generate_cross_provider_handoff_fixture(test, pair, &model, &options).await?;
        fixtures.push(fixture);
    }

    if fixtures.len() < 2 {
        skip_live(
            test,
            format!(
                "need at least two generated cross-provider handoff fixtures, got {}",
                fixtures.len()
            ),
        );
        return Ok(());
    }

    let mut failures = Vec::new();
    for target in &fixtures {
        let mut messages = Vec::new();
        for source in &fixtures {
            if source.label != target.label {
                messages.extend(source.messages.clone());
            }
        }
        messages.push(Message::User(UserMessage::text(
            "Great, thanks for all that help! Now just say 'Hello, handoff successful!' to confirm you received everything.",
        )));

        let response = complete_simple(
            &target.model,
            Context {
                system_prompt: Some("You are a helpful assistant.".to_owned()),
                messages,
                tools: vec![live_double_number_tool()],
            },
            target.options.clone(),
        )
        .await?;
        if response.stop_reason == StopReason::Error
            || response
                .error_message
                .as_deref()
                .is_some_and(|message| !message.trim().is_empty())
            || response.content.is_empty()
        {
            failures.push(format!(
                "{} returned {:?}: {:?}",
                target.label, response.stop_reason, response.error_message
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{test} cross-provider handoff failures: {failures:#?}"
    );
    Ok(())
}

async fn run_live_empty_azure_smoke(test: &str, case: LiveEmptyCase) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_empty_with_options(test, &model, options, case).await
}

async fn run_live_empty_cloudflare_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    requires_gateway: bool,
    case: LiveEmptyCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_cloudflare_options(test, api_key, requires_gateway) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_empty_with_options(test, &model, options, case).await
}

async fn run_live_tool_call_without_result_azure_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_tool_call_without_result_with_options(test, &model, options).await
}

async fn run_live_tool_call_without_result_cloudflare_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    requires_gateway: bool,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_cloudflare_options(test, api_key, requires_gateway) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_tool_call_without_result_with_options(test, &model, options).await
}

async fn run_live_unicode_tool_result_azure_smoke(
    test: &str,
    case: LiveUnicodeToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_unicode_tool_result_with_options(test, &model, options, case).await
}

async fn run_live_image_tool_result_azure_smoke(
    test: &str,
    case: LiveImageToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_image_tool_result_with_options(test, &model, options, case).await
}

async fn run_live_image_input_azure_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_image_input_with_options(test, &model, options).await
}

async fn run_live_responses_tool_result_images_azure_smoke(
    test: &str,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_responses_tool_result_images_with_options(test, &model, options).await
}

async fn run_live_unicode_tool_result_cloudflare_smoke(
    test: &str,
    provider: &str,
    model_id: &str,
    requires_gateway: bool,
    case: LiveUnicodeToolResultCase,
) -> Result<(), Box<dyn Error>> {
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_cloudflare_options(test, api_key, requires_gateway) else {
        return Ok(());
    };
    let model = get_model(provider, model_id)
        .ok_or_else(|| format!("missing model registry entry: {provider}/{model_id}"))?;
    run_live_unicode_tool_result_with_options(test, &model, options, case).await
}

fn live_bedrock_ready(test: &str) -> bool {
    if !live_tests_enabled() {
        skip_live(test, format!("{LIVE_GATE_ENV}=1 is not set"));
        return false;
    }
    let has_region = live_env("AWS_REGION")
        .or_else(|| live_env("AWS_DEFAULT_REGION"))
        .is_some();
    let has_auth = live_env("AWS_PROFILE").is_some()
        || (live_env("AWS_ACCESS_KEY_ID").is_some() && live_env("AWS_SECRET_ACCESS_KEY").is_some())
        || live_env("AWS_BEARER_TOKEN_BEDROCK").is_some()
        || (live_env("AWS_WEB_IDENTITY_TOKEN_FILE").is_some()
            && live_env("AWS_ROLE_ARN").is_some())
        || live_env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI").is_some()
        || live_env("AWS_CONTAINER_CREDENTIALS_FULL_URI").is_some();
    if !has_region || !has_auth {
        skip_live(
            test,
            "AWS region and AWS profile/access keys/bearer token are not set",
        );
        return false;
    }
    true
}

fn live_bedrock_extensive_ready(test: &str) -> bool {
    if !live_bedrock_ready(test) {
        return false;
    }
    if !live_gate_value_enabled(env::var(LIVE_BEDROCK_EXTENSIVE_ENV).ok().as_deref()) {
        skip_live(test, format!("{LIVE_BEDROCK_EXTENSIVE_ENV}=1 is not set"));
        return false;
    }
    true
}

async fn run_live_empty_bedrock_smoke(
    test: &str,
    case: LiveEmptyCase,
) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_empty_with_options(test, &model, live_text_options(None), case).await
}

async fn run_live_tool_call_without_result_bedrock_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_tool_call_without_result_with_options(test, &model, live_text_options(None)).await
}

async fn run_live_unicode_tool_result_bedrock_smoke(
    test: &str,
    case: LiveUnicodeToolResultCase,
) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_unicode_tool_result_with_options(test, &model, live_text_options(None), case).await
}

async fn run_live_image_tool_result_bedrock_smoke(
    test: &str,
    case: LiveImageToolResultCase,
) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_image_tool_result_with_options(test, &model, live_text_options(None), case).await
}

async fn run_live_image_input_bedrock_smoke(test: &str) -> Result<(), Box<dyn Error>> {
    if !live_bedrock_ready(test) {
        return Ok(());
    }
    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_image_input_with_options(test, &model, live_text_options(None)).await
}

macro_rules! live_empty_api_key_tests {
    (
        $empty_content_array:ident,
        $empty_string:ident,
        $whitespace_only:ident,
        $empty_assistant:ident,
        $provider:literal,
        $model_id:literal,
        $api_key_env:literal
    ) => {
        #[tokio::test]
        async fn $empty_content_array() -> Result<(), Box<dyn Error>> {
            run_live_empty_api_key_smoke(
                stringify!($empty_content_array),
                $provider,
                $model_id,
                $api_key_env,
                LiveEmptyCase::EmptyContentArray,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_string() -> Result<(), Box<dyn Error>> {
            run_live_empty_api_key_smoke(
                stringify!($empty_string),
                $provider,
                $model_id,
                $api_key_env,
                LiveEmptyCase::EmptyString,
            )
            .await
        }

        #[tokio::test]
        async fn $whitespace_only() -> Result<(), Box<dyn Error>> {
            run_live_empty_api_key_smoke(
                stringify!($whitespace_only),
                $provider,
                $model_id,
                $api_key_env,
                LiveEmptyCase::WhitespaceOnly,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_assistant() -> Result<(), Box<dyn Error>> {
            run_live_empty_api_key_smoke(
                stringify!($empty_assistant),
                $provider,
                $model_id,
                $api_key_env,
                LiveEmptyCase::EmptyAssistant,
            )
            .await
        }
    };
}

macro_rules! live_empty_oauth_tests {
    (
        $empty_content_array:ident,
        $empty_string:ident,
        $whitespace_only:ident,
        $empty_assistant:ident,
        $provider:literal,
        $model_id:literal
    ) => {
        #[tokio::test]
        async fn $empty_content_array() -> Result<(), Box<dyn Error>> {
            run_live_oauth_empty_smoke(
                stringify!($empty_content_array),
                $provider,
                $model_id,
                LiveEmptyCase::EmptyContentArray,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_string() -> Result<(), Box<dyn Error>> {
            run_live_oauth_empty_smoke(
                stringify!($empty_string),
                $provider,
                $model_id,
                LiveEmptyCase::EmptyString,
            )
            .await
        }

        #[tokio::test]
        async fn $whitespace_only() -> Result<(), Box<dyn Error>> {
            run_live_oauth_empty_smoke(
                stringify!($whitespace_only),
                $provider,
                $model_id,
                LiveEmptyCase::WhitespaceOnly,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_assistant() -> Result<(), Box<dyn Error>> {
            run_live_oauth_empty_smoke(
                stringify!($empty_assistant),
                $provider,
                $model_id,
                LiveEmptyCase::EmptyAssistant,
            )
            .await
        }
    };
}

macro_rules! live_empty_azure_tests {
    (
        $empty_content_array:ident,
        $empty_string:ident,
        $whitespace_only:ident,
        $empty_assistant:ident
    ) => {
        #[tokio::test]
        async fn $empty_content_array() -> Result<(), Box<dyn Error>> {
            run_live_empty_azure_smoke(
                stringify!($empty_content_array),
                LiveEmptyCase::EmptyContentArray,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_string() -> Result<(), Box<dyn Error>> {
            run_live_empty_azure_smoke(stringify!($empty_string), LiveEmptyCase::EmptyString).await
        }

        #[tokio::test]
        async fn $whitespace_only() -> Result<(), Box<dyn Error>> {
            run_live_empty_azure_smoke(stringify!($whitespace_only), LiveEmptyCase::WhitespaceOnly)
                .await
        }

        #[tokio::test]
        async fn $empty_assistant() -> Result<(), Box<dyn Error>> {
            run_live_empty_azure_smoke(stringify!($empty_assistant), LiveEmptyCase::EmptyAssistant)
                .await
        }
    };
}

macro_rules! live_empty_cloudflare_tests {
    (
        $empty_content_array:ident,
        $empty_string:ident,
        $whitespace_only:ident,
        $empty_assistant:ident,
        $provider:literal,
        $model_id:literal,
        $requires_gateway:literal
    ) => {
        #[tokio::test]
        async fn $empty_content_array() -> Result<(), Box<dyn Error>> {
            run_live_empty_cloudflare_smoke(
                stringify!($empty_content_array),
                $provider,
                $model_id,
                $requires_gateway,
                LiveEmptyCase::EmptyContentArray,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_string() -> Result<(), Box<dyn Error>> {
            run_live_empty_cloudflare_smoke(
                stringify!($empty_string),
                $provider,
                $model_id,
                $requires_gateway,
                LiveEmptyCase::EmptyString,
            )
            .await
        }

        #[tokio::test]
        async fn $whitespace_only() -> Result<(), Box<dyn Error>> {
            run_live_empty_cloudflare_smoke(
                stringify!($whitespace_only),
                $provider,
                $model_id,
                $requires_gateway,
                LiveEmptyCase::WhitespaceOnly,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_assistant() -> Result<(), Box<dyn Error>> {
            run_live_empty_cloudflare_smoke(
                stringify!($empty_assistant),
                $provider,
                $model_id,
                $requires_gateway,
                LiveEmptyCase::EmptyAssistant,
            )
            .await
        }
    };
}

macro_rules! live_empty_bedrock_tests {
    (
        $empty_content_array:ident,
        $empty_string:ident,
        $whitespace_only:ident,
        $empty_assistant:ident
    ) => {
        #[tokio::test]
        async fn $empty_content_array() -> Result<(), Box<dyn Error>> {
            run_live_empty_bedrock_smoke(
                stringify!($empty_content_array),
                LiveEmptyCase::EmptyContentArray,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_string() -> Result<(), Box<dyn Error>> {
            run_live_empty_bedrock_smoke(stringify!($empty_string), LiveEmptyCase::EmptyString)
                .await
        }

        #[tokio::test]
        async fn $whitespace_only() -> Result<(), Box<dyn Error>> {
            run_live_empty_bedrock_smoke(
                stringify!($whitespace_only),
                LiveEmptyCase::WhitespaceOnly,
            )
            .await
        }

        #[tokio::test]
        async fn $empty_assistant() -> Result<(), Box<dyn Error>> {
            run_live_empty_bedrock_smoke(
                stringify!($empty_assistant),
                LiveEmptyCase::EmptyAssistant,
            )
            .await
        }
    };
}

macro_rules! live_tool_call_without_result_api_key_test {
    ($name:ident, $provider:literal, $model_id:literal, $api_key_env:literal) => {
        #[tokio::test]
        async fn $name() -> Result<(), Box<dyn Error>> {
            run_live_tool_call_without_result_api_key_smoke(
                stringify!($name),
                $provider,
                $model_id,
                $api_key_env,
            )
            .await
        }
    };
}

macro_rules! live_tool_call_without_result_oauth_test {
    ($name:ident, $provider:literal, $model_id:literal) => {
        #[tokio::test]
        async fn $name() -> Result<(), Box<dyn Error>> {
            run_live_oauth_tool_call_without_result_smoke(stringify!($name), $provider, $model_id)
                .await
        }
    };
}

macro_rules! live_tool_call_without_result_cloudflare_test {
    ($name:ident, $provider:literal, $model_id:literal, $requires_gateway:literal) => {
        #[tokio::test]
        async fn $name() -> Result<(), Box<dyn Error>> {
            run_live_tool_call_without_result_cloudflare_smoke(
                stringify!($name),
                $provider,
                $model_id,
                $requires_gateway,
            )
            .await
        }
    };
}

macro_rules! live_stream_reasoning_api_key_tests {
    (
        $basic:ident,
        $tool:ident,
        $stream:ident,
        $thinking:ident,
        $multiturn:ident,
        $provider:literal,
        $model_id:literal,
        $api_key_env:literal,
        $reasoning:expr,
        $max_tokens:expr,
        $thinking_budgets:expr
    ) => {
        #[tokio::test]
        async fn $basic() -> Result<(), Box<dyn Error>> {
            run_live_basic_text_generation_api_key_smoke(
                stringify!($basic),
                $provider,
                $model_id,
                $api_key_env,
            )
            .await
        }

        #[tokio::test]
        async fn $tool() -> Result<(), Box<dyn Error>> {
            run_live_tool_call_smoke(stringify!($tool), $provider, $model_id, $api_key_env).await
        }

        #[tokio::test]
        async fn $stream() -> Result<(), Box<dyn Error>> {
            run_live_streaming_smoke(stringify!($stream), $provider, $model_id, $api_key_env).await
        }

        #[tokio::test]
        async fn $thinking() -> Result<(), Box<dyn Error>> {
            run_live_reasoning_api_key_smoke(
                stringify!($thinking),
                $provider,
                $model_id,
                $api_key_env,
                $reasoning,
                $max_tokens,
                $thinking_budgets,
            )
            .await
        }

        #[tokio::test]
        async fn $multiturn() -> Result<(), Box<dyn Error>> {
            run_live_tool_followup_api_key_smoke(
                stringify!($multiturn),
                $provider,
                $model_id,
                $api_key_env,
                Some($reasoning),
                $max_tokens,
                $thinking_budgets,
            )
            .await
        }
    };
}

macro_rules! live_stream_tools_api_key_tests {
    (
        $basic:ident,
        $tool:ident,
        $stream:ident,
        $multiturn:ident,
        $provider:literal,
        $model_id:literal,
        $api_key_env:literal
    ) => {
        #[tokio::test]
        async fn $basic() -> Result<(), Box<dyn Error>> {
            run_live_basic_text_generation_api_key_smoke(
                stringify!($basic),
                $provider,
                $model_id,
                $api_key_env,
            )
            .await
        }

        #[tokio::test]
        async fn $tool() -> Result<(), Box<dyn Error>> {
            run_live_tool_call_smoke(stringify!($tool), $provider, $model_id, $api_key_env).await
        }

        #[tokio::test]
        async fn $stream() -> Result<(), Box<dyn Error>> {
            run_live_streaming_smoke(stringify!($stream), $provider, $model_id, $api_key_env).await
        }

        #[tokio::test]
        async fn $multiturn() -> Result<(), Box<dyn Error>> {
            run_live_tool_followup_api_key_smoke(
                stringify!($multiturn),
                $provider,
                $model_id,
                $api_key_env,
                None,
                None,
                None,
            )
            .await
        }
    };
}

macro_rules! live_stream_reasoning_oauth_tests {
    (
        $basic:ident,
        $tool:ident,
        $stream:ident,
        $thinking:ident,
        $multiturn:ident,
        $provider:literal,
        $model_id:literal,
        $reasoning:expr,
        $max_tokens:expr,
        $thinking_budgets:expr,
        $transport:expr
    ) => {
        #[tokio::test]
        async fn $basic() -> Result<(), Box<dyn Error>> {
            run_live_oauth_basic_text_generation_smoke(
                stringify!($basic),
                $provider,
                $model_id,
                $transport,
            )
            .await
        }

        #[tokio::test]
        async fn $tool() -> Result<(), Box<dyn Error>> {
            run_live_oauth_tool_call_smoke(stringify!($tool), $provider, $model_id, $transport)
                .await
        }

        #[tokio::test]
        async fn $stream() -> Result<(), Box<dyn Error>> {
            run_live_oauth_streaming_smoke(stringify!($stream), $provider, $model_id, $transport)
                .await
        }

        #[tokio::test]
        async fn $thinking() -> Result<(), Box<dyn Error>> {
            run_live_oauth_reasoning_smoke(
                stringify!($thinking),
                $provider,
                $model_id,
                $reasoning,
                $max_tokens,
                $thinking_budgets,
                $transport,
            )
            .await
        }

        #[tokio::test]
        async fn $multiturn() -> Result<(), Box<dyn Error>> {
            run_live_oauth_tool_followup_smoke(
                stringify!($multiturn),
                $provider,
                $model_id,
                Some($reasoning),
                $max_tokens,
                $thinking_budgets,
                $transport,
            )
            .await
        }
    };
}

macro_rules! live_stream_tools_oauth_tests {
    (
        $basic:ident,
        $tool:ident,
        $stream:ident,
        $multiturn:ident,
        $provider:literal,
        $model_id:literal,
        $transport:expr
    ) => {
        #[tokio::test]
        async fn $basic() -> Result<(), Box<dyn Error>> {
            run_live_oauth_basic_text_generation_smoke(
                stringify!($basic),
                $provider,
                $model_id,
                $transport,
            )
            .await
        }

        #[tokio::test]
        async fn $tool() -> Result<(), Box<dyn Error>> {
            run_live_oauth_tool_call_smoke(stringify!($tool), $provider, $model_id, $transport)
                .await
        }

        #[tokio::test]
        async fn $stream() -> Result<(), Box<dyn Error>> {
            run_live_oauth_streaming_smoke(stringify!($stream), $provider, $model_id, $transport)
                .await
        }

        #[tokio::test]
        async fn $multiturn() -> Result<(), Box<dyn Error>> {
            run_live_oauth_tool_followup_smoke(
                stringify!($multiturn),
                $provider,
                $model_id,
                None,
                None,
                None,
                $transport,
            )
            .await
        }
    };
}

macro_rules! live_context_overflow_api_key_test {
    ($name:ident, $provider:literal, $model_id:literal, $api_key_env:literal, $expectation:expr) => {
        #[tokio::test]
        async fn $name() -> Result<(), Box<dyn Error>> {
            run_live_context_overflow_api_key_smoke(
                stringify!($name),
                $provider,
                $model_id,
                $api_key_env,
                $expectation,
            )
            .await
        }
    };
}

macro_rules! live_context_overflow_oauth_test {
    ($name:ident, $provider:literal, $model_id:literal, $expectation:expr) => {
        #[tokio::test]
        async fn $name() -> Result<(), Box<dyn Error>> {
            run_live_context_overflow_oauth_smoke(
                stringify!($name),
                $provider,
                $model_id,
                $expectation,
            )
            .await
        }
    };
}

macro_rules! live_unicode_api_key_tests {
    ($emoji:ident, $linkedin:ident, $provider:literal, $model_id:literal, $api_key_env:literal) => {
        #[tokio::test]
        async fn $emoji() -> Result<(), Box<dyn Error>> {
            run_live_unicode_tool_result_api_key_smoke(
                stringify!($emoji),
                $provider,
                $model_id,
                $api_key_env,
                LiveUnicodeToolResultCase::Emoji,
            )
            .await
        }

        #[tokio::test]
        async fn $linkedin() -> Result<(), Box<dyn Error>> {
            run_live_unicode_tool_result_api_key_smoke(
                stringify!($linkedin),
                $provider,
                $model_id,
                $api_key_env,
                LiveUnicodeToolResultCase::LinkedIn,
            )
            .await
        }
    };
}

macro_rules! live_unicode_oauth_tests {
    ($emoji:ident, $linkedin:ident, $provider:literal, $model_id:literal) => {
        #[tokio::test]
        async fn $emoji() -> Result<(), Box<dyn Error>> {
            run_live_oauth_unicode_tool_result_smoke(
                stringify!($emoji),
                $provider,
                $model_id,
                LiveUnicodeToolResultCase::Emoji,
            )
            .await
        }

        #[tokio::test]
        async fn $linkedin() -> Result<(), Box<dyn Error>> {
            run_live_oauth_unicode_tool_result_smoke(
                stringify!($linkedin),
                $provider,
                $model_id,
                LiveUnicodeToolResultCase::LinkedIn,
            )
            .await
        }
    };
}

macro_rules! live_unicode_cloudflare_tests {
    ($emoji:ident, $linkedin:ident, $provider:literal, $model_id:literal, $requires_gateway:literal) => {
        #[tokio::test]
        async fn $emoji() -> Result<(), Box<dyn Error>> {
            run_live_unicode_tool_result_cloudflare_smoke(
                stringify!($emoji),
                $provider,
                $model_id,
                $requires_gateway,
                LiveUnicodeToolResultCase::Emoji,
            )
            .await
        }

        #[tokio::test]
        async fn $linkedin() -> Result<(), Box<dyn Error>> {
            run_live_unicode_tool_result_cloudflare_smoke(
                stringify!($linkedin),
                $provider,
                $model_id,
                $requires_gateway,
                LiveUnicodeToolResultCase::LinkedIn,
            )
            .await
        }
    };
}

macro_rules! live_image_api_key_tests {
    ($image_only:ident, $text_and_image:ident, $provider:literal, $model_id:literal, $api_key_env:literal) => {
        #[tokio::test]
        async fn $image_only() -> Result<(), Box<dyn Error>> {
            run_live_image_tool_result_api_key_smoke(
                stringify!($image_only),
                $provider,
                $model_id,
                $api_key_env,
                LiveImageToolResultCase::ImageOnly,
            )
            .await
        }

        #[tokio::test]
        async fn $text_and_image() -> Result<(), Box<dyn Error>> {
            run_live_image_tool_result_api_key_smoke(
                stringify!($text_and_image),
                $provider,
                $model_id,
                $api_key_env,
                LiveImageToolResultCase::TextAndImage,
            )
            .await
        }
    };
}

macro_rules! live_image_api_key_image_only_test {
    ($image_only:ident, $provider:literal, $model_id:literal, $api_key_env:literal) => {
        #[tokio::test]
        async fn $image_only() -> Result<(), Box<dyn Error>> {
            run_live_image_tool_result_api_key_smoke(
                stringify!($image_only),
                $provider,
                $model_id,
                $api_key_env,
                LiveImageToolResultCase::ImageOnly,
            )
            .await
        }
    };
}

macro_rules! live_image_oauth_tests {
    ($image_only:ident, $text_and_image:ident, $provider:literal, $model_id:literal) => {
        #[tokio::test]
        async fn $image_only() -> Result<(), Box<dyn Error>> {
            run_live_oauth_image_tool_result_smoke(
                stringify!($image_only),
                $provider,
                $model_id,
                LiveImageToolResultCase::ImageOnly,
            )
            .await
        }

        #[tokio::test]
        async fn $text_and_image() -> Result<(), Box<dyn Error>> {
            run_live_oauth_image_tool_result_smoke(
                stringify!($text_and_image),
                $provider,
                $model_id,
                LiveImageToolResultCase::TextAndImage,
            )
            .await
        }
    };
}

macro_rules! live_image_input_api_key_test {
    ($test_name:ident, $provider:literal, $model_id:literal, $api_key_env:literal) => {
        #[tokio::test]
        async fn $test_name() -> Result<(), Box<dyn Error>> {
            run_live_image_input_api_key_smoke(
                stringify!($test_name),
                $provider,
                $model_id,
                $api_key_env,
            )
            .await
        }
    };
}

macro_rules! live_image_input_oauth_test {
    ($test_name:ident, $provider:literal, $model_id:literal) => {
        #[tokio::test]
        async fn $test_name() -> Result<(), Box<dyn Error>> {
            run_live_oauth_image_input_smoke(stringify!($test_name), $provider, $model_id, None)
                .await
        }
    };
}

macro_rules! live_anthropic_messages_e2e_tests {
    ($configured:ident, $forced:ident, $long_cache:ident, $provider:literal) => {
        #[tokio::test]
        async fn $configured() -> Result<(), Box<dyn Error>> {
            run_live_anthropic_messages_eager_tool_input_probe(
                stringify!($configured),
                $provider,
                false,
            )
            .await
        }

        #[tokio::test]
        async fn $forced() -> Result<(), Box<dyn Error>> {
            run_live_anthropic_messages_eager_tool_input_probe(stringify!($forced), $provider, true)
                .await
        }

        #[tokio::test]
        async fn $long_cache() -> Result<(), Box<dyn Error>> {
            run_live_anthropic_messages_long_cache_retention_probe(
                stringify!($long_cache),
                $provider,
            )
            .await
        }
    };
}

live_empty_api_key_tests!(
    live_google_generative_ai_empty_content_array,
    live_google_generative_ai_empty_string_content,
    live_google_generative_ai_whitespace_only_content,
    live_google_generative_ai_empty_assistant_message,
    "google",
    "gemini-2.5-flash",
    "GEMINI_API_KEY"
);

live_empty_api_key_tests!(
    live_openai_completions_empty_content_array,
    live_openai_completions_empty_string_content,
    live_openai_completions_whitespace_only_content,
    live_openai_completions_empty_assistant_message,
    "openai",
    "gpt-4o-mini",
    "OPENAI_API_KEY"
);

live_empty_api_key_tests!(
    live_openai_responses_empty_content_array,
    live_openai_responses_empty_string_content,
    live_openai_responses_whitespace_only_content,
    live_openai_responses_empty_assistant_message,
    "openai",
    "gpt-5-mini",
    "OPENAI_API_KEY"
);

live_empty_azure_tests!(
    live_azure_openai_responses_empty_content_array,
    live_azure_openai_responses_empty_string_content,
    live_azure_openai_responses_whitespace_only_content,
    live_azure_openai_responses_empty_assistant_message
);

live_empty_api_key_tests!(
    live_anthropic_messages_empty_content_array,
    live_anthropic_messages_empty_string_content,
    live_anthropic_messages_whitespace_only_content,
    live_anthropic_messages_empty_assistant_message,
    "anthropic",
    "claude-haiku-4-5",
    "ANTHROPIC_API_KEY"
);

live_empty_api_key_tests!(
    live_xai_empty_content_array,
    live_xai_empty_string_content,
    live_xai_whitespace_only_content,
    live_xai_empty_assistant_message,
    "xai",
    "grok-3",
    "XAI_API_KEY"
);

live_empty_api_key_tests!(
    live_groq_empty_content_array,
    live_groq_empty_string_content,
    live_groq_whitespace_only_content,
    live_groq_empty_assistant_message,
    "groq",
    "openai/gpt-oss-20b",
    "GROQ_API_KEY"
);

live_empty_api_key_tests!(
    live_cerebras_empty_content_array,
    live_cerebras_empty_string_content,
    live_cerebras_whitespace_only_content,
    live_cerebras_empty_assistant_message,
    "cerebras",
    "gpt-oss-120b",
    "CEREBRAS_API_KEY"
);

live_empty_cloudflare_tests!(
    live_cloudflare_workers_ai_empty_content_array,
    live_cloudflare_workers_ai_empty_string_content,
    live_cloudflare_workers_ai_whitespace_only_content,
    live_cloudflare_workers_ai_empty_assistant_message,
    "cloudflare-workers-ai",
    "@cf/moonshotai/kimi-k2.6",
    false
);

live_empty_cloudflare_tests!(
    live_cloudflare_ai_gateway_empty_content_array,
    live_cloudflare_ai_gateway_empty_string_content,
    live_cloudflare_ai_gateway_whitespace_only_content,
    live_cloudflare_ai_gateway_empty_assistant_message,
    "cloudflare-ai-gateway",
    "workers-ai/@cf/moonshotai/kimi-k2.6",
    true
);

live_empty_api_key_tests!(
    live_huggingface_empty_content_array,
    live_huggingface_empty_string_content,
    live_huggingface_whitespace_only_content,
    live_huggingface_empty_assistant_message,
    "huggingface",
    "moonshotai/Kimi-K2.5",
    "HF_TOKEN"
);

live_empty_api_key_tests!(
    live_together_empty_content_array,
    live_together_empty_string_content,
    live_together_whitespace_only_content,
    live_together_empty_assistant_message,
    "together",
    "moonshotai/Kimi-K2.6",
    "TOGETHER_API_KEY"
);

live_empty_api_key_tests!(
    live_zai_empty_content_array,
    live_zai_empty_string_content,
    live_zai_whitespace_only_content,
    live_zai_empty_assistant_message,
    "zai",
    "glm-4.5-air",
    "ZAI_API_KEY"
);

live_empty_api_key_tests!(
    live_mistral_conversations_empty_content_array,
    live_mistral_conversations_empty_string_content,
    live_mistral_conversations_whitespace_only_content,
    live_mistral_conversations_empty_assistant_message,
    "mistral",
    "devstral-medium-latest",
    "MISTRAL_API_KEY"
);

live_empty_api_key_tests!(
    live_minimax_empty_content_array,
    live_minimax_empty_string_content,
    live_minimax_whitespace_only_content,
    live_minimax_empty_assistant_message,
    "minimax",
    "MiniMax-M2.7",
    "MINIMAX_API_KEY"
);

live_empty_api_key_tests!(
    live_xiaomi_empty_content_array,
    live_xiaomi_empty_string_content,
    live_xiaomi_whitespace_only_content,
    live_xiaomi_empty_assistant_message,
    "xiaomi",
    "mimo-v2.5-pro",
    "XIAOMI_API_KEY"
);

live_empty_api_key_tests!(
    live_xiaomi_token_plan_cn_empty_content_array,
    live_xiaomi_token_plan_cn_empty_string_content,
    live_xiaomi_token_plan_cn_whitespace_only_content,
    live_xiaomi_token_plan_cn_empty_assistant_message,
    "xiaomi-token-plan-cn",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_CN_API_KEY"
);

live_empty_api_key_tests!(
    live_xiaomi_token_plan_ams_empty_content_array,
    live_xiaomi_token_plan_ams_empty_string_content,
    live_xiaomi_token_plan_ams_whitespace_only_content,
    live_xiaomi_token_plan_ams_empty_assistant_message,
    "xiaomi-token-plan-ams",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_AMS_API_KEY"
);

live_empty_api_key_tests!(
    live_xiaomi_token_plan_sgp_empty_content_array,
    live_xiaomi_token_plan_sgp_empty_string_content,
    live_xiaomi_token_plan_sgp_whitespace_only_content,
    live_xiaomi_token_plan_sgp_empty_assistant_message,
    "xiaomi-token-plan-sgp",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_SGP_API_KEY"
);

live_empty_api_key_tests!(
    live_kimi_coding_empty_content_array,
    live_kimi_coding_empty_string_content,
    live_kimi_coding_whitespace_only_content,
    live_kimi_coding_empty_assistant_message,
    "kimi-coding",
    "kimi-k2-thinking",
    "KIMI_API_KEY"
);

live_empty_api_key_tests!(
    live_vercel_ai_gateway_empty_content_array,
    live_vercel_ai_gateway_empty_string_content,
    live_vercel_ai_gateway_whitespace_only_content,
    live_vercel_ai_gateway_empty_assistant_message,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY"
);

live_empty_bedrock_tests!(
    live_bedrock_converse_empty_content_array,
    live_bedrock_converse_empty_string_content,
    live_bedrock_converse_whitespace_only_content,
    live_bedrock_converse_empty_assistant_message
);

live_empty_oauth_tests!(
    live_anthropic_oauth_empty_content_array,
    live_anthropic_oauth_empty_string_content,
    live_anthropic_oauth_whitespace_only_content,
    live_anthropic_oauth_empty_assistant_message,
    "anthropic",
    "claude-haiku-4-5"
);

live_empty_oauth_tests!(
    live_github_copilot_oauth_openai_empty_content_array,
    live_github_copilot_oauth_openai_empty_string_content,
    live_github_copilot_oauth_openai_whitespace_only_content,
    live_github_copilot_oauth_openai_empty_assistant_message,
    "github-copilot",
    "gpt-4o"
);

live_empty_oauth_tests!(
    live_github_copilot_oauth_anthropic_empty_content_array,
    live_github_copilot_oauth_anthropic_empty_string_content,
    live_github_copilot_oauth_anthropic_whitespace_only_content,
    live_github_copilot_oauth_anthropic_empty_assistant_message,
    "github-copilot",
    "claude-sonnet-4.6"
);

live_empty_oauth_tests!(
    live_openai_codex_oauth_empty_content_array,
    live_openai_codex_oauth_empty_string_content,
    live_openai_codex_oauth_whitespace_only_content,
    live_openai_codex_oauth_empty_assistant_message,
    "openai-codex",
    "gpt-5.5"
);

live_context_overflow_api_key_test!(
    live_anthropic_messages_detects_context_overflow,
    "anthropic",
    "claude-haiku-4-5",
    "ANTHROPIC_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_oauth_test!(
    live_anthropic_oauth_detects_context_overflow,
    "anthropic",
    "claude-sonnet-4-6",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_oauth_test!(
    live_github_copilot_oauth_openai_detects_context_overflow,
    "github-copilot",
    "gpt-4o",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_oauth_test!(
    live_github_copilot_oauth_anthropic_detects_context_overflow,
    "github-copilot",
    "claude-sonnet-4.6",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_openai_completions_detects_context_overflow,
    "openai",
    "gpt-4o-mini",
    "OPENAI_API_KEY",
    LiveContextOverflowExpectation::Error
);

#[tokio::test]
async fn live_openai_responses_detects_context_overflow() -> Result<(), Box<dyn Error>> {
    run_live_context_overflow_openai_responses_smoke(
        "live_openai_responses_detects_context_overflow",
    )
    .await
}

#[tokio::test]
async fn live_azure_openai_responses_detects_context_overflow() -> Result<(), Box<dyn Error>> {
    run_live_context_overflow_azure_smoke("live_azure_openai_responses_detects_context_overflow")
        .await
}

live_context_overflow_api_key_test!(
    live_google_generative_ai_detects_context_overflow,
    "google",
    "gemini-2.0-flash",
    "GEMINI_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_oauth_test!(
    live_openai_codex_oauth_detects_context_overflow,
    "openai-codex",
    "gpt-5.5",
    LiveContextOverflowExpectation::Error
);

#[tokio::test]
async fn live_bedrock_converse_detects_context_overflow() -> Result<(), Box<dyn Error>> {
    run_live_context_overflow_bedrock_smoke("live_bedrock_converse_detects_context_overflow").await
}

live_context_overflow_api_key_test!(
    live_xai_detects_context_overflow,
    "xai",
    "grok-3-fast",
    "XAI_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_groq_detects_context_overflow,
    "groq",
    "llama-3.3-70b-versatile",
    "GROQ_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_cerebras_detects_context_overflow,
    "cerebras",
    "qwen-3-235b-a22b-instruct-2507",
    "CEREBRAS_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_huggingface_detects_context_overflow,
    "huggingface",
    "moonshotai/Kimi-K2.5",
    "HF_TOKEN",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_together_detects_context_overflow,
    "together",
    "moonshotai/Kimi-K2.6",
    "TOGETHER_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_zai_detects_context_overflow_when_reported,
    "zai",
    "glm-4.5-air",
    "ZAI_API_KEY",
    LiveContextOverflowExpectation::ZaiInconsistent
);

live_context_overflow_api_key_test!(
    live_mistral_conversations_detects_context_overflow,
    "mistral",
    "devstral-medium-latest",
    "MISTRAL_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_minimax_detects_context_overflow,
    "minimax",
    "MiniMax-M2.7",
    "MINIMAX_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_xiaomi_detects_context_overflow,
    "xiaomi",
    "mimo-v2.5-pro",
    "XIAOMI_API_KEY",
    LiveContextOverflowExpectation::LengthZeroOutput
);

live_context_overflow_api_key_test!(
    live_xiaomi_token_plan_cn_detects_context_overflow,
    "xiaomi-token-plan-cn",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    LiveContextOverflowExpectation::LengthZeroOutput
);

live_context_overflow_api_key_test!(
    live_xiaomi_token_plan_ams_detects_context_overflow,
    "xiaomi-token-plan-ams",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    LiveContextOverflowExpectation::LengthZeroOutput
);

live_context_overflow_api_key_test!(
    live_xiaomi_token_plan_sgp_detects_context_overflow,
    "xiaomi-token-plan-sgp",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    LiveContextOverflowExpectation::LengthZeroOutput
);

live_context_overflow_api_key_test!(
    live_kimi_coding_detects_context_overflow,
    "kimi-coding",
    "kimi-k2-thinking",
    "KIMI_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_vercel_ai_gateway_detects_context_overflow,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_openrouter_anthropic_detects_context_overflow,
    "openrouter",
    "anthropic/claude-sonnet-4",
    "OPENROUTER_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_openrouter_deepseek_detects_context_overflow,
    "openrouter",
    "deepseek/deepseek-v3.2",
    "OPENROUTER_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_openrouter_mistral_detects_context_overflow,
    "openrouter",
    "mistralai/mistral-large-2512",
    "OPENROUTER_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_openrouter_google_detects_context_overflow,
    "openrouter",
    "google/gemini-2.5-flash",
    "OPENROUTER_API_KEY",
    LiveContextOverflowExpectation::Error
);

live_context_overflow_api_key_test!(
    live_openrouter_llama_detects_context_overflow,
    "openrouter",
    "meta-llama/llama-4-scout",
    "OPENROUTER_API_KEY",
    LiveContextOverflowExpectation::Error
);

#[tokio::test]
async fn live_ollama_detects_context_overflow_when_reported() -> Result<(), Box<dyn Error>> {
    let test = "live_ollama_detects_context_overflow_when_reported";
    let Some((model, _)) = live_ollama_model_options(test).await? else {
        return Ok(());
    };
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(Some("ollama".to_owned())),
        LiveContextOverflowExpectation::SilentTruncationOrError,
    )
    .await
}

#[tokio::test]
async fn live_ollama_stream_basic_text_generation() -> Result<(), Box<dyn Error>> {
    let test = "live_ollama_stream_basic_text_generation";
    let Some((model, options)) = live_ollama_model_options(test).await? else {
        return Ok(());
    };
    run_live_basic_text_generation_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_ollama_stream_handles_tool_call() -> Result<(), Box<dyn Error>> {
    let test = "live_ollama_stream_handles_tool_call";
    let Some((model, options)) = live_ollama_model_options(test).await? else {
        return Ok(());
    };
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_ollama_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    let test = "live_ollama_streams_text_deltas";
    let Some((model, options)) = live_ollama_model_options(test).await? else {
        return Ok(());
    };
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_ollama_streams_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_ollama_streams_thinking";
    let Some((model, mut options)) = live_ollama_model_options(test).await? else {
        return Ok(());
    };
    options.stream.max_tokens = Some(1_024);
    run_live_reasoning_stream_with_options(test, &model, options, ThinkingLevel::Medium).await
}

#[tokio::test]
async fn live_ollama_handles_multiturn_with_thinking_and_tools() -> Result<(), Box<dyn Error>> {
    let test = "live_ollama_handles_multiturn_with_thinking_and_tools";
    let Some((model, mut options)) = live_ollama_model_options(test).await? else {
        return Ok(());
    };
    options.reasoning = Some(ThinkingLevel::Medium);
    options.stream.max_tokens = Some(1_024);
    run_live_tool_followup_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_lm_studio_detects_context_overflow() -> Result<(), Box<dyn Error>> {
    let test = "live_lm_studio_detects_context_overflow";
    if live_local_http_get_text(test, "http://localhost:1234/v1/models", "LM Studio")
        .await
        .is_none()
    {
        return Ok(());
    }
    let model = live_local_openai_completions_model(
        "lm-studio",
        "local-model",
        "http://localhost:1234/v1",
        false,
        8_192,
        2_048,
        "LM Studio Local Model",
    );
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(Some("lm-studio".to_owned())),
        LiveContextOverflowExpectation::Error,
    )
    .await
}

#[tokio::test]
async fn live_llama_cpp_detects_context_overflow() -> Result<(), Box<dyn Error>> {
    let test = "live_llama_cpp_detects_context_overflow";
    if live_local_http_get_text(test, "http://localhost:8081/health", "llama.cpp")
        .await
        .is_none()
    {
        return Ok(());
    }
    let model = live_local_openai_completions_model(
        "llama.cpp",
        "local-model",
        "http://localhost:8081/v1",
        false,
        4_096,
        2_048,
        "llama.cpp Local Model",
    );
    run_live_context_overflow_with_options(
        test,
        &model,
        live_context_overflow_options(Some("llama.cpp".to_owned())),
        LiveContextOverflowExpectation::Error,
    )
    .await
}

live_tool_call_without_result_api_key_test!(
    live_google_generative_ai_filters_tool_call_without_result,
    "google",
    "gemini-2.5-flash",
    "GEMINI_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_openai_completions_filters_tool_call_without_result,
    "openai",
    "gpt-4o-mini",
    "OPENAI_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_openai_responses_filters_tool_call_without_result,
    "openai",
    "gpt-5-mini",
    "OPENAI_API_KEY"
);

#[tokio::test]
async fn live_azure_openai_responses_filters_tool_call_without_result() -> Result<(), Box<dyn Error>>
{
    run_live_tool_call_without_result_azure_smoke(
        "live_azure_openai_responses_filters_tool_call_without_result",
    )
    .await
}

live_tool_call_without_result_api_key_test!(
    live_anthropic_messages_filters_tool_call_without_result,
    "anthropic",
    "claude-haiku-4-5",
    "ANTHROPIC_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_xai_filters_tool_call_without_result,
    "xai",
    "grok-3-fast",
    "XAI_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_groq_filters_tool_call_without_result,
    "groq",
    "openai/gpt-oss-20b",
    "GROQ_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_cerebras_filters_tool_call_without_result,
    "cerebras",
    "gpt-oss-120b",
    "CEREBRAS_API_KEY"
);

live_tool_call_without_result_cloudflare_test!(
    live_cloudflare_workers_ai_filters_tool_call_without_result,
    "cloudflare-workers-ai",
    "@cf/moonshotai/kimi-k2.6",
    false
);

live_tool_call_without_result_cloudflare_test!(
    live_cloudflare_ai_gateway_filters_tool_call_without_result,
    "cloudflare-ai-gateway",
    "workers-ai/@cf/moonshotai/kimi-k2.6",
    true
);

live_tool_call_without_result_api_key_test!(
    live_huggingface_filters_tool_call_without_result,
    "huggingface",
    "moonshotai/Kimi-K2.5",
    "HF_TOKEN"
);

#[tokio::test]
async fn live_together_filters_tool_call_without_result() -> Result<(), Box<dyn Error>> {
    let test = "live_together_filters_tool_call_without_result";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_tool_call_without_result_with_options(test, &model, options).await
}

live_tool_call_without_result_api_key_test!(
    live_zai_filters_tool_call_without_result,
    "zai",
    "glm-4.5-air",
    "ZAI_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_mistral_conversations_filters_tool_call_without_result,
    "mistral",
    "devstral-medium-latest",
    "MISTRAL_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_minimax_filters_tool_call_without_result,
    "minimax",
    "MiniMax-M2.7",
    "MINIMAX_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_xiaomi_filters_tool_call_without_result,
    "xiaomi",
    "mimo-v2.5-pro",
    "XIAOMI_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_xiaomi_token_plan_cn_filters_tool_call_without_result,
    "xiaomi-token-plan-cn",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_CN_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_xiaomi_token_plan_ams_filters_tool_call_without_result,
    "xiaomi-token-plan-ams",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_AMS_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_xiaomi_token_plan_sgp_filters_tool_call_without_result,
    "xiaomi-token-plan-sgp",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_SGP_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_kimi_coding_filters_tool_call_without_result,
    "kimi-coding",
    "kimi-k2-thinking",
    "KIMI_API_KEY"
);

live_tool_call_without_result_api_key_test!(
    live_vercel_ai_gateway_filters_tool_call_without_result,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY"
);

#[tokio::test]
async fn live_bedrock_converse_filters_tool_call_without_result() -> Result<(), Box<dyn Error>> {
    run_live_tool_call_without_result_bedrock_smoke(
        "live_bedrock_converse_filters_tool_call_without_result",
    )
    .await
}

live_tool_call_without_result_oauth_test!(
    live_anthropic_oauth_filters_tool_call_without_result,
    "anthropic",
    "claude-haiku-4-5"
);

live_tool_call_without_result_oauth_test!(
    live_github_copilot_oauth_openai_filters_tool_call_without_result,
    "github-copilot",
    "gpt-4o"
);

live_tool_call_without_result_oauth_test!(
    live_github_copilot_oauth_anthropic_filters_tool_call_without_result,
    "github-copilot",
    "claude-sonnet-4.6"
);

live_tool_call_without_result_oauth_test!(
    live_openai_codex_oauth_filters_tool_call_without_result,
    "openai-codex",
    "gpt-5.5"
);

live_unicode_api_key_tests!(
    live_google_generative_ai_handles_emoji_tool_result,
    live_google_generative_ai_handles_linkedin_unicode_tool_result,
    "google",
    "gemini-2.5-flash",
    "GEMINI_API_KEY"
);

live_unicode_api_key_tests!(
    live_openai_completions_handles_emoji_tool_result,
    live_openai_completions_handles_linkedin_unicode_tool_result,
    "openai",
    "gpt-4o-mini",
    "OPENAI_API_KEY"
);

live_unicode_api_key_tests!(
    live_openai_responses_handles_emoji_tool_result,
    live_openai_responses_handles_linkedin_unicode_tool_result,
    "openai",
    "gpt-5-mini",
    "OPENAI_API_KEY"
);

#[tokio::test]
async fn live_azure_openai_responses_handles_emoji_tool_result() -> Result<(), Box<dyn Error>> {
    run_live_unicode_tool_result_azure_smoke(
        "live_azure_openai_responses_handles_emoji_tool_result",
        LiveUnicodeToolResultCase::Emoji,
    )
    .await
}

#[tokio::test]
async fn live_azure_openai_responses_handles_linkedin_unicode_tool_result()
-> Result<(), Box<dyn Error>> {
    run_live_unicode_tool_result_azure_smoke(
        "live_azure_openai_responses_handles_linkedin_unicode_tool_result",
        LiveUnicodeToolResultCase::LinkedIn,
    )
    .await
}

live_unicode_api_key_tests!(
    live_anthropic_messages_handles_emoji_tool_result,
    live_anthropic_messages_handles_linkedin_unicode_tool_result,
    "anthropic",
    "claude-haiku-4-5",
    "ANTHROPIC_API_KEY"
);

live_unicode_oauth_tests!(
    live_anthropic_oauth_handles_emoji_tool_result,
    live_anthropic_oauth_handles_linkedin_unicode_tool_result,
    "anthropic",
    "claude-haiku-4-5"
);

live_unicode_oauth_tests!(
    live_github_copilot_oauth_openai_handles_emoji_tool_result,
    live_github_copilot_oauth_openai_handles_linkedin_unicode_tool_result,
    "github-copilot",
    "gpt-4o"
);

live_unicode_oauth_tests!(
    live_github_copilot_oauth_anthropic_handles_emoji_tool_result,
    live_github_copilot_oauth_anthropic_handles_linkedin_unicode_tool_result,
    "github-copilot",
    "claude-sonnet-4.6"
);

live_unicode_api_key_tests!(
    live_xai_handles_emoji_tool_result,
    live_xai_handles_linkedin_unicode_tool_result,
    "xai",
    "grok-3",
    "XAI_API_KEY"
);

live_unicode_api_key_tests!(
    live_groq_handles_emoji_tool_result,
    live_groq_handles_linkedin_unicode_tool_result,
    "groq",
    "openai/gpt-oss-20b",
    "GROQ_API_KEY"
);

live_unicode_api_key_tests!(
    live_cerebras_handles_emoji_tool_result,
    live_cerebras_handles_linkedin_unicode_tool_result,
    "cerebras",
    "gpt-oss-120b",
    "CEREBRAS_API_KEY"
);

live_unicode_cloudflare_tests!(
    live_cloudflare_workers_ai_handles_emoji_tool_result,
    live_cloudflare_workers_ai_handles_linkedin_unicode_tool_result,
    "cloudflare-workers-ai",
    "@cf/moonshotai/kimi-k2.6",
    false
);

live_unicode_cloudflare_tests!(
    live_cloudflare_ai_gateway_handles_emoji_tool_result,
    live_cloudflare_ai_gateway_handles_linkedin_unicode_tool_result,
    "cloudflare-ai-gateway",
    "workers-ai/@cf/moonshotai/kimi-k2.6",
    true
);

live_unicode_api_key_tests!(
    live_huggingface_handles_emoji_tool_result,
    live_huggingface_handles_linkedin_unicode_tool_result,
    "huggingface",
    "moonshotai/Kimi-K2.5",
    "HF_TOKEN"
);

#[tokio::test]
async fn live_together_handles_emoji_tool_result() -> Result<(), Box<dyn Error>> {
    let test = "live_together_handles_emoji_tool_result";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_unicode_tool_result_with_options(
        test,
        &model,
        options,
        LiveUnicodeToolResultCase::Emoji,
    )
    .await
}

#[tokio::test]
async fn live_together_handles_linkedin_unicode_tool_result() -> Result<(), Box<dyn Error>> {
    let test = "live_together_handles_linkedin_unicode_tool_result";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_unicode_tool_result_with_options(
        test,
        &model,
        options,
        LiveUnicodeToolResultCase::LinkedIn,
    )
    .await
}

live_unicode_api_key_tests!(
    live_zai_handles_emoji_tool_result,
    live_zai_handles_linkedin_unicode_tool_result,
    "zai",
    "glm-4.5-air",
    "ZAI_API_KEY"
);

live_unicode_api_key_tests!(
    live_mistral_conversations_handles_emoji_tool_result,
    live_mistral_conversations_handles_linkedin_unicode_tool_result,
    "mistral",
    "devstral-medium-latest",
    "MISTRAL_API_KEY"
);

live_unicode_api_key_tests!(
    live_minimax_handles_emoji_tool_result,
    live_minimax_handles_linkedin_unicode_tool_result,
    "minimax",
    "MiniMax-M2.7",
    "MINIMAX_API_KEY"
);

live_unicode_api_key_tests!(
    live_xiaomi_handles_emoji_tool_result,
    live_xiaomi_handles_linkedin_unicode_tool_result,
    "xiaomi",
    "mimo-v2.5-pro",
    "XIAOMI_API_KEY"
);

live_unicode_api_key_tests!(
    live_xiaomi_token_plan_cn_handles_emoji_tool_result,
    live_xiaomi_token_plan_cn_handles_linkedin_unicode_tool_result,
    "xiaomi-token-plan-cn",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_CN_API_KEY"
);

live_unicode_api_key_tests!(
    live_xiaomi_token_plan_ams_handles_emoji_tool_result,
    live_xiaomi_token_plan_ams_handles_linkedin_unicode_tool_result,
    "xiaomi-token-plan-ams",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_AMS_API_KEY"
);

live_unicode_api_key_tests!(
    live_xiaomi_token_plan_sgp_handles_emoji_tool_result,
    live_xiaomi_token_plan_sgp_handles_linkedin_unicode_tool_result,
    "xiaomi-token-plan-sgp",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_SGP_API_KEY"
);

live_unicode_api_key_tests!(
    live_kimi_coding_handles_emoji_tool_result,
    live_kimi_coding_handles_linkedin_unicode_tool_result,
    "kimi-coding",
    "kimi-k2-thinking",
    "KIMI_API_KEY"
);

live_unicode_api_key_tests!(
    live_vercel_ai_gateway_handles_emoji_tool_result,
    live_vercel_ai_gateway_handles_linkedin_unicode_tool_result,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY"
);

#[tokio::test]
async fn live_bedrock_converse_handles_emoji_tool_result() -> Result<(), Box<dyn Error>> {
    run_live_unicode_tool_result_bedrock_smoke(
        "live_bedrock_converse_handles_emoji_tool_result",
        LiveUnicodeToolResultCase::Emoji,
    )
    .await
}

#[tokio::test]
async fn live_bedrock_converse_handles_linkedin_unicode_tool_result() -> Result<(), Box<dyn Error>>
{
    run_live_unicode_tool_result_bedrock_smoke(
        "live_bedrock_converse_handles_linkedin_unicode_tool_result",
        LiveUnicodeToolResultCase::LinkedIn,
    )
    .await
}

live_unicode_oauth_tests!(
    live_openai_codex_oauth_handles_emoji_tool_result,
    live_openai_codex_oauth_handles_linkedin_unicode_tool_result,
    "openai-codex",
    "gpt-5.5"
);

live_image_api_key_tests!(
    live_google_generative_ai_handles_image_only_tool_result,
    live_google_generative_ai_handles_text_and_image_tool_result,
    "google",
    "gemini-2.5-flash",
    "GEMINI_API_KEY"
);

live_image_api_key_tests!(
    live_openai_completions_handles_image_only_tool_result,
    live_openai_completions_handles_text_and_image_tool_result,
    "openai",
    "gpt-4o-mini",
    "OPENAI_API_KEY"
);

live_image_api_key_tests!(
    live_openai_responses_handles_image_only_tool_result,
    live_openai_responses_handles_text_and_image_tool_result,
    "openai",
    "gpt-5-mini",
    "OPENAI_API_KEY"
);

#[tokio::test]
async fn live_azure_openai_responses_handles_image_only_tool_result() -> Result<(), Box<dyn Error>>
{
    run_live_image_tool_result_azure_smoke(
        "live_azure_openai_responses_handles_image_only_tool_result",
        LiveImageToolResultCase::ImageOnly,
    )
    .await
}

#[tokio::test]
async fn live_azure_openai_responses_handles_text_and_image_tool_result()
-> Result<(), Box<dyn Error>> {
    run_live_image_tool_result_azure_smoke(
        "live_azure_openai_responses_handles_text_and_image_tool_result",
        LiveImageToolResultCase::TextAndImage,
    )
    .await
}

live_image_api_key_tests!(
    live_anthropic_messages_handles_image_only_tool_result,
    live_anthropic_messages_handles_text_and_image_tool_result,
    "anthropic",
    "claude-haiku-4-5",
    "ANTHROPIC_API_KEY"
);

live_image_api_key_tests!(
    live_openrouter_handles_image_only_tool_result,
    live_openrouter_handles_text_and_image_tool_result,
    "openrouter",
    "z-ai/glm-4.5v",
    "OPENROUTER_API_KEY"
);

live_image_api_key_tests!(
    live_mistral_conversations_handles_image_only_tool_result,
    live_mistral_conversations_handles_text_and_image_tool_result,
    "mistral",
    "pixtral-12b",
    "MISTRAL_API_KEY"
);

#[tokio::test]
async fn live_together_handles_image_only_tool_result() -> Result<(), Box<dyn Error>> {
    let test = "live_together_handles_image_only_tool_result";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_image_tool_result_with_options(
        test,
        &model,
        options,
        LiveImageToolResultCase::ImageOnly,
    )
    .await
}

#[tokio::test]
async fn live_together_handles_text_and_image_tool_result() -> Result<(), Box<dyn Error>> {
    let test = "live_together_handles_text_and_image_tool_result";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_image_tool_result_with_options(
        test,
        &model,
        options,
        LiveImageToolResultCase::TextAndImage,
    )
    .await
}

live_image_api_key_image_only_test!(
    live_xiaomi_handles_image_only_tool_result,
    "xiaomi",
    "mimo-v2.5-pro",
    "XIAOMI_API_KEY"
);

live_image_api_key_image_only_test!(
    live_xiaomi_token_plan_cn_handles_image_only_tool_result,
    "xiaomi-token-plan-cn",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_CN_API_KEY"
);

live_image_api_key_image_only_test!(
    live_xiaomi_token_plan_ams_handles_image_only_tool_result,
    "xiaomi-token-plan-ams",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_AMS_API_KEY"
);

live_image_api_key_image_only_test!(
    live_xiaomi_token_plan_sgp_handles_image_only_tool_result,
    "xiaomi-token-plan-sgp",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_SGP_API_KEY"
);

live_image_api_key_tests!(
    live_kimi_coding_handles_image_only_tool_result,
    live_kimi_coding_handles_text_and_image_tool_result,
    "kimi-coding",
    "kimi-for-coding",
    "KIMI_API_KEY"
);

live_image_api_key_tests!(
    live_vercel_ai_gateway_handles_image_only_tool_result,
    live_vercel_ai_gateway_handles_text_and_image_tool_result,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY"
);

#[tokio::test]
async fn live_bedrock_converse_handles_image_only_tool_result() -> Result<(), Box<dyn Error>> {
    run_live_image_tool_result_bedrock_smoke(
        "live_bedrock_converse_handles_image_only_tool_result",
        LiveImageToolResultCase::ImageOnly,
    )
    .await
}

#[tokio::test]
async fn live_bedrock_converse_handles_text_and_image_tool_result() -> Result<(), Box<dyn Error>> {
    run_live_image_tool_result_bedrock_smoke(
        "live_bedrock_converse_handles_text_and_image_tool_result",
        LiveImageToolResultCase::TextAndImage,
    )
    .await
}

live_image_oauth_tests!(
    live_anthropic_oauth_handles_image_only_tool_result,
    live_anthropic_oauth_handles_text_and_image_tool_result,
    "anthropic",
    "claude-sonnet-4-5"
);

live_image_oauth_tests!(
    live_github_copilot_oauth_openai_handles_image_only_tool_result,
    live_github_copilot_oauth_openai_handles_text_and_image_tool_result,
    "github-copilot",
    "gpt-4o"
);

live_image_oauth_tests!(
    live_github_copilot_oauth_anthropic_handles_image_only_tool_result,
    live_github_copilot_oauth_anthropic_handles_text_and_image_tool_result,
    "github-copilot",
    "claude-sonnet-4.6"
);

live_image_oauth_tests!(
    live_openai_codex_oauth_handles_image_only_tool_result,
    live_openai_codex_oauth_handles_text_and_image_tool_result,
    "openai-codex",
    "gpt-5.5"
);

live_image_input_api_key_test!(
    live_google_generative_ai_handles_image_input,
    "google",
    "gemini-2.5-flash",
    "GEMINI_API_KEY"
);

#[tokio::test]
async fn live_google_vertex_adc_handles_image_input() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_handles_image_input";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    let model = live_google_vertex_model()?;
    run_live_image_input_with_options(test, &model, options).await
}

live_image_input_api_key_test!(
    live_openai_completions_handles_image_input,
    "openai",
    "gpt-4o-mini",
    "OPENAI_API_KEY"
);

live_image_input_api_key_test!(
    live_openai_responses_handles_image_input,
    "openai",
    "gpt-5.4",
    "OPENAI_API_KEY"
);

live_image_input_api_key_test!(
    live_anthropic_messages_handles_image_input,
    "anthropic",
    "claude-haiku-4-5",
    "ANTHROPIC_API_KEY"
);

#[tokio::test]
async fn live_azure_openai_responses_handles_image_input() -> Result<(), Box<dyn Error>> {
    run_live_image_input_azure_smoke("live_azure_openai_responses_handles_image_input").await
}

live_image_input_api_key_test!(
    live_together_handles_image_input,
    "together",
    "moonshotai/Kimi-K2.6",
    "TOGETHER_API_KEY"
);

live_image_input_api_key_test!(
    live_openrouter_handles_image_input,
    "openrouter",
    "z-ai/glm-4.5v",
    "OPENROUTER_API_KEY"
);

live_image_input_api_key_test!(
    live_vercel_ai_gateway_google_handles_image_input,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY"
);

live_image_input_api_key_test!(
    live_vercel_ai_gateway_anthropic_handles_image_input,
    "vercel-ai-gateway",
    "anthropic/claude-opus-4.5",
    "AI_GATEWAY_API_KEY"
);

live_image_input_api_key_test!(
    live_vercel_ai_gateway_openai_handles_image_input,
    "vercel-ai-gateway",
    "openai/gpt-5.1-codex-max",
    "AI_GATEWAY_API_KEY"
);

live_image_input_api_key_test!(
    live_zai_handles_image_input,
    "zai",
    "glm-5.1",
    "ZAI_API_KEY"
);

live_image_input_api_key_test!(
    live_mistral_conversations_handles_image_input,
    "mistral",
    "pixtral-12b",
    "MISTRAL_API_KEY"
);

live_image_input_oauth_test!(
    live_anthropic_oauth_sonnet_handles_image_input,
    "anthropic",
    "claude-sonnet-4-6"
);

live_image_input_oauth_test!(
    live_anthropic_oauth_opus_handles_image_input,
    "anthropic",
    "claude-opus-4-6"
);

live_image_input_oauth_test!(
    live_github_copilot_oauth_openai_handles_image_input,
    "github-copilot",
    "gpt-5.3-codex"
);

live_image_input_oauth_test!(
    live_github_copilot_oauth_anthropic_handles_image_input,
    "github-copilot",
    "claude-sonnet-4.6"
);

live_image_input_oauth_test!(
    live_openai_codex_oauth_gpt_54_handles_image_input,
    "openai-codex",
    "gpt-5.4"
);

live_image_input_oauth_test!(
    live_openai_codex_oauth_gpt_55_handles_image_input,
    "openai-codex",
    "gpt-5.5"
);

#[tokio::test]
async fn live_openai_codex_oauth_gpt_55_websocket_handles_image_input() -> Result<(), Box<dyn Error>>
{
    run_live_oauth_image_input_smoke(
        "live_openai_codex_oauth_gpt_55_websocket_handles_image_input",
        "openai-codex",
        "gpt-5.5",
        Some(Transport::Websocket),
    )
    .await
}

#[tokio::test]
async fn live_bedrock_converse_handles_image_input() -> Result<(), Box<dyn Error>> {
    run_live_image_input_bedrock_smoke("live_bedrock_converse_handles_image_input").await
}

#[tokio::test]
async fn live_openai_responses_keeps_tool_result_images_in_function_call_output()
-> Result<(), Box<dyn Error>> {
    run_live_responses_tool_result_images_api_key_smoke(
        "live_openai_responses_keeps_tool_result_images_in_function_call_output",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
        Some(ThinkingLevel::Low),
    )
    .await
}

#[tokio::test]
async fn live_azure_openai_responses_keeps_tool_result_images_in_function_call_output()
-> Result<(), Box<dyn Error>> {
    run_live_responses_tool_result_images_azure_smoke(
        "live_azure_openai_responses_keeps_tool_result_images_in_function_call_output",
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_openai_keeps_tool_result_images_in_function_call_output()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_responses_tool_result_images_smoke(
        "live_github_copilot_oauth_openai_keeps_tool_result_images_in_function_call_output",
        "github-copilot",
        "gpt-5-mini",
        Some(ThinkingLevel::Low),
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_keeps_tool_result_images_in_function_call_output()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_responses_tool_result_images_smoke(
        "live_openai_codex_oauth_keeps_tool_result_images_in_function_call_output",
        "openai-codex",
        "gpt-5.5",
        Some(ThinkingLevel::Low),
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_normalizes_todowrite_tool_name() -> Result<(), Box<dyn Error>> {
    run_live_anthropic_oauth_tool_name_normalization_smoke(
        "live_anthropic_oauth_normalizes_todowrite_tool_name",
        live_string_arg_tool("todowrite", "Write a todo item", "task", "The task to add"),
        "You are a helpful assistant. Use the todowrite tool when asked to add todos.",
        "Add a todo: buy milk. Use the todowrite tool.",
        "todowrite",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_normalizes_builtin_read_tool_name() -> Result<(), Box<dyn Error>> {
    run_live_anthropic_oauth_tool_name_normalization_smoke(
        "live_anthropic_oauth_normalizes_builtin_read_tool_name",
        live_string_arg_tool("read", "Read a file", "path", "File path"),
        "You are a helpful assistant. Use the read tool to read files.",
        "Read the file /tmp/test.txt using the read tool.",
        "read",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_does_not_map_find_to_glob() -> Result<(), Box<dyn Error>> {
    run_live_anthropic_oauth_tool_name_normalization_smoke(
        "live_anthropic_oauth_does_not_map_find_to_glob",
        live_string_arg_tool("find", "Find files by pattern", "pattern", "Glob pattern"),
        "You are a helpful assistant. Use the find tool to search for files.",
        "Find all .ts files using the find tool.",
        "find",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_preserves_custom_tool_name() -> Result<(), Box<dyn Error>> {
    run_live_anthropic_oauth_tool_name_normalization_smoke(
        "live_anthropic_oauth_preserves_custom_tool_name",
        live_string_arg_tool("my_custom_tool", "A custom tool", "input", "Input value"),
        "You are a helpful assistant. Use my_custom_tool when asked.",
        "Use my_custom_tool with input 'hello'.",
        "my_custom_tool",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_cache_affinity_e2e() -> Result<(), Box<dyn Error>> {
    run_live_cache_affinity_api_key_smoke(
        "live_openai_responses_cache_affinity_e2e",
        "openai",
        "gpt-5.4",
        "OPENAI_API_KEY",
        "0195d6e4-4cf9-7f44-a2d8-f8f7f49ee9d3",
        "Reply with exactly: openai cache affinity e2e success",
        "openai cache affinity e2e success",
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_cache_affinity_e2e() -> Result<(), Box<dyn Error>> {
    run_live_oauth_cache_affinity_smoke(
        "live_openai_codex_oauth_cache_affinity_e2e",
        "openai-codex",
        "gpt-5.5",
        "0195d6e4-4cf9-7f44-a2d8-f8f7f49ee9d3",
        "Reply with exactly: cache affinity e2e success",
        "cache affinity e2e success",
        Some(Transport::Sse),
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_skips_reasoning_only_history_after_aborted_turn()
-> Result<(), Box<dyn Error>> {
    let test = "live_openai_responses_skips_reasoning_only_history_after_aborted_turn";
    let Some(api_key) = live_api_key(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let model = get_model("openai", "gpt-5-mini").ok_or_else(|| "missing openai/gpt-5-mini")?;
    let user = live_double_number_user_message();
    let first = complete_simple(
        &model,
        live_openai_reasoning_replay_first_context(
            "You are a helpful assistant. Use the tool.",
            user.clone(),
        ),
        live_openai_responses_high_reasoning_options(api_key.clone()),
    )
    .await?;
    assert_live_anthropic_e2e_accepted(test, &first);
    let thinking = first
        .content
        .iter()
        .find_map(|content| match content {
            AssistantContent::Thinking(thinking)
                if thinking
                    .thinking_signature
                    .as_deref()
                    .is_some_and(|signature| !signature.trim().is_empty()) =>
            {
                Some(thinking.clone())
            }
            _ => None,
        })
        .ok_or_else(|| format!("{test} missing OpenAI Responses thinking signature: {first:?}"))?;

    let mut corrupted = first;
    corrupted.content = vec![AssistantContent::Thinking(thinking)];
    corrupted.stop_reason = StopReason::Aborted;
    let (captured, hook) = live_capture_payload_hook();
    let mut options = live_openai_responses_high_reasoning_options(api_key);
    options.payload_hooks.push(hook);
    let second = complete_simple(
        &model,
        live_openai_reasoning_replay_followup_context(
            "You are a helpful assistant.",
            vec![
                user,
                Message::Assistant(corrupted),
                Message::User(UserMessage::text("Say hello to confirm you can continue.")),
            ],
        ),
        options,
    )
    .await?;
    assert_live_non_error_with_content(test, &second);
    let payload = live_captured_payload(test, &captured)?;
    assert_live_openai_responses_payload_omits_reasoning(test, &payload);
    Ok(())
}

#[tokio::test]
async fn live_openai_responses_handles_same_provider_different_model_tool_handoff()
-> Result<(), Box<dyn Error>> {
    let test = "live_openai_responses_handles_same_provider_different_model_tool_handoff";
    let Some(api_key) = live_api_key(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let model_a = get_model("openai", "gpt-5-mini").ok_or_else(|| "missing openai/gpt-5-mini")?;
    let model_b =
        get_model("openai", "gpt-5.2-codex").ok_or_else(|| "missing openai/gpt-5.2-codex")?;
    let user = live_double_number_user_message();
    let first = complete_simple(
        &model_a,
        live_openai_reasoning_replay_first_context(
            "You are a helpful assistant. Always use the tool when asked.",
            user.clone(),
        ),
        live_openai_responses_high_reasoning_options(api_key.clone()),
    )
    .await?;
    let tool_call = live_tool_call_from_response(test, &first, "OpenAI Responses")?;
    let (captured, hook) = live_capture_payload_hook();
    let mut options = live_openai_responses_high_reasoning_options(api_key);
    options.payload_hooks.push(hook);
    let second = complete_simple(
        &model_b,
        live_openai_reasoning_replay_followup_context(
            "You are a helpful assistant. Answer concisely.",
            vec![
                user,
                Message::Assistant(first),
                live_tool_result_for_call(&tool_call),
                Message::User(UserMessage::text(
                    "What was the result? Answer with just the number.",
                )),
            ],
        ),
        options,
    )
    .await?;
    assert_live_contains_text(test, &second, "42");
    let payload = live_captured_payload(test, &captured)?;
    assert_live_openai_responses_handoff_payload(test, &payload);
    Ok(())
}

#[tokio::test]
async fn live_openai_responses_handles_cross_provider_anthropic_tool_handoff()
-> Result<(), Box<dyn Error>> {
    let test = "live_openai_responses_handles_cross_provider_anthropic_tool_handoff";
    let Some(anthropic_api_key) = live_api_key(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    let Some(openai_api_key) = live_api_key(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let anthropic_model = get_model("anthropic", "claude-sonnet-4-5")
        .ok_or_else(|| "missing anthropic/claude-sonnet-4-5")?;
    let openai_model =
        get_model("openai", "gpt-5.2-codex").ok_or_else(|| "missing openai/gpt-5.2-codex")?;
    let user = live_double_number_user_message();
    let first = complete_simple(
        &anthropic_model,
        live_openai_reasoning_replay_first_context(
            "You are a helpful assistant. Always use the tool when asked.",
            user.clone(),
        ),
        live_anthropic_high_reasoning_options(anthropic_api_key),
    )
    .await?;
    let tool_call = live_tool_call_from_response(test, &first, "Anthropic")?;
    let (captured, hook) = live_capture_payload_hook();
    let mut options = live_openai_responses_high_reasoning_options(openai_api_key);
    options.payload_hooks.push(hook);
    let second = complete_simple(
        &openai_model,
        live_openai_reasoning_replay_followup_context(
            "You are a helpful assistant. Answer concisely.",
            vec![
                user,
                Message::Assistant(first),
                live_tool_result_for_call(&tool_call),
                Message::User(UserMessage::text(
                    "What was the result? Answer with just the number.",
                )),
            ],
        ),
        options,
    )
    .await?;
    assert_live_contains_text(test, &second, "42");
    let payload = live_captured_payload(test, &captured)?;
    assert_live_openai_responses_handoff_payload(test, &payload);
    Ok(())
}

#[tokio::test]
async fn live_cross_provider_handoff_matrix_matches_source_e2e() -> Result<(), Box<dyn Error>> {
    let test = "live_cross_provider_handoff_matrix_matches_source_e2e";
    run_live_cross_provider_handoff_matrix(test).await
}

#[tokio::test]
async fn live_openrouter_handles_prefilled_long_pipe_tool_call_id() -> Result<(), Box<dyn Error>> {
    let test = "live_openrouter_handles_prefilled_long_pipe_tool_call_id";
    let Some(api_key) = live_api_key(test, "OPENROUTER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("openrouter", "openai/gpt-5.2-codex")
        .ok_or_else(|| "missing openrouter/openai/gpt-5.2-codex")?;
    run_live_prefilled_long_pipe_tool_call_id_smoke(test, &model, live_text_options(Some(api_key)))
        .await
}

#[tokio::test]
async fn live_openai_codex_oauth_handles_prefilled_long_pipe_tool_call_id()
-> Result<(), Box<dyn Error>> {
    let test = "live_openai_codex_oauth_handles_prefilled_long_pipe_tool_call_id";
    let Some(resolution) = live_oauth_resolution(test, "openai-codex").await else {
        return Ok(());
    };
    let model = live_oauth_model("openai-codex", "gpt-5.5", &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.transport = Some(Transport::Sse);
    run_live_prefilled_long_pipe_tool_call_id_smoke(test, &model, options).await
}

#[tokio::test]
async fn live_github_copilot_to_openrouter_normalizes_pipe_tool_call_id()
-> Result<(), Box<dyn Error>> {
    let test = "live_github_copilot_to_openrouter_normalizes_pipe_tool_call_id";
    let Some(api_key) = live_api_key(test, "OPENROUTER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("openrouter", "openai/gpt-5.2-codex")
        .ok_or_else(|| "missing openrouter/openai/gpt-5.2-codex")?;
    run_live_generated_pipe_tool_call_id_handoff_smoke(
        test,
        &model,
        live_text_options(Some(api_key)),
        "hello world",
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_to_openai_codex_normalizes_pipe_tool_call_id()
-> Result<(), Box<dyn Error>> {
    let test = "live_github_copilot_to_openai_codex_normalizes_pipe_tool_call_id";
    let Some(resolution) = live_oauth_resolution(test, "openai-codex").await else {
        return Ok(());
    };
    let model = live_oauth_model("openai-codex", "gpt-5.5", &resolution)?;
    let mut options = live_text_options(Some(resolution.api_key));
    options.stream.transport = Some(Transport::Sse);
    run_live_generated_pipe_tool_call_id_handoff_smoke(test, &model, options, "test message").await
}

#[test]
fn live_anthropic_messages_generated_e2e_cases_cover_catalog_models() {
    let expected = get_providers()
        .into_iter()
        .flat_map(|provider| {
            live_anthropic_messages_models(&provider)
                .into_iter()
                .map(move |model| format!("{provider}/{}", model.id))
        })
        .collect::<BTreeSet<_>>();
    let actual = live_anthropic_messages_cases()
        .into_iter()
        .map(|case| case.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);

    let expected_providers = expected
        .iter()
        .filter_map(|name| {
            name.split_once('/')
                .map(|(provider, _)| provider.to_owned())
        })
        .collect::<BTreeSet<_>>();
    let selected_providers =
        live_select_one_anthropic_messages_case_per_provider(live_anthropic_messages_cases())
            .into_iter()
            .map(|case| case.provider)
            .collect::<BTreeSet<_>>();
    assert_eq!(selected_providers, expected_providers);
    assert!(
        !selected_providers.is_empty(),
        "expected at least one anthropic-messages E2E provider"
    );
}

#[tokio::test]
async fn live_opencode_zen_catalog_models_smoke_complete() -> Result<(), Box<dyn Error>> {
    run_live_opencode_zen_catalog_smoke("live_opencode_zen_catalog_models_smoke_complete").await
}

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_anthropic_accepts_configured_tool_streaming,
    live_anthropic_messages_anthropic_accepts_forced_eager_input_streaming,
    live_anthropic_messages_anthropic_accepts_long_cache_retention,
    "anthropic"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_github_copilot_accepts_configured_tool_streaming,
    live_anthropic_messages_github_copilot_accepts_forced_eager_input_streaming,
    live_anthropic_messages_github_copilot_accepts_long_cache_retention,
    "github-copilot"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_opencode_accepts_configured_tool_streaming,
    live_anthropic_messages_opencode_accepts_forced_eager_input_streaming,
    live_anthropic_messages_opencode_accepts_long_cache_retention,
    "opencode"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_opencode_go_accepts_configured_tool_streaming,
    live_anthropic_messages_opencode_go_accepts_forced_eager_input_streaming,
    live_anthropic_messages_opencode_go_accepts_long_cache_retention,
    "opencode-go"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_cloudflare_ai_gateway_accepts_configured_tool_streaming,
    live_anthropic_messages_cloudflare_ai_gateway_accepts_forced_eager_input_streaming,
    live_anthropic_messages_cloudflare_ai_gateway_accepts_long_cache_retention,
    "cloudflare-ai-gateway"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_fireworks_accepts_configured_tool_streaming,
    live_anthropic_messages_fireworks_accepts_forced_eager_input_streaming,
    live_anthropic_messages_fireworks_accepts_long_cache_retention,
    "fireworks"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_minimax_accepts_configured_tool_streaming,
    live_anthropic_messages_minimax_accepts_forced_eager_input_streaming,
    live_anthropic_messages_minimax_accepts_long_cache_retention,
    "minimax"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_kimi_coding_accepts_configured_tool_streaming,
    live_anthropic_messages_kimi_coding_accepts_forced_eager_input_streaming,
    live_anthropic_messages_kimi_coding_accepts_long_cache_retention,
    "kimi-coding"
);

live_anthropic_messages_e2e_tests!(
    live_anthropic_messages_vercel_ai_gateway_accepts_configured_tool_streaming,
    live_anthropic_messages_vercel_ai_gateway_accepts_forced_eager_input_streaming,
    live_anthropic_messages_vercel_ai_gateway_accepts_long_cache_retention,
    "vercel-ai-gateway"
);

#[tokio::test]
async fn live_google_generative_ai_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_google_generative_ai_immediate_abort",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_openai_completions_immediate_abort",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_openai_responses_immediate_abort",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_azure_openai_responses_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_azure_smoke("live_azure_openai_responses_immediate_abort").await
}

#[tokio::test]
async fn live_anthropic_oauth_auth_storage_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_oauth_smoke(
        "live_anthropic_oauth_auth_storage_immediate_abort",
        "anthropic",
        "claude-opus-4-1-20250805",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_mistral_conversations_immediate_abort",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_together_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_together_immediate_abort",
        "together",
        "moonshotai/Kimi-K2.6",
        "TOGETHER_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_minimax_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_minimax_immediate_abort",
        "minimax",
        "MiniMax-M2.7",
        "MINIMAX_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_xiaomi_immediate_abort",
        "xiaomi",
        "mimo-v2.5-pro",
        "XIAOMI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_handles_midstream_abort_then_new_message() -> Result<(), Box<dyn Error>> {
    run_live_midstream_abort_then_new_message_api_key_smoke(
        "live_xiaomi_handles_midstream_abort_then_new_message",
        "xiaomi",
        "mimo-v2.5-pro",
        "XIAOMI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_cn_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_xiaomi_token_plan_cn_immediate_abort",
        "xiaomi-token-plan-cn",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_cn_handles_midstream_abort_then_new_message()
-> Result<(), Box<dyn Error>> {
    run_live_midstream_abort_then_new_message_api_key_smoke(
        "live_xiaomi_token_plan_cn_handles_midstream_abort_then_new_message",
        "xiaomi-token-plan-cn",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_ams_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_xiaomi_token_plan_ams_immediate_abort",
        "xiaomi-token-plan-ams",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_ams_handles_midstream_abort_then_new_message()
-> Result<(), Box<dyn Error>> {
    run_live_midstream_abort_then_new_message_api_key_smoke(
        "live_xiaomi_token_plan_ams_handles_midstream_abort_then_new_message",
        "xiaomi-token-plan-ams",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_sgp_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_xiaomi_token_plan_sgp_immediate_abort",
        "xiaomi-token-plan-sgp",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_sgp_handles_midstream_abort_then_new_message()
-> Result<(), Box<dyn Error>> {
    run_live_midstream_abort_then_new_message_api_key_smoke(
        "live_xiaomi_token_plan_sgp_handles_midstream_abort_then_new_message",
        "xiaomi-token-plan-sgp",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_kimi_coding_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_kimi_coding_immediate_abort",
        "kimi-coding",
        "kimi-k2-thinking",
        "KIMI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_kimi_coding_handles_midstream_abort_then_new_message() -> Result<(), Box<dyn Error>> {
    run_live_midstream_abort_then_new_message_api_key_smoke(
        "live_kimi_coding_handles_midstream_abort_then_new_message",
        "kimi-coding",
        "kimi-k2-thinking",
        "KIMI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_vercel_ai_gateway_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_api_key_smoke(
        "live_vercel_ai_gateway_immediate_abort",
        "vercel-ai-gateway",
        "google/gemini-2.5-flash",
        "AI_GATEWAY_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_auth_storage_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_oauth_smoke(
        "live_openai_codex_oauth_auth_storage_immediate_abort",
        "openai-codex",
        "gpt-5.5",
    )
    .await
}

#[tokio::test]
async fn live_bedrock_converse_immediate_abort() -> Result<(), Box<dyn Error>> {
    run_live_immediate_abort_bedrock_smoke("live_bedrock_converse_immediate_abort").await
}

#[tokio::test]
async fn live_bedrock_converse_handles_abort_then_new_message() -> Result<(), Box<dyn Error>> {
    run_live_abort_then_new_message_bedrock_smoke(
        "live_bedrock_converse_handles_abort_then_new_message",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_text_smoke(
        "live_openai_responses_smoke_completes",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_response_id_smoke(
        "live_openai_responses_exposes_response_id",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_then_new_message_smoke(
        "live_openai_responses_abort_reports_source_usage_shape",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_invalid_api_key_reports_provider_error() -> Result<(), Box<dyn Error>>
{
    run_live_provider_error_smoke(
        "live_openai_responses_invalid_api_key_reports_provider_error",
        "openai",
        "gpt-5-mini",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    run_live_streaming_smoke(
        "live_openai_responses_streams_text_deltas",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    run_live_multiturn_smoke(
        "live_openai_responses_handles_multiturn_context",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_handles_tool_call() -> Result<(), Box<dyn Error>> {
    run_live_tool_call_smoke(
        "live_openai_responses_handles_tool_call",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_responses_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openai_responses_reports_total_usage_components",
        "openai",
        "gpt-5-mini",
        "OPENAI_API_KEY",
    )
    .await
}

live_stream_reasoning_api_key_tests!(
    live_openai_responses_gpt_54_stream_basic_text_generation,
    live_openai_responses_gpt_54_stream_handles_tool_call,
    live_openai_responses_gpt_54_streams_text_deltas,
    live_openai_responses_gpt_54_streams_thinking,
    live_openai_responses_gpt_54_handles_multiturn_with_thinking_and_tools,
    "openai",
    "gpt-5.4",
    "OPENAI_API_KEY",
    ThinkingLevel::High,
    Some(1_024),
    None
);

#[tokio::test]
async fn live_openai_completions_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_text_smoke(
        "live_openai_completions_smoke_completes",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_response_id_smoke(
        "live_openai_completions_exposes_response_id",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_then_new_message_smoke(
        "live_openai_completions_abort_reports_source_usage_shape",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_invalid_api_key_reports_provider_error()
-> Result<(), Box<dyn Error>> {
    run_live_provider_error_smoke(
        "live_openai_completions_invalid_api_key_reports_provider_error",
        "openai",
        "gpt-4o-mini",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    run_live_streaming_smoke(
        "live_openai_completions_streams_text_deltas",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    run_live_multiturn_smoke(
        "live_openai_completions_handles_multiturn_context",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_handles_tool_call() -> Result<(), Box<dyn Error>> {
    run_live_tool_call_smoke(
        "live_openai_completions_handles_tool_call",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openai_completions_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openai_completions_reports_total_usage_components",
        "openai",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_text_smoke(
        "live_anthropic_messages_smoke_completes",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_response_id_smoke(
        "live_anthropic_messages_exposes_response_id",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_anthropic_messages_abort_reports_source_usage_shape",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
        LiveAbortUsageExpectation::PositiveInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_invalid_api_key_reports_provider_error()
-> Result<(), Box<dyn Error>> {
    run_live_provider_error_smoke(
        "live_anthropic_messages_invalid_api_key_reports_provider_error",
        "anthropic",
        "claude-haiku-4-5",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    run_live_streaming_smoke(
        "live_anthropic_messages_streams_text_deltas",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    run_live_multiturn_smoke(
        "live_anthropic_messages_handles_multiturn_context",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_handles_tool_call() -> Result<(), Box<dyn Error>> {
    run_live_tool_call_smoke(
        "live_anthropic_messages_handles_tool_call",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_messages_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_anthropic_messages_reports_total_usage_components",
        "anthropic",
        "claude-haiku-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_opus_47_streams_reasoning_with_signature() -> Result<(), Box<dyn Error>> {
    run_live_anthropic_opus_47_reasoning_smoke(
        "live_anthropic_opus_47_streams_reasoning_with_signature",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_opus_45_interleaved_thinking() -> Result<(), Box<dyn Error>> {
    run_live_interleaved_thinking_api_key_smoke(
        "live_anthropic_opus_45_interleaved_thinking",
        "anthropic",
        "claude-opus-4-5",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_opus_46_interleaved_thinking() -> Result<(), Box<dyn Error>> {
    run_live_interleaved_thinking_api_key_smoke(
        "live_anthropic_opus_46_interleaved_thinking",
        "anthropic",
        "claude-opus-4-6",
        "ANTHROPIC_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_bedrock_opus_45_interleaved_thinking() -> Result<(), Box<dyn Error>> {
    run_live_interleaved_thinking_bedrock_smoke(
        "live_bedrock_opus_45_interleaved_thinking",
        "global.anthropic.claude-opus-4-5-20251101-v1:0",
    )
    .await
}

#[tokio::test]
async fn live_bedrock_opus_46_interleaved_thinking() -> Result<(), Box<dyn Error>> {
    run_live_interleaved_thinking_bedrock_smoke(
        "live_bedrock_opus_46_interleaved_thinking",
        "global.anthropic.claude-opus-4-6-v1",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_sonnet_45_disables_budget_thinking() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_anthropic_sonnet_45_disables_budget_thinking",
        "anthropic",
        "claude-sonnet-4-5",
        "ANTHROPIC_API_KEY",
        320,
        Some(0.0),
        35,
        None,
    )
    .await
}

#[tokio::test]
async fn live_anthropic_sonnet_46_disables_adaptive_thinking() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_anthropic_sonnet_46_disables_adaptive_thinking",
        "anthropic",
        "claude-sonnet-4-6",
        "ANTHROPIC_API_KEY",
        320,
        Some(0.0),
        35,
        None,
    )
    .await
}

#[tokio::test]
async fn live_google_gemini_25_flash_disables_thinking() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_google_gemini_25_flash_disables_thinking",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
        160,
        Some(0.0),
        35,
        None,
    )
    .await
}

#[tokio::test]
async fn live_google_gemini_3_flash_disables_thinking() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_google_gemini_3_flash_disables_thinking",
        "google",
        "gemini-3-flash-preview",
        "GEMINI_API_KEY",
        160,
        Some(0.0),
        35,
        None,
    )
    .await
}

#[tokio::test]
async fn live_google_gemini_31_pro_accepts_thinking_off() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_google_gemini_31_pro_accepts_thinking_off",
        "google",
        "gemini-3.1-pro-preview",
        "GEMINI_API_KEY",
        512,
        Some(0.0),
        20,
        None,
    )
    .await
}

#[tokio::test]
async fn live_google_vertex_gemini_25_flash_disables_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_gemini_25_flash_disables_thinking";
    let Some(options) = live_google_vertex_options(test) else {
        return Ok(());
    };
    let model = get_model("google-vertex", "gemini-2.5-flash")
        .ok_or_else(|| "missing model registry entry: google-vertex/gemini-2.5-flash")?;
    run_live_thinking_disable_with_options(test, &model, options, 160, Some(0.0), 35, None).await
}

#[tokio::test]
async fn live_google_vertex_gemini_3_flash_disables_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_gemini_3_flash_disables_thinking";
    let Some(options) = live_google_vertex_options(test) else {
        return Ok(());
    };
    let model = get_model("google-vertex", "gemini-3-flash-preview")
        .ok_or_else(|| "missing model registry entry: google-vertex/gemini-3-flash-preview")?;
    run_live_thinking_disable_with_options(test, &model, options, 160, Some(0.0), 35, None).await
}

#[tokio::test]
async fn live_openai_responses_gpt_54_mini_disables_thinking() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_openai_responses_gpt_54_mini_disables_thinking",
        "openai",
        "gpt-5.4-mini",
        "OPENAI_API_KEY",
        160,
        None,
        35,
        None,
    )
    .await
}

#[tokio::test]
async fn live_openrouter_qwen35_disables_thinking() -> Result<(), Box<dyn Error>> {
    run_live_thinking_disable_api_key_smoke(
        "live_openrouter_qwen35_disables_thinking",
        "openrouter",
        "qwen/qwen3.5-plus-02-15",
        "OPENROUTER_API_KEY",
        160,
        Some(0.0),
        35,
        Some(100),
    )
    .await
}

async fn run_live_anthropic_oauth_interactive_callback_login_to_auth_storage(
    test: &str,
) -> Result<(), Box<dyn Error>> {
    if !live_oauth_interactive_enabled(test) {
        return Ok(());
    }

    let flow = start_anthropic_oauth_login_flow().await?;
    live_print_oauth_callback_instructions(test, "Anthropic", &flow);
    let credentials = live_oauth_interactive_timeout_result(
        test,
        "Anthropic OAuth callback",
        finish_anthropic_oauth_login_from_callback_at(
            flow,
            ANTHROPIC_OAUTH_TOKEN_URL,
            live_now_millis_i64(),
        ),
    )
    .await?;
    save_live_oauth_credentials_to_auth_storage(
        test,
        "anthropic",
        StoredOAuthCredentials::from(credentials),
    )
    .await
}

async fn run_live_github_copilot_oauth_interactive_device_login_to_auth_storage(
    test: &str,
) -> Result<(), Box<dyn Error>> {
    if !live_oauth_interactive_enabled(test) {
        return Ok(());
    }

    let urls = github_copilot_urls("github.com");
    let login = live_oauth_interactive_timeout_result(
        test,
        "GitHub Copilot device authorization",
        login_github_copilot_device_flow_for_urls(&urls, None, |device| {
            eprintln!("{test}: complete GitHub Copilot device login:");
            eprintln!("{test}: open {}", device.verification_uri);
            eprintln!("{test}: enter code {}", device.user_code);
            if let Some(verification_uri_complete) = device.verification_uri_complete.as_deref() {
                eprintln!("{test}: direct verification URL {verification_uri_complete}");
            }
            eprintln!(
                "{test}: device code expires in {} seconds; credential will be stored in ~/.pi/agent/auth.json",
                device.expires_in_seconds
            );
        }),
    )
    .await?;
    save_live_oauth_credentials_to_auth_storage(
        test,
        "github-copilot",
        StoredOAuthCredentials::from(login.credentials),
    )
    .await
}

async fn run_live_openai_codex_oauth_interactive_callback_login_to_auth_storage(
    test: &str,
) -> Result<(), Box<dyn Error>> {
    if !live_oauth_interactive_enabled(test) {
        return Ok(());
    }

    let flow = start_openai_codex_oauth_login_flow("ri-live-oauth", Some("pi")).await?;
    live_print_oauth_callback_instructions(test, "OpenAI Codex", &flow);
    let credentials = live_oauth_interactive_timeout_result(
        test,
        "OpenAI Codex OAuth callback",
        finish_openai_codex_oauth_login_from_callback_at(
            flow,
            OPENAI_CODEX_OAUTH_TOKEN_URL,
            live_now_millis_i64(),
        ),
    )
    .await?;
    save_live_oauth_credentials_to_auth_storage(
        test,
        "openai-codex",
        StoredOAuthCredentials::from(credentials),
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_interactive_callback_login_to_auth_storage()
-> Result<(), Box<dyn Error>> {
    run_live_anthropic_oauth_interactive_callback_login_to_auth_storage(
        "live_anthropic_oauth_interactive_callback_login_to_auth_storage",
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_interactive_device_login_to_auth_storage()
-> Result<(), Box<dyn Error>> {
    run_live_github_copilot_oauth_interactive_device_login_to_auth_storage(
        "live_github_copilot_oauth_interactive_device_login_to_auth_storage",
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_interactive_callback_login_to_auth_storage()
-> Result<(), Box<dyn Error>> {
    run_live_openai_codex_oauth_interactive_callback_login_to_auth_storage(
        "live_openai_codex_oauth_interactive_callback_login_to_auth_storage",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_auth_storage_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_oauth_text_smoke(
        "live_anthropic_oauth_auth_storage_smoke_completes",
        "anthropic",
        "claude-haiku-4-5",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_auth_storage_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_oauth_response_id_smoke(
        "live_anthropic_oauth_auth_storage_exposes_response_id",
        "anthropic",
        "claude-haiku-4-5",
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_auth_storage_abort_reports_source_usage_shape()
-> Result<(), Box<dyn Error>> {
    let test = "live_anthropic_oauth_auth_storage_abort_reports_source_usage_shape";
    let Some(resolution) = live_oauth_resolution(test, "anthropic").await else {
        return Ok(());
    };
    let model = live_oauth_model("anthropic", "claude-sonnet-4.6", &resolution)?;
    let mut options =
        live_abort_options(Some(resolution.api_key), Arc::new(AtomicBool::new(false)));
    options.reasoning = Some(ThinkingLevel::High);
    options.thinking_budgets = Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    });
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::PositiveInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_anthropic_oauth_auth_storage_reports_total_usage_components()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_total_usage_smoke(
        "live_anthropic_oauth_auth_storage_reports_total_usage_components",
        "anthropic",
        "claude-haiku-4-5",
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_auth_storage_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_oauth_text_smoke(
        "live_github_copilot_oauth_auth_storage_smoke_completes",
        "github-copilot",
        "gpt-4o",
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_auth_storage_exposes_response_id() -> Result<(), Box<dyn Error>>
{
    run_live_oauth_response_id_smoke(
        "live_github_copilot_oauth_auth_storage_exposes_response_id",
        "github-copilot",
        "gpt-4o",
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_openai_abort_reports_source_usage_shape()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_abort_tokens_smoke(
        "live_github_copilot_oauth_openai_abort_reports_source_usage_shape",
        "github-copilot",
        "gpt-4o",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_anthropic_abort_reports_source_usage_shape()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_abort_tokens_smoke(
        "live_github_copilot_oauth_anthropic_abort_reports_source_usage_shape",
        "github-copilot",
        "claude-sonnet-4.6",
        LiveAbortUsageExpectation::PositiveInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_auth_storage_reports_total_usage_components()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_total_usage_smoke(
        "live_github_copilot_oauth_auth_storage_reports_total_usage_components",
        "github-copilot",
        "gpt-4o",
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_auth_storage_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_oauth_text_smoke(
        "live_openai_codex_oauth_auth_storage_smoke_completes",
        "openai-codex",
        "gpt-5.5",
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_auth_storage_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_oauth_response_id_smoke(
        "live_openai_codex_oauth_auth_storage_exposes_response_id",
        "openai-codex",
        "gpt-5.5",
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_auth_storage_abort_reports_source_usage_shape()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_abort_tokens_then_new_message_smoke(
        "live_openai_codex_oauth_auth_storage_abort_reports_source_usage_shape",
        "openai-codex",
        "gpt-5.5",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_openai_codex_oauth_auth_storage_reports_total_usage_components()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_total_usage_smoke(
        "live_openai_codex_oauth_auth_storage_reports_total_usage_components",
        "openai-codex",
        "gpt-5.5",
    )
    .await
}

live_stream_reasoning_oauth_tests!(
    live_anthropic_oauth_sonnet_stream_basic_text_generation,
    live_anthropic_oauth_sonnet_stream_handles_tool_call,
    live_anthropic_oauth_sonnet_streams_text_deltas,
    live_anthropic_oauth_sonnet_streams_thinking,
    live_anthropic_oauth_sonnet_handles_multiturn_with_thinking_and_tools,
    "anthropic",
    "claude-sonnet-4-6",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    }),
    None
);

live_stream_reasoning_oauth_tests!(
    live_anthropic_oauth_opus_stream_basic_text_generation,
    live_anthropic_oauth_opus_stream_handles_tool_call,
    live_anthropic_oauth_opus_streams_text_deltas,
    live_anthropic_oauth_opus_streams_adaptive_thinking_high,
    live_anthropic_oauth_opus_handles_multiturn_with_adaptive_thinking_and_tools,
    "anthropic",
    "claude-opus-4-6",
    ThinkingLevel::High,
    Some(4_096),
    None,
    None
);

#[tokio::test]
async fn live_anthropic_oauth_opus_streams_adaptive_thinking_medium() -> Result<(), Box<dyn Error>>
{
    run_live_oauth_reasoning_smoke(
        "live_anthropic_oauth_opus_streams_adaptive_thinking_medium",
        "anthropic",
        "claude-opus-4-6",
        ThinkingLevel::Medium,
        Some(4_096),
        None,
        None,
    )
    .await
}

live_stream_tools_oauth_tests!(
    live_github_copilot_oauth_openai_stream_basic_text_generation,
    live_github_copilot_oauth_openai_stream_handles_tool_call,
    live_github_copilot_oauth_openai_streams_text_deltas,
    live_github_copilot_oauth_openai_handles_multiturn_with_tools,
    "github-copilot",
    "gpt-5.3-codex",
    None
);

#[tokio::test]
async fn live_github_copilot_oauth_openai_streams_thinking() -> Result<(), Box<dyn Error>> {
    run_live_oauth_reasoning_smoke(
        "live_github_copilot_oauth_openai_streams_thinking",
        "github-copilot",
        "gpt-5-mini",
        ThinkingLevel::High,
        Some(1_024),
        None,
        None,
    )
    .await
}

#[tokio::test]
async fn live_github_copilot_oauth_openai_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    run_live_oauth_tool_followup_smoke(
        "live_github_copilot_oauth_openai_handles_multiturn_with_thinking_and_tools",
        "github-copilot",
        "gpt-5-mini",
        Some(ThinkingLevel::High),
        Some(1_024),
        None,
        None,
    )
    .await
}

live_stream_reasoning_oauth_tests!(
    live_github_copilot_oauth_anthropic_stream_basic_text_generation,
    live_github_copilot_oauth_anthropic_stream_handles_tool_call,
    live_github_copilot_oauth_anthropic_streams_text_deltas,
    live_github_copilot_oauth_anthropic_streams_thinking,
    live_github_copilot_oauth_anthropic_handles_multiturn_with_thinking_and_tools,
    "github-copilot",
    "claude-sonnet-4.6",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    }),
    None
);

live_stream_reasoning_oauth_tests!(
    live_openai_codex_oauth_gpt_54_stream_basic_text_generation,
    live_openai_codex_oauth_gpt_54_stream_handles_tool_call,
    live_openai_codex_oauth_gpt_54_streams_text_deltas,
    live_openai_codex_oauth_gpt_54_streams_thinking,
    live_openai_codex_oauth_gpt_54_handles_multiturn_with_thinking_and_tools,
    "openai-codex",
    "gpt-5.4",
    ThinkingLevel::High,
    Some(1_024),
    None,
    None
);

live_stream_reasoning_oauth_tests!(
    live_openai_codex_oauth_gpt_55_stream_basic_text_generation,
    live_openai_codex_oauth_gpt_55_stream_handles_tool_call,
    live_openai_codex_oauth_gpt_55_streams_text_deltas,
    live_openai_codex_oauth_gpt_55_streams_thinking_xhigh,
    live_openai_codex_oauth_gpt_55_handles_multiturn_with_thinking_and_tools,
    "openai-codex",
    "gpt-5.5",
    ThinkingLevel::XHigh,
    Some(1_024),
    None,
    None
);

live_stream_reasoning_oauth_tests!(
    live_openai_codex_oauth_gpt_55_websocket_stream_basic_text_generation,
    live_openai_codex_oauth_gpt_55_websocket_stream_handles_tool_call,
    live_openai_codex_oauth_gpt_55_websocket_streams_text_deltas,
    live_openai_codex_oauth_gpt_55_websocket_streams_thinking_xhigh,
    live_openai_codex_oauth_gpt_55_websocket_handles_multiturn_with_thinking_and_tools,
    "openai-codex",
    "gpt-5.5",
    ThinkingLevel::XHigh,
    Some(1_024),
    None,
    Some(Transport::Websocket)
);

#[tokio::test]
async fn live_google_generative_ai_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_text_smoke(
        "live_google_generative_ai_smoke_completes",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_response_id_smoke(
        "live_google_generative_ai_exposes_response_id",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>>
{
    let test = "live_google_generative_ai_abort_reports_source_usage_shape";
    let Some(api_key) = live_api_key(test, "GEMINI_API_KEY") else {
        return Ok(());
    };
    let model = get_model("google", "gemini-2.5-flash")
        .ok_or_else(|| "missing model registry entry: google/gemini-2.5-flash")?;
    let mut options = live_abort_options(Some(api_key), Arc::new(AtomicBool::new(false)));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::PositiveInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_invalid_api_key_reports_provider_error()
-> Result<(), Box<dyn Error>> {
    run_live_provider_error_smoke(
        "live_google_generative_ai_invalid_api_key_reports_provider_error",
        "google",
        "gemini-2.5-flash",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    run_live_streaming_smoke(
        "live_google_generative_ai_streams_text_deltas",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    run_live_multiturn_smoke(
        "live_google_generative_ai_handles_multiturn_context",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_handles_tool_call() -> Result<(), Box<dyn Error>> {
    run_live_tool_call_smoke(
        "live_google_generative_ai_handles_tool_call",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_google_generative_ai_reports_total_usage_components",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_streams_thinking() -> Result<(), Box<dyn Error>> {
    run_live_reasoning_api_key_smoke(
        "live_google_generative_ai_streams_thinking",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
        ThinkingLevel::High,
        Some(1_024),
        None,
    )
    .await
}

#[tokio::test]
async fn live_google_generative_ai_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    run_live_tool_followup_api_key_smoke(
        "live_google_generative_ai_handles_multiturn_with_thinking_and_tools",
        "google",
        "gemini-2.5-flash",
        "GEMINI_API_KEY",
        Some(ThinkingLevel::High),
        Some(1_024),
        None,
    )
    .await
}

#[tokio::test]
async fn live_google_vertex_adc_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_smoke_completes";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    run_live_google_vertex_text_with_options(test, options).await
}

#[tokio::test]
async fn live_google_vertex_api_key_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_api_key_smoke_completes";
    let Some(options) = live_google_vertex_api_key_options(test) else {
        return Ok(());
    };
    run_live_google_vertex_text_with_options(test, options).await
}

#[tokio::test]
async fn live_google_vertex_adc_exposes_response_id() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_exposes_response_id";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    run_live_google_vertex_response_id_with_options(test, options).await
}

#[tokio::test]
async fn live_google_vertex_api_key_exposes_response_id() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_api_key_exposes_response_id";
    let Some(options) = live_google_vertex_api_key_options(test) else {
        return Ok(());
    };
    run_live_google_vertex_response_id_with_options(test, options).await
}

#[tokio::test]
async fn live_google_vertex_adc_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_streams_text_deltas";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    let model = live_google_vertex_model()?;
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_google_vertex_adc_streams_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_streams_thinking";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    let model = live_google_vertex_model()?;
    run_live_reasoning_stream_with_options(test, &model, options, ThinkingLevel::Low).await
}

#[tokio::test]
async fn live_google_vertex_adc_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_handles_multiturn_context";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    let model = live_google_vertex_model()?;
    run_live_multiturn_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_google_vertex_adc_handles_tool_call() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_handles_tool_call";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    let model = live_google_vertex_model()?;
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_google_vertex_adc_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    let test = "live_google_vertex_adc_reports_total_usage_components";
    let Some(options) = live_google_vertex_adc_options(test) else {
        return Ok(());
    };
    let model = live_google_vertex_model()?;
    run_live_total_usage_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_mistral_conversations_smoke_completes() -> Result<(), Box<dyn Error>> {
    run_live_text_smoke(
        "live_mistral_conversations_smoke_completes",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_exposes_response_id() -> Result<(), Box<dyn Error>> {
    run_live_response_id_smoke(
        "live_mistral_conversations_exposes_response_id",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>>
{
    run_live_abort_tokens_then_new_message_smoke(
        "live_mistral_conversations_abort_reports_source_usage_shape",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_invalid_api_key_reports_provider_error()
-> Result<(), Box<dyn Error>> {
    run_live_provider_error_smoke(
        "live_mistral_conversations_invalid_api_key_reports_provider_error",
        "mistral",
        "devstral-medium-latest",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    run_live_streaming_smoke(
        "live_mistral_conversations_streams_text_deltas",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    run_live_multiturn_smoke(
        "live_mistral_conversations_handles_multiturn_context",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_handles_tool_call() -> Result<(), Box<dyn Error>> {
    run_live_tool_call_smoke(
        "live_mistral_conversations_handles_tool_call",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_mistral_conversations_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_mistral_conversations_reports_total_usage_components",
        "mistral",
        "devstral-medium-latest",
        "MISTRAL_API_KEY",
    )
    .await
}

live_stream_tools_api_key_tests!(
    live_mistral_devstral_stream_basic_text_generation,
    live_mistral_devstral_stream_handles_tool_call,
    live_mistral_devstral_streams_text_deltas,
    live_mistral_devstral_handles_multiturn_with_tools,
    "mistral",
    "devstral-medium-latest",
    "MISTRAL_API_KEY"
);

#[tokio::test]
async fn live_mistral_magistral_streams_thinking() -> Result<(), Box<dyn Error>> {
    run_live_reasoning_api_key_smoke(
        "live_mistral_magistral_streams_thinking",
        "mistral",
        "magistral-medium-latest",
        "MISTRAL_API_KEY",
        ThinkingLevel::High,
        Some(1_024),
        None,
    )
    .await
}

#[tokio::test]
async fn live_mistral_magistral_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    run_live_tool_followup_api_key_smoke(
        "live_mistral_magistral_handles_multiturn_with_thinking_and_tools",
        "mistral",
        "magistral-medium-latest",
        "MISTRAL_API_KEY",
        Some(ThinkingLevel::High),
        Some(1_024),
        None,
    )
    .await
}

live_stream_tools_api_key_tests!(
    live_mistral_pixtral_stream_basic_text_generation,
    live_mistral_pixtral_stream_handles_tool_call,
    live_mistral_pixtral_streams_text_deltas,
    live_mistral_pixtral_handles_multiturn_with_tools,
    "mistral",
    "pixtral-12b",
    "MISTRAL_API_KEY"
);

#[tokio::test]
async fn live_azure_openai_responses_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_azure_openai_responses_smoke_completes";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    let message = complete_simple(&model, live_context(), options).await?;
    assert_live_text_response(test, &message);
    Ok(())
}

#[tokio::test]
async fn live_azure_openai_responses_exposes_response_id() -> Result<(), Box<dyn Error>> {
    let test = "live_azure_openai_responses_exposes_response_id";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    let message = complete_simple(&model, live_response_id_context(), options).await?;
    assert_live_response_id(test, &message);
    Ok(())
}

#[tokio::test]
async fn live_azure_openai_responses_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>>
{
    let test = "live_azure_openai_responses_abort_reports_source_usage_shape";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(mut options) = live_azure_options(test, api_key) else {
        return Ok(());
    };
    options.stream.max_tokens = Some(2_048);
    options.stream.abort_flag = Some(Arc::new(AtomicBool::new(false)));

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_azure_openai_responses_invalid_api_key_reports_provider_error()
-> Result<(), Box<dyn Error>> {
    let test = "live_azure_openai_responses_invalid_api_key_reports_provider_error";
    if !live_network_enabled(test) {
        return Ok(());
    }
    let Some(options) = live_azure_options(test, LIVE_BOGUS_API_KEY.to_owned()) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    let message = complete_simple(&model, live_context(), options).await?;
    assert_live_provider_error(test, &message);
    Ok(())
}

#[tokio::test]
async fn live_azure_openai_responses_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    let test = "live_azure_openai_responses_streams_text_deltas";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_azure_openai_responses_handles_multiturn_context() -> Result<(), Box<dyn Error>> {
    let test = "live_azure_openai_responses_handles_multiturn_context";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_multiturn_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_azure_openai_responses_handles_tool_call() -> Result<(), Box<dyn Error>> {
    let test = "live_azure_openai_responses_handles_tool_call";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_azure_openai_responses_reports_total_usage_components() -> Result<(), Box<dyn Error>>
{
    let test = "live_azure_openai_responses_reports_total_usage_components";
    let Some(api_key) = live_api_key(test, "AZURE_OPENAI_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_azure_options(test, api_key) else {
        return Ok(());
    };

    let model = get_model("azure-openai-responses", "gpt-4o-mini")
        .ok_or_else(|| "missing model registry entry: azure-openai-responses/gpt-4o-mini")?;
    run_live_total_usage_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_bedrock_converse_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_smoke_completes";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    let message = complete_simple(&model, live_context(), live_text_options(None)).await?;
    assert_live_text_response(test, &message);
    Ok(())
}

#[tokio::test]
async fn live_bedrock_converse_extensive_model_catalog_smoke_complete() -> Result<(), Box<dyn Error>>
{
    let test = "live_bedrock_converse_extensive_model_catalog_smoke_complete";
    if !live_bedrock_extensive_ready(test) {
        return Ok(());
    }

    let models = get_models("amazon-bedrock");
    assert!(!models.is_empty(), "{test}: no Bedrock models registered");
    eprintln!("{test}: running {} Bedrock catalog models", models.len());
    let mut options = live_text_options(None);
    options.stream.timeout_ms = Some(10_000);

    for model in models {
        let case = format!("{test}/{}", model.id);
        let message = complete_simple(&model, live_context(), options.clone())
            .await
            .map_err(|error| format!("{case} failed before provider response: {error}"))?;
        assert_live_text_response(&case, &message);
        assert!(
            message.error_message.as_deref().unwrap_or("").is_empty(),
            "{case} returned error_message: {:?}",
            message.error_message
        );
        assert!(
            message.usage.input.saturating_add(message.usage.cache_read) > 0,
            "{case} expected positive input/cache-read usage, got {:?}",
            message.usage
        );
        assert!(
            message.usage.output > 0,
            "{case} expected positive output usage, got {:?}",
            message.usage
        );
    }

    Ok(())
}

#[tokio::test]
async fn live_bedrock_converse_handles_tool_call() -> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_handles_tool_call";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_tool_call_with_options(test, &model, live_text_options(None)).await
}

#[tokio::test]
async fn live_bedrock_converse_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_abort_reports_source_usage_shape";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    let mut options = live_abort_options(None, Arc::new(AtomicBool::new(false)));
    options.reasoning = Some(ThinkingLevel::Medium);
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_bedrock_converse_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_reports_total_usage_components";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_total_usage_with_options(test, &model, live_text_options(None)).await
}

#[tokio::test]
async fn live_bedrock_converse_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_streams_text_deltas";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_streaming_with_options(test, &model, live_text_options(None)).await
}

#[tokio::test]
async fn live_bedrock_converse_streams_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_streams_thinking";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    run_live_reasoning_stream_with_options(
        test,
        &model,
        live_text_options(None),
        ThinkingLevel::Medium,
    )
    .await
}

#[tokio::test]
async fn live_bedrock_converse_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    let test = "live_bedrock_converse_handles_multiturn_with_thinking_and_tools";
    if !live_bedrock_ready(test) {
        return Ok(());
    }

    let model = get_model(
        "amazon-bedrock",
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .ok_or_else(|| "missing model registry entry: amazon-bedrock/global sonnet")?;
    let mut options = live_text_options(None);
    options.reasoning = Some(ThinkingLevel::High);
    run_live_tool_followup_with_options(test, &model, options).await
}

live_stream_reasoning_api_key_tests!(
    live_deepseek_stream_basic_text_generation,
    live_deepseek_stream_handles_tool_call,
    live_deepseek_streams_text_deltas,
    live_deepseek_streams_thinking,
    live_deepseek_handles_multiturn_with_thinking_and_tools,
    "deepseek",
    "deepseek-v4-flash",
    "DEEPSEEK_API_KEY",
    ThinkingLevel::High,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_xai_stream_basic_text_generation,
    live_xai_stream_handles_tool_call,
    live_xai_streams_text_deltas,
    live_xai_streams_thinking,
    live_xai_handles_multiturn_with_thinking_and_tools,
    "xai",
    "grok-code-fast-1",
    "XAI_API_KEY",
    ThinkingLevel::Medium,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_groq_stream_basic_text_generation,
    live_groq_stream_handles_tool_call,
    live_groq_streams_text_deltas,
    live_groq_streams_thinking,
    live_groq_handles_multiturn_with_thinking_and_tools,
    "groq",
    "openai/gpt-oss-20b",
    "GROQ_API_KEY",
    ThinkingLevel::Medium,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_cerebras_stream_basic_text_generation,
    live_cerebras_stream_handles_tool_call,
    live_cerebras_streams_text_deltas,
    live_cerebras_streams_thinking,
    live_cerebras_handles_multiturn_with_thinking_and_tools,
    "cerebras",
    "gpt-oss-120b",
    "CEREBRAS_API_KEY",
    ThinkingLevel::Medium,
    Some(1_024),
    None
);

#[tokio::test]
async fn live_xai_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_xai_reports_total_usage_components",
        "xai",
        "grok-3-fast",
        "XAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xai_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_xai_abort_reports_source_usage_shape",
        "xai",
        "grok-3-fast",
        "XAI_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_groq_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_groq_reports_total_usage_components",
        "groq",
        "openai/gpt-oss-120b",
        "GROQ_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_groq_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_groq_abort_reports_source_usage_shape",
        "groq",
        "openai/gpt-oss-20b",
        "GROQ_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_cerebras_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_cerebras_reports_total_usage_components",
        "cerebras",
        "gpt-oss-120b",
        "CEREBRAS_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_cerebras_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_cerebras_abort_reports_source_usage_shape",
        "cerebras",
        "qwen-3-235b-a22b-instruct-2507",
        "CEREBRAS_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_cloudflare_workers_ai_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_workers_ai_reports_total_usage_components";
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_cloudflare_options(test, api_key, false) else {
        return Ok(());
    };
    let model = get_model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6")
        .ok_or_else(|| "missing model registry entry: cloudflare-workers-ai/kimi")?;
    run_live_total_usage_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_workers_ai_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>>
{
    let test = "live_cloudflare_workers_ai_abort_reports_source_usage_shape";
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(mut options) = live_cloudflare_options(test, api_key, false) else {
        return Ok(());
    };
    options.stream.max_tokens = Some(2_048);
    options.stream.abort_flag = Some(Arc::new(AtomicBool::new(false)));
    let model = get_model("cloudflare-workers-ai", "@cf/moonshotai/kimi-k2.6")
        .ok_or_else(|| "missing model registry entry: cloudflare-workers-ai/kimi")?;
    run_live_abort_tokens_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
    .map(|_| ())
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_reports_total_usage_components";
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(options) = live_cloudflare_options(test, api_key, true) else {
        return Ok(());
    };
    let model = get_model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    )
    .ok_or_else(|| "missing model registry entry: cloudflare-ai-gateway/kimi")?;
    run_live_total_usage_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>>
{
    let test = "live_cloudflare_ai_gateway_abort_reports_source_usage_shape";
    let Some(api_key) = live_api_key(test, "CLOUDFLARE_API_KEY") else {
        return Ok(());
    };
    let Some(mut options) = live_cloudflare_options(test, api_key, true) else {
        return Ok(());
    };
    options.stream.max_tokens = Some(2_048);
    options.stream.abort_flag = Some(Arc::new(AtomicBool::new(false)));
    let model = get_model(
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
    )
    .ok_or_else(|| "missing model registry entry: cloudflare-ai-gateway/kimi")?;
    run_live_abort_tokens_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
    .map(|_| ())
}

#[tokio::test]
async fn live_cloudflare_workers_ai_stream_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_workers_ai_stream_smoke_completes";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-workers-ai",
        "@cf/moonshotai/kimi-k2.6",
        false,
    )?
    else {
        return Ok(());
    };
    run_live_basic_text_generation_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_workers_ai_stream_handles_tool_call() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_workers_ai_stream_handles_tool_call";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-workers-ai",
        "@cf/moonshotai/kimi-k2.6",
        false,
    )?
    else {
        return Ok(());
    };
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_workers_ai_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_workers_ai_streams_text_deltas";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-workers-ai",
        "@cf/moonshotai/kimi-k2.6",
        false,
    )?
    else {
        return Ok(());
    };
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_workers_ai_streams_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_workers_ai_streams_thinking";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-workers-ai",
        "@cf/moonshotai/kimi-k2.6",
        false,
    )?
    else {
        return Ok(());
    };
    run_live_reasoning_stream_with_options(test, &model, options, ThinkingLevel::Medium).await
}

#[tokio::test]
async fn live_cloudflare_workers_ai_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_workers_ai_handles_multiturn_with_thinking_and_tools";
    let Some((model, mut options)) = live_cloudflare_model_options(
        test,
        "cloudflare-workers-ai",
        "@cf/moonshotai/kimi-k2.6",
        false,
    )?
    else {
        return Ok(());
    };
    options.reasoning = Some(ThinkingLevel::Medium);
    options.stream.max_tokens = Some(512);
    run_live_tool_followup_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_workers_stream_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_workers_stream_smoke_completes";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        true,
    )?
    else {
        return Ok(());
    };
    run_live_basic_text_generation_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_workers_stream_handles_tool_call() -> Result<(), Box<dyn Error>>
{
    let test = "live_cloudflare_ai_gateway_workers_stream_handles_tool_call";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        true,
    )?
    else {
        return Ok(());
    };
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_workers_streams_text_deltas() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_workers_streams_text_deltas";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        true,
    )?
    else {
        return Ok(());
    };
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_workers_streams_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_workers_streams_thinking";
    let Some((model, options)) = live_cloudflare_model_options(
        test,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        true,
    )?
    else {
        return Ok(());
    };
    run_live_reasoning_stream_with_options(test, &model, options, ThinkingLevel::Medium).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_workers_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_workers_handles_multiturn_with_thinking_and_tools";
    let Some((model, mut options)) = live_cloudflare_model_options(
        test,
        "cloudflare-ai-gateway",
        "workers-ai/@cf/moonshotai/kimi-k2.6",
        true,
    )?
    else {
        return Ok(());
    };
    options.reasoning = Some(ThinkingLevel::Medium);
    options.stream.max_tokens = Some(512);
    run_live_tool_followup_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_openai_byok_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_openai_byok_smoke_completes";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("gpt-5.1")?;
    run_live_basic_text_generation_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_openai_byok_handles_tool_call() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_openai_byok_handles_tool_call";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("gpt-5.1")?;
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_openai_byok_streams_text_deltas() -> Result<(), Box<dyn Error>>
{
    let test = "live_cloudflare_ai_gateway_openai_byok_streams_text_deltas";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("gpt-5.1")?;
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_openai_byok_streams_thinking() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_openai_byok_streams_thinking";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("gpt-5.1")?;
    run_live_reasoning_stream_with_options(test, &model, options, ThinkingLevel::Medium).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_openai_byok_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_openai_byok_handles_multiturn_with_thinking_and_tools";
    let Some(mut options) = live_cloudflare_gateway_byok_options(test, "OPENAI_API_KEY") else {
        return Ok(());
    };
    options.reasoning = Some(ThinkingLevel::Medium);
    options.stream.max_tokens = Some(512);
    let model = live_cloudflare_gateway_model("gpt-5.1")?;
    run_live_tool_followup_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_anthropic_byok_smoke_completes() -> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_anthropic_byok_smoke_completes";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("claude-sonnet-4-5")?;
    run_live_basic_text_generation_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_anthropic_byok_handles_tool_call() -> Result<(), Box<dyn Error>>
{
    let test = "live_cloudflare_ai_gateway_anthropic_byok_handles_tool_call";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("claude-sonnet-4-5")?;
    run_live_tool_call_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_anthropic_byok_streams_text_deltas()
-> Result<(), Box<dyn Error>> {
    let test = "live_cloudflare_ai_gateway_anthropic_byok_streams_text_deltas";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("claude-sonnet-4-5")?;
    run_live_streaming_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_anthropic_byok_streams_thinking() -> Result<(), Box<dyn Error>>
{
    let test = "live_cloudflare_ai_gateway_anthropic_byok_streams_thinking";
    let Some(options) = live_cloudflare_gateway_byok_options(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    let model = live_cloudflare_gateway_model("claude-sonnet-4-5")?;
    run_live_reasoning_stream_with_options(test, &model, options, ThinkingLevel::High).await
}

#[tokio::test]
async fn live_cloudflare_ai_gateway_anthropic_byok_handles_multiturn_with_thinking_and_tools()
-> Result<(), Box<dyn Error>> {
    let test =
        "live_cloudflare_ai_gateway_anthropic_byok_handles_multiturn_with_thinking_and_tools";
    let Some(mut options) = live_cloudflare_gateway_byok_options(test, "ANTHROPIC_API_KEY") else {
        return Ok(());
    };
    options.reasoning = Some(ThinkingLevel::High);
    options.stream.max_tokens = Some(512);
    let model = live_cloudflare_gateway_model("claude-sonnet-4-5")?;
    run_live_tool_followup_with_options(test, &model, options).await
}

live_stream_reasoning_api_key_tests!(
    live_huggingface_stream_basic_text_generation,
    live_huggingface_stream_handles_tool_call,
    live_huggingface_streams_text_deltas,
    live_huggingface_streams_thinking,
    live_huggingface_handles_multiturn_with_thinking_and_tools,
    "huggingface",
    "moonshotai/Kimi-K2.5",
    "HF_TOKEN",
    ThinkingLevel::Medium,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_together_stream_basic_text_generation,
    live_together_stream_handles_tool_call,
    live_together_streams_text_deltas,
    live_together_streams_thinking,
    live_together_handles_multiturn_with_thinking_and_tools,
    "together",
    "moonshotai/Kimi-K2.6",
    "TOGETHER_API_KEY",
    ThinkingLevel::High,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_openrouter_stream_basic_text_generation,
    live_openrouter_stream_handles_tool_call,
    live_openrouter_streams_text_deltas,
    live_openrouter_streams_thinking,
    live_openrouter_handles_multiturn_with_thinking_and_tools,
    "openrouter",
    "z-ai/glm-4.5v",
    "OPENROUTER_API_KEY",
    ThinkingLevel::Medium,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_zai_stream_basic_text_generation,
    live_zai_stream_handles_tool_call,
    live_zai_streams_text_deltas,
    live_zai_streams_thinking,
    live_zai_handles_multiturn_with_thinking_and_tools,
    "zai",
    "glm-5.1",
    "ZAI_API_KEY",
    ThinkingLevel::Medium,
    Some(1_024),
    None
);

live_stream_reasoning_api_key_tests!(
    live_minimax_stream_basic_text_generation,
    live_minimax_stream_handles_tool_call,
    live_minimax_streams_text_deltas,
    live_minimax_handles_thinking,
    live_minimax_handles_multiturn_with_thinking_and_tools,
    "minimax",
    "MiniMax-M2.7",
    "MINIMAX_API_KEY",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    })
);

live_stream_reasoning_api_key_tests!(
    live_kimi_coding_stream_basic_text_generation,
    live_kimi_coding_stream_handles_tool_call,
    live_kimi_coding_streams_text_deltas,
    live_kimi_coding_handles_thinking,
    live_kimi_coding_handles_multiturn_with_thinking_and_tools,
    "kimi-coding",
    "kimi-k2-thinking",
    "KIMI_API_KEY",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    })
);

live_stream_reasoning_api_key_tests!(
    live_xiaomi_stream_basic_text_generation,
    live_xiaomi_stream_handles_tool_call,
    live_xiaomi_streams_text_deltas,
    live_xiaomi_handles_thinking,
    live_xiaomi_handles_multiturn_with_thinking_and_tools,
    "xiaomi",
    "mimo-v2.5-pro",
    "XIAOMI_API_KEY",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    })
);

live_stream_reasoning_api_key_tests!(
    live_xiaomi_token_plan_cn_stream_basic_text_generation,
    live_xiaomi_token_plan_cn_stream_handles_tool_call,
    live_xiaomi_token_plan_cn_streams_text_deltas,
    live_xiaomi_token_plan_cn_handles_thinking,
    live_xiaomi_token_plan_cn_handles_multiturn_with_thinking_and_tools,
    "xiaomi-token-plan-cn",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    })
);

live_stream_reasoning_api_key_tests!(
    live_xiaomi_token_plan_ams_stream_basic_text_generation,
    live_xiaomi_token_plan_ams_stream_handles_tool_call,
    live_xiaomi_token_plan_ams_streams_text_deltas,
    live_xiaomi_token_plan_ams_handles_thinking,
    live_xiaomi_token_plan_ams_handles_multiturn_with_thinking_and_tools,
    "xiaomi-token-plan-ams",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    })
);

live_stream_reasoning_api_key_tests!(
    live_xiaomi_token_plan_sgp_stream_basic_text_generation,
    live_xiaomi_token_plan_sgp_stream_handles_tool_call,
    live_xiaomi_token_plan_sgp_streams_text_deltas,
    live_xiaomi_token_plan_sgp_handles_thinking,
    live_xiaomi_token_plan_sgp_handles_multiturn_with_thinking_and_tools,
    "xiaomi-token-plan-sgp",
    "mimo-v2.5-pro",
    "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    ThinkingLevel::High,
    Some(4_096),
    Some(ThinkingBudgets {
        high: Some(2_048),
        ..Default::default()
    })
);

live_stream_tools_api_key_tests!(
    live_vercel_ai_gateway_google_stream_basic_text_generation,
    live_vercel_ai_gateway_google_stream_handles_tool_call,
    live_vercel_ai_gateway_google_streams_text_deltas,
    live_vercel_ai_gateway_google_handles_multiturn_with_tools,
    "vercel-ai-gateway",
    "google/gemini-2.5-flash",
    "AI_GATEWAY_API_KEY"
);

live_stream_tools_api_key_tests!(
    live_vercel_ai_gateway_anthropic_stream_basic_text_generation,
    live_vercel_ai_gateway_anthropic_stream_handles_tool_call,
    live_vercel_ai_gateway_anthropic_streams_text_deltas,
    live_vercel_ai_gateway_anthropic_handles_multiturn_with_tools,
    "vercel-ai-gateway",
    "anthropic/claude-opus-4.5",
    "AI_GATEWAY_API_KEY"
);

live_stream_tools_api_key_tests!(
    live_vercel_ai_gateway_openai_stream_basic_text_generation,
    live_vercel_ai_gateway_openai_stream_handles_tool_call,
    live_vercel_ai_gateway_openai_streams_text_deltas,
    live_vercel_ai_gateway_openai_handles_multiturn_with_tools,
    "vercel-ai-gateway",
    "openai/gpt-5.1-codex-max",
    "AI_GATEWAY_API_KEY"
);

#[tokio::test]
async fn live_huggingface_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_huggingface_reports_total_usage_components",
        "huggingface",
        "moonshotai/Kimi-K2.5",
        "HF_TOKEN",
    )
    .await
}

#[tokio::test]
async fn live_huggingface_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_huggingface_abort_reports_source_usage_shape",
        "huggingface",
        "moonshotai/Kimi-K2.5",
        "HF_TOKEN",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_together_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    let test = "live_together_reports_total_usage_components";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_text_options(Some(api_key));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_total_usage_with_options(test, &model, options).await
}

#[tokio::test]
async fn live_together_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    let test = "live_together_abort_reports_source_usage_shape";
    let Some(api_key) = live_api_key(test, "TOGETHER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("together", "moonshotai/Kimi-K2.6")
        .ok_or_else(|| "missing model registry entry: together/Kimi-K2.6")?;
    let mut options = live_abort_options(Some(api_key), Arc::new(AtomicBool::new(false)));
    options.reasoning = Some(ThinkingLevel::High);
    run_live_abort_tokens_then_new_message_with_options(
        test,
        &model,
        options,
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_zai_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_zai_reports_total_usage_components",
        "zai",
        "glm-4.5-air",
        "ZAI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_zai_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_zai_abort_reports_source_usage_shape",
        "zai",
        "glm-4.5-air",
        "ZAI_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_minimax_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_minimax_reports_total_usage_components",
        "minimax",
        "MiniMax-M2.7",
        "MINIMAX_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_minimax_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_then_new_message_smoke(
        "live_minimax_abort_reports_source_usage_shape",
        "minimax",
        "MiniMax-M2.7",
        "MINIMAX_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_xiaomi_reports_total_usage_components",
        "xiaomi",
        "mimo-v2.5-pro",
        "XIAOMI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_cn_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_xiaomi_token_plan_cn_reports_total_usage_components",
        "xiaomi-token-plan-cn",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_CN_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_ams_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_xiaomi_token_plan_ams_reports_total_usage_components",
        "xiaomi-token-plan-ams",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_xiaomi_token_plan_sgp_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_xiaomi_token_plan_sgp_reports_total_usage_components",
        "xiaomi-token-plan-sgp",
        "mimo-v2.5-pro",
        "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_kimi_coding_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_kimi_coding_reports_total_usage_components",
        "kimi-coding",
        "kimi-k2-thinking",
        "KIMI_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_kimi_coding_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_smoke(
        "live_kimi_coding_abort_reports_source_usage_shape",
        "kimi-coding",
        "kimi-for-coding",
        "KIMI_API_KEY",
        LiveAbortUsageExpectation::PositiveInputZeroOutput,
    )
    .await
}

#[tokio::test]
async fn live_vercel_ai_gateway_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_vercel_ai_gateway_reports_total_usage_components",
        "vercel-ai-gateway",
        "google/gemini-2.5-flash",
        "AI_GATEWAY_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_vercel_ai_gateway_abort_reports_source_usage_shape() -> Result<(), Box<dyn Error>> {
    run_live_abort_tokens_then_new_message_smoke(
        "live_vercel_ai_gateway_abort_reports_source_usage_shape",
        "vercel-ai-gateway",
        "google/gemini-2.5-flash",
        "AI_GATEWAY_API_KEY",
        LiveAbortUsageExpectation::ZeroInputOutput,
    )
    .await
}

#[tokio::test]
async fn live_openrouter_anthropic_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openrouter_anthropic_reports_total_usage_components",
        "openrouter",
        "anthropic/claude-sonnet-4",
        "OPENROUTER_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openrouter_deepseek_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openrouter_deepseek_reports_total_usage_components",
        "openrouter",
        "deepseek/deepseek-chat",
        "OPENROUTER_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openrouter_mistral_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openrouter_mistral_reports_total_usage_components",
        "openrouter",
        "mistralai/mistral-small-3.2-24b-instruct",
        "OPENROUTER_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openrouter_google_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openrouter_google_reports_total_usage_components",
        "openrouter",
        "google/gemini-2.0-flash-001",
        "OPENROUTER_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openrouter_llama_reports_total_usage_components() -> Result<(), Box<dyn Error>> {
    run_live_total_usage_smoke(
        "live_openrouter_llama_reports_total_usage_components",
        "openrouter",
        "meta-llama/llama-4-scout",
        "OPENROUTER_API_KEY",
    )
    .await
}

#[tokio::test]
async fn live_openrouter_preserves_cache_write_tokens_on_stream_path() -> Result<(), Box<dyn Error>>
{
    let test = "live_openrouter_preserves_cache_write_tokens_on_stream_path";
    let Some(api_key) = live_api_key(test, "OPENROUTER_API_KEY") else {
        return Ok(());
    };
    let model = get_model("openrouter", "google/gemini-2.5-flash")
        .ok_or_else(|| "missing model registry entry: openrouter/google/gemini-2.5-flash")?;
    let context = live_openrouter_cache_write_context();
    let mut options = live_text_options(Some(api_key));
    options.stream.temperature = Some(0.0);
    options.stream.max_tokens = Some(32);
    options
        .payload_hooks
        .push(Arc::new(OpenRouterCacheControlPayloadHook));

    let first = complete_simple(&model, context.clone(), options.clone()).await?;
    assert_eq!(
        first.stop_reason,
        StopReason::Stop,
        "{test} first request failed: {first:?}"
    );
    let second = complete_simple(&model, context, options).await?;
    assert_eq!(
        second.stop_reason,
        StopReason::Stop,
        "{test} second request failed: {second:?}"
    );
    assert!(
        first.usage.cache_write > 0 || second.usage.cache_write > 0,
        "{test} expected cache_write_tokens on at least one request: first={:?} second={:?}",
        first.usage,
        second.usage
    );
    Ok(())
}

fn live_openrouter_images_options(api_key: String) -> ImagesOptions {
    ImagesOptions {
        api_key: Some(api_key),
        timeout_ms: Some(120_000),
        max_retries: Some(1),
        max_retry_delay_ms: Some(1_000),
        ..Default::default()
    }
}

fn assert_live_images_success(test: &str, output: &AssistantImages) {
    assert_ne!(
        output.stop_reason,
        ImagesStopReason::Error,
        "{test} returned provider error: {:?}",
        output.error_message
    );
    assert_ne!(
        output.stop_reason,
        ImagesStopReason::Aborted,
        "{test} was aborted: {:?}",
        output.error_message
    );
    assert!(
        output
            .response_id
            .as_deref()
            .map(str::trim)
            .is_some_and(|response_id| !response_id.is_empty()),
        "{test} returned no response_id: {output:?}"
    );
}

fn assert_live_images_contain_image(test: &str, output: &AssistantImages) {
    assert!(
        output
            .output
            .iter()
            .any(|item| matches!(item, ImagesContent::Image(_))),
        "{test} returned no image output: {output:?}"
    );
}

fn assert_live_images_contain_text(test: &str, output: &AssistantImages) {
    assert!(
        output.output.iter().any(|item| {
            matches!(item, ImagesContent::Text(text) if !text.text.trim().is_empty())
        }),
        "{test} returned no text output: {output:?}"
    );
}

#[tokio::test]
async fn live_openrouter_images_invalid_api_key_reports_provider_error()
-> Result<(), Box<dyn Error>> {
    let test = "live_openrouter_images_invalid_api_key_reports_provider_error";
    if !live_network_enabled(test) {
        return Ok(());
    }
    let model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
        .ok_or_else(|| "missing image model registry entry: openrouter/google image")?;
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text(
                "Generate a tiny simple monochrome square icon.",
            )],
        },
        ImagesOptions {
            api_key: Some(LIVE_BOGUS_API_KEY.to_owned()),
            timeout_ms: Some(60_000),
            max_retries: Some(0),
            ..Default::default()
        },
    )
    .await?;

    assert_eq!(
        output.stop_reason,
        ImagesStopReason::Error,
        "{test} expected provider error, got {:?}: {output:?}",
        output.stop_reason
    );
    assert!(
        output
            .error_message
            .as_deref()
            .map(str::trim)
            .is_some_and(|error| !error.is_empty()),
        "{test} returned no provider error message: {output:?}"
    );
    Ok(())
}

#[tokio::test]
async fn live_openrouter_images_smoke_generates() -> Result<(), Box<dyn Error>> {
    let test = "live_openrouter_images_smoke_generates";
    let Some(api_key) = live_api_key(test, "OPENROUTER_API_KEY") else {
        return Ok(());
    };
    let model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
        .ok_or_else(|| "missing image model registry entry: openrouter/google image")?;
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text(
                "Generate a tiny simple monochrome square icon.",
            )],
        },
        live_openrouter_images_options(api_key),
    )
    .await?;

    assert_live_images_success(test, &output);
    assert_live_images_contain_image(test, &output);
    Ok(())
}

#[tokio::test]
async fn live_openrouter_images_text_plus_image_output() -> Result<(), Box<dyn Error>> {
    let test = "live_openrouter_images_text_plus_image_output";
    let Some(api_key) = live_api_key(test, "OPENROUTER_API_KEY") else {
        return Ok(());
    };
    let model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
        .ok_or_else(|| "missing image model registry entry: openrouter/google image")?;
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![ImagesContent::text(
                "Generate a simple red circle on a plain white background and include a brief description.",
            )],
        },
        live_openrouter_images_options(api_key),
    )
    .await?;

    assert_live_images_success(test, &output);
    assert_live_images_contain_image(test, &output);
    assert_live_images_contain_text(test, &output);
    Ok(())
}

#[tokio::test]
async fn live_openrouter_images_accepts_image_input() -> Result<(), Box<dyn Error>> {
    let test = "live_openrouter_images_accepts_image_input";
    let Some(api_key) = live_api_key(test, "OPENROUTER_API_KEY") else {
        return Ok(());
    };
    let model = get_image_model("openrouter", "google/gemini-2.5-flash-image")
        .ok_or_else(|| "missing image model registry entry: openrouter/google image")?;
    let output = generate_images(
        &model,
        ImagesContext {
            input: vec![
                ImagesContent::text("Create a variation of this image with a blue background."),
                ImagesContent::Image(ImageContent {
                    mime_type: "image/png".to_owned(),
                    data: LIVE_RED_PIXEL_PNG_BASE64.to_owned(),
                }),
            ],
        },
        live_openrouter_images_options(api_key),
    )
    .await?;

    assert_live_images_success(test, &output);
    assert_live_images_contain_image(test, &output);
    Ok(())
}
