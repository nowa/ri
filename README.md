# ri

`ri` is a Rust workspace that ports the core runtime pieces of
[pi](https://github.com/earendil-works/pi-mono) into Rust:

- `ri-llm-provider` is the Rust counterpart of `packages/ai` /
  `@earendil-works/pi-ai`.
- `ri-agent-core` is the Rust counterpart of `packages/agent` /
  `@earendil-works/pi-agent-core`.

The goal is behavioral compatibility with pi's LLM provider and agent runtime
semantics, expressed with Rust's type system, traits, serde data models, and
async streams instead of a direct TypeScript API translation.

## Packages

| Rust surface | Pi package | Purpose |
| --- | --- | --- |
| `ri-llm-provider` | `packages/ai` / `pi-ai` | Unified multi-provider LLM API, model registry, streaming events, tool calls, usage tracking, provider payloads, OAuth helpers, and image APIs. |
| `ri-agent-core` | `packages/agent` / `pi-agent-core` | Stateful agent runtime with event streaming, tool execution, message queues, lifecycle events, and turn orchestration. |
| `ri_agent_core::harness` | `packages/agent` harness utilities | Session storage, prompt formatting, compaction, skills/resources, local execution env, provider hooks, and harness-level orchestration utilities. |

## Status

The workspace contains the Rust implementation and local test coverage for the
core pi-ai and pi-agent-core behavior that is practical to verify without live
provider credentials.

The test suite covers provider metadata, payload generation, streaming parsers,
SSE and eventstream behavior, abort handling, response IDs, usage accounting,
message transforms, tool calling, reasoning/thinking controls, agent loop
control flow, stateful agents, custom stream providers, proxy streaming, tool
execution, compaction, resources, session storage, and harness utilities.

Live provider E2E tests are not run by default. Provider behavior is covered
locally through mock HTTP servers, payload assertions, parser tests, and stream
event tests.

## Migration Test Accounting

The migration target is pi's LLM provider and agent-core behavior, not every
test in the pi monorepo. The active source baseline is the direct simple
`it/test` cases under:

- `packages/ai/test`: 721 cases.
- `packages/agent/test`: 150 cases.

That 871-case baseline intentionally excludes `packages/coding-agent`, skipped
source cases, and statically unexpanded `it.each` / `test.each` declarations.

Rust test totals must not be used as a one-to-one completion signal. The Rust
suite currently includes both:

- **Pi exact case parity**: behavior that corresponds to a specific pi-ai or
  pi-agent-core source test case.
- **Rust-specific coverage**: behavior needed because ri owns Rust-native
  implementations for HTTP transport, proxy construction, OAuth auth storage,
  proxy stream forwarding, streaming parsers, session storage, and harness
  integration.

New tests should be added only when they cover a missing Pi exact case or a
clearly necessary Rust-specific contract. Tests that inspect Rust source,
TypeScript source, Markdown files, `Cargo.toml`, or test names to prove
coverage should not be added. Coverage claims should come from behavior tests,
gated live tests, and explicit notes in [MIGRATION_STATUS.md](MIGRATION_STATUS.md).

This migration is still not certified complete by count alone. Strict external
parity still requires running the gated provider live/E2E matrix with real
credentials, local model services where applicable, and manual OAuth flows.

## Features

`ri-llm-provider` includes:

- Built-in model lookup with `get_model(provider, model_id)`.
- `stream`, `complete`, `stream_simple`, and `complete_simple` APIs.
- OpenAI, Azure OpenAI Responses, OpenAI Codex, Anthropic, Google, Vertex AI,
  Mistral, Amazon Bedrock, GitHub Copilot, OpenRouter, OpenAI-compatible, and
  related compatibility layers.
- Tool calling with streamed partial arguments.
- Text, image, thinking, and tool-result content blocks.
- Reasoning levels including `off`, `minimal`, `low`, `medium`, `high`, and
  `xhigh`, with provider-specific wire mappings.
- Usage and cost accounting, response IDs, diagnostics, abort flags, retry
  configuration, session IDs, and provider-specific extra options.
- OAuth helpers for providers such as OpenAI Codex, Anthropic, GitHub Copilot,
  and Google Vertex.

`ri-agent-core` includes:

- Stateful `Agent` and lower-level `agent_loop` APIs.
- Event streaming for agent, turn, message, and tool execution events.
- Parallel and sequential tool execution.
- Tool call and tool result hooks.
- Steering and follow-up queues.
- Context transforms and custom `AgentMessage` conversion before LLM calls.
- Custom stream providers, including proxy streaming through `/api/stream`.
- Prompt templates, skills/resources loading, session storage, compaction, and
  local execution environment utilities.

`ri_agent_core::harness` includes:

- `AgentHarness`, a higher-level runtime facade around `Agent`.
- Persistent and in-memory session storage.
- System prompt formatting with model, thinking level, active tools, resources,
  session, and local environment context.
- Skills and prompt template loading.
- Branch summary and context compaction helpers.
- Local execution environment utilities for file and shell operations.
- Provider auth, request, and payload hooks.
- Before-agent-start, context, tool-call, and tool-result hooks.
- Model selection, thinking-level selection, resource updates, queue updates,
  save points, aborts, and settled lifecycle events.

## Quick Start

Add the crates as path dependencies inside this repository or as Git
dependencies from `https://github.com/nowa/ri.git`.

```toml
[dependencies]
ri-llm-provider = { git = "https://github.com/nowa/ri.git" }
ri-agent-core = { git = "https://github.com/nowa/ri.git" }
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### Complete One LLM Request

```rust
use ri_llm_provider::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = get_model("openai", "gpt-5-mini").expect("model");

    let context = Context {
        system_prompt: Some("You are helpful.".to_owned()),
        messages: vec![Message::User(UserMessage::text("Hello"))],
        tools: vec![],
    };

    let mut options = StreamOptions::default();
    options.api_key = std::env::var("OPENAI_API_KEY").ok();

    let message = complete(&model, context, options).await?;
    println!("{message:#?}");

    Ok(())
}
```

### Run A Stateful Agent

```rust
use ri_agent_core::*;
use ri_llm_provider::*;

#[tokio::main]
async fn main() -> Result<(), String> {
    let model = get_model("openai", "gpt-5-mini").expect("model");
    let agent = Agent::new(AgentOptions::new(model));

    agent.subscribe(|event| {
        println!("{event:?}");
    });

    agent.prompt("Hello from ri").await?;
    Ok(())
}
```

## Core Concepts

### Provider Context

`ri-llm-provider` uses the same conceptual context as pi-ai:

- `system_prompt`
- ordered `messages`
- optional tool definitions
- assistant output made of typed content blocks
- tool results that can be fed back into later turns

The Rust representation uses enums and structs such as `Context`, `Message`,
`AssistantContent`, `ToolCall`, `ToolResultMessage`, and `Usage`.

### Agent Messages

`ri-agent-core` distinguishes between flexible agent-level messages and LLM
messages:

```text
AgentMessage[] -> transform_context -> AgentMessage[] -> convert_to_llm -> Message[] -> LLM
```

This mirrors pi-agent-core's `AgentMessage` vs LLM message model. Custom
application messages can be kept in agent state and filtered or converted before
each provider request.

### Events

Agent runs emit events for:

- `AgentStart` / `AgentEnd`
- `TurnStart` / `TurnEnd`
- `MessageStart` / `MessageUpdate` / `MessageEnd`
- `ToolExecutionStart` / `ToolExecutionEnd`

The `Agent` updates its state from these events and notifies sync or async
subscribers.

### Agent Harness

The harness is a core part of `ri-agent-core`, but it is intentionally kept as
the `ri_agent_core::harness` module instead of a separate crate. It sits above
the lower-level `Agent` and owns the application-facing runtime concerns that a
coding agent or UI needs around the raw agent loop.

`AgentHarness` combines:

- `Session` state and storage.
- `LocalExecutionEnv` file and shell utilities.
- model and thinking-level selection.
- active tool filtering and tool execution mode.
- loaded skills and prompt templates.
- generated system prompts.
- provider authentication and stream option patching.
- provider payload patching before network dispatch.
- message queues for steering, follow-up, and next-turn work.
- save-point and settlement events for durable app state.

This keeps crate boundaries simple while still making the harness API visible as
a first-class Rust surface.

## Development

Run all tests:

```bash
cargo test --workspace -- --test-threads=1
```

Useful focused commands:

```bash
cargo test -p ri-llm-provider
cargo test -p ri-agent-core
cargo fmt
```

When changing migration coverage, update
[MIGRATION_STATUS.md](MIGRATION_STATUS.md) with the exact parity or
Rust-specific reason instead of relying on the raw test count.

The workspace currently targets Rust 2024 edition.

## Contributing And Security

- See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull request
  guidelines.
- See [SECURITY.md](SECURITY.md) for vulnerability reporting.
- See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency license
  notes.

## Compatibility Notes

This is not a source-compatible port of the TypeScript API. The compatibility
target is pi's behavior and protocol shape:

- provider payload semantics
- stream event ordering
- tool call and tool result behavior
- reasoning/thinking controls
- context handoff between providers
- abort and error behavior
- agent loop and state transitions

TypeScript-specific pieces such as TypeBox declarations and dynamic module
loading are represented with Rust-native equivalents where applicable, primarily
serde, `serde_json::Value`, traits, and explicit provider modules.

## License

MIT
