# Migration Status

Objective: build Rust crates in this `ri` directory that correspond to
`pi-agent-core` and required dependencies such as `pi-ai`, named with the `ri-*`
pattern, and finish only when all `pi-agent-core` / `pi-ai` tests have Rust
counterparts that pass.

## Source Scope

- Source agent package: `/home/nowa/Projects/src/pi/packages/agent`
- Source AI package: `/home/nowa/Projects/src/pi/packages/ai`
- Source agent tests counted: 16 `*.test.ts` files, 150 direct simple `it/test`
  cases.
- Source AI tests counted: 68 `*.test.ts` files, 721 direct simple `it/test`
  cases after excluding direct `it.skip`, `it.skipIf`, and `it.each`/`test.each`
  declarations.
- Current simple baseline denominator: 871 direct source `it/test` cases across
  `packages/agent` and `packages/ai`. This excludes 8 explicit skipped simple
  cases, 126 direct conditional `it.skipIf` live cases, and 7
  `it.each`/`test.each` declarations that are not statically expanded in this
  baseline. It intentionally does not subtract the ordinary `it(...)` cases
  nested under the 221 conditional `describe.skipIf` live suites; those
  suite-level live conditions are tracked in this document and represented by
  gated Rust live tests, but they are not part of the simple denominator.
- This source scope intentionally excludes `packages/coding-agent`, which has
  its own much larger test suite: 121 `*.test.ts` files and about 1,233 broad
  static `it/test` declarations.

## Rust Artifacts Created

- `ri-llm-provider`
  - Core message/content/model/event types.
  - `AssistantMessageEventStream` source parity for terminal-event result
    settlement, ignored post-terminal pushes, and explicit `end(result)`
    closure without a terminal event.
  - API provider registry.
  - `stream`, `complete`, `stream_simple`, `complete_simple`.
  - Built-in model registry seed and thinking-level helpers.
  - Faux provider with queued responses, multi-model registrations,
    model-aware response factories, event deltas, terminal error/abort events,
    abort flags before and during paced streams, usage estimates, session cache
    simulation, and unregister behavior. Faux response factories can inspect
    `SimpleStreamOptions` for helper tests.
  - Rust-native `providers/simple-options.ts` parity for simple stream defaults:
    default `max_tokens` selection, 32k output cap when model output reaches the
    context window, explicit caller override preservation, xhigh-to-high budget
    clamping, and thinking-budget/max-token adjustment for budget-based
    reasoning models.
  - JSON repair/hash helpers, including malformed control-character repair,
    invalid escape repair, conservative partial streamed tool-argument parsing
    for common incomplete object/array/string cases, exact `shortHash` output
    fixtures used for foreign OpenAI Responses item IDs, and unpaired UTF-16
    surrogate escape replacement.
  - Assistant-message diagnostics helpers mirroring `utils/diagnostics.ts`,
    represented as typed Rust structs that serialize to the pi
    `{ type, timestamp, error, details }` diagnostic shape.
  - JSON-schema subset validation and coercion, including pi-style primitive
    coercion plus `allOf`/`anyOf`/`oneOf`, `enum`/`const`, object
    `additionalProperties`, tuple array items, and common string/number/array
    bounds used by TypeBox/plain JSON-schema tool parameters.
  - Context-overflow detection with provider-specific error-shape corpus and
    silent/length-stop overflow signals.
  - Environment API key lookup.
  - GitHub Copilot dynamic request headers from `providers/github-copilot-headers.ts`:
    user-vs-agent `X-Initiator`, `Openai-Intent`, vision-request detection for
    user/tool-result images, and caller header overrides across Anthropic,
    OpenAI Responses, and OpenAI Completions paths.
  - HTTP proxy URL resolution with `NO_PROXY` and `npm_config_*` handling,
    unsupported protocol errors, proxy-aware `reqwest` client construction for
    provider, image, Bedrock runtime ConverseStream, Google ADC token-refresh,
    Anthropic/OpenAI Codex OAuth authorization-code token exchanges,
    Anthropic/GitHub Copilot/OpenAI Codex OAuth token-refresh HTTP paths,
    GitHub Copilot OAuth device-flow requests, plus OpenAI Codex WebSocket HTTP
    CONNECT proxy tunneling.
  - Bedrock endpoint/region config helpers plus Converse payload helpers for
    message conversion, Claude thinking fields, GovCloud display omission, and
    application-inference-profile cache points, including image tool-result
    content, streamed Converse block aggregation, usage mapping, stop reasons,
    and SDK exception error events.
  - Azure OpenAI base URL/config normalization, default API version,
    deployment-name map resolution, and Responses payload construction with
    deployment overrides, tools, session cache keys, and reasoning options.
  - Google Vertex API-key/ADC/custom-base-URL client config helpers.
  - Google/Gemini shared tool conversion helpers with OpenAPI schema sanitization,
    message conversion, multimodal function response routing, thought-signature
    handling, `streamSimple` thinking payload construction/disable rules, and
    streamed chunk aggregation for response IDs, text/thinking signatures,
    function calls, usage, and safety/error finish reasons.
  - Fireworks and Together model metadata/base-URL/provider compatibility overrides.
  - OpenCode Zen/Go model metadata/base-URL/provider compatibility overrides.
  - Source-compatible base URL/API/header mappings for additional
    OpenAI-compatible and Anthropic-compatible providers used by live source
    tests: xAI, Groq, Cerebras, Hugging Face, Together, z.ai, MiniMax, Kimi
    Coding, Vercel AI Gateway, Xiaomi MiMo token-plan variants, and OpenRouter
    text backends.
  - Mistral chat payload helpers for tool schema serialization, image
    tool-result content, cross-provider tool-call ID normalization, missing
    tool-result synthesis, request header/session-affinity handling, stream
    chunk aggregation, response IDs, usage mapping, and reasoning mode selection.
  - Cross-provider message transform helpers: image downgrade, thinking cleanup,
    tool-call ID normalization, orphaned tool-result synthesis.
  - Anthropic raw SSE parsing helpers with malformed JSON repair, streamed
    tool-argument parsing, response IDs, usage mapping, partial usage
    preservation, and unknown event filtering.
  - Anthropic Messages payload helpers for eager tool input streaming compatibility,
    fine-grained streaming beta headers, thinking disable/adaptive payloads, and
    cache-control markers/TTL for system prompts, final user turns, and tools,
    assistant/tool-result replay with image tool-result content, including
    Fireworks tool-field compatibility omissions.
  - Anthropic client/header config helpers for GitHub Copilot Claude Bearer auth,
    Copilot static/dynamic headers, vision request detection, Fireworks
    session-affinity headers, Cloudflare AI Gateway `cf-aig-authorization`
    and BYOK upstream authorization preservation, and adaptive-model
    interleaved-thinking beta omission.
  - Anthropic Claude Code tool-name casing helpers, including OAuth payload,
    assistant-history replay, stream inbound restoration, and built-in provider
    wiring integration.
  - Anthropic OAuth helpers for PKCE generation, authorization URL
    construction, local callback server/state validation with pi-style
    success/error HTML pages and escaped details, manual redirect input login
    flow, token/refresh JSON requests, localhost callback preservation, token
    response parsing, and proxy-aware async authorization-code/refresh token
    exchange primitives.
  - OpenAI Codex OAuth helpers for authorization URL construction,
    local callback server/state validation, callback-driven login flow,
    form-encoded token/refresh requests, refresh failure message formatting,
    and proxy-aware async authorization-code/refresh token exchange primitives.
  - OpenAI Codex Responses helpers for ChatGPT JWT account-id extraction,
    SSE/WebSocket headers, request-body construction, URL resolution, reasoning
    effort mapping, cached WebSocket input-delta continuation, SSE frame parsing
    and retry/backoff request handling, WebSocket debug-stat accounting, and
    service-tier usage cost resolution. WebSocket transport fallback now writes
    pi-shaped `provider_transport_failure` diagnostics with nested `error` and
    `details` fields.
  - Session-scoped provider resource cleanup now covers the OpenAI Codex
    WebSocket session cache. This is the Rust-native counterpart of
    `session-resources.ts`: callers explicitly await `cleanup_session_resources`
    for one session or all sessions, and cached Codex WebSocket continuations
    are removed before later requests can reuse stale context.
  - GitHub Copilot OAuth helpers for device-flow request construction,
    slow-down-aware polling intervals, enterprise domain normalization,
    Copilot token refresh headers, base-URL derivation, proxy-aware async
    device-code/access-token poll/token-refresh primitives, complete
    device-flow orchestration with refresh-token exchange, and post-login
    model-policy enable requests for known GitHub Copilot models.
  - OAuth provider metadata registry for built-in Anthropic, GitHub Copilot,
    and OpenAI Codex providers, including callback-server markers and
    source-style register/unregister/reset behavior. The live external
    requirements manifest is guarded against this built-in provider set so
    each built-in OAuth provider has a stored-token auth-storage requirement
    before its live paths can be considered covered.
  - Source-compatible `~/.pi/agent/auth.json` auth storage resolution for API
    keys and OAuth credentials, including expired-token refresh/writeback,
    private file permissions, unknown OAuth provider pre-refresh validation,
    Anthropic/OpenAI Codex callback and manual-input login-to-auth-storage
    round trips, and GitHub Copilot device-flow auth-storage round trips with
    `enterpriseUrl` preservation.
  - OpenAI Responses stream and message conversion helpers for function-call
    partial JSON cleanup, foreign tool-call ID normalization, tool-result
    images, prompt-cache fields, session-affinity headers, default reasoning
    payload rules, function tools with strict defaults, text deltas/signature
    replay, incomplete terminal events, aborted reasoning history pruning,
    same-provider model handoff item-id omission, empty assistant-turn pruning,
    and service-tier usage cost multipliers.
  - OpenAI Completions payload helpers for empty-tools omission, tool choice,
    strict-mode compatibility, provider reasoning fields, z.ai tool streaming,
    Anthropic-style cache-control markers, tool-result image replay, and
    thinking-as-text replay, prompt-cache fields, session-affinity headers,
    Cloudflare AI Gateway `cf-aig-authorization` and BYOK upstream
    authorization preservation, empty user-block pruning, stream usage parsing,
    streamed text/thinking/tool delta aggregation, finish-reason mapping, and
    routed response model metadata.
  - Images API provider registry and `generate_images` dispatch, plus
    the full 28-model OpenRouter image model registry from the source generated
    catalog and image payload/response helpers for chat-completions image
    generation, inline data URLs, caller-supplied authorization/header
    preservation, usage, payload/response hooks,
    request timeout behavior, retry/backoff handling for retryable HTTP/network
    failures, non-retryable HTTP errors, invalid JSON responses, and
    aborted/error result mapping.
  - JSONL session metadata loading now reads only the first non-empty header
    line, matching the source line-reading metadata path without requiring the
    whole session file to decode successfully.
  - Gated live provider smoke tests for OpenAI Responses, OpenAI Completions,
    Anthropic Messages, Google/Gemini, Google Vertex API-key/ADC,
    Mistral Conversations, Azure OpenAI Responses, Bedrock Converse, and
    OpenRouter Images, plus gated live
    `response_id`, abort/token-usage shape across the non-skipped source
    `tokens` provider set, immediate pre-abort parity across the source
    `abort.test.ts` provider set, Bedrock abort-then-new-message parity from
    `abort.test.ts`, total-usage component consistency across the source
    `total-tokens` provider set, empty-message handling across the source
    `empty` provider set, orphaned tool-call-without-result handling across
    the source `tool-call-without-result` provider set, Unicode/emoji
    tool-result handling across the source `unicode-surrogate` provider set
    for Rust-representable strings, user image-input handling across the
    source `stream.test.ts` image-capable provider set, image tool-result
    handling across the source `image-tool-result` provider set, Responses API
    tool-result image `function_call_output` payload placement, Anthropic OAuth tool-name
    normalization, context-overflow live detection across the source remote
    and local provider sets, Cloudflare Workers AI and Cloudflare AI Gateway Workers
    `/compat` `stream.test.ts` live parity, Cloudflare AI Gateway
    OpenAI/Anthropic BYOK `stream.test.ts` live parity, OpenAI
    Responses/OpenAI Codex cache-affinity E2E, OpenAI Responses reasoning
    replay E2E, OpenRouter cache-write E2E, Anthropic Opus 4.7 reasoning smoke,
    Anthropic/Bedrock interleaved-thinking E2E, provider thinking-disable
    E2E, Anthropic Messages eager tool input streaming and long
    cache-retention E2E probes across the generated Anthropic-compatible
    provider catalog, Google Vertex ADC streaming-delta, thinking-delta,
    multi-turn context, live tool-call streaming, and total-usage checks,
    source `stream.test.ts` positive live matrices for OpenAI Responses
    `gpt-5.4`, Google/Gemini thinking/tool-followup, Bedrock
    streaming/thinking/tool-followup, DeepSeek, xAI, Groq, Cerebras,
    Hugging Face, Together, OpenRouter, z.ai, Mistral
    Devstral/Magistral/Pixtral, MiniMax, Kimi Coding, Xiaomi MiMo token-plan
    variants, Vercel AI Gateway Google/Anthropic/OpenAI routes, and
    local Ollama `gpt-oss:20b` positive stream parity, stored-token OAuth
    `stream.test.ts` matrices for Anthropic OAuth Sonnet/Opus, GitHub Copilot
    OpenAI/Anthropic, and OpenAI Codex SSE/WebSocket, and live provider-error
    checks for OpenAI Responses, OpenAI Completions,
    Anthropic, Google/Gemini, Mistral, and Azure OpenAI Responses. Bedrock has smoke,
    opt-in extensive per-model catalog smoke,
    empty-message, orphaned tool-call,
    Unicode/emoji tool-result, tool-call, streaming, thinking, tool-followup,
    immediate abort, abort-then-new-message, abort/token-usage, and
    total-usage live coverage. The
    OpenRouter Images live coverage asserts `response_id`, provider-error
    mapping, text+image output, and image-input generation. Anthropic OAuth,
    GitHub Copilot OAuth, and OpenAI Codex OAuth also have gated auth-storage
    live smoke/`response_id`/source `stream.test.ts` basic/tool/stream/thinking
    and follow-up coverage/empty-message/orphaned-tool-call/Unicode
    tool-result/image tool-result/immediate-abort/abort-token/total-usage coverage via
    `~/.pi/agent/auth.json`.
    These default to a no-network pass unless `RI_LIVE_PROVIDER_TESTS=1` and
    provider credentials are present. `RI_LIVE_PROVIDER_STRICT=1` makes any
    credential/service skip fail, turning the suite into a hard proof gate for
    a fully configured environment. Gate parsing/default-off behavior,
    strict-skip behavior, strict readiness reporting, and stored-OAuth
    `auth.json` readiness are covered by behavior tests. The previous
    migration meta-tests that scanned `provider_live.rs` or this status file to
    prove live-test matrix entries have been removed; live coverage should now
    be judged by the actual gated live tests and strict readiness behavior.
  - Usage helpers for enforcing `total_tokens == input + output + cache_read +
    cache_write` across provider usage parsers.

- `ri-agent-core`
  - Agent message/tool/state/event types.
  - Low-level prompt and continue agent loop with sequential tool execution,
    tool-result continuation turns, tool argument preparation, Pi-style
    schema validation/coercion before `beforeToolCall`, tool-call hooks
    with argument replacement and Pi-style pre-execution blocking, tool-result
    replacement plus field-level content/details/terminate/is-error patch
    hooks, hook-driven tool-result termination, partial tool execution update
    callbacks/events whose start/update argument payloads preserve the
    assistant's raw tool-call arguments, tool-result message
    lifecycle events, and assistant/tool-call/current-context hook snapshots
    for low-level `beforeToolCall`/`afterToolCall` parity. Transform/convert
    hooks for LLM request context. Prepare-next-turn hooks can
    replace the next-turn context/model/thinking state after `turn_end`, and
    should-stop-after-turn hooks can stop before the next provider request.
    Provider streams that close via `end(result)` without an explicit
    `done`/`error` terminal event use the final stream result before ending the
    assistant message, matching `streamAssistantResponse`.
    Terminal assistant responses with `error` or `aborted` stop immediately
    after `turn_end`, without running next-turn hooks or polling
    steering/follow-up queues, matching `agent-loop.ts`.
    Queued messages can be injected before the initial provider request or
    after all tool-result messages are written and before the next LLM request,
    and follow-up messages can restart the loop after the agent would otherwise
    stop. Tool batches terminate only when every finalized result terminates.
    Parallel tool execution is the default, emits completion events in
    completion order while persisting tool-result messages in source order, and
    sequential batches emit each tool's end and tool-result message before the
    next tool's start, including when per-tool sequential mode overrides a
    parallel loop config.
  - Stateful `Agent` wrapper with default/custom state initialization,
    thinking-level and session-id forwarding, subscriptions, text/image
    prompt normalization, prompt-message runs, source-compatible busy/continue
    validation errors, and steering/follow-up queues with one-at-a-time/all
    drain modes plus queued state introspection, queue clearing/reset helpers,
    abort handles that cancel active provider streams without clearing
    low-level `Agent` queues, and provider start failures persisted as
    assistant error messages with lifecycle events.
  - Custom stream-provider hooks for the low-level loop and `Agent` wrapper,
    dynamic per-request API key providers for low-level LLM calls with static
    key fallback, including a Rust-native `agent/src/proxy.ts` counterpart that
    posts model, context, and proxy-safe stream options to `/api/stream`,
    reconstructs SSE assistant events, preserves abort behavior, and keeps API
    keys and retry policy out of the proxy payload.
  - Harness basics: system prompt helper, system skill formatting, UTF-8/line
    truncation, token/usage estimate, compaction predicates, cut-point
    selection, compaction preparation, file-operation metadata extraction, and
    summarization conversation serialization/generation, compact execution, and
    branch-summary collection/preparation/generation, UUID v7 session id.
  - High-level `AgentHarness` implementation with env/session/model access,
    thinking-level and queue-mode state, subscriptions, queue update events,
    `before_agent_start` message/system-prompt hooks, before/after lifecycle
    hook error-path pending-write flushes, `after_agent_finish`
    observation/persisted-message hooks including provider-start failure and
    aborted-turn settlement,
    `context` hooks with
    assistant-error persistence on hook failure, `tool_call`/`tool_result`
    hooks through the direct loop including blocked tool-call error results and
    tool-result error-flag patching, fixed skill/prompt-template resources with
    `resources_update` events, source-style resource `source` metadata
    preservation, direct skill and prompt-template invocation,
    stream-options accessors, model/thinking selection events with session
    persistence, listener-safe pending `append_message`, custom-entry,
    custom-message, label, and session-name writes, dynamic system-prompt
    providers over resources, live listener events and
    prepare-next-turn refresh for model/thinking/system prompt/resources/tools,
    `save_point` events for flushed pending writes, `next_turn`
    injection/persistence, tool/active-tool state management with request-time
    active-tool filtering, running-turn steering/follow-up queues, abort queue
    clearing that preserves next-turn messages, source-parity
    `abort_and_wait` idle/event-listener settlement, idle waiting, and
    session prompt-context conversion that preserves LLM messages,
    custom messages, branch summaries, and compaction summaries,
    high-level session compaction that persists generated summaries through the session
    with `session_before_compact` cancellation/provided-summary hooks and
    hook success/error and generation-error pending-write flushes, session
    compact events, plus high-level
    session branch moves that generate and persist branch summaries with
    `session_before_branch_summary` supplied-summary/skip hooks,
    hook success/error and generation-error pending-write flushes, branch-summary
    events, and Pi-style `navigateTree` behavior through `navigate_tree` with
    `session_before_tree` cancellation/provided-summary hooks, `session_tree`
    events, target user/custom-message editor text restoration, and target-parent
    leaf movement.
  - In-memory and JSONL session storage/repositories with branching, labels,
    metadata, context building, full-session fork, before-target user-message
    fork, at-target fork, invalid-target errors, and source-shaped
    `bashExecution` message entries with LLM-context formatting,
    `excludeFromContext` filtering, JSONL round trips, token estimates, and
    compaction turn-start handling.
  - Skill and prompt-template invocation formatting plus skill/prompt-template
    loading, pi-style numeric/slice/all-argument substitution including
    multi-digit numeric placeholders, skill ignore-file handling, and
    pi-style skill metadata validation diagnostics for name/description rules.
  - Local execution environment foundation for filesystem operations, shell
    execution with per-command working-directory overrides and shell
    environment overrides, stdout/stderr callbacks, callback error propagation
    that terminates running commands, command timeout/abort handling,
    pre-aborted file-operation cancellation, symlink directory entries,
    text-line read limits, stable file error mapping, shell spawn error
    mapping, best-effort cleanup, and streaming shell-output capture/truncation
    with cancellation results and full-output log files.

## Rust Test Coverage Now

Current Rust tests: 1123 enumerated by `cargo test --workspace -- --list`.

- `ri-llm-provider`: 928 tests: 1 library test, 296 `provider_core` tests, and
  631 `provider_live` tests. This is 207 above the 721 direct simple source
  cases counted under `packages/ai/test`, because the Rust suite also includes
  Rust-specific registry, HTTP, proxy, transport, OAuth auth-storage, and gated
  live/E2E coverage.
- `ri-agent-core`: 195 tests across `agent_core`, `agent_harness`,
  `execution_env`, `harness_compaction`, `harness_truncate`, `proxy`,
  `resources`, and `session_storage`. This is 45 above the 150 direct simple
  source cases counted under `packages/agent/test`, because several Rust tests
  cover grouped source behavior plus Rust-specific session, harness, and
  execution-environment contracts.
- On 2026-05-20, 52 migration meta/audit/implementation-shape tests were
  removed. Those tests read Rust source, TypeScript source,
  `MIGRATION_STATUS.md`, or `Cargo.toml` to prove that source case titles,
  evidence markers, live-test runner names, documentation entries, source
  metadata, or implementation details existed. The remaining coverage claims
  should be based on behavior tests and explicit status notes, not tests that
  inspect the test suite, implementation source, or this document for proof.
- The proxy-aware `reqwest`/raw-TCP source-scanner tests were removed as
  over-specific implementation-shape tests.

## Source Parity Audit Notes

- The active source denominator remains 871 direct simple `it/test` cases: 721
  under `packages/ai/test` plus 150 under `packages/agent/test`. The
  `packages/coding-agent` suite is intentionally outside this migration scope.
- Provider behavior coverage now includes built-in HTTP providers for OpenAI
  Responses, OpenAI Completions, OpenAI Codex, Anthropic, Bedrock,
  Google/Gemini, Google Vertex, Mistral, Azure OpenAI Responses, OpenRouter
  Images, proxy-aware networking, OAuth auth-storage, Bedrock credential
  resolution/signing, streaming/SSE/eventstream parsing, payload transforms,
  model registries, and gated live provider matrices.
- Agent behavior coverage now includes the advanced loop, queue handling,
  stateful wrapper, high-level `AgentHarness` hooks, compaction and branch
  summary persistence, JSONL/session storage, resources, prompt templates,
  skills, truncation, and local execution environment behavior.
- The raw 1123-vs-871 count is not completion proof. Rust tests sometimes
  aggregate several source assertions, some source cases are Node/SDK-loader
  specific, and many provider live/E2E tests require credentials, local
  services, or manual OAuth interaction before they prove external parity.

## Completion Audit

This migration is not complete.

- Local behavior-test coverage is substantially broader than the original
  baseline, but strict external proof is still missing for the full provider
  live matrix with real credentials, local model services, and manual browser or
  device OAuth flows.
- Remaining provider risk is case-by-case semantic parity for provider-specific
  payload transforms, streaming edge cases, OAuth refresh/writeback behavior,
  image API networking, proxy behavior, and live E2E flows that cannot be
  certified by default-off gated tests alone.
- Remaining agent risk is case-by-case semantic parity for advanced abort/error
  termination paths, async listener settlement, lifecycle hook ordering, and
  session/harness integration edge cases even where Rust behavior tests now
  cover the main contracts. High-level compaction and branch-summary
  persistence hooks have direct Rust behavior coverage, including hook removal,
  supplied-summary, cancel/skip, error, event, and JSONL persistence paths.
- Latest local verification on 2026-05-21 after aligning Pi
  `utils/event-stream.ts` and low-level `streamAssistantResponse` EOF result
  behavior from `agent-loop.ts`: Rust event streams now ignore pushes after a
  terminal `done`/`error` event, close iteration on `end(result)`, and the
  low-level agent loop uses the stream result when a provider closes without an
  explicit terminal event:
  `cargo fmt`,
  `cargo test -p ri-llm-provider --test provider_core assistant_event_stream_ignores_pushes_after_terminal_event -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core assistant_event_stream_end_closes_without_terminal_event -- --exact`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_uses_stream_result_when_provider_ends_without_terminal_event -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core -- --test-threads=1`,
  `cargo test -p ri-agent-core -- --test-threads=1`, and
  `cargo test --workspace -- --list` passed; the list command enumerated 1123
  tests.
- Previous local verification on 2026-05-21 after aligning Pi low-level
  `afterToolCall` field-level patch semantics from `agent-loop.ts`: Rust hook
  results can now patch only `content`, `details`, `terminate`, or `isError`
  while preserving omitted fields, so a Pi-style `{ terminate: true }` result
  can stop the tool batch without replacing the original tool output:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_tool_result_hook_can_terminate_tool_batch -- --exact`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_runs_tool_call_and_tool_result_hooks -- --exact`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_tool_result_hook_can_override_error_flag -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_runs_tool_call_and_tool_result_hooks_through_direct_loop -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_tool_result_hook_can_override_error_flag -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1120 tests.
- Previous local verification on 2026-05-21 after aligning Pi low-level
  `validateToolArguments` ordering from `agent-loop.ts`: Rust now validates and
  coerces prepared tool arguments before tool-call hooks or executor calls,
  turns validation failures into error tool results, and preserves the existing
  Pi behavior that hook-replaced arguments are not schema-revalidated:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_validates_tool_arguments_before_hooks_and_execution -- --exact`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_executes_hook_replaced_args_without_schema_revalidation -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core validation_ -- --test-threads=1`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1120 tests.
- Previous local verification on 2026-05-21 after aligning Pi low-level
  terminal assistant `error`/`aborted` behavior from `agent-loop.ts`: Rust now
  emits `turn_end` and then ends the run without calling `prepareNextTurn`,
  `shouldStopAfterTurn`, steering polling after the response, follow-up polling,
  or a second provider request:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_terminal_assistant_error_or_abort_skips_next_turn_hooks_and_queues -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1119 tests.
- Previous local verification on 2026-05-21 after aligning Pi low-level
  `agentLoopContinue` empty-context validation from `agent-loop.ts`: Rust now
  returns `Cannot continue: no messages in context`, matching the source
  thrown error instead of using the older Rust-only wording:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_continue_validates_context_tail -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1118 tests.
- Previous local verification on 2026-05-21 after aligning Pi low-level
  `Agent.abort()` queue behavior from `agent.ts`: Rust `Agent::abort()` now
  only cancels the active run and preserves queued steering/follow-up messages;
  explicit clear helpers and `reset` still clear queues, while high-level
  harness abort keeps its Pi-style queue clearing behavior:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core agent_has_queued_messages_tracks_steering_follow_up_and_clears -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1118 tests.
- Previous local verification on 2026-05-21 after aligning Pi low-level
  `getApiKey` behavior from `agent-loop.ts`, `agent.ts`, and `types.ts`: Rust
  `AgentLoopConfig`/`AgentOptions` now accept an async API key provider, resolve
  it before each low-level provider request, override the static stream
  `api_key` when a refreshed key is returned, and fall back to the static key
  when the provider returns `None`:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_resolves_dynamic_api_key_before_each_provider_request -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1118 tests.
- Previous local verification on 2026-05-21 after aligning Pi sequential
  `executeToolCallsSequential` lifecycle ordering from `agent-loop.ts`: Rust
  sequential tool batches now emit each tool's `tool_execution_end` and
  tool-result message events before emitting the next tool's
  `tool_execution_start`, while still delaying context mutation until the
  batch's tool results are collected:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core -- --test-threads=1`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1117 tests.
- Previous local verification on 2026-05-21 after aligning Pi
  `tool_execution_start` / `tool_execution_update` event argument semantics
  from `agent-loop.ts`: Rust now emits tool start before argument preparation
  or `beforeToolCall`, exposes the assistant's raw tool-call arguments in
  start/update events, and still executes tools with validated/prepared or
  hook-replaced arguments:
  `cargo fmt`, `cargo fmt --check`, `git diff --check`,
  `cargo test -p ri-agent-core --test agent_core -- --test-threads=1`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1116 tests.
- Previous local verification on 2026-05-21 after aligning low-level Pi
  `beforeToolCall` / `afterToolCall` hook context snapshots from
  `agent-loop.ts` and `types.ts`: Rust tool-call and tool-result hooks now
  receive the assistant message, raw tool-call block, validated/current input,
  and current agent context snapshot containing the prompt and assistant
  tool-call message before tool-result messages are appended:
  `cargo fmt`, `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_tool_hooks_receive_assistant_and_context_snapshot -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `git diff --check`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1116 tests.
- Previous local verification on 2026-05-21 after aligning Pi `afterToolCall`
  / harness `tool_result` `isError` semantics from `agent-loop.ts`,
  `types.ts`, and `harness/agent-harness.ts`: tool-result hooks now receive
  the current error flag and can override it while replacing result content,
  details, and termination, so recovered failed tools can continue as
  non-error tool results:
  `cargo fmt`, `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_tool_result_hook_can_override_error_flag -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_tool_result_hook_can_override_error_flag -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `git diff --check`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1115 tests.
- Previous local verification on 2026-05-21 after aligning Pi `beforeToolCall`
  / harness `tool_call` blocking semantics from `agent-loop.ts`,
  `types.ts`, and `harness/agent-harness.ts`: tool-call hooks can now return
  `block` with an optional `reason`, the loop emits an error tool result
  without executing the tool or running after-tool hooks, and the Rust-native
  argument replacement hook behavior remains available:
  `cargo fmt`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_tool_call_hook_can_block_execution_with_error_result -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_tool_call_hook_can_block_tool_execution -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `cargo fmt --check`,
  `git diff --check`, `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1113 tests.
- Previous local verification on 2026-05-21 after aligning generated compaction
  and branch-summary auth handling with `harness/agent-harness.ts`: generated
  compaction and `navigateTree(..., { summarize: true })` branch summaries now
  require a configured auth provider and fail with `Auth` when auth is
  unavailable, while hook-supplied summaries still bypass provider generation
  and existing provider-generation error paths keep their `Compaction` /
  `BranchSummary` classification once auth is present:
  `cargo fmt`, `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_compaction_generation_requires_auth_provider -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_navigate_tree_summary_requires_auth_provider -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `git diff --check`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1111 tests.
- Previous local verification on 2026-05-21 after aligning harness
  compaction/branch-summary error classification with
  `harness/agent-harness.ts` and `harness/types.ts`: `session_before_compact`
  cancellation now returns a `Compaction` harness error instead of conflating
  cancellation with "nothing to compact", generated compaction failures map to
  `Compaction`, generated branch-summary failures map to `BranchSummary`, and
  caller-provided hook errors keep the caller's classification while pending
  session writes still flush with save points:
  `cargo fmt`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_summary_hook_cancel_and_skip_flush_jsonl_pending_writes_without_summary -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_compaction_generation_errors_flush_pending_writes_without_summary -- --exact`, and
  `cargo test -p ri-agent-core --test agent_harness agent_harness_branch_summary_generation_errors_flush_pending_writes_without_move -- --exact`
  passed; the workspace test count remains 1109.
- Previous local verification on 2026-05-21 after adding Rust-native
  `after_provider_response` parity from `harness/agent-harness.ts`: simple
  stream providers can emit response metadata hooks for HTTP/faux responses,
  built-in OpenAI-compatible HTTP streaming records status/headers before
  parsing SSE, and `AgentHarness` subscribers receive `AfterProviderResponse`
  before the assistant stream starts:
  `cargo fmt`, `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_emits_after_provider_response_before_assistant_stream -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core builtin_openai_completions_provider_applies_response_hooks -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core -- --test-threads=1`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `git diff --check`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1109 tests.
- Previous local verification on 2026-05-21 after adding Rust-native Pi
  `navigateTree` parity from `harness/agent-harness.ts` and
  `harness/types.ts`: `navigate_tree` emits `session_tree`, calls
  `session_before_tree` hooks with target/common-ancestor/branch preparation,
  supports hook cancellation and hook-provided summaries, returns editor text
  when navigating to a user or custom-message entry, and moves the leaf to the
  target parent before appending branch summaries:
  `cargo fmt`, `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_navigate_tree_emits_pi_tree_event_and_returns_editor_text -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_navigate_tree_hook_can_cancel_without_moving -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `git diff --check`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1107 tests.
- Previous local verification on 2026-05-21 after adding Rust-native
  `AgentToolUpdateCallback` / `tool_execution_update` parity from
  `packages/agent/src/types.ts` and `agent-loop.ts`: tool executors can emit
  partial `AgentToolResult` updates through an async callback, update events
  carry the tool call id/name, current args, and partial result, update callbacks
  settle event-sink listeners before final tool completion, and `Agent` state
  reduction keeps partial updates non-mutating while start/end events continue to
  manage pending tool calls: `cargo fmt`, `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_core agent_loop_emits_tool_execution_update_from_tool_callbacks -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`, `git diff --check`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1105 tests.
- Previous local verification on 2026-05-21 after adding Rust-native
  `harness/messages.ts` `bashExecution` parity through typed session message
  entries: pi-shaped JSONL `message.role = "bashExecution"` round trips,
  `bashExecutionToText` formatting, `excludeFromContext` filtering during LLM
  context conversion, session-context role preservation, compaction token
  accounting, and bash execution as a user-visible turn start:
  `cargo fmt`,
  `cargo test -p ri-agent-core --test session_storage session_bash_execution_messages_convert_to_llm_context_and_wire_shape -- --exact`,
  `cargo test -p ri-agent-core --test harness_compaction compaction_treats_bash_execution_as_user_visible_context -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1104 tests.
- Previous local verification on 2026-05-21 after adding Rust-native
  `utils/validation.ts` parity for TypeBox/plain JSON-schema validation and
  coercion keywords, including `allOf`/`oneOf`, object
  `additionalProperties`, tuple array items, `enum`/`const`, and common
  scalar/array bounds, plus `utils/overflow.ts` parity that keeps 400/413
  no-body Cerebras errors as overflow while leaving 429 no-body responses as
  non-overflow rate-limit candidates, exact `utils/hash.ts` `shortHash`
  fixtures for ASCII/UTF-16 emoji/foreign Responses item IDs, and
  `utils/oauth/pkce.ts` verifier/challenge shape plus SHA-256 derivation, and
  `providers/github-copilot-headers.ts` dynamic headers for OpenAI
  Responses/Completions while preserving Anthropic Copilot behavior:
  `cargo fmt --check`,
  `cargo test -p ri-llm-provider --test provider_core validation_enforces_json_schema_object_array_and_constraint_keywords -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core validation_enforces_combinators_and_additional_property_schemas -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core overflow_matches_provider_error_shapes_without_rate_limit_false_positives -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core overflow_matches_context_overflow_provider_error_corpus -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core json_repair_and_hash_match_core_semantics -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core openai_responses_message_conversion_hashes_foreign_tool_item_ids -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core oauth_pkce_generation_matches_source_shape_and_challenge_derivation -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core openai_responses_and_completions_apply_copilot_dynamic_headers -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core github_copilot_anthropic_client_config_matches_provider_headers -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core -- --test-threads=1`, and
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1102 tests.
- Previous full local verification on 2026-05-21 after adding Rust-native
  `harness/env/nodejs.ts` parity for per-command cwd, shell env, file error
  mapping, callback-error termination, and streaming shell capture cancellation
  behavior, plus `harness/session/repo-utils.ts` fork-position/error case
  coverage and session custom/summary prompt-context conversion:
  `cargo fmt --check`,
  `cargo test -p ri-agent-core --test execution_env -- --test-threads=1`,
  `cargo test -p ri-agent-core --test session_storage in_memory_repo_opens_deletes_and_forks_by_metadata -- --exact`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_includes_session_summaries_and_custom_messages_in_prompt_context -- --exact`,
  `cargo test -p ri-agent-core --test resources prompt_template_argument_substitution_matches_pi_placeholders -- --exact`,
  `cargo test -p ri-agent-core --test resources load_skills_reports_pi_metadata_validation_warnings_without_dropping_skill -- --exact`,
  `cargo test -p ri-agent-core -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1098 tests.
- Previous full local verification on 2026-05-21 after adding Rust-native
  `session-resources.ts` parity for OpenAI Codex WebSocket cache cleanup and
  `utils/diagnostics.ts` parity for Codex transport fallback diagnostics, plus
  `agent/src/proxy.ts` stream-provider/proxy parity and
  `providers/simple-options.ts` simple stream default/max-token parity,
  `utils/oauth/oauth-page.ts` callback HTML parity, and
  `utils/json-parse.ts` partial streamed JSON parsing parity:
  `cargo fmt --check`,
  `cargo test -p ri-llm-provider --test provider_core -- --test-threads=1`,
  `cargo test -p ri-llm-provider --test provider_core session_resource_cleanup_removes_openai_codex_websocket_cache_for_session -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core builtin_openai_codex_provider_auto_falls_back_to_sse_when_websocket_fails_before_events -- --exact`,
  `cargo test -p ri-agent-core --test proxy -- --test-threads=1`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1095 tests.
- Previous full local verification on 2026-05-20 after removing the meta-tests and
  adding summary hook removal, Bedrock runtime proxy coverage, and OpenRouter
  Images custom authorization preservation:
  `cargo fmt --check`,
  `cargo test -p ri-agent-core --test agent_harness agent_harness_removes_registered_summary_hooks -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core builtin_bedrock_provider_routes_runtime_request_through_resolved_proxy -- --exact`,
  `cargo test -p ri-llm-provider --test provider_core builtin_openrouter_images_provider_preserves_custom_authorization_header -- --exact`,
  `cargo test --workspace -- --list`, and
  `cargo test --workspace -- --test-threads=1` passed; the list command
  enumerated 1086 tests.

## Known Missing Work

This migration is not complete.

- Strict provider live/E2E completion still requires running the gated provider
  matrix with real API keys, provider-specific environment configuration,
  local Ollama/LM Studio/llama.cpp services, and stored OAuth credentials.
- Manual interactive OAuth proof still requires real Anthropic callback,
  GitHub Copilot device, and OpenAI Codex callback login flows with
  `RI_LIVE_PROVIDER_TESTS=1` and `RI_LIVE_OAUTH_INTERACTIVE_TESTS=1`.
- Provider-specific parity still needs continued source-by-source review for
  edge transforms, streaming deltas, usage accounting, cache behavior,
  response-id handling, image inputs/outputs, error mapping, and proxy routing
  across all providers.
- Agent-core parity still needs continued case-by-case review for termination
  edge cases, before/after lifecycle hook ordering, async listener settlement,
  and session/harness integration behavior outside the covered high-level
  compaction and branch-summary hook contracts.
- Test parity is not certified by raw count alone: 1123 Rust tests cover the
  current Rust-representable provider and agent matrix, but the 871 source-case
  denominator is not one-to-one with Rust tests and excludes `packages/coding-agent`.
