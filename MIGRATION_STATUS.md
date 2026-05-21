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
    for common incomplete object/array/string cases, and unpaired UTF-16
    surrogate escape replacement.
  - Assistant-message diagnostics helpers mirroring `utils/diagnostics.ts`,
    represented as typed Rust structs that serialize to the pi
    `{ type, timestamp, error, details }` diagnostic shape.
  - JSON-schema subset validation and coercion.
  - Context-overflow detection with provider-specific error-shape corpus and
    silent/length-stop overflow signals.
  - Environment API key lookup.
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
    tool-result continuation turns, tool argument preparation, tool-call hooks
    with argument replacement, tool-result replacement hooks, hook-driven
    tool-result termination, tool-result message lifecycle events, and
    transform/convert hooks for LLM request context. Prepare-next-turn hooks can
    replace the next-turn context/model/thinking state after `turn_end`, and
    should-stop-after-turn hooks can stop before the next provider request.
    Queued messages can be injected before the initial provider request or
    after all tool-result messages are written and before the next LLM request,
    and follow-up messages can restart the loop after the agent would otherwise
    stop. Tool batches terminate only when every finalized result terminates.
    Parallel tool execution is the default, emits completion events in
    completion order while persisting tool-result messages in source order, and
    per-tool sequential mode forces sequential execution even under parallel
    loop config.
  - Stateful `Agent` wrapper with default/custom state initialization,
    thinking-level and session-id forwarding, subscriptions, text/image
    prompt normalization, prompt-message runs, source-compatible busy/continue
    validation errors, and steering/follow-up queues with one-at-a-time/all
    drain modes plus queued state introspection, queue clearing/reset helpers,
    abort handles that cancel active provider streams, and provider start
    failures persisted as assistant error messages with lifecycle events.
  - Custom stream-provider hooks for the low-level loop and `Agent` wrapper,
    including a Rust-native `agent/src/proxy.ts` counterpart that posts model,
    context, and proxy-safe stream options to `/api/stream`, reconstructs SSE
    assistant events, preserves abort behavior, and keeps API keys and retry
    policy out of the proxy payload.
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
    hooks through the direct loop, fixed skill/prompt-template resources with
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
    high-level session compaction that persists generated summaries through the session
    with `session_before_compact` cancellation/provided-summary hooks and
    hook success/error and generation-error pending-write flushes, session
    compact events, plus high-level
    session branch moves that generate and persist branch summaries with
    `session_before_branch_summary` supplied-summary/skip hooks,
    hook success/error and generation-error pending-write flushes, and
    branch-summary events.
  - In-memory and JSONL session storage/repositories with branching, labels, metadata, and context building.
  - Skill and prompt-template invocation formatting plus skill/prompt-template
    loading, argument substitution, and skill ignore-file handling.
  - Local execution environment foundation for filesystem operations, shell
    execution, stdout/stderr callbacks, callback error propagation, command
    timeout/abort handling, pre-aborted file-operation cancellation, symlink
    directory entries, text-line read limits, shell spawn error mapping,
    best-effort cleanup, and shell-output capture/truncation with full-output
    log files.

## Rust Test Coverage Now

Current Rust tests: 1095 enumerated by `cargo test --workspace -- --list`.

- `ri-llm-provider`: 921 tests: 1 library test, 289 `provider_core` tests, and
  631 `provider_live` tests. This is 200 above the 721 direct simple source
  cases counted under `packages/ai/test`, because the Rust suite also includes
  Rust-specific registry, HTTP, proxy, transport, OAuth auth-storage, and gated
  live/E2E coverage.
- `ri-agent-core`: 174 tests across `agent_core`, `agent_harness`,
  `execution_env`, `harness_compaction`, `harness_truncate`, `proxy`,
  `resources`, and `session_storage`. This is 24 above the 150 direct simple
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
- The raw 1095-vs-871 count is not completion proof. Rust tests sometimes
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
- Latest local verification on 2026-05-21 after adding Rust-native
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
- Test parity is not certified by raw count alone: 1095 Rust tests cover the
  current Rust-representable provider and agent matrix, but the 871 source-case
  denominator is not one-to-one with Rust tests and excludes `packages/coding-agent`.
