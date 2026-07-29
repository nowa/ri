# Behavior-Parity Audit (2026-07-25)

The third parity audit, covering what the test-parity and prompt-text audits
could not: behavior pi implements but does not lock with tests — retry and
error handling, behavioral constants, streaming edge cases, and session/
compaction semantics. Baseline: pi v0.81.1 @ `dd6bea41`.

## Method

Three complementary techniques:

1. **Constants sweep** — every behavior-affecting literal (defaults, limits,
   timeouts, thresholds, detection sets, env names) in pi `packages/ai` +
   `packages/agent`, compared value-by-value:
   [behavior-audit-constants_audit_ai-2026-07-25.tsv](behavior-audit-constants_audit_ai-2026-07-25.tsv)
   (662 rows) and
   [behavior-audit-constants_audit_agent-2026-07-25.tsv](behavior-audit-constants_audit_agent-2026-07-25.tsv)
   (137 rows).
2. **Function-level walk** of the high-risk utils — retry/backoff, error-body
   composition, abort signals, SSE parsing, estimation, hashing:
   [behavior-audit-utils_audit_retry_error-2026-07-25.tsv](behavior-audit-utils_audit_retry_error-2026-07-25.tsv)
   (61 rows) and
   [behavior-audit-utils_audit_stream_misc-2026-07-25.tsv](behavior-audit-utils_audit_stream_misc-2026-07-25.tsv)
   (60 rows).
3. **Differential testing** — the same 57-case matrix (providers × contexts ×
   option sets) run through pi's real `streamSimple` (bun) and ri's real
   `complete_simple`, capturing outgoing payloads via the payload hook on both
   sides and diffing them. Tooling lives in `tools/differential/` (cases +
   pi-side dumper + comparator) and `tests/differential.rs` (ri side,
   env-gated via `RI_DIFF_CASES`/`RI_DIFF_OUT`).

## Result

~920 audited items plus 57 live payload comparisons. All actionable drifts
were fixed (workspace suite grew 1471 → 1527 tests, all passing; differential:
55/57 byte-identical after canonicalization, the other 2 are the documented
pi-bug divergences below).

Highest-impact fixes (the audit's justification):

- **Anthropic endpoint** missed the `/v1` path segment on every
  anthropic-messages provider (pre-existing mock tests had pinned the drift).
- **Bedrock** long-cache TTL sent the enum name (`ONE_HOUR`) instead of the
  wire value (`1h`) — misled by pi's own vitest mock; Claude 5 models missed
  prompt caching; HTTP throttling errors lacked the `Throttling error:`
  prefix and were misclassified as context overflow; simple-stream max-token
  and thinking-budget defaults were absent.
- **Codex retry semantics**: ri defaulted to 3 provider-internal retries (pi:
  0), capped every delay when a cap was set (pi: only 429 retry-after, 60s
  default), retried terminal quota 429s, and slept through aborts.
- **Error-body composition** now reproduces the SDK-shaped strings pi
  surfaces (status digits, `"<status> status code (no body)"`, `{}`
  suppression, 4000-char truncation) — these strings feed the retry and
  overflow classifiers, so the wording is load-bearing.
- **Session/compaction**: harness compaction consumed the whole append log
  instead of the active branch; `Session::branch()` ignored compaction
  boundaries; cursor paging windows were backwards vs pi; zero-usage
  assistant messages anchored context estimates; token estimates rounded
  per-block instead of per-message.
- **Tool-call id normalization** collapsed same-turn parallel calls sharing a
  Responses `call_id` (item part dropped); Responses replay emitted
  `"id": null`; pipe-split ids kept everything after the first `|`.
- **Detection sets**: nvidia/ant-ling/zai-cn URL detection, OpenRouter
  developer-role and cache-control gates, routing passthrough conditions,
  `string-thinking` format, xAI encrypted-reasoning include.
- **Anthropic/Mistral/Google**: raw-path max_tokens fallback was
  `model.max_tokens / 3` (pi: `model.max_tokens`); compat defaults are pi's
  constants; Mistral cached tokens were never parsed; Gemini flash-latest ids
  missed level-based thinking; vertex gemma-4 split from genai.
- Plus: IRSA token-file-only credentials, Cloudflare placeholder passthrough,
  scoped `PI_CACHE_RETENTION`/`AZURE_OPENAI_RESOURCE_NAME`, SSE BOM
  stripping, UTF-16 token estimation, Copilot device-code `interval`
  optionality, OAuth token-endpoint timeout, openrouter-images fail-fast and
  data-URL strictness, Radius state/scheme/registration-order.

## Documented divergences (intentional, kept)

- **pi upstream bugs not replicated**: `reasoning: "off"` on non-adaptive
  Anthropic models makes pi send `max_tokens: NaN→null` with a 1024 thinking
  budget (differential case `anthropic-claude-sonnet-4-5--reasoning-off`);
  codex SSE parsing never dispatches CRLF-framed events and drops a trailing
  unterminated frame; pi-messages uses only the first `data:` line per event.
- **Corrupted thinking signatures** on Responses replay: pi fails the request
  with a `JSON Parse error`; ri drops the corrupted reasoning item and sends
  the rest (fail-loud alignment needs Result plumbing through the payload
  builders — deferred).
- ~~**Feature gaps**~~ — all three were implemented after the audit; see
  "Feature gaps closed" below.
- **Runtime-inherent**: UTF-16 vs char counts in three truncation helpers
  (astral-plane only); `sanitize_surrogates` as identity (Rust strings cannot
  hold lone surrogates; U+FFFD at JSON ingestion); JS local-time parsing of
  timezone-less retry-after dates; localeCompare vs byte sort in skill/
  template listings; npm `ignore`'s literal `[!...]` (ri follows real
  gitignore negation); event-stream `end()` hang vs ri's error.
- **ri-only hardening kept**: agent-loop max-turn cap, WS frame/handshake
  caps, OAuth callback limits, JSON repair on SSE frames (pi throws),
  lazy idle-TTL check instead of a release-armed timer.
- **Gateway concurrency hardening** (longbridge/ri#8, production incident):
  non-2xx streaming errors carry the gateway's announced wait as a
  `(retry-after-ms: N)` suffix so hosts holding only the error string can
  honor it instead of backing off blindly, and a dropped stream consumer is
  treated as an abort so the request task stops reading and releases the
  upstream concurrency slot. The suffix is ri-only transport metadata: it is
  stripped before any error-classification pattern runs
  (`retry::strip_retry_after_hint`), because the delay's digits would
  otherwise match the retryable-status patterns and turn a terminal 4xx into
  a "retryable" error. ri's own `retry_assistant_call` (harness compaction and
  branch summarization) also prefers the announced delay over its blind
  exponential backoff — pi's `retry.ts` has no header access and stays blind,
  so this is deliberate ri-only hardening.
- **Returned agent event log omits per-chunk progress events**
  (longbridge/ri#7, production incident): pi's `agentLoop` returns messages
  only, so ri's extra event log is convenience — and retaining one
  `MessageUpdate` per SSE delta (each owning two copies of the
  partial message) grew it quadratically with output. Progress events are
  delivered to the `event_sink` in real time; the log keeps the lifecycle.

## Feature gaps closed (2026-07-25, after the audit)

The three gaps the audit deferred are now implemented:

- **GitHub Copilot entitlement filtering** — the `/models` listing is fetched
  at login and on every token refresh (`X-GitHub-Api-Version: 2026-06-01`,
  5s timeout, pi's `model_picker_enabled` / `policy.state` /
  `capabilities.supports.tool_calls` selectability rules), stored as
  `availableModelIds` on the credential, and applied through the provider's
  `filter_models` hook. As in pi, a listing failure fails the login/refresh,
  and a missing or malformed listing leaves the catalog unfiltered.
- **Bedrock and Vertex interactive logins** — pi's prompt sequences
  (Bedrock: bearer token / AWS profile / existing credential chain; Vertex:
  API key / ADC / service-account file, with project and location), including
  the informational notices and links. Both providers' `resolve()` now
  mirrors pi's full source chain, so env-only credentials from those logins
  resolve and report pi's source labels (`AWS_PROFILE`, `ECS task role`,
  `web identity token`, `gcloud application default credentials`, …).
- **`maxRetries` SDK-style retries** — a shared retry loop reproduces the
  openai-node / @anthropic-ai/sdk contract on the paths where pi forwards
  `options.maxRetries` (anthropic-messages, openai-completions,
  openai-responses, azure-openai-responses, openrouter-images):
  `x-should-retry` override, 408/409/429/5xx, connection-error retries, and
  the `retry-after-ms` → `retry-after` → `min(0.5s·2ⁿ, 8s)` minus up to 25%
  jitter delay ladder. Defaults are unchanged (0 retries = one attempt).
  Header parsing follows the SDKs' `parseFloat`, which reads an ISO-8601
  retry-after as *seconds* and never reaches `Date.parse` — pi's own codex
  loop uses `Number()` and does fall through to dates, so the two paths
  differ on that input in pi too.

## Cross-implementation interop (stage 4)

Fixtures written by pi's own `JsonlSessionStorage` are committed under
`crates/ri-agent-core/tests/fixtures/` and read back by ri in
`tests/interop_pi_fixtures.rs`; the reverse direction writes a session from
ri (`tests/interop_write_for_pi.rs`) and has pi's reader confirm it.
`tools/interop/verify.sh` runs both halves — regenerate the fixtures with
`generate-pi-fixtures.ts` only as a deliberate step when syncing baselines.

Feeding real pi bytes through ri immediately surfaced four defects that no
amount of same-side testing had caught, all now fixed:

- **`auth.json` credentials were unreadable.** ri's `Credential` enum derived
  its type tags from the variant names, emitting/expecting `o_auth` where pi
  writes `oauth` — so every OAuth credential in a pi-written `auth.json`
  failed to deserialize, and ri-written credentials were unreadable by pi.
- **`firstKeptEntryId` is optional in pi.** ri modeled it as a required
  `String`, so a pi session whose compaction kept nothing at all failed to
  open with an "invalid entry" error.
- **`retainedTail` was dropped on read.** pi stores the kept messages on the
  compaction entry and replays them after the summary; ri had no field, so a
  branch rooted at a retained-tail compaction silently lost those messages
  from the model context. ri now models the tail, stops the branch walk at
  such a compaction, and expands `summary + retainedTail` exactly like pi —
  closing the last outstanding gap from the earlier audits.
- **The tail's wire type is pi's `AgentMessage`**, not ri's tagged
  context-message type; modeling it with the wrong shape rejected otherwise
  valid pi files.

ri appends and never rewrites existing lines, which the interop test pins:
fields ri does not model stay byte-intact on disk for pi to read back.

## Empirical confirmations worth recording

- groq `qwen/qwen3*` hardcoded `reasoning_effort: "default"` matches pi's
  live behavior for every non-off level (differential cases
  `groq-qwen3--reasoning-*`), despite reading as if the thinkingLevelMap
  should win — do not "simplify" it.
- OpenRouter developer-role gate: `anthropic/*` ids get `developer`,
  other ids get `system` (differential `openrouter-*--basic`).

## Reusing the differential harness

```bash
cd tools/differential
bun pi-dump.ts cases.json > pi-payloads.json          # pi side (needs npm ci in pi)
RI_DIFF_CASES=$PWD/cases.json RI_DIFF_OUT=$PWD/ri-payloads.json \
  cargo test -p ri-llm-provider --test differential
python3 compare.py pi-payloads.json ri-payloads.json
```

Extend `cases.json` when syncing new pi baselines (see
[SYNC_PROCESS.md](SYNC_PROCESS.md)); disagreements found by the comparator
are ground truth — pi's dumper runs pi's real code.
