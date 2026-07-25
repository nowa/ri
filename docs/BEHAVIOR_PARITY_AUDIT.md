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
- **Feature gaps** (backlog): Copilot `/models` entitlement fetch +
  `availableModelIds` filtering (ri's `filter_models` hook exists, unwired);
  Bedrock/Vertex interactive logins; `maxRetries > 0` SDK-style retries on
  non-codex chat paths (pi delegates to provider SDKs; defaults of 0 match).
- **Runtime-inherent**: UTF-16 vs char counts in three truncation helpers
  (astral-plane only); `sanitize_surrogates` as identity (Rust strings cannot
  hold lone surrogates; U+FFFD at JSON ingestion); JS local-time parsing of
  timezone-less retry-after dates; localeCompare vs byte sort in skill/
  template listings; npm `ignore`'s literal `[!...]` (ri follows real
  gitignore negation); event-stream `end()` hang vs ri's error.
- **ri-only hardening kept**: agent-loop max-turn cap, WS frame/handshake
  caps, OAuth callback limits, JSON repair on SSE frames (pi throws),
  lazy idle-TTL check instead of a release-armed timer.

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
