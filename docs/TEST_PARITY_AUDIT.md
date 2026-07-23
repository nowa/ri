# pi Test-Case Parity Audit — 2026-07-23

A case-by-case comparison of every pi test (`packages/ai/test`,
`packages/agent/test`, pi @ v0.81.1 HEAD) against ri's suites. Raw
per-case verdicts: `docs/test-parity-audit-2026-07-23.tsv`
(file / case / verdict / detail).

## Matrix

| Slice | Cases | COVERED | GAP | NA-live | NA-node | NA-scope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Core streaming/util (empty, unicode, abort, tokens, overflow, text…) | 163 | 158 | 3 | 0 | 2 | 0 |
| Agent package (harness, storage, sqlite, loop) | 184 | 168 | 9 | 0 | 6 | 1 |
| Live/E2E/smoke (stream matrix, e2e, oauth-live) | 279 | 23 | 15 | 235 | 6 | 0 |
| Providers (anthropic/google/bedrock/mistral/copilot/…) | 252 | 156 | 55 | 24 | 11 | 6 |
| OpenAI family (completions/responses/azure/codex, runtime) | 301 | 210 | 72 | 5 | 14 | 0 |
| **Total** | **1179** | **715** | **154** | **264** | **39** | **7** |

Verdicts: COVERED = a named ri test asserts the same behavior (body
verified, not name-matched). NA-live = requires real credentials; ri's
gated `provider_live.rs` covers 205 of the 224 stream-matrix cases.
NA-node = Node/TS-specific with no Rust analogue. NA-scope = consciously
deferred per MIGRATION_STATUS.

## Gaps fixed during the audit (16 of 154)

Commits 7a2fb55, 0a15cf6:
- `truncate_tail` dropped oversized single lines ending in a newline
  (behavior fix + regression test)
- Bedrock adaptive/native-xhigh matchers stopped at Opus 4.7/Sonnet 4.6;
  extended to Opus 4.8 / Sonnet 5 / Fable 5
- Anthropic now omits `thinking:{type:disabled}` for Claude Fable 5
- text.rs content-extraction helpers; session stats across all three
  storage backends; JSONL header-metadata omission + repo
  create/list/fork metadata; SQLite append rollback restore;
  branch-summary retry events + hook-usage persistence; stored Copilot
  OAuth per-credential base URL through `Models::get_auth`

## Remaining gap backlog (138), categorized

### A. Behavior/catalog drift vs pi HEAD (needs code changes)
- Azure: Foundry root-endpoint normalization (`.ai.azure.com` → `/openai/v1`),
  `/openai/v1/responses` path normalization, `store:false` in payloads,
  prompt_cache_key clamping
- prompt_cache_key / session-id 64-char clamp only exists on the codex
  path (missing on openai-responses, completions, azure); codex session-id
  HEADER clamp also missing
- `sessionAffinityFormat:"openai-nosession"` ignored everywhere
- z.ai GLM-5.2 reasoning: ri emits legacy `enable_thinking` instead of
  `reasoning_effort` (+ replay half)
- Ant Ling thinkingFormat branch absent; generic chat-template
  boolean/effort kwargs substitution absent
- Copilot thinkingLevelMap overrides stale (minimal:low / max); stale
  xhigh maps (deepseek-v4-flash → max, openrouter opus-4.6 → max,
  gpt-5.5-pro restriction, opencode-go kimi-k2.6); Moonshot Kimi K2.7
  Code off-omission; opencode-go deepseek supportsReasoningEffort gate
- xAI retired models (grok-3 etc.) still in catalog; kimi-for-coding
  implied pricing still zero
- Bedrock: `<empty>` placeholder for only-unknown user blocks; endpoint
  pinning for explicit/scoped profiles
- Codex/WS: SSE headers-arrival timeout, WS connect timeout,
  pre-first-event and post-start idle timeouts all absent
- openai-completions error surfaces lack status prefix; OpenRouter
  `metadata.raw` dropped instead of deduplicated
- Mistral promptCacheKey (sessionId + retention-none omission)
- Copilot device-flow verification_uri http(s) validation/normalization;
  device-code poll abort hook; proxy scoped-env override param;
  Models runtime `transformHeaders` hook
- xAI OAuth wholly unported (14 offline-testable device-flow cases)

### B. Implemented but untested (tests only, ~80 cases)
Highlights: gpt-5.6 none-effort rows; responses-side OpenRouter affinity;
codex oauth select-prompt flow; completions maxTokens positive asserts;
per-model catalog compat flags (opencode long-cache-retention,
openrouter/xiaomi/qwen replay compat, zai effort metadata); deferred-tools
OAuth normalization + cross-provider history; models-runtime refresh
credential/abort/force propagation; pi-messages error paths; bedrock
custom headers; kimi-coding force-adaptive payloads; adaptive-model
catalog sweep; xai grok-4.5 responses; supports-xhigh new-model rows;
env ZAI_CODING_CN_API_KEY; sqlite in-memory settled-tool update guard.

### C. Live coverage gaps (gated suite additions)
No `provider_live.rs` smoke for NVIDIA NIM, Qwen Token Plan (intl+CN),
Ant Ling (19 matrix cases); xiaomi-token-plan-ams empty-signature smoke;
live opus reasoning smoke pinned to 4-7 vs pi's 4-8.

### D. Documented non-goals found undocumented (now recorded)
- `Compaction.retainedTail` field/semantics unimplemented
- Copilot account-picker model filtering (provider-owned-auth deferral)
