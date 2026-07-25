# Prompt-Text Parity Audit (2026-07-25)

A line-by-line comparison of every model-visible string in pi
(`packages/ai` + `packages/agent`, baseline v0.81.1 @ `dd6bea41`) against
ri: system-prompt construction, summarization/branch-summary prompts,
skill and prompt-template formatting, session-context serialization
markers, tool-error texts returned to the model, message-transform
placeholders, and provider-injected payload text.

Method: three parallel audit slices (harness prompts / agent-core texts /
provider-injected text), each walking the pi files top-to-bottom and
byte-comparing against the ri counterpart. Long prompt constants were
diffed programmatically. Per-item verdicts:
[prompt-parity-audit-2026-07-25.tsv](prompt-parity-audit-2026-07-25.tsv)
(156 items: 114 MATCH, 24 DRIFT, 14 MISSING_IN_RI, 4 EXTRA_IN_RI).

## Fixed after the audit (all drifts closed)

- **Tool-argument validation error text** (the largest block, ~18 DRIFT +
  2 MISSING): ri's hand-written wording replaced with typebox 1.1.38
  en_US locale text as pi emits it, including aggregation semantics —
  `required` as a single line listing all missing properties with the
  first-missing-property path, `additionalProperties: false` as one
  object-level line before property recursion, `uniqueItems` as one
  array-level line after item recursion, tuple `additionalItems: false`
  as `schema is false` at the first out-of-range index, anyOf/oneOf
  emitting candidate sub-errors before the summary line, union types as
  `either a or b`, and the `Unknown validation error` fallback. Ground
  truth captured by running pi's formatting over typebox 1.1.38 and
  locked by `validation_error_text_matches_typebox_en_us_locale`.
- **`Operation aborted` tool results**: the abort signal is now checked
  after each tool-call hook resolves and again before execution,
  matching pi's agent-loop ordering.
- **OpenAI Responses empty-text replay**: assistant text blocks are
  replayed even when empty/whitespace (pi has no guard); the empty-item
  drop was removed and the affected test realigned to pi's live-verified
  behavior.
- **Codex instructions fallback**: an empty-string system prompt now
  falls back to `You are a helpful assistant.` (pi truthiness).
- **Branch-summary token budget**: unset `context_window` falls back to
  128000 (pi `model.contextWindow || 128000`) instead of collapsing the
  budget to 0/unlimited.
- **Empty-string truthiness in summarization prompts**: `""` previous
  summary / custom instructions are treated as absent (no empty
  `<previous-summary>` block, no dangling `Additional focus:`), in both
  the compaction and branch-summary paths.
- **Skill/template edges**: slash-less `dirname` resolves to `/` (pi),
  empty frontmatter `name:` falls back to the directory name, and
  template names strip one `.md` suffix case-insensitively.

## Documented divergences (intentional, kept)

- Validation keywords ri's subset validator does not implement (pi
  delegates to typebox): `format`, `contains`, `dependencies` /
  `dependentRequired`, `if`/`then`/`else`, `not`, `propertyNames`,
  `unevaluatedItems` / `unevaluatedProperties`. Tool schemas in scope do
  not use them; add with typebox wording if that changes.
- `Compaction.retainedTail` stored-tail semantics (also listed in the
  test-parity audit residuals).
- ri-only extras that emit no model text in pi flows: the agent-loop
  max-turn cap message (assistant `error_message` metadata), the
  runtime invalid-regex validation line (pi fails at schema compile
  time), percent-based `should_compact`, and the unused
  `system_prompt_with_context` helper.
- Anthropic redacted-thinking replay drops a block whose signature is
  empty/missing (pi would emit `data: undefined`); signatures are always
  present in practice.
- Three truncation helpers count Unicode chars where pi counts UTF-16
  code units — divergent only on astral-plane characters.
