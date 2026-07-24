# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The workspace crates (`ri-llm-provider`, `ri-agent-core`) are versioned in
lockstep; one entry here covers both.

Versioning policy: the `major.minor` component tracks the pi release line the
workspace is behavior-compatible with (`0.81` = pi 0.81.x), while the patch
component is owned by ri and advances for ri bug fixes and small baseline
syncs. Each release entry records the exact pi baseline (version and commit).
When ri syncs to a new pi minor line, the version jumps accordingly
(e.g. `0.82.0`).

## [Unreleased]

## [0.81.0] - 2026-07-24

Initial release: a Rust port of pi's LLM provider (`packages/ai`) and agent
runtime (`packages/agent`) behavior.

Pi baseline: v0.81.1, commit `dd6bea41efa8caa7a10fe5a6401676dc5699f83f`
(2026-07-21), verified case-by-case (1179 pi test cases audited; see
`docs/TEST_PARITY_AUDIT.md`). Workspace test suite: 1469 passed / 0 failed.

### Added

- `ri-llm-provider`: unified multi-provider LLM API (`stream`, `complete`,
  `stream_simple`, `complete_simple`) with pi-compatible payload semantics,
  stream event ordering, tool calling with streamed partial arguments,
  reasoning/thinking controls (`off` through `xhigh`), usage and cost
  accounting, retries, aborts, and provider-specific extras.
- Providers: OpenAI (Completions + Responses), Azure OpenAI, OpenAI Codex,
  Anthropic, Google, Vertex AI, Mistral, Amazon Bedrock, GitHub Copilot,
  OpenRouter, xAI, Radius, and OpenAI-/Anthropic-compatible layers.
- Embedded model catalog with an ri-owned refresh generator
  (`generate-models`) that merges models.dev data under a conservative
  update policy, plus image-model metadata.
- OAuth support: OpenAI Codex, Anthropic, GitHub Copilot, Google Vertex, and
  xAI flows on a shared RFC 8628 device-code poller with abort support,
  interactive `AuthInteraction` prompts, and pi-shaped
  `~/.pi/agent/auth.json` credential storage.
- `ri-agent-core`: stateful `Agent` and `agent_loop` APIs with event
  streaming, parallel/sequential tool execution, steering and follow-up
  queues, context transforms, and tool call/result hooks.
- `ri_agent_core::harness`: session storage (in-memory, JSON, and SQLite
  with write-side materialization and lazy cursor reads), system prompt
  formatting, skills and prompt templates, compaction and branch summaries,
  local execution environment, and provider auth/request/payload hooks.
- Case-by-case test parity audit against pi HEAD with committed matrix and
  per-case verdicts (`docs/TEST_PARITY_AUDIT.md`,
  `docs/test-parity-audit-2026-07-23.tsv`).

[Unreleased]: https://github.com/nowa/ri/compare/v0.81.0...HEAD
[0.81.0]: https://github.com/nowa/ri/releases/tag/v0.81.0
