# Pi Baseline Sync Process

How to move ri to a newer pi release. The **baseline** is the exact pi
commit recorded in the latest `CHANGELOG.md` release entry (e.g.
`Pi baseline: v0.81.1, commit dd6bea41...`); a sync is a diff-driven port
from that commit to the new pi ref, scoped to the two packages ri mirrors:
`packages/ai` and `packages/agent`.

## 1. Scope the diff

```bash
cd ../pi && git fetch
# BASE = baseline commit from CHANGELOG.md, TARGET = new pi tag/commit
git log --oneline BASE..TARGET -- packages/ai packages/agent
git diff --stat BASE..TARGET -- packages/ai/src packages/agent/src
```

The commit list is the work list. Changes outside `packages/ai` and
`packages/agent` (coding-agent, tui, server, storage) are out of scope by
construction.

## 2. Classify and port each change

- **Model catalog updates** (`models.ts`, generated JSON — usually the bulk):
  do not transcribe line-by-line. Run ri's own generator,
  `cargo run --bin generate-models -- --write`, then walk pi's diff for
  hand-written corrections (thinking-level maps, pricing fixes, compat
  flags) and port those explicitly.
- **Behavior changes** (payloads, stream parsers, agent loop): port by hand
  into the corresponding ri module. The pi-file → ri-module mapping is
  documented throughout `MIGRATION_STATUS.md`.
- **New or changed tests**: pi ships tests with nearly every behavior
  change. `docs/test-parity-audit-2026-07-23.tsv` is the case ledger —
  diff the new baseline's test files against it; cases missing from the
  TSV are new work items. After porting, append them to the TSV with a
  verdict (COVERED / NA-live / NA-node / NA-scope), reusing the audit's
  classification rules from `docs/TEST_PARITY_AUDIT.md`.
- **Node-specific or out-of-scope**: skip, and record the NA verdict in
  the TSV so the ledger stays complete.

## 3. Version mapping and release

The `major.minor` component tracks the pi line; the patch component is
ri-owned (see `CHANGELOG.md` header):

- pi **patch** release (e.g. 0.81.2) → after syncing, ri takes its own next
  free patch number on the current line.
- pi **minor** release (e.g. 0.82.0) → after the sync is complete, ri jumps
  to that minor with patch 0 (e.g. `0.82.0`).

Then follow the standard release steps:

1. Bump `version` in both `crates/*/Cargo.toml`.
2. Roll CHANGELOG `[Unreleased]` into a dated entry that records the new
   pi baseline (version + commit); add a fresh `[Unreleased]`; update the
   compare links. Update the README tag pin.
3. `cargo fmt && cargo test --workspace` — must be green.
4. Commit, create an annotated tag (`git tag -a vX.Y.Z`).
5. Push `main` and the tag, then create the GitHub Release
   (`gh release create vX.Y.Z --notes-file ...`).

## 4. Verification depth

Scale verification to the size of the sync:

- **Small sync** (patch, few commits): the full local suite
  (`cargo test --workspace`) is enough.
- **Provider wire-format changes**: additionally run the gated live smokes
  for the touched providers (`RI_LIVE_PROVIDER_TESTS=1` plus credentials;
  see `tests/provider_live.rs` for per-provider requirements).
- **Large jumps** (several minors at once): rerun a case-by-case parity
  audit against the new baseline. The 2026-07 audit's method — parallel
  slices over the pi test tree, per-case verdicts merged into a TSV —is
  documented in `docs/TEST_PARITY_AUDIT.md` and reusable as-is.

## Key assets

- `CHANGELOG.md` — authoritative record of the current pi baseline commit
  (every sync's diff starts there).
- `docs/test-parity-audit-2026-07-23.tsv` — per-case coverage ledger; the
  anchor for incremental test triage.
- `docs/TEST_PARITY_AUDIT.md` — audit method and classification rules.
- `MIGRATION_STATUS.md` — pi-file → ri-module mapping and behavior notes.
- `cargo run --bin generate-models` — digests catalog-only changes without
  manual transcription.
