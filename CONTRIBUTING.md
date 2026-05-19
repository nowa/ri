# Contributing

Thanks for working on `ri`.

`ri` is a Rust port of the core `pi-ai` and `pi-agent-core` behavior. Changes
should preserve pi-compatible behavior where the Rust API intentionally mirrors
upstream concepts.

## Setup

Install the Rust toolchain required by the workspace:

```bash
rustup toolchain install 1.92
rustup override set 1.92
```

Build and test from the repository root:

```bash
cargo test --all
```

## Development Commands

```bash
cargo fmt
cargo test --all
cargo test -p ri-llm-provider
cargo test -p ri-agent-core
```

## Guidelines

- Keep Rust APIs idiomatic instead of mechanically copying TypeScript shapes.
- Preserve pi behavior for provider payloads, event ordering, tool execution,
  context transforms, abort handling, response IDs, usage accounting, and
  reasoning/thinking controls.
- Prefer enum and struct modeling over ad hoc stringly typed logic.
- Keep provider tests local and deterministic by using mock HTTP servers,
  payload assertions, parser tests, and stream event tests.
- Do not require live provider credentials for default tests.
- Do not commit secrets, API keys, OAuth tokens, generated build output, or
  local environment files.
- Keep `README.md`, `NOTICE.md`, and `THIRD_PARTY_NOTICES.md` aligned with
  material project changes.

## Pull Request Checklist

Before submitting a change, make sure:

- `cargo fmt` has been run.
- `cargo test --all` passes.
- New behavior has focused tests.
- Changes that intentionally differ from pi behavior are documented in the
  code, tests, or migration notes.
- Public documentation is updated when APIs or supported behavior change.

## License

By contributing to this repository, you agree that your contribution is licensed
under the MIT License used by this project.
