# ri

Rust workspace for migrating the core pieces of `/home/nowa/Projects/src/pi`:

- `crates/ri-llm-provider`: Rust counterpart for `packages/ai` / `pi-ai`.
- `crates/ri-agent-core`: Rust counterpart for `packages/agent` / `pi-agent-core`.

Current status: scaffold plus first functional slice. The workspace builds and
has Rust tests for provider model metadata, faux provider streaming/caching,
tool argument validation, overflow detection, JSON repair/hash, agent loop,
stateful `Agent`, and basic harness utilities.

Run:

```sh
cargo test --workspace
```

