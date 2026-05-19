# Third-Party Notices

`ri` depends on third-party Rust crates distributed through Cargo. Dependency
versions are locked in [Cargo.lock](Cargo.lock), and declared dependency
metadata is available through Cargo manifests.

This file records the current direct dependency license snapshot. Transitive
dependencies retain their own licenses and copyright notices.

## Direct Rust Dependencies

| Crate | Declared license |
| --- | --- |
| `async-trait` | `MIT OR Apache-2.0` |
| `base64` | `MIT OR Apache-2.0` |
| `chrono` | `MIT OR Apache-2.0` |
| `futures` | `MIT OR Apache-2.0` |
| `parking_lot` | `MIT OR Apache-2.0` |
| `regex` | `MIT OR Apache-2.0` |
| `reqwest` | `MIT OR Apache-2.0` |
| `ring` | `Apache-2.0 AND ISC` |
| `rustls` | `Apache-2.0 OR ISC OR MIT` |
| `rustls-pki-types` | `MIT OR Apache-2.0` |
| `serde` | `MIT OR Apache-2.0` |
| `serde_json` | `MIT OR Apache-2.0` |
| `thiserror` | `MIT OR Apache-2.0` |
| `tokio` | `MIT` |
| `tokio-rustls` | `MIT OR Apache-2.0` |
| `tokio-stream` | `MIT` |
| `uuid` | `Apache-2.0 OR MIT` |
| `webpki-roots` | `CDLA-Permissive-2.0` |

## Generating A Full Dependency Inventory

For source development, `Cargo.lock` is the authoritative dependency version
snapshot. Before publishing binary artifacts or vendored dependency bundles,
generate a full third-party notice inventory with a license auditing tool such
as `cargo-about` or `cargo-deny`.

A quick metadata view can be generated with:

```bash
cargo metadata --format-version 1
```

The generated inventory should include every transitive dependency and any
license text required by those dependencies.
