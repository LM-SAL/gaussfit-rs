# Vendored: rmpfit

This directory is a **verbatim vendored copy** of the third-party crate
[`rmpfit`](https://crates.io/crates/rmpfit) — a pure-Rust port of the
CMPFIT/MINPACK Levenberg-Marquardt solver. It backs the Gaussian fit in
`src/gaussian.rs`.

| | |
|---|---|
| Crate | `rmpfit` |
| Version | `2.0.0` |
| Author | Vadim Dyadkin <dyadkin@3lp.cx> |
| Upstream | https://git.3lp.cx/dyadkin/rmpfit |
| crates.io | https://crates.io/crates/rmpfit |
| License | MIT (see `LICENSE`) |
| Source | crates.io registry tarball |
| `src/lib.rs` sha256 | `db6a83434cd18369e674cf26226c93a09a7cf5ba157bdd0efe32c8912490f431` |
| Vendored on | 2026-07-27 |

## Why vendored

Zero external dependencies, single self-contained file, MIT-licensed, decades-stable
numeric core (MINPACK). Vendoring keeps the scientific solver auditable and the
build reproducible without pulling it from crates.io at build time. See the project
design notes / review discussion for the full rationale.

## Rules

- **Do not edit `src/lib.rs`.** Keep it byte-identical to the upstream release so
  it can be diffed and re-synced. If a local patch is ever unavoidable, document
  it explicitly here and in a comment at the patch site.
- Consumed via a path dependency in the root `Cargo.toml`:
  `rmpfit = { path = "third_party/rmpfit" }`.

## License note

The published crate ships no `LICENSE` file; only `license = "MIT"` metadata
(confirmed via the crates.io API). The `LICENSE` here is the standard MIT text
with the author's attribution. The copyright year (2021) is the crate's first
publish year; it is unverified against upstream because the upstream host is
behind an anti-bot wall. Correct it if the upstream `LICENSE` later becomes
reachable.

## How to update

1. `cargo fetch` the new version, or download the tarball from crates.io.
2. Replace `src/lib.rs` (and `README.md`) with the new release's files.
3. Update the version, sha256, and "Vendored on" date in this file, and the
   version in the vendored `Cargo.toml` (the update-check workflow reads the
   pinned version from there).
4. Run `cargo test` + the differential validation harness before committing.
