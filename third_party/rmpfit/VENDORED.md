# Vendored: rmpfit

This directory is a copy of the third-party crate [`rmpfit`](https://crates.io/crates/rmpfit),
a pure-Rust port of the CMPFIT/MINPACK Levenberg-Marquardt solver, carrying one local patch.
It backs the Gaussian fit in `src/gaussian.rs`.

| | |
|---|---|
| Crate | `rmpfit` |
| Version | `2.0.0` |
| Author | Vadim Dyadkin <dyadkin@3lp.cx> |
| Upstream | https://git.3lp.cx/dyadkin/rmpfit, commit `ec9e172ffe2cd67baed17cfa2a94a2bd4b52e558` per the registry tarball |
| crates.io | https://crates.io/crates/rmpfit |
| crates.io tarball sha256 | `e3e8006f8d69fcd1b9cbbf7b74999549dfe44b504b8c4f80727ae40a2b856528` (from the previous `Cargo.lock`) |
| License | MIT (see `LICENSE`) |
| Upstream `src/lib.rs` sha256 | `db6a83434cd18369e674cf26226c93a09a7cf5ba157bdd0efe32c8912490f431` |
| Vendored `src/lib.rs` sha256 | `ff51592ecf36549bdee174e48b831884c938599e7c3c31e6520feab9f56f35ea` |
| Vendored on | 2026-09-10 |

`Cargo.toml` and `README.md` are the published files, unchanged. The registry artifacts
(`Cargo.toml.orig`, `Cargo.lock`, `.cargo-ok`, `.cargo_vcs_info.json`, `.gitignore`) are not
copied; `Cargo.toml.orig` in particular breaks `maturin sdist`.

## Why vendored, and why patched

`src/lib.rs` is the upstream file plus `bound-snap.patch`, a single hunk in `MPFit::iterate`.
MPFIT scales a step that would cross a parameter bound so it stops at the bound, then snaps
coordinates within one ULP of a limit onto it. `x + alpha * step` can round a couple of ULP
past the limit; the snap misses it, the parameter never counts as pegged, the next step's
`alpha` collapses to ~1e-14 and the solver reports convergence with no progress. The patch
records which bound set `alpha` and lands that coordinate exactly on its limit. CMPFIT has the
same hazard. See "Bound-Limited Steps And The MPFIT Snap Hazard" in `docs/design-notes.md` for
the evidence (over 14,197 corpus windows: 18 fits change, all improvements, no stalls left).

Every local change is marked `gaussfit-rs local patch` at the patch site. Nothing else in the
file differs from upstream; check with

```bash
patch -R -p1 -d third_party/rmpfit -o - < third_party/rmpfit/bound-snap.patch | sha256sum
# db6a83434cd18369e674cf26226c93a09a7cf5ba157bdd0efe32c8912490f431
```

Known but not patched: `lmpar` clamps `par` with `max(paru)` where CMPFIT and MINPACK use
`min`. Fixing it changed one corpus row for the worse against the C reference, so it is only
reported upstream.

## Wiring

Consumed as a path dependency, `rmpfit = { path = "third_party/rmpfit" }` in the root
`Cargo.toml`. The sdist must ship this directory: `[tool.maturin] exclude` in `pyproject.toml`
excludes only `third_party/c_reference/**`, and CI installs from the sdist to prove it.
Dependabot ignores path dependencies (and skips any crate named in a `[patch]` table, so that
form buys nothing, and maturin leaves `[patch]` paths out of the sdist), so there is no
automated notice of new rmpfit releases; check crates.io when touching dependencies. The
scheduled poll deleted in commit f8cb42c can be restored from `f8cb42c^` if wanted.

The crate is not a workspace member, so `cargo fmt` and `cargo clippy` leave it alone.

## License note

The published crate ships no `LICENSE` file, only `license = "MIT"` metadata. `LICENSE` here
is the standard MIT text with the author's attribution; the copyright year (2021, the crate's
first publish year) is unverified because the upstream host is behind an anti-bot wall.

## How to update

1. `cargo fetch` the new version or download the tarball from crates.io; copy `Cargo.toml`,
   `README.md` and `src/lib.rs` here (not the registry artifacts listed above).
2. `patch -p1 -d third_party/rmpfit < third_party/rmpfit/bound-snap.patch`. If upstream has
   fixed the snap, drop the patch instead and delete the patch-site comments, the "why patched"
   paragraph above and the design-notes section; the regression tests decide:
   `src/tests/gaussian.rs::bound_pinned_window_matches_c_reference` and
   `python/gaussfit_rs/tests/test_fit_gaussian.py::test_f32_bound_limited_step_does_not_stall`.
3. Update the version, checksums and date in this file; regenerate `bound-snap.patch` with
   `diff -u --label a/src/lib.rs --label b/src/lib.rs <upstream lib.rs> src/lib.rs`.
4. `cargo test`, then the live parity test with the C reference installed
   (`tox -r -e py313-cparity`, 46 seeds): Rust must never be worse than C.
