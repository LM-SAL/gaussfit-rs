# Vendored: rmpfit

This directory is a copy of the third-party crate [`rmpfit`](https://crates.io/crates/rmpfit),
a pure-Rust port of the CMPFIT/MINPACK Levenberg-Marquardt solver, carrying two local patches
(`bound-snap`, `lmpar-clamp`). It backs the Gaussian fit in `src/gaussian.rs`.

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
| Vendored `src/lib.rs` sha256 | `e932b52ae3ee5ee43f62a71f0a605dc513735a837b8beb100d4339d780fdcd8e` |
| Vendored on | 2026-09-10; `lmpar-clamp.patch` added 2026-09-11 |

`Cargo.toml` and `README.md` are the published files, unchanged. The registry artifacts
(`Cargo.toml.orig`, `Cargo.lock`, `.cargo-ok`, `.cargo_vcs_info.json`, `.gitignore`) are not
copied; `Cargo.toml.orig` in particular breaks `maturin sdist`.

## Why vendored, and why patched

`src/lib.rs` is the upstream file plus two patches. `bound-snap.patch` is a single hunk in
`MPFit::iterate`: MPFIT scales a step that would cross a parameter bound so it stops at the bound,
then snaps coordinates within one ULP of a limit onto it. `x + alpha * step` can round a couple of
ULP past the limit; the snap misses it, the parameter never counts as pegged, the next step's
`alpha` collapses to ~1e-14 and the solver reports convergence with no progress. The patch records
which bound set `alpha` and lands that coordinate exactly on its limit. CMPFIT has the same hazard.
See "Bound-Limited Steps And The MPFIT Snap Hazard" in `docs/design-notes.rst` for the evidence
(over 14,197 corpus windows: 18 fits change, all improvements, no stalls left).

`lmpar-clamp.patch` fixes a port deviation: upstream rmpfit clamps the trust-region radius with
`self.par = self.par.max(paru)` where CMPFIT and MINPACK use `min` (in-repo C reference:
`third_party/c_reference/vendor/cmpfit-1.5/mpfit.c:2102`). Every other line of the `lmpar` sequence
matches. With the fix, 68 of 82,368 corpus fits move onto the C reference's values and one low-SNR
fit (`muse` seed 44, `low_snr`, absolute row 165) lands in a different local minimum, reduced
chi-square 0.804337 vs C's 0.755257 (1.065x), so it is covered by a named row-level exception,
`ACCEPTED_WORSE` in `python/gaussfit_rs/tests/test_c_parity.py`, rather than a wider tolerance. See
"`lmpar` Trust-Region Clamp" in `docs/design-notes.rst`.

Every local change is marked `gaussfit-rs local patch` at the patch site. Nothing else in the
file differs from upstream; check with

```bash
cp -r third_party/rmpfit /tmp/rmpfit-check
patch -R -p1 -d /tmp/rmpfit-check < third_party/rmpfit/lmpar-clamp.patch
patch -R -p1 -d /tmp/rmpfit-check < third_party/rmpfit/bound-snap.patch
sha256sum /tmp/rmpfit-check/src/lib.rs
# db6a83434cd18369e674cf26226c93a09a7cf5ba157bdd0efe32c8912490f431
```

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
2. `patch -p1 -d third_party/rmpfit < third_party/rmpfit/bound-snap.patch` and
   `patch -p1 -d third_party/rmpfit < third_party/rmpfit/lmpar-clamp.patch`. If upstream has
   fixed the snap, drop that patch instead and delete the patch-site comments, its paragraph
   above and the design-notes section; the regression tests decide:
   `src/tests/gaussian.rs::bound_pinned_window_matches_c_reference` and
   `python/gaussfit_rs/tests/test_fit_gaussian.py::test_f32_bound_limited_step_does_not_stall`.
3. Update the version, checksums and date in this file, then regenerate each patch against its own
   base — not both against upstream. `bound-snap.patch` is
   `diff -u --label a/src/lib.rs --label b/src/lib.rs <upstream lib.rs> <upstream + snap>`, and
   `lmpar-clamp.patch` is the same command from that intermediate file to `src/lib.rs`.
   Regenerating both from upstream in one go would fold the snap hunk into the lmpar patch and
   break the reverse check above.
4. `cargo test`, then the live parity test with the C reference installed
   (`tox -r -e py313-cparity`, 46 seeds): Rust must never be worse than C, except where
   `ACCEPTED_WORSE` in `python/gaussfit_rs/tests/test_c_parity.py` names a documented exception.
   Upstream releases that fix `bound-snap` or `lmpar` mean the matching patch can be dropped;
   when `lmpar` is fixed upstream, also delete `ACCEPTED_WORSE` and the design-notes entry — the
   gate fails if an exception is listed but no longer taken, so it cannot linger.
