# Vendored: rmpfit

This directory is a copy of the third-party crate [`rmpfit`](https://crates.io/crates/rmpfit),
a pure-Rust port of the CMPFIT/MINPACK Levenberg-Marquardt solver, carrying two local patches
(`workspace`, `perf`). It backs the Gaussian fit in `src/gaussian.rs`.

| | |
|---|---|
| Crate | `rmpfit` |
| Version | `2.0.3` |
| Author | Vadim Dyadkin <dyadkin@3lp.cx> |
| Upstream | https://git.3lp.cx/dyadkin/rmpfit, commit `e8a810bd9ba5f30ac251f903ee0e1ddf72413d2c` per the registry tarball |
| crates.io | https://crates.io/crates/rmpfit |
| crates.io tarball sha256 | `61612e393bf9916f7cd6c7f91d45986f3613bedb67b2b5064b09d4399e9a9775` (crates.io index checksum) |
| License | MIT (see `LICENSE`) |
| Upstream `src/lib.rs` sha256 | `2bb95e44e8afa4751c0dafcf79cdfbda1b47594a52b5f47ef56099458863c12f` |
| Vendored `src/lib.rs` sha256 | `45eae578eff147c134f8e1a84075f26617b6889adf3a2a4cebb1af61ab52c418` |
| Vendored on | 2026-10-22 |

`Cargo.toml` and `README.md` are the published files, unchanged. The registry artifacts
(`Cargo.toml.orig`, `Cargo.lock`, `.cargo-ok`, `.cargo_vcs_info.json`, `.gitignore`) are not
copied; `Cargo.toml.orig` in particular breaks `maturin sdist`.

## Upstream behaviour relied on

Two upstream behaviours matter here. A step limited by a parameter bound lands that coordinate
exactly on the limit (`MPFit::iterate`): `x + alpha * step` can round a couple of ULP past it,
miss MPFIT's one-ULP snap and stall the fit with `alpha` ~1e-14, as CMPFIT can.
`src/tests/gaussian.rs::bound_pinned_window_matches_c_reference` and
`python/gaussfit_rs/tests/test_fit_gaussian.py::test_bound_limited_step_does_not_stall` guard
it. And `MPFit::lmpar` clamps the trust-region parameter with `min` as CMPFIT and MINPACK do,
which is where the one row named in "Accepted C-Parity Exception" in `docs/design-notes.rst`
comes from. `src/gaussian.rs` bounds every parameter on both sides, so upstream's handling of
unbounded parameters in `check_limits` does not apply.

The crate's own `gaussian` test (numerical derivatives) fails on macOS arm64, upstream as well:
`init[0]` lands at 0.48044143 against an expected 0.48044336 at a 1e-6 tolerance (2.0.3 dropped
the iteration-count asserts that failed there first). Apple's `exp` and glibc's round two
near-halfway inputs of the fit 1 ULP apart, the forward-difference Jacobian amplifies that, and
the fit stops one iteration earlier at a point equally inside `ftol`: both answers are within
1.2e-6 of the true minimum, so the test's tolerance, not the solver, is at fault.
`gauss_analytical`, the path `src/gaussian.rs` uses, passes. The crate tests are not part of
this repo's test run (`cargo test` covers the root crate only).

## Why still vendored

`src/lib.rs` is the upstream file plus two performance patches, neither with a numerical effect.

`workspace.patch`: upstream allocates about 28
`Vec`s per fit (the `MPFit` buffers, the parsed parameter tables, and one derivative column per
analytical parameter on every Jacobian evaluation), which at the pipeline's 5-sample window was
a fifth of the fit. The patch adds `MPWorkspace`, a reusable set of those buffers, and
`MPFitter::mpfit_with_workspace`, which borrows them for one fit and leaves them behind, grown to
the problem's size. `MPFitter::mpfit` becomes a wrapper over a fresh workspace that copies the
residuals, covariance and errors into the status as before, so upstream callers see no change.
Every value written to `fjac` is the same, in the same order: the parity corpus (92 corpora,
38,083 rows) is bit-identical before and after. `src/gaussian.rs` keeps one workspace per thread
in a `thread_local!`, so after the first fit on a thread a fit makes no heap allocation. Numbers
under "Performance" in `docs/design-notes.rst`.

`perf.patch` is a second performance patch, again with no numerical effect: the Rust outputs over
the same 92-corpus parity set (and two 20,000-spectrum benchmark batches) are bit-identical before
and after. It makes three changes. The MINPACK `enorm` takes a branch-free fast path, the plain
sequential sum of squares, which is exactly what its three-sum algorithm reduces to when no
component is tiny or huge; the original is kept as `enorm_scaled` for the rest, and
`tests::enorm_fast_path_is_bit_identical_to_the_scaled_sum` pins the equivalence
(`cargo test --manifest-path third_party/rmpfit/Cargo.toml enorm_fast_path`). `fdjac2` no longer
zeroes `fjac` and the analytical derivative columns, since every element is overwritten before it
is read (a user `jacobian` must therefore fill every element of each column). And
`MPConfig::no_covar` lets a caller that derives its own errors (as `src/gaussian.rs` does) skip
the covariance pass at the end of the fit. Measured together, these cut the cycles per fit by 6 % at a 5-sample window and 3 % at 21
samples with the instruction count unchanged; numbers and the two rejected variants under
"Performance" in `docs/design-notes.rst`.

Every local change is marked `gaussfit-rs local patch` at the patch site. Nothing else in the
file differs from upstream; check with

```bash
cp -r third_party/rmpfit /tmp/rmpfit-check
patch -R -p1 -d /tmp/rmpfit-check < third_party/rmpfit/perf.patch
patch -R -p1 -d /tmp/rmpfit-check < third_party/rmpfit/workspace.patch
sha256sum /tmp/rmpfit-check/src/lib.rs
# 2bb95e44e8afa4751c0dafcf79cdfbda1b47594a52b5f47ef56099458863c12f
```

## Wiring

Consumed as a path dependency, `rmpfit = { path = "third_party/rmpfit" }` in the root
`Cargo.toml`. The sdist must ship this directory: `[tool.maturin] exclude` in `pyproject.toml`
excludes only `third_party/c_reference/**`, and CI installs from the sdist to prove it.
Dependabot ignores path dependencies (and skips any crate named in a `[patch]` table, so that
form buys nothing, and maturin leaves `[patch]` paths out of the sdist), so there is no
automated notice of new rmpfit releases; check crates.io when touching dependencies.

The crate is not a workspace member, so `cargo fmt` and `cargo clippy` leave it alone.

## License note

The published crate ships no `LICENSE` file, only `license = "MIT"` metadata. `LICENSE` here
is the standard MIT text with the author's attribution; the copyright year (2021, the crate's
first publish year) is unverified because the upstream host is behind an anti-bot wall.

## How to update

1. `cargo fetch` the new version or download the tarball from crates.io; copy `Cargo.toml`,
   `README.md` and `src/lib.rs` here (not the registry artifacts listed above).
2. Apply the patches in order: `patch -p1 -d third_party/rmpfit < third_party/rmpfit/<name>.patch`
   for `workspace`, then `perf`. If upstream gains a reusable workspace of its own, drop
   `workspace.patch` and point `src/gaussian.rs` at the upstream entry point; if it takes any
   part of `perf.patch`, drop that hunk. The gate for both is `benchmarks/throughput.py` plus
   the byte comparison under "Performance" in `docs/design-notes.rst`. Once neither patch is
   left, replace the path dependency with `rmpfit = "=<version>"` and delete this directory.
3. Update the version, checksums and date in this file, then regenerate each patch against its own
   base — not both against upstream. `workspace.patch` is
   `diff -u --label a/src/lib.rs --label b/src/lib.rs <upstream lib.rs> <upstream + workspace>`
   and `perf.patch` is the same command from that intermediate file to `src/lib.rs`.
   Regenerating them from upstream in one go would fold `workspace` hunks into `perf.patch` and
   break the reverse check above.
4. `cargo test`, then the live parity test with the C reference installed
   (`tox -r -e py313-cparity`, 46 seeds): Rust must never be worse than C, except where
   `ACCEPTED_WORSE` in `python/gaussfit_rs/tests/test_c_parity.py` names a documented exception.
   The gate fails if an exception is listed but no longer taken, so if an upstream change moves
   that row back under the normal contract, delete the entry and its design-notes section.
