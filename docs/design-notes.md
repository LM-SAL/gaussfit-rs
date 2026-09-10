# Design notes

Internal design decisions and known limitations. Not part of the published
Sphinx docs (no `myst-parser`, so `.md` files here are ignored by the build).

---

## Peak Detection And Velocity-Window Semantics

**Status:** resolved 2026-05-29 to preserve the original C extension behavior.
**Owner:** Nabil Freij.
**Code:** `src/spectrum.rs::fit_single_spectrum_core` (peak-detection block).
**Tests encoding current behavior:**
`src/tests/spectrum.rs::rejects_peak_inside_slack_zone`,
`src/tests/spectrum.rs::returns_no_local_max_when_no_sample_in_search_window`.

### Current Behavior

`fit_single_spectrum_core` does, in order:

1. Search for the brightest pixel within
   `|dopp_slit[i] - guide_velocity| <= velocity_range + dv_slac`, where
   `dv_slac = dv * npix_slack`.
2. Reject with `FLAG_NO_LOCAL_MAX` if the peak is empty, non-positive, or lies
   outside the strict `velocity_range`.
3. Normalize the fitting window by the accepted peak and run the bounded
   Levenberg-Marquardt fit.

This means `npix_slack` is a local-maximum vetting band, not an acceptance band.
A peak found only in the slack band is rejected. This matches the deleted C
extension and avoids false-positive fits when the expected line is absent.

### Tradeoff

The strict re-check can mask a valid lower in-range peak if a brighter spurious
sample appears in the slack band. We accept that tradeoff here because this
package is currently being used as a C replacement, and matching the C status
semantics is more important than changing the scientific selection rule.

If the desired behavior changes later, add an explicit signal-presence criterion
instead of silently accepting slack-band peaks. A minimum peak signal-to-noise
or absolute peak threshold would be clearer than overloading `npix_slack`.

---

## Bound-Limited Steps And The MPFIT Snap Hazard

**Status:** resolved 2026-09-10 by patching the vendored solver.
**Owner:** Nabil Freij.
**Code:** `third_party/rmpfit/src/lib.rs::MPFit::iterate` (bound clamp; local patch in
`third_party/rmpfit/bound-snap.patch`, documented in `third_party/rmpfit/VENDORED.md`),
consumed by `src/gaussian.rs::fit_gaussian_bounded_with_config`.
**Tests encoding current behavior:**
`src/tests/gaussian.rs::bound_pinned_window_matches_c_reference`,
`python/gaussfit_rs/tests/test_fit_gaussian.py::test_f32_bound_limited_step_does_not_stall`,
`python/gaussfit_rs/tests/test_c_parity.py::test_matches_live_c_extension` (46 seeds, needs the C reference).

### Current Behavior

MPFIT scales a Levenberg-Marquardt step that would cross a parameter bound by
`alpha = (limit - x) / step` so the step stops at the bound, then snaps any
coordinate within one ULP of a limit onto it. In floating point
`x + alpha * step` can land a couple of ULP past the limit, on either side. In
stock MPFIT (CMPFIT 1.5 and rmpfit 2.0.0 share this code line for line):

1. A landing just inside the bound is snapped, the parameter counts as pegged
   from then on, the outward gradient is zeroed, and the fit continues.
2. A landing just outside is not snapped. On the next iteration the parameter
   is not pegged, the step crosses the bound again, `alpha` collapses to about
   1e-14 for every parameter, the trust region collapses with it, and the
   solver reports `Par` (or `Both`) convergence with no progress. The result
   carries `FLAG_SUCCESS` and, on the parity corpus, a reduced chi-square up to
   74x higher than the C reference.

Which side a window lands on is decided by rounding order in the linear
algebra, so each build stalls on its own subset. Over 20 corpus seeds (14,197
windows fitted by both), stock rmpfit was worse than C on 13 rows and C was
worse than rmpfit on 13 others, by the same one-sided contract.

The vendored rmpfit records which bound set `alpha` and places that coordinate
exactly on its limit. Measured effect: 18 of 14,197 windows change, every one
an improvement; no window is worse than C f32, C f64 or stock rmpfit; Rust is
never worse than C on any of the 46 seeds the live parity test runs.

### Tradeoff

Rust now beats the C reference on the windows where C stalls. The parity
contract in `helpers.assert_fit_parity` is one-sided (Rust never worse), so
this is allowed and expected. Do not "repair" the corpus or widen the
tolerance to make those rows agree.

The patch is carried as a path dependency on `third_party/rmpfit`. Dependabot
ignores path dependencies (and skips any crate named in a `[patch]` table, so
that form buys nothing; maturin also omits `[patch]` paths from the sdist),
so new rmpfit releases must be checked by hand; `VENDORED.md` has the update
procedure, and the tests above decide when the patch can be dropped.
rmpfit's `lmpar` also clamps `par` with `max(paru)`
where MINPACK uses `min`; that is a separate port deviation, reported upstream
and deliberately not patched here because it changed one corpus row for the
worse.
