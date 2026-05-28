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
