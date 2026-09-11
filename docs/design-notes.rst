Design notes
============

Design decisions and known limitations, one section per decision: what the
code does, why, and what it costs. Each section names the code and the tests
that pin the behavior, so a change to either is a change to the decision.

Peak Detection And Velocity-Window Semantics
--------------------------------------------

- **Status:** resolved 2026-05-29 to preserve the original C extension behavior.
- **Owner:** Nabil Freij.
- **Code:** ``src/spectrum.rs::fit_single_spectrum_core`` (peak-detection block).
- **Tests encoding current behavior:**
  ``src/tests/spectrum.rs::rejects_peak_inside_slack_zone``,
  ``src/tests/spectrum.rs::returns_no_local_max_when_no_sample_in_search_window``.

Current Behavior
~~~~~~~~~~~~~~~~

``fit_single_spectrum_core`` does, in order:

1. Search for the brightest pixel within
   ``|dopp_slit[i] - guide_velocity| <= velocity_range + dv_slac``, where
   ``dv_slac = dv * npix_slack``.
2. Reject with ``FLAG_NO_LOCAL_MAX`` if the peak is empty, non-positive, or lies
   outside the strict ``velocity_range``.
3. Normalize the fitting window by the accepted peak and run the bounded
   Levenberg-Marquardt fit.

This means ``npix_slack`` is a local-maximum vetting band, not an acceptance
band. A peak found only in the slack band is rejected. This matches the MUSE C
extension (pinned as the test reference under ``third_party/c_reference``) and
avoids false-positive fits when the expected line is absent.

Tradeoff
~~~~~~~~

The strict re-check can mask a valid lower in-range peak if a brighter spurious
sample appears in the slack band. We accept that tradeoff here because this
package is used as a drop-in replacement for the C extension, and matching its
status semantics is more important than changing the scientific selection rule.

If the desired behavior changes later, add an explicit signal-presence criterion
instead of silently accepting slack-band peaks. A minimum peak signal-to-noise
or absolute peak threshold would be clearer than overloading ``npix_slack``.

Bound-Limited Steps And The MPFIT Snap Hazard
---------------------------------------------

- **Status:** resolved 2026-09-10 by patching the vendored solver.
- **Owner:** Nabil Freij.
- **Code:** ``third_party/rmpfit/src/lib.rs::MPFit::iterate`` (bound clamp; the
  local patch is ``third_party/rmpfit/bound-snap.patch``, documented in
  ``third_party/rmpfit/VENDORED.md``), consumed by
  ``src/gaussian.rs::fit_gaussian_bounded_with_config``.
- **Tests encoding current behavior:**
  ``src/tests/gaussian.rs::bound_pinned_window_matches_c_reference``,
  ``python/gaussfit_rs/tests/test_fit_gaussian.py::test_f32_bound_limited_step_does_not_stall``,
  ``python/gaussfit_rs/tests/test_c_parity.py::test_matches_live_c_extension``
  (46 seeds, needs the C reference).

Current Behavior
~~~~~~~~~~~~~~~~

MPFIT scales a Levenberg-Marquardt step that would cross a parameter bound by
``alpha = (limit - x) / step`` so the step stops at the bound, then snaps any
coordinate within one ULP of a limit onto it. In floating point
``x + alpha * step`` can land a couple of ULP past the limit, on either side. In
stock MPFIT (CMPFIT 1.5 and rmpfit 2.0.0 share this code line for line):

1. A landing just inside the bound is snapped, the parameter counts as pegged
   from then on, the outward gradient is zeroed, and the fit continues.
2. A landing just outside is not snapped. On the next iteration the parameter
   is not pegged, the step crosses the bound again, ``alpha`` collapses to about
   1e-14 for every parameter, the trust region collapses with it, and the
   solver reports ``Par`` (or ``Both``) convergence with no progress. The result
   carries ``FLAG_SUCCESS`` and, on the parity corpus, a reduced chi-square up
   to 74x higher than the C reference.

Which side a window lands on is decided by rounding order in the linear
algebra, so each build stalls on its own subset. Over 20 corpus seeds (14,197
windows fitted by both), stock rmpfit was worse than C on 13 rows and C was
worse than rmpfit on 13 others, judged by the same tolerance.

The vendored rmpfit records which bound set ``alpha`` and places that coordinate
exactly on its limit. Measured effect: 18 of 14,197 windows change, every one
an improvement; no window is worse than C f32, C f64 or stock rmpfit; Rust is
never worse than C on the 46 seeds the live parity test runs, nor on a
100-seed sweep (82,368 fits).

Tradeoff
~~~~~~~~

Rust now beats the C reference on the windows where C stalls. The parity
contract in ``helpers.assert_fit_parity`` is one-sided (Rust never worse), so
this is allowed and expected. Do not "repair" the corpus or widen the
tolerance to make those rows agree.

The patch is carried as a path dependency on ``third_party/rmpfit``. Dependabot
ignores path dependencies, and it also skips any crate named in a ``[patch]``
table, so that form buys nothing (maturin omits ``[patch]`` paths from the sdist
as well). New rmpfit releases must therefore be checked by hand; ``VENDORED.md``
has the update procedure, and the tests above decide when the patch can be
dropped.

``lmpar`` Trust-Region Clamp
----------------------------

- **Status:** resolved 2026-09-11 by patching the vendored solver.
- **Owner:** Nabil Freij.
- **Code:** ``third_party/rmpfit/src/lib.rs`` (the clamp in ``lmpar``), carried
  as ``third_party/rmpfit/lmpar-clamp.patch``; documented in ``VENDORED.md``.
- **Tests encoding current behavior:**
  ``python/gaussfit_rs/tests/test_c_parity.py::test_matches_live_c_extension``
  (``muse`` seed 44, family ``low_snr``, carries the named exception).

Current Behavior
~~~~~~~~~~~~~~~~

rmpfit clamps the trust-region radius with ``self.par = self.par.max(paru)``
where CMPFIT and MINPACK clamp the upper end with ``min`` (C reference:
``third_party/c_reference/vendor/cmpfit-1.5/mpfit.c:2102``). Every other line of the
``lmpar`` clamp/update sequence matches, so this is a one-token port deviation,
now corrected locally.

Tradeoff
~~~~~~~~

Measured effect over 100 seeds (82,368 corpus fits): 68 fits move onto the C
reference's values within tolerance, and one low-SNR fit (``muse`` seed 44,
family ``low_snr``, absolute row 165) moves to a different local minimum, reduced
chi-square 0.804337 against C's 0.755257 (1.065x). That single row is carried as
a **named exception** rather than a wider tolerance: ``ACCEPTED_WORSE`` in
``python/gaussfit_rs/tests/test_c_parity.py``, capped at 1.10x, restricted to
that row so the other 39 low-SNR rows stay under the gate, and asserted to be
taken exactly once, so the exception cannot silently go stale and the one-sided
contract stays meaningful everywhere else.

On a full MUSE run (2026-09-11) the change is close to chi-square-neutral and
slightly closer to the C reference: the clean GT product is bit-identical, the
gate's 0.5 % chi-square criterion is never newly violated, and roughly two thirds
of the pixels whose fits change move toward C. Per-product measurements, and the
provenance for the pipeline data they come from, are in
``docs/improvement-todos.md`` (B4); they are not repeated here because the MUSE
pipeline data is not part of this repository. If a future rmpfit release fixes
``lmpar`` upstream, drop the patch and delete the exception and this section.

Opt-In Unconstrained-Fit Indicator
----------------------------------

- **Status:** resolved 2026-09-11; opt-in, default off.
- **Owner:** Nabil Freij.
- **Code:** ``src/spectrum.rs::fit_prepared_spectrum_window`` (the predicate),
  ``src/api.rs`` (the ``quality`` keyword on the three spectrum entry points) and
  ``python/gaussfit_rs/fitting.py`` (the public wrappers).
- **Tests encoding current behavior:**
  ``src/tests/spectrum.rs::quality_flag_separates_constrained_and_unconstrained_fits``,
  ``python/gaussfit_rs/tests/test_fit_single_spectrum.py::test_quality_flag_is_opt_in_and_reports_unconstrained_successes``,
  ``python/gaussfit_rs/tests/test_fit_single_spectrum.py::test_batch_quality_flags_match_the_documented_criterion``,
  and the figure suite (``test_quality_per_family``, and the ``UNC`` marker in
  ``test_fit_gallery`` / ``test_family_overview``).

Current Behavior
~~~~~~~~~~~~~~~~

Both backends return ``FLAG_SUCCESS`` for spaxels whose parameters the data do not actually
constrain, and the pipeline propagates them into the moments and area statistics: on a real MUSE
run 8-9 % of spaxels carry a median velocity error of 200-244 km/s while their amplitude is 2.6-3.6
against 17.8-18.9 overall. With ``quality=True`` those spaxels are reported instead of being
indistinguishable from a good fit: the call appends a ninth column to the result array
(``fit_results[8]``, ``fit_results[:, 8]`` for a batch), 1 for a successful but unconstrained fit
and 0 otherwise, including for failed fits.

Criterion
~~~~~~~~~

A successful fit is flagged when any of:

* the velocity error is not smaller than the velocity interval it was fitted in, ``2*dv*npix``;
* the linewidth error is not smaller than ``width_max - width_min``;
* any of the three errors is exactly 0. That is physically impossible; the near-singular guard
  (``src/gaussian.rs``, ``|det| < 1e-30``) returns zeros on near-zero-flux windows, where the C
  backend's fallback still returns a finite number. Measured on the summed cube of a real run, 193
  of 3,459 solved spaxels are zeroed this way, and 19 of the 336 solved rows of the ``muse``
  corpus.

An amplitude term -- the normalised amplitude error against ``amplitude_rel_max -
amplitude_rel_min`` -- was implemented first and then dropped: in the pipeline configuration those
bounds are +/-10 % of the peak, a detection window rather than a physical range, so the term fires
on ordinary low-SNR fits. Measured on the summed cube of a real run, adding it raises the flag rate
from 11.2 % to 45.2 % of solved spaxels (1,384 of 3,459 against 205 for the velocity interval
alone), i.e. it swamps the population the indicator is for. Dropping it also makes the predicate
reproducible from the returned columns.

Measured rate (2026-09-11): 11.2 % of the 3,459 solved spaxels of the real run's summed cube at the
pipeline configuration (``npix=2``, ``+/-500 km/s``, ``dv=40.740``, ``width`` 5-200 km/s), 7.4 % of
the 336 solved rows of the ``muse`` C-reference corpus and 3.0 % of the 367 solved rows of the
``wide`` corpus. The Muse-era 8-9 % figure quoted above counts spaxels with a *median velocity
error* of 200-244 km/s, which is a different population and not the indicator's rate.

Limitations
~~~~~~~~~~~

* A parameter pinned at a bound with small formal errors is not caught. Measured on the corpora,
  including a pegged term would flag roughly 30 % of the ``muse`` solved rows, 15 of 15 ``broad``
  rows among them: amplitude saturation is normal in that configuration (79 of 336 rows; the
  ``wide`` corpus has none), so it is a modelling signal, not an unconstrained one. Widening the
  criterion would cost the specificity that makes the indicator usable as a mask.
* Faint lines whose errors are small for the wrong reason are not caught when the errors are
  nonzero. Just above the guard the cofactor is still near-singular, so a line far below the noise
  can return errors around 1e-6 (e.g. peak/noise ~0.2 with the amplitude and width bounds of the
  Rust tests) while carrying no information. Only the exactly-zero case is detectable from the
  returned columns.

Tests
~~~~~

``src/tests/spectrum.rs::quality_flag_separates_constrained_and_unconstrained_fits`` (clean,
velocity/width-driven, guard-zeroed and failed fits),
``python/gaussfit_rs/tests/test_fit_single_spectrum.py::test_quality_flag_is_opt_in_and_reports_unconstrained_successes``
and ``...::test_batch_quality_flags_match_the_documented_criterion`` (the batch indicator equals the
velocity/width/zero rule recomputed from the returned columns), plus the figure suite
(``test_quality_per_family``, and the ``UNC`` marker in ``test_fit_gallery`` /
``test_family_overview``).
