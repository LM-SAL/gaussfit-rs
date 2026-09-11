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
extension (pinned as the test reference under ``third_party/c_reference``).
It does not establish a line detection: a positive noise peak inside the strict
search interval can still be fitted successfully.

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
  ``src/gaussian.rs::fit_gaussian_bounded``.
- **Tests encoding current behavior:**
  ``src/tests/gaussian.rs::bound_pinned_window_matches_c_reference``,
  ``python/gaussfit_rs/tests/test_fit_gaussian.py::test_bound_limited_step_does_not_stall``,
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
exactly on its limit. With the bound-snap patch alone, the recorded measurements
were: 18 of 14,197 windows change, every one an improvement; no window is worse
than C f32, C f64 or stock rmpfit; Rust is never worse than C on the 46-seed live
parity test or a 100-seed sweep (82,368 fits). The subsequent ``lmpar`` correction
below introduces one documented exception to that last claim.

Tradeoff
~~~~~~~~

Rust now beats the C reference on the windows where C stalls. The parity
contract in ``helpers.assert_fit_parity`` is one-sided, apart from the named
``lmpar`` exception below, so improvements are allowed and expected. Do not
"repair" the corpus or widen the
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

Upstream rmpfit 2.0.0 clamps the Levenberg-Marquardt damping parameter ``par``
with ``self.par = self.par.max(paru)`` where CMPFIT and MINPACK use ``min``
(C reference:
``third_party/c_reference/vendor/cmpfit-1.5/mpfit.c:2102``). Every other line of the
``lmpar`` clamp/update sequence matches. The vendored code corrects this with
``self.par = self.par.min(paru)``. The trust-region radius is ``delta``; it is
not the variable changed by this patch.

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

The commit history of 2026-09-11 records separate pipeline comparisons on MUSE
simulation data. Those measurements are not an observational validation and are
not part of the committed parity fixtures. If a future rmpfit release fixes
``lmpar`` upstream, drop the local patch after checking parity. Keep the exception
until the corresponding row actually satisfies the normal gate: moving the same
fix upstream does not by itself change that row's behavior.

Unconstrained-Fit Indicator
---------------------------

- **Status:** resolved 2026-09-11; reported for every fit in the ninth result column.
- **Owner:** Nabil Freij.
- **Code:** ``src/spectrum.rs::fit_prepared_spectrum_window`` (the predicate),
  ``src/api.rs`` (the ``quality`` argument on the two Rust spectrum entry points) and
  ``python/gaussfit_rs/fitting.py`` (the public wrappers).

Current Behavior
~~~~~~~~~~~~~~~~

Both backends can return ``FLAG_SUCCESS`` for weakly constrained fits, including
fits to positive noise fluctuations. Convergence does not establish a detected
line or a reliable parameter measurement. With the ninth result column, the spectrum
entry points append a ninth float32 column (``fit_results[8]``, or
``fit_results[:, 8]`` for a batch) reporting three quality bits. Failed fits have
zero quality bits and must be checked separately using column 7. The default
eight-column output and the fitted values are unchanged by enabling quality.

Bits, not a boolean
~~~~~~~~~~~~~~~~~~~

* **1 (``QUALITY_UNCONSTRAINED``)** -- the velocity error is not smaller than ``2*dv*npix``, or the
  linewidth error is not smaller than ``width_max - width_min``. This reports a
  formal error at least as large as a parameter's allowed interval; it is a
  diagnostic threshold, not a calibrated detection or confidence test.
* **2 (``QUALITY_ZERO_ERROR``)** -- a formal error is exactly 0. The near-singular guard
  (``src/gaussian.rs``, nonfinite determinant or ``|det| < 1e-30``) returns zeros.
  Nonpositive computed variances also become zero, and residual scaling can
  produce zeros on an exact fit with a nonsingular matrix. The absolute
  determinant threshold depends on scale; a small determinant alone is not a
  condition-number test. This bit reports the error value, not its cause.
* **4 (``QUALITY_PEGGED``)** -- a fitted parameter sits exactly on one of its bounds (rmpfit clamps
  onto the bound exactly, so equality is the test).

They are reported separately because they mean different things and a single boolean forces one
policy on every consumer. Measured on the ``muse`` corpus (336 solved rows, 117 with bit 4 = 34.8 %),
pegging is *not* evidence of an unconstrained fit: 15 of 15 ``broad`` rows and 11 of 17 ``narrow``
rows set it, but so do 5 of 26 ``clean`` rows -- a broad line legitimately pins at ``width_max``.
After converting column 8 to integers (see the user guide), ``flags != 0``
selects any bit and ``(flags & (1|2)) != 0`` selects large or zero errors for
inspection. Neither is an automatic rejection policy: pegged and exact fits
can be legitimate, and an unflagged fit is not guaranteed to be useful.

The amplitude interval is deliberately excluded: the MUSE configuration uses
+/-10 % of the normalization peak, a detection window rather than a physical
range. Comparing the amplitude error with that interval would also flag
ordinary low-SNR fits.

The MUSE ``fit_single_spectrum_fallback`` calls ``ftoolss.fmpfit_f32_pywrap``,
which delegates to a C extension. Its agreement with the direct C path is not
an independent pure-Python solver comparison. The quality column is implemented
in gaussfit-rs; the C path would need its own implementation to return it.

Measured rates (2026-09-11), as fractions of successful fits in the synthetic
C-reference fixtures, using each fixture's recorded configuration:

.. list-table::
   :header-rows: 1

   * - sample (solved rows)
     - bit 1 span
     - bit 2 zero
     - bit 4 pegged
     - any (``!= 0``)
   * - ``muse`` corpus, 336
     - 1.8 %
     - 5.7 %
     - 34.8 %
     - 35.4 %
   * - ``wide`` corpus, 367
     - 3.0 %
     - 0 %
     - 11.2 %
     - 11.2 %

Saved Simulation Examples And Covariance Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The saved ``baseline_gpu/spectrum_summed_all.zarr`` input is MURaM-based MUSE
simulation output. Its configuration names a VDEM model and has a null
observational filename. An earlier report fitted a reconstructed velocity axis
with a +/-500 km/s search and sigma bounds of 5-200 km/s. The saved Fe XIX
pipeline output instead uses its response-derived per-slit axis, a +/-300 km/s
search, minimum sigma 59.216967 km/s, initial sigma equal to that minimum plus
10 km/s, and no peak-search slack. Both examples use ``npix=2`` and fitting
noise of 1 DN, not the cube's stored measurement-error array.

The earlier 11.2 % suspect and 42.5 % any-bit figures are not validated rates
for the saved pipeline configuration or observations. The following selected
examples establish problematic fits, not an occurrence rate:

* With the reconstructed settings, flattened row 379 fits pixels ``[522, 527)``
  with flux ``[6, -1, 9, 0, 4]`` DN. Rust returns sigma 7.5834 km/s, below the
  40.74-km/s sample spacing, and velocity error 1,124,729.75 km/s. C returns
  similar parameters and zero errors; both report success.
* The largest successful saved Fe XIX velocity error occurs at line index 0,
  y index 89, x index 30 (slit index 5, step index 0). Refitting with the saved
  configuration reproduces all eight Rust values and the mask ``[170, 175)``.
  Flux is ``[6, 0, 1, 1, 1]`` DN. Rust and C return velocity errors of
  3837.236572 and 3837.231201 km/s respectively; Rust quality is 5. The
  normalization peak is 1 DN because the 6-DN sample is outside the peak-search
  interval, although it lies inside the fit window. The fit reaches its
  amplitude upper, velocity lower, and sigma upper bounds.

The MUSE benchmark applies an intensity mask after fitting, before comparison
statistics. For the saved example, fitted net flux is 13.5453 DN against a
62.9021-DN threshold; both the GFAT intensity mask and the saved ground-truth
mask reject it. The saved combined mask also rejects it. Its presence in
``mom_gfat`` does not demonstrate contamination of the masked statistics.
The ninth-column diagnostics complement the caller's masking; their benefit
for the retained population needs to be measured on that population.

The stored ``clean_flux`` trace also needs care. MUSE preserves the expected
signal before Poisson sampling, but the inspected ``ph2dn`` path converts only
``flux``. The readout step truncates ``clean_flux`` to integers and labels it
DN. The saved zeros therefore do not establish zero emission or a calibrated
DN expectation. The floating-point pre-noise synthesis gives nonzero expected
photons in both cuts: approximately 0.38-0.95 per sample in the reconstructed
example and 0.022-0.051 in the saved pipeline example.

Both implementations form the weighted Gaussian Jacobian ``J`` and derive
formal errors from the inverse of ``J.T @ J``, scaled by
``sqrt(chi2 / (N - 3))`` when ``N > 3``. These are local covariance estimates,
not confidence intervals respecting the bounds. In the narrow example, the
C float32 cofactor calculation produces determinant ``-8.0409013e-11`` and
negative diagonal variances. ``mp_xerror_scipy`` accepts ``abs(det) > 1e-30``
and explicitly converts nonpositive variances to zero. This example does not
enter the Gauss-Jordan fallback; the success status is determined separately.

Rust computes covariance in float64 at its internal parameters and retains
positive variances here. Its million-scale uncertainty is numerically unstable:
direct SVD of the Jacobian at the rounded returned parameters gives about
2.89 million km/s. In the saved pipeline example, SVD gives 3837.23663 km/s,
agreeing with both backends. A rank-aware QR/SVD covariance calculation would
address inversion instability; it would not by itself change residual scaling
or establish a detection.

Limitations
~~~~~~~~~~~

* Bit 4 is a report, not a verdict, for the reason above; a consumer that masks on it will drop
  good broad lines.
* The span bit needs the returned errors and configured velocity/width intervals.
  Reconstructing pegging additionally needs the actual bounds and normalization
  peak; the physical amplitude alone does not identify its normalized bound.
* Residual scaling can produce small nonzero errors for a nearly exact noiseless
  fit, even with a nonsingular matrix. Covariance inversion can also be unstable.
  Zero-error detection alone does not identify all unreliable uncertainties.
* A positive peak is not a signal-to-noise test, and a zero quality mask is not a
  detection certificate. Apply the caller's intensity masks and evaluate any
  additional quality policy on the fits that survive them.

Tests
~~~~~

``src/tests/spectrum.rs::quality_bits_separate_constrained_unconstrained_zero_and_pegged_fits``
(clean, span, guard-zeroed, pegged and failed fits),
``python/gaussfit_rs/tests/test_fit_single_spectrum.py::test_quality_bits_are_opt_in_and_report_unconstrained_successes``
and ``...::test_batch_quality_bits_match_the_documented_criterion`` (bits 1 and 2 equal the
velocity/width/zero rule recomputed from the returned columns), plus the figure suite
(``test_quality_per_family``, and the ``q`` marker in ``test_fit_gallery`` /
``test_family_overview``).

Performance
-----------

- **Status:** measured 2026-09-11; per-fit allocation removed, the rest attributed.
- **Owner:** Nabil Freij.
- **Code:** ``third_party/rmpfit/workspace.patch`` (``MPWorkspace``, ``mpfit_with_workspace``),
  the per-thread workspace in ``src/gaussian.rs``, the per-thread window buffers in
  ``src/spectrum.rs``.
- **Harness:** ``benchmarks/throughput.py`` (``--npix`` sweeps the window; the fitted intercept
  is the per-fit overhead, the slope the per-sample arithmetic).

What A Fit Costs
~~~~~~~~~~~~~~~~

Single-threaded, a fit costs about 1.4 us fixed plus 0.08 us per sample, so at the pipeline's
5-sample window most of the time is per-fit work, not arithmetic. Upstream rmpfit made about
28 heap allocations per fit; the workspace patch removed all of them for 0.24 us per fit
(the intercept went from 1.78 to 1.43 us on the synthetic sweep, 0.2-0.35 us per row on real
MUSE rows) with bit-identical output over the parity corpus. glibc's tcache makes an
allocation cost about 10 ns, so the gain is bounded; a faster allocator does not help.

Against The C++ Batcher
~~~~~~~~~~~~~~~~~~~~~~~

Compared with MUSE's ``fastfit2`` C++ extension on real rows at matched thread counts, Rust is
level with a plain ``-O3`` build of the same cmpfit and 15-20 % slower per row single-threaded
than the extension Python's default ``CFLAGS`` produce (``-fno-strict-overflow`` alone halves
GCC's cost in the cmpfit helpers); at 32 threads Rust is faster. Iteration counts are not the
cause: on identical rows Rust accepts 3-5 % more steps and evaluates the model 1-5 % fewer times.
Nor are precision (the same C++ in double is within 2 % of float), ``exp``, the peak search or
the ISA (``x86-64-v3`` buys 3-4 %, documented as an opt-in build in the installation guide).
Instruction counts put the remainder in the port's per-iteration bookkeeping (``iterate``,
``qrfac``, ``transpose``): bounds checks and ``Vec`` indexing where cmpfit walks raw pointers.
A patch binding each stage's buffers to local slices recovered 3 % for an 864-line diff and was
not kept. The solver is under 1 % of a pipeline run; the end-to-end lever was the per-slit
staging on the caller's side, which the ``slit_index`` argument of ``fit_spectra_batch`` removes.
