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
- **Tests:** ``src/tests/spectrum.rs::rejects_peak_inside_slack_zone``,
  ``src/tests/spectrum.rs::returns_no_local_max_when_no_sample_in_search_window``.

``fit_single_spectrum_core`` searches for the brightest pixel within
``|dopp_slit[i] - guide_velocity| <= velocity_range + dv * npix_slack``, rejects with
``FLAG_NO_LOCAL_MAX`` if the peak is empty, non-positive, or lies outside the strict
``velocity_range``, and otherwise normalises the fitting window by the peak and runs the
bounded Levenberg-Marquardt fit. ``npix_slack`` is therefore a vetting band, not an acceptance
band: a peak found only in the slack band is rejected, as in the MUSE C extension pinned under
``third_party/c_reference``. Nothing here establishes a line detection; a positive noise peak
inside the strict interval is fitted.

**Tradeoff.** A brighter spurious sample in the slack band masks a valid lower in-range peak.
Accepted because this package is a drop-in replacement for the C extension and its status
semantics matter more than a better selection rule. If that changes, add an explicit
signal-presence criterion (a minimum peak signal-to-noise or absolute threshold) rather than
overloading ``npix_slack``.

Bound-Limited Steps And The MPFIT Snap Hazard
---------------------------------------------

- **Status:** resolved 2026-09-10 by patching the vendored solver.
- **Owner:** Nabil Freij.
- **Code:** ``third_party/rmpfit/bound-snap.patch`` (``MPFit::iterate``), documented in
  ``third_party/rmpfit/VENDORED.md``; consumed by ``src/gaussian.rs::fit_gaussian_bounded``.
- **Tests:** ``src/tests/gaussian.rs::bound_pinned_window_matches_c_reference``,
  ``python/gaussfit_rs/tests/test_fit_gaussian.py::test_bound_limited_step_does_not_stall``,
  ``python/gaussfit_rs/tests/test_c_parity.py::test_matches_live_c_extension`` (46 seeds, needs
  the C reference).

MPFIT scales a step that would cross a parameter bound by ``alpha = (limit - x) / step`` and
snaps coordinates within one ULP of a limit onto it. ``x + alpha * step`` can land a couple of
ULP past the limit on either side. Just inside, the parameter is pegged and the fit continues.
Just outside, it is not snapped: the next step crosses the bound again, ``alpha`` collapses to
about 1e-14 for every parameter, the trust region with it, and the solver reports ``Par`` (or
``Both``) with no progress, so ``FLAG_SUCCESS`` arrives with a reduced chi-square up to 74x the
C reference's. Rounding order decides the side, so each build stalls on its own subset of fits.
CMPFIT 1.5 and rmpfit 2.0.0 share this code line for line.

The vendored rmpfit records which bound set ``alpha`` and places that coordinate exactly on its
limit. Over 14,197 corpus windows 18 fits change, every one an improvement, and Rust is never
worse than C except for the one ``lmpar`` exception below.

**Tradeoff.** The parity contract in ``helpers.assert_fit_parity`` is one-sided: Rust may beat
C and does on these windows. Do not "repair" the corpus or widen the tolerance to make those
rows agree. The patch rides on a path dependency, which Dependabot ignores (as it does crates
named in a ``[patch]`` table), so new rmpfit releases are checked by hand; ``VENDORED.md`` has
the procedure and the tests above decide when the patch can be dropped.

``lmpar`` Trust-Region Clamp
----------------------------

- **Status:** resolved 2026-09-11 by patching the vendored solver.
- **Owner:** Nabil Freij.
- **Code:** ``third_party/rmpfit/lmpar-clamp.patch``, documented in ``VENDORED.md``.
- **Tests:** ``python/gaussfit_rs/tests/test_c_parity.py::test_matches_live_c_extension``
  (``muse`` seed 44, family ``low_snr``, carries the named exception).

Upstream rmpfit 2.0.0 clamps the Levenberg-Marquardt damping parameter with
``self.par = self.par.max(paru)`` where CMPFIT and MINPACK use ``min``
(``third_party/c_reference/vendor/cmpfit-1.5/mpfit.c:2102``); the vendored copy uses ``min``.
Over 100 corpus seeds (82,368 fits) 68 fits move onto the C reference's values within tolerance
and one low-SNR fit (``muse`` seed 44, row 165) lands in a different local minimum, reduced
chi-square 0.804 against C's 0.755 (1.065x).

**Tradeoff.** That row is a named exception, ``ACCEPTED_WORSE`` in ``test_c_parity.py``, capped
at 1.10x, restricted to that row and asserted to be taken exactly once, rather than a wider
tolerance, so the one-sided contract stays meaningful everywhere else. If upstream fixes
``lmpar``, drop the patch after checking parity, but keep the exception until the row passes
the normal gate.

Unconstrained-Fit Indicator
---------------------------

- **Status:** resolved 2026-09-11; reported for every fit in the ninth result column.
- **Owner:** Nabil Freij.
- **Code:** ``src/spectrum.rs::fit_single_spectrum_core`` (the predicate), ``src/api.rs`` (the
  ninth column of every result row), ``python/gaussfit_rs/fitting.py`` (the public wrappers).
- **Tests:**
  ``src/tests/spectrum.rs::quality_bits_separate_constrained_unconstrained_zero_and_pegged_fits``,
  ``python/gaussfit_rs/tests/test_fit_single_spectrum.py::test_quality_bits_report_unconstrained_successes``
  and ``::test_batch_quality_bits_match_the_documented_criterion``, and the figure suite
  (``test_quality_per_family``, the ``q`` marker in ``test_fit_gallery`` /
  ``test_family_overview``).

Both backends return ``FLAG_SUCCESS`` for weakly constrained fits, including fits to positive
noise fluctuations; convergence is not a detection. Column 8 of every result row therefore
carries three quality bits (zero for failed fits, which column 7 identifies):

* **1 (``QUALITY_UNCONSTRAINED``)**: the velocity error is not smaller than ``2 * dv * npix``, or
  the width error is not smaller than ``width_max - width_min``. A formal error at least as
  large as the parameter's allowed interval; a diagnostic threshold, not a calibrated test.
* **2 (``QUALITY_ZERO_ERROR``)**: a formal error is exactly zero, from the near-singular guard in
  ``src/gaussian.rs`` (non-finite determinant or ``|det| < 1e-30``), a non-positive computed
  variance, or residual scaling on an exact fit. The bit reports the value, not its cause.
* **4 (``QUALITY_PEGGED``)**: a fitted parameter sits exactly on a bound (rmpfit clamps onto the
  bound exactly, so equality is the test).

They are bits, not a boolean, because they mean different things and one flag would force one
policy on every consumer. Pegging in particular is a report, not a verdict: on the ``muse``
corpus 117 of 336 solved rows set bit 4, including 15 of 15 ``broad`` rows, a broad line
legitimately pinned at ``width_max``. On the synthetic fixtures 35 % of solved ``muse`` rows and
11 % of ``wide`` rows set some bit, almost all of it bit 4. After converting column 8 to
integers (see the user guide), ``flags != 0`` selects any bit and ``(flags & (1|2)) != 0`` large
or zero errors; neither is an automatic rejection policy. The amplitude interval is excluded on
purpose: the MUSE configuration uses +/-10 % of the normalisation peak, a detection window
rather than a physical range, and comparing the amplitude error with it would flag ordinary
low-SNR fits.

**Limitations.** Formal errors come from the inverse of ``J.T @ J`` scaled by
``sqrt(chi2 / (N - 3))``: local covariance estimates in float64 at the fitted parameters, not
confidence intervals that respect the bounds. Near a singular Hessian the inverse is unstable,
so a fit can carry a huge error or, through the guard, a zero one; zero-error detection does not
identify every unreliable uncertainty. Bit 1 needs the configured velocity and width intervals;
reconstructing bit 4 from physical amplitudes needs the normalisation peak as well. Apply the
caller's intensity masks first and evaluate any quality policy on the fits that survive them.

Performance
-----------

- **Status:** measured 2026-10-20 on a Ryzen 9 9950X; per-fit allocation removed, solver
  bookkeeping trimmed, the remainder attributed by instruction.
- **Owner:** Nabil Freij.
- **Code:** ``third_party/rmpfit/workspace.patch`` (``MPWorkspace``, ``mpfit_with_workspace``)
  and ``third_party/rmpfit/perf.patch`` (``enorm`` fast path, ``fdjac2`` fills, finite check,
  ``MPConfig::no_covar``), the per-thread workspace in ``src/gaussian.rs``, the per-thread
  window buffers in ``src/spectrum.rs``.
- **Harness:** ``benchmarks/throughput.py`` times the public batch path (``--npix`` sweeps the
  window; the fitted intercept is the per-fit overhead, the slope the per-sample arithmetic).
  ``src/tests/bench.rs`` fits the same data without Python, NumPy or Rayon
  (``taskset -c N cargo test --release bench_fit_window -- --ignored --nocapture``); under
  ``perf stat -e cycles:u,instructions:u`` with a ``BENCH_ROUNDS=0`` run subtracted it gives
  counts per fit. Judge changes by those counts: wall-clock A/B of identical binaries varies
  2-8 % on this machine even pinned and interleaved.
- **Gate for any change:** outputs over the 92 ``build_cases`` corpora (46 seeds x muse/wide)
  compared as raw bytes before and after. Every patch above is bit-identical.

What A Fit Costs
~~~~~~~~~~~~~~~~

Three parameters, analytical Jacobian, about 6 accepted steps and 12 evaluations (Jacobian
calls included) per fit. The fit is throughput-bound, not latency-bound:

.. list-table::
   :header-rows: 1

   * - window (samples)
     - instructions per fit
     - cycles per fit
     - batch, one thread
   * - 5
     - 40,700
     - 10,100 (IPC 4.0)
     - 1.9 us
   * - 9
     - 48,100
     - 11,300
     - 2.1 us
   * - 21
     - 82,100
     - 17,200
     - 3.3 us

Where the instructions go (callgrind, 5-sample window): about 23 % are slice bounds checks,
``Vec`` pointer and length reloads and loop counters, spread over every solver stage; ``enorm``
8 % (about a hundred calls per fit on 3- to 5-element vectors); the model and glibc ``exp``
16 %; ``iterate`` 7 %, half of it the box-constraint logic; per-fit setup (``MPFit::new``,
``parse_params``, ``init_lm``) 5 %; ``qrfac`` 4 %, ``lmpar`` 3 %, ``transpose`` 2 %; the peak
search and window copy under 2 %. No single loop dominates.

What the patches bought: the workspace removed about 28 heap allocations per fit, 0.24 us (the
intercept went from 1.78 to 1.43 us). ``perf.patch`` cut cycles per fit by 7 % at 5 samples, 6 %
at 9 and 4 % at 21 with the instruction count unchanged, by inlining ``enorm`` and removing the
data-dependent branches from its loop (its three-sum form reduces exactly to the plain sum of
squares when no component is tiny or huge) and by dropping fills and a covariance pass whose
results were never read. Against MUSE's ``fastfit2`` C++ batcher (the same cmpfit, measured
before that reference was dropped from the repo) Rust was level with an ``-O3`` build
single-threaded, 15-20 % slower than the extension built with Python's default ``CFLAGS``, and
faster at 32 threads.

Measured And Rejected
~~~~~~~~~~~~~~~~~~~~~

All bit-identical in output; recorded so they are not tried twice.

* Binding each solver stage's buffers to local slices to drop bounds checks: 3 % for an
  864-line diff.
* ``MPFit`` owning the workspace buffers for a fit (moved out in ``new``, back in ``Drop``) so
  the compiler need not reload ``Vec`` headers: +3 % instructions, +4 % cycles. Moving 23
  vectors costs more than the reloads.
* A lane-wise, branch-free peak search (eight running maxima merged by lowest index): 1-2 %
  slower. The serial scan is under 2 % of a fit and mispredicts only a few times.
* Precision, ``exp`` and the ISA are not the cost: double against float is within 2 %, and
  ``-C target-cpu=x86-64-v3`` buys 3-4 %, kept as the opt-in build in the installation guide.

The remaining lever is the 23 % in bounds checks and header reloads, which needs the hot loops
of ``qrfac``, ``transpose``, ``lmpar`` and ``iterate`` rewritten over iterators or split slices.
The solver is under 1 % of a pipeline run; the end-to-end lever was per-slit staging on the
caller's side, which the ``slit_index`` argument of ``fit_spectra_batch`` removes.

Threads And NumPy's BLAS Pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit_spectra_batch`` releases the GIL and fits rows in parallel with Rayon: 1.9 us per fit on
one thread, 0.15 us on 16 and 0.12 us on 32 (16 cores with SMT). Results never depend on thread
counts. NumPy's OpenBLAS workers busy-wait for about 60 ms after a BLAS call on the SMT siblings
of the fitting threads: a 400,000-spectrum batch started right after a matrix product ran 1.75x
slower, and every batch after that window at full speed. The cost is one spin window per
BLAS-to-fit transition, negligible for a long batch. ``benchmarks/throughput.py`` sets
``OPENBLAS_NUM_THREADS=1`` before importing NumPy; a pipeline should do the same only if its own
BLAS work is light.
