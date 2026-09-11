# Improving gaussfit-rs vs the C backend

Status: revised 2026-09-10 (fourth revision). Numbers tagged **[measured]** /
**[inferred]** / **[speculative]**.

Reproduce with the committed harnesses (not the session-scoped `/tmp` files):
`muse/benchmark/probe_fastfit_staging.py TARGET_SCRIPT [--out JSON]` (per-call
staging probe; the target script's own `savepath` decides what is written) and
`muse/benchmark/ab_fastfit_cpp_vs_rust.py` (needs `SPECTRA_DIR`, optional
`CPP_SO`/`CPP_THREADS`/`RAYON_NUM_THREADS`). Raw evidence
(`/tmp/fastfit_instrument.json`, `/tmp/gfat_ab.json`, `/tmp/gfat_capture.json`) is
**session-scoped**; the figures below are what is recorded.
Build under test: `maturin develop --release` at HEAD `1a8acdf`, rebuilt
2026-09-10 19:00:17; the bound-snap patch is in that binary (regression test
passes).

## 1. Where the time goes

| quantity | value | tag |
| --- | --- | --- |
| spectra fitted per run | 636,300 (525 calls x 1212 rows) | measured |
| time inside the Rust batches | 226.7 ms; 0.35 us/row at pipeline block sizes | measured |
| solver share of the four timed GFAT phases | 177.9 ms / 19.6946 s = **0.90 %** | measured |
| GFAT share of logged phase timers | ~26 % (19.69 s of 75.5 s) | measured |
| fit configuration | 5-sample window (`npix=2`), `dv` 25.8-40.7 km/s, noise 1.0: 3 parameters from 5 points | measured |
| non-solver hot spot | `np.take` staging = 21.1 s of the 24.7 s of GFAT calls (86 %) | measured |

Caller-side work (the actual end-to-end lever) is in
`muse/docs/fastfit-staging-performance.md` (M1-M4).

## 2. Rust-side work

### R1 Measured head-to-head against the C++ batcher
`ab_fastfit_cpp_vs_rust.py`, real spectra, 42,420 rows per line, min of 3,
**matched thread counts**, identical inputs to both:

| threads | line | rust, 1 call | rust, 35 calls (pipeline pattern) | C++ | C++ faster by |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 5.29 us/row | 5.37 | **3.34** | 1.59x / 1.61x |
| 1 | 1 | 3.24 | 3.26 | **2.09** | 1.55x / 1.56x |
| 1 | 2 | 3.95 | 3.97 | **2.47** | 1.60x / 1.61x |
| 8 | 0 / 1 / 2 | 0.65 / 0.40 / 0.47 | 0.68 / 0.42 / 0.50 | **0.45 / 0.28 / 0.32** | 1.42-1.54x |
| 32 | 0 / 1 / 2 | 0.24 / 0.16 / 0.18 | 0.31 / 0.19 / 0.21 | **0.17 / 0.12 / 0.13** | 1.27-1.77x |

Both scale ~20x from 1 to 32 threads; agreement is 0 flag mismatches over 127,260
rows with velocities within 0.41 km/s — same work, and **the C++ implementation
is consistently ~1.3-1.8x faster per row (≈1.5x typical)**. Numbers drift a few
percent run to run (machine load); the earlier "parity" reading was a
dispatch-noise artifact of 1212-row batches.
Rerun 2026-09-11 after the workspace patch, HEAD and patched side by side: section 5 of
[performance-plan.md](performance-plan.md). The one-thread gap is now 1.24-1.33x.

### R2 Close the per-row gap [highest-value solver item]
Where the ~1.5x goes: **measured**, in [performance-plan.md](performance-plan.md), which
supersedes the code-level inference that used to sit here. A fit costs about 1.4 us fixed plus
0.08 us per sample, so at the pipeline's 5-sample window three quarters of the time is per-fit
setup and allocation; the f64 arithmetic is roughly 0.4 us. The earlier attribution of the gap to
the C++ build's f32 precision was wrong, and an f32 LM path is off the table (section 3 there).
Steps 1-2 of that plan (reusable Jacobian columns, a per-thread solver workspace; landed
2026-09-11) removed every heap allocation from a fit and cut the intercept from 1.78 to 1.43 us
without changing a bit of output; the measurements and what is left are recorded there.
- **Evaluation cap:** rmpfit's default `max_fev = 200*(nfree+1)` = 800 for 3
  parameters is left in place (`MPConfig { ..MPConfig::new() }`,
  `src/gaussian.rs`) where C has no cap — a silent status divergence
  (see B3).

### R3 Expose per-row cost metadata [unblocks R2]
C exposes `niter`/`nfev`/`c_time`; rust returns only the 8-column row + window.
Add opt-in `return_meta=True` with `nfev`, `niter`, and the convergence reason per
row; assert `nfev >= 1` and stable values on the parity corpus. Without it, R2
cannot be attributed beyond the code-level inference above.
Done 2026-09-11 as `meta=True` (Step 4 of [performance-plan.md](performance-plan.md)).
Result there: Rust accepts 3-5 % more steps and evaluates 1-5 % fewer times than C++, so the
remaining gap is cost per evaluation, not iteration count.

### R4 Accept the slit axis / strided views in the batch entry point
Today the caller must slice per slit, which creates the 86 % staging cost on the
muse side (`fitting_block.py:465-480`). Accepting a `(n_slit, n_wave)` Doppler
grid plus a per-row slit index (as the C++ prototype's `fit_spectra_batch`
already does) removes that staging inside the kernel.
Done 2026-09-11: `fit_spectra_batch_slits` (Step 3 of
[performance-plan.md](performance-plan.md)); the muse-side switch is a separate PR.

### R5 Specialise the 5-point fit [deferred, speculative]
5 samples x 3 parameters: fixed-size unrolled/SIMD model and derivative loops,
fused error computation, reused workspace. State the configuration alongside any
fits/s figure — the prior review's 290-293k fits/s single-thread and this
document's 183k were measured on different data and window sizes.

### R6 API naming note
The raw pyo3 functions take positional `sg_xpixels`; the public wrapper
`python/gaussfit_rs/fitting.py` is keyword-only and renames it to `n_pixels`
(error strings say `n_pixels`). Document the mapping once.

## 3. Better (correctness/robustness)

### B1 The honest "rust is better" headline
Recorded in this repo's prior review sessions (transcripts under
`~/.claude/projects/-home-nabil-Git-gaussfit-rs/`), identical inputs, four
solvers, 20 seeds, windows stuck relative to the best of the four:

| solver | windows stuck | worst chi2 ratio |
| --- | --- | --- |
| C f32 | 13 | 2739x |
| C f64 | 8 | 24x |
| rmpfit (stock) | 13 | 74x |
| **rmpfit + bound-snap fix (current rust)** | **0** | **1.0x** |

Plus: the C extension **does not return** on NaN noise or a NaN guide velocity in
the fit window, or on `muse` seed 27 (killed after 30-60 s); rust drops the pixel.
The patch itself is essentially free (293k vs 290k fits/s single-thread). Use
this in the README, not "18 of 14,197 windows improve".

### B2 Status semantics: `MaxIter` and the evaluation cap
The C status gate accepts `MaxIter` (reports success) while rust flags it, and
rmpfit's 800-evaluation cap (R2) can trigger where C is uncapped. Both are silent
divergences visible in `FLAGS`; document them (the parity corpus peaks near 202
evaluations, so neither binds there).

### B3 Error fields: agreement, and an absolute near-singular guard — no regression risk
Same estimator in both backends (`sqrt(cofactor/det(J^T J)) * sqrt(chi2/dof)`,
`mpfit.c:2611` vs `src/gaussian.rs:251-299`) with the same `dof = n - 3` clamp.
Over 631,897 real pixels they agree within 1e-4 for 91.7 % and within 1e-3 for
98.3 %; every difference is at the float32 noise floor except one case: on 8,790
pixels (1.39 %, 8,753 of them in `mom_guide`) rust reports exactly 0 where C
reports 1e-10..3e-5. Those are **trivial fits, not degenerate ones** — reduced
chi2 ~1e-13, amplitude ~5e-5 ("nothing there" spectra fitted exactly) — and both
backends return FLAG_SUCCESS with finite parameters there. Both values are zero
to any consumer, so rust's 0 carries **no regression risk**, and the earlier
"rust is worse / match C's fallback" framing in this document was wrong.
Mechanism [measured]: the error pass normalises the window and its noise by the
peak (`spectrum.rs:140-141`), so the determinant scales as `peak^6`; computing
`det(J^T J)` at the fitted parameters for the reproduced case gives 6.495e-33 at
peak 5.7e-5 and 6.495e-27 at 5.7e-4, against the **absolute** guard
`|det| < 1e-30 -> [0, 0, 0]` (`gaussian.rs:281`). The threshold crossing lands
exactly where rust's zeros start (0 at 5.7e-5, non-zero from 5.7e-4 upward), so
the guard is confirmed rather than inferred. C's fallback fires at a different
scale (it returns 0 at peak 5.7), so **neither implementation is scale-invariant**.
Improvement worth considering (new, beyond the original plan): make the guard
relative — compare `det` against the product of the diagonal terms — so
tiny-amplitude fits are treated like any other. That is a behaviour change needing
the parity corpus and a decision, but it would avoid guard-induced zero uncertainties
on these fits. Exact fits can still report zeros because errors scale with residuals.

### B4 `lmpar` `max(paru)` deviation — **patched** 2026-09-11
Fix applied: `third_party/rmpfit/src/lib.rs` now clamps with
`self.par = self.par.min(paru)`, matching the C reference's
`*par = mp_dmin1(*par, paru)` (`third_party/c_reference/vendor/cmpfit-1.5/mpfit.c:2102`);
the other lines of the `lmpar` sequence already matched. Carried as
`third_party/rmpfit/lmpar-clamp.patch`, documented in `VENDORED.md` (vendored
sha256 updated; reverse-applying both patches reproduces the recorded upstream
sha256), with its own `docs/design-notes.rst` section.
Parity gate: the one row this moves off the C reference (`muse` seed 44,
`low_snr`, chi2 0.804 vs 0.755 = 1.065x) is a **named exception** in
`ACCEPTED_WORSE` (`python/gaussfit_rs/tests/test_c_parity.py`), capped at 1.10x
and asserted to be taken exactly once so it cannot go stale. Tests: `cargo test`
14 passed; python suite 153 passed; live C parity 117 passed / 1 skipped
(the known C hang on seed 27).
Effect on a real run [measured], pre-patch vs post-patch with identical
deterministic inputs: input spectra, masks' provenance, `spec_*`, `scaling`,
`outliers` bit-identical; `mom_gt` (clean) 21-63 of 127,260 pixels differ at
1e-7..1e-5; noisy/inverted moments 4-7 % of pixels differ, but the movers are the
low-amplitude, high-uncertainty ones — in `mom_gfat` the 565 pixels shifting by
>0.01 km/s have median amplitude 5.4 (vs 18.9 overall) and median
`error_velocity` 77 km/s, and exactly **1** pixel shifts by >1 km/s against a
97 km/s formal error; in `mom_inv`, 684 pixels shift by >0.01 km/s with median
amplitude 1.06 and median error 137 km/s, none by >1 km/s. Logfile area
statistics differ in 89 of 472 keys at 1e-5..1e-4 relative.
Direction and contract check [measured] (per-pixel, pre-patch vs post-patch `.so`, same
deterministic inputs, against the C backend captured on the same five calls). **Pipeline-scale, not
corpus-scale**: the gate's chi2 criterion — `rust > C * (1 + RTOL) + CHI_ATOL`, i.e. 0.5 % — is
violated by 0 of 127,260 pixels on `mom_gt`, `mom_guide`, `mom_inv` both before and after; by the
same single pixel on `mom_gt_noise` both before and after; and by **1 pixel before, 0 after on
`mom_gfat`** (the patch fixes the only violation on the real data rather than introducing one).
The clean GT case is bit-identical (0 pixels changed). On the noisy/inverted cases the changes are
**roughly chi2-neutral**: e.g. `mom_gfat` 515 improved / 494 worse, `mom_inv` 461/531,
`mom_gt_noise` 382/307, and the largest absolute regressions are +1.06 on a pixel whose chi2 is
7.7e5 and +0.75 on one whose chi2 is 8.5e5 (both <1e-5 relative; the largest relative regression
anywhere is 4e-4). About two thirds of the changed pixels move *closer* to C
(434/689, 669/992, 697/1009) and the median |rust−C| velocity distance falls
(3.34e-06 -> 1.91e-06, 2.86e-06 -> 1.91e-06, 3.81e-06 -> 2.62e-06), as expected since C clamps with
`min`; the remaining third are sub-0.1 % moves in the other direction on pixels whose formal errors
are hundreds of km/s. The "68 of 82,368 corpus fits move onto C's values" figure above is the
100-seed corpus measurement, not this pipeline data.
Still worth doing: report the deviation upstream (it exists in released rmpfit).

### B5 Unconstrained fits — **opt-in quality bits implemented** 2026-09-11
Both backends can return FLAG_SUCCESS for parameters the data do not constrain, and the pipeline
propagates these fits into the moments and area statistics. `quality=True` on the three spectrum
entry points now appends a ninth column with separate bits for errors exceeding parameter spans,
zero errors, and parameters at bounds. Zero errors can also come from exact fits, and pegging can
be legitimate saturation; consumers must choose how to handle each bit.

The default eight-column output is preserved. See [the user guide](userguide.rst) for usage and
"Opt-In Unconstrained-Fit Indicator" in [the design notes](design-notes.rst) for the criteria,
measurements, limitations, and test coverage.

Follow-up (muse side): decide whether `sdc_benchmark` should consume the bits. If it does, the C
path needs the same column (`muse/fastfit/fitting_block.py` preallocates `(N, 8)` for C results)
and the flag must be threaded through the wrapper's `quality` argument.


## 4. The carlos_dev C++ prototype: faster per row, still not worth adopting

`origin/carlos_dev` (`muse/fastfit2/`, tip `a5d01f5`, 9 ahead / 91 behind main)
builds cleanly on the current toolchain (C++17 + OpenMP, cmpfit-1.5 sources;
`-DUSE_SIMD` consumed nowhere) and was benchmarked in R1. Caveat: identical
inputs to both backends with pipeline-like parameters — not a bit-faithful
reproduction of a pipeline run (the run's sigma medians are 32-59 km/s vs 19-39
km/s here, because guides and `width_guess` differ).

Verdict [measured]: **CPP is ~1.3-1.8x faster per row at scale (≈1.5x typical) and
numerically equivalent at float32 level — and that is irrelevant end-to-end,
because the solver is 0.90 % of GFAT.** Do not adopt it: 2 months stale, unwired
(`fastfit2/__init__.py` comments the batch import out), f32-only error pass. Take
R2's precision/allocation findings and R5's ideas, plus its Python layer's
`extract/kernel/compose` timers (muse M4).

## 5. Is rust at its limit?

- **Against the C++ implementation: no.** ~1.5x per-row deficit measured (R1),
  attributable to the deliberate f64 LM and per-eval allocation (R2).
- **For pipeline impact: yes.** The solve is 0.90 % of GFAT, so even a 1.5x
  kernel win moves a full run by ~0.3 %. 86 % of GFAT is caller-side staging (M1).
- **Quality: ahead, with one open improvement.** 0 stuck windows vs C f32's 13
  (worst 2739x) and no hangs on NaN inputs. The only place rust and C differ
  elsewhere is an absolute `|det| < 1e-30` guard that zeroes the error field on
  tiny-amplitude fits (B3): no regression risk, but making the guard relative
  would avoid guard-induced zero uncertainties. Opt-in quality bits now report
  large or zero errors and pegged parameters (B5); pipeline adoption remains open.
