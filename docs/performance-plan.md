# Performance plan

Status: drafted 2026-09-11 against HEAD `777e4b5` on branch `mygod`; Steps 1-3 landed the same day,
results in section 5. Numbers tagged
**[measured]** / **[inferred]**. Supersedes the speed analysis in R2 of
[improvement-todos.md](improvement-todos.md); the correctness items there (B1-B5) are untouched.

## 0. What the measurements say

Single thread, `RAYON_NUM_THREADS=1`, muse env, current `--release` build, synthetic spectra with
noise 0.02 and 100 % success. The fit window is `2 * npix + 1` samples; the pipeline runs `npix=2`.

| window | us / fit | note |
| --- | --- | --- |
| 5 samples | 1.78 | pipeline configuration |
| 11 samples | 2.29 | |
| 21 samples | 3.09 | `benchmarks/throughput.py` default: 286 k fits/s |
| 61 samples | 6.43 | |

**[measured]** A straight line through those points is **1.4 us fixed + 0.083 us per sample**. At
the pipeline's 5-sample window the fixed part is 75 % of the fit. Precision is on the slope, not the
intercept: the whole 5-sample arithmetic costs about 0.4 us. So the doc's R2 attribution of the
~1.5x gap to the C++ f32 build's precision is wrong; the gap is per-fit overhead.

**[measured]** Allocator experiments on the 5-sample case:

| allocator | us / fit |
| --- | --- |
| glibc, default (tcache on) | 1.78 |
| glibc with `GLIBC_TUNABLES=glibc.malloc.tcache_count=0` | 2.70 |
| `LD_PRELOAD=/usr/lib64/libtbbmalloc_proxy.so.2` | 2.14 |

Taking the malloc fast path away costs 0.9 us per fit, which only happens when a fit makes dozens
of small allocations. A faster allocator does not exist here: glibc's tcache already wins. The lever
is to stop allocating, not to allocate faster.

**[inferred from code]** Allocation inventory per fit, all inside the vendored
`third_party/rmpfit/src/lib.rs`:

| site | allocations | when |
| --- | --- | --- |
| `MPFit::new` (line 498) | 8 `vec![0.; ..]` buffers | once per fit |
| `parse_params` (line 907) | 11 `Vec`s grown by `push` | once per fit |
| `init_lm` (line 969) | `x`, `qtf`, `fjac` | once per fit |
| `fdjac2` (line 631) | outer `Vec<Option<Vec>>` + one column per free parameter = 4 | **every Jacobian evaluation** |
| `terminate` (line 1103) | `covar`, `xerror` | once per fit |

That is roughly 24 + 4 x iterations per fit. gaussfit-rs discards `resid`, `covar`, and `xerror`
from `MPStatus` and computes its own full-Hessian errors in `scipy_style_errors`
(`src/gaussian.rs:251`).

Expected ceiling for removing all of it **[inferred]**: 0.5-0.9 us per fit, i.e. 1.78 -> ~1.0-1.2 us
at 5 samples. That is the size of the measured C++ gap in R1.

## 1. Steps

Ordered by value per line changed. Each step is independently shippable and gated by the
protocol in section 2.

### Step 1: reuse the Jacobian column buffers across iterations

**What.** Third vendored patch to rmpfit. `fdjac2` builds a fresh `Vec<Option<Vec<f64>>>` and one
`vec![0.0; m]` per free parameter every time it is called (line 631-635). Move that container onto
`MPFit` as a field, allocate it once in `init_lm` after `nfree` is known, and hand `&mut self.derivs`
to `MPFitter::jacobian` each iteration.

**Where.** `third_party/rmpfit/src/lib.rs` only: struct field, `new`, `init_lm`, `fdjac2`. The
`MPFitter::jacobian` signature (`derivs: &mut [Option<Vec<f64>>]`) does not change, so
`src/gaussian.rs:91` is untouched.

**Size.** About ten lines. No numerics change: same values written to `fjac` in the same order, so
the C parity sweep and the corpus must be bit-identical, not merely within tolerance.

**Bookkeeping.** New `third_party/rmpfit/jacobian-workspace.patch`, sha256 and section in
`third_party/rmpfit/VENDORED.md`, following the two existing patches.

**Removes.** 4 allocations per iteration.

### Step 2: caller-owned workspace for the per-fit buffers

**What.** Add `MPWorkspace` to rmpfit holding every `Vec` that `MPFit` currently allocates in `new`,
`parse_params`, `init_lm`, and the Step 1 `derivs`. Add
`MPFitter::mpfit_with_workspace(&mut self, xall: &mut [f64], ws: &mut MPWorkspace)`; the existing
`mpfit` becomes a one-line wrapper that creates a fresh workspace, so upstream behaviour and the
crate's own tests are unchanged. Inside `MPFit`, `vec![..]` assignments become `clear()` +
`resize()`, and the `push` loops in `parse_params` are preceded by `clear()`.

`terminate` moves `fvec` out as `resid` and allocates `covar` and `xerror`. For the workspace entry
point leave those three fields as empty `Vec`s (no allocation) and document that the covariance
stays in the workspace's `fjac`. gaussfit-rs ignores all three today.

**Where.** `third_party/rmpfit/src/lib.rs` for the workspace and entry point.
`src/gaussian.rs:206-214`: a `thread_local!` `RefCell<MPWorkspace>` and the call switched to
`mpfit_with_workspace`. Rayon workers each get their own, so no contention. The single-fit entry
points get the same benefit for free.

**Size.** Around sixty lines in rmpfit, ten in gaussfit. Same numerics, same bit-identical gate.

**Removes.** The remaining ~24 allocations per fit. After Steps 1 and 2 a fit makes zero heap
allocations at any window size up to the existing 512-sample stack limit in `spectrum.rs`.

**Decision point.** Measure after Step 1. If Step 1 alone already lands within noise of the C++
figure, Step 2 is optional and its patch-maintenance cost can be weighed against a fraction of a
microsecond.

### Step 3: slit-indexed batch entry point (the only end-to-end lever)

**Why first-class.** The solver is 0.9 % of GFAT; the per-slit `np.take` staging in
`muse/fastfit/fitting_block.py:467-472` is 86 % [measured, improvement-todos.md section 1]. Steps 1
and 2 move a full run by well under 1 %. This step lets the caller delete that staging.

**What.** New `pyfunction` `fit_spectra_batch_slits` in `src/api.rs`, a sibling of
`fit_spectra_batch_guided` (line 320) with two differences:

- `dopp_slit` is `(n_slit, n_pixels)` C-contiguous `float32` instead of 1-D.
- a new `slit_index: (n_rows,) int32` argument; row `i` fits against `dopp_slit[slit_index[i]]`.

Validation: `slit_index` length matches rows, every value in `[0, n_slit)`, `n_pixels >= sg_xpixels`.
Body: identical parallel loop; `dopp_window` (line 393) becomes a per-row slice
`&dopp_data[slit * n_pixels .. slit * n_pixels + sg_xpixels]`. Factor the loop body shared with the
guided function into one helper taking a `Fn(usize) -> &[f32]` for the Doppler row so the two
entry points do not diverge.

Python: `fit_spectra_batch_slits` in `python/gaussfit_rs/fitting.py` next to line 334, keyword-only,
same `n_pixels` naming and `ascontiguousarray` coercion as the others; export from `__init__.py`.

**How muse uses it.** With `flux` shaped `(..., n_wave)` and C-contiguous, and `slit` somewhere in
the leading axes, the whole block becomes one call with no copies:

```python
rows = flux.reshape(-1, SG_xpixels)                     # view
noise_rows = spec_noise.reshape(-1, SG_xpixels)         # view
slit_index = np.indices(row_shape_with_slit)[slit_axis].ravel().astype(np.int32)
fit, idx = fit_spectra_batch_slits(rows, sg_dopp, noise_rows, guides.ravel(), slit_index, ...)
gfit_pool = fit.reshape(*flux.shape[:-1], 8)            # no moveaxis, no per-slit assignment
```

The "no slit axis in the spectra" branch (every spectrum fit once per slit) keeps its loop; there is
no staging to remove there. The muse change is a separate repo and a separate PR.

**Size.** Thirty to forty lines of Rust, twenty of Python, tests mirroring the guided ones plus one
that checks `fit_spectra_batch_slits` with a constant `slit_index` equals `fit_spectra_batch_guided`
row for row.

### Step 4: opt-in iteration and evaluation counts (only if needed)

**When.** Only if Steps 1 and 2 leave a residual gap to C++ that the scaling test cannot explain.
`MPStatus` already carries `n_iter` and `n_fev` (line 316), so this is cheap to add when it is
needed and dead weight otherwise.

**What.** `meta=False` keyword on the three spectrum entry points returning a separate
`(n_rows, 2) int32` array `[n_iter, n_fev]` (NaN rows get `-1`). Do not append more float columns:
the ninth `quality` column already stretches the row contract. Assert `n_fev >= 1` on success and
stable values across the parity corpus.

## 2. Measurement protocol, every step

1. `benchmarks/throughput.py` at `RAYON_NUM_THREADS=1` and at default threads, min of 5.
2. The window-scaling sweep from section 0 (`npix` in 2, 5, 10, 30 with `pixels = 4 * window + 1`).
   Add an `--npix` flag to `throughput.py` so this is a committed harness rather than a scratch
   script; report the fitted intercept and slope, since the intercept is what Steps 1-2 attack.
3. `cargo test` and `pytest python/gaussfit_rs/tests`.
4. Live C parity (`test_c_parity.py`, muse env). For Steps 1 and 2 the requirement is
   **bit-identical** output on the parity corpus, not within tolerance: no floating-point operation
   changes.
5. When a C++ comparison is wanted, `muse/benchmark/ab_fastfit_cpp_vs_rust.py` with matched thread
   counts, as in R1.

Record the numbers in this file under a dated heading so the next revision starts from evidence.

## 3. Not doing, and why

- **f32 LM path (R2a).** The B1 result (0 stuck windows vs C f32's 13, worst 2739x) depends on the
  f64 solve, and the measured slope says precision costs ~0.4 us of 1.78 at 5 samples. Wrong
  trade in both directions.
- **Specialised 5-point solver (R5).** Highest parity risk in the list. Steps 1 and 2 capture most
  of its upside for a tenth of the code. Revisit only if the post-Step-2 intercept is still
  dominated by something the scaling test attributes to solver bookkeeping.
- **`-C target-cpu=native`.** Five-sample loops do not vectorise usefully and it breaks wheel
  portability. Anyone who wants it locally can set `RUSTFLAGS` at build time.
- **Skipping rmpfit's exit-path `covar()`.** A few dozen flops on a 3x3; its two allocations fold
  into Step 2.
- **Alternative global allocator (mimalloc, jemalloc).** Measured: TBB malloc is slower than glibc
  tcache here, and a `#[global_allocator]` adds a C build dependency to the wheel for nothing.
- **Raising or removing the 800-evaluation cap (R2, B2).** A status-semantics question, not a
  speed one; the corpus peaks near 202 evaluations so it does not bind.

## 4. Doc follow-up

When Step 1 lands, rewrite the "where the ~1.5x goes" bullets in improvement-todos.md R2 to point
here: the gap is allocation and per-fit setup, not f64 versus f32.

## 5. Results, 2026-09-11

Steps 1-3 landed on `mygod`; Step 4 was not needed. Same harness as section 0 through the committed
`--npix` sweep (`pixels = max(60, 4 * window + 1)`, noise 0.05, 100 k spectra, min of 5),
interleaved three times against a wheel of the pre-change HEAD on the same idle 32-core machine;
the spread across rounds is under 0.02 us on every number below.

| | HEAD `7a124b5` | after Steps 1-2 | change |
| --- | --- | --- | --- |
| fitted intercept, 1 thread | 1.71-1.73 us | 1.47-1.49 us | **-0.24 us (-14 %)** |
| fitted slope, 1 thread | 0.081 us/sample | 0.086 us/sample | +0.005, at the noise floor |
| 5-sample window, 1 thread | 2.26-2.28 us/fit | 2.05-2.07 us/fit | -10 % |
| 5-sample window, 32 threads | 0.119 us/fit | 0.109 us/fit | -8 % |
| 21-sample window, 32 threads | 0.188 us/fit | 0.183 us/fit | -3 % |
| heap allocations per fit (bare solver, counting allocator) | 28, plus 4 per Jacobian after the first | 0 | |

Per step, at the gates of section 2:

- **Step 1** (derivative columns reused across Jacobians): allocations 28 + 4 per iteration -> 28;
  time within noise of HEAD (intercept 1.77 vs 1.78 us on single runs). Corpus bit-identical.
- **Step 2** (`MPWorkspace`, `MPFitter::mpfit_with_workspace`, one `thread_local!` workspace in
  `src/gaussian.rs`): allocations 28 -> 0 and the table above. Corpus bit-identical: 92 corpora,
  38,083 rows of `fit_spectra_batch_guided(quality=True)` output over 46 seeds of both parameter
  sets, compared with `np.array_equal`. Live C parity 117 passed, 1 skipped (the C hang seed).
  Steps 1 and 2 ship as one vendored patch, `third_party/rmpfit/workspace.patch`.
- **Step 3** (`fit_spectra_batch_slits`): shipped with tests, including constant `slit_index`
  equals `fit_spectra_batch_guided` row for row. The muse-side switch is a separate PR and is
  where the end-to-end gain is.
- **Step 4** (`meta=True` on the four spectrum entry points, `[n_iter, n_fev]` per row): done,
  because the C++ A/B below left a 1.24-1.33x gap at one thread that the synthetic scaling line
  does not explain. The counts settle it (see "Iteration counts" below): Rust does not iterate
  more than C++ in any way that matters, so the gap is per-iteration cost.

What the measurements corrected in this plan:

- The section 0 ceiling of 0.5-0.9 us for removing every allocation was too high. With tcache on, a
  malloc/free pair costs about 10 ns, so 28 of them are roughly 0.3 us; the measured gain is
  0.24 us. The 0.9 us tcache-off figure measured slow allocation, not the cost of fast allocation.
- Step 1 alone was worth nothing measurable: fits here converge in a few iterations, so the
  per-Jacobian columns were a small share of the allocations.
- The remaining 1.47 us intercept is not allocation. Untested candidates, cheapest first: the three
  `[0.0f32; 512]` stack buffers `src/spectrum.rs` zeroes for every fit (6 KB of memset, estimated
  0.1-0.2 us; a smaller `MAX_STACK_WINDOW` or uninitialised buffers would remove it), then the 5x3
  `qrfac` / `lmpar` / `qrsolv` bookkeeping, which is R5 territory.

### C++ A/B, 2026-09-11

`muse/benchmark/ab_fastfit_cpp_vs_rust.py` on the `baseline_gpu` run output
(`spectrum_summed_all.zarr`, 42,420 rows per line, 5-sample window, min of 3), matched thread
counts, the `rust 1 call` column, all in us/row; the baseline is the same HEAD wheel as above.
Agreement with C++ is unchanged by the patch: 0 flag mismatches on every line, max velocity
difference 0.19 km/s on line 0 and 0.02 km/s on lines 1-2. The 32-thread rows time 6-11 ms runs
and move by 10-20 % between invocations (C++ line 2 read 0.186 then 0.240), so read them as
"parity or better", not as ratios.

| threads | line | HEAD | patched | C++ | C++ faster by, HEAD -> patched |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 4.62 | 4.27 | 3.46-3.49 | 1.33x -> 1.24x |
| 1 | 1 | 2.92 | 2.73 | 2.04-2.05 | 1.43x -> 1.33x |
| 1 | 2 | 3.41 | 3.19 | 2.41-2.46 | 1.39x -> 1.32x |
| 8 | 0 | 0.617 | 0.563 | 0.459-0.474 | 1.30x -> 1.23x |
| 8 | 1 | 0.393 | 0.363 | 0.283-0.297 | 1.32x -> 1.28x |
| 8 | 2 | 0.459 | 0.433 | 0.325-0.328 | 1.40x -> 1.33x |
| 32 | 0 | 0.245 | 0.217 | 0.257-0.265 | Rust faster, 1.05x -> 1.22x |
| 32 | 1 | 0.167 | 0.150 | 0.235-0.238 | Rust faster, 1.42x -> 1.57x |
| 32 | 2 | 0.184 | 0.178 | 0.186-0.240 | Rust faster, 1.01x -> 1.35x |

The patch saves 0.19-0.35 us per row at one thread on real data, in line with the 0.24 us
synthetic figure, and the remaining gap to C++ is 0.7-0.8 us per row. Real rows cost 2.7-4.3 us
against 2.05 us for the synthetic 5-sample fit, on both solvers, so the extra is iterations, not
setup. That reorders the candidates in section 1: Step 4 to count iterations and evaluations per
row first, then the per-iteration `qrfac` / `lmpar` / `qrsolv` path (R5), with the stack-buffer
zeroing above a smaller, fixed win.

### Iteration counts, Rust versus C++, 2026-09-11

Identical inputs to both: Rust through `meta=True`, C++ through the `fit_counts_ext` diagnostic in
`third_party/cpp_reference` (the unmodified fastfit2 core with its `mpfit` call shimmed to read
`mp_result.niter` and `nfev`). Rows are those both solvers fitted successfully. rmpfit starts its
evaluation counter at 1 where cmpfit starts at 0, so Rust's `n_fev` is shown minus one.

| input | rows | iterations, mean Rust / C++ | median | evaluations, mean Rust / C++ | rows Rust > C++ / Rust < C++ |
| --- | --- | --- | --- | --- | --- |
| corpus `wide`, 45 seeds | 17,109 | 7.65 / 7.24 | 7 / 7 | 14.5 / 15.1 | 45 % / 6 % |
| corpus `muse`, 45 seeds | 15,263 | 8.13 / 7.76 | 7 / 7 | 15.8 / 16.2 | 41 % / 4 % |
| baseline_gpu line 0 | 1,047 | 10.08 / 9.79 | 9 / 9 | 21.0 / 21.1 | 29 % / 2 % |
| baseline_gpu line 1 | 1,130 | 8.15 / 7.86 | 7 / 7 | 15.4 / 16.2 | 37 % / 8 % |
| baseline_gpu line 2 | 1,158 | 9.29 / 9.03 | 8 / 8 | 17.8 / 18.3 | 32 % / 6 % |

Rust accepts 3-5 % more steps and evaluates the model 1-5 % fewer times: the f64 solve wastes fewer
trial steps, and the f32 solve reaches its tolerances a step or so earlier. Neither moves the row
cost by more than a few percent, so the whole one-thread gap is cost per evaluation. Dividing the
A/B row times by the mean evaluation counts:

| line | Rust, us per evaluation | C++ | ratio |
| --- | --- | --- | --- |
| 0 | 4.27 / 21.0 = 0.203 | 3.46 / 21.1 = 0.164 | 1.24x |
| 1 | 2.73 / 15.4 = 0.178 | 2.05 / 16.2 = 0.127 | 1.40x |
| 2 | 3.19 / 17.8 = 0.179 | 2.41 / 18.3 = 0.132 | 1.36x |

Where a fixed 0.04-0.05 us per evaluation (0.7-0.8 us per row) can come from, none of it measured
yet: the three `[0.0f32; 512]` stack buffers `src/spectrum.rs` zeroes per row (6 KB of memset,
estimated 0.1-0.2 us, and C++ zeroes nothing: its workspace is per-thread vectors); the f64
arithmetic and bounds-checked `Vec` indexing on the port's per-iteration path (`qrfac`, `lmpar`,
`qrsolv`); `fdjac2` copying the derivative columns into `fjac` where cmpfit writes them in place;
and `scipy_style_errors` re-evaluating five exponentials that the C build reads from the final
Jacobian. The first is a one-line experiment with the corpus snapshot and the A/B as gates.

### Where the remaining gap is, 2026-09-11

Method: a native replay harness over the three `baseline_gpu` lines (1,212 rows each, 1,024
pixels, the A/B parameters), single-threaded, core-pinned, best of 20 passes; the same rows through
scratch copies of `src/gaussian.rs`, `src/spectrum.rs`, the vendored rmpfit and the vendored
fastfit2 C++ core with `rdtsc` region timers around the same six solver stages; callgrind on both
with the stage functions made non-inlinable, for exact instruction attribution; and ablation
builds. Nothing in the repo changed for this; the harness lived in the session scratch directory.

**What the C++ number is.** The setuptools build the A/B runs against is not a plain `-O3` build
of the same sources: Python's default `CFLAGS` add `-fno-strict-overflow`, and that one flag makes
GCC's code for the cmpfit helpers much better (per row on line 1: `fdjac2` 170 -> 85 ns, `lmpar`
150 -> 43 ns, the QR block 572 -> 508 ns; whole row 2.33 -> 1.98 us). Every comparison below is
against that fast build, which the Python batch extension reproduces (2.07 us/row on line 1; the
Rust Python batch path costs 0.1 us/row over the native core).

| us/row, native, best of 20 x 3 interleaved rounds | line 1 | line 0 | line 2 |
| --- | --- | --- | --- |
| Rust, current | 2.43-2.46 | 4.01-4.10 | 2.91-2.94 |
| Rust, HEAD before the workspace patch | 2.92-2.96 | 4.54-4.76 | 3.41-3.52 |
| Rust, `-C target-cpu=x86-64-v3` | 2.35-2.39 | 3.95-3.99 | 2.84 |
| C++ f32, plain `-O3` | 2.39-2.41 | 3.73-3.76 | 2.73-2.75 |
| C++ f64, plain `-O3`, same code | 2.33-2.34 | 3.81-3.83 | 2.74-2.75 |
| C++ f32, setuptools flags (the A/B build) | 2.07 | 3.19-3.35 | 2.24-2.39 |

The gap to the shipped C++ is 0.36 us on line 1 and 0.55-0.8 us on lines 0 and 2, and it is all
inside the solver call: per row on line 1 Rust spends about 2.0 us in `mpfit_with_workspace`
against 1.48 us for the fast cmpfit, while the peak search (240-360 ns on both sides), the window
build (25 ns vs 1 ns) and Rust's error pass (19 ns) are equal or negligible.

**What it is not**, each measured:

- Not precision: the identical C++ code in double is within 2 % of float, and the same fits in
  Rust with a float `exp` are not faster.
- Not `exp`: 20 M instructions for `exp` against 12 M for `expf` over the run, no time difference.
- Not allocation, and not the workspace layout: the workspace build is 0.5 us per row faster than
  HEAD in this harness, in every stage, not only the allocating ones.
- Not the peak search: equal on both sides; a bounds-check-free zipped rewrite is slower (LLVM
  emits worse code for it), and a two-pass version meant to vectorise is slower too.
- Not the ISA: `x86-64-v3` buys 2-3 %, `native` 1 %.
- Not the model evaluation in `gaussian.rs`: zipped loops with the derivative columns hoisted out
  of their `Option`s gain 1-1.5 %.
- Not the stack-buffer zeroing guessed at above: 25 ns per row, measured.

**What it is.** Instruction counts per row (callgrind, line 1, stages non-inlined): Rust 57.4 k,
C++ 45.3 k. The excess is the port's per-iteration bookkeeping: `iterate` 9.1 k, `qrfac` 7.8 k,
`transpose` 3.6 k, `gnorm` 1.4 k, `check_limits` 1.2 k, `check_is_finite` 1.0 k, `fdjac2` 2.7 k,
against cmpfit's `mpfit` body 10.6 k, `mp_qrfac` 5.9 k, `mp_lmpar` 1.9 k, `mp_fdjac2` 0.9 k. Per
line, the arithmetic is the minority: `transpose` executes 1.0 k instructions on its own source
lines and 2.6 k in the `Vec` and slice indexing inlined behind them; its hot lines are plain
`self.wa4[i] += self.fjac[ij] * temp` forms, each a `&mut Vec` header load, a bounds check and
the multiply-add. cmpfit runs the same loops on raw pointers.

**Next lever, if the 20 % matters.** It is inside the vendored solver: take slices once per stage
(`let fjac = &mut self.fjac[..]`, or index-free iterator forms) in `qrfac`, `transpose`,
`iterate`, `lmpar` and `qrsolv`, which removes the header reloads and lets LLVM elide the checks.
That is a larger patch to `third_party/rmpfit`, and the bit-identical gate applies.

### Slice-indexed stages and compile flags, 2026-09-11

The lever named above was tried and dropped: a patch binding each stage's buffers to local slices
and indexing those (`qrfac`, `transpose`, `iterate`, `lmpar`, `qrsolv` and the small stages) was
bit-identical and measured as below, but it is an 864-line diff to the vendored file for 3 %, so
it was not kept. Same harness as the previous subsection, best of 20 x 3 interleaved rounds,
core-pinned.

| us/row, native | line 1 | line 0 | line 2 |
| --- | --- | --- | --- |
| Rust, workspace patch | 2.47-2.48 | 4.02-4.08 | 2.91-2.95 |
| Rust, + stage-slices | 2.40-2.41 | 3.98-4.03 | 2.85 |
| Rust, + stage-slices, `-C target-cpu=x86-64-v3` | 2.29 | 3.84 | 2.75-2.77 |
| C++, setuptools build | 2.07-2.09 | 3.17-3.36 | 2.39 |

Through Python at one thread it moved the ratio to the C++ batch extension from 1.24x / 1.33x /
1.32x to 1.22x / 1.25x / 1.30x on lines 1 / 0 / 2. Removing the header reloads was worth 3 %; the rest
of the port's excess is the bounds checks and LLVM's loop codegen, and it is only reachable by
rewriting the hot loops in iterator form or with unchecked indexing inside the vendored file.

Compile flags, each measured in the harness: `panic = "abort"` is within noise (1 %);
`-C target-cpu=x86-64-v3` is 4 %, `native` no better than v3; `lto`, `codegen-units = 1` and
`opt-level = 3` were already set. A v3 build does not run on pre-AVX2 CPUs, and a wheel cannot be
tagged for a micro-architecture level on PyPI, so v3 stays an opt-in `RUSTFLAGS` documented in
`docs/installation.rst`. Shipping the 4 % in one portable wheel would need runtime dispatch (the
hot entry compiled twice under `#[target_feature]`, chosen by `is_x86_feature_detected!`), which
is more machinery than 4 % justifies while the solver is under 1 % of a pipeline run.

**Where this leaves the comparison.** Single-threaded, per row, Rust is level with a plain `-O3`
build of the C++ reference and 15-20 % behind the build Python's CFLAGS produce (a few points
less with a v3 build); at 32 threads Rust is ahead (section 5, C++ A/B). The end-to-end lever is
still Step 3.
