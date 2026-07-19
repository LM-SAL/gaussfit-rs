# Performance TODO

Baseline: 100,000 noisy 60-pixel spectra on a 12-core Apple Silicon machine.

- Current Rust/rmpfit: 0.343 s with 1 thread; 0.0421 s with 12 threads.
- Serial C extension: 0.343 s.
- Previous hand-rolled Rust solver: 0.126 s with 1 thread; 0.0155 s with 12 threads.

## Work

- [x] Use the analytical Gaussian Jacobian already supported by `rmpfit`.
  - Derivatives match central differences within `1e-8`.
  - 10,000 differential fits had no flag mismatches with the C extension.
- [x] Re-run the baseline and inspect the remaining per-fit allocation cost.
  - The Gaussian problem now borrows its inputs instead of allocating three converted vectors.
  - Remaining workspace allocation is inside the verbatim vendored solver; no local fork was added.
- [x] Add a reproducible benchmark for single-thread and multicore throughput.
  - `RAYON_NUM_THREADS=1 python benchmarks/throughput.py`
  - `python benchmarks/throughput.py`
- [x] Replace the inaccurate “bit-comparable” wording with the actual compatibility guarantee.
- [x] Run Rust and Python tests and record the final measurements here.
  - 12 Rust tests and 35 Python tests pass in release mode.

## Current Result

- Analytical Jacobian plus borrowed inputs: 0.294 s with 1 thread; 0.0364 s with default threads.
- Throughput: 340,000 fits/s with 1 thread; 2.75 million fits/s with default threads.
- Improvement over the starting rmpfit baseline: about 14% single-thread and 14% multicore.
