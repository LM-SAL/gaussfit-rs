"""
Measure public batch-fitting throughput with deterministic noisy spectra.

Run once with ``RAYON_NUM_THREADS=1`` and once without it to compare single-thread and default
multicore performance; give the multicore run ``--spectra 400000`` or more so a batch lasts tens
of milliseconds. Pass several ``--npix`` values to sweep the fit window (``2 * npix + 1``
samples); the script then fits ``us_per_fit = intercept + slope * window`` and reports both, since
the intercept is the per-fit overhead and the slope the per-sample arithmetic. The grid keeps a
``DV`` km/s pixel spacing and grows with the window, so the peak search over the whole spectrum
is part of the intercept. ``src/tests/bench.rs`` generates the same data for a Python-free timing
of the fit alone.
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

# NumPy's OpenBLAS workers busy-wait for ~60 ms after a BLAS call, on the SMT siblings of the
# fitting threads; a short batch started inside that window runs up to 1.75x slower. Pin them.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from gaussfit_rs import FLAG_SUCCESS, fit_spectra_batch

DV = 12.0  # km/s per pixel; passed as dv, as the pipeline passes its grid's median spacing
NOISE = 0.05  # 1-sigma noise per sample, against amplitudes of 0.2 to 2
MIN_SUCCESS = 0.95  # below this the batch is timing failure paths, not fits


def measure(spectra: int, pixels: int, npix: int, repeat: int) -> tuple[float, float]:
    """
    Return best-of-``repeat`` seconds for one batch of ``spectra`` fits, and the converged fraction.
    """
    rng = np.random.default_rng(123)
    velocity = ((np.arange(pixels) - (pixels - 1) / 2) * DV).astype(np.float32)
    amplitudes = rng.uniform(0.2, 2.0, (spectra, 1)).astype(np.float32)
    means = rng.uniform(-80.0, 80.0, (spectra, 1)).astype(np.float32)
    sigmas = rng.uniform(10.0, 60.0, (spectra, 1)).astype(np.float32)
    data = np.ascontiguousarray(
        amplitudes * np.exp(-0.5 * ((velocity[None, :] - means) / sigmas) ** 2)
        + np.float32(NOISE) * rng.standard_normal((spectra, pixels), dtype=np.float32)
    )
    noise = np.full_like(data, NOISE)

    def run():
        return fit_spectra_batch(
            spectra=data,
            dopp_slit=velocity,
            spec_noise=noise,
            guide_velocities=0.0,
            velocity_range=200.0,
            npix=npix,
            npix_slack=2,
            dv=DV,
            width_min=5.0,
            amplitude_rel_min=0.1,
            amplitude_rel_max=2.0,
            width_max=100.0,
            width_guess=30.0,
        )

    run()
    durations = []
    for _ in range(repeat):
        start = perf_counter()
        fits, _ = run()
        durations.append(perf_counter() - start)

    success = float(np.mean(fits[:, 7] == FLAG_SUCCESS))
    if success < MIN_SUCCESS:
        msg = f"only {success:.1%} of the benchmark fits converged; the batch is not measuring fits"
        raise RuntimeError(msg)
    return min(durations), success


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", type=int, default=100_000)
    parser.add_argument("--pixels", type=int, default=60)
    parser.add_argument("--npix", type=int, nargs="+", default=[10])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    threads = os.environ.get("RAYON_NUM_THREADS") or f"default({os.cpu_count()})"
    windows, costs = [], []
    for npix in args.npix:
        window = 2 * npix + 1
        # Keep the window well inside the grid so every fit sees the full window.
        pixels = max(args.pixels, 4 * window + 1)
        elapsed, success = measure(args.spectra, pixels, npix, args.repeat)
        windows.append(window)
        costs.append(1e6 * elapsed / args.spectra)
        print(
            f"threads={threads} spectra={args.spectra} pixels={pixels} npix={npix} "
            f"window={window} seconds={elapsed:.6f} us_per_fit={costs[-1]:.3f} "
            f"fits_per_second={args.spectra / elapsed:.0f} success={success:.4f}"
        )
    if len(windows) > 1:
        slope, intercept = np.polyfit(windows, costs, 1)
        print(f"fit: us_per_fit = {intercept:.3f} + {slope:.4f} * window")


if __name__ == "__main__":
    main()
