"""
Measure public batch-fitting throughput with deterministic noisy spectra.

Run once with ``RAYON_NUM_THREADS=1`` and once without it to compare single-thread and default
multicore performance. Pass several ``--npix`` values to sweep the fit window (``2 * npix + 1``
samples); the script then fits ``us_per_fit = intercept + slope * window`` and reports both, since
the intercept is the per-fit overhead and the slope the per-sample arithmetic.
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import numpy as np

from gaussfit_rs import FLAG_SUCCESS, fit_spectra_batch


def measure(spectra: int, pixels: int, npix: int, repeat: int) -> float:
    """
    Return the best-of-``repeat`` seconds for one batch of ``spectra`` fits.
    """
    rng = np.random.default_rng(123)
    velocity = np.linspace(-300.0, 300.0, pixels, dtype=np.float32)
    amplitudes = rng.uniform(0.2, 2.0, (spectra, 1)).astype(np.float32)
    means = rng.uniform(-80.0, 80.0, (spectra, 1)).astype(np.float32)
    sigmas = rng.uniform(10.0, 60.0, (spectra, 1)).astype(np.float32)
    data = np.ascontiguousarray(
        amplitudes * np.exp(-0.5 * ((velocity[None, :] - means) / sigmas) ** 2)
        + np.float32(0.05) * rng.standard_normal((spectra, pixels), dtype=np.float32)
    )
    noise = np.full_like(data, 0.05)

    def run():
        return fit_spectra_batch(
            spectra=data,
            dopp_slit=velocity,
            spec_noise=noise,
            guide_velocity=0.0,
            velocity_range=200.0,
            npix=npix,
            npix_slack=2,
            dv=12.0,
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

    if not np.all(fits[:, 7] == FLAG_SUCCESS):
        raise RuntimeError("benchmark data did not produce successful fits")
    return min(durations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", type=int, default=100_000)
    parser.add_argument("--pixels", type=int, default=60)
    parser.add_argument("--npix", type=int, nargs="+", default=[10])
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    threads = os.environ.get("RAYON_NUM_THREADS", "default")
    windows, costs = [], []
    for npix in args.npix:
        window = 2 * npix + 1
        # Keep the window well inside the grid so every fit sees the full window.
        pixels = max(args.pixels, 4 * window + 1)
        elapsed = measure(args.spectra, pixels, npix, args.repeat)
        windows.append(window)
        costs.append(1e6 * elapsed / args.spectra)
        print(
            f"threads={threads} spectra={args.spectra} pixels={pixels} npix={npix} "
            f"window={window} seconds={elapsed:.6f} us_per_fit={costs[-1]:.3f} "
            f"fits_per_second={args.spectra / elapsed:.0f}"
        )
    if len(windows) > 1:
        slope, intercept = np.polyfit(windows, costs, 1)
        print(f"fit: us_per_fit = {intercept:.3f} + {slope:.4f} * window")


if __name__ == "__main__":
    main()
