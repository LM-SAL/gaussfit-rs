"""Measure public batch-fitting throughput with deterministic noisy spectra.

Run once with ``RAYON_NUM_THREADS=1`` and once without it to compare
single-thread and default multicore performance.
"""

from __future__ import annotations

import argparse
import os
from time import perf_counter

import numpy as np

from gaussfit_rs import FLAG_SUCCESS, fit_spectra_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", type=int, default=100_000)
    parser.add_argument("--pixels", type=int, default=60)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(123)
    velocity = np.linspace(-300.0, 300.0, args.pixels, dtype=np.float32)
    amplitudes = rng.uniform(0.2, 2.0, (args.spectra, 1)).astype(np.float32)
    means = rng.uniform(-80.0, 80.0, (args.spectra, 1)).astype(np.float32)
    sigmas = rng.uniform(10.0, 60.0, (args.spectra, 1)).astype(np.float32)
    spectra = np.ascontiguousarray(
        amplitudes * np.exp(-0.5 * ((velocity[None, :] - means) / sigmas) ** 2)
        + np.float32(0.05)
        * rng.standard_normal((args.spectra, args.pixels), dtype=np.float32)
    )
    noise = np.full_like(spectra, 0.05)

    def run():
        return fit_spectra_batch(
            spectra=spectra,
            dopp_slit=velocity,
            spec_noise=noise,
            guide_velocity=0.0,
            velocity_range=200.0,
            npix=10,
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
    for _ in range(args.repeat):
        start = perf_counter()
        fits, _ = run()
        durations.append(perf_counter() - start)

    if not np.all(fits[:, 7] == FLAG_SUCCESS):
        raise RuntimeError("benchmark data did not produce successful fits")

    elapsed = min(durations)
    threads = os.environ.get("RAYON_NUM_THREADS", "default")
    print(
        f"threads={threads} spectra={args.spectra} pixels={args.pixels} "
        f"seconds={elapsed:.6f} fits_per_second={args.spectra / elapsed:.0f}"
    )


if __name__ == "__main__":
    main()
