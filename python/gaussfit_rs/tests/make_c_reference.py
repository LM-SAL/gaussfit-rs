"""
Record reference fits from MUSE's C extension for the parity and figure tests.

Run in an environment where ``muse`` is installed::

    python -m gaussfit_rs.tests.make_c_reference

One fixture is written per entry of ``PARAMETER_SETS``. The C extension is the
reference implementation; two intentional differences are kept out of the cases:
all-negative windows (C fits them, Rust reports no peak) and inf pixels or zero
noise (the C solver does not return).
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
FIT_KEYS = {
    "velocity_range": float,
    "npix": int,
    "npix_slack": int,
    "width_min": float,
    "amplitude_rel_min": float,
    "amplitude_rel_max": float,
    "width_max": float,
    "width_guess": float,
}
# "wide" exercises loose bounds on a fine grid. "muse" mirrors the Level 2 pipeline:
# 40 km/s pixels, a 9-pixel fit window and amplitude bounds of 0.9 to 1.1.
PARAMETER_SETS = {
    "wide": {
        "n_pixels": 60,
        "dv": 600.0 / 59.0,
        "sigma_range": (8.0, 120.0),
        "velocity_range": 200.0,
        "npix": 5,
        "npix_slack": 2,
        "width_min": 5.0,
        "amplitude_rel_min": 0.1,
        "amplitude_rel_max": 2.0,
        "width_max": 200.0,
        "width_guess": 30.0,
    },
    "muse": {
        "n_pixels": 128,
        "dv": 40.0,
        "sigma_range": (15.0, 120.0),
        "velocity_range": 100.0,
        "npix": 4,
        "npix_slack": 1,
        "width_min": 5.0,
        "amplitude_rel_min": 0.9,
        "amplitude_rel_max": 1.1,
        "width_max": 200.0,
        "width_guess": 20.0,
    },
}


def build_cases(params, seed=7):  # noqa: C901
    """
    Return ``(dopp, spectra, noise, guides, labels)`` for one parameter set.

    Families: clean, white_noise, poisson, low_snr, continuum, blend, wings,
    asymmetric, narrow, broad, flat_top, hot_pixel, masked and degenerate.
    Guides are the true centre, zero, an offset within the search range, or NaN.
    """
    rng = np.random.default_rng(seed)
    n, dv = params["n_pixels"], params["dv"]
    dopp = ((np.arange(n) - (n - 1) / 2) * dv).astype(np.float32)
    sigma_lo, sigma_hi = params["sigma_range"]
    vmax = 0.8 * dopp[-1]
    cases = []

    def gaussian(amp, vel, sigma):
        return amp * np.exp(-0.5 * ((dopp - vel) / sigma) ** 2)

    def draw():
        return 10 ** rng.uniform(-1, 3), rng.uniform(-vmax, vmax), rng.uniform(sigma_lo, sigma_hi)

    def guide_for(vel):
        offset = vel + rng.uniform(-params["velocity_range"], params["velocity_range"])
        return rng.choice([vel, 0.0, offset, np.nan], p=[0.5, 0.25, 0.2, 0.05])

    def add(label, spec, err, guide):
        cases.append((label, np.asarray(spec, np.float32), np.asarray(err, np.float32), guide))

    def add_noisy(label, profile, amp, vel, level=0.02):
        add(label, profile + rng.normal(0, level * amp, n), np.full(n, level * amp), guide_for(vel))

    for _ in range(40):
        amp, vel, sigma = draw()
        add("clean", gaussian(amp, vel, sigma), np.full(n, 0.01 * amp), guide_for(vel))
    for _ in range(60):
        amp, vel, sigma = draw()
        err = np.full(n, rng.choice([0.01, 0.05, 0.2]) * amp)
        if rng.random() < 0.5:
            err = err * rng.uniform(0.5, 2.0, n)
        add("white_noise", gaussian(amp, vel, sigma) + rng.normal(0, err), err, guide_for(vel))
    for _ in range(40):
        _, vel, sigma = draw()
        counts = rng.poisson(gaussian(10 ** rng.uniform(0.7, 3.7), vel, sigma) + 2.0)
        add("poisson", counts, np.sqrt(np.maximum(counts, 1.0)), guide_for(vel))
    for _ in range(40):
        amp, vel, sigma = draw()
        add_noisy("low_snr", gaussian(amp, vel, sigma), amp, vel, level=rng.uniform(0.2, 1.0))
    for _ in range(40):
        amp, vel, sigma = draw()
        slope = 1 + rng.uniform(-0.5, 0.5) * dopp / dopp[-1]
        background = amp * rng.uniform(0.05, 0.5) * slope
        add_noisy("continuum", gaussian(amp, vel, sigma) + background, amp, vel)
    for _ in range(40):
        amp, vel, sigma = draw()
        separation = rng.uniform(1, 3) * sigma * rng.choice([-1, 1])
        companion = gaussian(
            amp * rng.uniform(0.3, 1.5), vel + separation, sigma * rng.uniform(0.7, 1.3)
        )
        add_noisy("blend", gaussian(amp, vel, sigma) + companion, amp, vel)
    for _ in range(20):
        amp, vel, sigma = draw()
        lorentzian = amp * rng.uniform(0.1, 0.4) / (1 + ((dopp - vel) / sigma) ** 2)
        add_noisy("wings", gaussian(amp, vel, sigma) + lorentzian, amp, vel)
    for _ in range(20):
        amp, vel, sigma = draw()
        sigmas = np.where(dopp < vel, sigma, sigma * rng.uniform(1.3, 2.5))
        add_noisy("asymmetric", amp * np.exp(-0.5 * ((dopp - vel) / sigmas) ** 2), amp, vel)
    for _ in range(20):
        amp, vel, _ = draw()
        add_noisy("narrow", gaussian(amp, vel, rng.uniform(0.2, 0.8) * dv), amp, vel)
    for _ in range(20):
        amp, vel, _ = draw()
        add_noisy("broad", gaussian(amp, vel, rng.uniform(2, 6) * params["npix"] * dv), amp, vel)
    for _ in range(20):
        amp, vel, sigma = draw()
        spec = gaussian(amp, vel, sigma) + rng.normal(0, 0.02 * amp, n)
        add(
            "flat_top",
            np.minimum(spec, amp * rng.uniform(0.6, 0.9)),
            np.full(n, 0.02 * amp),
            guide_for(vel),
        )
    for _ in range(20):
        amp, vel, sigma = draw()
        spec = gaussian(amp, vel, sigma) + rng.normal(0, 0.02 * amp, n)
        spec[rng.integers(n)] = amp * rng.uniform(2, 5)
        add("hot_pixel", spec, np.full(n, 0.02 * amp), guide_for(vel))
    for _ in range(30):
        amp, vel, sigma = draw()
        spec = gaussian(amp, vel, sigma) + rng.normal(0, 0.02 * amp, n)
        spec[rng.choice(n, size=rng.integers(1, 6), replace=False)] = np.nan
        add("masked", spec, np.full(n, 0.02 * amp), guide_for(vel))
    add("degenerate", np.zeros(n), np.ones(n), 0.0)
    add("degenerate", np.full(n, np.nan), np.ones(n), 0.0)
    sparse = np.full(n, np.nan)
    sparse[n // 2 : n // 2 + 2] = [5.0, 4.0]
    add("degenerate", sparse, np.ones(n), 0.0)
    edge = dopp[-1] - 0.5 * dv
    add("degenerate", gaussian(5.0, edge, 2 * dv), np.full(n, 0.05), edge)

    labels, spectra, noise, guides = (np.array(column) for column in zip(*cases, strict=True))
    return dopp, spectra, noise, guides.astype(np.float32), labels


def run_c(dopp, spectra, noise, guides, params):
    """
    Fit every spectrum with ``muse``'s C extension.
    """
    from muse.fastfit.fit_single_spectrum_ext import _fit_single_spectrum_c  # noqa: PLC0415

    fit_params = {key: cast(params[key]) for key, cast in FIT_KEYS.items()}
    fits = np.empty((len(spectra), 8), dtype=np.float32)
    indices = np.empty((len(spectra), 2), dtype=np.int32)
    dv = float(np.median(np.gradient(dopp)))
    for i, (spec, err, guide) in enumerate(zip(spectra, noise, guides, strict=True)):
        fits[i], indices[i] = _fit_single_spectrum_c(
            spectrum=spec,
            dopp_slit=dopp,
            spec_noise=err,
            guide_velocity=float(guide),
            dv=dv,
            SG_xpixels=len(dopp),
            **fit_params,
        )
    return fits, indices


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for name, params in PARAMETER_SETS.items():
        dopp, spectra, noise, guides, labels = build_cases(params)
        fits, indices = run_c(dopp, spectra, noise, guides, params)
        out = DATA_DIR / f"c_reference_{name}.npz"
        np.savez_compressed(
            out,
            dopp=dopp,
            spectra=spectra,
            noise=noise,
            guides=guides,
            labels=labels,
            fits=fits,
            indices=indices,
            **{key: params[key] for key in FIT_KEYS},
        )
        flags, counts = np.unique(fits[:, 7], return_counts=True)
        summary = dict(zip(flags.tolist(), counts.tolist(), strict=True))
        print(f"wrote {out.name}: {len(spectra)} spectra, C flags {summary}")  # noqa: T201
