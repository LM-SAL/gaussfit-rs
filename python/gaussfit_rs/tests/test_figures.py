"""
Figure tests: fitted profiles per spectrum family, Rust against the recorded C reference.

Deselected by default. To run them and inspect the pictures::

    pytest python/gaussfit_rs/tests/test_figures.py --mpl -m mpl_image_compare
    # write every figure to a directory instead of comparing hashes:
    pytest python/gaussfit_rs/tests/test_figures.py -m mpl_image_compare \
        --mpl-generate-path=/tmp/fit_figures

After an intentional change to the fits or the plotting, refresh the hash library with::

    pytest python/gaussfit_rs/tests/test_figures.py -m mpl_image_compare \
        --mpl-generate-hash-library=python/gaussfit_rs/tests/figure_hashes_mpl_3110_ft_2143.json

and state the reason in the commit; ``tox -e py313-figure`` runs the same tests.

Coverage: every ``c_reference_*`` fixture (the ``wide`` and ``muse`` parameter sets) across all
spectrum families in them -- clean, narrow, broad, asymmetric, blend, wings, flat_top, continuum,
poisson, white_noise, hot_pixel, masked, degenerate and low_snr -- with the fitted parameters, their
formal errors, reduced chi-square, both flags and the quality bits printed per
panel.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gaussfit_rs import (
    FLAG_SUCCESS,
    QUALITY_PEGGED,
    QUALITY_UNCONSTRAINED,
    QUALITY_ZERO_ERROR,
    fit_gaussian,
)
from gaussfit_rs.tests.helpers import FIXTURES, figure_test, fit_fixture

PANELS = 12

FIXTURE_IDS = [fixture.stem.removeprefix("c_reference_") for fixture in FIXTURES]
CASES = [
    (fixture, family)
    for fixture in FIXTURES
    for family in np.unique(np.load(fixture)["labels"]).tolist()
]
CASE_IDS = [f"{fixture.stem.removeprefix('c_reference_')}-{family}" for fixture, family in CASES]


def _gaussian(x, params):
    amp, vel, sigma = params[:3]
    return amp * np.exp(-0.5 * ((x - vel) / sigma) ** 2)


def _view(dopp, window, guide, velocity_range, margin=6):
    """
    Pixel slice to draw: the fit window, or the search window when no peak was found.
    """
    left, right = window
    if right <= left:
        inside = (
            np.flatnonzero(np.abs(dopp - guide) <= velocity_range)
            if np.isfinite(guide)
            else np.arange(dopp.size)
        )
        left, right = inside[0], inside[-1] + 1
    return slice(max(left - margin, 0), min(right + margin, dopp.size))


def _panel_text(row, fits, ref_fits, quality):
    """
    Fitted parameters and quality bits, compact enough for a panel caption.
    """
    if fits[row, 7] != FLAG_SUCCESS:
        return f"#{row} no fit (flag {fits[row, 7]:.0f})"
    amp, vel, sigma = fits[row, :3]
    e_amp, e_vel, e_sigma = fits[row, 3:6]
    marker = f" q={int(quality[row])}"
    return (
        f"#{row} A={amp:.3g}+/-{e_amp:.2g} V={vel:.3g}+/-{e_vel:.2g}\n"
        f"    S={sigma:.4g}+/-{e_sigma:.2g}\n"
        f"chi2={fits[row, 6]:.3g} flag R/C {fits[row, 7]:.0f}/{ref_fits[row, 7]:.0f}{marker}"
    )


@pytest.mark.parametrize(("fixture", "family"), CASES, ids=CASE_IDS)
@figure_test
def test_fit_gallery(fixture, family):
    """
    Up to twelve fits of one family: data, fit window, Rust against the C reference, parameters.
    """
    ref = np.load(fixture)
    fits, indices = fit_fixture(ref)
    quality = fits[:, 8]
    dopp, velocity_range = ref["dopp"], float(ref["velocity_range"])
    rows = np.flatnonzero(ref["labels"] == family)[:PANELS]
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
    for ax, row in zip(axes.flat, rows, strict=False):
        view = _view(dopp, indices[row], ref["guides"][row], velocity_range)
        x, y, err = dopp[view], ref["spectra"][row][view], ref["noise"][row][view]
        ax.errorbar(x, y, err, fmt=".", color="0.4", ms=4, lw=0.6)
        left, right = indices[row]
        if right > left:
            ax.axvspan(dopp[left], dopp[right - 1], color="0.92", zorder=0)
        fine = np.linspace(x[0], x[-1], 300)
        if fits[row, 7] == FLAG_SUCCESS:
            ax.plot(fine, _gaussian(fine, fits[row]), color="C0", lw=1.5, label="Rust")
        if ref["fits"][row, 7] == FLAG_SUCCESS:
            ax.plot(fine, _gaussian(fine, ref["fits"][row]), "--", color="C3", lw=1.2, label="C")
        ax.set_title(_panel_text(row, fits, ref["fits"], quality), fontsize=7.5)
        ax.set_xlabel("Doppler velocity [km/s]", fontsize=8)
    for ax in axes.flat[len(rows) :]:
        ax.set_axis_off()
    handles, _ = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        f"{family}: {fixture.stem} (window shaded; q = quality bits: "
        f"{QUALITY_UNCONSTRAINED} unconstrained, {QUALITY_ZERO_ERROR} zero error, "
        f"{QUALITY_PEGGED} pegged)"
    )
    return fig


@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURE_IDS)
@figure_test
def test_family_overview(fixture):
    """
    One representative spectrum per family, so the whole test range is inspectable at a glance.
    """
    ref = np.load(fixture)
    fits, indices = fit_fixture(ref)
    quality = fits[:, 8]
    dopp, velocity_range = ref["dopp"], float(ref["velocity_range"])
    families = np.unique(ref["labels"]).tolist()
    ncols = 4
    nrows = -(-len(families) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.6 * nrows), constrained_layout=True)
    for ax, family in zip(axes.flat, families, strict=False):
        rows = np.flatnonzero(ref["labels"] == family)
        # Show a fit that worked when the family has one, so every panel is inspectable.
        solved = rows[fits[rows, 7] == FLAG_SUCCESS]
        row = int(solved[0]) if solved.size else int(rows[0])
        view = _view(dopp, indices[row], ref["guides"][row], velocity_range)
        x, y, err = dopp[view], ref["spectra"][row][view], ref["noise"][row][view]
        ax.errorbar(x, y, err, fmt=".", color="0.45", ms=3, lw=0.5)
        fine = np.linspace(x[0], x[-1], 300)
        if fits[row, 7] == FLAG_SUCCESS:
            ax.plot(fine, _gaussian(fine, fits[row]), color="C0", lw=1.4)
        ax.set_title(f"{family} ({rows.size} rows)", fontsize=9)
        ax.text(
            0.02,
            0.96,
            _panel_text(row, fits, ref["fits"], quality),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    for ax in axes.flat[len(families) :]:
        ax.set_axis_off()
    fig.suptitle(f"{fixture.stem}: one example per family (parameters printed per panel)")
    return fig


@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURE_IDS)
@figure_test
def test_quality_per_family(fixture):
    """
    How well each family is constrained: success rate, unconstrained share, velocity error, chi2.
    """
    ref = np.load(fixture)
    fits, _ = fit_fixture(ref)
    quality = fits[:, 8].astype(int)
    families = np.unique(ref["labels"]).tolist()
    solved = fits[:, 7] == FLAG_SUCCESS
    stats = {
        "success [%]": [100.0 * np.mean(solved[ref["labels"] == family]) for family in families],
        "unconstrained of solved [%]": [
            100.0 * np.mean(quality[(ref["labels"] == family) & solved] & QUALITY_UNCONSTRAINED)
            if np.any((ref["labels"] == family) & solved)
            else 0.0
            for family in families
        ],
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for ax, (name, values) in zip(axes[:2], stats.items(), strict=True):
        ax.bar(families, values, color="C0")
        ax.set_title(name)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", rotation=60)
    for family in families:
        rows = (ref["labels"] == family) & solved
        if not rows.any():
            continue
        axes[2].plot(
            np.maximum(fits[rows, 4], 1e-9),
            np.maximum(fits[rows, 6], 1e-9),
            "o",
            ms=3,
            alpha=0.6,
            label=family,
        )
    axes[2].set_xlabel("velocity error [km/s]")
    axes[2].set_ylabel("reduced chi2")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=6, ncols=2)
    axes[2].set_title("solved rows: error against chi2")
    fig.suptitle(f"{fixture.stem}: fit quality per family")
    return fig


@figure_test
def test_parity_summary():
    """
    Largest Rust-versus-C difference per family and parameter, against the parity tolerance.
    """
    panels = [
        ("amplitude", 0, "relative to C"),
        ("velocity", 1, "in pixels"),
        ("width", 2, "relative to C"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for marker, fixture in zip("os", FIXTURES, strict=False):
        ref = np.load(fixture)
        fits, _ = fit_fixture(ref)
        dv = float(np.median(np.gradient(ref["dopp"])))
        ok = (ref["fits"][:, 7] == FLAG_SUCCESS) & (fits[:, 7] == FLAG_SUCCESS)
        families = np.unique(ref["labels"])
        for ax, (name, column, _) in zip(axes, panels, strict=True):
            scale = dv if column == 1 else np.maximum(np.abs(ref["fits"][:, column]), 1e-6)
            diff = np.abs(fits[:, column] - ref["fits"][:, column]) / scale
            worst = [
                np.max(diff[ok & (ref["labels"] == family)], initial=0.0) for family in families
            ]
            ax.plot(families, np.maximum(worst, 1e-7), marker, ls="none", label=fixture.stem)
            ax.set_title(name)
    for ax, (_, _, unit) in zip(axes, panels, strict=True):
        ax.axhline(5e-3, color="C3", ls="--", lw=1, label="close tolerance")
        ax.set_yscale("log")
        ax.set_ylabel(f"max |Rust - C| ({unit})")
        ax.tick_params(axis="x", rotation=60)
    axes[0].legend(fontsize=8)
    return fig


@figure_test
def test_low_level_gaussian_fits():
    """
    ``fit_gaussian`` on arbitrary data: a clean, a noisy, a bound-pinned and a misfit case.
    """
    rng = np.random.default_rng(7)
    x = np.linspace(0.0, 100.0, 60, dtype=np.float32)
    truth = 12.0 * np.exp(-0.5 * ((x - 50.0) / 8.0) ** 2)

    cases = [
        ("clean", truth, 0.05, [12.0, 50.0, 8.0], [0.0, 30.0, 1.0], [50.0, 70.0, 20.0]),
        (
            "noisy",
            truth + rng.normal(0.0, 2.0, x.size),
            2.0,
            [12.0, 50.0, 8.0],
            [0.0, 30.0, 1.0],
            [50.0, 70.0, 20.0],
        ),
        (
            "bound-pinned (sigma <= 6)",
            truth,
            0.05,
            [12.0, 50.0, 6.0],
            [0.0, 30.0, 1.0],
            [50.0, 70.0, 6.0],
        ),
        (
            "misfit (flat top)",
            12.0 * (np.abs(x - 50.0) < 8.0),
            0.05,
            [12.0, 50.0, 8.0],
            [0.0, 30.0, 1.0],
            [50.0, 70.0, 20.0],
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for ax, (name, data, noise, initial, lower, upper) in zip(axes.flat, cases, strict=True):
        y = np.ascontiguousarray(data, dtype=np.float32)
        err = np.full(x.size, noise, dtype=np.float32)
        fit = fit_gaussian(
            x=x,
            y=y,
            error=err,
            initial=np.asarray(initial, dtype=np.float32),
            lower_bounds=np.asarray(lower, dtype=np.float32),
            upper_bounds=np.asarray(upper, dtype=np.float32),
        )
        assert fit[7] == FLAG_SUCCESS
        if name.startswith("bound-pinned"):
            assert fit[2] == upper[2]
        ax.errorbar(x, y, err, fmt=".", color="0.45", ms=4, lw=0.6)
        ax.plot(x, _gaussian(x, np.asarray(initial)), ":", color="0.6", lw=1.2, label="initial")
        if fit[7] == FLAG_SUCCESS:
            ax.plot(x, _gaussian(x, fit), color="C0", lw=1.6, label="fit")
        ax.set_title(
            f"{name}: A={fit[0]:.3g}+/-{fit[3]:.2g} V={fit[1]:.3g}+/-{fit[4]:.2g} "
            f"S={fit[2]:.4g}+/-{fit[5]:.2g} chi2={fit[6]:.3g} flag {fit[7]:.0f}",
            fontsize=8,
        )
        ax.set_xlabel("x")
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("fit_gaussian: arbitrary data")
    return fig
