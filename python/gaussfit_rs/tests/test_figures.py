"""
Figure tests: fitted profiles per spectrum family, Rust against the recorded C reference.

Deselected by default; run with ``pytest --mpl -m mpl_image_compare`` or ``tox -e py313-figure``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from gaussfit_rs import FLAG_SUCCESS
from gaussfit_rs.tests.helpers import DATA_DIR, FIXTURES, figure_test, fit_fixture

MUSE_FIXTURE = DATA_DIR / "c_reference_muse.npz"
FAMILIES = np.unique(np.load(MUSE_FIXTURE)["labels"]).tolist()
PANELS = 12


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


@pytest.mark.parametrize("family", FAMILIES)
@figure_test
def test_fit_gallery(family):
    ref = np.load(MUSE_FIXTURE)
    fits, indices = fit_fixture(ref)
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
        flags = f"flag Rust {fits[row, 7]:.0f} / C {ref['fits'][row, 7]:.0f}"
        ax.set_title(f"#{row} guide {ref['guides'][row]:.0f} km/s, {flags}", fontsize=9)
        ax.set_xlabel("Doppler velocity [km/s]", fontsize=8)
    for ax in axes.flat[len(rows) :]:
        ax.set_axis_off()
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{family}: {MUSE_FIXTURE.stem} (window shaded)")
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
