"""
Shared helpers for the gaussfit-rs test suite.
"""

from functools import wraps
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gaussfit_rs import FLAG_SUCCESS, fit_spectra_batch
from gaussfit_rs.tests.make_c_reference import FIT_KEYS

DATA_DIR = Path(__file__).parent / "data"
FIXTURES = sorted(DATA_DIR.glob("c_reference_*.npz"))
RTOL = 5e-3  # float32 cmpfit versus float64 rmpfit
CHI_ATOL = 5e-3  # reduced chi-square of noise-free fits is rounding noise
MIN_CONSTRAINED = 0.5  # fraction of successful fits whose errors must be comparable


def fit_fixture(ref):
    """
    Fit every spectrum of a recorded C-reference fixture with the Rust backend.

    ``fits[:, 8]`` holds the quality bits (see :data:`gaussfit_rs.QUALITY_UNCONSTRAINED`).
    """
    params = {key: cast(ref[key]) for key, cast in FIT_KEYS.items()}
    return fit_spectra_batch(
        spectra=ref["spectra"],
        dopp_slit=ref["dopp"],
        spec_noise=ref["noise"],
        guide_velocities=ref["guides"],
        dv=float(np.median(np.gradient(ref["dopp"]))),
        **params,
    )


def assert_fit_parity(
    fits, c_fits, dv, *, indices=None, c_indices=None, labels=None, accepted_worse=None
):
    """
    Assert that Rust fit results satisfy the parity contract with MUSE's C extension.

    Flags and fit windows must be identical, and failed fits must be NaN. Successful parameters
    and chi-square must be finite in both implementations. Parameters must agree to
    ``RTOL`` (relative for amplitude and width, in pixels for velocity, plus ``CHI_ATOL`` for chi-
    square) unless Rust's reduced chi-square is no higher than C's within that same tolerance: the
    float32 C solver can stop at a different point of a flat, bound-limited chi-square valley.
    Rust is never allowed to be worse. Errors are compared only for well-constrained fits (both
    solvers report every error positive and below half the parameter, or one pixel for velocity)
    where the parameters agree: a fit pinned at a bound or narrower than a pixel has a near-singular
    Hessian whose float32 inverse is not meaningful. At least ``MIN_CONSTRAINED`` of the successful
    fits must qualify, so the error check cannot silently vanish.

    ``accepted_worse`` names documented exceptions to the one-sided rule: a mapping of label to
    ``(absolute row indices, cap)``, where ``cap`` is the maximum tolerated ``Rust chi2 / C chi2``
    for those rows only. Every entry must match a decision recorded in ``docs/design-notes.rst``;
    the function returns how many rows used an exception so a caller can fail when one goes stale.
    """
    labels = np.zeros(len(fits), dtype=int) if labels is None else np.asarray(labels)
    successful = c_fits[:, 7] == FLAG_SUCCESS
    n_constrained = 0
    n_accepted = 0
    for label in np.unique(labels):
        rows = labels == label
        np.testing.assert_array_equal(fits[rows, 7], c_fits[rows, 7], err_msg=f"{label}: flags")
        if indices is not None:
            np.testing.assert_array_equal(
                indices[rows], c_indices[rows], err_msg=f"{label}: windows"
            )
        assert np.isnan(fits[rows & ~successful, :7]).all(), f"{label}: failed fits must be NaN"
        r, c = fits[rows & successful], c_fits[rows & successful]
        for name, values in (("Rust", r), ("C", c)):
            assert np.isfinite(values[:, [0, 1, 2, 6]]).all(), (
                f"{label}: {name} successful parameters and chi-square must be finite"
            )
        scale = np.stack(
            [np.maximum(np.abs(c[:, 0]), 1e-6), np.full(len(c), dv), np.abs(c[:, 2])], axis=1
        )
        chi_close = np.abs(r[:, 6] - c[:, 6]) <= RTOL * c[:, 6] + CHI_ATOL
        close = (np.abs(r[:, :3] - c[:, :3]) <= RTOL * scale).all(axis=1) & chi_close
        chi_worse = r[:, 6] > c[:, 6] * (1 + RTOL) + CHI_ATOL
        # Documented exceptions (see the docstring and docs/design-notes.rst): the local lmpar fix
        # makes one specific low-SNR row land in a slightly worse local minimum than C. Only the
        # named rows are exempt, and only up to the recorded ratio cap.
        accepted = np.zeros(len(r), dtype=bool)
        exception = (accepted_worse or {}).get(label)
        if exception is not None:
            allowed_rows, cap = exception
            label_rows = np.flatnonzero(rows & successful)
            wanted = np.isin(
                label_rows, np.fromiter(allowed_rows, dtype=int, count=len(allowed_rows))
            )
            accepted = chi_worse & wanted & (r[:, 6] <= c[:, 6] * cap + CHI_ATOL)
        n_accepted += int(accepted.sum())
        worse = ~close & chi_worse & ~accepted
        assert not worse.any(), (
            f"{label}: Rust fit worse than C, reduced chi-square {r[worse, 6]} vs {c[worse, 6]}"
        )
        r_err, c_err = r[:, 3:6], c[:, 3:6]
        bound = np.stack(
            [0.5 * np.abs(c[:, 0]), np.full(len(c), dv), 0.5 * np.abs(c[:, 2])], axis=1
        )
        constrained = close & ((r_err > 0) & (c_err > 0) & (r_err < bound) & (c_err < bound)).all(
            axis=1
        )
        error_close = (np.abs(r_err - c_err) <= RTOL * (np.abs(c_err) + scale)).all(axis=1)
        assert error_close[constrained].all(), f"{label}: errors differ on well-constrained fits"
        n_constrained += int(constrained.sum())
    assert n_constrained >= MIN_CONSTRAINED * successful.sum(), (
        "too few well-constrained fits to compare errors"
    )
    return n_accepted


def _clean_version(version):
    return "dev" if ("dev" in version or "rc" in version) else version.replace(".", "")


def get_hash_library_name():
    """
    Generate the figure hash library name for this environment.
    """
    freetype_version = mpl.ft2font.__freetype_version__.replace(".", "")
    return f"figure_hashes_mpl_{_clean_version(mpl.__version__)}_ft_{freetype_version}.json"


def figure_test(test_function):
    """
    Mark a test whose returned (or current) figure is verified against the hash library.

    The test name is the hash identifier. A PNG is also written to the pytest-mpl results directory,
    and every such test carries the ``mpl_image_compare`` marker for filtering.
    """
    hash_library_file = Path(__file__).parent / get_hash_library_name()

    @pytest.mark.mpl_image_compare(
        hash_library=hash_library_file,
        savefig_kwargs={"metadata": {"Software": None}},
        style="default",
    )
    @wraps(test_function)
    def test_wrapper(*args, **kwargs):
        ret = test_function(*args, **kwargs)
        return plt.gcf() if ret is None else ret

    return test_wrapper
