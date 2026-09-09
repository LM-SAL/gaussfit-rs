"""
Parity with MUSE's C extension, the reference implementation.

``test_matches_recorded_c_reference`` always runs against the fixtures written by
``make_c_reference.py``; ``test_matches_live_c_extension`` also runs when the pinned
``gaussfit_c_reference`` package is installed. See ``helpers.assert_fit_parity`` for the contract,
checked family by family so a failure names the kind of spectrum.
"""

import numpy as np
import pytest

from gaussfit_rs import FLAG_SUCCESS
from gaussfit_rs.tests.helpers import DATA_DIR, FIXTURES, assert_fit_parity, fit_fixture
from gaussfit_rs.tests.make_c_reference import PARAMETER_SETS, build_cases, run_c


@pytest.mark.parametrize("backend", ["Rust", "C"])
@pytest.mark.parametrize("column", [0, 1, 2, 6], ids=["amplitude", "velocity", "width", "chi2"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_parity_rejects_nonfinite_success(backend, column, value):
    # Keep another row constrained so the coverage check cannot hide the bug.
    c_fits = np.tile([10.0, 0.0, 2.0, 0.1, 0.1, 0.1, 1.0, FLAG_SUCCESS], (2, 1))
    fits = c_fits.copy()
    assert_fit_parity(fits, c_fits, dv=1.0)
    (fits if backend == "Rust" else c_fits)[0, column] = value
    with pytest.raises(AssertionError, match=f"{backend} successful parameters and chi-square"):
        assert_fit_parity(fits, c_fits, dv=1.0)


def _check(ref, c_fits, c_indices, labels):
    fits, indices = fit_fixture(ref)
    dv = float(np.median(np.gradient(ref["dopp"])))
    assert_fit_parity(fits, c_fits, dv, indices=indices, c_indices=c_indices, labels=labels)


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda path: path.stem)
def test_matches_recorded_c_reference(fixture):
    ref = np.load(fixture)
    _check(ref, ref["fits"], ref["indices"], ref["labels"])


@pytest.mark.parametrize("name", list(PARAMETER_SETS))
def test_matches_live_c_extension(name):
    pytest.importorskip("gaussfit_c_reference.fit_single_spectrum_ext")
    ref = np.load(DATA_DIR / f"c_reference_{name}.npz")
    dopp, spectra, noise, guides, labels = build_cases(PARAMETER_SETS[name])
    np.testing.assert_array_equal(
        spectra, ref["spectra"], err_msg="fixture is stale; rerun make_c_reference.py"
    )
    c_fits, c_indices = run_c(dopp, spectra, noise, guides, PARAMETER_SETS[name])
    _check(ref, c_fits, c_indices, labels)
