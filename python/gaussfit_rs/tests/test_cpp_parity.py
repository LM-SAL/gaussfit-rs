"""
Parity with MUSE's fastfit2 C++ extensions, the batch implementation gaussfit-rs is benchmarked
against.

Runs only when the pinned ``gaussfit_cpp_reference`` package is installed (``tox -e py313-cparity``
builds it next to the C reference). The C++ port shares its solver, ``cmpfit-1.5`` built with
``MPFIT_FLOAT``, with the C reference, so the contract is the same one-sided one as
``test_c_parity.py``, checked with the same helper, over the same seeds, with the same row-level
exception list and the same hang list.
"""

import numpy as np
import pytest

from gaussfit_rs import FLAG_SUCCESS
from gaussfit_rs.tests.helpers import DATA_DIR, assert_fit_parity, fit_fixture
from gaussfit_rs.tests.make_c_reference import FIT_KEYS, PARAMETER_SETS, build_cases
from gaussfit_rs.tests.test_c_parity import ACCEPTED_WORSE, HANGS_C, N_SEEDS

cpp_single = pytest.importorskip("gaussfit_cpp_reference.fit_single_spectrum_ext")
cpp_batch = pytest.importorskip("gaussfit_cpp_reference.fit_batch_spectrum_ext")


def run_cpp(dopp, spectra, noise, guides, params):
    """
    Fit every spectrum with the pinned C++ single-spectrum extension.
    """
    fit_params = {key: cast(params[key]) for key, cast in FIT_KEYS.items()}
    fits = np.empty((len(spectra), 8), dtype=np.float32)
    indices = np.empty((len(spectra), 2), dtype=np.int32)
    dv = float(np.median(np.gradient(dopp)))
    for i, (spec, err, guide) in enumerate(zip(spectra, noise, guides, strict=True)):
        fits[i], indices[i] = cpp_single._fit_single_spectrum_c(
            spectrum=spec,
            dopp_slit=dopp,
            spec_noise=err,
            guide_velocity=float(guide),
            dv=dv,
            SG_xpixels=len(dopp),
            **fit_params,
        )
    return fits, indices


def run_cpp_batch(dopp, spectra, noise, guides, params, n_threads=0):
    """
    Fit every spectrum in one call to the pinned C++ batch extension; windows come from its mask.
    """
    n = len(dopp)
    mask = np.zeros((len(spectra), n), dtype=bool)
    f32 = np.float32
    fits = cpp_batch.fit_spectra_batch(
        np.ascontiguousarray(spectra, dtype=f32),
        dopp.reshape(1, -1),
        1,
        np.ascontiguousarray(noise, dtype=f32),
        0,
        np.ascontiguousarray(guides, dtype=f32),
        1,
        f32(params["velocity_range"]),
        int(params["npix"]),
        int(params["npix_slack"]),
        f32(np.median(np.gradient(dopp))),
        f32(params["width_min"]),
        n,
        f32(params["amplitude_rel_min"]),
        f32(params["amplitude_rel_max"]),
        f32(params["width_max"]),
        f32(params["width_guess"]),
        mask,
        len(spectra),
        n_threads,
    )
    indices = np.zeros((len(spectra), 2), dtype=np.int32)
    fitted = mask.any(axis=1)
    indices[fitted, 0] = np.argmax(mask[fitted], axis=1)
    indices[fitted, 1] = n - np.argmax(mask[fitted, ::-1], axis=1)
    return fits, indices


@pytest.mark.parametrize("seed", range(N_SEEDS))
@pytest.mark.parametrize("name", list(PARAMETER_SETS))
def test_matches_live_cpp_extension(name, seed):
    if (name, seed) in HANGS_C:
        pytest.skip("the shared cmpfit solver does not return on this seed")
    params = PARAMETER_SETS[name]
    dopp, spectra, noise, guides, labels = build_cases(params, seed=seed)
    c_fits, c_indices = run_cpp(dopp, spectra, noise, guides, params)
    live = {"dopp": dopp, "spectra": spectra, "noise": noise, "guides": guides, **params}
    fits, indices = fit_fixture(live)
    assert_fit_parity(
        fits,
        c_fits,
        float(np.median(np.gradient(dopp))),
        indices=indices,
        c_indices=c_indices,
        labels=labels,
        accepted_worse=ACCEPTED_WORSE.get((name, seed)),
    )


@pytest.mark.parametrize("name", list(PARAMETER_SETS))
def test_cpp_batch_matches_cpp_single(name):
    # The benchmark uses the batch entry point; pin it to the single-spectrum core it mirrors.
    params = PARAMETER_SETS[name]
    dopp, spectra, noise, guides, _labels = build_cases(params)
    single_fits, single_indices = run_cpp(dopp, spectra, noise, guides, params)
    batch_fits, batch_indices = run_cpp_batch(dopp, spectra, noise, guides, params)
    np.testing.assert_array_equal(batch_fits, single_fits)
    # The batch mask records only the windows of fits that succeeded; the single entry point also
    # reports the window of a fit it rejected (too few valid samples, or no convergence).
    fitted = single_fits[:, 7] == FLAG_SUCCESS
    assert fitted.sum() > len(fitted) // 2
    np.testing.assert_array_equal(batch_indices[fitted], single_indices[fitted])


@pytest.mark.parametrize("name", list(PARAMETER_SETS))
def test_cpp_matches_recorded_c_reference(name):
    # Same cmpfit, same float32 problem: the port must reproduce the C fixtures exactly.
    ref = np.load(DATA_DIR / f"c_reference_{name}.npz")
    params = {key: cast(ref[key]) for key, cast in FIT_KEYS.items()}
    fits, indices = run_cpp(ref["dopp"], ref["spectra"], ref["noise"], ref["guides"], params)
    np.testing.assert_array_equal(indices, ref["indices"])
    np.testing.assert_array_equal(fits, ref["fits"])
