"""
Tests for fit_single_spectrum and fit_spectra_batch.
"""

import numpy as np
import pytest

from gaussfit_rs import (
    FLAG_NO_LOCAL_MAX,
    FLAG_SUCCESS,
    FitResult,
    fit_single_spectrum,
    fit_spectra_batch,
    fit_spectra_batch_guided,
)

SIGMA_TRUE = 30.0  # km/s
COMMON_KW = {
    "guide_velocity": 0.0,
    "velocity_range": 200.0,
    "npix": 10,
    "npix_slack": 2,
    "dv": 12.0,
    "width_min": 5.0,
    "amplitude_rel_min": 0.1,
    "amplitude_rel_max": 2.0,
    "width_max": 100.0,
    "width_guess": 30.0,
}


def _make_spectrum(amp=1.0, vel=0.0, sigma=SIGMA_TRUE, noise=0.05, n=60):
    v = np.linspace(-300.0, 300.0, n, dtype=np.float32)
    profile = amp * np.exp(-0.5 * ((v - vel) / sigma) ** 2)
    spec = (profile + noise * np.random.default_rng(0).standard_normal(n)).astype(np.float32)
    err = np.full(n, noise, dtype=np.float32)
    return v, spec, err


@pytest.fixture
def batch_data():
    rng = np.random.default_rng(42)
    n = 200
    v = np.linspace(-300, 300, 60, dtype=np.float32)
    noise_level = 0.05
    spectra = np.stack(
        [
            np.exp(-0.5 * (v / SIGMA_TRUE) ** 2).astype(np.float32)
            + (noise_level * rng.standard_normal(60)).astype(np.float32)
            for _ in range(n)
        ]
    )
    noise = np.full((n, 60), noise_level, dtype=np.float32)
    return {"v": v, "spectra": spectra, "noise": noise, "n": n}


def test_single_recovers_gaussian():
    v, spec, err = _make_spectrum(noise=0.02)
    r, (il, ir) = fit_single_spectrum(spectrum=spec, dopp_slit=v, spec_noise=err, **COMMON_KW)
    assert r[7] == FLAG_SUCCESS
    assert abs(r[1]) < 5.0, f"velocity {r[1]} km/s off by >5"
    assert abs(r[2] - SIGMA_TRUE) < 5.0, f"sigma {r[2]} off by >5"
    assert il < ir


def test_single_fit_result_namedtuple():
    v, spec, err = _make_spectrum(noise=0.02)
    r_arr, _ = fit_single_spectrum(spectrum=spec, dopp_slit=v, spec_noise=err, **COMMON_KW)
    fr = FitResult.from_array(r_arr)
    assert fr.converged
    assert isinstance(fr.velocity, float)


def test_single_no_local_max_outside_window():
    v, spec, err = _make_spectrum(vel=0.0, noise=0.02)
    r, _ = fit_single_spectrum(
        spectrum=spec,
        dopp_slit=v,
        spec_noise=err,
        **{**COMMON_KW, "guide_velocity": 290.0, "velocity_range": 5.0},
    )
    assert r[7] == FLAG_NO_LOCAL_MAX


def test_single_output_shape():
    v, spec, err = _make_spectrum()
    r, idx = fit_single_spectrum(spectrum=spec, dopp_slit=v, spec_noise=err, **COMMON_KW)
    assert r.shape == (8,)
    assert len(idx) == 2


def test_single_n_pixels_parameter():
    v, spec, err = _make_spectrum()
    r40, _ = fit_single_spectrum(
        spectrum=spec, dopp_slit=v, spec_noise=err, n_pixels=40, **COMMON_KW
    )
    assert r40[7] in (FLAG_SUCCESS, FLAG_NO_LOCAL_MAX)


def test_single_non_contiguous_arrays():
    v, spec, err = _make_spectrum(noise=0.02)
    r, _ = fit_single_spectrum(
        spectrum=spec[::-1].copy()[::-1],  # non-contiguous
        dopp_slit=v,
        spec_noise=err,
        **COMMON_KW,
    )
    assert r[7] in (FLAG_SUCCESS, FLAG_NO_LOCAL_MAX)


def test_single_raises_on_zero_n_pixels():
    v, spec, err = _make_spectrum()
    with pytest.raises(ValueError, match="n_pixels must be positive"):
        fit_single_spectrum(spectrum=spec, dopp_slit=v, spec_noise=err, n_pixels=0, **COMMON_KW)


def test_single_raises_on_negative_n_pixels():
    v, spec, err = _make_spectrum()
    with pytest.raises(ValueError, match="n_pixels must be positive"):
        fit_single_spectrum(spectrum=spec, dopp_slit=v, spec_noise=err, n_pixels=-1, **COMMON_KW)


def test_single_rejects_invalid_fit_parameters():
    v, spec, err = _make_spectrum()
    with pytest.raises(ValueError, match="finite"):
        fit_single_spectrum(
            spectrum=spec,
            dopp_slit=v,
            spec_noise=err,
            **{**COMMON_KW, "width_min": np.nan},
        )
    with pytest.raises(ValueError, match="amplitude_rel_min"):
        fit_single_spectrum(
            spectrum=spec,
            dopp_slit=v,
            spec_noise=err,
            **{**COMMON_KW, "amplitude_rel_min": 2.0, "amplitude_rel_max": 1.0},
        )


def test_single_large_window_uses_heap_fallback():
    n = 700
    v = np.linspace(-350.0, 349.0, n, dtype=np.float32)
    spec = np.exp(-0.5 * (v / SIGMA_TRUE) ** 2).astype(np.float32)
    err = np.full(n, 0.01, dtype=np.float32)
    r, (il, ir) = fit_single_spectrum(
        spectrum=spec,
        dopp_slit=v,
        spec_noise=err,
        **{
            **COMMON_KW,
            "npix": 300,
            "npix_slack": 0,
            "dv": 1.0,
            "velocity_range": 50.0,
        },
    )
    assert ir - il > 512
    assert r[7] == FLAG_SUCCESS


def test_batch_output_shapes(batch_data):
    fits, idx = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        **COMMON_KW,
    )
    assert fits.shape == (batch_data["n"], 8)
    assert idx.shape == (batch_data["n"], 2)
    assert fits.dtype == np.float32
    assert idx.dtype == np.int32


def test_batch_matches_single_spectrum(batch_data):
    fits, indices = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        **COMMON_KW,
    )
    for i in range(10):
        r_single, (il, ir) = fit_single_spectrum(
            spectrum=batch_data["spectra"][i],
            dopp_slit=batch_data["v"],
            spec_noise=batch_data["noise"][i],
            **COMMON_KW,
        )
        np.testing.assert_array_equal(fits[i], r_single, err_msg=f"row {i} mismatch")
        assert indices[i, 0] == il
        assert indices[i, 1] == ir


def test_guided_batch_matches_single_spectrum(batch_data):
    guide_velocities = np.linspace(-15.0, 15.0, batch_data["n"], dtype=np.float32)
    batch_kw = {key: value for key, value in COMMON_KW.items() if key != "guide_velocity"}
    fits, indices = fit_spectra_batch_guided(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        guide_velocities=guide_velocities,
        **batch_kw,
    )
    for i in range(10):
        r_single, (il, ir) = fit_single_spectrum(
            spectrum=batch_data["spectra"][i],
            dopp_slit=batch_data["v"],
            spec_noise=batch_data["noise"][i],
            **{**COMMON_KW, "guide_velocity": float(guide_velocities[i])},
        )
        np.testing.assert_array_equal(fits[i], r_single, err_msg=f"row {i} mismatch")
        assert indices[i, 0] == il
        assert indices[i, 1] == ir


def test_batch_all_converge_on_clean_data(batch_data):
    n = 100
    spectra = np.stack(
        [np.exp(-0.5 * (batch_data["v"] / SIGMA_TRUE) ** 2).astype(np.float32) for _ in range(n)]
    )
    noise = np.full((n, 60), 0.01, dtype=np.float32)
    fits, _ = fit_spectra_batch(
        spectra=spectra, dopp_slit=batch_data["v"], spec_noise=noise, **COMMON_KW
    )
    assert np.all(fits[:, 7] == FLAG_SUCCESS)


def test_batch_n_pixels_limits_columns(batch_data):
    fits, _ = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        n_pixels=40,
        **COMMON_KW,
    )
    assert fits.shape == (batch_data["n"], 8)


def test_batch_raises_on_1d_spectra(batch_data):
    with pytest.raises(ValueError, match="2-D"):
        fit_spectra_batch(
            spectra=batch_data["spectra"][0],
            dopp_slit=batch_data["v"],
            spec_noise=batch_data["noise"],
            **COMMON_KW,
        )


def test_batch_requires_exact_noise_shape(batch_data):
    with pytest.raises(ValueError, match="spec_noise shape"):
        fit_spectra_batch(
            spectra=batch_data["spectra"],
            dopp_slit=batch_data["v"],
            spec_noise=batch_data["noise"][:, :50],
            n_pixels=50,
            **COMMON_KW,
        )
