"""
Tests for fit_spectra_batch and fit_single_spectrum.
"""

import numpy as np
import pytest

from gaussfit_rs import (
    FLAG_NO_LOCAL_MAX,
    FLAG_SUCCESS,
    QUALITY_PEGGED,
    QUALITY_UNCONSTRAINED,
    QUALITY_ZERO_ERROR,
    FitResult,
    fit_single_spectrum,
    fit_spectra_batch,
)

SIGMA_TRUE = 30.0  # km/s
COMMON_KW = {
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


def _single(spec, v, err, guide=0.0, **overrides):
    return fit_single_spectrum(
        spectrum=spec,
        dopp_slit=v,
        spec_noise=err,
        guide_velocity=guide,
        **{**COMMON_KW, **overrides},
    )


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
    v, spec, err = _make_spectrum(amp=2.0, vel=15.0)
    r, (il, ir) = _single(spec, v, err)
    assert r.shape == (9,)
    assert r.dtype == np.float32
    assert r[7] == FLAG_SUCCESS
    assert abs(r[0] - 2.0) < 0.1
    assert abs(r[1] - 15.0) < 2.0
    assert abs(r[2] - SIGMA_TRUE) < 3.0
    assert 0 <= il < ir <= v.size


def test_single_no_local_max_outside_window():
    v, spec, err = _make_spectrum(vel=0.0)
    r, window = _single(spec, v, err, guide=250.0, velocity_range=10.0)
    assert r[7] == FLAG_NO_LOCAL_MAX
    assert np.isnan(r[:7]).all()
    assert window == (0, 0)


def test_single_accepts_non_contiguous_and_float64_inputs():
    v, spec, err = _make_spectrum()
    wide = np.stack([spec, spec]).astype(np.float64)
    r, _ = _single(wide[:, ::1][0], v.astype(np.float64), err[::1])
    r_strided, _ = _single(np.repeat(spec, 2)[::2], v, np.repeat(err, 2)[::2])
    assert r[7] == FLAG_SUCCESS
    np.testing.assert_array_equal(r, r_strided)


def test_single_rejects_invalid_options():
    v, spec, err = _make_spectrum()
    with pytest.raises(ValueError, match="finite"):
        _single(spec, v, err, width_min=np.nan)
    with pytest.raises(ValueError, match="amplitude_rel_min"):
        _single(spec, v, err, amplitude_rel_min=2.0, amplitude_rel_max=1.0)
    with pytest.raises(ValueError, match="xtol"):
        _single(spec, v, err, xtol=np.nan)
    with pytest.raises(ValueError, match="max_iter"):
        _single(spec, v, err, max_iter=0)


def test_single_large_window_fits():
    v, spec, err = _make_spectrum(n=700)
    r, (il, ir) = _single(spec, v, err, npix=300, npix_slack=0, dv=1.0, velocity_range=50.0)
    assert ir - il > 512
    assert r[7] == FLAG_SUCCESS


def test_batch_output_shapes(batch_data):
    fits, idx = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        guide_velocities=0.0,
        **COMMON_KW,
    )
    assert fits.shape == (batch_data["n"], 9)
    assert idx.shape == (batch_data["n"], 2)
    assert fits.dtype == np.float32
    assert idx.dtype == np.int32
    assert np.all(fits[:, 7] == FLAG_SUCCESS)


def test_batch_matches_single_spectrum(batch_data):
    guide_velocities = np.linspace(-15.0, 15.0, batch_data["n"], dtype=np.float32)
    fits, indices = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        guide_velocities=guide_velocities,
        **COMMON_KW,
    )
    for i in range(10):
        r_single, (il, ir) = _single(
            batch_data["spectra"][i], batch_data["v"], batch_data["noise"][i], guide_velocities[i]
        )
        np.testing.assert_array_equal(fits[i], r_single, err_msg=f"row {i} mismatch")
        assert (indices[i, 0], indices[i, 1]) == (il, ir)


def _slit_grids(v):
    # Three slits whose velocity grids are offset from one another.
    return np.stack([v - 20.0, v, v + 20.0]).astype(np.float32)


def test_slit_table_routes_each_row_to_its_slit(batch_data):
    dopp = _slit_grids(batch_data["v"])
    slit_index = (np.arange(batch_data["n"]) % len(dopp)).astype(np.int32)
    fits, indices = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=dopp,
        spec_noise=batch_data["noise"],
        guide_velocities=0.0,
        slit_index=slit_index,
        **COMMON_KW,
    )
    for i in range(10):
        r_single, (il, ir) = _single(
            batch_data["spectra"][i], dopp[slit_index[i]], batch_data["noise"][i]
        )
        np.testing.assert_array_equal(fits[i], r_single, err_msg=f"row {i} mismatch")
        assert (indices[i, 0], indices[i, 1]) == (il, ir)
    # The line sits at 0 on grid 1, so grids 0 and 2 shift its velocity by -20 and +20.
    velocities = fits[:, 1]
    assert np.all(velocities[slit_index == 0] < -10.0)
    assert np.all(np.abs(velocities[slit_index == 1]) < 10.0)
    assert np.all(velocities[slit_index == 2] > 10.0)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"spectra": "1d"}, r"2-D"),
        ({"spec_noise": "short"}, r"spec_noise shape"),
        ({"dopp_slit": "short"}, r"dopp_slit columns"),
        ({"guide_velocities": np.zeros(3, dtype=np.float32)}, r"guide_velocities length"),
        ({"dopp_slit": "table"}, r"slit_index is required"),
        ({"dopp_slit": "table", "slit_index": [0, 3]}, r"slit_index values"),
        ({"dopp_slit": "table", "slit_index": [0, -1]}, r"slit_index values"),
        ({"dopp_slit": "table", "slit_index": [0]}, r"slit_index length"),
    ],
)
def test_batch_validates_shapes(override, match):
    v, spec, err = _make_spectrum()
    kwargs = {
        "spectra": np.stack([spec, spec]),
        "dopp_slit": v,
        "spec_noise": np.stack([err, err]),
        "guide_velocities": 0.0,
        **COMMON_KW,
    }
    for key, value in override.items():
        if not isinstance(value, str):
            kwargs[key] = value
        elif value == "1d":
            kwargs[key] = spec
        elif value == "short":
            kwargs[key] = kwargs[key][..., :50]
        else:
            kwargs[key] = _slit_grids(v)
    with pytest.raises(ValueError, match=match):
        fit_spectra_batch(**kwargs)


@pytest.mark.parametrize("guide", [np.nan, np.inf, -np.inf])
def test_nonfinite_guide_reports_no_local_max(guide):
    # Matches the C extension: a nonfinite guide matches no pixel, so no peak is found.
    v, spec, err = _make_spectrum()
    r, window = _single(spec, v, err, guide=guide)
    assert r[7] == FLAG_NO_LOCAL_MAX
    assert np.isnan(r[:7]).all()
    assert window == (0, 0)
    fits, idx = fit_spectra_batch(
        spectra=np.stack([spec, spec]),
        dopp_slit=v,
        spec_noise=np.stack([err, err]),
        guide_velocities=np.array([guide, 0.0], dtype=np.float32),
        **COMMON_KW,
    )
    assert fits[0, 7] == FLAG_NO_LOCAL_MAX
    assert fits[1, 7] == FLAG_SUCCESS
    np.testing.assert_array_equal(idx[0], [0, 0])


def test_quality_bits_report_unconstrained_successes():
    """
    Column 8 carries the documented quality bits.
    """
    v, spec, err = _make_spectrum(amp=1.0, noise=0.05)

    fits = _single(spec, v, err)[0]
    assert fits.shape == (9,)
    assert fits[7] == FLAG_SUCCESS
    assert fits[8] == 0  # a real line constrains its parameters

    # A faint line over structured noise fits "successfully" but is not constrained by the data:
    # its velocity and width errors exceed the intervals those parameters were bounded to.
    checker = np.where(np.arange(v.size) % 2 == 0, 3.0, -3.0)
    faint = (0.5 * np.exp(-0.5 * (v / SIGMA_TRUE) ** 2) + checker).astype(np.float32)
    faint_err = np.full(v.size, 3.0, dtype=np.float32)
    fits_u = _single(faint, v, faint_err)[0]
    assert fits_u[7] == FLAG_SUCCESS
    assert fits_u[4] >= 2.0 * COMMON_KW["dv"] * COMMON_KW["npix"]
    assert int(fits_u[8]) & QUALITY_UNCONSTRAINED

    # Failed fits are reported by the result flag, not the indicator.
    flat = np.full(v.size, -1.0, dtype=np.float32)
    fits_f = _single(flat, v, err)[0]
    assert fits_f[7] == FLAG_NO_LOCAL_MAX
    assert fits_f[8] == 0  # failed fits carry no indicator


def test_batch_quality_bits_match_the_documented_criterion():
    """
    The batch bits (ninth column) equal the documented span and zero tests on the returned errors.
    """
    v, spec, err = _make_spectrum(amp=1.0, noise=0.05)
    checker = np.where(np.arange(v.size) % 2 == 0, 3.0, -3.0)
    noisy = (0.5 * np.exp(-0.5 * (v / SIGMA_TRUE) ** 2) + checker).astype(np.float32)
    # A noiseless line far too faint to constrain: its determinant falls under the guard, so all
    # three errors come back exactly zero. This exercises the guard, not an exact-fit residual.
    tiny = (1.0e-6 * np.exp(-0.5 * (v / SIGMA_TRUE) ** 2)).astype(np.float32)
    spectra = np.stack([spec, noisy, tiny])
    noises = np.stack([err, np.full(v.size, 3.0, dtype=np.float32), err])
    kwargs = {"spectra": spectra, "dopp_slit": v, "spec_noise": noises, "guide_velocities": 0.0}

    fits, indices = fit_spectra_batch(**kwargs, **COMMON_KW)
    flags = fits[:, 8]
    assert fits.shape[1] == 9
    assert flags.dtype == np.float32
    assert indices.shape == (3, 2)
    assert fits[0, 7] == FLAG_SUCCESS
    assert flags[0] == 0
    assert int(flags[1]) & QUALITY_UNCONSTRAINED
    assert np.array_equal(fits[2, 3:6], np.zeros(3, dtype=np.float32))  # the guard zeroed them
    assert int(flags[2]) & QUALITY_ZERO_ERROR
    assert not int(flags[2]) & QUALITY_UNCONSTRAINED  # zero errors stay under any span

    # The span and zero bits are reproducible from the returned columns alone; the pegged bit is
    # not (it needs the bounds, and the fit works in peak-normalised amplitude units).
    span_velocity = 2.0 * COMMON_KW["dv"] * COMMON_KW["npix"]
    span_width = COMMON_KW["width_max"] - COMMON_KW["width_min"]
    derived = (
        (QUALITY_UNCONSTRAINED * (fits[:, 4] >= span_velocity))
        | (QUALITY_UNCONSTRAINED * (fits[:, 5] >= span_width))
        | (QUALITY_ZERO_ERROR * (fits[:, 3:6] == 0).any(axis=1))
    ).astype(np.float32)
    assert np.array_equal(flags % QUALITY_PEGGED, derived % QUALITY_PEGGED)


def test_fit_result_named_tuple():
    v, spec, err = _make_spectrum(amp=2.0)
    row, _ = _single(spec, v, err)
    result = FitResult.from_array(row)
    assert result.converged
    assert result.quality == 0
    assert abs(result.amplitude - 2.0) < 0.1
    assert result.flag == FLAG_SUCCESS


def test_meta_counts_are_opt_in_and_match_between_entry_points(batch_data):
    v, spec, err = _make_spectrum()
    r, _window, counts = _single(spec, v, err, meta=True)
    assert r[7] == FLAG_SUCCESS
    assert counts.dtype == np.int32
    assert counts.shape == (2,)
    assert counts[1] >= counts[0] >= 1
    assert len(_single(spec, v, err)) == 2
    _, _, failed = _single(spec, v, err, guide=np.nan, meta=True)
    np.testing.assert_array_equal(failed, [-1, -1])

    fits, _, batch_counts = fit_spectra_batch(
        spectra=batch_data["spectra"],
        dopp_slit=batch_data["v"],
        spec_noise=batch_data["noise"],
        guide_velocities=0.0,
        meta=True,
        **COMMON_KW,
    )
    assert batch_counts.shape == (batch_data["n"], 2)
    assert (batch_counts[fits[:, 7] == FLAG_SUCCESS] >= 1).all()
    for i in range(5):
        _, _, single = _single(
            batch_data["spectra"][i], batch_data["v"], batch_data["noise"][i], meta=True
        )
        np.testing.assert_array_equal(batch_counts[i], single, err_msg=f"row {i}")
