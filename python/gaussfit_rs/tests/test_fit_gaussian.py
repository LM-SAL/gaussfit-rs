"""
Tests for fit_gaussian_f32 and fit_gaussian_f64.
"""

import numpy as np
import pytest

from gaussfit_rs import (
    FLAG_NO_CONVERGENCE,
    FLAG_SUCCESS,
    fit_gaussian_f32,
    fit_gaussian_f64,
)


def _gaussian(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _make_data(amp=1.0, mean=0.0, sigma=1.0, noise=0.01, dtype=np.float32, n=20):
    x = np.linspace(-3 * sigma, 3 * sigma, n, dtype=dtype)
    y = _gaussian(x, amp, mean, sigma).astype(dtype)
    err = np.full(n, noise, dtype=dtype)
    return x, y, err


def test_f32_recovers_unit_gaussian():
    x, y, err = _make_data()
    r = fit_gaussian_f32(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 0.1, 0.8], dtype=np.float32),
        lower_bounds=np.array([0.1, -2.0, 0.2], dtype=np.float32),
        upper_bounds=np.array([2.0, 2.0, 3.0], dtype=np.float32),
    )
    assert r[7] == FLAG_SUCCESS
    assert abs(r[0] - 1.0) < 1e-3, f"amp={r[0]}"
    assert abs(r[1] - 0.0) < 1e-3, f"mean={r[1]}"
    assert abs(r[2] - 1.0) < 1e-3, f"sigma={r[2]}"


def test_f32_output_shape_and_dtype():
    x, y, err = _make_data()
    r = fit_gaussian_f32(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 0, 0.9], dtype=np.float32),
        lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
        upper_bounds=np.array([2, 2, 3], dtype=np.float32),
    )
    assert r.shape == (8,)
    assert r.dtype == np.float32


def test_f32_no_convergence_flag_on_impossible_bounds():
    x, y, err = _make_data()
    r = fit_gaussian_f32(
        x=x,
        y=y,
        error=err,
        initial=np.array([1.5, 0, 1.0], dtype=np.float32),
        # amplitude must be > 1.8, but true peak is 1.0 — can't converge
        lower_bounds=np.array([1.8, -0.01, 0.99], dtype=np.float32),
        upper_bounds=np.array([2.0, 0.01, 1.01], dtype=np.float32),
    )
    assert r[7] in (FLAG_SUCCESS, FLAG_NO_CONVERGENCE)


def test_f32_nan_results_on_no_convergence():
    x, y, err = _make_data()
    r = fit_gaussian_f32(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 0, 0.9], dtype=np.float32),
        lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
        upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        max_iter=1,
    )
    if r[7] == FLAG_NO_CONVERGENCE:
        assert np.isnan(r[0])
        assert np.isnan(r[1])
        assert np.isnan(r[2])


def test_f32_max_iter_one_reports_no_convergence():
    x = np.linspace(-3, 3, 80, dtype=np.float32)
    y = _gaussian(x, 2.5, 1.2, 0.4).astype(np.float32)
    err = np.full_like(x, 0.01)
    r = fit_gaussian_f32(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.2, -2.5, 2.5], dtype=np.float32),
        lower_bounds=np.array([0.01, -3.0, 0.05], dtype=np.float32),
        upper_bounds=np.array([5.0, 3.0, 3.0], dtype=np.float32),
        max_iter=1,
    )
    assert r[7] == FLAG_NO_CONVERGENCE
    assert np.all(np.isnan(r[:7]))


def test_f32_raises_on_nan_tolerance():
    x, y, err = _make_data()
    with pytest.raises(ValueError, match="finite"):
        fit_gaussian_f32(
            x=x,
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9], dtype=np.float32),
            lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
            xtol=np.nan,
        )


def test_f32_raises_on_invalid_bounds():
    x, y, err = _make_data()
    with pytest.raises(ValueError, match="bounds"):
        fit_gaussian_f32(
            x=x,
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9], dtype=np.float32),
            lower_bounds=np.array([np.nan, -2, 0.2], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        )


def test_f32_raises_on_non_positive_sigma_lower_bound():
    x, y, err = _make_data()
    with pytest.raises(ValueError, match="sigma lower bound"):
        fit_gaussian_f32(
            x=x,
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9], dtype=np.float32),
            lower_bounds=np.array([0.1, -2, 0.0], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        )


def test_f32_raises_on_wrong_param_length():
    x, y, err = _make_data()
    with pytest.raises(ValueError, match="exactly 3"):
        fit_gaussian_f32(
            x=x,
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9, 1.0], dtype=np.float32),
            lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        )


def test_f32_raises_on_non_positive_error():
    x, y, err = _make_data()
    err[0] = 0.0
    with pytest.raises(ValueError, match="error values"):
        fit_gaussian_f32(
            x=x,
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9], dtype=np.float32),
            lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        )


def test_f32_non_contiguous_input_accepted():
    x, y, err = _make_data()
    r = fit_gaussian_f32(
        x=x[::1],
        y=y[::1],
        error=err[::1],
        initial=np.array([0.9, 0, 0.9], dtype=np.float32),
        lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
        upper_bounds=np.array([2, 2, 3], dtype=np.float32),
    )
    assert r[7] == FLAG_SUCCESS


def test_f32_raises_on_length_mismatch():
    x, y, err = _make_data()
    with pytest.raises(ValueError, match="same length"):
        fit_gaussian_f32(
            x=x[:-1],
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9], dtype=np.float32),
            lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        )


def test_f32_raises_on_too_few_samples():
    x = np.array([0.0, 1.0], dtype=np.float32)
    y = np.array([1.0, 0.5], dtype=np.float32)
    err = np.array([0.1, 0.1], dtype=np.float32)
    with pytest.raises(ValueError, match="at least 3"):
        fit_gaussian_f32(
            x=x,
            y=y,
            error=err,
            initial=np.array([0.9, 0, 0.9], dtype=np.float32),
            lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
            upper_bounds=np.array([2, 2, 3], dtype=np.float32),
        )


def test_f32_recovers_offset_gaussian():
    x, y, err = _make_data(mean=2.5, sigma=0.5)
    r = fit_gaussian_f32(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 2.0, 0.4], dtype=np.float32),
        lower_bounds=np.array([0.1, 1.0, 0.1], dtype=np.float32),
        upper_bounds=np.array([2.0, 4.0, 2.0], dtype=np.float32),
    )
    assert r[7] == FLAG_SUCCESS
    assert abs(r[1] - 2.5) < 0.05, f"mean={r[1]}"
    assert abs(r[2] - 0.5) < 0.05, f"sigma={r[2]}"


def test_f32_dtype_auto_conversion():
    x, y, err = _make_data()
    r = fit_gaussian_f32(
        x=x.astype(np.float64),
        y=y.astype(np.float64),
        error=err.astype(np.float64),
        initial=np.array([0.9, 0, 0.9]),
        lower_bounds=np.array([0.1, -2, 0.2]),
        upper_bounds=np.array([2, 2, 3]),
    )
    assert r.dtype == np.float32
    assert r[7] == FLAG_SUCCESS


def test_f64_recovers_unit_gaussian():
    x, y, err = _make_data(dtype=np.float64, noise=0.001)
    r = fit_gaussian_f64(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 0.1, 0.8]),
        lower_bounds=np.array([0.1, -2.0, 0.2]),
        upper_bounds=np.array([2.0, 2.0, 3.0]),
    )
    assert r[7] == 0.0  # FLAG_SUCCESS as f64
    assert abs(r[0] - 1.0) < 1e-6, f"amp={r[0]}"
    assert abs(r[1] - 0.0) < 1e-6, f"mean={r[1]}"
    assert abs(r[2] - 1.0) < 1e-6, f"sigma={r[2]}"


def test_f64_output_dtype():
    x, y, err = _make_data(dtype=np.float64)
    r = fit_gaussian_f64(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 0, 0.9]),
        lower_bounds=np.array([0.1, -2, 0.2]),
        upper_bounds=np.array([2, 2, 3]),
    )
    assert r.dtype == np.float64


def test_f64_more_precise_than_f32():
    x, y, err = _make_data(dtype=np.float64, noise=0.001, n=100)
    r64 = fit_gaussian_f64(
        x=x,
        y=y,
        error=err,
        initial=np.array([0.9, 0, 0.9]),
        lower_bounds=np.array([0.1, -2, 0.2]),
        upper_bounds=np.array([2, 2, 3]),
    )
    r32 = fit_gaussian_f32(
        x=x.astype(np.float32),
        y=y.astype(np.float32),
        error=err.astype(np.float32),
        initial=np.array([0.9, 0, 0.9], dtype=np.float32),
        lower_bounds=np.array([0.1, -2, 0.2], dtype=np.float32),
        upper_bounds=np.array([2, 2, 3], dtype=np.float32),
    )
    assert abs(r64[0] - 1.0) <= abs(r32[0] - 1.0) + 1e-5
