use num_traits::Float;
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::{pyfunction, Bound, PyResult, Python};
use rayon::prelude::*;

use crate::gaussian::{fit_gaussian_bounded_with_config, FitConfig};
use crate::spectrum::fit_single_spectrum_core;
use crate::{FLAG_NO_CONVERGENCE, FLAG_SUCCESS, FTOL, GTOL, MAX_ITER_PY, XTOL};

fn validate_n_pixels(n_pixels: isize) -> PyResult<usize> {
    if n_pixels <= 0 {
        return Err(PyValueError::new_err("n_pixels must be positive"));
    }
    Ok(n_pixels as usize)
}

fn validate_fit_config<F: Float>(
    xtol: F,
    ftol: F,
    gtol: F,
    max_iter: isize,
) -> PyResult<FitConfig<F>> {
    let zero = F::zero();
    if !xtol.is_finite() || !ftol.is_finite() || !gtol.is_finite() {
        return Err(PyValueError::new_err("xtol, ftol, and gtol must be finite"));
    }
    if xtol <= zero || ftol <= zero || gtol <= zero || max_iter <= 0 {
        return Err(PyValueError::new_err(
            "xtol, ftol, gtol, and max_iter must be positive",
        ));
    }
    Ok(FitConfig {
        xtol,
        ftol,
        gtol,
        max_iter: max_iter as usize,
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_spectrum_options(
    guide_velocity: f32,
    velocity_range: f32,
    npix: i32,
    npix_slack: i32,
    dv: f32,
    width_min: f32,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
) -> PyResult<()> {
    let values = [
        guide_velocity,
        velocity_range,
        dv,
        width_min,
        amplitude_rel_min,
        amplitude_rel_max,
        width_max,
        width_guess,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("fit parameters must be finite"));
    }
    if velocity_range < 0.0 {
        return Err(PyValueError::new_err("velocity_range must be non-negative"));
    }
    if npix <= 0 {
        return Err(PyValueError::new_err("npix must be positive"));
    }
    if npix_slack < 0 {
        return Err(PyValueError::new_err("npix_slack must be non-negative"));
    }
    if dv <= 0.0 {
        return Err(PyValueError::new_err("dv must be positive"));
    }
    if width_min <= 0.0 || width_max <= width_min || width_guess <= 0.0 {
        return Err(PyValueError::new_err(
            "width_min, width_max, and width_guess must define positive increasing widths",
        ));
    }
    if amplitude_rel_min >= amplitude_rel_max {
        return Err(PyValueError::new_err(
            "amplitude_rel_min must be less than amplitude_rel_max",
        ));
    }
    Ok(())
}

fn validate_bounds<F: Float>(bounds: [[F; 2]; 3]) -> PyResult<()> {
    for bound in bounds {
        if !bound[0].is_finite() || !bound[1].is_finite() || bound[0] >= bound[1] {
            return Err(PyValueError::new_err(
                "lower_bounds and upper_bounds must be finite and increasing",
            ));
        }
    }
    Ok(())
}

fn validate_initial<F: Float>(initial: [F; 3]) -> PyResult<()> {
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("initial values must be finite"));
    }
    Ok(())
}

fn validate_fit_arrays<F: Float>(x: &[F], y: &[F], error: &[F]) -> PyResult<()> {
    if x.iter().any(|value| !value.is_finite()) || y.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("x and y values must be finite"));
    }
    if error
        .iter()
        .any(|value| !value.is_finite() || *value <= F::zero())
    {
        return Err(PyValueError::new_err(
            "error values must be finite and positive",
        ));
    }
    Ok(())
}

/// Fit a single Gaussian to one spectrum.
///
/// Returns an 8-element array `[amp, vel, sigma, amp_err, vel_err, sig_err,
/// reduced_chi2, flag]` and the `(i_left, i_right)` pixel window used.
#[pyfunction]
#[pyo3(signature = (spectrum, dopp_slit, spec_noise, guide_velocity, velocity_range,
    npix, npix_slack, dv, width_min, sg_xpixels, amplitude_rel_min, amplitude_rel_max,
    width_max, width_guess, xtol=XTOL, ftol=FTOL, gtol=GTOL, max_iter=MAX_ITER_PY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fit_single_spectrum<'py>(
    py: Python<'py>,
    spectrum: PyReadonlyArray1<'py, f32>,
    dopp_slit: PyReadonlyArray1<'py, f32>,
    spec_noise: PyReadonlyArray1<'py, f32>,
    guide_velocity: f32,
    velocity_range: f32,
    npix: i32,
    npix_slack: i32,
    dv: f32,
    width_min: f32,
    sg_xpixels: isize,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
    xtol: f32,
    ftol: f32,
    gtol: f32,
    max_iter: isize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, (i32, i32))> {
    let sg_xpixels = validate_n_pixels(sg_xpixels)?;
    validate_spectrum_options(
        guide_velocity,
        velocity_range,
        npix,
        npix_slack,
        dv,
        width_min,
        amplitude_rel_min,
        amplitude_rel_max,
        width_max,
        width_guess,
    )?;
    let config = validate_fit_config(xtol, ftol, gtol, max_iter)?;

    let spectrum = spectrum
        .as_slice()
        .map_err(|_| PyValueError::new_err("spectrum must be contiguous float32"))?;
    let dopp_slit = dopp_slit
        .as_slice()
        .map_err(|_| PyValueError::new_err("dopp_slit must be contiguous float32"))?;
    let spec_noise = spec_noise
        .as_slice()
        .map_err(|_| PyValueError::new_err("spec_noise must be contiguous float32"))?;

    if spectrum.len() < sg_xpixels || dopp_slit.len() < sg_xpixels || spec_noise.len() < sg_xpixels
    {
        return Err(PyValueError::new_err(
            "input arrays must be at least n_pixels long",
        ));
    }

    let result = fit_single_spectrum_core(
        &spectrum[..sg_xpixels],
        &dopp_slit[..sg_xpixels],
        &spec_noise[..sg_xpixels],
        guide_velocity,
        velocity_range,
        npix,
        npix_slack,
        dv,
        width_min,
        amplitude_rel_min,
        amplitude_rel_max,
        width_max,
        width_guess,
        config,
    );

    if result.status < 0 {
        return Err(PyRuntimeError::new_err(
            "Rust fastfit backend reported an internal error",
        ));
    }

    Ok((
        result.fit_results.to_vec().into_pyarray(py),
        (result.i_left, result.i_right),
    ))
}

/// Fit a bounded Gaussian to arbitrary (x, y ± error) data.
///
/// Returns an 8-element array `[amp, mean, sigma, amp_err, mean_err, sig_err,
/// reduced_chi2, flag]`. `flag` is `FLAG_SUCCESS` or `FLAG_NO_CONVERGENCE`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_gaussian_f32<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f32>,
    y: PyReadonlyArray1<'py, f32>,
    error: PyReadonlyArray1<'py, f32>,
    initial: PyReadonlyArray1<'py, f32>,
    lower_bounds: PyReadonlyArray1<'py, f32>,
    upper_bounds: PyReadonlyArray1<'py, f32>,
    xtol: f32,
    ftol: f32,
    gtol: f32,
    max_iter: isize,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let x = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be contiguous float32"))?;
    let y = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be contiguous float32"))?;
    let error = error
        .as_slice()
        .map_err(|_| PyValueError::new_err("error must be contiguous float32"))?;
    let initial = initial
        .as_slice()
        .map_err(|_| PyValueError::new_err("initial must be contiguous float32"))?;
    let lower_bounds = lower_bounds
        .as_slice()
        .map_err(|_| PyValueError::new_err("lower_bounds must be contiguous float32"))?;
    let upper_bounds = upper_bounds
        .as_slice()
        .map_err(|_| PyValueError::new_err("upper_bounds must be contiguous float32"))?;

    if x.len() != y.len() || x.len() != error.len() {
        return Err(PyValueError::new_err(
            "x, y, and error must have the same length",
        ));
    }
    if x.len() < 3 {
        return Err(PyValueError::new_err("at least 3 samples are required"));
    }
    if initial.len() < 3 || lower_bounds.len() < 3 || upper_bounds.len() < 3 {
        return Err(PyValueError::new_err(
            "initial, lower_bounds, and upper_bounds must have at least 3 elements",
        ));
    }

    let bounds = [
        [lower_bounds[0], upper_bounds[0]],
        [lower_bounds[1], upper_bounds[1]],
        [lower_bounds[2], upper_bounds[2]],
    ];
    let initial = [initial[0], initial[1], initial[2]];
    validate_fit_arrays(x, y, error)?;
    validate_initial(initial)?;
    validate_bounds(bounds)?;
    let config = validate_fit_config(xtol, ftol, gtol, max_iter)?;
    let fit_results = match fit_gaussian_bounded_with_config(x, y, error, initial, bounds, config) {
        Some(outcome) => {
            let dof = (x.len() as i32 - 3).max(1) as f32;
            [
                outcome.params[0],
                outcome.params[1],
                outcome.params[2],
                outcome.errors[0],
                outcome.errors[1],
                outcome.errors[2],
                outcome.bestnorm / dof,
                FLAG_SUCCESS,
            ]
        }
        None => {
            let mut failed = [f32::NAN; 8];
            failed[7] = FLAG_NO_CONVERGENCE;
            failed
        }
    };

    Ok(fit_results.to_vec().into_pyarray(py))
}

/// Fit a bounded Gaussian to (x, y ± error) data in **double precision**.
///
/// Identical to `fit_gaussian_f32` but operates on `float64` arrays and uses
/// 64-bit arithmetic throughout. Useful when the data span a wide dynamic range
/// or when f32 rounding would be significant.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_gaussian_f64<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    error: PyReadonlyArray1<'py, f64>,
    initial: PyReadonlyArray1<'py, f64>,
    lower_bounds: PyReadonlyArray1<'py, f64>,
    upper_bounds: PyReadonlyArray1<'py, f64>,
    xtol: f64,
    ftol: f64,
    gtol: f64,
    max_iter: isize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x
        .as_slice()
        .map_err(|_| PyValueError::new_err("x must be contiguous float64"))?;
    let y = y
        .as_slice()
        .map_err(|_| PyValueError::new_err("y must be contiguous float64"))?;
    let error = error
        .as_slice()
        .map_err(|_| PyValueError::new_err("error must be contiguous float64"))?;
    let initial = initial
        .as_slice()
        .map_err(|_| PyValueError::new_err("initial must be contiguous float64"))?;
    let lower_bounds = lower_bounds
        .as_slice()
        .map_err(|_| PyValueError::new_err("lower_bounds must be contiguous float64"))?;
    let upper_bounds = upper_bounds
        .as_slice()
        .map_err(|_| PyValueError::new_err("upper_bounds must be contiguous float64"))?;

    if x.len() != y.len() || x.len() != error.len() {
        return Err(PyValueError::new_err(
            "x, y, and error must have the same length",
        ));
    }
    if x.len() < 3 {
        return Err(PyValueError::new_err("at least 3 samples are required"));
    }
    if initial.len() < 3 || lower_bounds.len() < 3 || upper_bounds.len() < 3 {
        return Err(PyValueError::new_err(
            "initial, lower_bounds, and upper_bounds must have at least 3 elements",
        ));
    }

    let bounds = [
        [lower_bounds[0], upper_bounds[0]],
        [lower_bounds[1], upper_bounds[1]],
        [lower_bounds[2], upper_bounds[2]],
    ];
    let initial = [initial[0], initial[1], initial[2]];
    validate_fit_arrays(x, y, error)?;
    validate_initial(initial)?;
    validate_bounds(bounds)?;
    let config = validate_fit_config(xtol, ftol, gtol, max_iter)?;
    let fit_results: [f64; 8] =
        match fit_gaussian_bounded_with_config(x, y, error, initial, bounds, config) {
            Some(outcome) => {
                let dof = (x.len() as i32 - 3).max(1) as f64;
                [
                    outcome.params[0],
                    outcome.params[1],
                    outcome.params[2],
                    outcome.errors[0],
                    outcome.errors[1],
                    outcome.errors[2],
                    outcome.bestnorm / dof,
                    FLAG_SUCCESS as f64,
                ]
            }
            None => {
                let mut failed = [f64::NAN; 8];
                failed[7] = FLAG_NO_CONVERGENCE as f64;
                failed
            }
        };

    Ok(fit_results.to_vec().into_pyarray(py))
}

/// Fit Gaussians to N spectra in parallel using Rayon.
///
/// Accepts a C-contiguous `(N, M)` spectra array and a shared `(M,)` Doppler
/// velocity grid. Returns `(fit_results (N, 8), indices (N, 2))`. The GIL is
/// released for the parallel computation.
#[pyfunction]
#[pyo3(signature = (spectra, dopp_slit, spec_noise, guide_velocity, velocity_range,
    npix, npix_slack, dv, width_min, sg_xpixels, amplitude_rel_min, amplitude_rel_max,
    width_max, width_guess, xtol=XTOL, ftol=FTOL, gtol=GTOL, max_iter=MAX_ITER_PY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fit_spectra_batch<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArray2<'py, f32>,
    dopp_slit: PyReadonlyArray1<'py, f32>,
    spec_noise: PyReadonlyArray2<'py, f32>,
    guide_velocity: f32,
    velocity_range: f32,
    npix: i32,
    npix_slack: i32,
    dv: f32,
    width_min: f32,
    sg_xpixels: isize,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
    xtol: f32,
    ftol: f32,
    gtol: f32,
    max_iter: isize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
    let sg_xpixels = validate_n_pixels(sg_xpixels)?;
    validate_spectrum_options(
        guide_velocity,
        velocity_range,
        npix,
        npix_slack,
        dv,
        width_min,
        amplitude_rel_min,
        amplitude_rel_max,
        width_max,
        width_guess,
    )?;
    let config = validate_fit_config(xtol, ftol, gtol, max_iter)?;

    let n_spectra = spectra.shape()[0];
    let n_pixels = spectra.shape()[1];

    if n_pixels < sg_xpixels {
        return Err(PyValueError::new_err(
            "spectra columns must be at least sg_xpixels",
        ));
    }
    if spec_noise.shape()[0] != n_spectra || spec_noise.shape()[1] != n_pixels {
        return Err(PyValueError::new_err(
            "spec_noise shape must match spectra shape",
        ));
    }
    if dopp_slit.len() < sg_xpixels {
        return Err(PyValueError::new_err(
            "dopp_slit must be at least sg_xpixels long",
        ));
    }

    let spectra_data = spectra
        .as_slice()
        .map_err(|_| PyValueError::new_err("spectra must be a C-contiguous float32 array"))?;
    let dopp_data = dopp_slit
        .as_slice()
        .map_err(|_| PyValueError::new_err("dopp_slit must be a contiguous float32 array"))?;
    let noise_data = spec_noise
        .as_slice()
        .map_err(|_| PyValueError::new_err("spec_noise must be a C-contiguous float32 array"))?;

    let dopp_window = &dopp_data[..sg_xpixels];

    let mut fit_values = vec![f32::NAN; n_spectra * 8];
    let mut idx_values = vec![0i32; n_spectra * 2];

    py.detach(|| {
        fit_values
            .par_chunks_mut(8)
            .zip(idx_values.par_chunks_mut(2))
            .enumerate()
            .for_each(|(i, (fit_row, idx_row))| {
                let row_start = i * n_pixels;
                let result = fit_single_spectrum_core(
                    &spectra_data[row_start..row_start + sg_xpixels],
                    dopp_window,
                    &noise_data[row_start..row_start + sg_xpixels],
                    guide_velocity,
                    velocity_range,
                    npix,
                    npix_slack,
                    dv,
                    width_min,
                    amplitude_rel_min,
                    amplitude_rel_max,
                    width_max,
                    width_guess,
                    config,
                );
                fit_row.copy_from_slice(&result.fit_results);
                idx_row[0] = result.i_left;
                idx_row[1] = result.i_right;
            })
    });

    let fit_out = Array2::<f32>::from_shape_vec((n_spectra, 8), fit_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build fit result array"))?;
    let idx_out = Array2::<i32>::from_shape_vec((n_spectra, 2), idx_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build index result array"))?;

    Ok((fit_out.into_pyarray(py), idx_out.into_pyarray(py)))
}

/// Fit Gaussians to N spectra with one guide velocity per spectrum.
///
/// This is equivalent to `fit_spectra_batch`, except `guide_velocities[i]` is
/// used when fitting row `i`.
#[pyfunction]
#[pyo3(signature = (spectra, dopp_slit, spec_noise, guide_velocities, velocity_range,
    npix, npix_slack, dv, width_min, sg_xpixels, amplitude_rel_min, amplitude_rel_max,
    width_max, width_guess, xtol=XTOL, ftol=FTOL, gtol=GTOL, max_iter=MAX_ITER_PY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fit_spectra_batch_guided<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArray2<'py, f32>,
    dopp_slit: PyReadonlyArray1<'py, f32>,
    spec_noise: PyReadonlyArray2<'py, f32>,
    guide_velocities: PyReadonlyArray1<'py, f32>,
    velocity_range: f32,
    npix: i32,
    npix_slack: i32,
    dv: f32,
    width_min: f32,
    sg_xpixels: isize,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
    xtol: f32,
    ftol: f32,
    gtol: f32,
    max_iter: isize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
    let sg_xpixels = validate_n_pixels(sg_xpixels)?;
    validate_spectrum_options(
        0.0,
        velocity_range,
        npix,
        npix_slack,
        dv,
        width_min,
        amplitude_rel_min,
        amplitude_rel_max,
        width_max,
        width_guess,
    )?;
    let config = validate_fit_config(xtol, ftol, gtol, max_iter)?;

    let n_spectra = spectra.shape()[0];
    let n_pixels = spectra.shape()[1];

    if n_pixels < sg_xpixels {
        return Err(PyValueError::new_err(
            "spectra columns must be at least sg_xpixels",
        ));
    }
    if spec_noise.shape()[0] != n_spectra || spec_noise.shape()[1] != n_pixels {
        return Err(PyValueError::new_err(
            "spec_noise shape must match spectra shape",
        ));
    }
    if guide_velocities.len() != n_spectra {
        return Err(PyValueError::new_err(
            "guide_velocities length must match spectra rows",
        ));
    }
    if dopp_slit.len() < sg_xpixels {
        return Err(PyValueError::new_err(
            "dopp_slit must be at least sg_xpixels long",
        ));
    }

    let spectra_data = spectra
        .as_slice()
        .map_err(|_| PyValueError::new_err("spectra must be a C-contiguous float32 array"))?;
    let dopp_data = dopp_slit
        .as_slice()
        .map_err(|_| PyValueError::new_err("dopp_slit must be a contiguous float32 array"))?;
    let noise_data = spec_noise
        .as_slice()
        .map_err(|_| PyValueError::new_err("spec_noise must be a C-contiguous float32 array"))?;
    let guide_data = guide_velocities.as_slice().map_err(|_| {
        PyValueError::new_err("guide_velocities must be a contiguous float32 array")
    })?;

    if guide_data.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("guide_velocities must be finite"));
    }

    let dopp_window = &dopp_data[..sg_xpixels];

    let mut fit_values = vec![f32::NAN; n_spectra * 8];
    let mut idx_values = vec![0i32; n_spectra * 2];

    py.detach(|| {
        fit_values
            .par_chunks_mut(8)
            .zip(idx_values.par_chunks_mut(2))
            .enumerate()
            .for_each(|(i, (fit_row, idx_row))| {
                let row_start = i * n_pixels;
                let result = fit_single_spectrum_core(
                    &spectra_data[row_start..row_start + sg_xpixels],
                    dopp_window,
                    &noise_data[row_start..row_start + sg_xpixels],
                    guide_data[i],
                    velocity_range,
                    npix,
                    npix_slack,
                    dv,
                    width_min,
                    amplitude_rel_min,
                    amplitude_rel_max,
                    width_max,
                    width_guess,
                    config,
                );
                fit_row.copy_from_slice(&result.fit_results);
                idx_row[0] = result.i_left;
                idx_row[1] = result.i_right;
            })
    });

    let fit_out = Array2::<f32>::from_shape_vec((n_spectra, 8), fit_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build fit result array"))?;
    let idx_out = Array2::<i32>::from_shape_vec((n_spectra, 2), idx_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build index result array"))?;

    Ok((fit_out.into_pyarray(py), idx_out.into_pyarray(py)))
}
