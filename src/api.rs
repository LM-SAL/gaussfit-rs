use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::{pyfunction, Bound, PyResult, Python};
use rayon::prelude::*;

use crate::gaussian::{fit_gaussian_bounded, FitConfig};
use crate::spectrum::fit_single_spectrum_core;
use crate::{FLAG_NO_CONVERGENCE, FLAG_SUCCESS, FTOL, GTOL, MAX_ITER_PY, XTOL};

fn validate_fit_config(xtol: f32, ftol: f32, gtol: f32, max_iter: isize) -> PyResult<FitConfig> {
    if !xtol.is_finite() || !ftol.is_finite() || !gtol.is_finite() {
        return Err(PyValueError::new_err("xtol, ftol, and gtol must be finite"));
    }
    if xtol <= 0.0 || ftol <= 0.0 || gtol <= 0.0 || max_iter <= 0 {
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

/// Fit Gaussians to N spectra in parallel.
///
/// `spectra` and `spec_noise` are `(N, M)`, `dopp_slit` is `(n_slit, M)`, and
/// row `i` is fitted against `dopp_slit[slit_index[i]]` (row 0 when
/// `slit_index` is None, which requires a single Doppler row) with guide
/// velocity `guide_velocities[i]`. Each result row is `[amp, vel, sigma,
/// amp_err, vel_err, sig_err, reduced_chi2, flag, quality]`, the ninth column
/// being the quality bits (0 for failed fits); the second array holds the `(i_left,
/// i_right)` fit window per row and the third the solver counts `[n_iter,
/// n_fev]` (-1 unless the fit converged). `flag` is `FLAG_SUCCESS`, `FLAG_NO_LOCAL_MAX`
/// (no positive peak in the search window, including a non-finite guide, or
/// fewer than 3 valid samples) or `FLAG_NO_CONVERGENCE`; all other fields are
/// NaN unless the flag is success.
#[pyfunction]
#[pyo3(signature = (spectra, dopp_slit, spec_noise, guide_velocities, velocity_range, npix,
    npix_slack, dv, width_min, amplitude_rel_min, amplitude_rel_max, width_max, width_guess,
    slit_index=None, xtol=XTOL, ftol=FTOL, gtol=GTOL, max_iter=MAX_ITER_PY))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn fit_spectra_batch<'py>(
    py: Python<'py>,
    spectra: PyReadonlyArray2<'py, f32>,
    dopp_slit: PyReadonlyArray2<'py, f32>,
    spec_noise: PyReadonlyArray2<'py, f32>,
    guide_velocities: PyReadonlyArray1<'py, f32>,
    velocity_range: f32,
    npix: i32,
    npix_slack: i32,
    dv: f32,
    width_min: f32,
    amplitude_rel_min: f32,
    amplitude_rel_max: f32,
    width_max: f32,
    width_guess: f32,
    slit_index: Option<PyReadonlyArray1<'py, i32>>,
    xtol: f32,
    ftol: f32,
    gtol: f32,
    max_iter: isize,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<i32>>,
)> {
    validate_spectrum_options(
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

    let [n_spectra, n_pixels] = [spectra.shape()[0], spectra.shape()[1]];
    let [n_slit, dopp_width] = [dopp_slit.shape()[0], dopp_slit.shape()[1]];
    if dopp_width != n_pixels {
        return Err(PyValueError::new_err(
            "dopp_slit columns must match spectra columns",
        ));
    }
    if spec_noise.shape() != [n_spectra, n_pixels] {
        return Err(PyValueError::new_err(
            "spec_noise shape must match spectra shape",
        ));
    }
    if guide_velocities.len() != n_spectra {
        return Err(PyValueError::new_err(
            "guide_velocities length must match spectra rows",
        ));
    }
    let slit_data = match &slit_index {
        Some(slit_index) => {
            if slit_index.len() != n_spectra {
                return Err(PyValueError::new_err(
                    "slit_index length must match spectra rows",
                ));
            }
            let slit_data = slit_index.as_slice().map_err(|_| {
                PyValueError::new_err("slit_index must be a contiguous int32 array")
            })?;
            if slit_data
                .iter()
                .any(|&slit| slit < 0 || slit as usize >= n_slit)
            {
                return Err(PyValueError::new_err(
                    "slit_index values must be in [0, dopp_slit rows)",
                ));
            }
            Some(slit_data)
        }
        None if n_slit == 1 => None,
        None => {
            return Err(PyValueError::new_err(
                "slit_index is required when dopp_slit has more than one row",
            ))
        }
    };

    let spectra_data = spectra
        .as_slice()
        .map_err(|_| PyValueError::new_err("spectra must be a C-contiguous float32 array"))?;
    let dopp_data = dopp_slit
        .as_slice()
        .map_err(|_| PyValueError::new_err("dopp_slit must be a C-contiguous float32 array"))?;
    let noise_data = spec_noise
        .as_slice()
        .map_err(|_| PyValueError::new_err("spec_noise must be a C-contiguous float32 array"))?;
    let guide_data = guide_velocities.as_slice().map_err(|_| {
        PyValueError::new_err("guide_velocities must be a contiguous float32 array")
    })?;

    let stride = 9;
    let mut fit_values = vec![f32::NAN; n_spectra * stride];
    let mut idx_values = vec![0i32; n_spectra * 2];
    let mut meta_values = vec![-1i32; n_spectra * 2];

    py.detach(|| {
        fit_values
            .par_chunks_mut(stride)
            .zip(idx_values.par_chunks_mut(2))
            .zip(meta_values.par_chunks_mut(2))
            .enumerate()
            .for_each(|(i, ((fit_row, idx_row), meta_row))| {
                let row = i * n_pixels..(i + 1) * n_pixels;
                let slit = slit_data.map_or(0, |slit_data| slit_data[i] as usize);
                let result = fit_single_spectrum_core(
                    &spectra_data[row.clone()],
                    &dopp_data[slit * n_pixels..(slit + 1) * n_pixels],
                    &noise_data[row],
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
                fit_row[..8].copy_from_slice(&result.fit_results);
                fit_row[8] = f32::from(result.quality);
                idx_row[0] = result.i_left;
                idx_row[1] = result.i_right;
                meta_row[0] = result.n_iter;
                meta_row[1] = result.n_fev;
            })
    });

    let fit_out = Array2::<f32>::from_shape_vec((n_spectra, stride), fit_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build fit result array"))?;
    let idx_out = Array2::<i32>::from_shape_vec((n_spectra, 2), idx_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build index result array"))?;
    let meta_out = Array2::<i32>::from_shape_vec((n_spectra, 2), meta_values)
        .map_err(|_| PyRuntimeError::new_err("failed to build solver count array"))?;

    Ok((
        fit_out.into_pyarray(py),
        idx_out.into_pyarray(py),
        meta_out.into_pyarray(py),
    ))
}

fn contiguous<'a>(array: &'a PyReadonlyArray1<'_, f32>, name: &str) -> PyResult<&'a [f32]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous float32 array")))
}

/// Fit a bounded Gaussian to arbitrary (x, y ± error) data.
///
/// Returns `[amp, mean, sigma, amp_err, mean_err, sig_err, reduced_chi2,
/// flag, n_iter, n_fev]`; `flag` is `FLAG_SUCCESS` or `FLAG_NO_CONVERGENCE`,
/// and the counts are -1 unless the fit converged.
#[pyfunction]
#[pyo3(signature = (x, y, error, initial, lower_bounds, upper_bounds, xtol=XTOL, ftol=FTOL,
    gtol=GTOL, max_iter=MAX_ITER_PY))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_gaussian<'py>(
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
    let (x, y, error) = (
        contiguous(&x, "x")?,
        contiguous(&y, "y")?,
        contiguous(&error, "error")?,
    );
    let (initial, lower, upper) = (
        contiguous(&initial, "initial")?,
        contiguous(&lower_bounds, "lower_bounds")?,
        contiguous(&upper_bounds, "upper_bounds")?,
    );
    if x.len() != y.len() || x.len() != error.len() {
        return Err(PyValueError::new_err(
            "x, y, and error must have the same length",
        ));
    }
    if x.len() < 3 {
        return Err(PyValueError::new_err("at least 3 samples are required"));
    }
    if initial.len() != 3 || lower.len() != 3 || upper.len() != 3 {
        return Err(PyValueError::new_err(
            "initial, lower_bounds, and upper_bounds must have exactly 3 elements",
        ));
    }
    if x.iter().chain(y).any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("x and y values must be finite"));
    }
    if error
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(PyValueError::new_err(
            "error values must be finite and positive",
        ));
    }
    if initial.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err("initial values must be finite"));
    }
    let bounds: [[f32; 2]; 3] = std::array::from_fn(|i| [lower[i], upper[i]]);
    for bound in bounds {
        if !bound[0].is_finite() || !bound[1].is_finite() || bound[0] >= bound[1] {
            return Err(PyValueError::new_err(
                "lower_bounds and upper_bounds must be finite and increasing",
            ));
        }
    }
    // The solver may evaluate the model anywhere inside the box, and the
    // Gaussian divides by sigma, so sigma = 0 must be unreachable.
    if bounds[2][0] <= 0.0 {
        return Err(PyValueError::new_err("sigma lower bound must be positive"));
    }
    let config = validate_fit_config(xtol, ftol, gtol, max_iter)?;

    let initial = [initial[0], initial[1], initial[2]];
    let mut row = [f32::NAN; 10];
    row[7] = FLAG_NO_CONVERGENCE;
    row[8] = -1.0;
    row[9] = -1.0;
    if let Some(outcome) = fit_gaussian_bounded(x, y, error, initial, bounds, config) {
        let dof = (x.len() as i32 - 3).max(1) as f32;
        row = [
            outcome.params[0],
            outcome.params[1],
            outcome.params[2],
            outcome.errors[0],
            outcome.errors[1],
            outcome.errors[2],
            outcome.bestnorm / dof,
            FLAG_SUCCESS,
            outcome.n_iter as f32,
            outcome.n_fev as f32,
        ];
    }
    Ok(row.to_vec().into_pyarray(py))
}
