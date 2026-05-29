mod api;
mod gaussian;
mod spectrum;

#[cfg(test)]
mod tests;

use pyo3::prelude::{pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;

pub use gaussian::{fit_gaussian_bounded_with_config, FitConfig, FitOutcome};
pub use spectrum::{fit_single_spectrum_core, FitSingleSpectrumResult};

/// Fit-status flag: optimisation converged successfully.
const FLAG_SUCCESS: f32 = 0.0;
/// Fit-status flag: no spectral peak found within the search window.
const FLAG_NO_LOCAL_MAX: f32 = 1.0;
/// Fit-status flag: LM optimiser did not converge within `max_iter` iterations.
const FLAG_NO_CONVERGENCE: f32 = 2.0;

const XTOL: f32 = 1.0e-6;
const FTOL: f32 = 1.0e-6;
const GTOL: f32 = 1.0e-6;
const MAX_ITER: usize = 2000;
const MAX_ITER_PY: isize = MAX_ITER as isize;

#[pymodule]
#[pyo3(name = "_gaussfit_rs")]
fn gaussfit_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(api::fit_single_spectrum, module)?)?;
    module.add_function(wrap_pyfunction!(api::fit_gaussian_f32, module)?)?;
    module.add_function(wrap_pyfunction!(api::fit_gaussian_f64, module)?)?;
    module.add_function(wrap_pyfunction!(api::fit_spectra_batch, module)?)?;
    module.add_function(wrap_pyfunction!(api::fit_spectra_batch_guided, module)?)?;
    module.add("FLAG_SUCCESS", FLAG_SUCCESS)?;
    module.add("FLAG_NO_LOCAL_MAX", FLAG_NO_LOCAL_MAX)?;
    module.add("FLAG_NO_CONVERGENCE", FLAG_NO_CONVERGENCE)?;
    Ok(())
}
