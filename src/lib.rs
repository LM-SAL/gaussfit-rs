mod api;
mod gaussian;
mod spectrum;

#[cfg(test)]
mod tests;

use pyo3::prelude::{pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;

/// Fit-status flag: optimisation converged successfully.
const FLAG_SUCCESS: f32 = 0.0;
/// Fit-status flag: no spectral peak found within the search window.
const FLAG_NO_LOCAL_MAX: f32 = 1.0;
/// Fit-status flag: LM optimiser did not converge within `max_iter` iterations.
const FLAG_NO_CONVERGENCE: f32 = 2.0;

// Opt-in quality bits, reported in the ninth result column when `quality=True`
// (see "Opt-In Unconstrained-Fit Indicator" in docs/design-notes.rst). A
// consumer that wants the permissive "any of them" answer tests `!= 0`.
/// Quality bit: a parameter's error is not smaller than the velocity or width interval it was
/// bounded to, so that parameter is not constrained by the data.
const QUALITY_UNCONSTRAINED: u8 = 1;
/// Quality bit: a formal error is exactly zero, including singular or exact fits.
const QUALITY_ZERO_ERROR: u8 = 2;
/// Quality bit: a fitted parameter sits exactly on one of its bounds. Reported, not judged: it
/// fires on legitimate saturation as well as on degenerate fits.
const QUALITY_PEGGED: u8 = 4;

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
    module.add_function(wrap_pyfunction!(api::fit_spectra_batch_guided, module)?)?;
    module.add("FLAG_SUCCESS", FLAG_SUCCESS)?;
    module.add("FLAG_NO_LOCAL_MAX", FLAG_NO_LOCAL_MAX)?;
    module.add("FLAG_NO_CONVERGENCE", FLAG_NO_CONVERGENCE)?;
    module.add("QUALITY_UNCONSTRAINED", QUALITY_UNCONSTRAINED)?;
    module.add("QUALITY_ZERO_ERROR", QUALITY_ZERO_ERROR)?;
    module.add("QUALITY_PEGGED", QUALITY_PEGGED)?;
    Ok(())
}
