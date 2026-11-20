mod api;
mod gaussian;
mod spectrum;
mod workers;

#[cfg(test)]
mod tests;

use pyo3::prelude::{pymodule, Bound, PyModule, PyResult, Python};
use pyo3::types::PyModuleMethods;
use pyo3::wrap_pyfunction;

const FLAG_SUCCESS: f32 = 0.0;
const FLAG_NO_LOCAL_MAX: f32 = 1.0;
const FLAG_NO_CONVERGENCE: f32 = 2.0;

/// The velocity or width error spans its allowed interval.
const QUALITY_UNCONSTRAINED: u8 = 1;
/// A singular or exact fit can give a formal error of zero.
const QUALITY_ZERO_ERROR: u8 = 2;
/// A parameter reached a bound, which can also happen for a well-determined fit.
const QUALITY_PEGGED: u8 = 4;

const XTOL: f32 = 1.0e-6;
const FTOL: f32 = 1.0e-6;
const GTOL: f32 = 1.0e-6;
const MAX_ITER: usize = 2000;
const MAX_ITER_PY: isize = MAX_ITER as isize;

// Safe without the GIL: the only mutable state is the per-thread workspaces in gaussian.rs and
// spectrum.rs.
#[pymodule(gil_used = false)]
#[pyo3(name = "_gaussfit_rs")]
fn gaussfit_rs(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<workers::WorkerPool>()?;
    module.add_function(wrap_pyfunction!(api::fit_spectra_batch, module)?)?;
    module.add_function(wrap_pyfunction!(api::fit_gaussian, module)?)?;
    module.add("FLAG_SUCCESS", FLAG_SUCCESS)?;
    module.add("FLAG_NO_LOCAL_MAX", FLAG_NO_LOCAL_MAX)?;
    module.add("FLAG_NO_CONVERGENCE", FLAG_NO_CONVERGENCE)?;
    module.add("QUALITY_UNCONSTRAINED", QUALITY_UNCONSTRAINED)?;
    module.add("QUALITY_ZERO_ERROR", QUALITY_ZERO_ERROR)?;
    module.add("QUALITY_PEGGED", QUALITY_PEGGED)?;
    Ok(())
}
