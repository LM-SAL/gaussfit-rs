//! A Python-owned Rayon pool for one call or stage.
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::num::NonZeroUsize;

#[pyclass(frozen, name = "_WorkerPool", module = "gaussfit_rs._gaussfit_rs")]
pub(crate) struct WorkerPool(pub(crate) ThreadPool);

#[pymethods]
impl WorkerPool {
    #[getter]
    fn num_threads(&self) -> usize {
        self.0.current_num_threads()
    }

    #[new]
    fn new(workers: NonZeroUsize) -> PyResult<Self> {
        ThreadPoolBuilder::new()
            .num_threads(workers.get())
            .build()
            .map(Self)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }
}

pub(crate) fn run<T: Send>(pool: Option<&WorkerPool>, work: impl FnOnce() -> T + Send) -> T {
    match pool {
        Some(pool) => pool.0.install(work),
        None => work(),
    }
}
