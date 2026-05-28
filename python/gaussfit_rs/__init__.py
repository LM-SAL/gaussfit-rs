"""
gaussfit-rs.
"""

from .fitting import (
    FLAG_NO_CONVERGENCE,
    FLAG_NO_LOCAL_MAX,
    FLAG_SUCCESS,
    FitResult,
    fit_gaussian_f32,
    fit_gaussian_f64,
    fit_single_spectrum,
    fit_spectra_batch,
    fit_spectra_batch_guided,
)
from .version import version as __version__

__all__ = [
    "FLAG_NO_CONVERGENCE",
    "FLAG_NO_LOCAL_MAX",
    "FLAG_SUCCESS",
    "FitResult",
    "__version__",
    "fit_gaussian_f32",
    "fit_gaussian_f64",
    "fit_single_spectrum",
    "fit_spectra_batch",
    "fit_spectra_batch_guided",
]
