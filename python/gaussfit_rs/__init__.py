"""
gaussfit-rs.
"""

from importlib.metadata import version as _get_version

from .fitting import (
    FLAG_NO_CONVERGENCE,
    FLAG_NO_LOCAL_MAX,
    FLAG_SUCCESS,
    QUALITY_PEGGED,
    QUALITY_UNCONSTRAINED,
    QUALITY_ZERO_ERROR,
    FitResult,
    fit_gaussian,
    fit_single_spectrum,
    fit_spectra_batch,
)

__version__ = _get_version("gaussfit-rs")

__all__ = [
    "FLAG_NO_CONVERGENCE",
    "FLAG_NO_LOCAL_MAX",
    "FLAG_SUCCESS",
    "QUALITY_PEGGED",
    "QUALITY_UNCONSTRAINED",
    "QUALITY_ZERO_ERROR",
    "FitResult",
    "__version__",
    "fit_gaussian",
    "fit_single_spectrum",
    "fit_spectra_batch",
]
