"""
Python API for the gaussfit-rs Rust backend.

Output array layout — 8 elements, dtype float32:

======  ===============  =====================================================
Index   Field            Description
======  ===============  =====================================================
0       amplitude        Fitted peak amplitude (same units as input spectrum)
1       velocity         Line-centre velocity [km/s]
2       sigma            Gaussian width 1-σ [km/s]
3       amplitude_err    1-σ uncertainty on amplitude
4       velocity_err     1-σ uncertainty on velocity [km/s]
5       sigma_err        1-σ uncertainty on sigma [km/s]
6       reduced_chi2     Reduced χ² of best fit
7       flag             Status: FLAG_SUCCESS / FLAG_NO_LOCAL_MAX / FLAG_NO_CONVERGENCE
======  ===============  =====================================================

All fields except *flag* are ``NaN`` when ``flag != FLAG_SUCCESS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ._gaussfit_rs import fit_gaussian_f32 as _fit_gaussian_f32
from ._gaussfit_rs import fit_gaussian_f64 as _fit_gaussian_f64
from ._gaussfit_rs import fit_single_spectrum as _fit_single_spectrum
from ._gaussfit_rs import fit_spectra_batch as _fit_spectra_batch
from ._gaussfit_rs import fit_spectra_batch_guided as _fit_spectra_batch_guided

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "FLAG_NO_CONVERGENCE",
    "FLAG_NO_LOCAL_MAX",
    "FLAG_SUCCESS",
    "FitResult",
    "fit_gaussian_f32",
    "fit_gaussian_f64",
    "fit_single_spectrum",
    "fit_spectra_batch",
    "fit_spectra_batch_guided",
]


class FitResult(NamedTuple):
    """
    Named result from a single Gaussian fit.

    Attributes
    ----------
    amplitude : float
        Fitted peak amplitude (same units as input spectrum).
    velocity : float
        Fitted line-centre velocity [km/s].
    sigma : float
        Fitted Gaussian width (1-sigma) [km/s].
    amplitude_err : float
        1-sigma uncertainty on *amplitude*.
    velocity_err : float
        1-sigma uncertainty on *velocity* [km/s].
    sigma_err : float
        1-sigma uncertainty on *sigma* [km/s].
    reduced_chi2 : float
        Reduced chi-squared of the best fit.
    flag : float
        Fit status code: :data:`FLAG_SUCCESS`, :data:`FLAG_NO_LOCAL_MAX`, or
        :data:`FLAG_NO_CONVERGENCE`.

    Notes
    -----
    All fields except *flag* are ``nan`` when ``flag != FLAG_SUCCESS``.
    """

    amplitude: float
    velocity: float
    sigma: float
    amplitude_err: float
    velocity_err: float
    sigma_err: float
    reduced_chi2: float
    flag: float

    @classmethod
    def from_array(cls, arr: NDArray[np.float32]) -> FitResult:
        """
        Construct a :class:`FitResult` from an 8-element fit array.
        """
        return cls(*arr[:8].tolist())

    @property
    def converged(self) -> bool:
        """
        ``True`` if the fit converged (``flag == FLAG_SUCCESS``).
        """
        return self.flag == FLAG_SUCCESS


# Fit-status flag values returned in output[..., 7]
FLAG_SUCCESS: float = 0.0
FLAG_NO_LOCAL_MAX: float = 1.0
FLAG_NO_CONVERGENCE: float = 2.0


def fit_single_spectrum(
    *,
    spectrum: NDArray[np.float32],
    dopp_slit: NDArray[np.float32],
    spec_noise: NDArray[np.float32],
    guide_velocity: float,
    velocity_range: float,
    npix: int,
    npix_slack: int,
    dv: float,
    width_min: float,
    n_pixels: int | None = None,
    amplitude_rel_min: float,
    amplitude_rel_max: float,
    width_max: float,
    width_guess: float,
    xtol: float = 1.0e-6,
    ftol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    max_iter: int = 2000,
) -> tuple[NDArray[np.float32], tuple[int, int]]:
    """
    Fit a single Gaussian to one spectrum using the Rust backend.

    The function searches for the brightest peak within ``guide_velocity ±
    velocity_range`` (km/s), normalises by that peak, then fits a bounded
    Levenberg-Marquardt Gaussian using the surrounding ``2*npix`` pixel window.

    Parameters
    ----------
    spectrum:
        Spectral data, 1-D float32.  All three arrays are trimmed to
        *n_pixels* before processing; pass pre-sliced arrays to avoid
        any copy overhead.
    dopp_slit:
        Doppler velocity at each pixel [km/s], same length as *spectrum*.
    spec_noise:
        1-sigma noise estimate at each pixel (same units as *spectrum*).
    guide_velocity:
        Expected line-centre velocity [km/s] used to locate the peak.
    velocity_range:
        Half-width of the velocity search window around *guide_velocity* [km/s].
    npix:
        Half-width of the fitting window around the peak (pixels).
    npix_slack:
        Extra pixels added to the search window for the peak-finding step.
    dv:
        Pixel scale [km/s per pixel]; used to compute velocity bounds.
    width_min:
        Minimum allowed Gaussian sigma [km/s].
    n_pixels:
        Number of pixels to use from the start of each array.  Defaults to
        ``len(spectrum)``.  Pass this when the arrays are longer than the
        intended fitting window.
    amplitude_rel_min:
        Minimum allowed amplitude relative to the detected peak (e.g. 0.1).
    amplitude_rel_max:
        Maximum allowed amplitude relative to the detected peak (e.g. 2.0).
    width_max:
        Maximum allowed Gaussian sigma [km/s].
    width_guess:
        Initial guess for the Gaussian sigma [km/s].
    xtol:
        Convergence tolerance on the parameter step size (default 1e-6).
    ftol:
        Convergence tolerance on the cost-function change (default 1e-6).
    gtol:
        Convergence tolerance on the gradient norm (default 1e-6).
    max_iter:
        Maximum LM iterations (default 2000).

    Returns
    -------
    fit_results : ndarray, shape (8,), float32
        ``[amplitude, velocity, sigma, amplitude_err, velocity_err, sigma_err,
        reduced_chi2, flag]``.  See module docstring for details.
    (i_left, i_right) : tuple[int, int]
        Pixel indices of the fitting window used (half-open, ``[i_left, i_right)``).
    """
    spectrum = np.ascontiguousarray(spectrum, dtype=np.float32)
    n_px = int(n_pixels) if n_pixels is not None else len(spectrum)
    return _fit_single_spectrum(
        spectrum,
        np.ascontiguousarray(dopp_slit, dtype=np.float32),
        np.ascontiguousarray(spec_noise, dtype=np.float32),
        float(guide_velocity),
        float(velocity_range),
        int(npix),
        int(npix_slack),
        float(dv),
        float(width_min),
        n_px,
        float(amplitude_rel_min),
        float(amplitude_rel_max),
        float(width_max),
        float(width_guess),
        float(xtol),
        float(ftol),
        float(gtol),
        int(max_iter),
    )


def fit_spectra_batch(
    *,
    spectra: NDArray[np.float32],
    dopp_slit: NDArray[np.float32],
    spec_noise: NDArray[np.float32],
    guide_velocity: float,
    velocity_range: float,
    npix: int,
    npix_slack: int,
    dv: float,
    width_min: float,
    n_pixels: int | None = None,
    amplitude_rel_min: float,
    amplitude_rel_max: float,
    width_max: float,
    width_guess: float,
    xtol: float = 1.0e-6,
    ftol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    max_iter: int = 2000,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Fit Gaussians to N spectra in parallel using all available CPU cores.

    Equivalent to calling :func:`fit_single_spectrum` for each row of
    *spectra*, but processed concurrently in Rust via Rayon.  The GIL is
    released for the duration of the computation.

    Parameters
    ----------
    spectra:
        Spectral data, shape ``(N, M)``, C-contiguous float32.
    dopp_slit:
        Doppler velocity grid [km/s], shape ``(n_pixels,)`` — shared across
        all spectra (same wavelength solution assumed).
    spec_noise:
        Per-spaxel noise, shape ``(N, M)``, C-contiguous float32.
    n_pixels:
        Number of columns to use from the start of *spectra*.  Defaults to
        the full column count ``spectra.shape[1]``.
    guide_velocity, velocity_range, npix, npix_slack, dv, width_min,
    amplitude_rel_min, amplitude_rel_max, width_max, width_guess,
    xtol, ftol, gtol, max_iter:
        Same as :func:`fit_single_spectrum`.

    Returns
    -------
    fit_results : ndarray, shape (N, 8), float32
        One result row per input spectrum.  Column layout same as
        :func:`fit_single_spectrum`.
    indices : ndarray, shape (N, 2), int32
        ``[:, 0]`` = i_left, ``[:, 1]`` = i_right for each spectrum.

    """
    spectra = np.ascontiguousarray(spectra, dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2-D (N, M) array")
    n_px = int(n_pixels) if n_pixels is not None else spectra.shape[1]
    return _fit_spectra_batch(
        spectra,
        np.ascontiguousarray(dopp_slit, dtype=np.float32),
        np.ascontiguousarray(spec_noise, dtype=np.float32),
        float(guide_velocity),
        float(velocity_range),
        int(npix),
        int(npix_slack),
        float(dv),
        float(width_min),
        n_px,
        float(amplitude_rel_min),
        float(amplitude_rel_max),
        float(width_max),
        float(width_guess),
        float(xtol),
        float(ftol),
        float(gtol),
        int(max_iter),
    )


def fit_spectra_batch_guided(
    *,
    spectra: NDArray[np.float32],
    dopp_slit: NDArray[np.float32],
    spec_noise: NDArray[np.float32],
    guide_velocities: NDArray[np.float32],
    velocity_range: float,
    npix: int,
    npix_slack: int,
    dv: float,
    width_min: float,
    n_pixels: int | None = None,
    amplitude_rel_min: float,
    amplitude_rel_max: float,
    width_max: float,
    width_guess: float,
    xtol: float = 1.0e-6,
    ftol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    max_iter: int = 2000,
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Fit Gaussians to N spectra in parallel with one guide velocity per row.

    Same as :func:`fit_spectra_batch`, except ``guide_velocities`` has shape ``(N,)`` and supplies
    the expected line-centre velocity for each spectrum.
    """
    spectra = np.ascontiguousarray(spectra, dtype=np.float32)
    if spectra.ndim != 2:
        raise ValueError("spectra must be a 2-D (N, M) array")
    n_px = int(n_pixels) if n_pixels is not None else spectra.shape[1]
    return _fit_spectra_batch_guided(
        spectra,
        np.ascontiguousarray(dopp_slit, dtype=np.float32),
        np.ascontiguousarray(spec_noise, dtype=np.float32),
        np.ascontiguousarray(guide_velocities, dtype=np.float32),
        float(velocity_range),
        int(npix),
        int(npix_slack),
        float(dv),
        float(width_min),
        n_px,
        float(amplitude_rel_min),
        float(amplitude_rel_max),
        float(width_max),
        float(width_guess),
        float(xtol),
        float(ftol),
        float(gtol),
        int(max_iter),
    )


def fit_gaussian_f32(
    *,
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    error: NDArray[np.float32],
    initial: NDArray[np.float32],
    lower_bounds: NDArray[np.float32],
    upper_bounds: NDArray[np.float32],
    xtol: float = 1.0e-6,
    ftol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    max_iter: int = 2000,
) -> NDArray[np.float32]:
    """
    Fit a bounded Gaussian to (x, y ± error) data using the Rust backend.

    Uses a Levenberg-Marquardt solver with box constraints on all three
    parameters (amplitude, mean, sigma).

    Parameters
    ----------
    x:
        Independent variable, shape ``(N,)``, float32.  Must have at least 3
        elements.
    y:
        Observed values, shape ``(N,)``, float32.
    error:
        1-sigma uncertainties on *y*, shape ``(N,)``, float32.  All values
        must be finite and positive.
    initial:
        Starting guess ``[amplitude, mean, sigma]``, shape ``(3,)``.
    lower_bounds:
        Lower bounds ``[amp_min, mean_min, sigma_min]``, shape ``(3,)``.
    upper_bounds:
        Upper bounds ``[amp_max, mean_max, sigma_max]``, shape ``(3,)``.
    xtol:
        Convergence tolerance on the parameter step size.
    ftol:
        Convergence tolerance on the cost-function change.
    gtol:
        Convergence tolerance on the gradient norm.
    max_iter:
        Maximum number of outer LM iterations.

    Returns
    -------
    ndarray, shape (8,), float32
        ``[amplitude, mean, sigma, amplitude_err, mean_err, sigma_err,
        reduced_chi2, flag]``.  *flag* is :data:`FLAG_SUCCESS` (0) on
        convergence or :data:`FLAG_NO_CONVERGENCE` (2) otherwise.
    """
    return _fit_gaussian_f32(
        np.ascontiguousarray(x, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
        np.ascontiguousarray(error, dtype=np.float32),
        np.ascontiguousarray(initial, dtype=np.float32),
        np.ascontiguousarray(lower_bounds, dtype=np.float32),
        np.ascontiguousarray(upper_bounds, dtype=np.float32),
        float(xtol),
        float(ftol),
        float(gtol),
        int(max_iter),
    )


def fit_gaussian_f64(
    *,
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    error: NDArray[np.float64],
    initial: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
    xtol: float = 1.0e-10,
    ftol: float = 1.0e-10,
    gtol: float = 1.0e-10,
    max_iter: int = 2000,
) -> NDArray[np.float64]:
    """
    Fit a bounded Gaussian to (x, y ± error) data in double precision.

    Identical to :func:`fit_gaussian_f32` but operates on ``float64`` arrays
    throughout.  Use when the data span a wide dynamic range or when f32
    rounding would be significant.

    Parameters and return value have the same structure as
    :func:`fit_gaussian_f32`, but all arrays are ``float64``.
    Default tolerances are tighter (1e-10) to exploit the extra precision.
    """
    return _fit_gaussian_f64(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(error, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(lower_bounds, dtype=np.float64),
        np.ascontiguousarray(upper_bounds, dtype=np.float64),
        float(xtol),
        float(ftol),
        float(gtol),
        int(max_iter),
    )
