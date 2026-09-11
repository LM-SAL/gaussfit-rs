"""
Python API for the gaussfit-rs Rust backend.

Result row layout, dtype float32:

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
8       quality          The QUALITY_* bits, 0 for failed fits
======  ===============  =====================================================

Fields 0-6 are ``NaN`` when ``flag != FLAG_SUCCESS``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ._gaussfit_rs import fit_gaussian as _fit_gaussian
from ._gaussfit_rs import fit_spectra_batch as _fit_spectra_batch

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "FLAG_NO_CONVERGENCE",
    "FLAG_NO_LOCAL_MAX",
    "FLAG_SUCCESS",
    "QUALITY_PEGGED",
    "QUALITY_UNCONSTRAINED",
    "QUALITY_ZERO_ERROR",
    "FitResult",
    "fit_gaussian",
    "fit_single_spectrum",
    "fit_spectra_batch",
]

_SPECTRA_NDIM = 2

# Fit-status flag values returned in column 7
FLAG_SUCCESS: float = 0.0
FLAG_NO_LOCAL_MAX: float = 1.0
FLAG_NO_CONVERGENCE: float = 2.0

# Quality bits, returned in column 8 as a float. Bits are reported
# separately instead of folded into one boolean because they answer different questions; a
# consumer that wants "any of them" tests ``fits[:, 8] != 0``.
QUALITY_UNCONSTRAINED: int = 1
"""
A parameter's error is not smaller than the velocity or width interval it was bounded to.
"""

QUALITY_ZERO_ERROR: int = 2
"""
A formal error is exactly zero, including singular or exact fits.
"""

QUALITY_PEGGED: int = 4
"""
A fitted parameter sits exactly on a bound: reported, not judged (saturation also sets it).
"""


class FitResult(NamedTuple):
    """
    A fit result row by name; ``FitResult.from_array(row)`` accepts 8 or 9 elements.

    ``quality`` is the bit mask of column 8, or 0 for the 8-element rows of :func:`fit_gaussian`.
    """

    amplitude: float
    velocity: float
    sigma: float
    amplitude_err: float
    velocity_err: float
    sigma_err: float
    reduced_chi2: float
    flag: float
    quality: int = 0

    @classmethod
    def from_array(cls, arr: NDArray[np.float32]) -> FitResult:
        """
        Build from a result row of 8 or 9 elements.
        """
        amplitude, velocity, sigma, amp_err, vel_err, sig_err, chi2, flag, *rest = arr.tolist()
        quality = int(rest[0]) if rest else 0
        return cls(amplitude, velocity, sigma, amp_err, vel_err, sig_err, chi2, flag, quality)

    @property
    def converged(self) -> bool:
        """
        ``True`` if the fit converged (``flag == FLAG_SUCCESS``).
        """
        return self.flag == FLAG_SUCCESS


def fit_spectra_batch(
    *,
    spectra: ArrayLike,
    dopp_slit: ArrayLike,
    spec_noise: ArrayLike,
    guide_velocities: ArrayLike,
    velocity_range: float,
    npix: int,
    npix_slack: int,
    dv: float,
    width_min: float,
    width_max: float,
    width_guess: float,
    amplitude_rel_min: float,
    amplitude_rel_max: float,
    slit_index: ArrayLike | None = None,
    xtol: float = 1.0e-6,
    ftol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    max_iter: int = 2000,
    meta: bool = False,
) -> (
    tuple[NDArray[np.float32], NDArray[np.int32]]
    | tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.int32]]
):
    """
    Fit a Gaussian to every row of ``spectra`` in parallel, releasing the GIL.

    For each row the brightest pixel within ``guide_velocity ± velocity_range`` (km/s) is found,
    the ``2 * npix + 1`` pixel window around it is normalised by that peak, and a bounded
    Levenberg-Marquardt Gaussian is fitted to it.

    Parameters
    ----------
    spectra:
        Spectral data, shape ``(N, M)``; converted to C-contiguous float32.
    dopp_slit:
        Doppler velocity of each pixel [km/s]: shape ``(M,)`` when every row shares one grid, or
        ``(n_slit, M)`` with ``slit_index`` naming the grid row of each spectrum.
    spec_noise:
        1-sigma noise per pixel (same units as *spectra*), shape ``(N, M)``.
    guide_velocities:
        Expected line-centre velocity [km/s] per row, shape ``(N,)``, or one scalar for all rows.
        A non-finite guide matches no pixel, so that row returns :data:`FLAG_NO_LOCAL_MAX`.
    velocity_range:
        Half-width of the peak search window around the guide [km/s].
    npix:
        Half-width of the fitting window around the peak (pixels).
    npix_slack:
        Extra pixels added to the search window when vetting the peak; a peak found only in the
        slack band is rejected.
    dv:
        Pixel scale [km/s per pixel]; sets the velocity bounds ``peak ± dv * npix``.
    width_min, width_max:
        Bounds on the Gaussian sigma [km/s].
    width_guess:
        Initial guess for sigma [km/s].
    amplitude_rel_min, amplitude_rel_max:
        Amplitude bounds relative to the detected peak (e.g. 0.1 and 2.0).
    slit_index:
        Row of ``dopp_slit`` to use for each spectrum, shape ``(N,)``, int32 values in
        ``[0, n_slit)``. Required when ``dopp_slit`` has more than one row. For a block with a
        slit axis, ``np.indices(flux.shape[:-1])[slit_axis].ravel()`` pairs with
        ``flux.reshape(-1, M)``.
    xtol, ftol, gtol:
        Convergence tolerances on the parameter step, the cost-function change and the gradient
        norm (default 1e-6 each).
    max_iter:
        Maximum Levenberg-Marquardt iterations (default 2000).
    meta:
        When True, a third return value carries the solver's ``[n_iter, n_fev]`` per row
        (accepted Levenberg-Marquardt iterations and model evaluations, Jacobian calls
        included) as int32, ``-1`` where the fit did not run or did not converge.

    Returns
    -------
    fit_results : ndarray, shape (N, 9), float32
        One row per spectrum; see the module docstring for the columns. Column 8 holds the
        quality bits (:data:`QUALITY_UNCONSTRAINED`, :data:`QUALITY_ZERO_ERROR`,
        :data:`QUALITY_PEGGED`), 0 when none apply and for failed fits.
    indices : ndarray, shape (N, 2), int32
        ``(i_left, i_right)`` of the fitting window per row, half-open, ``(0, 0)`` when no peak
        was found.
    counts : ndarray, shape (N, 2), int32
        Only with ``meta=True``: ``[n_iter, n_fev]`` per row.
    """
    spectra = np.ascontiguousarray(spectra, dtype=np.float32)
    if spectra.ndim != _SPECTRA_NDIM:
        msg = "spectra must be a 2-D (N, M) array"
        raise ValueError(msg)
    guides = np.asarray(guide_velocities, dtype=np.float32)
    if guides.ndim == 0:
        guides = np.full(spectra.shape[0], guides, dtype=np.float32)
    fits, indices, counts = _fit_spectra_batch(
        spectra,
        np.ascontiguousarray(np.atleast_2d(np.asarray(dopp_slit, dtype=np.float32))),
        np.ascontiguousarray(spec_noise, dtype=np.float32),
        np.ascontiguousarray(guides),
        float(velocity_range),
        int(npix),
        int(npix_slack),
        float(dv),
        float(width_min),
        float(amplitude_rel_min),
        float(amplitude_rel_max),
        float(width_max),
        float(width_guess),
        None if slit_index is None else np.ascontiguousarray(slit_index, dtype=np.int32),
        float(xtol),
        float(ftol),
        float(gtol),
        int(max_iter),
    )
    return (fits, indices, counts) if meta else (fits, indices)


def fit_single_spectrum(
    *,
    spectrum: ArrayLike,
    dopp_slit: ArrayLike,
    spec_noise: ArrayLike,
    guide_velocity: float,
    **options: float,
) -> (
    tuple[NDArray[np.float32], tuple[int, int]]
    | tuple[NDArray[np.float32], tuple[int, int], NDArray[np.int32]]
):
    """
    Fit one spectrum; ``options`` are the keyword arguments of :func:`fit_spectra_batch`.

    Returns the 9-element result row and the ``(i_left, i_right)`` fitting window, plus the
    ``[n_iter, n_fev]`` counts with ``meta=True``.
    """
    meta = options.pop("meta", False)
    fits, indices, counts = fit_spectra_batch(
        spectra=np.asarray(spectrum, dtype=np.float32)[np.newaxis],
        dopp_slit=dopp_slit,
        spec_noise=np.asarray(spec_noise, dtype=np.float32)[np.newaxis],
        guide_velocities=guide_velocity,
        meta=True,
        **options,
    )
    window = (int(indices[0, 0]), int(indices[0, 1]))
    return (fits[0], window, counts[0]) if meta else (fits[0], window)


def fit_gaussian(
    *,
    x: ArrayLike,
    y: ArrayLike,
    error: ArrayLike,
    initial: ArrayLike,
    lower_bounds: ArrayLike,
    upper_bounds: ArrayLike,
    xtol: float = 1.0e-6,
    ftol: float = 1.0e-6,
    gtol: float = 1.0e-6,
    max_iter: int = 2000,
    meta: bool = False,
) -> NDArray[np.float32] | tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Fit a bounded Gaussian ``amp * exp(-0.5 * ((x - mean) / sigma)**2)`` to ``(x, y ± error)``.

    Parameters
    ----------
    x, y, error:
        Samples, shape ``(N,)`` with ``N >= 3``; all finite, ``error`` positive. Converted to
        float32; the solve itself runs in float64.
    initial:
        Starting guess ``[amplitude, mean, sigma]``, clamped into the bounds.
    lower_bounds, upper_bounds:
        Bounds on ``[amplitude, mean, sigma]``; finite, increasing, and ``sigma > 0`` at the lower
        end.
    xtol, ftol, gtol, max_iter, meta:
        As in :func:`fit_spectra_batch`.

    Returns
    -------
    ndarray, shape (8,), float32
        ``[amplitude, mean, sigma, amplitude_err, mean_err, sigma_err, reduced_chi2, flag]``;
        *flag* is :data:`FLAG_SUCCESS` or :data:`FLAG_NO_CONVERGENCE`, and everything else is NaN
        when it is not success. With ``meta=True`` a second value holds ``[n_iter, n_fev]``.
    """
    row = _fit_gaussian(
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
    return (row[:8], row[8:].astype(np.int32)) if meta else row[:8]
