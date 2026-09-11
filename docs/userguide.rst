User Guide
==========

Fitting a single spectrum
-------------------------

:func:`~gaussfit_rs.fit_single_spectrum` searches for the brightest emission-line
peak within a velocity window, normalises by that peak, then runs a bounded
Levenberg-Marquardt Gaussian fit on the surrounding pixel window.

.. code-block:: python

   import numpy as np
   from gaussfit_rs import fit_single_spectrum, FitResult

   velocity_grid = np.linspace(-300, 300, 60, dtype=np.float32)  # km/s
   spectrum = np.exp(-0.5 * (velocity_grid / 30.0) ** 2).astype(np.float32)
   noise = np.full_like(spectrum, 0.05)

   result_arr, (i_left, i_right) = fit_single_spectrum(
       spectrum=spectrum,
       dopp_slit=velocity_grid,
       spec_noise=noise,
       guide_velocity=0.0,
       velocity_range=200.0,
       npix=10,
       npix_slack=2,
       dv=12.0,
       width_min=5.0,
       width_max=100.0,
       width_guess=30.0,
       amplitude_rel_min=0.1,
       amplitude_rel_max=2.0,
   )

   r = FitResult.from_array(result_arr)
   if r.converged:
       print(f"velocity = {r.velocity:.1f} ± {r.velocity_err:.1f} km/s")

Batch fitting (IFU data)
------------------------

:func:`~gaussfit_rs.fit_spectra_batch` is the recommended entry point for IFU
data.  It accepts a 2-D ``(N, M)`` array and processes all N rows concurrently
using Rayon, releasing the Python GIL for the full computation.
``spec_noise`` must have exactly the same shape as ``spectra``.  When
``n_pixels`` is supplied, the first ``n_pixels`` columns of both arrays are used.

.. code-block:: python

   from gaussfit_rs import fit_spectra_batch

   fits, indices = fit_spectra_batch(
       spectra=spectra_2d,       # (N, M) float32, C-contiguous
       dopp_slit=velocity_grid,  # (M,) float32
       spec_noise=noise_2d,      # (N, M) float32, C-contiguous
       guide_velocity=0.0,
       velocity_range=200.0,
       npix=10,
       npix_slack=2,
       dv=12.0,
       width_min=5.0,
       width_max=100.0,
       width_guess=30.0,
       amplitude_rel_min=0.1,
       amplitude_rel_max=2.0,
   )

   # fits[i, 7] == FLAG_SUCCESS means row i converged
   converged_mask = fits[:, 7] == 0.0
   velocities = fits[converged_mask, 1]   # km/s

:func:`~gaussfit_rs.fit_spectra_batch_guided` is the same call with one guide velocity per
row.  When the Doppler grid differs per slit, :func:`~gaussfit_rs.fit_spectra_batch_slits`
takes the grids as one ``(n_slit, M)`` table plus an ``int32`` ``slit_index`` naming the grid
row for each spectrum, so a block with a slit axis is fitted in a single call from a reshaped
view, with no per-slit staging copies.

Output format
-------------

Both functions return an 8-element float array per spectrum:

.. list-table::
   :header-rows: 1
   :widths: 5 20 50

   * - Index
     - Field
     - Description
   * - 0
     - amplitude
     - Fitted peak amplitude (same units as input spectrum)
   * - 1
     - velocity
     - Line-centre velocity [km/s]
   * - 2
     - sigma
     - Gaussian width 1-σ [km/s]
   * - 3
     - amplitude_err
     - 1-σ uncertainty on amplitude
   * - 4
     - velocity_err
     - 1-σ uncertainty on velocity [km/s]
   * - 5
     - sigma_err
     - 1-σ uncertainty on sigma [km/s]
   * - 6
     - reduced_chi2
     - Reduced χ² of best fit
   * - 7
     - flag
     - Status code (0 = success, 1 = no peak, 2 = no convergence)

Fields 0-6 are ``NaN`` when ``flag != FLAG_SUCCESS``.

Use :class:`~gaussfit_rs.FitResult` to unpack by name:

.. code-block:: python

   from gaussfit_rs import FitResult

   r = FitResult.from_array(result_arr)
   r.velocity      # float [km/s]
   r.sigma         # float [km/s]
   r.converged     # bool

Opt-in unconstrained-fit indicator
----------------------------------

``fit_single_spectrum``, ``fit_spectra_batch``, ``fit_spectra_batch_guided`` and
``fit_spectra_batch_slits`` accept ``quality=True``, which appends one column to the result array:

* single spectrum: ``(fit_results, window)``, where ``fit_results`` has shape ``(9,)``
* batch: ``(fit_results, indices)``, where ``fit_results`` has shape ``(N, 9)``

Column 8 is 0 for failed fits (the flag column reports those) and otherwise a mask of these bits,
exported as :data:`~gaussfit_rs.QUALITY_UNCONSTRAINED`, :data:`~gaussfit_rs.QUALITY_ZERO_ERROR`
and :data:`~gaussfit_rs.QUALITY_PEGGED`:

.. list-table::
   :header-rows: 1

   * - bit
     - meaning
   * - 1
     - a parameter's error is not smaller than the interval it was bounded to: velocity
       (``2 * dv * npix``) or linewidth (``width_max - width_min``). The data do not constrain it.
   * - 2
     - a formal error is exactly 0. This can result from the near-singular guard or from an exact
       fit, because formal errors scale with the residuals. It does not prove the fit is unconstrained.
   * - 4
     - a fitted parameter sits exactly on a bound. Reported, not judged: legitimate saturation
       sets this bit too (a broad line pinned at ``width_max``), so it is the most common bit and
       the one not to mask on blindly.

Convert the float32 quality column to integers before testing individual bits:

.. code-block:: python

   from gaussfit_rs import FLAG_SUCCESS, QUALITY_UNCONSTRAINED, QUALITY_ZERO_ERROR

   flags = fits[:, 8].astype("uint8")
   failed = fits[:, 7] != FLAG_SUCCESS
   unconstrained = (flags & QUALITY_UNCONSTRAINED) != 0
   suspect = (flags & (QUALITY_UNCONSTRAINED | QUALITY_ZERO_ERROR)) != 0

``suspect`` includes zero-error fits for inspection; it is not an automatic rejection rule for
exact fits. Failed fits must be checked separately because their quality bits are zero.
``flags != 0`` also includes pegged parameters, which may be legitimate saturation.
The amplitude interval is deliberately excluded: in the pipeline configuration it is a detection
window of +/-10 % of the peak, so comparing against its width flags ordinary low-SNR fits.

It is off by default, so result arity, dtypes and numerics are unchanged for existing callers.
See "Opt-In Unconstrained-Fit Indicator" in the design notes for the measurements, the per-family
breakdown and the cases no bit catches.

Opt-in solver counts
--------------------

The same four functions accept ``meta=True``, which adds a third return value: an ``int32``
array of ``[n_iter, n_fev]`` per fit (accepted Levenberg-Marquardt iterations and model
evaluations, Jacobian calls included), ``-1`` where the fit did not run or did not converge.
It exists for profiling and for comparing against other MPFIT builds; the default call is
unchanged.

Input validation and tolerances
-------------------------------

All fitting functions require finite, positive tolerances and ``max_iter``.
Spectrum fitting also requires positive ``npix`` and ``dv``, non-negative
``npix_slack`` and ``velocity_range``, increasing amplitude and width bounds,
and finite numeric inputs.  Low-level Gaussian fitting requires finite ``x`` and
``y`` arrays, finite positive ``error`` values, finite initial parameters, and
finite increasing bounds.

The f32 functions default to:

.. code-block:: python

   fit_single_spectrum(
       ...,
       xtol=1e-6,     # parameter step tolerance
       ftol=1e-6,     # cost-function change tolerance
       gtol=1e-6,     # gradient norm tolerance
       max_iter=2000,
   )

Tighten these for high-SNR data; relax (e.g. ``gtol=1e-4``) for bulk runs
where exact convergence is less critical.

Low-level Gaussian fitting
--------------------------

:func:`~gaussfit_rs.fit_gaussian_f32` fits a Gaussian to arbitrary
(x, y ± error) data without the spectral peak-finding step.

.. code-block:: python

   from gaussfit_rs import fit_gaussian_f32
   import numpy as np

   result = fit_gaussian_f32(
       x=x, y=y, error=err,
       initial=np.array([1.0, 0.0, 1.0], dtype=np.float32),
       lower_bounds=np.array([0.1, -5.0, 0.1], dtype=np.float32),
       upper_bounds=np.array([2.0, 5.0, 5.0], dtype=np.float32),
   )
