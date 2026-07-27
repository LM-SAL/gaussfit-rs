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
