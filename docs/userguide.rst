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

:func:`~gaussfit_rs.fit_spectra_batch` is the entry point for IFU data. It takes a 2-D
``(N, M)`` array and fits every row concurrently with Rayon, releasing the GIL for the whole
computation. ``spec_noise`` has the same shape as ``spectra``; ``guide_velocities`` is one value
per row or a scalar for all rows.

.. code-block:: python

   from gaussfit_rs import fit_spectra_batch

   fits, indices = fit_spectra_batch(
       spectra=spectra_2d,       # (N, M) float32, C-contiguous
       dopp_slit=velocity_grid,  # (M,) float32
       spec_noise=noise_2d,      # (N, M) float32, C-contiguous
       guide_velocities=0.0,
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

   converged = fits[:, 7] == 0.0   # FLAG_SUCCESS
   velocities = fits[converged, 1]  # km/s

When the velocity grid differs per slit, pass the grids as one ``(n_slit, M)`` table and
``slit_index``, an ``int32`` array naming the grid row of each spectrum. A block with a slit axis
is then fitted in one call from a reshaped view, with no per-slit copies:

.. code-block:: python

   rows = flux.reshape(-1, n_wave)                     # view
   slit_index = np.indices(flux.shape[:-1])[slit_axis].ravel()
   fits, indices = fit_spectra_batch(
       spectra=rows, dopp_slit=grids, spec_noise=noise.reshape(-1, n_wave),
       guide_velocities=guides.ravel(), slit_index=slit_index, **options,
   )
   fits = fits.reshape(*flux.shape[:-1], 9)

Output format
-------------

Both functions return one 9-element float32 row per spectrum:

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
   * - 8
     - quality
     - Quality bits, see below (0 for failed fits)

Fields 0-6 are ``NaN`` when ``flag != FLAG_SUCCESS``. :class:`~gaussfit_rs.FitResult` unpacks a
row by name. The second return value is the fitting window, ``(i_left, i_right)`` per spectrum,
half-open and ``(0, 0)`` when no peak was found.

Fit quality indicator
---------------------

Column 8 is 0 for failed fits (the flag column reports those) and otherwise a mask of these bits, exported as :data:`~gaussfit_rs.QUALITY_UNCONSTRAINED`,
:data:`~gaussfit_rs.QUALITY_ZERO_ERROR` and :data:`~gaussfit_rs.QUALITY_PEGGED`:

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
``flags != 0`` also includes pegged parameters, which may be legitimate saturation. The amplitude
interval is deliberately excluded: in the pipeline configuration it is a detection window of
+/-10 % of the peak, so comparing against its width flags ordinary low-SNR fits. See "Unconstrained-Fit
Indicator" in the design notes for the measurements and the cases no bit
catches.

Solver counts
-------------

``meta=True`` adds one more return value: an ``int32`` array of ``[n_iter, n_fev]`` per fit
(accepted Levenberg-Marquardt iterations and model evaluations, Jacobian calls included), ``-1``
where the fit did not run or did not converge. It is there for profiling and for comparing
against other MPFIT builds.

Tolerances
----------

Every fitting function takes ``xtol``, ``ftol``, ``gtol`` (default ``1e-6``) and ``max_iter``
(default 2000), which must be finite and positive. Tighten them for high-SNR data; relax them
(for example ``gtol=1e-4``) for bulk runs where exact convergence matters less. The other
options must be finite, with positive ``npix`` and ``dv``, non-negative ``npix_slack`` and
``velocity_range``, and increasing amplitude and width bounds.

Low-level Gaussian fitting
--------------------------

:func:`~gaussfit_rs.fit_gaussian` fits a bounded Gaussian to arbitrary ``(x, y ± error)`` data
without the peak search, for use outside the spectral pipeline:

.. code-block:: python

   from gaussfit_rs import fit_gaussian

   row = fit_gaussian(
       x=x, y=y, error=err,
       initial=[1.0, 0.0, 10.0],          # amplitude, mean, sigma
       lower_bounds=[0.0, -50.0, 1.0],
       upper_bounds=[5.0, 50.0, 40.0],
   )

The row has the same layout as above without the peak-search flag: *flag* is
``FLAG_SUCCESS`` or ``FLAG_NO_CONVERGENCE``.
