API Reference
=============

Functions
---------

.. autofunction:: gaussfit_rs.fit_single_spectrum

.. autofunction:: gaussfit_rs.fit_spectra_batch

.. autofunction:: gaussfit_rs.fit_gaussian_f32

Result type
-----------

.. autoclass:: gaussfit_rs.FitResult
   :members: from_array, converged

Constants
---------

.. py:data:: gaussfit_rs.FLAG_SUCCESS
   :value: 0.0

   Fit converged successfully.

.. py:data:: gaussfit_rs.FLAG_NO_LOCAL_MAX
   :value: 1.0

   No spectral peak found within the search window.

.. py:data:: gaussfit_rs.FLAG_NO_CONVERGENCE
   :value: 2.0

   LM optimiser did not converge within ``max_iter`` iterations.
