API Reference
=============

Functions
---------

.. autofunction:: gaussfit_rs.fit_single_spectrum

.. autofunction:: gaussfit_rs.fit_spectra_batch

.. autofunction:: gaussfit_rs.fit_gaussian

Result type
-----------

.. autoclass:: gaussfit_rs.FitResult
   :members: from_array, converged

Constants
---------

Fit status, column 7 of every result row:

.. autodata:: gaussfit_rs.FLAG_SUCCESS
.. autodata:: gaussfit_rs.FLAG_NO_LOCAL_MAX
.. autodata:: gaussfit_rs.FLAG_NO_CONVERGENCE

Quality bits, summed in column 8 of converged spectrum fits:

.. autodata:: gaussfit_rs.QUALITY_UNCONSTRAINED
.. autodata:: gaussfit_rs.QUALITY_ZERO_ERROR
.. autodata:: gaussfit_rs.QUALITY_PEGGED
