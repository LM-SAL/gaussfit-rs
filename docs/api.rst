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

Result rows and constants
-------------------------

.. automodule:: gaussfit_rs.fitting

The constants are re-exported from the ``gaussfit_rs`` package.

.. autodata:: gaussfit_rs.fitting.FLAG_SUCCESS
.. autodata:: gaussfit_rs.fitting.FLAG_NO_LOCAL_MAX
.. autodata:: gaussfit_rs.fitting.FLAG_NO_CONVERGENCE
.. autodata:: gaussfit_rs.fitting.QUALITY_UNCONSTRAINED
.. autodata:: gaussfit_rs.fitting.QUALITY_ZERO_ERROR
.. autodata:: gaussfit_rs.fitting.QUALITY_PEGGED
