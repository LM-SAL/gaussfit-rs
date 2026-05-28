Changelog
=========

Unreleased
----------

Initial release.

- Parallel batch fitting via Rayon (:func:`~gaussfit_rs.fit_spectra_batch`)
- Single-spectrum fitting (:func:`~gaussfit_rs.fit_single_spectrum`)
- f32 and f64 low-level Gaussian fitting
- :class:`~gaussfit_rs.FitResult` named result type
- Bounded Levenberg-Marquardt optimiser with configurable tolerances
