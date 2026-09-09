# Pinned C reference

This test-only package builds MUSE's Gaussian fitting C extension without
installing MUSE or ftoolss. It is independent of the gaussfit-rs Rust wheel
and is not a runtime dependency. A C17 compiler and Python development
headers are required; pip installs the NumPy headers and setuptools build
dependencies in isolation.

From the repository root, in an activated Python environment:

```bash
python -m pip install './third_party/c_reference' '.[tests]'
python -c 'import gaussfit_c_reference.fit_single_spectrum_ext'
python -m pytest python/gaussfit_rs/tests/test_c_parity.py
```

Or run `tox -e py313-cparity`. This environment installs the local reference
and requires it to import before running the tests. The GitHub Actions
`c_parity` job uses this environment. Ordinary test environments still run
the recorded fixtures and skip the live comparison if this package is absent.

## Provenance and build

`SOURCES.json` records the upstream repository, revision, path, and SHA-256
of each copied file. The C files are unchanged:

- `fit_single_spectrum_ext.c` comes from MUSE revision
  `74df595bcf2367f2cebee41acba2646bdf22f248`.
- `gaussian_deviate.c`, `mpfit.c`, `mpfit.h`, and CMPFIT's `DISCLAIMER`
  come from ftoolss revision `9a8a9defd61d6e560a7cf1f44dae21eb4b1c8e90`.
  They also match the files vendored in the local MUSE build used to
  validate this snapshot. This is ftoolss's modified CMPFIT, including
  float32 support, rather than a stock CMPFIT release.

The build uses `MPFIT_FLOAT=1`, C17, and optimization, as MUSE does.
Only the Python package namespace changes: the extension is installed as
`gaussfit_c_reference.fit_single_spectrum_ext`. No MUSE Python modules are
imported and no Rust implementation is used by the reference.

See `LICENSE.muse`, `LICENSE.ftoolss`, and
`vendor/cmpfit-1.5/DISCLAIMER` for the upstream licensing notices.
This product includes software developed by the University of Chicago,
as Operator of Argonne National Laboratory.

## Updating the reference

Keep updates explicit: copy the files from identified upstream revisions,
retain their licenses, and update the paths, revisions, and hashes in
`SOURCES.json`. Rebuild and run the parity suite before replacing fixtures:

```bash
python -m pip install --force-reinstall ./third_party/c_reference
python -m pytest python/gaussfit_rs/tests/test_c_parity.py
```

Do not modify the C implementation to make Rust tests pass.

To regenerate the recorded fixtures with the installed reference:

```bash
python -m gaussfit_rs.tests.make_c_reference
```

Review fixture changes and figure comparisons before updating image hashes.
Passing these tests establishes the documented tolerance contract on the
selected spectra. It does not establish equivalence for all inputs;
the generator documents intentionally excluded cases.
