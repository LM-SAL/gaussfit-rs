# Pinned C++ reference

This test-only package builds MUSE's `fastfit2` C++ Gaussian fitting extensions, the batch
implementation gaussfit-rs is benchmarked against, without installing MUSE. It is independent of
the gaussfit-rs Rust wheel and is not a runtime dependency. A C++17 compiler with OpenMP and
Python development headers are required; pip installs the NumPy headers and setuptools build
dependencies in isolation.

From the repository root, in an activated Python environment:

```bash
python -m pip install './third_party/cpp_reference' '.[tests]'
python -c 'import gaussfit_cpp_reference.fit_batch_spectrum_ext'
python -m pytest python/gaussfit_rs/tests/test_cpp_parity.py
```

Or run `tox -e py313-cparity`, which installs this package next to the C reference and requires
both to import before running the tests. Ordinary test environments skip the C++ comparison if
this package is absent.

## Provenance and build

`SOURCES.json` records the upstream repository, revision, path, and SHA-256 of each copied
file. The files under `vendor/` are unchanged:

- `fit_single_spectrum_ext.cpp` and `fit_batch_spectrum_ext.cpp` come from MUSE branch
  `carlos_dev` at revision `a5d01f5b0bb843e4c9c0e3d3d918af3821853adf` (2026-07-14),
  `muse/fastfit2/`.
- At that revision `muse/fastfit2/cmpfit-1.5` is a symlink to
  `../fastfit/_vendor_mpfit/cmpfit-1.5`, which is populated from ftoolss; `mpfit.c`, `mpfit.h`
  and CMPFIT's `DISCLAIMER` are therefore the same files, byte for byte, as in
  `third_party/c_reference` (ftoolss revision `9a8a9defd61d6e560a7cf1f44dae21eb4b1c8e90`).
  The C++ port is documented upstream as a step-by-step mirror of the C extension over that
  solver; what it adds is a per-thread workspace and an OpenMP batch loop.

The build mirrors `muse/fastfit2/setup_package.py` at the pinned revision: `MPFIT_FLOAT=1`,
C++17, `-O3 -fno-math-errno -fopenmp -DOMP_BLOCK_SIZE=32 -DUSE_SIMD`. Only the Python package
namespace changes: the extensions are installed as
`gaussfit_cpp_reference.fit_single_spectrum_ext` and
`gaussfit_cpp_reference.fit_batch_spectrum_ext`. No MUSE Python modules are imported and no
Rust implementation is used by the reference.

`diagnostic/fit_counts_ext.cpp` is gaussfit-rs's own and is not upstream code. It includes the
vendored single-spectrum source with `mpfit` renamed to a shim that records `mp_result.niter`
and `mp_result.nfev`, and exports `_fit_single_spectrum_counts`, which returns the reference's
fit plus those two counts. It exists so the iteration counts of the Rust solver (`meta=True`)
can be compared with the reference's on identical inputs; it is not used by the parity test.

See `LICENSE.muse`, `LICENSE.ftoolss`, and `vendor/cmpfit-1.5/DISCLAIMER` for the upstream
licensing notices. This product includes software developed by the University of Chicago, as
Operator of Argonne National Laboratory.

## Updating the reference

Keep updates explicit: copy the files from identified upstream revisions, retain their licenses,
and update the paths, revisions, and hashes in `SOURCES.json`. Rebuild and run the parity suite:

```bash
python -m pip install --force-reinstall ./third_party/cpp_reference
python -m pytest python/gaussfit_rs/tests/test_cpp_parity.py
```

tox caches the built reference, so after changing anything under `vendor/` recreate the
environment with `tox -r -e py313-cparity`.

Do not modify the C++ implementation to make Rust tests pass.

The solver is the C reference's, so the same caveats apply: no evaluation cap
(`conf.maxfev = 0`) and an iteration count that only advances on accepted steps, so the inputs
listed in `third_party/c_reference/README.md` never return, and `HANGS_C` in `test_c_parity.py`
is shared by both live tests. Run any new sweep under a watchdog. The parity contract is the
one-sided one documented in `helpers.assert_fit_parity`.
