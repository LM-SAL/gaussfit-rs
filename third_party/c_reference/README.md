# Pinned C reference

Test-only package that builds the reference C Gaussian fitter for
`python/gaussfit_rs/tests/test_c_parity.py`. It is not a dependency of the gaussfit-rs wheel.
It needs a C17 compiler and Python development headers.

From the repository root, in an activated Python environment:

```bash
python -m pip install './third_party/c_reference' '.[tests]'
python -m pytest python/gaussfit_rs/tests/test_c_parity.py
```

Or run `tox -e py313-cparity`, as the CI `c_parity` job does. Without this package the parity
tests still check the recorded fixtures and skip the live comparison.

## Sources

The files in `vendor/` are copied unchanged:

- `fit_single_spectrum_ext.c`: revision `74df595bcf2367f2cebee41acba2646bdf22f248` of the private
  upstream repository.
- `gaussian_deviate.c` and `cmpfit-1.5/`: [ftoolss](https://github.com/steinhh/ftools) revision
  `9a8a9defd61d6e560a7cf1f44dae21eb4b1c8e90`, a modified CMPFIT with float32 support.

To update, copy the new files unchanged, update the revisions above, reinstall with
`pip install --force-reinstall ./third_party/c_reference` (or `tox -r -e py313-cparity`, as tox
caches the build) and run the parity tests. Do not modify the C code to make Rust tests pass.

## Licenses

See `LICENSE.ftoolss` and `vendor/cmpfit-1.5/DISCLAIMER`. This product includes software
developed by the University of Chicago, as Operator of Argonne National Laboratory.
