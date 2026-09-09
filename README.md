# gaussfit-rs

Fast, parallel Gaussian fitting for spectroscopic data with a Rust core and a Python API.

Run `tox -e py313-cparity` to build the pinned MUSE C reference and compare
Rust fits against it and the recorded fixtures. This requires a C compiler
as well as Rust. See [the C reference instructions](third_party/c_reference/README.md)
for manual installation, source provenance, and fixture updates.
