Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.3
- Rust >= 1.85 and `maturin <https://github.com/PyO3/maturin>`_ (source builds only)

From PyPI
---------

Wheels are published for Linux x86-64, macOS arm64 and Windows x86-64:

.. code-block:: bash

   pip install gaussfit-rs

From source
-----------

.. code-block:: bash

   git clone https://github.com/LM-SAL/gaussfit-rs
   cd gaussfit-rs
   pip install maturin
   maturin develop --release

The wheels target baseline x86-64 so they run everywhere. On a machine with AVX2 (any x86-64 from
2013 on) a source build with

.. code-block:: bash

   RUSTFLAGS="-C target-cpu=x86-64-v3" maturin develop --release

is 3-4 % faster per fit; see "Performance" in the design notes. It is a build-time choice, not a
default, because such a build does not run on older CPUs.

Development
-----------

.. code-block:: bash

   pip install maturin
   maturin develop --release --extras tests  # build into the active environment
   cargo test                                # Rust unit tests
   pytest --pyargs gaussfit_rs               # Python tests, including parity with the recorded C reference
   prek run --all-files                      # lint hooks (ruff, docformatter, cargo fmt, clippy, cargo test)

With the pinned C reference installed (``pip install ./third_party/c_reference``), the parity
tests also run the C extension live over many generator seeds, and
``python -m gaussfit_rs.tests.make_c_reference`` re-records the fixtures. ``tox -e py313-figure``
runs the figure tests against the committed hash library.

Releases: bump ``version`` in ``Cargo.toml`` (the only place it is set) and run
``towncrier build --version X.Y.Z``, which moves the fragments in ``changelog/`` into
``CHANGELOG.rst``; merge, then tag ``vX.Y.Z`` on ``main``; the workflow builds the wheels and the
sdist and uploads them to PyPI.
