Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.3
- Rust >= 1.99 and `maturin <https://github.com/PyO3/maturin>`_ (source builds only)

From PyPI
---------

Wheels are published for Linux x86-64 and arm64, macOS arm64 and Windows x86-64. Each platform
has one wheel for CPython 3.12 and later, and one each for free-threaded CPython 3.14 and 3.15,
on which gaussfit-rs runs without the GIL:

.. code-block:: bash

   pip install gaussfit-rs

From source
-----------

.. code-block:: bash

   git clone https://github.com/LM-SAL/gaussfit-rs
   cd gaussfit-rs
   pip install maturin
   maturin develop --release

On an x86-64-v3 CPU, this source build was 3–4 % faster per fit than the
baseline x86-64 wheel in the :ref:`performance measurements <fit-performance>`.

.. code-block:: bash

   RUSTFLAGS="-C target-cpu=x86-64-v3" maturin develop --release

The resulting extension requires a CPU that supports x86-64-v3.

Development
-----------

.. code-block:: bash

   pip install maturin
   maturin develop --release --extras tests
   cargo test
   pytest --pyargs gaussfit_rs
   prek run --all-files

With the pinned C reference installed (``pip install ./third_party/c_reference``), the parity
tests also run the C extension live over many generator seeds, and
``python -m gaussfit_rs.tests.make_c_reference`` re-records the fixtures. ``tox -e py313-figure``
runs the figure tests against the committed hash library.

Releases: bump ``version`` in ``Cargo.toml`` (the only place it is set) and run
``towncrier build --version X.Y.Z``, which moves the fragments in ``changelog/`` into
``CHANGELOG.rst``; merge, then tag ``vX.Y.Z`` on ``main``; the workflow builds the wheels and the
sdist and uploads them to PyPI.
