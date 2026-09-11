Installation
============

Requirements
------------

- Python >= 3.12
- NumPy >= 2.0
- Rust >= 1.85 (build only)

From PyPI
---------

Once wheels are published, install the package with:

.. code-block:: bash

   pip install gaussfit-rs

From source
-----------

Requires Rust ≥ 1.85 and `maturin <https://github.com/PyO3/maturin>`_.

.. code-block:: bash

   git clone https://github.com/LM-SAL/gaussfit-rs
   cd gaussfit-rs
   pip install maturin
   maturin develop --release

The wheel targets baseline x86-64 so it runs everywhere. On a machine with AVX2 (any x86-64 from
2013 on) a source build with

.. code-block:: bash

   RUSTFLAGS="-C target-cpu=x86-64-v3" maturin develop --release

is 3-4 % faster per fit; see "Performance" in the design notes for the measurement.
It is a build-time choice, not a default, because such a build does not run on older CPUs.
