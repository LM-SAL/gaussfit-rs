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
