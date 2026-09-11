"""
Build the pinned fastfit2 C++ reference independently of MUSE and the Rust wheel.

Flags mirror ``muse/fastfit2/setup_package.py`` at the pinned revision: C++17, ``-O3``,
``-fno-math-errno``, OpenMP, ``MPFIT_FLOAT`` for the vendored cmpfit. The same flags are passed
to ``mpfit.c``, as upstream does, so a ``-std=c++17`` warning on the C file is expected.
"""

import sys

import numpy as np
from setuptools import Extension, setup

if sys.platform == "win32":
    COMPILE = ["/std:c++17", "/O2", "/openmp"]
    LINK = []
    LIBS = []
else:
    COMPILE = ["-std=c++17", "-O3", "-fno-math-errno", "-fopenmp", "-DOMP_BLOCK_SIZE=32", "-DUSE_SIMD"]
    LINK = ["-fopenmp"]
    LIBS = ["m"] if sys.platform == "linux" else []
MACROS = [("MPFIT_FLOAT", "1"), ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
INCLUDE = [np.get_include(), "vendor", "vendor/cmpfit-1.5"]


def extension(name, source):
    return Extension(
        f"gaussfit_cpp_reference.{name}",
        sources=[source, "vendor/cmpfit-1.5/mpfit.c"],
        include_dirs=INCLUDE,
        define_macros=MACROS,
        extra_compile_args=COMPILE,
        extra_link_args=LINK,
        libraries=LIBS,
        language="c++",
    )


setup(
    ext_modules=[
        extension("fit_single_spectrum_ext", "vendor/fit_single_spectrum_ext.cpp"),
        extension("fit_batch_spectrum_ext", "vendor/fit_batch_spectrum_ext.cpp"),
        # gaussfit-rs diagnostic over the unmodified single-spectrum core; see README.
        extension("fit_counts_ext", "diagnostic/fit_counts_ext.cpp"),
    ],
)
