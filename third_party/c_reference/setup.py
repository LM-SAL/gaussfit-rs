"""
Build the pinned C reference independently of MUSE and the Rust wheel.
"""

import sys

import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "gaussfit_c_reference.fit_single_spectrum_ext",
            sources=["vendor/fit_single_spectrum_ext.c", "vendor/cmpfit-1.5/mpfit.c"],
            include_dirs=[np.get_include(), "vendor", "vendor/cmpfit-1.5"],
            define_macros=[("MPFIT_FLOAT", "1"), ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=["/std:c17", "/O2"]
            if sys.platform == "win32"
            else ["-std=c17", "-O3"],
            libraries=["m"] if sys.platform == "linux" else [],
            language="c",
        ),
    ],
)
