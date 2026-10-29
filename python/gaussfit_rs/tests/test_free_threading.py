"""
Tests for free-threaded CPython.
"""

import subprocess
import sys
import sysconfig

import pytest


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="needs free-threaded CPython"
)
def test_import_keeps_the_gil_disabled():
    # A fresh interpreter, as a module this test run imported earlier may have enabled the GIL.
    code = "import sys, gaussfit_rs; sys.exit(sys._is_gil_enabled())"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603
