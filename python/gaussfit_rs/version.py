"""
Version.
"""

import importlib.metadata as _meta

version = _meta.version("gaussfit-rs")

__all__ = ["version"]
