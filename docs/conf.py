"""
Sphinx configuration for gaussfit-rs.
"""

from importlib.metadata import version

project = "gaussfit-rs"
author = "LMSAL"
copyright = "2026, LMSAL"  # noqa: A001
version = release = version("gaussfit-rs")
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
html_theme = "furo"
html_title = "gaussfit-rs"
