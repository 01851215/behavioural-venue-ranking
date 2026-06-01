"""Sphinx configuration for BVR documentation."""

project   = "Behavioral Venue Ranking"
copyright = "2026, Chris Liu"
author    = "Chris Liu"
release   = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

templates_path    = ["_templates"]
exclude_patterns  = []
html_theme        = "sphinx_rtd_theme"
html_static_path  = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "torch":  ("https://pytorch.org/docs/stable", None),
}
