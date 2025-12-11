# Configuration file for the Sphinx documentation builder.
# This file generates API documentation that can be integrated with Docusaurus.

import os
import sys

# Add the WindGym package to the path
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "WindGym"
copyright = "2024, DTU Wind Energy"
author = "DTU Wind Energy"
version = "0.0.2"
release = "0.0.2"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Support for Google/NumPy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_markdown_builder",  # Output as Markdown for Docusaurus
]

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gymnasium": ("https://gymnasium.farama.org/", None),
}

# Templates path
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Options for Markdown output ---------------------------------------------
# These settings configure the sphinx-markdown-builder
markdown_http_base = ""
markdown_uri_doc_suffix = ".md"

# Mock imports for modules that may not be available during doc build
autodoc_mock_imports = [
    "dynamiks",
    "hipersim",
    "py_wake",
    "wetb",
    "stable_baselines3",
    "pettingzoo",
    "h2lib",
    "IPython",
    "wandb",
    "tensorboard",
]
