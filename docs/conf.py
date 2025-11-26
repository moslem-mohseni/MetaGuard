"""
Sphinx configuration for MetaGuard documentation.

Author: Moslem Mohseni
"""

import os
import sys
from datetime import datetime

# Add source directory to path
sys.path.insert(0, os.path.abspath("../src"))

# Project information
project = "MetaGuard"
copyright = f"{datetime.now().year}, Moslem Mohseni"
author = "Moslem Mohseni"

# Get version from package
try:
    from metaguard import __version__
    version = __version__
    release = __version__
except ImportError:
    version = "1.1.0"
    release = "1.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_copybutton",
]

# Templates and static files
templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Source file settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

# HTML theme settings
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
}

# HTML output settings
html_title = "MetaGuard Documentation"
html_short_title = "MetaGuard"
html_show_sourcelink = True
html_show_sphinx = False
html_show_copyright = True

# Napoleon settings (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_mock_imports = ["sklearn", "pandas", "numpy", "joblib"]

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST parser settings (for Markdown)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]

# Copy button settings
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
