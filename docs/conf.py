import os
import sys

# Make the src/ package importable without installing it
sys.path.insert(0, os.path.abspath('../src'))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = 'DiskMELTS'
copyright = '2025, Chengyan Xie, Dingshan Deng'
author    = 'Chengyan Xie, Dingshan Deng'
release   = '0.1.0'

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # pull docstrings from source
    'sphinx.ext.napoleon',      # support Google/NumPy docstring style
    'sphinx.ext.viewcode',      # add [source] links to API pages
    'sphinx.ext.intersphinx',   # cross-link to numpy/scipy/torch docs
    'myst_parser',              # write docs in Markdown
]

# ---------------------------------------------------------------------------
# Mock heavy dependencies so the doc build does not require installing them
# ---------------------------------------------------------------------------
autodoc_mock_imports = [
    'torch',
    'sklearn',
    'scipy',
]

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    'members':          True,
    'undoc-members':    False,
    'show-inheritance': True,
    'member-order':     'bysource',
}
autodoc_typehints = 'description'

# ---------------------------------------------------------------------------
# Napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_google_docstring  = True
napoleon_numpy_docstring   = False
napoleon_use_param         = True
napoleon_use_rtype         = True

# ---------------------------------------------------------------------------
# intersphinx — cross-references to external packages
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':  ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

# ---------------------------------------------------------------------------
# MyST-Parser settings
# ---------------------------------------------------------------------------
myst_enable_extensions = ['colon_fence']
source_suffix = {
    '.rst': 'restructuredtext',
    '.md':  'markdown',
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,
    'titles_only':      False,
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
