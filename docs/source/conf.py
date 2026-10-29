from pathlib import Path
import sys

# Make the src/ package importable without installing it
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = 'DiskMELTS'
copyright = '2026, Chengyan Xie, Dingshan Deng'
author    = 'Chengyan Xie, Dingshan Deng'
release   = '2.3.0'

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',       # pull docstrings from source
    'sphinx.ext.napoleon',      # support Google/NumPy docstring style
    'sphinx.ext.viewcode',      # add [source] links to API pages
    'sphinx.ext.mathjax',       # render the equations used in the guides
    'myst_parser',              # write docs in Markdown
    'sphinx_design',            # guide cards on the landing page
    'nbsphinx',                 # render tutorial notebooks
    'nbsphinx_link',            # use the maintained notebooks/ files directly
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
napoleon_numpy_docstring   = True
napoleon_use_param         = True
napoleon_use_rtype         = True

# ---------------------------------------------------------------------------
# MyST-Parser settings
# ---------------------------------------------------------------------------
myst_enable_extensions = ['colon_fence', 'dollarmath']
nbsphinx_execute = 'never'     # fitting/training examples are expensive or need local grids
source_suffix = {
    '.rst': 'restructuredtext',
    '.md':  'markdown',
}

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------
html_theme = 'furo'
html_title = 'DiskMELTS documentation'
html_static_path = ['_static']

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
