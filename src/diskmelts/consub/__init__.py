"""Continuum subtraction and plotting for disk spectra.

Use ``calculate`` to return five arrays and ``plot`` to display or save them.
The diagnostic DataFrame interface remains available as ``subtract_continuum``.
"""
from ._core import (
    generate_line_free_regions, load_regions, read_spectrum,
    subtract_continuum, plot_consub, save_consub,
)
from ._api import calculate, plot

__all__ = [
    "calculate", "plot", "read_spectrum", "load_regions",
    "generate_line_free_regions", "subtract_continuum", "plot_consub", "save_consub",
]
