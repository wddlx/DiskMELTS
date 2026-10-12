"""Array-based continuum API."""
from pathlib import Path

import numpy as np
import pandas as pd

from ._core import subtract_continuum, plot_consub


def calculate(wav, flux, error=None, preset="ttauri", exclude=None,
              linefreefile="", *, boxsizes=None, special=None, method="robust",
              shift=True, correction_scale=0.75, special_blend=0.15):
    """Calculate and subtract a continuum, returning five NumPy arrays.

    Parameters
    ----------
    wav, flux : array-like
        Matching one-dimensional wavelength (microns) and raw flux arrays.
        Wavelength must be finite, positive, unique, and increasing. Nonfinite
        flux is excluded from fitting and returned as NaN in the residual.
    error : array-like or None, optional
        Measurement errors in the same units as flux. Missing or nonpositive
        values are replaced by local noise estimates. None estimates all errors.
    preset : str or None, optional
        ``water``, ``ttauri``, ``carbon_rich``, or ``carbon_extended`` selects
        bundled windows. Carbon presets exclude 12.5--16.5 microns. None uses
        a custom linefreefile, or an unrestricted robust fit if no file is given.
    exclude : sequence of (float, float) or None, optional
        Additional wavelength intervals excluded from fitting and correction.
    linefreefile : str, pathlib.Path or None, optional
        Used only with preset=None. A CSV containing ``xmin,xmax`` specifies
        fit and correction windows. None or an empty string fits all nonexcluded
        pixels and disables line-free correction, so cont equals contOrig.
    boxsizes : number, sequence or None, optional
        Pixel count, regional ``(xmin,xmax,pixels)`` rows, or automatic widths.
    special : sequence of (float, float) or None, optional
        Intervals where the reference correction is suppressed.
    method : {"robust", "arpls"}, optional
        Baseline estimator. The default uses robust local fits and PCHIP.
    shift : bool, optional
        Apply the reference correction when windows are available.
    correction_scale, special_blend : float, optional
        Correction-bin width and special-region transition width in microns.

    Returns
    -------
    wav, consub, err, cont, contOrig : tuple of numpy.ndarray
        Five independent, one-dimensional arrays of the input length:
        wavelength, continuum-subtracted flux, measurement/estimated noise
        error, final continuum, and continuum before the reference correction.
        The returned err does not include correlated continuum uncertainty.

    Notes
    -----
    Large excluded intervals require interpolation and may be poorly constrained.
    Use ``subtract_continuum`` for support masks and uncertainty diagnostics.
    All default reference tables are bundled with the installed package.
    """
    custom_file = linefreefile is not None and str(linefreefile) != ""
    if preset is not None and custom_file:
        raise ValueError("Set preset=None to use linefreefile")
    result = subtract_continuum(
        wav, flux, error=error, preset=preset, exclude=exclude,
        line_free_regions=linefreefile if custom_file else None,
        reference_regions="curated" if preset is not None else None,
        boxsizes=boxsizes, special=special, method=method, shift=shift,
        correction_scale=correction_scale, special_blend=special_blend,
    )
    return tuple(result[name].to_numpy(copy=True) for name in
                 ("wave", "flux", "err", "cont", "cont_unshifted"))


def plot(wav, consub, error=None, cont=None, contOrig=None, savedir=None, *,
         ranges=None, title=None, filename="continuum.png"):
    """Plot continuum and residual arrays, optionally saving the figure.

    Parameters
    ----------
    wav, consub : array-like
        Wavelength in microns and continuum-subtracted flux.
    error : array-like or None, optional
        Errors for the shaded residual band. None omits the band.
    cont : array-like
        Final continuum; required. The raw spectrum is reconstructed as
        consub + cont. It can be passed as the fourth positional argument.
    contOrig : array-like or None, optional
        Unshifted continuum. None omits that curve.
    savedir : str, pathlib.Path or None, optional
        Output directory, created if needed. None creates no files.
    ranges : (float, float), sequence of pairs or None, optional
        Plot limits in microns; None shows the full wavelength range.
    title : str or None, optional
        Figure title.
    filename : str, optional
        Image basename inside savedir, default ``continuum.png``. A suffix such
        as .png, .pdf or .svg selects the Matplotlib output format.

    Returns
    -------
    matplotlib.figure.Figure
        Open figure, displayed by Jupyter's Matplotlib backend. In scripts,
        call matplotlib.pyplot.show() to display it interactively.
    """
    x = np.asarray(wav, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("wav must be a finite, increasing 1D array with at least 2 samples")
    if cont is None:
        raise ValueError("cont is required to plot the continuum")
    values = {}
    for name, value in (("flux", consub), ("cont", cont),
                        ("err", error), ("cont_unshifted", contOrig)):
        if value is not None:
            array = np.asarray(value, dtype=float)
            if array.shape != x.shape:
                raise ValueError(f"{name} must have the same shape as wav")
            values[name] = array
    if "flux" not in values:
        raise ValueError("consub is required")
    if "err" in values and np.any(values["err"] < 0):
        raise ValueError("error must be nonnegative")
    output = None
    if savedir is not None:
        if not filename or Path(filename).name != filename:
            raise ValueError("filename must be a basename, not a path")
        output = Path(savedir) / filename
    result = pd.DataFrame({"wave": x, **values,
                           "raw_flux": values["flux"] + values["cont"],
                           "reference_mask": False, "weak_support": False})
    return plot_consub(result, ranges=ranges, title=title, output=output)
