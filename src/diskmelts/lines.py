"""Catalogue and plot markers for emission lines that DiskMELTS does not fit.

The bundled ``data/unfitted_lines.csv`` lists vacuum wavelengths (µm) of
H I recombination lines, H2 pure-rotational and HD rotational lines, and
atomic fine-structure lines ([Ne II], [Ne III], [Ar II], [Ar III], [Fe II],
[Ni II], ...) in the MIRI range. These lines are not part of any fitted model;
marking them helps judge residuals and avoid them in ``fit_ranges``.

Sources (MPE ISO line lists, https://www.mpe.mpg.de/ir/ISO/linelists/):
H2/HD from the van Dishoeck (1994) compilation (Jennings et al. 1987;
Ulivi et al. 1991), fine-structure lines from Feuchtgruber et al. (1997,
2001), and H I from P. van Hoof (l-mixed averages). H I lines are limited to
Δn <= 3 and n_upper <= 14.

Typical use::

    from diskmelts import plot_line_markers
    plot_line_markers(ax)                         # all lines in the x-range
    plot_line_markers(ax, species=['H2', 'NeII'])  # a subset

OH is not in the catalogue: it is usually not in LTE, so DiskMELTS does not
recommend fitting it. ``plot_oh_model`` instead overlays the LTE OH spectrum
at a chosen temperature (100-6000 K), scaled to the residual peak, for a
visual comparison.
"""

import warnings
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

LINE_LIST_FILE = 'unfitted_lines.csv'
LINE_CATEGORIES = ('HI', 'molecular', 'atomic')
OH_MODEL_COLORS = ('tab:purple', 'tab:brown', 'tab:olive', 'tab:cyan')


def load_line_list(species=None, category=None, wav_range=None):
    """Return the bundled unfitted-line catalogue as a DataFrame.

    Args:
        species (str, list, or None): keep these species, e.g. 'H2', 'HD',
            'HI', 'NeII', 'ArIII' (the ``species`` column); None keeps all
        category (str, list, or None): keep these categories: 'HI',
            'molecular' (H2, HD), or 'atomic' (fine-structure lines)
        wav_range ((lo, hi) or None): keep lines with lo <= wavelength <= hi µm

    Returns:
        pandas.DataFrame with columns species, transition, wavelength_um
        (vacuum, µm), label (plot text), category, reference, source_url;
        sorted by wavelength.
    """
    path = resources.files('diskmelts').joinpath('data').joinpath(LINE_LIST_FILE)
    with path.open() as handle:
        table = pd.read_csv(handle)
    for column, value in (('species', species), ('category', category)):
        if value is None:
            continue
        wanted = [value] if isinstance(value, str) else list(value)
        unknown = set(wanted) - set(table[column])
        if unknown:
            raise ValueError(f'Unknown {column} {sorted(unknown)}; '
                             f'available: {sorted(set(table[column]))}')
        table = table[table[column].isin(wanted)]
    if wav_range is not None:
        lo, hi = sorted(wav_range)
        table = table[(table['wavelength_um'] >= lo) & (table['wavelength_um'] <= hi)]
    return table.sort_values('wavelength_um').reset_index(drop=True)


def plot_line_markers(ax, species=None, category=None, wav_range=None, *,
                      color='gray', ls='--', lw=0.8, alpha=0.6, label=True,
                      fontsize=6.5, text_y=0.98, max_names=3):
    """Draw unfitted lines as vertical dashed lines with names on an axis.

    Args:
        ax (matplotlib.axes.Axes): axis with wavelength (µm) on x
        species, category: line selection, as in ``load_line_list``
        wav_range ((lo, hi) or None): lines to draw; defaults to the
            current x-limits of ``ax``
        color, ls, lw, alpha: line style (default gray dashed, alpha 0.6)
        label (bool): write each line's name, rotated, beside its marker
        fontsize (float): label size
        text_y (float): label anchor in axes fraction. Below 1 the labels
            hang down from it inside the axis; at 1 or above they stand
            above the axis, clear of legends and text boxes.
        max_names (int): lines too close to label separately share one
            label ('H I 7-6 / H I 11-8'); beyond this many names the rest
            are counted as '+n'

    Returns:
        pandas.DataFrame of the drawn lines.
    """
    if wav_range is None:
        wav_range = tuple(sorted(ax.get_xlim()))
    lines = load_line_list(species, category, wav_range)
    if lines.empty:
        return lines
    xlim = ax.get_xlim()
    for wav in lines['wavelength_um']:
        ax.axvline(wav, color=color, ls=ls, lw=lw, alpha=alpha, zorder=2)
    if label:
        # Width of one rotated label in data units, from the axis size.
        axis_pt = ax.get_position().width * ax.figure.get_figwidth() * 72
        width = 1.3 * fontsize / axis_pt * abs(xlim[1] - xlim[0])
        # Labels go left of their line when there is room, otherwise right
        # of it; a line with no room on either side joins the previous label.
        placed = []
        edge = -np.inf
        for wav, text in zip(lines['wavelength_um'], lines['label']):
            if wav - width >= edge:
                placed.append([wav, 'right', [text]])
                edge = wav
            elif wav >= edge and placed[-1][1] == 'right':
                placed.append([wav, 'left', [text]])
                edge = wav + width
            else:
                placed[-1][2].append(text)
        outside = text_y >= 1
        for wav, side, texts in placed:
            names = texts[:max_names]
            if len(texts) > max_names:
                names.append(f'+{len(texts) - max_names}')
            ax.text(wav, text_y, ' / '.join(names),
                    transform=ax.get_xaxis_transform(), rotation=90, ha=side,
                    va='bottom' if outside else 'top', fontsize=fontsize,
                    color=color, alpha=min(1.0, alpha + 0.3), zorder=2,
                    clip_on=not outside)
    ax.set_xlim(xlim)
    return lines


@lru_cache(maxsize=2)
def _oh_broad_model(sigma):
    from diskmelts.v2_fitting import OH_BROAD_T, _specs_with_extra, load_fitting_models
    return load_fitting_models(OH_BROAD_T, sigma=sigma,
                               model_specs=_specs_with_extra(OH_BROAD_T, sigma))[OH_BROAD_T]


def check_oh_model_args(T, logN=14.0, sigma=0):
    """Raise ValueError unless every T, logN, and sigma suit the OH bank."""
    from diskmelts.v2_fitting import OH_BROAD_T_SPECS

    if isinstance(sigma, bool) or sigma not in OH_BROAD_T_SPECS:
        raise ValueError(f'OH model sigma must be 0 or 2, not {sigma!r}')
    spec = OH_BROAD_T_SPECS[sigma]
    temperatures = np.atleast_1d(np.asarray(T, dtype=float))
    if temperatures.size == 0:
        raise ValueError('give at least one OH temperature')
    for name, values, bounds in (('T', temperatures, spec['T_bounds']),
                                 ('logN', np.atleast_1d(float(logN)), spec['logN_bounds'])):
        bad = values[~((values >= bounds[0]) & (values <= bounds[1]))]
        if bad.size:
            raise ValueError(f'OH model {name}={bad[0]:g} is outside {bounds}')


def oh_model_flux(T, wavelength, logN=14.0, sigma=0):
    """LTE OH flux from the bundled 100-6000 K OH comparison bank.

    Args:
        T (float): temperature in K, 100-6000
        wavelength (array): wavelengths in µm, any order or shape
        logN (float): log10 OH column density (cm^-2), 13-19.5
        sigma (0 or 2): grid line broadening

    Returns:
        np.ndarray of flux in Jy for A=1 au² at 140 pc, shaped like
        ``wavelength``. Pixels outside 4.9-27 µm get zero flux (unlike
        ``synthesize_spectrum('OH_broadT', ...)``, which raises there).
    """
    from diskmelts.fitting import generate_spectrum
    from diskmelts.v2_fitting import OH_BROAD_T_SPECS

    check_oh_model_args(T, logN, sigma)
    spec = OH_BROAD_T_SPECS[sigma]
    wavelength = np.asarray(wavelength, dtype=float)
    lo = spec['wavelength_ranges'][0][0]
    hi = spec['wavelength_ranges'][-1][1]
    inside = (wavelength >= lo) & (wavelength <= hi)
    flux = np.zeros_like(wavelength)
    if inside.any():
        flux[inside] = generate_spectrum(float(T), float(logN), 1.0,
                                         _oh_broad_model(sigma),
                                         obs_wav=wavelength[inside])
    return flux


def plot_oh_model(ax, wavelength, residual, T, *, logN=14.0, sigma=0,
                  offset=-0.4, colors=OH_MODEL_COLORS, lw=0.8, legend=True):
    """Overlay LTE OH models on a residual spectrum for comparison.

    Each model is scaled so its peak equals the residual peak, then shifted
    by ``offset`` times that peak (default -0.4), so the OH pattern sits just
    below the residual. Both peaks are taken over the given ``wavelength``
    points; pass only the plotted part of the spectrum.

    Args:
        ax (matplotlib.axes.Axes): residual axis, wavelength (µm) on x
        wavelength, residual (array): residual spectrum (µm, Jy)
        T (float or list): OH temperature(s) in K, 100-6000
        logN (float): log10 OH column density (cm^-2); the default 14 is
            optically thin, where the normalised shape depends on T only
        sigma (0 or 2): grid line broadening; use the fit's sigma
        offset (float): vertical shift in units of the residual peak
        colors (sequence): one color per temperature, cycled
        lw (float): line width
        legend (bool): add a legend entry per temperature

    Returns:
        dict mapping each T to the plotted (scaled, offset) flux array.
    """
    check_oh_model_args(T, logN, sigma)
    wavelength = np.asarray(wavelength, dtype=float)
    residual = np.asarray(residual, dtype=float)
    temperatures = np.atleast_1d(np.asarray(T, dtype=float))
    finite = np.isfinite(residual)
    peak = float(np.max(residual[finite])) if finite.any() else 0.0
    curves = {}
    if peak <= 0:
        warnings.warn('OH models not drawn: the residual has no positive peak',
                      stacklevel=2)
        return curves
    for index, temperature in enumerate(temperatures):
        flux = oh_model_flux(temperature, wavelength, logN=logN, sigma=sigma)
        model_peak = float(np.max(flux)) if flux.size else 0.0
        if model_peak <= 0:
            warnings.warn(f'OH model at {temperature:g} K not drawn: no OH '
                          'emission in the plotted range (model covers 4.9-27 µm)',
                          stacklevel=2)
            continue
        curve = flux * (peak / model_peak) + offset * peak
        ax.plot(wavelength, curve, color=colors[index % len(colors)], lw=lw,
                label=f'OH LTE {temperature:.0f} K (scaled, {offset:+g}×peak)',
                zorder=3)
        curves[float(temperature)] = curve
    if legend and curves:
        ax.legend(fontsize=7, loc='lower left')
    return curves
