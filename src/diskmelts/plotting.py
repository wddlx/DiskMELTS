"""
diskmelts/plotting.py — Visualise spectral fitting results from fitting.py.

Two public functions:
    plot_fit         : plot observed + model (top panel) and residuals + chi-sq (bottom panel)
    plot_validation  : scatter plots of predicted vs. true from validation.py results

Typical workflow
----------------
    from diskmelts.plotting import plot_fit

    obs_wav, obs_flux = ...
    h2o  = fit_nested(obs_wav, obs_flux, 'H2O',  pretrained, fit_ranges=...)
    cmol = fit_nested(obs_wav, h2o['residual'], 'C2H2', pretrained, fit_ranges=...)
    plot_fit(obs_wav, obs_flux, [h2o, cmol],
             fit_ranges=[(11,12),(12.4,14.5),(17,18.5)],
             name='J16120505', save_path='figures/fit.png')
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Molecule display labels
# ---------------------------------------------------------------------------
_MOL_LABEL = {
    'H2O':       r'H$_2$O',
    'H2O_hot':   r'hot H$_2$O',
    'H2O_warm':  r'warm H$_2$O',
    'H2O_1':     r'H$_2$O comp 1',
    'H2O_2':     r'H$_2$O comp 2',
    'C2H2':      r'C$_2$H$_2$',
    'HCN':       r'HCN',
    'CO2':       r'CO$_2$',
    '13C12CH2':  r'$^{13}$C$^{12}$CH$_2$',
    '13CO2':     r'$^{13}$CO$_2$',
    'C4H2':      r'C$_4$H$_2$',
    'HC3N':      r'HC$_3$N',
}

_MOL_COLOR = {
    'H2O':      'blue',
    'H2O_1':    'blue',
    'H2O_2':    'C9',
    'H2O_hot':  'blue',
    'H2O_warm': 'C9',
    'C2H2':     'C3',
    'HCN':      'C4',
    'CO2':      'C2',
    '13C12CH2': 'C5',
    '13CO2':    'C6',
    'C4H2':     'brown',
    'HC3N':     'gray',
}

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
_PLOT_STYLE = {
    'figure.dpi':      150,
    'font.size':       9,
    'font.family':     'serif',
    'font.serif':      ['Times', 'Times New Roman'],
    'xtick.top':       True,
    'xtick.direction': 'in',
    'ytick.right':     True,
    'ytick.direction': 'in',
    'mathtext.fontset': 'cm',
}


def _get_palette():
    """Return a colorblind-safe palette (seaborn if available, else fallback)."""
    try:
        import seaborn as sns
        return sns.color_palette('colorblind')
    except ImportError:
        return ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7',
                '#56B4E9', '#F0E442']


def _norm_ranges(fit_ranges):
    """Normalise a single (lo, hi) or list of (lo, hi) to a list of tuples."""
    if fit_ranges is None:
        return None
    if isinstance(fit_ranges[0], (int, float)):
        return [tuple(fit_ranges)]
    return [tuple(r) for r in fit_ranges]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_fit(
    obs_wav,
    obs_flux,
    results,
    sigma=None,
    fit_ranges=None,
    wav_plot_range=None,
    name='',
    save_path=None,
    params=None,
    uncertainty=None,
):
    """
    Two-panel spectral fit figure.

    The top panel shows the observed spectrum, total model, and per-molecule
    components with an optional parameter text box (T, logN, log10A ± errors).
    The bottom panel shows the residual with a ±1σ band and reduced χ².

    Args:
        obs_wav (np.ndarray): observed wavelength axis in µm
        obs_flux (np.ndarray): observed flux in Jy
        results (dict or list[dict]): one result dict from fit_nested / fit_molecules,
                                      or a list of such dicts.
                                      Each dict must contain 'model_flux'.
                                      'mol' key is used for legend labels.
                                      'sigma' key (if present) is used for χ².
        sigma (float or None): noise std (Jy) for χ² and ±1σ band; overrides
                               any sigma stored in results; if neither is
                               available it is estimated from the residual std
        fit_ranges: (lo, hi) or [(lo1,hi1), ...] µm — shaded in both panels
        wav_plot_range (tuple or None): (lo_µm, hi_µm) x-axis range; defaults
                                        to full observed wavelength range
        name (str): source name used in the figure title
        save_path (str or None): file path to save figure (PNG/PDF); if None the
                                 figure is returned without saving
        params (dict or None): label -> {'T', 'logN', 'log10A', ...} from
                               fit_molecules result['params'].  When provided, a
                               parameter text box with best-fit values is drawn.
        uncertainty (dict or None): label -> {'T': {'minus', 'plus'}, ...} from
                                    fit_molecules result['uncertainty'].  Combined
                                    with params to show ± errorbars in the text box.

    Returns:
        (matplotlib.figure.Figure)
    """
    obs_wav  = np.asarray(obs_wav,  dtype=np.float64)
    obs_flux = np.asarray(obs_flux, dtype=np.float64)

    if isinstance(results, dict):
        results = [results]

    # --- Accumulate total model and per-molecule fluxes ---
    total_model = np.zeros_like(obs_flux)
    mol_models  = []    # list of (key, display_label, flux array)
    for res in results:
        flux = np.asarray(res['model_flux'], dtype=np.float64)
        total_model += flux
        if 'component_fluxes' in res and res['component_fluxes']:
            for key, comp_flux in res['component_fluxes'].items():
                lbl = _MOL_LABEL.get(key, key)
                mol_models.append((key, lbl, np.asarray(comp_flux, dtype=np.float64)))
        else:
            key = res.get('mol', f'mol{len(mol_models)+1}')
            lbl = _MOL_LABEL.get(key, key)
            mol_models.append((key, lbl, flux))

    residual = obs_flux - total_model

    # --- Noise for chi-square ---
    if sigma is None:
        for res in results:
            if 'sigma' in res and res['sigma'] is not None:
                sigma = float(res['sigma'])
                break
    if sigma is None or sigma <= 0:
        sigma = float(np.std(residual))

    # --- Reduced chi-square over fit_ranges (or full range) ---
    ranges = _norm_ranges(fit_ranges)
    if ranges is not None:
        chi_mask = np.zeros(len(obs_wav), dtype=bool)
        for lo, hi in ranges:
            chi_mask |= (obs_wav >= lo) & (obs_wav <= hi)
    else:
        chi_mask = np.ones(len(obs_wav), dtype=bool)

    n_fit    = int(chi_mask.sum())
    chi2_red = float(np.sum((residual[chi_mask] / sigma) ** 2) / n_fit) if n_fit > 0 else float('nan')

    # --- Wavelength plot limits ---
    lo_p = float(obs_wav.min()) if wav_plot_range is None else wav_plot_range[0]
    hi_p = float(obs_wav.max()) if wav_plot_range is None else wav_plot_range[1]
    pmask = (obs_wav >= lo_p) & (obs_wav <= hi_p)
    w = obs_wav[pmask]

    palette = _get_palette()

    with plt.style.context([_PLOT_STYLE]):
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.06},
        )

        # --- Top panel: observed + total model + individual molecules ---
        ax_top.plot(w, obs_flux[pmask], color='k', lw=1.5, label='Observed', zorder=3)
        ax_top.plot(w, total_model[pmask], color=palette[0], lw=1.2, ls='-',
                    label='Total model', zorder=4)
        for ci, (key, lbl, flux) in enumerate(mol_models):
            color = _MOL_COLOR.get(key, palette[(ci + 1) % len(palette)])
            ax_top.plot(w, flux[pmask], color=color, lw=0.9, ls='--',
                        label=lbl, zorder=3)
        ax_top.axhline(0, color='grey', lw=0.4, ls=':', zorder=1)
        if ranges is not None:
            for lo, hi in ranges:
                ax_top.axvspan(lo, hi, alpha=0.07, color='steelblue', zorder=1)
        ax_top.set_ylabel('Flux (Jy)')
        ax_top.legend(fontsize=8, loc='upper left')
        ax_top.tick_params(labelbottom=False)
        ax_top.set_title(f'{name} — spectral fit' if name else 'Spectral fit',
                         fontsize=10)

        # --- Parameter text box with errorbars ---
        if params is not None:
            unc = uncertainty or {}
            lines = []
            for label, p in params.items():
                u = unc.get(label, {})
                u_T     = u.get('T',      {})
                u_logN  = u.get('logN',   {})
                u_log10A = u.get('log10A', {})
                T_str = f"T={p['T']:.0f}"
                if u_T:
                    T_str += f"(+{u_T['plus']:.0f}/-{u_T['minus']:.0f})"
                logN_str = f"logN={p['logN']:.2f}"
                if u_logN:
                    logN_str += f"(+{u_logN['plus']:.2f}/-{u_logN['minus']:.2f})"
                la_val = p.get('log10A', np.log10(max(p.get('A', 1.0), 1e-300)))
                la_str = f"log10A={la_val:.2f}"
                if u_log10A:
                    la_str += f"(+{u_log10A['plus']:.2f}/-{u_log10A['minus']:.2f})"
                lines.append(f"{label}: {T_str} K  {logN_str}  {la_str}")
            ax_top.text(
                0.99, 0.98, '\n'.join(lines),
                transform=ax_top.transAxes, va='top', ha='right',
                fontsize=7, family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.80, ec='lightgrey'),
                zorder=5,
            )

        # --- Bottom panel: residuals + chi-square ---
        ax_bot.plot(w, residual[pmask], color=palette[1], lw=0.8)
        ax_bot.axhline(0, color='k', lw=0.5, ls=':')
        ax_bot.axhspan(-sigma, sigma, alpha=0.12, color='grey')
        if ranges is not None:
            for lo, hi in ranges:
                ax_bot.axvspan(lo, hi, alpha=0.07, color='steelblue', zorder=1)
        ax_bot.set_xlim(lo_p, hi_p)
        ax_bot.set_xlabel(r'Wavelength ($\mu$m)')
        ax_bot.set_ylabel('Residual (Jy)')
        ax_bot.text(0.99, 0.92, rf'$\chi^2_{{\rm red}} = {chi2_red:.2f}$',
                    transform=ax_bot.transAxes, ha='right', va='top', fontsize=9)

        plt.tight_layout()
        plt.show()
        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved → {save_path}')

    return fig


# ---------------------------------------------------------------------------
# Validation plotting
# ---------------------------------------------------------------------------

def _r2_score(true, pred):
    """R² (coefficient of determination) between true and pred arrays."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')


def plot_validation(
    mol,
    val_results,
    save_path=None,
):
    """
    Scatter plot of predicted vs. true parameters from validate_nt or validate_full.

    Automatically detects whether A data is present:
        validate_nt output   (keys T, logN only)         → 1 × 2 panels
        validate_full output (keys T, logN, log10A, NA)  → 1 × 4 panels

    Points with residual > 3σ in any parameter are highlighted in red.

    Args:
        mol (str): molecule name; used in the figure title
        val_results (dict): output of validate_nt or validate_full
        save_path (str or None): file path to save figure; returned without
                                 saving if None

    Returns:
        (matplotlib.figure.Figure)
    """
    mol_display = _MOL_LABEL.get(mol, mol)
    has_A       = 'log10A_true' in val_results

    if has_A:
        param_meta = [
            ('T_true',       'T_pred',       'T (K)',                   'Temperature'),
            ('logN_true',    'logN_pred',     r'$\log_{10}\,N$',         'Column density'),
            ('log10A_true',  'log10A_pred',   r'$\log_{10}\,A$',         'Scaling factor'),
            ('log10NA_true', 'log10NA_pred',  r'$\log_{10}(N{\cdot}A)$', r'$N \times A$'),
        ]
        figsize = (13, 3.5)
    else:
        param_meta = [
            ('T_true',    'T_pred',    'T (K)',              'Temperature'),
            ('logN_true', 'logN_pred', r'$\log_{10}\,N$',    'Column density'),
        ]
        figsize = (7, 3.5)

    n_panels = len(param_meta)

    outlier_mask = np.zeros(len(val_results['T_true']), dtype=bool)
    for true_key, pred_key, _, _ in param_meta:
        resid = val_results[pred_key] - val_results[true_key]
        sigma = np.std(resid)
        if sigma > 0:
            outlier_mask |= np.abs(resid) > 3 * sigma

    palette = _get_palette()

    with plt.style.context([_PLOT_STYLE]):
        fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
        fig.suptitle(f'{mol_display} — predicted vs. true', fontsize=11)

        for col, (true_key, pred_key, axis_label, title) in enumerate(param_meta):
            ax  = axes[0, col]
            yt  = val_results[true_key]
            yp  = val_results[pred_key]
            r2  = _r2_score(yt, yp)

            normal = ~outlier_mask
            ax.scatter(yt[normal], yp[normal], s=14, alpha=0.65,
                       color=palette[0], edgecolors='none')
            if outlier_mask.any():
                ax.scatter(yt[outlier_mask], yp[outlier_mask], s=28, alpha=0.9,
                           color='tomato', marker='x', linewidths=1.0,
                           label=r'$>3\sigma$')
                ax.legend(fontsize=7, markerscale=1.0)

            lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            pad  = (lims[1] - lims[0]) * 0.05
            lims = [lims[0] - pad, lims[1] + pad]
            ax.plot(lims, lims, 'k--', lw=0.8)
            ax.set_xlim(lims); ax.set_ylim(lims)

            n_out = int(outlier_mask.sum())
            ax.set_xlabel(f'True {axis_label}')
            ax.set_ylabel(f'Predicted {axis_label}')
            ax.set_title(
                f'{title}   $R^2={r2:.3f}$' + (f'  [{n_out} outliers]' if n_out else ''),
                fontsize=8,
            )

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved → {save_path}')

    return fig


def plot_validation_split(
    mol,
    val_pretrain,
    val_holdout,
    save_path=None,
):
    """
    Two-row validation scatter plot: in-pretrain (top row) and holdout (bottom row).

    Each row has two panels — Temperature and column density — with predicted
    vs. true scatter, a 1:1 reference line, R², and >3σ outlier highlighting.
    Outlier masks are computed independently per row.

    Args:
        mol (str): molecule name; used in the figure title
        val_pretrain (dict): output of validate_nt for in-pretrain grid points
        val_holdout  (dict): output of validate_nt_holdout for unseen grid points
        save_path (str or None): file path to save figure; returned without
                                 saving if None

    Returns:
        (matplotlib.figure.Figure)
    """
    mol_display = _MOL_LABEL.get(mol, mol)

    param_meta = [
        ('T_true',    'T_pred',    'T (K)',           'Temperature'),
        ('logN_true', 'logN_pred', r'$\log_{10}\,N$', 'Column density'),
    ]
    row_data   = [val_pretrain, val_holdout]
    row_labels = ['In-pretrain', 'Holdout (unseen)']

    palette = _get_palette()

    with plt.style.context([_PLOT_STYLE]):
        fig, axes = plt.subplots(2, 2, figsize=(7, 7))
        fig.suptitle(f'{mol_display} — predicted vs. true', fontsize=11)

        for row, (data, row_label) in enumerate(zip(row_data, row_labels)):
            outlier_mask = np.zeros(len(data['T_true']), dtype=bool)
            for true_key, pred_key, _, _ in param_meta:
                resid = data[pred_key] - data[true_key]
                sigma = np.std(resid)
                if sigma > 0:
                    outlier_mask |= np.abs(resid) > 3 * sigma

            for col, (true_key, pred_key, axis_label, title) in enumerate(param_meta):
                ax = axes[row, col]
                yt = data[true_key]
                yp = data[pred_key]
                r2 = _r2_score(yt, yp)

                normal = ~outlier_mask
                ax.scatter(yt[normal], yp[normal], s=14, alpha=0.65,
                           color=palette[0], edgecolors='none')
                if outlier_mask.any():
                    ax.scatter(yt[outlier_mask], yp[outlier_mask], s=28, alpha=0.9,
                               color='tomato', marker='x', linewidths=1.0,
                               label=r'$>3\sigma$')
                    ax.legend(fontsize=7)

                lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
                pad  = (lims[1] - lims[0]) * 0.05
                lims = [lims[0] - pad, lims[1] + pad]
                ax.plot(lims, lims, 'k--', lw=0.8)
                ax.set_xlim(lims); ax.set_ylim(lims)

                n_out = int(outlier_mask.sum())
                ax.set_xlabel(f'True {axis_label}')
                ax.set_ylabel(f'Predicted {axis_label}')
                ax.set_title(
                    f'[{row_label}] {title}   $R^2={r2:.3f}$'
                    + (f'  [{n_out} out]' if n_out else ''),
                    fontsize=8,
                )

        plt.tight_layout()

        if save_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved → {save_path}')

    return fig
