"""Paper-versus-fit parameter figures for the saved real-observation runs."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


FIELDS = (
    ('logN', r'$\log_{10} N$ (cm$^{-2}$)', 'dex'),
    ('T_K', r'$T$ (K)', 'K'),
    ('logA', r'$\log_{10} A$ (au$^2$)', 'dex'),
    ('logNA', r'$\log_{10}(N A)$ (cm$^{-2}$ au$^2$)', 'dex'),
)
COLORS = {'H2O_warm': '#14b8cf', 'H2O_hot': '#2449e8'}


def _finite(value):
    return bool(np.isfinite(pd.to_numeric(value, errors='coerce')))


def _errors(row, field, paper):
    prefix = 'paper_' if paper else ''
    if field == 'T_K':
        keys = (f'{prefix}T_sigma_minus_K', f'{prefix}T_sigma_plus_K')
    else:
        keys = (f'{prefix}{field}_sigma_minus', f'{prefix}{field}_sigma_plus')
    values = [pd.to_numeric(row.get(key, np.nan), errors='coerce') for key in keys]
    if paper and field == 'logA' and row['component'] in ('HC3N', 'C4H2'):
        values = [0.0 if not _finite(value) else value for value in values]
    if all(_finite(value) and value >= 0 for value in values):
        return tuple(float(value) for value in values)
    return None


def _quantity(row, field, paper):
    prefix = 'paper_' if paper else ''
    if field == 'logNA':
        n, a = row.get(f'{prefix}logN'), row.get(f'{prefix}logA')
        if not (_finite(n) and _finite(a)):
            return np.nan, None
        nerr = _errors(row, 'logN', paper)
        aerr = _errors(row, 'logA', paper)
        error = (tuple(np.hypot(nerr[i], aerr[i]) for i in range(2))
                 if nerr is not None and aerr is not None else None)
        return float(n + a), error
    value = row.get(f'{prefix}{field}', np.nan)
    return (float(value), _errors(row, field, paper)) if _finite(value) else (np.nan, None)


def _axis_limits(x, y):
    low, high = float(min(min(x), min(y))), float(max(max(x), max(y)))
    span = high - low
    pad = max(0.12 * span, 0.3 if span < 1 else 0.1)
    return low - pad, high + pad


def plot_literature_comparisons(params, output_dir, summary_csv):
    """Plot every molecule, and write paired-difference sample statistics.

    Missing published errors are omitted, never interpreted as zero. The
    product errors assume independent log N and log A uncertainties.
    """
    output_dir, summary_csv = Path(output_dir), Path(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    records, figures = [], []
    params = params.copy()
    params['molecule'] = params.component.str.replace(r'^H2O_.*$', 'H2O', regex=True)
    for molecule, subset in params.groupby('molecule', sort=False):
        fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), dpi=150)
        fig.suptitle(f'{molecule}: literature values vs. v2 fits', fontsize=16, y=0.99)
        color_map = {component: COLORS.get(component, '#9354ad')
                     for component in subset.component.unique()}
        for col, (field, label, unit) in enumerate(FIELDS):
            common = []
            for _, row in subset.iterrows():
                x, _ = _quantity(row, field, True)
                y, _ = _quantity(row, field, False)
                if _finite(x) and _finite(y):
                    common.append((x, y))
            if common:
                x_all, y_all = zip(*common)
                limits = _axis_limits(x_all, y_all)
            else:
                limits = (0.0, 1.0)
            for ridx, method in enumerate(('models', 'grids')):
                ax = axes[ridx, col]
                points = []
                method_rows = subset[subset.method == method]
                for _, row in method_rows.iterrows():
                    x, xerr = _quantity(row, field, True)
                    y, yerr = _quantity(row, field, False)
                    if not (_finite(x) and _finite(y)):
                        continue
                    points.append((x, y))
                    color = color_map[row.component]
                    ax.errorbar(x, y,
                                xerr=np.array(xerr).reshape(2, 1) if xerr else None,
                                yerr=np.array(yerr).reshape(2, 1) if yerr else None,
                                fmt='o', ms=6.5, mfc=color, mec='black', mew=0.8,
                                ecolor=color, elinewidth=1.1, capsize=2,
                                alpha=0.88, zorder=3)
                    if molecule != 'H2O' and len(method_rows) <= 5:
                        ax.annotate(row.source.replace('J16', ''), (x, y),
                                    xytext=(4, 4), textcoords='offset points',
                                    fontsize=6.5, color='#444444')
                ax.plot(limits, limits, '--', color='#333333', lw=1.1, zorder=1)
                if points:
                    delta = np.array([y - x for x, y in points])
                    mean = float(np.mean(delta))
                    std = float(np.std(delta, ddof=1)) if len(delta) > 1 else np.nan
                    if np.isfinite(std):
                        line = np.array(limits)
                        ax.fill_between(line, line - std, line + std,
                                        color='#b7bec8', alpha=0.18, zorder=0)
                    stats = (rf'$\mu(\Delta)={mean:+.2f}$' + '\n' +
                             (rf'$\sigma(\Delta)={std:.2f}$' if np.isfinite(std)
                              else r'$\sigma(\Delta)=\mathrm{n/a}$') +
                             f'\nn={len(delta)}')
                    if molecule == 'H2O':
                        text_x, text_y, ha, va = 0.04, 0.96, 'left', 'top'
                    else:
                        corners = [(0.04, 0.96, 'left', 'top'),
                                   (0.96, 0.96, 'right', 'top'),
                                   (0.04, 0.04, 'left', 'bottom'),
                                   (0.96, 0.04, 'right', 'bottom')]
                        scale = limits[1] - limits[0]
                        text_x, text_y, ha, va = max(
                            corners,
                            key=lambda c: min((c[0] - (x - limits[0]) / scale)**2 +
                                              (c[1] - (y - limits[0]) / scale)**2
                                              for x, y in points))
                    ax.text(text_x, text_y, stats, transform=ax.transAxes,
                            ha=ha, va=va, fontsize=8,
                            bbox={'facecolor': 'white', 'edgecolor': 'none',
                                  'alpha': 0.78, 'pad': 2})
                    records.append({'molecule': molecule, 'method': method,
                                    'quantity': field, 'unit': unit,
                                    'n': len(delta), 'mean_fit_minus_paper': mean,
                                    'std_fit_minus_paper': std,
                                    'rms_fit_minus_paper': float(np.sqrt(np.mean(delta**2)))})
                else:
                    ax.text(0.5, 0.5, 'No comparable paper value',
                            ha='center', va='center', transform=ax.transAxes,
                            color='#777777', fontsize=9)
                ax.set_xlim(limits)
                ax.set_ylim(limits)
                ax.set_aspect('equal', adjustable='box')
                ax.grid(color='#dddddd', alpha=0.55, linewidth=0.5)
                ax.set_xlabel(f'Paper {label}', fontsize=9)
                ax.set_ylabel(f'Fitted {label}', fontsize=9)
                ax.set_title(f'{method.capitalize()} | {label}', fontsize=10)
                ax.tick_params(labelsize=8)
        if molecule == 'H2O':
            handles = [plt.Line2D([], [], marker='o', linestyle='',
                                  mfc=COLORS[c], mec='black', markersize=7,
                                  label=c.replace('_', ' '))
                       for c in ('H2O_warm', 'H2O_hot')]
            fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.969),
                       ncol=2, frameon=False, fontsize=9)
        fig.text(0.5, 0.012,
                 'Error bars: paper (horizontal), fit (vertical). Gray band: ±1 sample '
                 'standard deviation of fitted − paper values. Bars beyond axes are clipped.',
                 ha='center', fontsize=9, color='#555555')
        fig.tight_layout(rect=(0, 0.035, 1, 0.94), w_pad=2.0, h_pad=2.0)
        path = output_dir / f'{molecule}_parameter_comparison.pdf'
        fig.savefig(path, bbox_inches='tight')
        fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches='tight')
        plt.close(fig)
        figures.append(path)
    pd.DataFrame.from_records(records).to_csv(summary_csv, index=False)
    return figures
