"""Compare v2 surrogate and direct-grid fits on the 14 local paper spectra.

Run from the repository root in the data_reduction environment::

    conda run -n data_reduction python examples/dev_v2_testsWithRO.py

Use --sources FZTau J16120505 or --methods models grids to run a subset.
Completed source/method outputs are reused unless --overwrite is supplied.
The paper values are comparison targets, never optimizer starting values.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from diskmelts import fit_observation, fit_observation_grids, plot_saved_observation
from diskmelts import plot_literature_comparisons


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'Realobs_data' / 'Consub_data'
RESULTS = ROOT / 'realobs_results' / 'v2_paper_comparison'
FIGURES = ROOT / 'figures' / 'realobs' / 'v2_paper_comparison'

YOUNG = {
    'FZTau': 'FZTau_1d_v8.2_csub.csv',
    'GKTau': 'GKTau_1d_v8.2_csub.csv',
    'HPTau': 'HPTau_1d_v8.2_csub.csv',
    'GQLup': 'GQLup_1d_v8.2_csub.csv',
    'IQTau': 'IQTau_1d_v8.2_csub.csv',
    'AS209': 'AS209_1d_v8.2_csub.csv',
    'CITau': 'CITau_1d_v8.2_csub.csv',
}
USCO = {
    'J16142029': 'j16142029_v9.0_contsub_RVcorr.csv',
    'J16075796': 'j16075796_v9.0_contsub_RVcorr.csv',
    'J16153456': 'j16153456_v9.0_contsub_RVcorr.csv',
    'J16123916': 'j16123916_v9.0_contsub_RVcorr.csv',
    'J16120505': 'j16120505_v9.0_contsub_RVcorr.csv',
    'J16153220': 'j16153220_v9.0_contsub_RVcorr.csv',
    'J16064385': 'j16064385_v9.0_contsub_RVcorr.csv',
}
SOURCES = {**YOUNG, **USCO}


def source_distance_pc(source):
    table_name = ('literature_young_sources.csv' if source in YOUNG
                  else 'literature_usco_sources.csv')
    table = pd.read_csv(ROOT / 'Realobs_data' / table_name)
    row = table.loc[table.source == source, 'd_pc']
    if len(row) != 1 or not np.isfinite(row.iloc[0]) or row.iloc[0] <= 0:
        raise ValueError(f'Missing valid published distance for {source}')
    return float(row.iloc[0])


def stages_for(source):
    """Literature-motivated molecules/windows, within both v2 model domains."""
    if source not in SOURCES:
        raise KeyError(f'Unknown local source: {source}')
    two_water = source != 'J16153220'
    if source in YOUNG:
        # Romero-Mirza et al. use 12-27 µm. The trained H2O bank ends at 25.
        water_range = (12.0, 25.0)
        fit_ranges = [(12.0, 12.9), (14.4, 14.9), (15.5, 25.0)]
    else:
        # Xie et al. fit the two water components outside the organic bands.
        water_range = (11.5, 19.0)
        fit_ranges = [(11.5, 12.2), (15.6, 18.6)]
    water = {
        'name': 'H2O', 'mol': 'H2O',
        'wavelength_range': water_range, 'fit_ranges': fit_ranges,
        'h2o_components': 2 if two_water else 1,
        'component_names': (['H2O_warm', 'H2O_hot'] if two_water
                            else ['H2O_warm']),
        'T_bounds': ({'H2O_warm': (200.0, 600.0),
                      'H2O_hot': (600.0, 1100.0)} if two_water
                     else (300.0, 600.0)),
        'logN_bounds': (15.0, 19.0),
        'loga_bounds': (-3.0, 3.0),
    }
    stages = [water]
    if source in YOUNG or source == 'J16142029':
        return stages

    table = pd.read_csv(ROOT / 'Realobs_data' / 'literature_usco_sources.csv')
    row = table.loc[table.source == source]
    if len(row) != 1:
        raise ValueError(f'No unique Upper Sco literature row for {source}')
    carbon = str(row.iloc[0].reported_C_molecules).split(';')
    main = [mol for mol in carbon
            if mol in ('C2H2', '13C12CH2', 'HCN', 'CO2', '13CO2')]
    if main:
        stages.append({
            'name': 'C_main', 'mol': main,
            'wavelength_range': (12.9, 16.25),
            'fit_ranges': (12.9, 16.25),
            'T_bounds': (100.0, 1400.0),
            'logN_bounds': (13.0, 19.5),
            'loga_bounds': (-3.0, 3.0),
        })
    for mol, interval in [('HC3N', (15.0, 15.2)),
                          ('C4H2', (15.8, 16.0))]:
        if mol in carbon:
            stages.append({
                'name': mol, 'mol': mol,
                'wavelength_range': interval,
                'fit_ranges': interval,
                'T_bounds': (100.0, 1400.0),
                'logN_bounds': (13.0, 19.5),
                # Xie et al. fix the rare-species emitting area to 1 au².
                'loga_bounds': (-1e-6, 1e-6),
            })
    return stages


def run_source(source, method, *, distance_pc=None, overwrite=False, n_samples=4096,
               n_refine=12, grid_maxiter=30, grid_popsize=5):
    path = DATA / SOURCES[source]
    if not path.is_file():
        raise FileNotFoundError(path)
    stages = stages_for(source)
    distance_pc = (source_distance_pc(source) if distance_pc is None
                   else float(distance_pc))
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError('distance_pc must be finite and positive')
    result_dir = RESULTS / method / source
    figure_dir = FIGURES / method / source
    final = result_dir / f'{source}_final_running.csv'
    config_path = result_dir / 'fit_config.json'
    saved_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    expected = [result_dir / f'{source}_{stage["name"]}_fit_params.csv'
                for stage in stages]
    if (final.is_file() and all(item.is_file() for item in expected) and
            saved_config.get('distance_pc') == distance_pc and not overwrite):
        print(f'Reusing {method} fit for {source}', flush=True)
        active = [stage['name'] for stage in stages]
        sigma = None
    else:
        print(f'Fitting {source} with {method}', flush=True)
        if method == 'models':
            result = fit_observation(
                path, stages, name=source, output_dir=result_dir,
                comparison_csv=result_dir / 'Fitted_Parameters.csv',
                distance_pc=distance_pc,
                detection_screening=False, n_samples=n_samples,
                n_refine=n_refine, n_top=min(10, n_refine), seed=42,
            )
        elif method == 'grids':
            result = fit_observation_grids(
                path, stages, name=source, grid_root=ROOT / 'Model_grids',
                output_dir=result_dir, detection_screening=False,
                distance_pc=distance_pc,
                maxiter=grid_maxiter, popsize=grid_popsize,
                n_refine=3, seed=42,
            )
        else:
            raise ValueError(method)
        active = result['fits']
        sigma = result['sigma_noise']
        config_path.write_text(json.dumps({
            'source': source, 'method': method,
            'distance_pc': distance_pc, 'reference_distance_pc': 140.0,
            'stages': [stage['name'] for stage in stages],
            'n_samples': n_samples if method == 'models' else None,
            'n_refine': n_refine if method == 'models' else 3,
            'grid_maxiter': grid_maxiter if method == 'grids' else None,
            'grid_popsize': grid_popsize if method == 'grids' else None,
        }, indent=2) + '\n')
    return plot_saved_observation(
        path, result_dir, stages, name=source,
        figure_dir=figure_dir, active_stages=active, sigma_noise=sigma,
    )


def _literature():
    usco = pd.read_csv(ROOT / 'Realobs_data' / 'literature_usco_table2.csv')
    usco = usco.loc[usco.status == 'fit'].copy()
    for isotope, parent in [('13C12CH2', 'C2H2'), ('13CO2', 'CO2')]:
        for source in usco.loc[usco.component == isotope, 'source']:
            child = (usco.source == source) & (usco.component == isotope)
            primary = usco.loc[(usco.source == source) &
                               (usco.component == parent)].iloc[0]
            for field in ('T_K', 'T_minus_K', 'T_plus_K', 'logA',
                          'logA_minus', 'logA_plus'):
                usco.loc[child, field] = primary[field]
    young = pd.read_csv(ROOT / 'Realobs_data' / 'literature_water_two_component.csv')
    young['source'] = young.source.replace({
        'FZ Tau': 'FZTau', 'GK Tau': 'GKTau', 'HP Tau': 'HPTau',
        'GQ Lup': 'GQLup', 'IQ Tau': 'IQTau', 'AS 209': 'AS209',
        'CI Tau': 'CITau',
    })
    young['component'] = 'H2O_' + young.component.replace({'cold': 'warm'})
    young['logN'] = np.log10(young.N_1e18 * 1e18)
    young['logA'] = np.log10(young.A_au2)
    for prefix, value, minus, plus in [
        ('logN', 'N_1e18', 'N_minus_1e18', 'N_plus_1e18'),
        ('logA', 'A_au2', 'A_minus_au2', 'A_plus_au2'),
    ]:
        young[f'{prefix}_minus'] = (np.log10(young[value]) -
                                    np.log10(young[value] - young[minus]))
        young[f'{prefix}_plus'] = (np.log10(young[value] + young[plus]) -
                                   np.log10(young[value]))
    fields = ['source', 'component', 'logN', 'logN_minus', 'logN_plus',
              'T_K', 'T_minus_K', 'T_plus_K', 'logA', 'logA_minus',
              'logA_plus']
    return pd.concat([usco[fields], young[fields]], ignore_index=True)


def collect_comparison():
    """Read saved fits and compare like-for-like components with the papers."""
    literature = _literature().set_index(['source', 'component'])
    rows, quality = [], []
    for source in SOURCES:
        stages = stages_for(source)
        fit_ranges = [item for stage in stages for item in (
            stage['fit_ranges'] if isinstance(stage['fit_ranges'], list)
            else [stage['fit_ranges']])]
        for method in ('models', 'grids'):
            folder = RESULTS / method / source
            config_path = folder / 'fit_config.json'
            config = json.loads(config_path.read_text()) if config_path.is_file() else {}
            final = folder / f'{source}_plot_data.csv'
            if final.is_file():
                data = pd.read_csv(final)
                mask = np.logical_or.reduce([
                    (data.wave >= lo) & (data.wave <= hi)
                    for lo, hi in fit_ranges
                ])
                residual = data.loc[mask, 'residual'].to_numpy(dtype=float)
                quality.append({'source': source, 'method': method,
                                'fit_pixels': int(mask.sum()),
                                'rms_Jy': float(np.sqrt(np.mean(residual ** 2)))})
            for stage in stages:
                path = folder / f'{source}_{stage["name"]}_fit_params.csv'
                if not path.is_file():
                    continue
                for fitted in pd.read_csv(path).itertuples(index=False):
                    key = (source, str(fitted.component))
                    ref = literature.loc[key] if key in literature.index else None
                    rows.append({
                        'source': source, 'method': method,
                        'component': fitted.component,
                        'distance_pc': config.get('distance_pc', np.nan),
                        'T_K': fitted.T, 'T_sigma_minus_K': fitted.T_minus,
                        'T_sigma_plus_K': fitted.T_plus,
                        'logN': fitted.logN, 'logN_sigma_minus': fitted.logN_minus,
                        'logN_sigma_plus': fitted.logN_plus,
                        'logA': fitted.log10A,
                        'logA_sigma_minus': fitted.log10A_minus,
                        'logA_sigma_plus': fitted.log10A_plus,
                        'paper_T_K': np.nan if ref is None else ref.T_K,
                        'paper_T_sigma_minus_K': np.nan if ref is None else ref.T_minus_K,
                        'paper_T_sigma_plus_K': np.nan if ref is None else ref.T_plus_K,
                        'paper_logN': np.nan if ref is None else ref.logN,
                        'paper_logN_sigma_minus': np.nan if ref is None else ref.logN_minus,
                        'paper_logN_sigma_plus': np.nan if ref is None else ref.logN_plus,
                        'paper_logA': np.nan if ref is None else ref.logA,
                        'paper_logA_sigma_minus': np.nan if ref is None else ref.logA_minus,
                        'paper_logA_sigma_plus': np.nan if ref is None else ref.logA_plus,
                    })
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / 'parameter_comparison.csv', index=False)
    pd.DataFrame(quality).to_csv(RESULTS / 'spectral_quality.csv', index=False)
    return pd.DataFrame(rows), pd.DataFrame(quality)


def plot_parameter_comparison(params):
    """Compare fitted H2O parameters with each other and the papers."""
    water = params[params.component.isin(('H2O_warm', 'H2O_hot'))]
    if water.empty:
        return None
    sources = [source for source in SOURCES if source in set(water.source)]
    fields = [('T_K', 'Temperature (K)'),
              ('logN', r'log$_{10}$ N (cm$^{-2}$)'),
              ('logA', r'log$_{10}$ A (au$^2$)')]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)
    for row, component in enumerate(('H2O_warm', 'H2O_hot')):
        subset = water[water.component == component]
        for col, (field, ylabel) in enumerate(fields):
            ax = axes[row, col]
            for method, offset, color in [('models', -0.18, '#2563eb'),
                                          ('grids', 0.18, '#ea580c')]:
                selected = subset[subset.method == method].set_index('source')
                xs, ys = [], []
                for i, source in enumerate(sources):
                    if source in selected.index:
                        xs.append(i + offset)
                        ys.append(float(selected.loc[source, field]))
                if xs:
                    ax.scatter(xs, ys, s=30, color=color, label=method, zorder=3)
            paper_field = f'paper_{field}'
            paper = subset.drop_duplicates('source').set_index('source')
            xs, ys = [], []
            for i, source in enumerate(sources):
                if source in paper.index and pd.notna(paper.loc[source, paper_field]):
                    xs.append(i)
                    ys.append(float(paper.loc[source, paper_field]))
            if xs:
                ax.scatter(xs, ys, s=42, marker='x', color='black',
                           label='paper', zorder=4)
            ax.set_title(f'{component.removeprefix("H2O_")} water')
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.set_xticks(range(len(sources)), sources, rotation=70, fontsize=8)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / 'water_parameter_comparison.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report():
    params, quality = collect_comparison()
    summary_figure = plot_parameter_comparison(params) if len(params) else None
    comparison_figures = plot_literature_comparisons(
        params, FIGURES / 'parameters', RESULTS / 'parameter_scatter.csv') if len(params) else []
    count = quality.groupby('method').source.nunique().to_dict() if len(quality) else {}
    lines = [
        '# V2 real-observation comparison', '',
        'References: [Xie et al. 2026, Tables 1–2](https://arxiv.org/html/2606.27477v1) '
        'and [Romero-Mirza et al. 2024, Table 3](https://arxiv.org/html/2409.03831).', '',
        f'Completed sources: models {count.get("models", 0)}/14; '
        f'grids {count.get("grids", 0)}/14.', '',
        'Upper Sco uses H2O at 11.5–12.2 and 15.6–18.6 µm, main carbon '
        'species at 12.9–16.25 µm, HC3N at 15.0–15.2 µm, and C4H2 at '
        '15.8–16.0 µm. MR6 has one H2O component. The seven young disks '
        'use water only, fitted over 12–25 µm with the 13–14.4 µm organic '
        'complex masked. Their paper used 12–27 µm, so the fitted physical '
        'parameters are not exact reproductions. The younger paper’s cold '
        'component is labeled H2O_warm here to match the fitter output.', '',
        'Both methods use the same masks for each source. The v2 H2O '
        'grid spectra and checkpoints assume 140 pc; each fit uses the '
        'published source distance and reports the corresponding physical '
        'area A in au². The source distance can be overridden at run time. '
        'Distance is supplied rather than inferred because the spectrum '
        'alone constrains only A/d². '
        'The v2 H2O '
        'checkpoint is trained only to logN=19 and 25 µm; the paper’s MR5 '
        'hot-water value (logN=19.78) lies outside that domain. The grid '
        'method is constrained to the same logN interval for a direct '
        'comparison. Grid error bars include the local full grid step in '
        'quadrature; model error bars are optimizer summaries. Neither is '
        'equivalent to the papers’ posterior intervals. The young paper '
        'also used 1 km/s turbulence and different input reductions, which '
        'can affect fitted columns and areas. The isotope components here '
        'have independent T and A, while the Upper Sco paper ties them to '
        'their parent species; their fitted values are exploratory.', '',
    ]
    if len(quality):
        pivot = quality.pivot(index='source', columns='method', values='rms_Jy')
        lines += ['## Spectral residual RMS on fitted pixels', '',
                  '| Source | Models (Jy) | Grids (Jy) |',
                  '|---|---:|---:|']
        for source in SOURCES:
            if source in pivot.index:
                row = pivot.loc[source]
                fmt = lambda x: '—' if pd.isna(x) else f'{x:.4g}'
                lines.append(f'| {source} | {fmt(row.get("models", np.nan))} | '
                             f'{fmt(row.get("grids", np.nan))} |')
        lines.append('')
        paired = pivot.dropna(subset=['models', 'grids'])
        if len(paired):
            better_grid = int((paired['grids'] < paired['models']).sum())
            lines.append(f'The grid fit has a lower fitted-pixel RMS for '
                         f'{better_grid} of {len(paired)} paired sources. '
                         'These are residual scores, not evidence that one '
                         'physical parameter set is uniquely correct.')
            median_difference = 100 * np.median(paired.grids / paired.models - 1)
            lines.append(f'The median grid/model RMS difference is '
                         f'{median_difference:+.2f}%.')
            lines.append('')
    if len(params):
        rows = []
        for sample, source_set in [('young', set(YOUNG)), ('Upper Sco', set(USCO))]:
            for method in ('models', 'grids'):
                selected = params[(params.source.isin(source_set)) &
                                  (params.method == method) &
                                  params.paper_T_K.notna()]
                if len(selected):
                    rows.append((sample, method, len(selected),
                                 np.median(np.abs(selected.T_K - selected.paper_T_K)),
                                 np.nanmedian(np.abs(selected.logN - selected.paper_logN)),
                                 np.nanmedian(np.abs(selected.logA - selected.paper_logA))))
        if rows:
            lines += ['## Median absolute difference from reported fits', '',
                      '| Sample | Method | Components | T (K) | logN (dex) | logA (dex) |',
                      '|---|---|---:|---:|---:|---:|']
            for sample, method, count, T, logN, logA in rows:
                lines.append(f'| {sample} | {method} | {count} | {T:.1f} | '
                             f'{logN:.3f} | {logA:.3f} |')
            lines.append('')
    if len(params):
        water = params[params.component.str.startswith('H2O')]
        if len(water):
            lines += ['## Water temperatures', '',
                      '| Source | Component | Paper (K) | Models (K) | Grids (K) |',
                      '|---|---|---:|---:|---:|']
            for (source, component), group in water.groupby(['source', 'component']):
                by_method = group.set_index('method')
                paper = group.iloc[0].paper_T_K
                model = by_method.loc['models', 'T_K'] if 'models' in by_method.index else np.nan
                grid = by_method.loc['grids', 'T_K'] if 'grids' in by_method.index else np.nan
                fmt = lambda x: '—' if pd.isna(x) else f'{x:.0f}'
                lines.append(f'| {source} | {component} | {fmt(paper)} | '
                             f'{fmt(model)} | {fmt(grid)} |')
            lines.append('')
    lines += [
        'All fitted T, logN, and logA values are in '
        '[parameter_comparison.csv](parameter_comparison.csv); spectral '
        'scores are in [spectral_quality.csv](spectral_quality.csv). '
        'Each method has its own source CSVs and v1-style stage and '
        'combined plots under `realobs_results/v2_paper_comparison/` and '
        '`figures/realobs/v2_paper_comparison/`.', '',
    ]
    if summary_figure is not None:
        lines += ['[Water parameter comparison](../../figures/realobs/'
                  'v2_paper_comparison/water_parameter_comparison.png).', '']
    if comparison_figures:
        lines += ['## Paper versus fitted parameters', '',
                  'One figure per molecule compares paper values on the x axis '
                  'with model and grid fits on the y axis for log N, T, log A, '
                  'and log(N A). Both axes show their available asymmetric '
                  'errors. The gray band shows one sample standard deviation '
                  'of fitted minus paper values; the annotations show its '
                  'mean and standard deviation. Full statistics are in '
                  '[parameter_scatter.csv](parameter_scatter.csv). The N A '
                  'uncertainties combine log N and log A errors in quadrature '
                  'without covariance; these are approximate. Isotope T and '
                  'A are copied from the tied parent species in the paper. '
                  'Paper isotope N errors were not reported, and fixed paper '
                  'areas have zero error. Error bars beyond a panel limit are '
                  'clipped visually; parameter_comparison.csv retains their '
                  'full values.', '']
        for figure in comparison_figures:
            relative = figure.relative_to(ROOT)
            lines.append(f'- [{figure.stem}]'
                         f'(../../{relative.as_posix()})')
        lines.append('')
    path = RESULTS / 'REPORT.md'
    path.write_text('\n'.join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument('--methods', nargs='+', choices=['models', 'grids'],
                        default=['models', 'grids'])
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--distance-pc', type=float,
                        help='Override published distance (one selected source only)')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--n-samples', type=int, default=4096)
    parser.add_argument('--n-refine', type=int, default=12)
    parser.add_argument('--grid-maxiter', type=int, default=30)
    parser.add_argument('--grid-popsize', type=int, default=5)
    args = parser.parse_args()
    if args.distance_pc is not None and len(args.sources) != 1:
        parser.error('--distance-pc requires exactly one --sources value')
    if not args.report_only:
        for source in args.sources:
            for method in args.methods:
                run_source(source, method, overwrite=args.overwrite,
                           distance_pc=args.distance_pc,
                           n_samples=args.n_samples, n_refine=args.n_refine,
                           grid_maxiter=args.grid_maxiter,
                           grid_popsize=args.grid_popsize)
                print(f'Updated {write_report()}', flush=True)
    else:
        print(f'Updated {write_report()}', flush=True)


if __name__ == '__main__':
    main()
