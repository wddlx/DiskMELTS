"""Read saved v2 observational fits and plot them against the data.

The figures follow the v1 package's two-panel ``plot_fit`` examples: observed
spectrum and fitted components above, residual and noise below. Reading the
CSV products makes plotting independent of the expensive fitting run.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from diskmelts.fitting import load_observed_spectrum
from diskmelts.plotting import plot_fit


LINE_FREE_WINDOWS = ((11.45, 11.55), (13.10, 13.20),
                     (15.65, 15.72), (15.90, 15.91))


def _ranges(value, active_molecules=None):
    if isinstance(value, dict):
        selected = [entry for key, entry in value.items()
                    if active_molecules is None or key in active_molecules]
        return [pair for entry in selected for pair in _ranges(entry)]
    if len(value) == 2 and all(np.isscalar(item) for item in value):
        return [tuple(map(float, value))]
    return [tuple(map(float, pair)) for pair in value]


def _same_array(actual, expected, description):
    if actual.shape != expected.shape or not np.allclose(
        actual, expected, rtol=1e-7, atol=1e-12, equal_nan=True,
    ):
        raise ValueError(f'{description} disagrees with the saved fitting sequence')


def read_stage_result(result_dir, source_name, stage_name):
    """Reconstruct one fit result from its spectrum, parameter, and running CSVs."""
    prefix = Path(result_dir) / f'{source_name}_{stage_name}'
    paths = {
        'spectrum': Path(f'{prefix}_fit_spectrum.csv'),
        'params': Path(f'{prefix}_fit_params.csv'),
        'running': Path(f'{prefix}_running.csv'),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing saved fit product(s):\n' + '\n'.join(missing))
    spectrum = pd.read_csv(paths['spectrum'])
    parameter_rows = pd.read_csv(paths['params'])
    running = pd.read_csv(paths['running'])
    required_spectrum = {'wave', 'flux', 'model', 'residual'}
    required_params = {'component', 'model_mol', 'T', 'logN', 'A', 'log10A'}
    if not required_spectrum.issubset(spectrum) or not required_params.issubset(parameter_rows):
        raise ValueError(f'Incomplete saved fit CSV for {stage_name}')

    params = {}
    uncertainty = {}
    component_fluxes = {}
    for row in parameter_rows.itertuples(index=False):
        label = str(row.component)
        column = f'{label}_model'
        if column not in spectrum:
            raise ValueError(f'{stage_name} spectrum has no {column} column')
        component_fluxes[label] = spectrum[column].to_numpy(dtype=float)
        params[label] = {
            'T': float(row.T), 'logN': float(row.logN),
            'A': float(row.A), 'log10A': float(row.log10A),
            'model_mol': str(row.model_mol),
        }
        uncertainty[label] = {
            key: {'minus': float(getattr(row, f'{key}_minus')),
                  'plus': float(getattr(row, f'{key}_plus'))}
            for key in ('T', 'logN', 'log10A')
        }

    wav = spectrum['wave'].to_numpy(dtype=float)
    flux = spectrum['flux'].to_numpy(dtype=float)
    model = spectrum['model'].to_numpy(dtype=float)
    residual = spectrum['residual'].to_numpy(dtype=float)
    _same_array(sum(component_fluxes.values(), np.zeros_like(model)), model,
                f'{stage_name} component sum')
    _same_array(flux - model, residual, f'{stage_name} residual')
    _same_array(running['wave'].to_numpy(dtype=float), wav,
                f'{stage_name} running wavelength')
    _same_array(running['residual'].to_numpy(dtype=float), residual,
                f'{stage_name} running residual')
    return {
        'obs_wav': wav, 'obs_flux': flux, 'model_flux': model,
        'residual': residual, 'params': params,
        'uncertainty': uncertainty, 'component_fluxes': component_fluxes,
        'running_cumulative': running['cumulative_model'].to_numpy(dtype=float),
        'paths': paths,
    }


def plot_saved_observation(input_path, result_dir, stages, *, name=None,
                           figure_dir=None, active_stages=None,
                           line_free_windows=LINE_FREE_WINDOWS,
                           sigma_noise=None, skiprows=1, delimiter=',',
                           wav_col=0, flux_col=1):
    """Read stage CSVs, validate their subtraction sequence, and save v1-style plots.

    ``active_stages`` should be the names returned by a fresh fitting run. If
    omitted, stages with complete saved CSV products are plotted. This also
    permits replotting without loading a trained checkpoint or fitting again.
    """
    input_path = Path(input_path)
    result_dir = Path(result_dir)
    figure_dir = Path(figure_dir) if figure_dir else result_dir / 'figures'
    source_name = name or input_path.stem
    obs_wav, obs_flux = load_observed_spectrum(
        input_path, skiprows=skiprows, delimiter=delimiter,
        wav_col=wav_col, flux_col=flux_col,
    )
    if sigma_noise is None and line_free_windows:
        chunks = [obs_flux[(obs_wav >= lo) & (obs_wav <= hi)]
                  for lo, hi in line_free_windows
                  if np.any((obs_wav >= lo) & (obs_wav <= hi))]
        if chunks:
            sigma_noise = float(np.nanstd(np.concatenate(chunks)))
    requested = None if active_stages is None else set(active_stages)
    configured = {stage['name'] for stage in stages}
    if requested is not None and not requested <= configured:
        raise ValueError(f'Unknown active stages: {sorted(requested - configured)}')
    figure_dir.mkdir(parents=True, exist_ok=True)

    prior_residual = obs_flux.copy()
    cumulative = np.zeros_like(obs_flux)
    fits = {}
    figures = {}
    plot_columns = {'wave': obs_wav, 'observed_flux': obs_flux}
    combined_ranges = []
    displayed_ranges = []
    for stage in stages:
        stage_name = stage['name']
        prefix = result_dir / f'{source_name}_{stage_name}'
        if requested is not None and stage_name not in requested:
            continue
        if requested is None and not Path(f'{prefix}_fit_spectrum.csv').is_file():
            continue
        fit = read_stage_result(result_dir, source_name, stage_name)
        _same_array(fit['obs_wav'], obs_wav, f'{stage_name} observed wavelength')
        _same_array(fit['obs_flux'], prior_residual, f'{stage_name} input flux')
        prior_residual = fit['residual']
        cumulative += fit['model_flux']
        _same_array(fit['running_cumulative'], cumulative,
                    f'{stage_name} cumulative model')
        fits[stage_name] = fit
        for label, flux in fit['component_fluxes'].items():
            plot_columns[f'{stage_name}_{label}_model'] = flux
        active_molecules = set(fit['params']) | {
            param['model_mol'] for param in fit['params'].values()
        }
        fit_ranges = _ranges(stage['fit_ranges'], active_molecules)
        for wav_range in fit_ranges:
            if wav_range not in combined_ranges:
                combined_ranges.append(wav_range)
        displayed_ranges.extend(_ranges(stage['wavelength_range']))
        wav_plot_range = (min(lo for lo, _ in _ranges(stage['wavelength_range'])),
                          max(hi for _, hi in _ranges(stage['wavelength_range'])))
        figure_path = figure_dir / f'{source_name}_{stage_name}_fit.png'
        fig = plot_fit(
            obs_wav, fit['obs_flux'], fit, sigma=sigma_noise,
            fit_ranges=fit_ranges, wav_plot_range=wav_plot_range,
            name=f'{source_name}_{stage_name}', save_path=str(figure_path),
            params=fit['params'], uncertainty=fit['uncertainty'], show=False,
        )
        plt.close(fig)
        figures[stage_name] = figure_path

    if fits:
        final_path = result_dir / f'{source_name}_final_running.csv'
        if not final_path.is_file():
            raise FileNotFoundError(f'Missing final running spectrum: {final_path}')
        final = pd.read_csv(final_path)
        _same_array(final['wave'].to_numpy(dtype=float), obs_wav,
                    'final wavelength')
        _same_array(final['residual'].to_numpy(dtype=float), prior_residual,
                    'final residual')
        _same_array(final['cumulative_model'].to_numpy(dtype=float), cumulative,
                    'final cumulative model')
        plot_columns['total_model'] = cumulative
        plot_columns['residual'] = prior_residual
        plot_data_path = result_dir / f'{source_name}_plot_data.csv'
        pd.DataFrame(plot_columns).to_csv(plot_data_path, index=False)
        params = {}
        uncertainty = {}
        for fit in fits.values():
            params.update(fit['params'])
            uncertainty.update(fit['uncertainty'])
        combined_path = figure_dir / f'{source_name}_combined_fit.png'
        fig = plot_fit(
            obs_wav, obs_flux, list(fits.values()), sigma=sigma_noise,
            fit_ranges=combined_ranges,
            wav_plot_range=(min(lo for lo, _ in displayed_ranges),
                            max(hi for _, hi in displayed_ranges)),
            name=source_name, save_path=str(combined_path),
            params=params, uncertainty=uncertainty, show=False,
        )
        plt.close(fig)
        figures['combined'] = combined_path
        figures['plot_data'] = plot_data_path
    return figures
