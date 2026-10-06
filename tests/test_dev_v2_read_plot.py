"""Saved observation plots must reflect the CSV products from the fit."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


pytest.importorskip('torch')
from examples.dev_v2_read_plot import plot_saved_observation, read_stage_result


@pytest.fixture
def saved_observation(tmp_path):
    wav = np.array([11.0, 12.0, 13.0])
    observed = np.array([0.12, 0.21, 0.11])
    model = np.array([0.10, 0.20, 0.10])
    residual = observed - model
    spectrum = tmp_path / 'observed.csv'
    pd.DataFrame({'wave': wav, 'flux': observed}).to_csv(spectrum, index=False)
    prefix = tmp_path / 'source_C2H2'
    pd.DataFrame({
        'wave': wav, 'flux': observed, 'model': model,
        'residual': residual, 'C2H2_model': model,
    }).to_csv(Path(f'{prefix}_fit_spectrum.csv'), index=False)
    pd.DataFrame([{
        'component': 'C2H2', 'model_mol': 'C2H2', 'T': 500.0,
        'T_minus': 10.0, 'T_plus': 12.0,
        'logN': 16.0, 'logN_minus': 0.1, 'logN_plus': 0.2,
        'A': 1.0, 'log10A': 0.0,
        'log10A_minus': 0.05, 'log10A_plus': 0.06,
    }]).to_csv(Path(f'{prefix}_fit_params.csv'), index=False)
    running = Path(f'{prefix}_running.csv')
    pd.DataFrame({
        'wave': wav, 'residual': residual, 'cumulative_model': model,
    }).to_csv(running, index=False)
    pd.DataFrame({
        'wave': wav, 'residual': residual, 'cumulative_model': model,
    }).to_csv(tmp_path / 'source_final_running.csv', index=False)
    return spectrum, running


def test_saved_stage_can_be_replotted_without_models(saved_observation, tmp_path):
    spectrum, _ = saved_observation
    result = read_stage_result(tmp_path, 'source', 'C2H2')
    assert result['params']['C2H2']['T'] == 500.0
    figures = plot_saved_observation(
        spectrum, tmp_path,
        [{'name': 'C2H2', 'wavelength_range': (11, 13),
          'fit_ranges': [(11, 12), (12, 13)]}],
        name='source', figure_dir=tmp_path / 'figures',
        active_stages=['C2H2'], line_free_windows=(), skiprows=1,
    )
    assert figures['C2H2'].is_file()
    assert figures['combined'].is_file()
    plotted = pd.read_csv(figures['plot_data'])
    np.testing.assert_allclose(plotted['observed_flux'] - plotted['total_model'],
                               plotted['residual'])


def test_saved_stage_rejects_inconsistent_running_spectrum(saved_observation, tmp_path):
    _, running_path = saved_observation
    running = pd.read_csv(running_path)
    running.loc[1, 'residual'] = 1.0
    running.to_csv(running_path, index=False)
    with pytest.raises(ValueError, match='running residual'):
        read_stage_result(tmp_path, 'source', 'C2H2')
