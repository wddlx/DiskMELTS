"""Grid-only fitting recovers a known spectrum and records grid-limited errors."""

import numpy as np
import pandas as pd
import pytest


pytest.importorskip('torch')
from diskmelts.v2_grid_fitting import (
    combine_grid_uncertainty,
    fit_grid_stage,
    fit_observation_grids,
    load_grid_cube,
)


def _small_grid(tmp_path, molecule='C2H2', wavelength_range=(11.0, 13.0)):
    grid_dir = tmp_path / 'grids' / molecule
    grid_dir.mkdir(parents=True)
    lo, hi = wavelength_range
    wav = np.linspace(lo, hi, 60)
    basis = np.stack([np.exp(-((wav - center) / 0.09) ** 2)
                      for center in (lo + 0.3, (lo + hi) / 2, hi - 0.3)])

    def spectrum(T, logN):
        return 0.01 * (basis[0] + (1 + (T - 100) / 25) * basis[1]
                       + (1 + (logN - 13) / 0.125) * basis[2])

    for T in (100, 125, 150):
        for logN in (13.0, 13.125, 13.25):
            pd.DataFrame({'wave': wav, 'Line': spectrum(T, logN)}).to_csv(
                grid_dir / f'T{T}N{logN}.csv', index=False,
            )
    return wav, spectrum


@pytest.mark.parametrize('molecule,wavelength_range', [
    ('C6H6', (13.0, 17.0)),
    ('C3H4', (15.0, 18.0)),
    ('CH3', (15.0, 18.0)),
])
def test_new_molecules_fit_from_direct_grids(tmp_path, molecule,
                                             wavelength_range):
    wav, spectrum = _small_grid(tmp_path, molecule, wavelength_range)
    observed = spectrum(113.0, 13.07)
    fit = fit_grid_stage(
        wav, observed, molecule, wavelength_range, wavelength_range,
        grid_root=tmp_path / 'grids', T_bounds=(100, 150),
        logN_bounds=(13, 13.25), sigma=1e-4,
        maxiter=20, popsize=5, n_refine=2,
    )
    assert molecule in fit['params']
    assert np.isfinite(fit['model_flux']).all()
    assert np.linalg.norm(fit['residual']) < np.linalg.norm(observed)


def test_grid_fit_and_saved_errors(tmp_path):
    wav, spectrum = _small_grid(tmp_path)
    distance_pc = 177.0
    observed = 1.3 * (140.0 / distance_pc) ** 2 * spectrum(113.0, 13.07)
    fit = fit_grid_stage(
        wav, observed, 'C2H2', (11, 13), (11, 13),
        grid_root=tmp_path / 'grids', T_bounds=(100, 150),
        logN_bounds=(13, 13.25), sigma=1e-4,
        distance_pc=distance_pc,
        maxiter=30, popsize=5, n_refine=2,
    )
    param = fit['params']['C2H2']
    detail = fit['uncertainty_details']['C2H2']
    assert param['T'] == pytest.approx(113, abs=1)
    assert param['logN'] == pytest.approx(13.07, abs=0.01)
    assert param['A'] == pytest.approx(1.3, abs=0.02)
    assert detail['T_grid_step'] == 25
    assert detail['logN_grid_step'] == 0.125
    assert detail['T_total_sigma'] == pytest.approx(
        np.hypot(detail['T_fit_sigma'], 25),
    )
    assert detail['logN_total_sigma'] == pytest.approx(
        np.hypot(detail['logN_fit_sigma'], 0.125),
    )
    assert combine_grid_uncertainty(3, 25) == pytest.approx(np.hypot(3, 25))

    input_path = tmp_path / 'observation.csv'
    pd.DataFrame({'wave': wav, 'flux': observed}).to_csv(input_path, index=False)
    stages = [{'name': 'C2H2', 'mol': 'C2H2',
               'wavelength_range': (11, 13), 'fit_ranges': (11, 13),
               'T_bounds': (100, 150), 'logN_bounds': (13, 13.25)}]
    saved = fit_observation_grids(
        input_path, stages, name='demo', grid_root=tmp_path / 'grids',
        output_dir=tmp_path / 'results', sigma_noise=1e-4,
        distance_pc=distance_pc,
        detection_screening=False, maxiter=20, popsize=5, n_refine=2,
    )
    error_table = pd.read_csv(saved['products']['C2H2']['grid_uncertainty'])
    assert error_table.loc[0, 'T_total_sigma'] >= 25
    assert error_table.loc[0, 'logN_total_sigma'] >= 0.125


def test_grid_must_be_complete(tmp_path):
    wav, _ = _small_grid(tmp_path)
    (tmp_path / 'grids' / 'C2H2' / 'T125N13.125.csv').unlink()
    with pytest.raises(ValueError, match='incomplete'):
        load_grid_cube('C2H2', wav, grid_root=tmp_path / 'grids',
                       T_bounds=(100, 150), logN_bounds=(13, 13.25))


def test_fixed_area_is_excluded_from_covariance(tmp_path):
    wav, spectrum = _small_grid(tmp_path)
    fit = fit_grid_stage(
        wav, spectrum(113, 13.07), 'C2H2', (11, 13), (11, 13),
        grid_root=tmp_path / 'grids', T_bounds=(100, 150),
        logN_bounds=(13, 13.25), loga_bounds=(-1e-6, 1e-6),
        sigma=1e-4, maxiter=20, popsize=5, n_refine=2,
    )
    details = fit['uncertainty_details']['C2H2']
    assert fit['params']['C2H2']['A'] == pytest.approx(1, abs=1e-5)
    assert details['log10A_fit_sigma'] == 0
    assert details['T_total_sigma'] >= 25
    assert details['logN_total_sigma'] >= 0.125


def test_grid_stage_rejects_unapplied_isotope_constraints(tmp_path):
    wav, spectrum = _small_grid(tmp_path)
    input_path = tmp_path / 'observation.csv'
    pd.DataFrame({'wave': wav, 'flux': spectrum(113, 13.07)}).to_csv(
        input_path, index=False,
    )
    stages = [{'name': 'carbon', 'mol': ['C2H2', '13C12CH2'],
               'wavelength_range': (11, 13), 'fit_ranges': (11, 13),
               'isotopologue_groups': [('C2H2', '13C12CH2')],
               'Tshare': ['less'], 'Ashare': [True]}]
    with pytest.raises(ValueError, match='does not support.*Tshare'):
        fit_observation_grids(
            input_path, stages, grid_root=tmp_path / 'grids',
            output_dir=tmp_path / 'results', detection_screening=False,
        )
    assert not (tmp_path / 'results').exists()


def test_repeated_molecule_has_independent_component_bounds(tmp_path):
    wav, spectrum = _small_grid(tmp_path)
    observed = 0.7 * spectrum(110, 13.04) + 1.4 * spectrum(140, 13.20)
    stage = {
        'name': 'acetylene', 'mol': ['C2H2', 'C2H2'],
        'component_names': ['C2H2_thin', 'C2H2_thick'],
        'wavelength_range': (11, 13), 'fit_ranges': (11, 13),
        'T_bounds': (100, 150),
        'logN_bounds': {'C2H2_thin': (13.0, 13.1),
                        'C2H2_thick': (13.15, 13.25)},
        'loga_bounds': {'C2H2_thin': (-1.0, 0.0),
                        'C2H2_thick': (0.0, 1.0)},
    }
    input_path = tmp_path / 'observation.csv'
    pd.DataFrame({'wave': wav, 'flux': observed}).to_csv(input_path, index=False)
    result = fit_observation_grids(
        input_path, [stage], grid_root=tmp_path / 'grids',
        output_dir=tmp_path / 'results', sigma_noise=1e-4,
        detection_screening=False, maxiter=12, popsize=5, n_refine=2,
    )
    params = result['fits']['acetylene']['params']
    assert set(params) == {'C2H2_thin', 'C2H2_thick'}
    assert 13.0 <= params['C2H2_thin']['logN'] <= 13.1
    assert 13.15 <= params['C2H2_thick']['logN'] <= 13.25
    assert 0.1 <= params['C2H2_thin']['A'] <= 1.0
    assert 1.0 <= params['C2H2_thick']['A'] <= 10.0
    spectrum_csv = pd.read_csv(result['products']['acetylene']['spectrum'])
    assert {'C2H2_thin_model', 'C2H2_thick_model'} <= set(spectrum_csv.columns)


def test_repeated_grid_molecule_requires_unique_labels(tmp_path):
    wav, spectrum = _small_grid(tmp_path)
    with pytest.raises(ValueError, match='component labels must be unique'):
        fit_grid_stage(
            wav, spectrum(125, 13.125), ['C2H2', 'C2H2'],
            (11, 13), (11, 13), grid_root=tmp_path / 'grids',
        )
