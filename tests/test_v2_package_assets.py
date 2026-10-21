"""Production v2 inference must work without local slab grids."""

import numpy as np
import pytest


pytest.importorskip('torch')
from diskmelts import (fit_stage, generate_spectrum, load_fitting_models,
                       fit_grid_stage, fit_observation, synthesize_spectrum)


def test_synthesize_spectrum_area_distance_and_wavelength():
    wave = np.linspace(13.8, 14.2, 80)
    returned_wave, flux = synthesize_spectrum(
        'HCN', 300, 18, A=2, sigma=2, wavelength=wave)
    _, distant_flux = synthesize_spectrum(
        'HCN', 300, 18, A=2, sigma=2, wavelength=wave, distance_pc=280)
    model = load_fitting_models('HCN', sigma=2)['HCN']
    expected = generate_spectrum(300, 18, 2, model, obs_wav=wave)
    assert np.array_equal(returned_wave, wave)
    np.testing.assert_allclose(flux, expected)
    np.testing.assert_allclose(distant_flux, flux / 4)
    assert np.isfinite(flux).all() and flux.max() > 0
    with pytest.raises(ValueError, match='trained range'):
        synthesize_spectrum('C6H6', 700, 18, sigma=2)


def test_fit_observation_accepts_arrays_with_errors(tmp_path):
    wave = np.linspace(13.8, 14.2, 81)
    _, flux = synthesize_spectrum('HCN', 300, 18, wavelength=wave)
    error = np.full(wave.shape, 0.01)
    flux[0] = np.nan
    error[-1] = np.nan
    stages = [{'name': 'hcn', 'mol': 'HCN',
               'wavelength_range': (11.5, 17.0),
               'fit_ranges': (13.8, 14.2)}]
    result = fit_observation(
        stages=stages, obs_wav=wave, obs_flux=flux, obs_err=error,
        name='array_source', output_dir=tmp_path,
        detection_screening=False, n_samples=64, n_refine=2, n_top=2,
    )
    assert result['sigma_noise'] == pytest.approx(0.01)
    assert result['obs_wav'].size == wave.size - 1
    assert np.isnan(result['obs_err'][-1])
    assert 'hcn' in result['fits']
    assert np.isfinite(result['fits']['hcn']['model_flux']).all()
    assert (tmp_path / 'array_source_hcn_running.csv').is_file()


def test_fit_observation_rejects_mixed_and_misaligned_input(tmp_path):
    stages = [{'name': 'hcn', 'mol': 'HCN',
               'wavelength_range': (11.5, 17.0),
               'fit_ranges': (13.8, 14.2)}]
    with pytest.raises(ValueError, match='either input_path'):
        fit_observation('spectrum.csv', stages, obs_wav=[13.8], obs_flux=[1])
    with pytest.raises(ValueError, match='obs_err must match'):
        fit_observation(stages=stages, obs_wav=[13.8, 13.9],
                        obs_flux=[1, 1], obs_err=[0.01])


def test_bundled_model_fit_without_grids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / 'Model_grids').exists()
    models = load_fitting_models('C2H2')
    wave = np.linspace(13.5, 14.1, 65)
    flux = generate_spectrum(500, 16, 1.0, models['C2H2'], obs_wav=wave)
    fit = fit_stage(
        wave, flux, 'C2H2', (13.5, 14.1), (13.5, 14.1), models,
        n_samples=64, n_refine=2, n_top=2, seed=4,
    )
    assert fit['params']['C2H2']['T'] == pytest.approx(500, abs=2)
    assert fit['params']['C2H2']['logN'] == pytest.approx(16, abs=0.02)
    with pytest.raises(ValueError, match='grid_root is required'):
        fit_grid_stage(wave, flux, 'C2H2', (13.5, 14.1), (13.5, 14.1))


def test_sigma2_water_checkpoint_selection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = load_fitting_models('H2O', sigma=0)['H2O']
    broadened = load_fitting_models('H2O', sigma=2)['H2O']
    assert original['manifest_path'] != broadened['manifest_path']
    wave = np.linspace(16.7, 17.2, 70)
    flux = generate_spectrum(400, 16, 1.0, broadened, obs_wav=wave)
    assert np.all(np.isfinite(flux))
    assert np.linalg.norm(flux) > 0
    with pytest.raises(ValueError, match='sigma must be 0 or 2'):
        load_fitting_models('H2O', sigma=1)
    with pytest.raises(ValueError, match='sigma=2'):
        load_fitting_models('NO_SUCH_MOLECULE', sigma=2)


def test_sigma2_checkpoints_cover_trained_bands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    molecules = ('C2H2', 'HCN', 'CO2', '13C12CH2', '13CO2',
                 'C4H2', 'HC3N', 'C2H6', 'C2H4', 'CH4', 'C2HD',
                 'CO17O', 'CO18O', 'C3H4', 'CH3', 'C6H6', 'OH')
    models = load_fitting_models(molecules, sigma=2)
    for molecule in molecules:
        model = models[molecule]
        assert model.get('grid_sigma') in (None, 2)
        if model.get('is_model_bank'):
            low, high = model['wav_ranges'][-1]
        else:
            low, high = float(model['wav'][0]), float(model['wav'][-1])
        wave = np.linspace(low, high, 80)
        flux = generate_spectrum(500, 16, 1, model, obs_wav=wave)
        assert np.isfinite(flux).all()
        assert np.max(flux) > 0


@pytest.mark.parametrize('grid_sigma', [0, 2])
def test_c2h2_bank_fits_both_bands(tmp_path, monkeypatch, grid_sigma):
    monkeypatch.chdir(tmp_path)
    models = load_fitting_models('C2H2', sigma=grid_sigma)
    c2h2 = models['C2H2']
    assert c2h2['is_model_bank']
    assert c2h2['wav_ranges'] == [(6.0, 9.0), (11.5, 17.0)]
    short_wave = np.linspace(7.3, 7.8, 80)
    assert np.max(generate_spectrum(500, 16, 1, c2h2, obs_wav=short_wave)) > 0
    gap_wave = np.linspace(9.1, 11.4, 20)
    assert np.all(generate_spectrum(500, 16, 1, c2h2, obs_wav=gap_wave) == 0)
    combined_wave = np.concatenate((np.linspace(7.3, 7.8, 40),
                                    np.linspace(13.5, 14.0, 40)))
    combined_flux = generate_spectrum(500, 16, 1, c2h2, obs_wav=combined_wave)
    fit = fit_stage(
        combined_wave, combined_flux, 'C2H2', (6.0, 17.0),
        [(7.3, 7.8), (13.5, 14.0)], models,
        n_samples=64, n_refine=2, n_top=2, seed=4,
    )
    assert np.linalg.norm(fit['residual']) < 0.05 * np.linalg.norm(combined_flux)


def test_sigma2_c6h6_fits_without_grids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from diskmelts import SIGMA2_MODEL_SPECS

    models = load_fitting_models('C6H6', sigma=2)
    assert SIGMA2_MODEL_SPECS['C6H6']['T_bounds'] == (100.0, 675.0)
    wave = np.linspace(14.7, 14.95, 100)
    flux = generate_spectrum(400, 16, 1, models['C6H6'], obs_wav=wave)
    fit = fit_stage(
        wave, flux, 'C6H6', (13.0, 17.0), (14.7, 14.95), models,
        n_samples=128, n_refine=4, n_top=4, seed=4,
    )
    assert np.linalg.norm(fit['residual']) < 0.05 * np.linalg.norm(flux)


@pytest.mark.parametrize('grid_sigma', [0, 2])
def test_oh_fit_stage_infers_registry(tmp_path, monkeypatch, grid_sigma):
    monkeypatch.chdir(tmp_path)
    models = load_fitting_models('OH', sigma=grid_sigma)
    wave = np.linspace(16.0, 17.0, 80)
    flux = generate_spectrum(500, 16, 1, models['OH'], obs_wav=wave)
    fit = fit_stage(
        wave, flux, 'OH', (15.0, 26.0), (16.0, 17.0), models,
        n_samples=64, n_refine=2, n_top=2, seed=4,
    )
    assert 'OH' in fit['params']


@pytest.mark.parametrize('molecule,window', [
    ('C6H6', (14.5, 15.2)),
    ('C3H4', (15.68, 15.85)),
    ('CH3', (16.4, 16.6)),
])
def test_new_bundled_molecules_fit_without_grids(tmp_path, monkeypatch,
                                                  molecule, window):
    monkeypatch.chdir(tmp_path)
    models = load_fitting_models(molecule)
    wave = np.linspace(*window, 100)
    flux = generate_spectrum(500, 16, 1.0, models[molecule], obs_wav=wave)
    fit = fit_stage(
        wave, flux, molecule, window, window, models,
        n_samples=64, n_refine=2, n_top=2, seed=4,
    )
    assert molecule in fit['params']
    assert np.isfinite(fit['model_flux']).all()
    assert np.linalg.norm(fit['residual']) < np.linalg.norm(flux)
