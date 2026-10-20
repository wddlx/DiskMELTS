"""Production v2 inference must work without local slab grids."""

import numpy as np
import pytest


pytest.importorskip('torch')
from diskmelts import (fit_stage, generate_spectrum, load_fitting_models,
                       fit_grid_stage)


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


def test_sigma2_c2h2_bank_fits_both_bands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = load_fitting_models('C2H2', sigma=2)
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
    from diskmelts import SIGMA2_MODEL_SPECS
    fit = fit_stage(
        combined_wave, combined_flux, 'C2H2', (6.0, 17.0),
        [(7.3, 7.8), (13.5, 14.0)], models,
        model_specs=SIGMA2_MODEL_SPECS,
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


def test_sigma2_only_species_fit_stage_infers_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    models = load_fitting_models('OH', sigma=2)
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
