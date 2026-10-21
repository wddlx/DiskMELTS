"""Regression tests for fitting, training, validation, and continuum edge cases."""

import warnings

import numpy as np
import pytest


pytest.importorskip('torch')
from diskmelts import consub, fitting, trainmodel, validation
from diskmelts import v2_fitting


def _gaussian_models(monkeypatch, wave, centers):
    def fake_flux(model, T, logN, A, obs_wav):
        width = 0.05 + 0.1 * (T - 200.0) / 600.0
        return A * (logN - 13.0) * np.exp(
            -0.5 * ((obs_wav - centers[model['mol']]) / width) ** 2
        )

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    return {mol: {'mol': mol, 'wav': wave} for mol in centers}


def test_n_top_one_returns_one_solution(monkeypatch):
    wave = np.linspace(13.0, 14.0, 60)
    models = _gaussian_models(monkeypatch, wave, {'HCN': 13.5})
    observed = fitting._mol_flux_on_wav(models['HCN'], 400, 16, 1.0, wave)
    fit = fitting.fit_molecules(
        wave, observed, 'HCN', models, (13.0, 14.0),
        T_bounds=(200, 800), logN_bounds=(14, 18),
        n_samples=16, n_refine=1, n_top=1, seed=1, verbose=False,
    )
    assert len(fit['top_list']) == 1
    assert len(fitting._distinct_solutions(
        [{'x_scaled': np.array([v])} for v in (0.0, 0.5, 1.0)], 1, None)) == 1


def test_nonfinite_flux_is_excluded_from_fit(monkeypatch):
    wave = np.linspace(13.0, 14.0, 60)
    models = _gaussian_models(monkeypatch, wave, {'HCN': 13.5})
    observed = fitting._mol_flux_on_wav(models['HCN'], 400, 16, 1.0, wave)
    observed[[10, 30]] = np.nan
    fit = fitting.fit_molecules(
        wave, observed, 'HCN', models, (13.0, 14.0),
        T_bounds=(200, 800), logN_bounds=(14, 18),
        n_samples=16, n_refine=1, n_top=1, seed=1, verbose=False,
    )
    assert np.isfinite(fit['top_list'][0]['mse_loss'])
    assert fit['params']['HCN']['T'] == pytest.approx(400, rel=0.05)


def test_single_model_outside_fit_window_is_rejected(monkeypatch):
    wave = np.linspace(13.0, 14.0, 60)
    models = _gaussian_models(monkeypatch, np.linspace(5.0, 9.0, 10), {'CH4': 7.0})
    with pytest.raises(ValueError, match='outside all selected fit pixels'):
        fitting.fit_molecules(wave, np.ones_like(wave), 'CH4', models,
                              (13.0, 14.0), verbose=False)


def test_synthesize_spectrum_warns_outside_model_range():
    wav, _ = v2_fitting.synthesize_spectrum('HCN', 400, 16)
    wavelength = np.linspace(wav[0] - 0.5, wav[-1], 50)
    with pytest.warns(UserWarning, match='outside the HCN model range'):
        _, flux = v2_fitting.synthesize_spectrum('HCN', 400, 16, wavelength=wavelength)
    assert np.all(flux[wavelength < wav[0]] == 0)


def test_fit_stage_rescales_global_candidates(monkeypatch):
    def fake_fit_molecules(**kwargs):
        return {'model_flux': np.zeros(2), 'residual': np.zeros(2),
                'component_fluxes': {}, 'top_list': [{'mse_loss': 4.0}],
                'global_top': [{'mse_loss': 8.0}]}

    monkeypatch.setattr(v2_fitting, 'fit_molecules', fake_fit_molecules)
    fit = v2_fitting.fit_stage(
        np.array([12.0, 12.1]), np.zeros(2), 'C2H2', (12.0, 12.1),
        (12.0, 12.1), {'C2H2': object()}, distance_pc=280.0,
    )
    assert fit['global_top'][0]['mse_loss'] == pytest.approx(0.5)


def test_screening_drops_undetected_isotopologue_from_groups(monkeypatch, tmp_path):
    captured = {}

    def fake_fit_stage(obs_wav, obs_flux, molecule, *args, **kwargs):
        captured.update(molecule=molecule, **kwargs)
        n = len(obs_wav)
        return {'model_flux': np.zeros(n), 'residual': np.zeros(n),
                'component_fluxes': {}, 'top_list': [], 'params': {}}

    monkeypatch.setattr(v2_fitting, 'fit_stage', fake_fit_stage)
    monkeypatch.setattr(v2_fitting, 'load_fitting_models',
                        lambda mols, **kw: {m: {} for m in mols})
    monkeypatch.setattr(v2_fitting, 'detect_stage_molecules',
                        lambda *a, **kw: ['C2H2', 'CO2'])
    for name in ('save_fit_outputs', 'save_running_spectrum', 'print_fit_params',
                 'save_fitted_comparison'):
        monkeypatch.setattr(v2_fitting, name, lambda *a, **kw: {})
    stage = {
        'name': 'carbon', 'mol': ['C2H2', '13C12CH2', 'CO2', '13CO2'],
        'wavelength_range': (13.0, 16.0), 'fit_ranges': (13.0, 16.0),
        'isotopologue_groups': [('C2H2', '13C12CH2'), ('CO2', '13CO2')],
        'Tshare': ['less', 'equal'], 'Ashare': [True, True],
        'shared_area_groups': [('C2H2', 'CO2')],
    }
    wave = np.linspace(13.0, 16.0, 50)
    v2_fitting.fit_observation(
        stages=[stage], obs_wav=wave, obs_flux=np.zeros_like(wave),
        obs_err=np.full_like(wave, 0.01), name='s', output_dir=tmp_path,
    )
    assert captured['molecule'] == ['C2H2', 'CO2']
    assert captured['isotopologue_groups'] is None
    assert captured['Tshare'] is None and captured['Ashare'] is None
    assert captured['shared_area_groups'] == [('C2H2', 'CO2')]


def test_screened_options_keep_surviving_groups_and_names():
    stage = {'mol': ['H2O', 'C2H2', '13C12CH2', 'CO2', '13CO2'],
             'h2o_components': 2, 'component_names': ['warm', 'hot', 'C2H2',
                                                      '13C12CH2', 'CO2', '13CO2'],
             'isotopologue_groups': [('C2H2', '13C12CH2'), ('CO2', '13CO2')],
             'Tshare': ['less', 'equal'], 'Ashare': [True, False]}
    options = v2_fitting._screened_group_options(stage, ['H2O', 'CO2', '13CO2'])
    assert options['component_names'] == ['warm', 'hot', 'CO2', '13CO2']
    assert options['isotopologue_groups'] == [('CO2', '13CO2')]
    assert options['Tshare'] == ['equal']
    assert options['Ashare'] == [False]


def test_weight_keys_follow_layer_order():
    state = {f'net.{i}.weight': None for i in (0, 2, 4, 6, 8, 10, 12)}
    state['net.0.bias'] = None
    assert trainmodel._weight_keys(state)[-2:] == ['net.10.weight', 'net.12.weight']


def test_nt_validation_selects_bank_tile():
    tile = {'wav': np.linspace(13.0, 15.0, 5)}
    bank = {'is_model_bank': True,
            'tiles': [{'wav_range': (5.0, 9.0), 'model': {}},
                      {'wav_range': (12.0, 16.0), 'model': tile}]}
    assert validation._single_model_for_ranges(bank, [(13.0, 14.0)]) is tile
    with pytest.raises(ValueError, match='no model-bank tile covers'):
        validation._single_model_for_ranges(bank, [(8.0, 13.0)])


def test_continuum_extrapolates_slope_past_last_window():
    wave = np.linspace(5.0, 28.0, 4600)
    rng = np.random.default_rng(0)
    flux = 1.0 + 0.02 * (wave - 15.0) + rng.normal(0, 1e-3, wave.size)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = consub.subtract_continuum(wave, flux, error=np.full_like(wave, 1e-3))
    edge = wave > 27.2
    assert result.weak_support[edge].all()
    assert np.max(np.abs(result.flux[edge])) < 0.006


def test_arpls_converges_without_spurious_warning():
    wave = np.linspace(5.0, 25.0, 3000)
    flux = 1.0 + 0.01 * (wave - 15.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error', category=Warning)
        warnings.filterwarnings('ignore', message='Continuum is interpolated')
        result = consub.subtract_continuum(wave, flux, method='arpls', preset=None)
    assert result.attrs['converged']


def test_single_exclude_pair_is_accepted():
    wave = np.linspace(5.0, 25.0, 3000)
    flux = 1.0 + 0.01 * (wave - 15.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = consub.subtract_continuum(wave, flux, exclude=(14.0, 15.0),
                                           boxsizes=np.array(31))
    assert result.excluded[(wave > 14.1) & (wave < 14.9)].all()
