"""Selected isotopologues can share one fitted emitting area."""

import numpy as np
import pytest


pytest.importorskip('torch')
from diskmelts import fitting


def test_repeated_model_molecule_uses_component_bounds(monkeypatch):
    wave = np.linspace(13.0, 14.0, 41)
    bases = (np.exp(-((wave - 13.3) / 0.1) ** 2),
             np.exp(-((wave - 13.7) / 0.1) ** 2))

    def fake_flux(model, T, logN, A, obs_wav):
        shape = bases[0] + (logN - 14.0) * bases[1]
        return A * np.interp(obs_wav, wave, shape)

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    observed = fake_flux(None, 300, 14.5, 0.8, wave) + fake_flux(None, 500, 16.5, 1.5, wave)
    fit = fitting.fit_molecules(
        wave, observed, ['C2H2', 'C2H2'], {'C2H2': {'mol': 'C2H2'}},
        (13.0, 14.0), component_names=['C2H2_thin', 'C2H2_thick'],
        T_bounds=(200, 600),
        logN_bounds={'C2H2_thin': (14.0, 15.0),
                     'C2H2_thick': (16.0, 17.0)},
        loga_bounds={'C2H2_thin': (-1.0, 0.0),
                     'C2H2_thick': (0.0, 1.0)},
        n_samples=16, n_refine=2, n_top=2, seed=7, verbose=False,
    )
    params = fit['params']
    assert set(params) == {'C2H2_thin', 'C2H2_thick'}
    assert 14.0 <= params['C2H2_thin']['logN'] <= 15.0
    assert 16.0 <= params['C2H2_thick']['logN'] <= 17.0
    assert 0.1 <= params['C2H2_thin']['A'] <= 1.0
    assert 1.0 <= params['C2H2_thick']['A'] <= 10.0


def test_repeated_model_molecule_requires_unique_labels():
    with pytest.raises(ValueError, match='component labels must be unique'):
        fitting.fit_molecules(
            np.array([13.0, 14.0]), np.array([1.0, 1.0]),
            ['C2H2', 'C2H2'], {'C2H2': {}}, (13.0, 14.0),
            verbose=False,
        )


def test_shared_area_groups_use_one_amplitude_in_joint_fit(monkeypatch):
    wave = np.linspace(13.0, 16.0, 80)
    profiles = {
        'C2H2': np.exp(-0.5 * ((wave - 13.7) / 0.12) ** 2),
        '13C12CH2': np.exp(-0.5 * ((wave - 14.7) / 0.15) ** 2),
    }

    def fake_flux(model, T, logN, A, obs_wav):
        return A * np.interp(obs_wav, wave, profiles[model['mol']])

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    models = {mol: {'mol': mol} for mol in profiles}
    observed = 2.3 * (profiles['C2H2'] + profiles['13C12CH2'])
    fit = fitting.fit_molecules(
        wave, observed, ['C2H2', '13C12CH2'], models, (13.0, 16.0),
        T_bounds=(300.0, 500.0), logN_bounds=(15.0, 17.0),
        loga_bounds=(-1.0, 1.0),
        shared_area_groups=[('C2H2', '13C12CH2')],
        n_samples=8, n_refine=2, n_top=2, seed=1, verbose=False,
    )
    a_main = fit['params']['C2H2']['A']
    a_rare = fit['params']['13C12CH2']['A']
    assert fit['shared_area_groups'] == [('C2H2', '13C12CH2')]
    assert a_main == a_rare
    assert a_main == pytest.approx(2.3, rel=1e-4)
    np.testing.assert_allclose(fit['model_flux'], observed, rtol=1e-4)
    assert all(item['params']['C2H2']['A'] == item['params']['13C12CH2']['A']
               for item in fit['top_list'])


@pytest.mark.parametrize('mode', ['equal', 'less'])
@pytest.mark.parametrize('component_order', [
    ('CO2', 'CO17O', 'CO18O'),
    ('CO17O', 'CO18O', 'CO2'),
])
def test_isotopologue_temperature_and_area_constraints(monkeypatch, mode,
                                                        component_order):
    wave = np.linspace(13.0, 16.0, 61)
    profiles = {
        'CO2': np.exp(-0.5 * ((wave - 14.2) / 0.10) ** 2),
        'CO17O': np.exp(-0.5 * ((wave - 14.8) / 0.10) ** 2),
        'CO18O': np.exp(-0.5 * ((wave - 15.4) / 0.10) ** 2),
    }

    def fake_flux(model, T, logN, A, obs_wav):
        profile = np.interp(obs_wav, wave, profiles[model['mol']])
        return A * (1 + (T - 400) / 3000) * profile

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    models = {mol: {'mol': mol} for mol in profiles}
    observed = sum(1.7 * profile for profile in profiles.values())
    fit = fitting.fit_molecules(
        wave, observed, list(component_order), models, (13.0, 16.0),
        T_bounds={'CO2': (300, 700), 'CO17O': (200, 600),
                  'CO18O': (250, 550)},
        logN_bounds=(15.0, 17.0), loga_bounds=(-1.0, 1.0),
        isotopologue_groups=[('CO2', 'CO17O', 'CO18O')],
        Tshare=[mode], Ashare=[True],
        n_samples=8, n_refine=1, n_top=1, seed=2, verbose=False,
    )
    for solution in (fit, *fit['top_list'], *fit['global_top']):
        params = solution['params']
        main = params['CO2']
        for rare_name in ('CO17O', 'CO18O'):
            rare = params[rare_name]
            if mode == 'equal':
                assert rare['T'] == main['T']
            else:
                assert rare['T'] <= main['T']
            assert rare['A'] == main['A']
    assert fit['Tshare'] == [mode]
    assert fit['Ashare'] == [True]


def test_isotopologue_constraints_reject_infeasible_bounds():
    with pytest.raises(ValueError, match='no feasible temperature range'):
        fitting.fit_molecules(
            np.array([13.0, 14.0]), np.array([1.0, 1.0]),
            ['CO2', 'CO17O'], {'CO2': {}, 'CO17O': {}}, (13.0, 14.0),
            T_bounds={'CO2': (100, 200), 'CO17O': (300, 400)},
            isotopologue_groups=[('CO2', 'CO17O')],
            Tshare=['less'], verbose=False,
        )


def test_mixed_group_modes_follow_group_order(monkeypatch):
    wave = np.linspace(13.0, 16.0, 31)
    names = ('C2H2', '13C12CH2', 'CO2', '13CO2')
    shapes = {name: np.exp(-0.5 * ((wave - (13.3 + i * 0.7)) / 0.12) ** 2)
              for i, name in enumerate(names)}

    def fake_flux(model, T, logN, A, obs_wav):
        return A * np.interp(obs_wav, wave, shapes[model['mol']])

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    fit = fitting.fit_molecules(
        wave, sum(shapes.values()), names,
        {name: {'mol': name} for name in names}, (13.0, 16.0),
        isotopologue_groups=[('C2H2', '13C12CH2'), ('CO2', '13CO2')],
        Tshare=['less', 'equal'], Ashare=[False, True],
        T_bounds=(200, 800), logN_bounds=(14, 18),
        n_samples=4, n_refine=1, n_top=1, seed=3, verbose=False,
    )
    params = fit['params']
    assert params['13C12CH2']['T'] <= params['C2H2']['T']
    assert params['13CO2']['T'] == params['CO2']['T']
    assert params['13CO2']['A'] == params['CO2']['A']
    assert fit['Ashare'] == [False, True]


@pytest.mark.parametrize('kwargs, message', [
    ({'Tshare': ['equal']}, 'require isotopologue_groups'),
    ({'isotopologue_groups': [('CO2', 'CO17O')], 'Tshare': ['cold']},
     'free, equal, or less'),
    ({'isotopologue_groups': [('CO2', 'CO17O')], 'Ashare': [1]},
     'one Boolean'),
    ({'isotopologue_groups': [('CO2', 'CO17O'), ('CO17O', 'CO18O')]},
     'cannot overlap'),
])
def test_isotopologue_options_validate_groups(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fitting.fit_molecules(
            np.array([13.0, 14.0]), np.array([1.0, 1.0]),
            ['CO2', 'CO17O', 'CO18O'],
            {'CO2': {}, 'CO17O': {}, 'CO18O': {}}, (13.0, 14.0),
            verbose=False, **kwargs,
        )


@pytest.mark.parametrize('groups, message', [
    ([('C2H2', 'missing')], 'unknown component'),
    ([('C2H2', '13C12CH2'), ('C2H2', '13C12CH2')], 'overlap'),
    ([('C2H2', 'C2H2')], 'distinct labels'),
])
def test_shared_area_groups_reject_invalid_labels(groups, message):
    with pytest.raises(ValueError, match=message):
        fitting.fit_molecules(
            np.array([13.0, 14.0]), np.array([1.0, 1.0]),
            ['C2H2', '13C12CH2'],
            {'C2H2': {}, '13C12CH2': {}}, (13.0, 14.0),
            shared_area_groups=groups, verbose=False,
        )


def test_less_tie_samples_when_main_sits_on_rare_lower_bound(monkeypatch):
    # Data prefer T below both lower bounds, so the main temperature pins at
    # the rare species' lower bound, where the 'less' prior width is zero.
    wave = np.linspace(13.0, 16.0, 61)

    def fake_flux(model, T, logN, A, obs_wav):
        centre = {'CO2': 14.2, '13CO2': 15.2}[model['mol']] + (T - 400) / 2000
        return A * (logN - 14.0) * np.exp(-0.5 * ((obs_wav - centre) / 0.1) ** 2)

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    models = {mol: {'mol': mol} for mol in ('CO2', '13CO2')}
    observed = (fake_flux(models['CO2'], 100, 16, 1, wave)
                + fake_flux(models['13CO2'], 100, 15, 1, wave))
    fit = fitting.fit_molecules(
        wave, observed, ['CO2', '13CO2'], models, (13.0, 16.0),
        T_bounds=(200, 1400), logN_bounds=(14.5, 17.0), loga_bounds=(-1.0, 1.0),
        isotopologue_groups=[('CO2', '13CO2')], Tshare=['less'],
        n_samples=16, n_refine=2, n_top=2, seed=3, verbose=False,
    )
    assert fit['params']['CO2']['T'] == pytest.approx(200, abs=1)
    assert np.isfinite(fit['uncertainty']['CO2']['T']['q50'])


def test_screening_out_main_isotopologue_drops_its_tie():
    stage = {'mol': ['C2H2', '13C12CH2', 'C2HD'],
             'isotopologue_groups': [('C2H2', '13C12CH2', 'C2HD')],
             'Tshare': ['less'], 'Ashare': [True]}
    options = fitting._screened_group_options(stage, {'13C12CH2', 'C2HD'})
    assert options['isotopologue_groups'] is None
    assert options['Tshare'] is None and options['Ashare'] is None
    kept = fitting._screened_group_options(stage, {'C2H2', 'C2HD'})
    assert kept['isotopologue_groups'] == [('C2H2', 'C2HD')]
    assert kept['Tshare'] == ['less']
