from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_load_model_grid_filters_before_returning(tmp_path):
    from diskmelts import load_model_grid

    for T in (100, 1500):
        for logN in (13.0, 20.0):
            pd.DataFrame({
                'wave': [5.0, 10.0, 15.0],
                'Line': [1.0, 2.0, 3.0],
            }).to_csv(tmp_path / f'T{T}N{logN}.csv', index=False)

    models = load_model_grid(
        tmp_path,
        T_range=(100, 1400),
        logN_range=(13.0, 19.5),
        wav_range=(8.0, 12.0),
    )

    assert list(models) == [(100, 13.0)]
    np.testing.assert_allclose(models[(100, 13.0)]['wavelength'], [10.0])


def test_select_grid_holdout_is_reproducible_and_preserves_boundaries():
    from diskmelts import select_grid_holdout

    models = {
        (T, logN): {} for T in (100, 200, 300, 400, 500)
        for logN in (13.0, 14.0, 15.0, 16.0, 17.0)
    }
    first = select_grid_holdout(models, fraction=0.1, seed=7)
    second = select_grid_holdout(models, fraction=0.1, seed=7)

    assert first == second
    assert len(first) == 2
    assert all(T not in (100, 500) for T, _ in first)
    assert all(logN not in (13.0, 17.0) for _, logN in first)


def test_tune_pca_components_uses_unseen_spectra():
    from diskmelts import tune_pca_components

    wav = np.linspace(5.0, 8.0, 12)
    models = {}
    for index, T in enumerate((100, 200, 300, 400, 500, 600)):
        flux = 1.0 + 0.1 * index + np.sin(wav) + 0.02 * index * np.cos(2 * wav)
        models[(T, 15.0)] = {'wavelength': wav, 'flux': flux}

    result = tune_pca_components(
        models,
        holdout_keys=[(300, 15.0), (500, 15.0)],
        wav_range=(5.0, 8.0),
        candidates=(1, 2, 3),
        seed=2,
    )

    assert result['selected'] in (1, 2, 3)
    assert result['n_holdout'] == 2
    assert set(result['rmse']) == {1, 2, 3}
    assert result['rmse'][3] <= result['rmse'][1] + 1e-6


def _bank():
    return {
        'is_model_bank': True,
        'mol': 'H2O',
        'wav_ranges': [(4.9, 7.4), (7.2, 11.2)],
        'T_ranges': [(50.0, 700.0), (600.0, 1300.0)],
        'logN_ranges': [(13.0, 19.0)],
        'tiles': [
            {
                'model': {'tile_value': 1.0},
                'wav_range': (4.9, 7.4),
                'T_range': (50.0, 700.0),
                'logN_range': (13.0, 19.0),
            },
            {
                'model': {'tile_value': 3.0},
                'wav_range': (7.2, 11.2),
                'T_range': (600.0, 1300.0),
                'logN_range': (13.0, 19.0),
            },
        ],
    }


def test_model_bank_routes_and_blends_overlaps(monkeypatch):
    pytest.importorskip('torch')
    from diskmelts import fitting

    def fake_flux(model, T, logN, A, wav_grid):
        return np.full(len(wav_grid), model['tile_value'] * A)

    monkeypatch.setattr(fitting, '_single_model_flux_on_wav', fake_flux)
    wav = np.array([6.0, 7.3, 8.0])
    flux = fitting._mol_flux_on_wav(_bank(), 650.0, 17.0, 2.0, wav)

    assert flux[0] == pytest.approx(2.0)
    assert 2.0 < flux[1] < 6.0
    assert flux[2] == pytest.approx(6.0)


def test_model_bank_uses_manifest_parameter_coverage():
    pytest.importorskip('torch')
    from diskmelts.fitting import _model_bank_bounds

    bank = _bank()
    assert _model_bank_bounds(bank, 'T_ranges', (200, 1400)) == (50.0, 1300.0)
    assert _model_bank_bounds(bank, 'logN_ranges', (14, 19)) == (13.0, 19.0)


def test_train_model_bank_writes_cartesian_manifest(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from diskmelts import trainmodel

    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {'X_pre_v': np.array([[100.0, 14.0]])}

    monkeypatch.setattr(trainmodel, 'pretrain_forward_model', fake_train)
    manifest = trainmodel.train_model_bank(
        mol='H2O',
        pretrain_csv=tmp_path / 'full.csv',
        model_dir=tmp_path / 'models',
        wavelength_ranges=[(4.9, 7.4), (7.2, 11.2)],
        logN_ranges=[(13, 19), (18.5, 22)],
        T_ranges=[(50, 700), (600, 1300)],
        n_epochs=1,
    )

    assert len(calls) == 8
    assert len(manifest['tiles']) == 8
    assert (tmp_path / 'models' / 'H2O_model_bank.json').is_file()
    assert {call['T_range'] for call in calls} == {(50.0, 700.0), (600.0, 1300.0)}


def test_train_model_bank_accepts_pca_per_wavelength(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from diskmelts import trainmodel

    calls = []

    def fake_train(**kwargs):
        calls.append(kwargs)
        return {'X_pre_v': np.array([[100.0, 14.0]])}

    monkeypatch.setattr(trainmodel, 'pretrain_forward_model', fake_train)
    manifest = trainmodel.train_model_bank(
        mol='H2O',
        pretrain_csv=tmp_path / 'full.csv',
        model_dir=tmp_path / 'models',
        wavelength_ranges=[(4.9, 7.4), (7.2, 11.2)],
        logN_ranges=[(13.0, 19.0)],
        T_ranges=[(100.0, 1400.0)],
        n_pca=[15, 21],
        n_epochs=1,
    )

    assert [call['n_pca'] for call in calls] == [15, 21]
    assert manifest['n_pca'] == [15, 21]
    assert [tile['n_pca'] for tile in manifest['tiles']] == [15, 21]


def test_small_model_bank_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip('torch')
    from diskmelts import generate_spectrum, load_model_bank, train_model_bank

    rows = []
    for T in (100.0, 300.0, 500.0, 700.0):
        for logN in (14.0, 18.0):
            rows.append({
                'H2O_T': T,
                'H2O_logN': logN,
                'H2O_A': 1.0,
                'H2O_log10_peak': -2.0 + 0.001 * T + 0.02 * logN,
                'wav_5.000000': 0.2 + T / 10000.0,
                'wav_6.000000': 0.4 + logN / 100.0,
                'wav_7.000000': 0.6 + T / 20000.0,
                'wav_8.000000': 0.8 + logN / 200.0,
            })
    csv_path = tmp_path / 'full.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    trained = train_model_bank(
        mol='H2O',
        pretrain_csv=csv_path,
        model_dir=tmp_path / 'models',
        wavelength_ranges=[(5.0, 7.0), (6.0, 8.0)],
        logN_ranges=[(13.0, 19.0)],
        T_ranges=[(50.0, 800.0)],
        device=torch.device('cpu'),
        hidden=(4,),
        n_epochs=2,
        batch_size=4,
        n_pca=2,
        early_stopping_patience=0,
        seed=3,
    )
    bank = load_model_bank(trained['manifest_path'], device=torch.device('cpu'))
    wav = np.array([5.5, 6.5, 7.5])
    flux = generate_spectrum(400.0, 16.0, 1.0, bank, obs_wav=wav)

    assert len(bank['tiles']) == 2
    assert flux.shape == wav.shape
    assert np.all(np.isfinite(flux))


def test_plot_bank_reconstruction(tmp_path):
    pytest.importorskip('matplotlib')
    from diskmelts import plot_bank_reconstruction

    wav = np.linspace(11.5, 17.0, 20)
    true = np.vstack([np.sin(wav), np.cos(wav)])
    predicted = true + 0.01
    output = tmp_path / 'bank_difference.png'
    figure = plot_bank_reconstruction(
        wav, true, predicted, mol='HCN', save_path=output
    )

    assert output.is_file()
    assert figure is not None
