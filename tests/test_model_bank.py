from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
