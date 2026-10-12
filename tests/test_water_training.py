from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.mark.parametrize('temperature_transform', ['linear', 'log10'])
def test_window_normalization_preserves_physical_targets_and_metadata(tmp_path, temperature_transform):
    from diskmelts.trainmodel import pretrain_forward_model

    rows = [dict(H2O_T=100 + 100*i, H2O_logN=15., H2O_log10_peak=-2.,
                 wav_5=1., wav_6=.01*(i+1), wav_7=.02*(i+1), wav_8=.03*(i+1))
            for i in range(12)]
    csv = tmp_path/'pretrain.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    path = tmp_path/'model.pt'
    args = dict(mol='H2O', pretrain_csv=csv, wav_range=(6,8), device='cpu',
                model_path=path, n_epochs=1, hidden=(4,), n_pca=2,
                peak_normalization='window', shape_loss='spectral',
                temperature_transform=temperature_transform)
    result = pretrain_forward_model(**args)
    np.testing.assert_allclose(result['Y_pre_v'].max(axis=1), 1.)
    for x, shape, peak in zip(result['X_pre_v'], result['Y_pre_v'], result['log10p_v']):
        index = int((x[0]-100)/100)
        np.testing.assert_allclose(shape*10**peak, np.array([.01,.02,.03])*(index+1)*.01, rtol=2e-6)
    loaded = pretrain_forward_model('H2O', model_path=path, device='cpu')
    assert loaded['peak_normalization'] == 'window'
    assert loaded['shape_loss'] == 'spectral'
    assert loaded['temperature_transform'] == temperature_transform
    from diskmelts import generate_spectrum
    np.testing.assert_allclose(generate_spectrum(600.,15.,1.,loaded),
                               generate_spectrum(600.,15.,1.,result))
    expected_input = result['X_pre_v'].copy()
    if temperature_transform == 'log10':
        expected_input[:, 0] = np.log10(expected_input[:, 0])
    np.testing.assert_allclose(result['X_pre_v_t'].cpu(),
                               result['xp_sc'].transform(expected_input), atol=1e-6)
    with pytest.raises(ValueError, match='peak_normalization'):
        pretrain_forward_model(**{**args, 'peak_normalization':'csv'})


@pytest.mark.parametrize('temperature_transform', ['linear', 'log10'])
def test_nt_validation_is_invariant_to_peak_shape_factorization(temperature_transform):
    from diskmelts.validation import _fit_only_nt
    from diskmelts import generate_spectrum

    log_temperature = temperature_transform == 'log10'
    input_mean = 2. if log_temperature else 0.
    input_scale = 2. if log_temperature else 1000.
    target_shape = (np.log10(600.)-input_mean)/input_scale if log_temperature else .6

    def model(factor):
        shape, peak = torch.nn.Linear(2,2), torch.nn.Linear(2,1)
        with torch.no_grad():
            shape.weight.copy_(torch.tensor([[factor,0.],[0.,0.]]))
            shape.bias.copy_(torch.tensor([0.,factor]))
            peak.weight.copy_(torch.tensor([[0.,1.]]))
            peak.bias.fill_(-np.log10(factor))
        return dict(net_shape=shape,net_peak=peak,pca=None,wav=np.array([6.,7.]),
                    temperature_transform=temperature_transform,
                    xp_sc=SimpleNamespace(mean_=np.array([input_mean,15.]),scale_=np.array([input_scale,1.])),
                    yp_sc_shape=SimpleNamespace(mean_=np.zeros(2),scale_=np.ones(2)),
                    yp_sc_peak=SimpleNamespace(mean_=np.zeros(1),scale_=np.ones(1)))
    np.testing.assert_allclose(generate_spectrum(600.,16.,1.,model(.001)),
                               np.array([target_shape,1.])*10., rtol=2e-6)
    results = [_fit_only_nt(np.array([target_shape,1.]),1.,np.array([6.,7.]),model(f),
                           (100,1400),(13,19),n_starts=10,n_steps=500,seed=4)
               for f in (1.,.001)]
    np.testing.assert_allclose(results, [[600.,16.],[600.,16.]], atol=.1)


def test_faint_float32_spectrum_retrieval(monkeypatch):
    from diskmelts import fitting

    def flux(model, T, logN, A, wav):
        return A*1e-12*np.array([1.,T/1000.,(logN-12.)/8.],dtype=np.float32).astype(float)

    monkeypatch.setattr(fitting,'_mol_flux_on_wav',flux)
    wav=np.array([1.,2.,3.])
    result=fitting.fit_nested(wav,flux({},725.,16.2,1.3,wav),'H2O',{'H2O':{}},
                              (1.,3.),T_bounds=(100,1400),logN_bounds=(13,19),
                              loga_bounds=(-2,2),n_restarts=3,seed=2)
    assert result['params']['T'] == pytest.approx(725.,abs=.1)
    assert result['params']['logN'] == pytest.approx(16.2,abs=.001)
    assert result['params']['A'] == pytest.approx(1.3,rel=1e-4)
    joint = fitting.fit_molecules(wav, flux({},725.,16.2,1.3,wav), 'H2O', {'H2O':{}},
        (1.,3.), T_bounds=(100,1400), logN_bounds=(13,19), loga_bounds=(-2,2),
        n_samples=32, n_refine=3, seed=2)
    np.testing.assert_allclose(joint['model_flux'], flux({},725.,16.2,1.3,wav), rtol=1e-4)


def test_empty_window_does_not_supply_a_false_peak_target(tmp_path):
    from diskmelts.trainmodel import pretrain_forward_model

    rows = [dict(H2O_T=100+100*i, H2O_logN=15., H2O_log10_peak=-2.,
                 wav_5=1., wav_6=0. if i<3 else .1, wav_7=0. if i<3 else .3)
            for i in range(12)]
    csv = tmp_path/'pretrain.csv'
    pd.DataFrame(rows).to_csv(csv, index=False)
    path = tmp_path/'model.pt'
    result = pretrain_forward_model('H2O', csv, (6,7), device='cpu',
        model_path=path, n_epochs=1, hidden=(4,), n_pca=1,
        peak_normalization='window', shape_loss='spectral')
    assert result['yp_sc_peak'].mean_[0] == pytest.approx(np.log10(.003), abs=1e-6)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['peak_loss_excludes_dark']
    assert checkpoint['n_dark_training'] + checkpoint['n_dark_validation'] == 3
