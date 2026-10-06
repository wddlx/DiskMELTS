"""Observation wrappers report physical area at the supplied distance."""

import numpy as np
import pytest


pytest.importorskip('torch')
from diskmelts import v2_fitting as dev_v2_fitting


def test_model_stage_scales_flux_and_noise_to_reference_distance(monkeypatch):
    observed = np.array([0.25, 0.5])
    wav = np.array([12.0, 12.1])

    def fake_fit_molecules(**kwargs):
        np.testing.assert_allclose(kwargs['obs_flux'], [1.0, 2.0])
        assert kwargs['sigma'] == pytest.approx(0.4)
        return {
            'model_flux': np.array([1.0, 2.0]),
            'residual': np.zeros(2),
            'component_fluxes': {'C2H2': np.array([1.0, 2.0])},
            'top_list': [{'mse_loss': 4.0}],
            'params': {'C2H2': {'A': 1.5}},
        }

    monkeypatch.setattr(dev_v2_fitting, 'fit_molecules', fake_fit_molecules)
    result = dev_v2_fitting.fit_stage(
        wav, observed, 'C2H2', (12.0, 12.1), (12.0, 12.1),
        pretrained={'C2H2': object()}, distance_pc=280.0, sigma=0.1,
    )
    np.testing.assert_allclose(result['model_flux'], observed)
    np.testing.assert_allclose(result['component_fluxes']['C2H2'], observed)
    assert result['params']['C2H2']['A'] == 1.5
    assert result['top_list'][0]['mse_loss'] == pytest.approx(0.25)
    assert result['distance_pc'] == 280.0
