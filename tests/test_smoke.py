"""Smoke tests for the bundled production fitting workflow."""

from __future__ import annotations

import numpy as np
import pytest


def test_import_diskmelts():
    pytest.importorskip("torch")

    import diskmelts

    assert hasattr(diskmelts, "load_fitting_models")
    assert hasattr(diskmelts, "fit_stage")
    assert hasattr(diskmelts, "load_observed_spectrum")


def test_pretrained_h2o_fitting_smoke():
    torch = pytest.importorskip("torch")
    assert torch is not None

    from diskmelts import fit_stage, generate_spectrum, load_fitting_models

    pretrained = load_fitting_models("H2O")
    obs_wav = np.linspace(11.0, 18.6, 350)
    obs_flux = generate_spectrum(
        T=650.0,
        logN=17.0,
        A=1.0,
        pretrained_mol=pretrained["H2O"],
        obs_wav=obs_wav,
    )

    fit = fit_stage(
        obs_wav, obs_flux, "H2O", (11.0, 18.6),
        [(11.0, 12.0), (16.5, 18.5)], pretrained,
        n_samples=256, n_refine=4, n_top=4, seed=42, verbose=False,
    )

    params = fit["params"]["H2O"]
    residual_rms = float(np.sqrt(np.mean(fit["residual"] ** 2)))
    flux_rms = float(np.sqrt(np.mean(obs_flux ** 2)))

    assert np.isfinite(params["T"])
    assert np.isfinite(params["logN"])
    assert np.isfinite(params["A"])
    assert residual_rms < max(1e-20, 0.05 * flux_rms)
