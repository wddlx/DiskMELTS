"""Smoke tests for the committed pretrained DiskMELTS fitting workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent


def test_import_diskmelts():
    pytest.importorskip("torch")

    import diskmelts

    assert hasattr(diskmelts, "load_models")
    assert hasattr(diskmelts, "fit_molecules")
    assert hasattr(diskmelts, "load_observed_spectrum")


def test_pretrained_h2o_fitting_smoke():
    torch = pytest.importorskip("torch")
    assert torch is not None

    model_path = ROOT / "Trained_model" / "net_H2O_forward_11to19.pt"
    assert model_path.exists(), "the committed H2O checkpoint is required"

    from diskmelts import fit_molecules, generate_spectrum, load_models

    pretrained = load_models(
        model_paths={"H2O": str(model_path)},
    )

    obs_wav = pretrained["H2O"]["wav"]
    obs_flux = generate_spectrum(
        T=650.0,
        logN=17.0,
        A=1.0,
        pretrained_mol=pretrained["H2O"],
        obs_wav=obs_wav,
    )

    fit = fit_molecules(
        obs_wav,
        obs_flux,
        mol="H2O",
        pretrained=pretrained,
        fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
        n_samples=512,
        n_refine=4,
        n_top=4,
        seed=42,
        verbose=False,
    )

    params = fit["params"]["H2O"]
    residual_rms = float(np.sqrt(np.mean(fit["residual"] ** 2)))
    flux_rms = float(np.sqrt(np.mean(obs_flux ** 2)))

    assert np.isfinite(params["T"])
    assert np.isfinite(params["logN"])
    assert np.isfinite(params["A"])
    assert residual_rms < max(1e-20, 0.05 * flux_rms)
