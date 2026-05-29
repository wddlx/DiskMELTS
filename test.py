"""Pytest smoke tests for the pretrained DiskMELTS fitting workflow.

Run from the repository root:

    pytest test.py

The pretrained fitting test is skipped unless both the H2O checkpoint and its
matching pretrain CSV are present. This keeps the test useful for source
checkouts while still exercising the main user workflow in release bundles.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


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
    pretrain_csv = ROOT / "Pretrain_grid" / "pretrain_H2O_11to19.csv"
    if not model_path.exists() or not pretrain_csv.exists():
        pytest.skip(
            "pretrained H2O checkpoint and matching Pretrain_grid CSV are required"
        )

    from diskmelts import fit_molecules, generate_spectrum, load_models

    pretrained = load_models(
        model_paths={"H2O": str(model_path)},
        pretrain_csv_paths={"H2O": str(pretrain_csv)},
        wav_ranges={"H2O": (11.0, 19.0)},
        n_pca={"H2O": 21},
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
