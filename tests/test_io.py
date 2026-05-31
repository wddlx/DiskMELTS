from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_load_spectrum_basic(spec_csv):
    pytest.importorskip("torch")
    from diskmelts import load_observed_spectrum

    wav, flux = load_observed_spectrum(str(spec_csv))
    assert len(wav) == 3
    assert len(flux) == 3
    assert wav[0] == pytest.approx(11.0)
    assert wav[1] == pytest.approx(14.5)
    assert flux[1] == pytest.approx(0.25)


def test_load_spectrum_drops_nan_flux(tmp_path):
    pytest.importorskip("torch")
    from diskmelts import load_observed_spectrum

    p = tmp_path / "spec_nan.csv"
    p.write_text("wave,flux\n11.0,0.1\n12.0,\n13.0,0.3\n")
    wav, flux = load_observed_spectrum(str(p))
    assert len(wav) == 2
    assert 12.0 not in wav


def test_load_spectrum_all_valid(tmp_path):
    pytest.importorskip("torch")
    from diskmelts import load_observed_spectrum

    p = tmp_path / "spec_ok.csv"
    lines = ["wave,flux"] + [f"{w:.1f},{w * 0.01:.4f}" for w in range(10, 20)]
    p.write_text("\n".join(lines) + "\n")
    wav, flux = load_observed_spectrum(str(p))
    assert len(wav) == 10
    assert np.all(np.isfinite(flux))


def test_save_running_spectrum(tmp_path):
    pytest.importorskip("torch")
    from diskmelts import save_running_spectrum

    wav = np.linspace(11.0, 19.0, 50)
    residual = np.random.default_rng(0).normal(0, 0.01, 50)
    cumulative = np.ones(50) * 0.5

    path = save_running_spectrum(wav, residual, cumulative, str(tmp_path), "test_src")
    df = pd.read_csv(path)
    assert list(df.columns) == ["wave", "residual", "cumulative_model"]
    assert len(df) == 50
    assert df["wave"].iloc[0] == pytest.approx(11.0)
