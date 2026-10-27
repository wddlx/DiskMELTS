from __future__ import annotations

import os

import numpy as np
import pytest

# Short MCMC chains keep the suite fast; accuracy tests set their own length.
os.environ.setdefault("DISKMELTS_N_MCMC", "10000")


@pytest.fixture
def spec_csv(tmp_path):
    """Minimal wave,flux CSV with three valid rows."""
    p = tmp_path / "spectrum.csv"
    p.write_text("wave,flux\n11.0,0.10\n14.5,0.25\n19.0,0.05\n")
    return p


@pytest.fixture
def synthetic_wav_flux():
    """100-point wavelength grid spanning 11–19 µm with zero flux."""
    wav = np.linspace(11.0, 19.0, 100)
    flux = np.zeros_like(wav)
    return wav, flux
