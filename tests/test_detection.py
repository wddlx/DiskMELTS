from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def wav_flux_13to15():
    wav = np.linspace(13.0, 15.5, 2000)
    flux = np.zeros_like(wav)
    return wav, flux


def _inject_peak(flux, wav, lo, hi, amplitude=1.0):
    mask = (wav >= lo) & (wav <= hi)
    flux = flux.copy()
    flux[mask] = amplitude
    return flux


# ---------------------------------------------------------------------------
# dict-style detect_peaks (per-molecule checks)
# ---------------------------------------------------------------------------

def test_detect_dict_single_detected(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    flux = _inject_peak(flux, wav, 13.705, 13.715, amplitude=1.0)
    stage = {"mol": "C2H2", "detect_peaks": {"C2H2": (13.705, 13.715)}}
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert result == ["C2H2"]


def test_detect_dict_not_detected(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    stage = {"mol": "C2H2", "detect_peaks": {"C2H2": (13.705, 13.715)}}
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert result == []


def test_detect_dict_partial_multi(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    flux = _inject_peak(flux, wav, 13.705, 13.715, amplitude=1.0)
    # HCN peak (13.99–14.05) left at zero → not detected
    stage = {
        "mol": ["C2H2", "HCN"],
        "detect_peaks": {"C2H2": (13.705, 13.715), "HCN": (13.99, 14.05)},
    }
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert result == ["C2H2"]
    assert "HCN" not in result


def test_detect_dict_all_detected(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    flux = _inject_peak(flux, wav, 13.705, 13.715, amplitude=1.0)
    flux = _inject_peak(flux, wav, 13.99, 14.05, amplitude=1.0)
    stage = {
        "mol": ["C2H2", "HCN"],
        "detect_peaks": {"C2H2": (13.705, 13.715), "HCN": (13.99, 14.05)},
    }
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert set(result) == {"C2H2", "HCN"}


# ---------------------------------------------------------------------------
# list-style detect_peaks (whole-stage check)
# ---------------------------------------------------------------------------

def test_detect_list_detected(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    flux = _inject_peak(flux, wav, 13.705, 13.715, amplitude=1.0)
    stage = {"mol": ["C2H2", "HCN"], "detect_peaks": [(13.705, 13.715)]}
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert set(result) == {"C2H2", "HCN"}


def test_detect_list_not_detected(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    stage = {"mol": ["C2H2", "HCN"], "detect_peaks": [(13.705, 13.715)]}
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert result == []


# ---------------------------------------------------------------------------
# Bypass conditions
# ---------------------------------------------------------------------------

def test_detect_screening_disabled(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    stage = {"mol": "C2H2", "detect_peaks": {"C2H2": (13.705, 13.715)}}
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1, screening=False)
    assert "C2H2" in result


def test_detect_sigma_none_bypasses(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    stage = {"mol": "C2H2", "detect_peaks": {"C2H2": (13.705, 13.715)}}
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=None)
    assert "C2H2" in result


def test_detect_no_peaks_key_bypasses(wav_flux_13to15):
    pytest.importorskip("torch")
    from diskmelts import detect_stage_molecules

    wav, flux = wav_flux_13to15
    stage = {"mol": "C2H2"}  # no detect_peaks key
    result = detect_stage_molecules(wav, flux, stage, sigma_noise=0.1)
    assert "C2H2" in result
