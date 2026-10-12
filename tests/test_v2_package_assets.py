"""Production v2 inference must work without local slab grids."""

import numpy as np
import pytest


pytest.importorskip('torch')
from diskmelts import (fit_stage, generate_spectrum, load_fitting_models,
                       fit_grid_stage)


def test_bundled_model_fit_without_grids(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / 'Model_grids').exists()
    models = load_fitting_models('C2H2')
    wave = np.linspace(13.5, 14.1, 65)
    flux = generate_spectrum(500, 16, 1.0, models['C2H2'], obs_wav=wave)
    fit = fit_stage(
        wave, flux, 'C2H2', (13.5, 14.1), (13.5, 14.1), models,
        n_samples=64, n_refine=2, n_top=2, seed=4,
    )
    assert fit['params']['C2H2']['T'] == pytest.approx(500, abs=2)
    assert fit['params']['C2H2']['logN'] == pytest.approx(16, abs=0.02)
    with pytest.raises(ValueError, match='grid_root is required'):
        fit_grid_stage(wave, flux, 'C2H2', (13.5, 14.1), (13.5, 14.1))
