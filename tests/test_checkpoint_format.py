"""Bundled checkpoints store preprocessing as arrays and load without scikit-learn."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip('torch')
from diskmelts.trainmodel import _load_checkpoint, convert_legacy_checkpoint


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = sorted((ROOT / 'src/diskmelts/models/Trained_model').glob('sigma*/*/*.pt'))


def test_all_bundled_checkpoints_are_array_only():
    assert len(CHECKPOINTS) == 46
    for path in CHECKPOINTS:
        # weights_only=True rejects any pickled scikit-learn or NumPy object.
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        for key in ('xp_sc', 'yp_sc_shape', 'yp_sc_peak'):
            assert set(ckpt[key]) == {'mean', 'scale'}, (path, key)
        assert ckpt['pca'] is None or {'components', 'mean'} <= set(ckpt['pca'])


def test_loading_models_does_not_import_sklearn(tmp_path):
    code = ("import sys; from diskmelts import synthesize_spectrum; "
            "synthesize_spectrum('H2O', 500, 18); synthesize_spectrum('HCN', 500, 17); "
            "assert 'sklearn' not in sys.modules, 'sklearn was imported'")
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'),
               MPLCONFIGDIR=str(tmp_path / 'mpl'))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_legacy_checkpoint_converts_to_identical_preprocessing(tmp_path):
    sklearn = pytest.importorskip('sklearn')
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2)) * [300, 1.5] + [700, 16]
    Y = rng.random((40, 25)).astype(np.float32)
    pca = PCA(n_components=5, random_state=0).fit(Y)
    Z = pca.transform(Y)
    scalers = {'xp_sc': StandardScaler().fit(X), 'yp_sc_shape': StandardScaler().fit(Z),
               'yp_sc_peak': StandardScaler().fit(X[:, :1])}
    legacy = tmp_path / 'legacy.pt'
    torch.save({**scalers, 'pca': pca, 'wav': np.linspace(13, 14, 25)}, legacy)

    converted = Path(convert_legacy_checkpoint(legacy, tmp_path / 'array.pt'))
    torch.load(converted, weights_only=True)  # no pickled objects remain
    ckpt = _load_checkpoint(converted)
    for key, scaler in scalers.items():
        data = X if key == 'xp_sc' else (Z if key == 'yp_sc_shape' else X[:, :1])
        # scikit-learn keeps float32 input in float32; the arrays compute in float64.
        np.testing.assert_allclose(ckpt[key].transform(data), scaler.transform(data),
                                   rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(ckpt[key].mean_, scaler.mean_)
        np.testing.assert_array_equal(ckpt[key].scale_, scaler.scale_)
    np.testing.assert_array_equal(ckpt['pca'].inverse_transform(Z), pca.inverse_transform(Z))
    np.testing.assert_allclose(ckpt['pca'].transform(Y), Z, rtol=1e-5, atol=1e-6)
    assert ckpt['pca'].n_components_ == 5
    np.testing.assert_array_equal(ckpt['wav'], np.linspace(13, 14, 25))
    # Legacy files still load directly through the fallback path.
    assert _load_checkpoint(legacy)['pca'] is not None
