from __future__ import annotations

import pytest


def test_mlp_forward_shape():
    torch = pytest.importorskip("torch")
    from diskmelts.trainmodel import MLP

    model = MLP(n_in=2, n_out=10)
    x = torch.randn(4, 2)
    y = model(x)
    assert y.shape == (4, 10)


def test_mlp_custom_hidden():
    torch = pytest.importorskip("torch")
    from diskmelts.trainmodel import MLP

    model = MLP(n_in=3, n_out=5, hidden=(32, 64))
    x = torch.randn(2, 3)
    y = model(x)
    assert y.shape == (2, 5)


def test_mlp_output_finite():
    torch = pytest.importorskip("torch")
    from diskmelts.trainmodel import MLP

    model = MLP(n_in=2, n_out=8)
    x = torch.randn(16, 2)
    y = model(x)
    assert torch.isfinite(y).all(), "MLP output contains NaN or inf"


def test_mlp_single_sample():
    torch = pytest.importorskip("torch")
    from diskmelts.trainmodel import MLP

    model = MLP(n_in=2, n_out=3)
    x = torch.tensor([[600.0, 17.0]])
    y = model(x)
    assert y.shape == (1, 3)
