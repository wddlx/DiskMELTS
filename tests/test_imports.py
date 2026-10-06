from __future__ import annotations

import pytest


PUBLIC_SYMBOLS = [
    # fitting
    "load_models",
    "load_model_bank",
    "generate_spectrum",
    "fit_nested",
    "fit_molecules",
    "load_observed_spectrum",
    "detect_stage_molecules",
    "save_fit_outputs",
    "save_running_spectrum",
    "print_fit_params",
    "save_fitted_comparison",
    # trainmodel
    "load_model_grid",
    "generate_pre_training_set",
    "pretrain_forward_model",
    "train_model_bank",
    "select_grid_holdout",
    "tune_pca_components",
    # validation
    "validate_nt",
    "validate_nt_holdout",
    "validate_full",
    # plotting
    "plot_fit",
    "plot_validation",
    "plot_validation_split",
    "plot_bank_reconstruction",
]


def test_diskmelts_importable():
    pytest.importorskip("torch")
    import diskmelts  # noqa: F401


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS)
def test_public_symbol(name):
    pytest.importorskip("torch")
    import diskmelts

    assert hasattr(diskmelts, name), f"'{name}' missing from diskmelts public API"
