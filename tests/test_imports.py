from __future__ import annotations

import pytest


PUBLIC_SYMBOLS = [
    # bundled v2 workflows
    "consub",
    "MODEL_SPECS",
    "SIGMA2_MODEL_SPECS",
    "load_fitting_models",
    "fit_stage",
    "fit_observation",
    "fit_grid_stage",
    "fit_observation_grids",
    "combine_grid_uncertainty",
    "read_stage_result",
    "plot_saved_observation",
    "plot_literature_comparisons",
    # fitting
    "load_models",
    "load_model_bank",
    "generate_spectrum",
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
    "convert_legacy_checkpoint",
    "grid_sha256",
    # validation
    "validate_nt_holdout",
    "validate_full",
    # plotting
    "plot_fit",
    "plot_validation",
    "plot_bank_reconstruction",
    "synthesize_spectrum",
]
DEPRECATED_SYMBOLS = ["fit_nested", "validate_nt", "plot_validation_split"]


def test_diskmelts_importable():
    pytest.importorskip("torch")
    import diskmelts  # noqa: F401


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS)
def test_public_symbol(name):
    pytest.importorskip("torch")
    import diskmelts

    assert hasattr(diskmelts, name), f"'{name}' missing from diskmelts public API"


def test_all_matches_public_symbols():
    pytest.importorskip("torch")
    import diskmelts

    assert set(diskmelts.__all__) == set(PUBLIC_SYMBOLS)
    assert not set(DEPRECATED_SYMBOLS) & set(diskmelts.__all__)


@pytest.mark.parametrize("name", DEPRECATED_SYMBOLS)
def test_deprecated_symbol_still_importable(name):
    pytest.importorskip("torch")
    import diskmelts

    assert callable(getattr(diskmelts, name))


def test_fit_nested_warns_and_matches_fit_molecules():
    pytest.importorskip("torch")
    import numpy as np
    from diskmelts import fit_molecules, fit_nested, load_fitting_models, synthesize_spectrum

    wave, flux = synthesize_spectrum("HCN", 450, 16.5, A=0.5,
                                     wavelength=np.linspace(13.5, 14.5, 400))
    models = load_fitting_models("HCN")
    with pytest.warns(FutureWarning, match="fit_nested is deprecated"):
        old = fit_nested(wave, flux, "HCN", models, (13.5, 14.5), n_restarts=4, n_top=2, seed=0)
    new = fit_molecules(wave, flux, "HCN", models, (13.5, 14.5), n_refine=4, n_top=2,
                        seed=0, verbose=False, errorbars=False)
    assert old["params"] == {key: new["params"]["HCN"][key] for key in ("T", "logN", "A")}
    assert old["params"]["T"] == pytest.approx(450, abs=5)
    # top_list keeps distinct solutions only, so it can be shorter than n_top.
    assert [row["rank"] for row in old["top_list"]][:1] == [1]
