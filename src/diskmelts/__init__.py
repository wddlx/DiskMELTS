"""
diskmelts — Machine-learning Enabled Line fitting Tool for Spectra

A neural surrogate-assisted framework for retrieving molecular gas parameters
(T, logN, A) from JWST mid-IR spectra of protoplanetary disks.
"""

from diskmelts.trainmodel import (
    load_model_grid,
    generate_pre_training_set,
    pretrain_forward_model,
    train_model_bank,
    select_grid_holdout,
    tune_pca_components,
)
from diskmelts.fitting import (
    load_models,
    load_model_bank,
    generate_spectrum,
    fit_nested,
    fit_molecules,
    load_observed_spectrum,
    save_fit_outputs,
    detect_stage_molecules,
    save_running_spectrum,
    print_fit_params,
    save_fitted_comparison,
)
from diskmelts.validation import (
    validate_nt,
    validate_nt_holdout,
    validate_full,
)
from diskmelts.plotting import (
    plot_fit,
    plot_validation,
    plot_validation_split,
    plot_bank_reconstruction,
)
from diskmelts.v2_fitting import (
    MODEL_SPECS,
    load_fitting_models,
    fit_stage,
    fit_observation,
)
from diskmelts.v2_grid_fitting import (
    fit_grid_stage,
    fit_observation_grids,
    combine_grid_uncertainty,
)
from diskmelts.v2_read_plot import (
    read_stage_result,
    plot_saved_observation,
)
from diskmelts.v2_paper_comparison import plot_literature_comparisons

__all__ = [
    "MODEL_SPECS",
    "load_fitting_models",
    "fit_stage",
    "fit_observation",
    "fit_grid_stage",
    "fit_observation_grids",
    "combine_grid_uncertainty",
    "read_stage_result",
    "plot_saved_observation",
    "plot_literature_comparisons",
    # trainmodel
    "load_model_grid",
    "generate_pre_training_set",
    "pretrain_forward_model",
    "train_model_bank",
    "select_grid_holdout",
    "tune_pca_components",
    # fitting
    "load_models",
    "load_model_bank",
    "generate_spectrum",
    "fit_nested",
    "fit_molecules",
    "load_observed_spectrum",
    "save_fit_outputs",
    "detect_stage_molecules",
    "save_running_spectrum",
    "print_fit_params",
    "save_fitted_comparison",
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
