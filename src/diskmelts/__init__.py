"""
diskmelts — Machine-learning Enabled Line fitting Tool for Spectra

A neural surrogate-assisted framework for retrieving molecular gas parameters
(T, logN, A) from JWST mid-IR spectra of protoplanetary disks.
"""

from diskmelts.trainmodel import (
    load_model_grid,
    generate_pre_training_set,
    pretrain_forward_model,
)
from diskmelts.fitting import (
    load_models,
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
)

__all__ = [
    # trainmodel
    "load_model_grid",
    "generate_pre_training_set",
    "pretrain_forward_model",
    # fitting
    "load_models",
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
]
