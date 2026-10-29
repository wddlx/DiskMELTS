# Deprecated functions

These functions still work in 2.3.0 but raise a `FutureWarning` and will be
removed in a future release. They are no longer part of `diskmelts.__all__`.

| Deprecated | Use instead | Notes |
| --- | --- | --- |
| `fit_nested` | `fit_molecules(..., errorbars=False)` | `fit_nested` now calls `fit_molecules` and returns the old flat `params` (`T`, `logN`, `A`); `n_restarts` sets the number of local refinements. |
| `validate_nt` | `validate_nt_holdout` | `validate_nt` tested spectra seen in training, which overstates accuracy. |
| `plot_validation_split` | `plot_validation` | It paired `validate_nt` with held-out results. |

```{eval-rst}
.. autofunction:: diskmelts.fitting.fit_nested

.. autofunction:: diskmelts.validation.validate_nt

.. autofunction:: diskmelts.plotting.plot_validation_split
```
