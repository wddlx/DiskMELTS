# diskmelts.validation

Validation functions for pretrained forward models.

Validation requires the local slab-model grid. These inputs live in the Git-ignored `Pretrain_grid/` and `Model_grids/`
directories; they are not required for ordinary observed-spectrum fitting.

Two validation functions are provided:

- **`validate_nt_holdout`** — checks $(T, \log N)$ recovery on grid points
  that were withheld from training (out-of-distribution test).
- **`validate_full`** — checks $(T, \log N, A)$ recovery on synthetic
  spectra with randomly drawn $A$, testing the complete fitting pipeline.

```{eval-rst}
.. autofunction:: diskmelts.validation.validate_nt_holdout
```

---

```{eval-rst}
.. autofunction:: diskmelts.validation.validate_full
```

`validate_full` runs `fit_molecules` with `errorbars=False` for each synthetic
spectrum. Plot either result with `plot_validation`.

`validate_nt`, which tested spectra the model was trained on, is deprecated
since 2.3.0; use `validate_nt_holdout`. See {doc}`deprecated`.
