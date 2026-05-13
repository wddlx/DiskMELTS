# diskmelts.validation

Validation functions for pretrained forward models.

Two complementary strategies are provided:

- **`validate_nt`** — checks $(T, \log N)$ recovery on spectra the model
  was trained on (in-distribution test).
- **`validate_nt_holdout`** — checks $(T, \log N)$ recovery on grid points
  that were withheld from training (out-of-distribution test).
- **`validate_full`** — checks $(T, \log N, A)$ recovery on synthetic
  spectra with randomly drawn $A$, testing the complete fitting pipeline.

```{eval-rst}
.. autofunction:: diskmelts.validation.validate_nt
```

---

```{eval-rst}
.. autofunction:: diskmelts.validation.validate_nt_holdout
```

---

```{eval-rst}
.. autofunction:: diskmelts.validation.validate_full
```
