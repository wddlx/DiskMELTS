# diskmelts.trainmodel

Training utilities for per-molecule forward surrogate models.

Retraining a molecule from the full local model grids uses two steps:

1. **`generate_pre_training_set`** — build a CSV that enumerates every
   $(T, \log N)$ grid point with a peak-normalised spectrum.
2. **`pretrain_forward_model`** — train (or load) the two-MLP forward model
   from that CSV.

The `MLP` class is used internally and is not part of the public API.

Users of the bundled self-contained checkpoints can skip these steps and load
the `.pt` files directly. See {doc}`../quickstart` for fitting and
{doc}`../training` for the complete local-data workflow.

---

```{eval-rst}
.. autofunction:: diskmelts.trainmodel.load_model_grid
```

---

```{eval-rst}
.. autofunction:: diskmelts.trainmodel.generate_pre_training_set
```

---

```{eval-rst}
.. autofunction:: diskmelts.trainmodel.pretrain_forward_model
```
