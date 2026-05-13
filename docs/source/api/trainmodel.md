# diskmelts.trainmodel

Training utilities for per-molecule forward surrogate models.

Each molecule requires two steps before fitting:

1. **`generate_pre_training_set`** — build a CSV that enumerates every
   $(T, \log N)$ grid point with a peak-normalised spectrum.
2. **`pretrain_forward_model`** — train (or load) the two-MLP forward model
   from that CSV.

The `MLP` class is used internally and is not part of the public API.

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
