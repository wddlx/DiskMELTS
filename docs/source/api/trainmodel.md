# diskmelts.trainmodel

Training utilities for per-molecule forward surrogate models.

Retraining a molecule from the full local model grids uses these steps:

1. **`generate_pre_training_set`** — build a CSV that enumerates every
   $(T, \log N)$ grid point with a peak-normalised spectrum.
2. **`pretrain_forward_model`** — train (or load) the two-MLP forward model
   from that CSV.
3. **`select_grid_holdout`** — exclude complete grid spectra for honest unseen
   validation.
4. **`tune_pca_components`** — choose the PCA dimension from holdout spectral
   reconstruction.

`train_model_bank` trains the maintained wavelength-only H2O bank and remains
available for reproducing archived wavelength/T/logN segmented experiments.
It accepts either one PCA count or one independently tuned count per wavelength.
Both training functions accept `peak_normalization`, `shape_loss`,
`temperature_transform`, `scheduler_patience`, and `min_lr`. The per-model
trainer also supports `shape_loss='spectral_edge'` with `shape_edge_ranges` and
`shape_edge_weight`; it emphasizes selected intervals while calculating the
PCA-space loss exactly as weighted wavelength-space error. The legacy defaults
preserve CSV normalization and linear-temperature inputs. The water examples
select local-window normalization and log-temperature inputs; the checkpoint
carries the preprocessing needed for inference.

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

---

```{eval-rst}
.. autofunction:: diskmelts.trainmodel.train_model_bank
```

---

```{eval-rst}
.. autofunction:: diskmelts.trainmodel.select_grid_holdout
```

---

```{eval-rst}
.. autofunction:: diskmelts.trainmodel.tune_pca_components
```
