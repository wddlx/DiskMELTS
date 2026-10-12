# diskmelts.fitting

Spectral fitting with pretrained forward surrogate models.

For observed spectra, use the v2 package workflow in {doc}`v2`, starting with
`load_fitting_models` and `fit_stage` or `fit_observation`. The lower-level
`fit_molecules` runs a global Sobol search followed by local refinement.
`fit_nested` is a simpler single-molecule fitter with random restarts.

**Forward model convention:**

```
flux(T, logN, A) = A × peak(T, logN) × shape(T, logN)
```

## Loading models

Current checkpoints are self-contained. The recommended v2 loader uses the
bundled paths and checks model metadata. The lower-level `load_models` accepts
explicit `model_paths`; its CSV, wavelength-range, and PCA arguments are
compatibility options for legacy checkpoints.

`model_paths` may also point to a segmented model-bank JSON manifest. DiskMELTS
then blends overlapping tiles while fitting one shared physical parameter set.

```python
from diskmelts import load_fitting_models
pretrained = load_fitting_models(['H2O'])
```

```{eval-rst}
.. autofunction:: diskmelts.fitting.load_models
```

```{eval-rst}
.. autofunction:: diskmelts.fitting.load_model_bank
```

---

## Generating spectra

```{eval-rst}
.. autofunction:: diskmelts.fitting.generate_spectrum
```

---

## Fitting

```{eval-rst}
.. autofunction:: diskmelts.fitting.fit_molecules
```

---

```{eval-rst}
.. autofunction:: diskmelts.fitting.fit_nested
```

---

## I/O helpers

```{eval-rst}
.. autofunction:: diskmelts.fitting.load_observed_spectrum
```

---

```{eval-rst}
.. autofunction:: diskmelts.fitting.detect_stage_molecules
```

---

```{eval-rst}
.. autofunction:: diskmelts.fitting.save_fit_outputs
```

---

```{eval-rst}
.. autofunction:: diskmelts.fitting.save_running_spectrum
```

---

```{eval-rst}
.. autofunction:: diskmelts.fitting.print_fit_params
```

---

```{eval-rst}
.. autofunction:: diskmelts.fitting.save_fitted_comparison
```
