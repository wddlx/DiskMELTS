# diskmelts.fitting

Spectral fitting with pretrained forward surrogate models.

The main entry point for most users is `fit_molecules`, which runs a
global Sobol search followed by local L-BFGS-B refinement.
`fit_nested` is a simpler single-molecule fitter with random restarts.

**Forward model convention:**

```
flux(T, logN, A) = A × peak(T, logN) × shape(T, logN)
```

## Loading models

Current checkpoints are self-contained, so normal fitting only passes
`model_paths`. The CSV, wavelength-range, and PCA arguments are compatibility
options for legacy checkpoints.

```python
pretrained = load_models(
    model_paths={
        'H2O': 'Trained_model/net_H2O_forward_11to19.pt',
    },
)
```

```{eval-rst}
.. autofunction:: diskmelts.fitting.load_models
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
