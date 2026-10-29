# diskmelts.fitting

Spectral fitting with pretrained forward surrogate models.

For observed spectra, use the v2 package workflow in {doc}`v2`, starting with
`load_fitting_models` and `fit_stage` or `fit_observation`. Both call the
lower-level `fit_molecules`, which fits one or several molecules with a
global Sobol search, local refinement, and MCMC error bars. Use it directly
only when you need its full option set without distance scaling or the
bundled-model bookkeeping.

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
`load_models` calls `load_model_bank` for such manifests; call
`load_model_bank` directly only to load a subset of tiles with its
`wav_ranges`, `T_range`, or `logN_range` filters.

For a joint fit of main and rare isotopologues, `fit_molecules` accepts
`isotopologue_groups=[('C2H2', '13C12CH2')]`, `Tshare=['equal']`, and
`Ashare=[True]`. The first label is the main species. Each group can instead
use `Tshare='less'` (every rare temperature at most the main temperature) or
`'free'`; column densities stay independent. Constraints are enforced during
both global sampling and local refinement. The earlier `shared_area_groups`
option remains available for area ties alone. See the
{doc}`quick start <../quickstart>` for a complete example and bounds.

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

## I/O helpers

`load_observed_spectrum` reads a continuum-subtracted text file by column
position (it is what `fit_observation` uses for `input_path`). To read a raw
JWST FITS file or a CSV with named `wave`/`flux`/`err` columns, use
`diskmelts.consub.read_spectrum` ({doc}`consub`).

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
