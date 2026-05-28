# Quick Start

## Workflow overview

DiskMELTS has two main workflows:

1. **Train** a forward surrogate for each molecule (done once per molecule).
2. **Fit** a real observed spectrum using the trained surrogates.

A ready-to-run example script is provided for each:

| Task | Script |
|---|---|
| Train + validate | `examples/dev_v1_pt_validation.py` |
| Fit a real spectrum | `examples/dev_v1_realobs.py` |

---

## 1. Train a forward model

```python
import numpy as np
from diskmelts import load_model_grid, generate_pre_training_set, pretrain_forward_model

# Load the slab-model grid for H2O
models = load_model_grid('Model_grids/H2O')
wav_grid = next(iter(models.values()))['wavelength']

# Build the pretrain CSV (one row per grid point)
generate_pre_training_set(
    mol='H2O',
    models=models,
    output_path='Pretrain_grid/pretrain_H2O_11to19.csv',
    wav_out=wav_grid,
)

# Train the two-MLP forward model (or load if checkpoint exists)
pretrained = pretrain_forward_model(
    mol='H2O',
    pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
    wav_range=(11.0, 19.0),
    model_path='Trained_model/net_H2O_forward_11to19.pt',
    n_pca=21,
)
```

Per-molecule defaults:

| Molecule | `wav_range` (µm) | `fit_ranges` (µm) | `n_pca` |
|---|---|---|---|
| H2O | 11–19 | [(11,12), (16.5,18.5)] | 21 |
| C2H2 | 11–17.5 | (12, 16.5) | 15 |
| HCN | 11–17.5 | (12, 17.0) | 15 |
| CO2 | 11–17.5 | (12, 16.5) | 15 |

---

## 2. Validate the forward model

```python
from diskmelts import load_models, validate_nt, validate_nt_holdout, plot_validation_split

pretrained = load_models(
    model_paths       = {'H2O': 'Trained_model/net_H2O_forward_11to19.pt'},
    pretrain_csv_paths= {'H2O': 'Pretrain_grid/pretrain_H2O_11to19.csv'},
    wav_ranges        = {'H2O': (11.0, 19.0)},
    n_pca             = {'H2O': 21},
)

# NT validation on in-pretrain rows (A = 1 fixed)
nt_pretrain = validate_nt(
    'H2O', pretrained,
    pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
)

# NT validation on held-out grid points (unseen during training)
nt_holdout = validate_nt_holdout(
    'H2O', pretrained, models, holdout_keys,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
)

plot_validation_split('H2O', nt_pretrain, nt_holdout,
                      save_path='figures/val_H2O_split.png')
```

---

## 3. Fit a real spectrum

The recommended entry point is `examples/dev_v1_realobs.py`.
Edit the `CONFIGURATION` block at the top, then run from the repo root:

```bash
python examples/dev_v1_realobs.py
```

Key configuration variables:

| Variable | Default | Description |
|---|---|---|
| `INPUT_PATH` | — | Path to continuum-subtracted CSV |
| `NAME` | — | Source name for filenames and plot title |
| `N_SAMPLES` | 20000 | Sobol samples for global search |
| `N_REFINE` | 32 | Candidates passed to L-BFGS-B |
| `N_TOP` | 20 | Solutions kept for uncertainty estimation |
| `DETECTION_SCREENING` | `True` | Skip molecules below 3σ |
| `STAGES` | list of dicts | Ordered fit stages |

Or call `fit_molecules` directly in a script or notebook:

```python
from diskmelts import load_models, load_observed_spectrum, fit_molecules, plot_fit

pretrained = load_models(
    model_paths       = {'H2O': 'Trained_model/net_H2O_forward_11to19.pt'},
    pretrain_csv_paths= {'H2O': 'Pretrain_grid/pretrain_H2O_11to19.csv'},
    wav_ranges        = {'H2O': (11.0, 19.0)},
    n_pca             = {'H2O': 21},
)

obs_wav, obs_flux = load_observed_spectrum('Realobs_data/Consub_data/my_source.csv')

fit = fit_molecules(
    obs_wav, obs_flux,
    mol='H2O',
    pretrained=pretrained,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    h2o_components=2,
    component_names=['H2O_warm', 'H2O_hot'],
    n_samples=20000,
    sigma=0.001,   # noise std in Jy for uncertainty weighting
)

plot_fit(obs_wav, obs_flux, fit,
         fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
         name='MySource',
         save_path='figures/myfit.png',
         params=fit['params'],
         uncertainty=fit['uncertainty'])
```

### Two-component H2O

Set `h2o_components=2` and `component_names=['H2O_warm', 'H2O_hot']` to fit
two water components simultaneously using the same pretrained H2O surrogate.
Use `T_prior` with `T_prior_log10=True` to apply log-temperature Gaussian
priors that help separate the two components:

```python
fit = fit_molecules(
    obs_wav, obs_flux,
    mol='H2O',
    pretrained=pretrained,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    h2o_components=2,
    component_names=['H2O_warm', 'H2O_hot'],
    sigma=0.001,
    T_prior={'H2O_warm': (2.6, 0.2), 'H2O_hot': (2.9, 0.1)},
    T_prior_log10=True,
)
```

---

## Forward model convention

```
flux(T, logN, A) = A × peak(T, logN) × shape(T, logN)
```

The linear amplitude `A` is solved analytically with NNLS (or `lsq_linear`
when bounds are set) after the nonlinear `(T, logN)` Sobol search, making
the optimisation much faster than treating `A` as a free nonlinear parameter.
