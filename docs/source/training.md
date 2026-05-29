# Training New Models

Training is an optional advanced workflow. Most users should use the pretrained
models and start with {doc}`quickstart`.

Use training only when you have your own slab-model grids, want to add a new
molecule, or need a different wavelength range or physical setup.

## 1. Build the pretrain CSV

```python
from diskmelts import load_model_grid, generate_pre_training_set

models = load_model_grid('Model_grids/H2O')
wav_grid = next(iter(models.values()))['wavelength']

generate_pre_training_set(
    mol='H2O',
    models=models,
    output_path='Pretrain_grid/pretrain_H2O_11to19.csv',
    wav_out=wav_grid,
)
```

Each slab-grid file should be named like `T500N17.0.csv` and contain columns
named `wave` and `Line`.

## 2. Train or load the surrogate

```python
from diskmelts import pretrain_forward_model

pretrained_h2o = pretrain_forward_model(
    mol='H2O',
    pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
    wav_range=(11.0, 19.0),
    model_path='Trained_model/net_H2O_forward_11to19.pt',
    n_pca=21,
)
```

If `model_path` already exists, `pretrain_forward_model` loads the checkpoint
instead of retraining.

## Default settings

| Molecule | Training `wav_range` (micron) | Typical fitting range (micron) | `n_pca` |
|---|---:|---:|---:|
| H2O | 11.0-19.0 | 11.0-12.0 and 16.5-18.5 | 21 |
| C2H2 | 12.0-16.5 | 12.0-16.5 | 15 |
| HCN | 12.0-17.0 | 12.0-17.0 | 15 |
| CO2 | 12.0-16.5 | 12.0-16.5 | 15 |

## Validation

After training, run a recovery check on pretrain or held-out grid spectra:

```python
from diskmelts import load_models, validate_nt, plot_validation

pretrained = load_models(
    model_paths={'H2O': 'Trained_model/net_H2O_forward_11to19.pt'},
    pretrain_csv_paths={'H2O': 'Pretrain_grid/pretrain_H2O_11to19.csv'},
    wav_ranges={'H2O': (11.0, 19.0)},
    n_pca={'H2O': 21},
)

nt_pretrain = validate_nt(
    'H2O',
    pretrained,
    pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
)

plot_validation(
    'H2O',
    nt_pretrain,
    save_path='figures/val_H2O.png',
)
```
