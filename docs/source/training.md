# Training and Validation

Training is a separate workflow from fitting observed spectra. The supplied
checkpoints are ready for fitting, while retraining requires the full local slab
model grids.

## Local-only data

The following directories are intentionally ignored by Git:

```text
Model_grids/
Pretrain_grid/
```

Place each molecule's slab-model CSV files under:

```text
Model_grids/<molecule>/
```

Grid files must use names such as `T500N17.0.csv` and contain `wave` and `Line`
columns. The workflow generates or reuses:

```text
Pretrain_grid/pretrain_<molecule>_11to19.csv
```

If `Model_grids/<molecule>/` is absent, the example script and notebook stop
early with a message explaining that the local data is missing.

## PCA configuration

The example training workflows use these component counts:

| Molecule | PCA components |
|---|---:|
| `H2O` | 21 |
| `C2H2` | 15 |
| `13C12CH2` | 15 |
| `HCN` | 15 |
| `CO2` | 15 |
| `13CO2` | 15 |

New checkpoints store the fitted PCA object, input/output scalers, wavelength
axis, molecule name, wavelength range, and PCA count. They can therefore be
loaded later with only `model_paths`.

## Example script

From the repository root:

```bash
python examples/dev_v1_pt_validation.py
```

Set `MOL` in the configuration block, or use the optional environment override:

```bash
DISKMELTS_MOL=CO2 python examples/dev_v1_pt_validation.py
```

The script performs:

1. model-grid loading
2. deterministic holdout selection
3. pretraining CSV generation or reuse
4. forward-model training or checkpoint loading
5. in-pretraining and unseen-holdout validation

The default settings are intended for scientific runs and may take substantial
time. The environment overrides used for short diagnostics include
`DISKMELTS_HOLDOUT_N`, `DISKMELTS_VAL_NT_FRAC`,
`DISKMELTS_VAL_NT_STARTS`, and `DISKMELTS_VAL_NT_STEPS`.

## Notebook

The equivalent interactive workflow is:

```text
notebooks/Example_Training_Validation.ipynb
```

It resolves all data and output paths relative to the repository root, whether
Jupyter starts in the root directory or in `notebooks/`.

## Loading a trained checkpoint

Once training has saved a self-contained checkpoint, fitting does not need the
pretraining CSV:

```python
from diskmelts import load_models

pretrained = load_models(
    model_paths={
        'H2O': 'Trained_model/net_H2O_forward_11to19.pt',
    },
)
```

The optional `pretrain_csv_paths`, `wav_ranges`, and `n_pca` arguments to
`load_models` remain only for loading older checkpoints that do not contain
their own scaler and wavelength metadata.
