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
columns. The v2 workflow generates one complete table and reuses portions of it:

```text
Pretrain_grid/pretrain_<molecule>_full.csv
```

If `Model_grids/<molecule>/` is absent, the example script and notebook stop
early with a message explaining that the local data is missing.

## Segmented H2O bank

The initial v2 H2O configuration trains five overlapping wavelength ranges,
three overlapping `logN` ranges, and three overlapping temperature ranges:

- wavelength: `(4.9, 7.4)`, `(7.2, 11.2)`, `(9.9, 15.1)`,
  `(13.5, 19.2)`, `(19.0, 25.0)` µm;
- `logN`: `(13, 19)`, `(18.5, 22)`, `(21.5, 24)`;
- temperature: `(50, 700)`, `(600, 1300)`, `(1200, 2000)` K.

The Cartesian product produces 45 checkpoints and an `H2O_model_bank.json`
manifest under `Trained_model/H2O_bank_v2/`. During fitting, overlapping tile
predictions are blended smoothly. All tiles describe one H2O component and
share one fitted `(T, logN, A)` parameter set.

The initial carbon-bearing configuration covers `HCN`, `C2H2`, `CO2`,
`13C12CH2`, and `13CO2`. Each molecule uses one wavelength range
`11.5–17.0` µm, one `logN` range `13–19.5`, and three temperature ranges
`200–800`, `600–1500`, and `1200–2000` K, producing three checkpoints per
molecule. The same `logN` and temperature configuration is intended for later
carbon-bearing molecules; only their wavelength configuration needs to change.

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
python examples/dev_v2_pt_validation.py
```

Select molecule groups using the three switches near the top of the script:

```python
run_H2O = True
run_Cmol = True
run_rarer = True
```

`run_Cmol` currently selects `HCN`, `C2H2`, `CO2`, `13C12CH2`, and `13CO2`.
`run_rarer` is reserved for later species such as `C4H2`, `HC3N`, and `C2H6`;
it has no effect until their wavelength configurations are added.

For each selected molecule, the script performs:

1. model-grid loading
2. complete pretraining CSV generation or reuse
3. training or loading every configured forward-model tile
4. JSON manifest generation
5. tile validation using each tile's withheld 10% validation split
6. assembled-bank reconstruction on 100 grid spectra with a residual plot

The default settings are intended for scientific runs and may take substantial
time. Set `DISKMELTS_VALIDATE=0` to train without retrieval validation.
Diagnostic controls include `DISKMELTS_VAL_N`, `DISKMELTS_VAL_NT_STARTS`,
`DISKMELTS_VAL_NT_STEPS`, and `DISKMELTS_BANK_RECON_N`.

## Notebook

The existing notebook demonstrates the original single-checkpoint workflow:

```text
notebooks/Example_Training_Validation.ipynb
```

Use `examples/dev_v2_pt_validation.py` for segmented v2 banks until the training
notebook is migrated to the bank API.

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

Load the complete v2 bank by passing its manifest to `load_models`:

```python
pretrained = load_models({
    'H2O': 'Trained_model/H2O_bank_v2/H2O_model_bank.json',
})
```

To reduce memory and evaluation cost for a narrow fit, load only intersecting
tiles:

```python
from diskmelts import load_model_bank

h2o = load_model_bank(
    'Trained_model/H2O_bank_v2/H2O_model_bank.json',
    wav_ranges=[(13.5, 19.2)],
    T_range=(200, 700),
    logN_range=(18.5, 22),
)
pretrained = {'H2O': h2o}
```
