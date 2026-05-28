# DISKMELTS

DiskMELTS (Machine-learning Enabled Line fitting Tool for Spectra) is a neural surrogate-assisted framework designed to retrieve molecular gas parameters, specifically temperature ($T$), column density ($\log N$), and emitting area ($A$), from JWST mid-IR spectra (11-19$\mu$m) of protoplanetary disks. Beyond its pre-trained capabilities, the package provides a flexible training suite, allowing users to develop custom models for fitting spectra at any wavelength.

---

## Required packages

| Package | Purpose |
|---|---|
| `torch` | MLP training and inference |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scipy` | L-BFGS-B optimiser, NNLS, Sobol sampling, `lsq_linear` |
| `sklearn` | `StandardScaler`, PCA, `train_test_split` |
| `matplotlib` | Plotting |

---

## Workflow

### Examples of the workflow can be found in notebooks/Example_*.ipynb

### 1. Train a forward model and validate model accuracy (Functions in `Trainmodel.py` and `Validation.py`, examples in `notebooks/Example_Example_Training_Validation.ipynb` or `dev_v1_pt_validation.py`. For advanced users only, train your own models.) 

```python
from Trainmodel import load_model_grid, generate_pre_training_set, pretrain_forward_model

models = load_model_grid('Model_grids/H2O')

generate_pre_training_set(
    mol='H2O', models=models,
    output_path='Pretrain_grid/pretrain_H2O_11to19.csv',
    wav_out=wav_grid,          # 11–19 µm, 3221-point array
)

pretrained = pretrain_forward_model(
    mol='H2O',
    pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
    wav_range=(11.0, 19.0),    # wavelength axis for PCA
    model_path='Trained_model/net_H2O_forward_11to19.pt',
    n_pca=21,
)
```
Per-molecule defaults:

| Molecule | `wav_range` (µm) | `fit_ranges` (µm)      | `n_pca` |
| -------- | ---------------- | ---------------------- | ------- |
| H2O      | 11–19            | [(11,12), (16.5,18.5)] | 21      |
| C2H2     | 11–17.5          | (12, 16.5)             | 15      |
| HCN      | 11–17.5          | (12, 17.0)             | 15      |
| CO2      | 11–17.5          | (12, 16.5)             | 15      |
| 13C12CH2 | 11–17.5          | (12, 16.5)             | 15      |

```python
from Fitting    import load_models
from Validation import validate_nt, validate_full
from Plotting   import plot_validation

pretrained = load_models(
    model_paths={'H2O': 'Trained_model/net_H2O_forward_11to19.pt'},
    pretrain_csv_paths={'H2O': 'Pretrain_grid/pretrain_H2O_11to19.csv'},
    wav_ranges={'H2O': (11.0, 19.0)},
    n_pca={'H2O': 21},
)

# T, logN only (A = 1 fixed) — tests forward-model accuracy
nt = validate_nt('H2O', pretrained,
                 pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
                 fit_ranges=[(11.0, 12.0), (16.5, 18.5)])
plot_validation('H2O', nt, save_path='figures/val_H2O_nt.png')

# T, logN, A full pipeline — tests end-to-end retrieval
full = validate_full('H2O', models, pretrained,
                     fit_ranges=[(11.0, 12.0), (16.5, 18.5)])
plot_validation('H2O', full, save_path='figures/val_H2O_full.png')
```

Or use the end-to-end script (edit `MOL` at the top):

```bash
python Package/dev_v1_pt_validation.py
```
---

### 2. Fit a real spectrum (`Fitting.py` + `dev_v1_realobs.py`)

The real-observation workflow is a sequential fit-and-subtract pipeline:

1. Load the continuum-subtracted spectrum.
2. Estimate noise from line-free continuum windows.
3. Fit the molecules one by one. If you want to fit several molecules together, put them in the same stage. Between each stage, the data will be subtracted with the fitted model and the residual will be used for the next stage fitting. For H2O, there is an option to fit 2 components together. 
4. Save per-stage residuals and a combined plot.

Edit the `CONFIGURATION` block in `dev_v1_realobs.py`, then run:

```bash
conda run -n data_reduction python Package/dev_v1_realobs.py
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
| `STAGES` | list of dicts | Ordered fit stages; each specifies `mol`, `fit_ranges`, optional bounds |

**Two-component H2O** — set `h2o_components=2` and `component_names=['H2O_warm', 'H2O_hot']` in the stage dict. Use `T_prior` with `T_prior_log10=True` to apply log10(T) Gaussian priors.


---

## Forward model convention

```
flux(T, logN, A) = A * peak(T, logN) * shape(T, logN)
```

Each molecule has two MLPs:
- `net_shape`: `(T, logN)` → `n_pca` PCA coefficients of the peak-normalised spectral shape
- `net_peak`: `(T, logN)` → `log10(peak flux)` in Jy

Inputs are standardised with `StandardScaler`. The linear amplitude `A` is solved analytically with NNLS (or `lsq_linear` when bounds are set) after the nonlinear `(T, logN)` search.



## Contributors

- Chengyan Xie (University of Arizona)
- Dingshan Deng (University of Arizona)
