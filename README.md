# DiskMELTS

DiskMELTS (Machine-learning Enabled Line fitting Tool for Spectra) is a
neural surrogate-assisted package for fitting molecular emission in JWST
mid-infrared spectra of protoplanetary disks. Its primary workflow is to load
pretrained molecular surrogate models and retrieve temperature (`T`), column
density (`logN`), and emitting area scaling (`A`) from observed spectra.

## Installation

```bash
conda env create -f environment.yaml
conda activate diskmelts
```

or install into an existing environment:

```bash
pip install -e .
```

## Required packages

| Package | Purpose |
|---|---|
| `torch` | Pretrained model inference |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scipy` | L-BFGS-B optimizer, NNLS, Sobol sampling |
| `scikit-learn` | Model scaling and PCA metadata |
| `matplotlib` | Plotting |

## Quick fitting example

```python
from diskmelts import load_models, load_observed_spectrum, fit_molecules, plot_fit

pretrained = load_models(
    model_paths={
        'H2O': 'Trained_model/net_H2O_forward_11to19.pt',
    },
    pretrain_csv_paths={
        'H2O': 'Pretrain_grid/pretrain_H2O_11to19.csv',
    },
    wav_ranges={
        'H2O': (11.0, 19.0),
    },
    n_pca={
        'H2O': 21,
    },
)

obs_wav, obs_flux = load_observed_spectrum('Realobs_data/Consub_data/my_source.csv')

fit = fit_molecules(
    obs_wav,
    obs_flux,
    mol='H2O',
    pretrained=pretrained,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    n_samples=20000,
    n_refine=32,
    n_top=20,
    sigma=0.001,
)

plot_fit(
    obs_wav,
    obs_flux,
    fit,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    name='my_source',
    save_path='figures/my_source_h2o.png',
    params=fit['params'],
    uncertainty=fit['uncertainty'],
)
```

You can also read spectra with your own code. DiskMELTS only requires
one-dimensional `obs_wav` and `obs_flux` arrays.

For a complete staged fit-and-subtract workflow, edit and run:

```bash
python examples/dev_v1_realobs.py
```

## Forward model convention

```text
flux(T, logN, A) = A * peak(T, logN) * shape(T, logN)
```

Each molecule has two pretrained MLPs:

- `net_shape`: `(T, logN)` to PCA coefficients of the peak-normalized spectral shape
- `net_peak`: `(T, logN)` to `log10(peak flux)` in Jy

The linear amplitude `A` is solved analytically with NNLS or bounded least
squares after the nonlinear `(T, logN)` search.

## Contributors

- Chengyan Xie (University of Arizona)
- Dingshan Deng (University of Arizona)
