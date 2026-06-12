# Quick Start

DiskMELTS is intended primarily for fitting spectra with pretrained surrogate
models. This page shows the standard fitting workflow for observed data. Run
the path-based examples below from the repository root.

## 1. Load the pretrained models

Bundled model checkpoints are self-contained: each `.pt` file stores the
network weights, scalers, PCA metadata, and wavelength grid needed for fitting.
No `Pretrain_grid/*.csv` file is needed for this workflow.

```python
from diskmelts import load_models

pretrained = load_models(
    model_paths={
        'H2O':  'Trained_model/net_H2O_forward_11to19.pt',
        'C2H2': 'Trained_model/net_C2H2_forward_11to19.pt',
        'HCN':  'Trained_model/net_HCN_forward_11to19.pt',
        'CO2':  'Trained_model/net_CO2_forward_11to19.pt',
    },
)
```

## 2. Load a spectrum

If your continuum-subtracted spectrum is a simple CSV, use the built-in helper:

```python
from diskmelts import load_observed_spectrum

obs_wav, obs_flux = load_observed_spectrum(
    'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv'
)
```

You can also read the spectrum with your own code. The fitting functions only
need two one-dimensional arrays: `obs_wav` in microns and `obs_flux` in Jy.

## 3. Fit H2O

```python
from diskmelts import fit_molecules, plot_fit

fit_h2o = fit_molecules(
    obs_wav,
    obs_flux,
    mol='H2O',
    pretrained=pretrained,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    h2o_components=1,
    n_samples=20000,
    n_refine=32,
    n_top=20,
    sigma=0.001,
    seed=42,
)

plot_fit(
    obs_wav,
    obs_flux,
    fit_h2o,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    name='my_source',
    save_path='figures/my_source_h2o.png',
    params=fit_h2o['params'],
    uncertainty=fit_h2o['uncertainty'],
)
```

## 4. Fit C-bearing molecules

After subtracting water, C$_2$H$_2$, HCN, and CO$_2$ are usually fitted on the
residual spectrum. The example below fits C$_2$H$_2$ and HCN together.

```python
residual_after_h2o = obs_flux - fit_h2o['model_flux']

fit_c = fit_molecules(
    obs_wav,
    residual_after_h2o,
    mol=['C2H2', 'HCN'],
    pretrained=pretrained,
    fit_ranges=(12.0, 16.5),
    n_samples=20000,
    n_refine=32,
    n_top=20,
    sigma=0.001,
    seed=42,
)
```

## Two-component H2O

To fit warm and hot water components simultaneously, set `h2o_components=2`.

```python
fit_h2o_two = fit_molecules(
    obs_wav,
    obs_flux,
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

## Example script

For a complete staged workflow, start from:

```bash
python examples/dev_v1_realobs.py
```

Edit the configuration block at the top of that script for your source name,
input spectrum path, molecules, wavelength masks, and output directory.

The default script uses the committed
`Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv` spectrum. It checks
that the spectrum and checkpoints exist before starting the fit.

The equivalent interactive workflow is:

```text
notebooks/Example_Fitting.ipynb
```

The notebook resolves the repository root whether Jupyter is launched from the
repository root or from `notebooks/`.

Training and checkpoint validation require additional local files. See
{doc}`training`.
