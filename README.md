# DiskMELTS

DiskMELTS (Machine-learning Enabled Line fitting Tool for Spectra) is a
package for continuum subtraction and fitting molecular emission in JWST
mid-infrared spectra of protoplanetary disks. Its fitting workflow loads
pretrained molecular surrogate models and retrieves temperature (`T`), column
density (`logN`), and emitting area scaling (`A`) from observed spectra.

## Installation

Requires Python 3.9 or newer. Start from a clone of this repository.

### Conda

From the repository root, create and activate the supplied environment. It
installs DiskMELTS in editable mode, with FITS and notebook support:

```bash
conda env create --file environment.yaml
conda activate diskmelts
```

### Custom environment

In an existing Python environment, run from the repository root:

```bash
python -m pip install -e .
```

For FITS reading, use `python -m pip install -e ".[io]"`. CSV and array inputs
need no optional dependencies. Select this same environment as your Jupyter kernel;
restart an existing kernel after upgrading the package.

## Runtime dependencies

| Package | Purpose |
|---|---|
| `torch` | Pretrained model inference |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scipy` | L-BFGS-B optimizer, NNLS, Sobol sampling |
| `scikit-learn` | Model scaling and PCA metadata |
| `matplotlib` | Plotting |

## Continuum subtraction

For a CSV with `wave,flux,err` columns (wavelength in µm and flux/error in the
same units):

```python
import diskmelts

wav, flux, error = diskmelts.consub.read_spectrum("raw_spectrum.csv")
wav, consub, err, cont, contOrig = diskmelts.consub.calculate(
    wav, flux, error, preset="ttauri", exclude=[[12, 13]],
)
fig = diskmelts.consub.plot(
    wav, consub, err, cont, contOrig, ranges=[(5, 12), (12, 25)], savedir=None,
)
```

`calculate()` returns five NumPy arrays. `plot()` displays the raw spectrum,
continuum, and residual; `savedir="figures/consub"` also writes `continuum.png`.
Its `error` and `contOrig` inputs may be `None`.

Presets are `water`, `ttauri`, `carbon_rich`, and `carbon_extended`.
Use `preset=None, linefreefile="windows.csv"` for custom `xmin,xmax` windows.
With `preset=None, linefreefile=None` (or `""`), fitting uses all nonexcluded
pixels and skips the line-free correction, so `cont` equals `contOrig`.
Reference tables are bundled; normal continuum subtraction needs no model grids.

The private `0-Consub/` experiment directory is excluded from Git and is not
required by the installed package. Its runtime code and reference tables are
maintained under `src/diskmelts/consub/`.
The distributed tests in `tests/` exercise the installed continuum API and
bundled references without that private directory.

Returned `err` represents measurement/noise error, not correlated continuum
uncertainty. Broad excluded intervals require inspection. See the
[continuum guide](docs/source/continuum.md) for presets, regional box sizes,
special regions, FITS input, saving CSVs, and uncertainty diagnostics.

## Quick fitting example

```python
from diskmelts import load_fitting_models, load_observed_spectrum, fit_stage

pretrained = load_fitting_models(['H2O'])

obs_wav, obs_flux = load_observed_spectrum(
    'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv'
)

fit = fit_stage(
    obs_wav,
    obs_flux,
    molecule='H2O',
    wavelength_range=(11.0, 19.0),
    pretrained=pretrained,
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    distance_pc=122.5,
    n_samples=20000,
    n_refine=32,
    n_top=20,
    sigma=0.001,
)

print(fit['params'])
```

You can also read spectra with your own code. DiskMELTS only requires
one-dimensional `obs_wav` and `obs_flux` arrays.

For a complete staged fit-and-subtract workflow, edit and run:

```bash
python examples/dev_v2_realobs.py
```

The v2 fitting API is `load_fitting_models`, `fit_stage`, `fit_observation`, and
`plot_saved_observation`. The production checkpoints are packaged under
`src/diskmelts/models/` and included in the wheel. Model fitting requires no
`Model_grids/` or `Pretrain_grid/` directory. The model metadata, including
PCA counts and network layers, are in `diskmelts.MODEL_SPECS` and checked when
loading. The checkpoint flux convention is 1 au² at 140 pc; `distance_pc`
sets the source distance and makes the returned emitting area physical.

For an optional fit directly from slab grids, use `fit_grid_stage` or
`fit_observation_grids` and supply `grid_root='Model_grids'` explicitly. This
separate method does require the full local grids. See the
[quick start](docs/source/quickstart.md) for both workflows and saved-result
plotting.

## Training and validation data

`Model_grids/` and `Pretrain_grid/` are intentionally ignored by Git because
they contain the full local training data. The generated checkpoint directories
under the top-level `Trained_model/` are also optional; the package bundles
only the production checkpoints needed for v2 fitting. To run
`examples/dev_v2_pt_validation.py` or
`notebooks/Example_Training_Validation.ipynb`, place the complete model grids
under `Model_grids/<molecule>/`. The workflows create or reuse the corresponding
pretraining CSV under `Pretrain_grid/`.

The maintained v2 workflow uses a hybrid design after restricting the expanded
IRIS grids to the intended physical domain. H2O uses five overlapping
wavelength checkpoints covering 4.9–25.0 µm, all sharing `T=100–1400 K` and
`logN=13–19`. Primary carbon-bearing and rarer molecules use one checkpoint
each, the same temperature range, and `logN=13–19.5`.

Five to ten percent of complete `(T, logN)` grid spectra are excluded before
PCA or neural-network training. They are used to select the PCA dimension and
to validate spectral reconstruction and full parameter fitting. The
`run_H2O`, `run_Cmol`, and `run_rarer` switches select which groups run.

## Forward model convention

```text
flux(T, logN, A, d) = A * (140 pc / d)^2 * peak(T, logN) * shape(T, logN)
```

Each molecule has two pretrained MLPs:

- `net_shape`: `(T, logN)` to PCA coefficients of the peak-normalized spectral shape
- `net_peak`: `(T, logN)` to `log10(peak flux)` in Jy

The linear amplitude `A` is solved analytically with NNLS or bounded least
squares after the nonlinear `(T, logN)` search.

## Contributors

- Chengyan Xie (University of Arizona)
- Dingshan Deng (University of Arizona)
