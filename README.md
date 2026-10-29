# DiskMELTS

DiskMELTS (Machine-learning Enabled Line fitting Tool for Spectra) is a
package for continuum subtraction and fitting molecular emission in JWST
mid-infrared spectra of protoplanetary disks. Its fitting workflow loads
pretrained molecular surrogate models and retrieves temperature (`T`), column
density (`logN`), and emitting area scaling (`A`) from observed spectra.

**Documentation:** [diskmelts.readthedocs.io](https://diskmelts.readthedocs.io/en/latest/)

## Installation

DiskMELTS is available on [PyPI](https://pypi.org/project/diskmelts/) and
requires Python 3.9 or newer. It bundles 46 production checkpoints, 23 each
for the sigma=0 and sigma=2 grids, covering the same 18 species (see
[Available molecules](#available-molecules)).
Install it from any directory. These commands use
`python` in an activated environment; the
[installation guide](https://diskmelts.readthedocs.io/en/latest/installation.html)
shows how to create one on macOS, Linux, Windows, or Conda:

```bash
python -m pip install diskmelts
python -c "import diskmelts; print('DiskMELTS is ready')"
```

Optional features can be installed in the same way:

```bash
python -m pip install "diskmelts[io]"            # FITS input
python -m pip install "diskmelts[io,notebooks]"  # FITS input and JupyterLab
```

The pip package includes the production checkpoints and continuum reference
tables. Full slab grids for training and direct-grid fitting are separate
downloads; pass their local path as `grid_root`.
Example notebooks and spectra can be downloaded from the repository when
needed; they are not installed by pip.

To install from a checkout with Conda or Mamba, run:

```bash
git clone https://github.com/wddlx/DiskMELTS.git
cd DiskMELTS
conda env create -f environment.yaml
conda activate diskmelts
```

Use `environment-notebooks.yaml` or `environment-dev.yaml` for the corresponding
notebook or contributor dependencies.

## Minimal workflow

Subtract the continuum, fit one molecule, and print the result:

```python
from diskmelts import consub, fit_stage, load_fitting_models

# 1. Raw spectrum: CSV with wave (µm), flux (Jy), and optional err columns.
raw_wave, raw_flux, raw_err = consub.read_spectrum("my_raw_spectrum.csv")
wave, flux, err, cont, cont_unshifted = consub.calculate(
    raw_wave, raw_flux, raw_err, preset="ttauri",
)

# 2. Fit C2H2 near 13.7 µm with the bundled surrogate model.
models = load_fitting_models("C2H2")
fit = fit_stage(
    wave, flux, molecule="C2H2", pretrained=models,
    wavelength_range=(13.0, 15.0), fit_ranges=(13.5, 14.1),
    distance_pc=140.0, n_samples=1024, n_refine=4, n_top=4,
    n_mcmc=20000,  # quick look; omit for the converged default (~1 min)
    # errorbars=False,  # skip the MCMC error bars and return the best fit only
)

# 3. Best fit, then posterior medians and 16-84% error bars (from MCMC).
print(fit["params"]["C2H2"])
print(fit["uncertainty"]["C2H2"])
```

The sections below explain each step, and the
[first-spectrum guide](https://diskmelts.readthedocs.io/en/latest/quickstart.html)
covers multi-stage fits, saved outputs, and plots.

## Runtime dependencies

| Package | Purpose |
|---|---|
| `torch` | Pretrained model inference |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scipy` | L-BFGS-B optimizer, NNLS, Sobol sampling |
| `scikit-learn` | PCA and scaling when training new models |
| `matplotlib` | Plotting |
| `emcee` | MCMC sampling of parameter error bars |

## Continuum subtraction

Try the [continuum subtraction notebook](https://github.com/wddlx/DiskMELTS/blob/main/notebooks/Example_Consub.ipynb)
with the included raw Upper Sco J16120505 spectrum. It demonstrates the
five-array result and a plot; change `RAW_CSV` to another file in
[`Realobs_data/Raw_data/`](https://github.com/wddlx/DiskMELTS/tree/main/Realobs_data/Raw_data) to process another source.

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

Returned `err` represents measurement/noise error, not correlated continuum
uncertainty. Broad excluded intervals require inspection. See the
[continuum guide](https://diskmelts.readthedocs.io/en/latest/continuum.html) for presets, regional box sizes,
special regions, FITS input, saving CSVs, and uncertainty diagnostics.

## Quick fitting example

Use a continuum-subtracted CSV with a header and `wave,flux` as its first two
columns, with wavelength in µm and flux in Jy. For a runnable example without
cloning the repository, download the
[J16120505 spectrum](https://github.com/wddlx/DiskMELTS/blob/main/Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv)
as `j16120505_contsub.csv`, or run:

```bash
curl -L -o j16120505_contsub.csv https://raw.githubusercontent.com/wddlx/DiskMELTS/main/Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv
```

```python
from diskmelts import fit_stage, load_fitting_models, load_observed_spectrum

models = load_fitting_models('C2H2')

obs_wav, obs_flux = load_observed_spectrum(
    'j16120505_contsub.csv'
)

fit = fit_stage(
    obs_wav,
    obs_flux,
    molecule='C2H2',
    wavelength_range=(13.0, 15.0),
    pretrained=models,
    fit_ranges=(13.5, 14.1),
    distance_pc=122.5,
    n_samples=1024,
    n_refine=4,
    n_top=4,
)

print(fit['params']['C2H2'])
```

You can also read spectra with your own code. DiskMELTS only requires
one-dimensional `obs_wav` and `obs_flux` arrays. Replace the sample CSV with
your own file to fit your source; update `distance_pc` and fit intervals.

For a complete staged fit-and-subtract workflow from a clone of the
repository, run:

```bash
python examples/dev_v2_realobs.py
python examples/dev_v2_realobs.py --input /path/to/my_spectrum.csv --name my_source --distance-pc 140 --sigma 2
```

Adjust that script's stage windows for your spectrum.
`python examples/synthetic_array_fit.py` generates an HCN spectrum and fits it
from in-memory arrays, without an observation file.

The v2 fitting API is `load_fitting_models`, `fit_stage`, `fit_observation`, and
`plot_saved_observation`. The production checkpoints are included in the
wheel, so model fitting needs no local model grids. The checkpoint flux
convention is 1 au² at 140 pc; `distance_pc` sets the source distance and
makes the returned emitting area physical. `fit_observation` accepts either a
CSV path or in-memory `obs_wav`, `obs_flux`, and optional `obs_err` arrays
(new in 2.2.1).

Every fit returns its best fit and, by default, error bars from MCMC sampling
of the posterior with [emcee](https://emcee.readthedocs.io/). Reported values
are posterior medians with 16–84% error bars; the noise level accounts for
correlated pixels and model mismatch, and a `*` marks a range that reaches a
fit bound. Sampling takes about a minute per stage. Pass `errorbars=False`
for best fits only. See
[What the uncertainties mean](https://diskmelts.readthedocs.io/en/latest/quickstart.html#uncertainties).

For a synthetic spectrum, `wave, flux = diskmelts.synthesize_spectrum('HCN',
T=300, logN=18, A=1)` returns the native wavelength axis and model flux (new
in 2.2.1). See the
[generation guide](https://diskmelts.readthedocs.io/en/latest/generate_spectra.html).

The [fitting notebook](https://github.com/wddlx/DiskMELTS/blob/main/notebooks/Example_Fitting.ipynb)
requires 13C12CH2 to be no hotter than C2H2 and gives them the same emitting
area; it ties 13CO2 to CO2 in both temperature and area. Point its input at
your own continuum-subtracted CSV, or pass arrays. Set
`isotopologue_groups`, `Tshare` (`free`, `equal`, or `less`), and `Ashare` in a
joint stage to choose those constraints. See the
[fitting guide](https://diskmelts.readthedocs.io/en/latest/quickstart.html) for examples.

For an optional fit directly from slab grids, use `fit_grid_stage` or
`fit_observation_grids` and supply `grid_root='Model_grids/sigma0'` explicitly. This
separate method does require the full local grids. See the
[quick start](https://diskmelts.readthedocs.io/en/latest/quickstart.html) for both workflows and saved-result
plotting.
For complete sigma=2 grids, pass `grid_root='Model_grids/sigma2'`.

## Available molecules

Each molecule is available for two slab-model grids: sigma=0 (default) assumes
no turbulent broadening, so lines are only thermally broadened, and sigma=2
assumes 2 km/s turbulent broadening. Select sigma=2 with `load_fitting_models(..., sigma=2)` or
`fit_observation(..., sigma=2)`. Fit windows must lie inside a model's band.

| Molecule | Band (µm) | T (K) | logN (cm⁻²) |
|---|---|---|---|
| H2O | 4.9–25.0 (five overlapping tiles) | 100–1400 | 13–19 |
| C2H2 | 6–9 and 11.5–17 | 100–1400 | 13–19.5 |
| HCN, CO2, 13C12CH2, 13CO2 | 11.5–17 | 100–1400 | 13–19.5 |
| HC3N, CO17O, CO18O | 13–18 | 100–1400 | 13–19.5 |
| C4H2 | 14–18 | 100–1400 | 13–19.5 |
| C3H4, CH3 | 15–18 | 100–1400 | 13–19.5 |
| C6H6 | 13–17 | 100–675 | 13–19.5 |
| C2H6 | 9–15 | 100–1400 | 13–19.5 |
| C2H4 | 8–14 | 100–1400 | 13–19.5 |
| CH4 | 5–9 | 100–1400 | 13–19.5 |
| C2HD | 11.5–20.5 | 100–1400 | 13–19.5 |
| OH | 15–26 | 100–1400 | 13–19.5 |

`diskmelts.MODEL_SPECS` records the exact domains, PCA sizes, and network
layers. Version 2.3.0 retrains 26 of these checkpoints (13C12CH2, 13CO2,
C2H2 11.5–17 µm, C2H4, C2H6, C2HD, C4H2, CH4, CO17O, CO18O, CO2, HC3N, and HCN at
both sigmas) with lower held-out error; see the
[changelog](https://diskmelts.readthedocs.io/en/latest/changelog.html).
Upgrade with `python -m pip install -U diskmelts`.

## Training and validation data

Generated data under `Model_grids/` and `Pretrain_grid/` are intentionally
ignored by Git because they contain the full local training data. Generated
checkpoint directories under the top-level `Trained_model/` are also optional; the package bundles
only the production checkpoints needed for v2 fitting. To run
`examples/dev_v2_pt_validation.py` or
`notebooks/Example_Training_Validation.ipynb`, place the complete model grids
under `Model_grids/sigma0/<molecule>/`. The workflows create or reuse the corresponding
pretraining CSV under `Pretrain_grid/`.

The maintained v2 workflow uses a hybrid design after restricting the expanded
IRIS grids to the intended physical domain. H2O uses five overlapping
wavelength checkpoints covering 4.9–25.0 µm, all sharing `T=100–1400 K` and
`logN=13–19`. Primary carbon-bearing and rarer molecules use one checkpoint
each and `logN=13–19.5`; C6H6 is limited to `T=100–675 K`, while the others
use `T=100–1400 K`.
The new C2HD checkpoint spans 11.5–20.5 µm to include its 19 µm band;
CO17O and CO18O span 13–18 µm. The updated 13C12CH2 checkpoint keeps
11.5–17 µm. To retrain these four grids, run:

```bash
DISKMELTS_MOLECULES=13C12CH2,C2HD,CO17O,CO18O python examples/dev_v2_pt_validation.py
```

C6H6 and C3H4 also have bundled fitting checkpoints. The C6H6 checkpoint
covers 13–17 µm and 100–675 K; C3H4 covers 15–18 µm and 100–1400 K. Load
either by name with `load_fitting_models`. Direct grid fitting
uses the same names with local `Model_grids/sigma0/C6H6/` and `Model_grids/sigma0/C3H4/`
files. See the [training guide](https://diskmelts.readthedocs.io/en/latest/training.html)
for peak audits and retraining commands.

The package also includes a CH3 checkpoint for 15–18 µm,
`T=100–1400 K`, and `logN=13–19.5`. Its strongest grid feature in this window
is near 16.4–16.6 µm. Use `load_fitting_models('CH3')` after installing this
package; direct-grid fitting uses local `Model_grids/sigma0/CH3/` files.

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
