# DiskMELTS

DiskMELTS (Machine-learning Enabled Line fitting Tool for Spectra) is a
package for continuum subtraction and fitting molecular emission in JWST
mid-infrared spectra of protoplanetary disks. Its fitting workflow loads
pretrained molecular surrogate models and retrieves temperature (`T`), column
density (`logN`), and emitting area scaling (`A`) from observed spectra.

**Documentation:** [diskmelts.readthedocs.io](https://diskmelts.readthedocs.io/en/latest/)

## Installation

DiskMELTS is available on [PyPI](https://pypi.org/project/diskmelts/).
The C6H6 and C3H4 checkpoints are included starting with version 2.1.2.
The new CH3 checkpoint is available in this source checkout; it has not yet
been published to PyPI.
With Python 3.9 or newer, install it from any directory. These commands use
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

The private `0-Consub/` experiment directory is excluded from Git and is not
required by the installed package. Its runtime code and reference tables are
maintained under `src/diskmelts/consub/`.
The distributed tests in `tests/` exercise the installed continuum API and
bundled references without that private directory.

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

For a complete staged fit-and-subtract workflow from a checkout, edit and run:

```bash
python examples/dev_v2_realobs.py
```

The v2 fitting API is `load_fitting_models`, `fit_stage`, `fit_observation`, and
`plot_saved_observation`. The production checkpoints are packaged under
`src/diskmelts/models/` and included in the wheel. Model fitting requires no
`Model_grids/` or `Pretrain_grid/` directory. `diskmelts.MODEL_SPECS` records
PCA counts, network layers, and `grid_sigma=0` for every current checkpoint.
The checkpoint settings are checked when
loading. The checkpoint flux convention is 1 au² at 140 pc; `distance_pc`
sets the source distance and makes the returned emitting area physical.
The [fitting notebook](https://github.com/wddlx/DiskMELTS/blob/main/notebooks/Example_Fitting.ipynb) requires 13C12CH2 to
be no hotter than C2H2 and gives them the same emitting area. It ties 13CO2
to CO2 in both temperature and area. Set
`isotopologue_groups`, `Tshare` (`free`, `equal`, or `less`), and `Ashare` in a
joint stage to choose those constraints. See the
[fitting guide](https://diskmelts.readthedocs.io/en/latest/quickstart.html) for examples.

For an optional fit directly from slab grids, use `fit_grid_stage` or
`fit_observation_grids` and supply `grid_root='Model_grids/sigma0'` explicitly. This
separate method does require the full local grids. See the
[quick start](https://diskmelts.readthedocs.io/en/latest/quickstart.html) for both workflows and saved-result
plotting.

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

The source checkout also includes a CH3 checkpoint for 15–18 µm,
`T=100–1400 K`, and `logN=13–19.5`. Its strongest grid feature in this window
is near 16.4–16.6 µm. Use `load_fitting_models('CH3')` after installing this
checkout; direct-grid fitting uses local `Model_grids/sigma0/CH3/` files.

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
