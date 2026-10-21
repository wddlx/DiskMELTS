# Training and Validation

Training is a separate workflow from fitting observed spectra. The supplied
checkpoints are ready for fitting, while retraining requires the full local slab
model grids.

The production fitting checkpoints are bundled inside `diskmelts/models/`.
Training scripts, generated checkpoints under the repository's top-level
`Trained_model/`, `Model_grids/`, and `Pretrain_grid/` are optional material.
They are not read by `load_fitting_models` or `fit_observation`.
The training notebook loads a bundled checkpoint from any folder after
`pip install diskmelts`. Its `BASE_DIR` defaults to the current directory;
change it to the folder holding your downloaded training script and slab
grids before enabling training. No repository marker search is required.

## Local-only data

The following local directories are intentionally ignored by Git:

```text
Model_grids/
  sigma0/
    <molecule>/
  sigma2/
    <molecule>/
Pretrain_grid/
```

Place each molecule's slab-model CSV files under:

```text
Model_grids/sigma0/<molecule>/
```

The original checkpoints use the `sigma0` grids. Put sigma=2
grids under `Model_grids/sigma2/<molecule>/`. The grid fitter reads
`<grid_root>/<molecule>/`, so pass `Model_grids/sigma0` for current grids or
`Model_grids/sigma2` for complete new grids. Both bundled sigma versions now
cover the same 18 species, including OH.

Train or validate sigma=2 water without overwriting the sigma=0 artifacts:

```bash
DISKMELTS_GRID_SIGMA=2 DISKMELTS_MOLECULES=H2O python examples/dev_v2_pt_validation.py
```

The generated pretraining table is under `Pretrain_grid/sigma2/`, the local
bank under `Trained_model/sigma2/`, and validation results under
`figures/validation/sigma2/`. Bundled checkpoints live in
`src/diskmelts/models/Trained_model/sigma0/` and `sigma2/`.
The separate band trainer also accepts `--sigma 2`; its default remains
`--sigma 0`, and it writes sigma=2 artifacts to separate output folders.

To reproduce the five sigma=2 carbon checkpoints, run:

```bash
DISKMELTS_GRID_SIGMA=2 DISKMELTS_MOLECULES=C2H2,HCN,CO2,13C12CH2,13CO2 python examples/dev_v2_pt_validation.py
python examples/train_band_checkpoint.py C2H2 6 9 --sigma 2
```

The first command trains the 11.5–17 µm band for each molecule. The second
trains the separate C2H2 6–9 µm band. Both use 10% unseen grid points for
holdout checks. The packaged C2H2 manifest combines these two checkpoints
for each sigma version and leaves the 9–11.5 µm gap uncovered.

The additional single molecule sigma=2 checkpoints use the same 10% complete
grid holdout and the maintained wavelength windows. Run the standard group with:

```bash
DISKMELTS_GRID_SIGMA=2 DISKMELTS_MOLECULES=C4H2,HC3N,C2H6,C2H4,CH4,C2HD,CO17O,CO18O DISKMELTS_COMPARE_V1=0 python examples/dev_v2_pt_validation.py
```

C3H4 and CH3 use 15–18 µm, while OH uses 15–26 µm. Their dense bands selected
75 PCA components from candidates through 75. C6H6 retains the 13–17 µm,
100–675 K production domain and selected 30 components:

```bash
python examples/train_band_checkpoint.py C3H4 15 18 --sigma 2 --pca-candidates 10 15 20 25 30 50 75 --validate-retrieval
python examples/train_band_checkpoint.py CH3 15 18 --sigma 2 --pca-candidates 10 15 20 25 30 50 75 --validate-retrieval
python examples/train_band_checkpoint.py OH 15 26 --sigma 2 --pca-candidates 10 15 20 25 30 50 75 --validate-retrieval
python examples/train_band_checkpoint.py C6H6 13 17 --t-min 100 --t-max 675 --sigma 2 --pca-candidates 10 15 20 25 30 --validate-retrieval
```

The first eleven additional models held out 281 of 2,809 in domain spectra.
C6H6 held out 127 of 1,272 cool grid spectra. Median and
maximum full window reconstruction RMSE, relative to each spectrum's peak,
were:

| Molecule | Band (µm) | Checked spectra | Median | Maximum |
| --- | --- | ---: | ---: | ---: |
| C4H2 | 14–18 | 100 | 0.271% | 3.180% |
| HC3N | 13–18 | 100 | 0.248% | 1.462% |
| C2H6 | 9–15 | 100 | 0.132% | 0.416% |
| C2H4 | 8–14 | 100 | 0.121% | 0.567% |
| CH4 | 5–9 | 100 | 0.163% | 0.800% |
| C2HD | 11.5–20.5 | 100 | 0.161% | 0.620% |
| CO17O | 13–18 | 100 | 0.222% | 0.941% |
| CO18O | 13–18 | 100 | 0.160% | 0.975% |
| C3H4 | 15–18 | 281 | 0.275% | 0.513% |
| CH3 | 15–18 | 281 | 0.152% | 0.346% |
| OH | 15–26 | 281 | 0.054% | 0.571% |
| C6H6 | 13–17 | 127 | 0.060% | 0.484% |

All twelve also passed T/logN retrieval checks on 30 unseen spectra and full
T/logN/A fits on ten. For C3H4, CH3, and OH respectively, T retrieval RMSE
was 7.55, 2.06, and 6.32 K; logN RMSE was 0.017, 0.010, and 0.067 dex.
For C6H6, those errors were 5.45 K and 0.025 dex. The band trainer checks
all held out spectra for reconstruction.

Each of the five initial carbon runs selected 2,809 grid spectra in T=100–1400 K and
logN=13–19.5, trained on 2,528, and held out 281. All six checkpoints
selected 30 PCA components. Spectral reconstruction RMSE relative to each
held-out spectrum's peak flux was:

| Molecule | Band (µm) | Checked spectra | Median | Maximum |
| --- | --- | ---: | ---: | ---: |
| HCN | 11.5–17 | 100 | 0.198% | 0.909% |
| C2H2 | 11.5–17 | 100 | 0.220% | 1.362% |
| C2H2 | 6–9 | 281 | 0.089% | 1.516% |
| CO2 | 11.5–17 | 100 | 0.203% | 1.541% |
| 13C12CH2 | 11.5–17 | 100 | 0.187% | 1.545% |
| 13CO2 | 11.5–17 | 100 | 0.215% | 0.756% |

For C2H2 at 6–9 µm, the nonzero grid spectra peak between 7.317 and
7.785 µm; 28 of the 2,809 selected spectra are dark in this window and
are excluded from the peak loss.
In the 11.5–17 µm training windows, the peak wavelengths span 13.94–14.04
µm (HCN), 13.70–13.71 µm (C2H2), 14.94–14.98 µm (CO2), 13.67–13.73 µm
(13C12CH2), and 15.38–15.42 µm (13CO2).

The main-band validation also ran T/logN retrieval on 30 unseen spectra
and full T/logN/A fits on ten per molecule, matching the sigma=0 settings.
Temperature retrieval RMSE was 10.0 K (HCN), 10.3 K (C2H2), 11.0 K
(CO2), 13.8 K (13C12CH2), and 11.8 K (13CO2). The 6–9 µm C2H2 band
has a separate spectral reconstruction check; it was not given a separate
parameter retrieval check. Column density and emitting area can trade off
in a fit, so the reconstruction errors are not parameter uncertainties.

The sigma=2 water bank trained on 2,337 grid points, with 260 complete points
held out. All five wavelength tiles selected 30 PCA components. On 100 held-out
spectra, the assembled bank had 0.0368% median and 0.184% maximum spectral
RMSE relative to each spectrum's peak flux. Per-tile T/logN retrieval used
30 held-out spectra per tile. A separate 10-spectrum full-bank fit check gave
1.85 K temperature and 0.170 dex logN RMSE; logN and emitting area remain
partly degenerate. These are held-out checks, not uncertainty estimates for
observed spectra.

### Sigma=0 and sigma=2 holdout comparison

The two versions withheld the same `(T, logN)` grid keys in each matching
domain. The table compares median peak-relative reconstruction RMSE against
each version's **own** grids, over the same wavelength window. H2O and the
standard single models use 100 held out spectra, C6H6 uses 127, and
C3H4/CH3/OH and the short C2H2 band use 281. Broadened sigma=2
spectra are different physical targets, so these ratios measure surrogate
fidelity rather than whether one broadening is preferable for observations.

| Molecule | Sigma=0 | Sigma=2 |
| --- | ---: | ---: |
| H2O | 0.058% | 0.037% |
| HCN | 0.232% | 0.198% |
| C2H2 (11.5–17 µm) | 0.161% | 0.220% |
| C2H2 (6–9 µm) | 0.108% | 0.089% |
| CO2 | 0.168% | 0.203% |
| 13C12CH2 | 0.157% | 0.187% |
| 13CO2 | 0.165% | 0.215% |
| C4H2 | 0.211% | 0.271% |
| HC3N | 0.213% | 0.248% |
| C2H6 | 0.155% | 0.132% |
| C2H4 | 0.146% | 0.121% |
| CH4 | 0.150% | 0.163% |
| C2HD | 0.173% | 0.161% |
| CO17O | 0.217% | 0.222% |
| CO18O | 0.208% | 0.160% |
| C6H6 | 0.051% | 0.060% |
| C3H4 | 0.292% | 0.275% |
| CH3 | 0.166% | 0.152% |
| OH | 0.068% | 0.054% |

Across the original 17 shared species, sigma=2 is lower for eight and higher
for nine. The additional C2H2 short-band comparison also favors sigma=2.
OH's sigma=0 checkpoint has 0.317% maximum error on 281 holdouts; the
sigma=2 maximum is 0.571%. Thirty-spectrum sigma=0 OH T/logN retrieval RMSE
is 4.10 K/0.049 dex, compared with 6.32 K/0.067 dex for sigma=2. Each version
is compared with its own grid truth. These tests use noiseless grid spectra and do not
measure observational fit bias or parameter uncertainty.

Grid files must use names such as `T500N17.0.csv` and contain `wave` and `Line`
columns. The v2 workflow writes a domain-restricted training table:

```text
Pretrain_grid/pretrain_<molecule>_single_<domain>.csv
```

If `Model_grids/sigma0/<molecule>/` is absent, the example script and notebook stop
early with a message explaining that the local data is missing.

## Hybrid v2 domains

The maintained workflow uses five wavelength checkpoints for H2O and one
checkpoint per carbon-bearing molecule. It uses the expanded v2 grids only
inside these domains:

| Group/species | Wavelength (µm) | T (K) | logN |
|---|---:|---:|---:|
| H2O bank | 4.9–7.4, 7.2–11.2, 9.9–15.1, 13.5–19.2, 19–25 | 100–1400 | 13–19 |
| HCN, C2H2, CO2, 13C12CH2, 13CO2 | 11.5–17.0 | 100–1400 | 13–19.5 |
| C4H2 | 14–18 | 100–1400 | 13–19.5 |
| HC3N | 13–18 | 100–1400 | 13–19.5 |
| C2H6 | 9–15 | 100–1400 | 13–19.5 |
| C2H4 | 8–14 | 100–1400 | 13–19.5 |
| CH4 | 5–9 | 100–1400 | 13–19.5 |
| C2HD | 11.5–20.5 | 100–1400 | 13–19.5 |
| CO17O, CO18O | 13–18 | 100–1400 | 13–19.5 |
| C6H6 | 13–17 | 100–675 | 13–19.5 |
| C3H4 | 15–18 | 100–1400 | 13–19.5 |
| CH3 | 15–18 | 100–1400 | 13–19.5 |

## CH3 checkpoint

The refreshed CH3 grid has signal in every one of its 2,809 spectra within
the trained domain. Peaks in the 15–18 µm window lie at 16.396–16.596 µm.
Training used 2,528 spectra and held out 281 complete grid points. The
held-out PCA comparison selected 75 components. The trained checkpoint's
relative spectral RMSE on those grid points was 0.166% median and 0.376%
maximum, measured against each spectrum's peak flux. Reproduce the training
from local grids with:

```bash
python examples/train_band_checkpoint.py CH3 15 18 --pca-candidates 10 15 20 25 30 50 75
```

Version 2.2.1 bundles the CH3 checkpoint for model fitting.
Direct grid fitting uses the local `Model_grids/sigma0/CH3/` files.

## C6H6 and C3H4 checkpoints

Across the full C6H6 grids, strongest peaks span 13.42–14.85 µm. The cool
100–675 K subset peaks at 14.84–14.85 µm; its checkpoint retains the
13–17 µm window used for the earlier comparison. C3H4 peaks at
15.71–15.81 µm within its 15–18 µm window. C6H6 uses 1,145 training spectra
and 127 separate holdouts
with 30 PCA components; C3H4 uses 2,528 training spectra and 281 holdouts
with 75 components. Retrain from local grids with:

```bash
python examples/train_band_checkpoint.py C6H6 13 17 --t-min 100 --t-max 675 --pca-candidates 10 15 20 25 30
python examples/train_band_checkpoint.py C3H4 15 18 --pca-candidates 10 15 20 25 30 50 75
```

Median reconstruction error relative to the grid peak is 0.051% for C6H6
(maximum 0.263%) and 0.29% for C3H4 (maximum 0.68%). A C6H6 trial limited
to 100–600 K had 0.060% median error. Extending it to 800 K introduced a
sharp spectral change at 700 K: holdouts at 700–800 K had 7.45% median error
and 12.02% maximum error. Use direct grid fitting for hotter C6H6 rather than
extrapolating the bundled checkpoint. These numbers measure spectra, not
uncertainty in fitted parameters.

The sigma=2 C6H6 checkpoint uses the same 127 withheld `(T, logN)` keys and
cool domain as sigma=0. Its median/maximum reconstruction error is
0.060%/0.484%, compared with 0.051%/0.263% for sigma=0. On 30 matching
unseen grid keys, T RMSE is 5.45 K versus 5.52 K and logN RMSE is 0.025
versus 0.062 dex for sigma=2 and sigma=0 respectively. The sigma=2 grid
also changes sharply between 675 and 700 K, so the same upper temperature
limit applies.

Use `--audit-only` to inspect peak positions before training, or `--tune-only`
to also compare PCA sizes without starting neural-network training. The
checkpoints under `src/diskmelts/models/Trained_model/sigma0/` are bundled for model
fitting. Direct grid fitting uses separate local `Model_grids/sigma0/C6H6/` and
`Model_grids/sigma0/C3H4/` directories. In the supplied C6H6 grids, some high-column
files had nonfinite flux; those entries must be replaced or regenerated before
fitting those column densities.

## Separate C2H2 6–9 µm checkpoint

Use the separate band trainer to reproduce the 6–9 µm checkpoint bundled
alongside the 11.5–17 µm C2H2 checkpoint:

```bash
python examples/train_band_checkpoint.py C2H2 6 9
```

Pass `--audit-only` to inspect signal counts and peak wavelengths before
training. In the local grids used for this run, C2H2's 6–9 µm peaks lie at
7.32–7.81 µm. The trainer stops if every selected spectrum is zero. This run
uses T=100–1400 K and logN=13–19.5, keeps 10% of complete grid spectra out of
training, and reports
reconstruction errors on that holdout.

New checkpoints are written under `Trained_model/<molecule>_w<range>_v2/`
for sigma=0 or `Trained_model/sigma2/<molecule>_w<range>_v2/` for sigma=2,
with training tables in `Pretrain_grid/` and validation summaries under
`figures/validation/`. Those directories are local and ignored by Git; a new
training run must be copied into `src/diskmelts/models/Trained_model/` to be
included in a future pip package build.

The new C2HD grid has its strongest sampled band around 14.5 µm and a
substantial second band near 19 µm, particularly in cooler models. Its
temperature, column-density, network, and holdout settings follow 13C12CH2,
while the wavelength window includes both bands. Sampled CO17O and CO18O
grids peak near 14.5–15.5 µm. Their 13–18 µm windows retain the weaker
high-temperature wings beyond the initial 13.5–17.5 µm estimate.

The four grids were trained with 2,528 spectra and validated on 281 excluded
spectra each. A 30-component PCA was selected for every checkpoint. On the
sampled noiseless holdouts, the median spectral reconstruction error relative
to the model peak was 0.16% (13C12CH2), 0.17% (C2HD), 0.22% (CO17O), and
0.21% (CO18O). Full fits can exchange column density against emitting area;
these reconstruction values are not observational parameter uncertainties.

No spectra outside the listed T/logN domains enter the pretraining CSV, PCA,
scalers, or neural networks. All five H2O wavelength models use the same
complete-grid holdout and represent one physical H2O component with shared
fitted `(T, logN, A)` parameters. Archived 45-tile banks remain loadable, but
the maintained H2O bank no longer tiles T or logN.

## Holdout and PCA selection

The script reproducibly discards 10% of complete grid models by default
(`DISKMELTS_HOLDOUT_FRACTION`; allowed range 0.05–0.10). Boundary grid rows are
kept in training so the checkpoint retains the full interpolation domain.
Holdouts never enter PCA or MLP training.

Candidate PCA dimensions are evaluated by reconstructing unseen spectra. The
smallest dimension within 5% of the best holdout error is selected. H2O is
tuned independently for each wavelength checkpoint; carbon/rarer species are
tuned once for their single checkpoint.

New checkpoints store the selected PCA object, input/output scalers, wavelength
axis, molecule name, physical ranges, PCA count, and a fingerprint of the input
grid. If the grid values change, the script rebuilds the pretraining table and
rejects reuse of a checkpoint made from the old grid. Choose a new checkpoint
path for retraining after such a change.
Checkpoints can be loaded later with only `model_paths`.

## Water normalization and optimization

The corrected water workflow writes a separate bank under
`Trained_model/H2O_wavelength_bank_v2_fixed/`, preserving the previous bank.
It uses the same five windows and complete-grid holdout, with:

- `peak_normalization='window'`: normalize each cropped spectrum by its own
  peak and adjust the log-peak target so `shape * peak` preserves physical flux.
  Fully zero windows still train the shape head but do not train the peak head:
  they have no defined log-peak. Some supplied cold-water grids contain such
  windows; this treatment does not regenerate or repair those input grids.
- `shape_loss='spectral'`: weight PCA coefficient errors by their variance,
  making the loss proportional to reconstructed spectral MSE rather than
  giving every standardized coefficient equal weight.
- `temperature_transform='log10'`: standardize log-temperature internally.
  Public fitting and spectrum-generation functions still accept kelvin.
- `scheduler_patience=100`, `min_lr=1e-6`: allow continued learning before
  reducing the learning rate. Checkpoints record these settings and the seed.

NT validation now compares local normalized shapes and local physical peaks
for both old and new checkpoints. Optimizer starts are reproducible. Older
window-only validation figures made with inconsistent normalization must not
be used to judge the water models. Full retrieval uses scaled parameter and
loss coordinates, finite differences resolvable by float32 networks, and a
derivative-free final refinement. Samples are drawn without replacement when
the requested count fits within the validation set.

Run a matched old/new comparison with:

```bash
DISKMELTS_V2_MODE=water_compare python examples/dev_v2_pt_validation.py
```

It saves per-spectrum results and summaries under
`figures/validation/water_bank_comparison/`. These noiseless grid tests assess
the surrogate and optimizer; observational continuum, blending, and multiple
temperature components require separate checks.

An edge-weighted H2O seam diagnostic was tested for the 19–25 µm tile:

```bash
DISKMELTS_V2_MODE=water_seam_trial python examples/dev_v2_pt_validation.py
```

It emphasizes the shared 19.0–19.2 µm interval and writes a separate bank under
`Trained_model/H2O_wavelength_bank_v2_seam_trial/`. On the 100-spectrum
reconstruction check it worsened median error from 0.058% to 2.03% of peak.
The default fit workflow continues to use the sigma=0
`H2O_wavelength_bank_v2_fixed`; pass `sigma=2` to select the new water bank.

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
`run_rarer` selects `C4H2`, `HC3N`, `C2H6`, `C2H4`, and `CH4`.
For an isolated run without editing the file, set a comma-separated override,
for example `DISKMELTS_MOLECULES=HCN,C2H2`.
To train the updated 13C12CH2 grid and the three new molecules together:

```bash
DISKMELTS_MOLECULES=13C12CH2,C2HD,CO17O,CO18O python examples/dev_v2_pt_validation.py
```

The updated 13C12CH2 output uses the local `_single_v2_newgrid` directory so
the previous local checkpoint remains available for comparison. Production
checkpoints are copied into `src/diskmelts/models/Trained_model/sigma0/` or
`src/diskmelts/models/Trained_model/sigma2/` only after
holdout validation.
The default `DISKMELTS_V2_MODE=train` runs this molecule workflow. The
`water_compare` and `water_seam_trial` modes above run separately and ignore
the molecule switches.

For each selected molecule, the script performs:

1. model-grid loading
2. T/logN/wavelength restriction
3. reproducible 5–10% complete-model holdout
4. PCA selection on unseen spectral reconstruction
5. training/loading five H2O wavelength checkpoints or one carbon checkpoint
6. unseen T/logN retrieval, full T/logN/A fitting, and reconstruction plots

The default settings are intended for scientific runs and may take substantial
time. Set `DISKMELTS_VALIDATE=0` to train without retrieval validation.
Diagnostic controls include `DISKMELTS_VAL_N`, `DISKMELTS_VAL_NT_STARTS`,
`DISKMELTS_VAL_NT_STEPS`, `DISKMELTS_VAL_FULL_N`, and `DISKMELTS_RECON_N`.

## Notebook

The optional notebook shows how to launch the maintained v2 training script
for one molecule, verify local grid availability, and inspect its checkpoint:

```text
notebooks/Example_Training_Validation.ipynb
```

The notebook starts with training disabled and `GRID_SIGMA=0`. Set
`GRID_SIGMA=2` for the new grid set and `RUN_TRAINING=True` after placing
the complete grids under the matching `Model_grids/sigma*/` directory. Use
`examples/dev_v2_pt_validation.py` for the authoritative domains, holdout,
PCA selection, and validation settings.

## Loading a trained checkpoint

Normal v2 fitting loads the package's checkpoints without a path:

```python
from diskmelts import load_fitting_models
pretrained = load_fitting_models(['H2O', 'C2H2'])
# Select separately trained sigma=2 checkpoints:
broadened = load_fitting_models(['H2O', 'C2H2', 'C3H4', 'OH'], sigma=2)
```

To use a newly trained checkpoint before bundling it, pass an alternate root
containing the `Trained_model/` directory, or use the lower-level `load_models`
API with explicit paths. Self-contained checkpoints do not need the pretraining
CSV:

```python
from diskmelts import load_models

pretrained = load_models(
    model_paths={
        'H2O': 'Trained_model/H2O_wavelength_bank_v2_fixed/H2O_model_bank.json',
    },
)
```

The optional `pretrain_csv_paths`, `wav_ranges`, and `n_pca` arguments to
`load_models` remain only for loading older checkpoints that do not contain
their own scaler and wavelength metadata. Archived bank manifests can still be
passed to `load_models` when reproducing earlier segmented-v2 results.
