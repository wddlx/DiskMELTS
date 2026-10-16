# Training and Validation

Training is a separate workflow from fitting observed spectra. The supplied
checkpoints are ready for fitting, while retraining requires the full local slab
model grids.

The production fitting checkpoints are bundled inside `diskmelts/models/`.
Training scripts, generated checkpoints under the repository's top-level
`Trained_model/`, `Model_grids/`, and `Pretrain_grid/` are optional material.
They are not read by `load_fitting_models` or `fit_observation`.

## Local-only data

The following local directories are intentionally ignored by Git:

```text
Model_grids/
Pretrain_grid/
```

Place each molecule's slab-model CSV files under:

```text
Model_grids/<molecule>/
```

If you have the local `Generating-Grids.py` and `generate_grids_sig2.slurm`
scripts, submit the SLURM script from `Model_grids/` to generate 2.0 km/s
grids under `Model_grids/<molecule>_sig2/`. Those scripts and generated grids
are local files and are not included in Git or the pip package. The
direct-grid fitter looks for `<grid_root>/<molecule>/`, so choose a grid root
with that layout when fitting the new grids.

Grid files must use names such as `T500N17.0.csv` and contain `wave` and `Line`
columns. The v2 workflow writes a domain-restricted training table:

```text
Pretrain_grid/pretrain_<molecule>_single_<domain>.csv
```

If `Model_grids/<molecule>/` is absent, the example script and notebook stop
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

Use `--audit-only` to inspect peak positions before training, or `--tune-only`
to also compare PCA sizes without starting neural-network training. The
checkpoints under `src/diskmelts/models/Trained_model/` are bundled for model
fitting. Direct grid fitting uses separate local `Model_grids/C6H6/` and
`Model_grids/C3H4/` directories. In the supplied C6H6 grids, some high-column
files had nonfinite flux; those entries must be replaced or regenerated before
fitting those column densities.

## Separate C2H2 6–9 µm checkpoint

Use the separate band trainer to make a 6–9 µm checkpoint without replacing
the production C2H2 model:

```bash
python examples/train_band_checkpoint.py C2H2 6 9
```

Pass `--audit-only` to inspect signal counts and peak wavelengths before
training. In the local grids used for this run, C2H2's 6–9 µm peaks lie at
7.32–7.81 µm. The trainer stops if every selected spectrum is zero. This run
uses T=100–1400 K and logN=13–19.5, keeps 10% of complete grid spectra out of
training, and reports
reconstruction errors on that holdout.

New checkpoints are written under `Trained_model/<molecule>_w<range>_v2/`,
with training tables in `Pretrain_grid/` and validation summaries under
`figures/validation/`. Those directories are local and ignored by Git; these
checkpoints are not automatically included in the pip package.

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
The fit workflow continues to use `H2O_wavelength_bank_v2_fixed`.

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
checkpoints are copied into `src/diskmelts/models/Trained_model/` only after
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

The notebook starts with training disabled. Set `RUN_TRAINING=True` after
placing the complete grids under `Model_grids/`. Use
`examples/dev_v2_pt_validation.py` for the authoritative domains, holdout,
PCA selection, and validation settings.

## Loading a trained checkpoint

Normal v2 fitting loads the package's checkpoints without a path:

```python
from diskmelts import load_fitting_models
pretrained = load_fitting_models(['H2O', 'C2H2'])
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
