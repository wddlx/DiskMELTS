# DiskMELTS v2 Plan

This plan records the v2 direction established on 2026-08-01. It separates completed implementation from HPC generation/training and scientific validation still requiring review.

## 1. Expanded IRIS model grids

Status: generator implemented; HPC production/inspection remains an external run.

- Use `Generating-grids.py` from the HPC Jupyter working directory without changing directories.
- Generate molecules sequentially into `<molecule>/` directories.
- Molecule set: `H2O`, `C2H2`, `HCN`, `CO2`, `13CO2`, `13C12CH2`, `C4H2`, `CH4`, `C2H6`, `HC3N`, `OH`, `CH3`, and `C2H4`.
- Grid coverage: `logN=13–24` in steps of 0.125 (89 values) and `T=50–2000 K` in steps of 25 K (79 values).
- Spectral coverage: 4.9–27.0 um using 12 non-overlapping MIRI/MRS-derived segments with segment-specific resolving power.
- Thermal broadening uses molecule-specific monoisotopic molecular masses.
- Before training a new species, verify file count, wavelength ordering, `wave`/`Line` columns, finite flux, and representative low/mid/high T and logN spectra.

## 2. Full pretraining tables

Status: implemented for generic single molecules; generated tables are local/Git-ignored.

- Build one complete `Pretrain_grid/pretrain_<mol>_full.csv` per molecule from its full model grid.
- Keep `A=1`; store the full peak-normalized shape and `log10(peak flux)` targets.
- Train all segmented checkpoints from filtered portions of this one table rather than regenerating data per tile.
- Existing pretraining CSVs are reused. Regenerate intentionally when the underlying IRIS grid or wavelength sampling changes.

## 3. H2O segmented bank

Status: 45 checkpoints and manifest currently present; assembled-bank reconstruction plot still needs completion/review.

- Wavelength tiles: 4.9–7.4, 7.2–11.2, 9.9–15.1, 13.5–19.2, and 19.0–25.0 um.
- `logN` tiles: 13–19, 18.5–22, and 21.5–24.
- Temperature tiles: 50–700, 600–1300, and 1200–2000 K.
- Cartesian product: 45 checkpoints, PCA=21, original MLP architecture and hyperparameters.
- Run/review withheld tile retrieval validation and the 100-spectrum assembled-bank difference plot.
- Investigate any discontinuities or high residual bands before observational fitting.

## 4. Initial carbon-bearing banks

Status: implemented and current artifacts present for all five molecules.

- Molecules: `HCN`, `C2H2`, `CO2`, `13C12CH2`, and `13CO2`.
- Wavelength tile: 11.5–17.0 um.
- `logN` tile: 13–19.5.
- Temperature tiles: 200–800, 600–1500, and 1200–2000 K.
- Three checkpoints per molecule, PCA=15, original MLP architecture and hyperparameters.
- Review the three tile-validation plots and 100-spectrum assembled-bank reconstruction plot for every molecule.

## 5. Rarer carbon-bearing and additional molecules

Status: planned; `run_rarer` currently selects no models.

- First candidates include `C4H2`, `HC3N`, and `C2H6`; grid generation also supports `CH4`, `OH`, `CH3`, and `C2H4`.
- Reuse carbon `logN=13–19.5` and temperature tiles `200–800`, `600–1500`, and `1200–2000 K` unless grid inspection motivates a change.
- Define molecule-specific wavelength ranges before adding each molecule to `RARER_MOLECULES` and `MODEL_CONFIGS` in `examples/dev_v2_pt_validation.py`.
- Add plotting labels/colors, detection windows, documentation, and tests when a molecule becomes supported by fitting.

## 6. Training and validation controls

Status: implemented.

- `run_H2O` controls H2O.
- `run_Cmol` controls all five current carbon banks.
- `run_rarer` is reserved for later configured rarer species.
- `DISKMELTS_VALIDATE=0` trains only; validation runs reuse existing checkpoints.
- Tile validation uses each checkpoint's withheld 10% split.
- Assembled-bank validation defaults to 100 grid spectra and saves `(model-grid)/peak` residual curves plus median and 16–84% summaries.
- Short diagnostic checkpoints are reused automatically; move them out of the production model directory before full training.

## 7. V2 fitting integration

Status: model-bank loading and blending implemented; observational configuration and science validation remain.

- Load one bank through its JSON manifest with `load_models`, or use `load_model_bank` filters for narrow wavelength/T/logN fits.
- Treat all tiles in a bank as one physical molecule with one shared `(T, logN,A)` parameter set.
- Blend overlapping tile predictions; never fit tiles as independent molecular components.
- Use wavelength restrictions, bounds, and physically motivated priors to avoid loading/evaluating irrelevant tiles.
- Update `examples/dev_v2_realobs.py` stages to use v2 manifests after bank validation is accepted.
- Establish the joint H2O subtraction and `HCN`/`C2H2`/`CO2`/isotopologue fitting order and detection windows.
- Validate recovery on synthetic mixtures before trusting real-observation parameters.

## 8. Release and documentation work

Status: partially implemented.

- Migrate `notebooks/Example_Training_Validation.ipynb` from the legacy single-checkpoint workflow to model banks.
- Update fitting notebook examples to load manifests and demonstrate filtered banks/priors.
- Decide whether the public release version should become 2.0.0; `pyproject.toml` currently remains 0.1.0.
- Decide which v2 manifests/checkpoints and validation plots should be committed versus distributed externally.
- Run the complete test suite without mutating tracked scientific results, then update README/docs and prepare release notes.

## Immediate next steps

1. Complete and inspect the missing H2O 100-spectrum assembled-bank reconstruction plot.
2. Review carbon tile and assembled-bank validation plots for wavelength-localized errors.
3. Add wavelength configurations for the first `run_rarer` species.
4. Update real-observation fitting stages to use validated manifests and explicit physical priors/bounds.
5. Run synthetic blended-spectrum retrieval tests across tile overlap boundaries.
