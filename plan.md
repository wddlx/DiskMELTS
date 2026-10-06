# DiskMELTS v2 Plan

This plan records the v2 direction established on 2026-08-01 and updated on 2026-10-05. It separates completed implementation and training from scientific validation still requiring review.

## 1. Expanded IRIS model grids

Status: generator implemented; HPC production/inspection remains an external run.

- Use `Generating-grids.py` from the HPC Jupyter working directory without changing directories.
- Generate molecules sequentially into `<molecule>/` directories.
- Molecule set: `H2O`, `C2H2`, `HCN`, `CO2`, `13CO2`, `13C12CH2`, `C4H2`, `CH4`, `C2H6`, `HC3N`, `OH`, `CH3`, and `C2H4`.
- Grid coverage: `logN=13–24` in steps of 0.125 (89 values) and `T=50–2000 K` in steps of 25 K (79 values).
- Spectral coverage: 4.9–27.0 um using 12 non-overlapping MIRI/MRS-derived segments with segment-specific resolving power.
- Thermal broadening uses molecule-specific monoisotopic molecular masses.
- Before training a new species, verify file count, wavelength ordering, `wave`/`Line` columns, finite flux, and representative low/mid/high T and logN spectra.

## 2. Hybrid pretraining on restricted v2 grids

Status: workflow implemented; corrected H2O production bank and all ten carbon checkpoints trained. Matched H2O holdout checks are complete; observational and mixed-species validation remain.

- Train five wavelength checkpoints for H2O and one checkpoint per carbon-bearing molecule.
- H2O wavelength ranges are 4.9–7.4, 7.2–11.2, 9.9–15.1, 13.5–19.2, and 19–25.0 um. All share `T=100–1400 K`, `logN=13–19`, one holdout set, and one fitted `(T,logN,A)` parameter set.
- `HCN`, `C2H2`, `CO2`, `13C12CH2`, and `13CO2` use 11.5–17.0 um, `T=100–1400 K`, and `logN=13–19.5`.
- Grid models outside the configured T/logN domain are removed before CSV, PCA, scaler, or MLP fitting.
- The maintained H2O bank remains wavelength-only with one shared physical `(T, logN, A)`. The corrected checkpoints use local-window peak normalization, spectral PCA loss, log-temperature inputs, and omit empty windows from the undefined peak loss. The original bank is preserved.

## 3. Rarer carbon-bearing molecules

Status: all five single-checkpoint models trained and validated on complete-grid holdouts on 2026-10-05. A subsequent C4H2 audit traced its inflated RMSE to a retrieval outlier; fixed optimizer starts give 0.123 dex on the same 30 holdouts without retraining.

- All use `T=100–1400 K`, `logN=13–19.5`, and the same PCA candidate set as the primary carbon molecules.
- Wavelengths: `C4H2=14–18`, `HC3N=13–18`, `C2H6=9–15`, `C2H4=8–14`, and `CH4=5–9` um.
- `run_rarer` selects all five configured molecules.

## 4. Training and validation controls

Status: implemented in code; production holdout summaries are available under `figures/validation/`.

- `run_H2O` controls H2O.
- `run_Cmol` controls all five current carbon banks.
- `run_rarer` controls all five configured rarer species.
- `DISKMELTS_VALIDATE=0` trains only; validation runs reuse existing checkpoints.
- Reproducibly withhold 5–10% of complete grid models before pretraining; preserve domain boundaries in the training set.
- Tune PCA independently for every H2O wavelength range and once per carbon/rarer model using unseen-spectrum reconstruction.
- Validate unseen T/logN retrieval, full T/logN/A fitting, and wavelength-dependent reconstruction; save a JSON metric summary per molecule. Matched corrected H2O and legacy comparisons are under `figures/validation/water_bank_comparison/`.
- Checkpoint metadata are verified before reuse so stale physical ranges or PCA settings cannot be loaded silently.

## 5. V2 fitting integration

Status: hybrid model paths and physical fitting bounds implemented; the H2O production bank passed the selected v1 tolerance check, but observational and synthetic-mixture validation remain.

- Load the H2O JSON bank manifest and one `.pt` checkpoint per carbon/rarer molecule with `load_models`.
- Use `T=100–1400 K`; use `logN=13–19` for H2O and `13–19.5` for carbon/rarer molecules.
- `examples/dev_v2_realobs.py` prefers the new H2O wavelength-bank manifest and single-v2 carbon checkpoints, with v1 fallbacks if artifacts are absent.
- Establish the joint H2O subtraction and `HCN`/`C2H2`/`CO2`/isotopologue fitting order and detection windows.
- Require holdout and synthetic-mixture recovery comparable to or better than v1 before trusting real-observation parameters.

## 6. Release and documentation work

Status: partially implemented.

- Align `notebooks/Example_Training_Validation.ipynb` with the new domain restriction, complete-model holdout, and PCA tuning workflow.
- Update fitting notebook examples to prefer the H2O wavelength bank and new single-v2 carbon checkpoints.
- Decide whether the public release version should become 2.0.0; `pyproject.toml` currently remains 0.1.0.
- Decide which v2 manifests/checkpoints and validation plots should be committed versus distributed externally.
- Run the complete test suite without mutating tracked scientific results, then update README/docs and prepare release notes.

## Immediate next steps

1. Completed H2O normalization-consistent validation, local-window retraining, spectral loss, empty-window peak masking, log-temperature inputs, and scaled local optimization. On the same 30 holdouts the five window T RMSEs improved from `(11.55,25.44,18.56,18.03,25.32)` K to `(2.12,5.99,3.50,2.21,3.03)` K; logN RMSE improved from `(.325,.456,.091,.109,.072)` to `(.019,.023,.067,.020,.013)` dex. The ten-spectrum full fit is 1.32 K / 0.0048 dex in logNA; median/max full-grid reconstruction is 0.058%/0.166% of peak. The common-range v1 test passes the 10% gate by a wide margin.
2. The 19.0–19.2 µm overlap remains locally inaccurate. Edge-weighted retraining worsened reconstruction and was rejected; explore a targeted architecture/PCA or tile design before synthetic-mixture and observational validation.
3. Refresh the primary carbon v1 comparison summaries under the 10% tolerance gate and tune any failures.
4. Establish a staged molecular fitting order and detection windows, then test synthetic mixtures before observational inference.
5. Run staged fits on the observational validation sample using accepted checkpoints.
