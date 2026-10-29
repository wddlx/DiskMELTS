# Changelog

## 2.3.0

### Retrained checkpoints

- 26 of the 46 bundled checkpoints were retrained with the shared
  {ref}`training recipe <training-recipe>`: 13C12CH2, 13CO2, C2H2
  (11.5–17 µm), C2H4, C2H6, C2HD, C4H2, CH4, CO17O, CO18O, CO2, HC3N, and HCN,
  at both sigma=0 and sigma=2. Domains, network sizes, and PCA dimensions are
  unchanged, so existing scripts run as before, but fitted values can
  change. In the J16120505 example notebook, the carbon-stage best fit moved
  from C2H2/HCN/13C12CH2 T = 185/325/162 K to 265/368/220 K, closer to a
  direct slab-grid fit of the same spectrum (235/387/198 K). The C2H2 column
  moved from logN 17.62 to 16.82 (grid: 16.95). Fits of comparable quality
  can span a wide range of T and logN, so small surrogate errors can move the
  best fit. Refit important sources after upgrading.
- On held-out grid spectra, median shape error fell 3–8x and temperature
  retrieval RMSE typically fell from 10–20 K to 2–4 K. See
  {ref}`bundled-checkpoints-2-3-0`.
- `examples/dev_v2_pt_validation.py` now trains single-molecule checkpoints
  with the same recipe.

### Error bars

- Fit uncertainties come from emcee posterior sampling (new
  {doc}`api/uncertainty` module; `emcee>=3` is now a dependency). Reported
  values are posterior medians with 16th/84th percentiles. Saved results also
  include the best fit (`*_best`), percentiles (`*_q16/_q50/_q84`), a bound
  flag (`*_at_bound`), the likelihood noise, and split R-hat.
- Runs are extended up to three times until R-hat < 1.1; a warning is raised
  if they still have not converged. Set the default length with `n_mcmc` or
  the `DISKMELTS_N_MCMC` environment variable.
- `errorbars=False` skips sampling in `fit_molecules`, `fit_stage`,
  `fit_observation`, `fit_grid_stage`, and `fit_observation_grids`.
- Direct-grid fitting uses the same sampler with batched grid interpolation.
- A sampled `logN + log10(A)` summary (`logNA*` columns) replaces the
  quadrature sum used in literature comparisons.

### API

- `fit_observation_grids` and `plot_saved_observation` now accept the
  spectrum as `obs_wav`/`obs_flux` arrays (and `obs_err` for the grid fit),
  like `fit_observation`. Existing positional CSV calls are unchanged; the
  notebook no longer writes a temporary CSV to plot array input.
- `fit_observation` and `fit_stage` have rewritten docstrings, and the API
  page compares them side by side.
- `grid_sha256` is exported and documented.
- Deprecated, still working with a `FutureWarning`: `fit_nested` (now a thin
  wrapper around `fit_molecules`), `validate_nt`, and `plot_validation_split`.
  `validate_full` now uses `fit_molecules`. See {doc}`api/deprecated`.

### Fixes

- Chains that never move no longer report R-hat = 1.
- `Tshare='less'` now gives a flat prior in physical temperature for both
  surrogate and grid fits.
- Unbounded emitting areas are sampled as log10(A) over at least [-6, 6],
  widened to include the best fit ±4 dex, instead of a range centred on the
  best fit.
- A `Tshare='less'` fit whose main temperature lands exactly on the rare
  species' lower bound no longer fails with a non-finite posterior.
- If detection screening removes a group's main isotopologue, that group's
  ties are dropped instead of being re-anchored on a rare species.
- `shape_loss='spectral_edge'` applies PCA coefficient scales once.
- `read_stage_result` returns best-fit values and skips missing percentiles.

## 2.2.2

- Checkpoints store plain arrays instead of scikit-learn objects;
  scikit-learn is needed only for training. Python 3.9 or newer.

## 2.2.1

- `synthesize_spectrum` generates one molecule's spectrum in two lines.
- `fit_observation` accepts wavelength, flux, and optional error arrays.
- Sigma=0 OH and C2H2 6–9 µm checkpoints, giving both sigma values the same
  18 species.

## 2.2.0

- Sigma=2 checkpoints for 18 species, selected with `sigma=2`.
