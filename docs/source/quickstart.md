# Your first spectrum

Install with `python -m pip install diskmelts`. You can fit your own
continuum-subtracted spectrum without a repository checkout. For a sample,
download the
[J16120505 continuum-subtracted CSV](https://github.com/wddlx/DiskMELTS/blob/main/Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv)
as `j16120505_contsub.csv`, or use:

```bash
curl -L -o j16120505_contsub.csv https://raw.githubusercontent.com/wddlx/DiskMELTS/main/Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv
```

The CSV reader uses the first two columns for wavelength (µm) and flux (Jy)
and skips one header row. The sample's extra `err` and `cont` columns are fine.
You can use a spectrum anywhere on your computer; no repository layout or
`pyproject.toml` search is needed. For example,
`Path('/path/to/my_spectrum.csv')` or `Path('my_spectrum.csv')` works when
the file exists at that location.

## Step 1: prepare the spectrum

Start with the {doc}`continuum subtraction guide <continuum>` or
{doc}`Example_Consub <notebooks/Example_Consub>` when your spectrum still
contains its continuum. `consub.read_spectrum` reads a CSV whose header
names the columns `wave`, `flux`, and optionally `err` (wavelength in µm),
or a JWST FITS spectrum with `wavelength`, `fluxdensity`, and
`fluxdensity_stddev` columns (install the `io` extra). Correct the spectrum to
the source rest frame first: the continuum windows and molecular models use
rest wavelengths (the sample `_RVcorr` file is already corrected).

```python
from diskmelts import consub

raw_wave, raw_flux, raw_error = consub.read_spectrum('my_raw_spectrum.csv')
wave, flux, error, continuum, baseline = consub.calculate(
    raw_wave, raw_flux, raw_error, preset='ttauri',
)
```

The resulting `wave` and continuum-subtracted `flux` can be passed directly
to `fit_stage`. The returned `error` is a measurement/noise estimate; it does
not include correlated continuum uncertainty.

If your spectrum is already continuum-subtracted, load it instead.
`load_observed_spectrum` ignores column names: it skips one header row and
reads the first two columns as wavelength (µm) and flux (Jy). This line uses
the downloaded sample; change the path for your own CSV:

```python
from diskmelts import load_observed_spectrum
wave, flux = load_observed_spectrum('j16120505_contsub.csv')
```

## Step 2: fit one molecule

The installed package includes the production v2 checkpoints. Fitting needs
no local model grids or training tables. Run this after either Step 1 path.
`distance_pc` defaults to 140; the sample source J16120505 is at 122.5 pc,
so give your own source's distance in parsecs:

```python
from diskmelts import load_fitting_models, fit_stage

models = load_fitting_models('C2H2')
fit = fit_stage(
    wave, flux,
    molecule='C2H2',
    wavelength_range=(13.0, 15.0),
    fit_ranges=(13.5, 14.1),
    pretrained=models,
    distance_pc=122.5,
    n_samples=1024, n_refine=4, n_top=4,
)
print(fit['params']['C2H2'])
```

`fit['params']['C2H2']` holds `T` (K), `logN` (log10 column density in
cm⁻²), `A` (emitting area in au²), and `log10A`. The fit samples `(T, logN)`
broadly (a Sobol sequence), solves the linear area exactly for every sample,
then refines the best candidates. `n_samples`, `n_refine`, and `n_top` set
those three phases; the example values are fast, while the defaults
(20000, 32, 10) are slower and more thorough. The area is limited by
`loga_bounds`, default `(-2, 2)`, meaning 0.01–100 au²; widen it if a fitted
area sits on a bound.

The same `fit_stage` call accepts wavelength and flux arrays read from any
source; a CSV is not required. Pixels with nonfinite flux are skipped. If you have measurement errors, pass one
representative scalar as `sigma`, for example
`sigma=float(np.nanmedian(error))`. The current fitter does not weight each
pixel by its individual error.

`fit_stage` returns fitted parameters, component spectra, the total model,
residuals, and posterior error bars (see {ref}`uncertainties`). Sampling the
posterior takes about a minute per stage; pass `n_mcmc=20000` for a quick
look, use the default for results you report, or pass `errorbars=False` to
skip the error bars and get the best fit only. It accepts one molecule or a list for a
joint fit. `fit_ranges` accepts one `(low, high)` pair, a list of pairs, or a
molecule-to-ranges mapping. Every fit interval must lie within
`wavelength_range` and inside each selected checkpoint's trained domain.
`MODEL_SPECS` records each checkpoint's wavelength, temperature, column-density,
PCA, and network settings; the loader verifies them. Only requested models are
loaded.

### Choosing sigma=0 or sigma=2

Every molecule has two sets of checkpoints, trained on slab-model grids that
assume a turbulent line broadening of 0 km/s (sigma=0, the default) or
2 km/s (sigma=2). With sigma=0 the lines are broadened only thermally, which
assumes the turbulent velocity is much smaller than the thermal velocity.
Choose sigma=2 when you expect non-negligible turbulence. Select the sigma=2
set with
`load_fitting_models(['H2O', 'C2H2'], sigma=2)`, or pass `sigma=2` to
`fit_observation`. All 18 molecules cover the same bands at both sigmas; see
the table in the README. Both C2H2 banks cover 6–9 and 11.5–17 µm (there is
no C2H2 checkpoint between 9 and 11.5 µm), OH covers 15–26 µm, and C6H6 covers
13–17 µm and 100–675 K. `fit_stage` takes the matching parameter bounds from
the loaded checkpoints.

```{note}
The keyword `sigma` means two different things. In `load_fitting_models`,
`synthesize_spectrum`, and `fit_observation`, `sigma` (0 or 2) selects the
model grid. In `fit_stage` and `fit_molecules`, `sigma` is the flux noise in
Jy used for uncertainty weighting; choose the grid when you load the models.
```

For C6H6 and C3H4, load `['C6H6', 'C3H4']` and fit them in separate stages
with `wavelength_range` `(13.0, 17.0)` and `(15.0, 18.0)`, respectively.
The cool C6H6 grids peak near 14.84 µm, so include that region in its
`fit_ranges` if your observation covers it. The bundled C6H6 checkpoint is
trained for `T=100–675 K`; C3H4 covers `T=100–1400 K`. Both cover
`logN=13–19.5`. Fit hotter C6H6 with the direct grid workflow below.

The package includes a CH3 checkpoint covering 15–18 µm,
`T=100–1400 K`, and `logN=13–19.5`. Its strongest feature lies around
16.4–16.6 µm, so a CH3 stage can use `wavelength_range=(15.0, 18.0)` and
`fit_ranges=(16.3, 16.7)`.

## Isotopologue temperature and area constraints

Fit the main and rare isotopologues in the **same stage**. The first label
in each `isotopologue_groups` entry is the main species. `Tshare` and `Ashare`
have one entry per group:

```python
models = load_fitting_models(['C2H2', '13C12CH2'])
fit = fit_stage(
    wave, flux,
    molecule=['C2H2', '13C12CH2'],
    wavelength_range=(11.5, 17.0),
    fit_ranges=(12.9, 16.25),
    pretrained=models,
    isotopologue_groups=[('C2H2', '13C12CH2')],
    Tshare=['equal'], Ashare=[True],
    T_bounds={'C2H2': (300, 900), '13C12CH2': (300, 900)},
    logN_bounds={'C2H2': (14, 19), '13C12CH2': (13, 18)},
    loga_bounds=(-2, 2),
    distance_pc=140.0,
)
```

`Tshare='equal'` fits one temperature for a group. `Tshare='less'` enforces
each rare species' temperature to be no greater than the main species'
temperature; it does not order the rare species among themselves.
`Tshare='free'` leaves their temperatures independent. These modes appear in
a list, such as `Tshare=['less', 'equal']` for two groups. `Ashare=[True]`
fits one emitting area for the group; `False` leaves areas independent.
Column densities remain independent in every mode.

The same three keys work in a `fit_observation` stage. Each group needs a main
and at least one distinct rare component; groups cannot overlap. Incompatible
temperature bounds raise an error. `T_bounds`, `logN_bounds`, and
`loga_bounds` may differ by component label or molecule; a tuple applies to
every component. For a shared-area group, the allowed areas must overlap. If
detection screening removes a group member, the remaining members are fitted
and any group left with one member is dropped. The older `shared_area_groups` option still ties areas
without specifying a temperature relationship. No constraint spans separate
stages, because each stage fits the preceding residual.

To fit two components of the same molecule together, repeat its name in
`mol`, assign unique `component_names`, and key bounds by those labels. This
works in both `fit_observation` and `fit_observation_grids` stages:

```python
acetylene_stage = {
    'name': 'acetylene',
    'mol': ['C2H2', 'C2H2'],
    'component_names': ['C2H2_thin', 'C2H2_thick'],
    'wavelength_range': (12.9, 14.8),
    'fit_ranges': (12.9, 14.8),
    'T_bounds': (100, 1400),
    'logN_bounds': {'C2H2_thin': (13, 18), 'C2H2_thick': (19, 22)},
    'loga_bounds': {'C2H2_thin': (-4, 2), 'C2H2_thick': (-2, 4)},
}
```

Set `detection_screening=False` when you require both components regardless
of the detection screen. Direct-grid fits also require grid files covering
the full requested T and logN rectangle.

## Ordered stages, saved results, and plots

`fit_observation` runs several stages in order and saves every result. It
accepts a CSV path or, since 2.2.1, arrays you have already read from FITS,
separate files, or your own I/O code. With the Step 1 variables:

```python
from diskmelts import fit_observation

stages = [{'name': 'hcn', 'mol': 'HCN',
           'wavelength_range': (11.5, 17.0),
           'fit_ranges': (13.8, 14.2)}]
result = fit_observation(
    stages=stages, obs_wav=wave, obs_flux=flux, obs_err=error,
    name='my_source', output_dir='results/my_source', distance_pc=140.0,
)
```

`obs_err` is optional. Its positive finite median supplies one noise value for
detection screening and the noise level of the error-bar likelihood; the
spectral optimization remains unweighted by individual errors. If no error array is
given, the workflow estimates one noise value from `line_free_windows` as in
the file-based workflow. Arrays must have matching one-dimensional shapes;
pixels with nonfinite wavelength or flux are dropped, and remaining wavelengths
must be unique and increasing. Supply either arrays or `input_path`, and set
`name` to choose the saved-file prefix for array input.
The fitting notebook writes the filtered array input to a small CSV only when
calling `plot_saved_observation`, whose saved-result plotting interface reads
a source file.

```python
from pathlib import Path
from diskmelts import fit_observation, plot_saved_observation

# Set this to your own continuum-subtracted CSV, wherever you keep it.
source = Path('my_spectrum.csv')
# Repository example only, if you downloaded its files:
# source = Path('Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv')
stages = [
    {
        'name': 'water', 'mol': 'H2O',
        'wavelength_range': (11.0, 19.0),
        'fit_ranges': [(11.0, 12.0), (16.5, 18.5)],
        'h2o_components': 2,
        'component_names': ['H2O_warm', 'H2O_hot'],
        'T_bounds': {'H2O_warm': (200, 600), 'H2O_hot': (600, 1100)},
    },
    {
        'name': 'carbon', 'mol': ['C2H2', 'HCN', '13C12CH2'],
        'wavelength_range': (11.5, 17.0),
        'fit_ranges': (12.9, 14.8),
        'isotopologue_groups': [('C2H2', '13C12CH2')],
        'Tshare': ['less'], 'Ashare': [True],
    },
    {
        'name': 'CO2', 'mol': ['CO2', '13CO2', 'HC3N'],
        'wavelength_range': (11.5, 17.0),
        'fit_ranges': [(14.8, 15.2), (16.1, 16.4)],
        'isotopologue_groups': [('CO2', '13CO2')],
        'Tshare': ['equal'], 'Ashare': [True],
    },
]
output = Path('results/my_source')
result = fit_observation(
    source, stages, name='my_source', output_dir=output,
    distance_pc=140.0, detection_screening=False,
    n_samples=4096, n_refine=12, n_top=10, seed=42,
)
figures = plot_saved_observation(
    source, output, stages, name='my_source',
    active_stages=result['fits'], sigma_noise=result['sigma_noise'],
    show=True, savefig=False,
)
```

The stages run in order. Each stage fits the previous stage's residual and
writes parameter, spectrum, and running-residual CSVs.

Stage dictionary keys:

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Unique stage name; used in output file names |
| `mol` | yes | One molecule name or a list for a joint fit |
| `wavelength_range` | yes | Allowed and plotted range in µm |
| `fit_ranges` | yes | `(lo, hi)`, a list of pairs, or a dict by molecule; must lie inside `wavelength_range` and the model band |
| `h2o_components` | no | 1 (default) or 2 water components |
| `component_names` | no | Unique labels, one per fitted component |
| `T_bounds`, `logN_bounds`, `loga_bounds` | no | A tuple for all components, or a dict by label or molecule |
| `isotopologue_groups`, `Tshare`, `Ashare` | no | Isotopologue temperature and area ties (above) |
| `shared_area_groups` | no | Area ties without a temperature relation |
| `detect_peaks` | no | Windows checked by detection screening (below) |

With `detection_screening=True` (the default), a stage that has
`detect_peaks` is checked first: a molecule whose peak in its window is below
3σ is dropped, and a stage with nothing detected is skipped with a printed
message. Stages without `detect_peaks` are always fitted. σ is the median of
`obs_err` when given; otherwise it is the scatter in `line_free_windows`.

Each stage writes these files to `output_dir` (prefix `{name}_{stage}`):

| File | Contents |
|---|---|
| `{prefix}_fit_params.csv` | One row per component. `T`, `logN`, `log10A`, `A`, `logNA` (log10 N·A, sampled directly so the N–A correlation is kept): posterior medians, with `_minus`/`_plus` error bars, `_q16`/`_q50`/`_q84` percentiles, `_at_bound` flags, and `_best` best-fit values; plus `uncertainty_method`, `noise_correlation_length`, `sigma_likelihood`, and `mcmc_r_hat` |
| `{prefix}_fit_spectrum.csv` | Wavelength, input flux, total best-fit model, residual, and per-component models |
| `{prefix}_fit_top.csv` | The ranked top solutions |
| `{prefix}_running.csv` | Residual and cumulative model after this stage |
| `{name}_final_running.csv` | The same after the last stage |
| `Fitted_Parameters.csv` | One row per source for H2O, C2H2, HCN, and CO2 only: posterior medians of logN and log10A, and T, with error bars; use the per-stage params files for every component |

The returned dict has `fits` (one `fit_stage` result per stage), `residual`,
`cumulative_model`, `sigma_noise`, `grid_sigma`, and `products` (saved paths). Adjust the stage
wavelength windows and `distance_pc` for your observation. The shown stage
settings follow the repository's J16120505 example; that source path is
commented out above. The plotter checks the
saved subtraction sequence and creates stage and combined figures. It can be
called later without loading checkpoints or rerunning optimization. Set
`output_dir` explicitly; otherwise fit CSVs go under the current working
directory. Set `figure_dir` when saving figures. The source CSV may have a
header row; the default
reader skips one row and takes the first two columns as wavelength in µm and
flux in Jy. Override `skiprows`, `wav_col`, or `flux_col` if needed.

In a notebook, use `plot_saved_observation(..., show=True, savefig=False)`
to display each stage and combined figure inline without writing PNGs.
Set `savefig=True` to save them as well. With `savefig=False`, the returned
entries for the figures are Matplotlib Figure objects; otherwise they are
paths to the saved PNGs. The numeric plot-data CSV is written in either case.

![Upper Sco J16120505 constrained carbon isotopologue fit](_static/examples/upper_sco_carbon_tied_fit.png)

This carbon-stage figure is from
{doc}`Example_Fitting <notebooks/Example_Fitting>`. Its parameter box shows
that 13C12CH2 is no hotter than C2H2 and shares its area; their
column densities differ. The lower panel shows the residual for that stage.

For the repository's J16120505 example, run
`python examples/dev_v2_realobs.py`. Use
`python examples/dev_v2_realobs.py --input /path/to/my_spectrum.csv --name my_source --distance-pc 140 --sigma 2`
for another CSV, after adjusting the script's stage windows. The script uses
the packaged checkpoints. The repository also includes
`python examples/synthetic_array_fit.py`, which generates HCN and fits its
wavelength, flux, and error arrays without an input file.

## Distance and area

The checkpoints predict flux for `A=1 au²` at 140 pc. Supply the known source
distance using `distance_pc` (default 140). The fitted physical area scales as
`A * (140 / distance_pc)**2` in the observed flux. Distance and area cannot both
be measured from one spectrum because only `A / distance_pc**2` is constrained.
When `sigma` is supplied, the model fitter scales it with the flux to the
140 pc reference.

(uncertainties)=
## What the uncertainties mean

After the optimizer finds the best fit, each stage samples its posterior
with the [emcee](https://emcee.readthedocs.io/) ensemble sampler, using the
surrogate models. The reported value of every parameter is the **posterior
median**, and its error bars run to the **16th and 84th percentiles**. These
are what the printout, the plots, and the saved CSV files show.
`fit['uncertainty'][label][key]` holds `q16`, `q50`, `q84`, `minus`
(`q50 - q16`), `plus` (`q84 - q50`), `best`, and `at_bound` for `T`, `logN`,
and `log10A`.

The likelihood compares the fitted pixels with one noise level, adjusted for
two effects a plain χ² ignores:

- **Correlated noise.** MIRI spectra are oversampled and resampled, so
  neighbouring pixels are not independent. The noise is scaled by the square
  root of the residual's autocorrelation length (summed over at most 10
  pixels; typically 5–10 for MIRI).
- **Model mismatch.** If the best fit's reduced χ² exceeds 1, the noise is
  scaled up to match; it is never scaled down below the supplied value.

Without a noise level, the best-fit residual RMS is used. Priors are flat
inside the fit bounds.

How to read the results:

- **Best fit vs median.** The best fit (`best`, and the `*_best` CSV
  columns) is the single best-fitting model, and the plotted model spectra
  use it. Where temperature, column density, and area are degenerate, most
  of the probability can lie away from that point, so the median and the
  best fit can differ markedly. Report the median and its error bars.
- **`at_bound` (printed as `*`).** The 16–84% range or the best fit reaches
  a fit bound, so the bound, not the data, limits it. Widen the bound, or
  quote the value as a limit.
- **Convergence.** Each stage starts with `n_mcmc` likelihood evaluations
  (default 300 000, or set `DISKMELTS_N_MCMC`) and checks R-hat, which
  compares the walkers. Below 1.1 it stops; otherwise it doubles the run, up
  to three times, so stages with many components get longer chains
  automatically. It warns if R-hat is still above 1.1; then increase
  `n_mcmc`.
- **Not included.** Continuum-subtraction errors, slab-model assumptions,
  and the uncertainty of earlier stages, whose models are subtracted as if
  exact. Fit strongly blended species in the same stage when their
  covariance matters.

`fit['solution_spread']` keeps the spread of the optimizer's top distinct
solutions. Use it only to spot separate good solutions, not as an error bar.

### Best fits only

Error bars are on by default. Pass `errorbars=False` to `fit_stage`,
`fit_observation`, `fit_grid_stage`, or `fit_observation_grids` to skip the
MCMC step, for example while exploring stage settings:

```python
result = fit_observation(source, stages, name='my_source',
                         output_dir='results/quick', errorbars=False)
```

`fit['uncertainty']` is then empty, the printout and plots show best fits
without error bars, and the saved `T`, `logN`, `log10A`, and `A` columns hold
the best fits (with `uncertainty_method` set to `none` and empty
`_minus`/`_plus` columns).

## Optional direct-grid fit

The separate grid API requires local slab CSVs. They are **not** included in
the package and are **not** consulted by model fitting.

For C6H6 and C3H4, use the same stage dictionaries for grid fitting with
`grid_root='Model_grids/sigma0'`, provided both local grid directories exist. Grid
fitting reads the complete spectra for each requested T/logN point and rejects
nonfinite flux values. Set `sigma_noise` or choose `line_free_windows` outside
the C3H4 feature near 15.71–15.81 µm when estimating noise for that stage.

```python
from diskmelts import fit_observation, fit_observation_grids

new_stages = [
    {'name': 'c6h6', 'mol': 'C6H6',
     'wavelength_range': (13.0, 17.0), 'fit_ranges': (14.5, 15.2)},
    {'name': 'c3h4', 'mol': 'C3H4',
     'wavelength_range': (15.0, 18.0), 'fit_ranges': (15.7, 15.9)},
]
model_fits = fit_observation(
    source, new_stages, output_dir='results/new_models',
    distance_pc=122.5, detection_screening=False,
)
grid_fits = fit_observation_grids(
    source, new_stages, grid_root='Model_grids/sigma0',
    output_dir='results/new_grids', distance_pc=122.5,
    detection_screening=False,
)
```

Grid stages accept the same isotopologue and area ties as model stages, so
the `stages` list above can be reused. Local grids may also extend beyond the
surrogate models' trained domain (logN 13–19.5 for most molecules), which
makes them the choice for optically thick components such as
`'logN_bounds': {'C2H2_thick': (19.0, 24.0)}`. For an unconstrained
comparison, drop the tie keys:

```python
from diskmelts import fit_observation_grids, plot_saved_observation

grid_stages = [
    {key: value for key, value in stage.items()
     if key not in ('isotopologue_groups', 'Tshare', 'Ashare',
                    'shared_area_groups')}
    for stage in stages
]
result = fit_observation_grids(
    source, grid_stages, name='my_source',
    grid_root='Model_grids/sigma0', output_dir='results/my_source_grids',
    distance_pc=122.5,
)
plot_saved_observation(
    source, 'results/my_source_grids', grid_stages, name='my_source',
    figure_dir='figures/my_source_grids', active_stages=result['fits'],
)
```

`Model_grids/sigma0/<molecule>/T{temperature}N{log_column}.csv` must contain `wave`
and `Line` columns and cover the fitted domain. The grid fitter interpolates
between T/logN points, refines the best solutions, samples the posterior on
the interpolated grid exactly as for model fits, and adds the full local grid
step in quadrature to each side of the T/logN error bars. For example, a 25 K
step gives `T_total_sigma = sqrt(T_fit_sigma**2 + 25**2)` K in the
`*_grid_uncertainty.csv` file. Its `_q16`/`_q84` columns are the pure
posterior percentiles, while `_minus`/`_plus` include the grid step. With
`errorbars=False` no grid step is reported. See {doc}`training` for obtaining and
using optional grids.
