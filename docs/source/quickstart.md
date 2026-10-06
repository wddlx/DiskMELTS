# Quick Start: fit an observed spectrum

The installed `diskmelts` package includes the production v2 checkpoints. A
model fit needs no `Model_grids/`, `Pretrain_grid/`, training script, or network
configuration file. Inputs are a continuum-subtracted spectrum in Jy and its
source distance in parsecs. Run this example from a writable working directory.

## One stage

```python
from diskmelts import load_fitting_models, fit_stage, load_observed_spectrum

wave, flux = load_observed_spectrum('my_continuum_subtracted.csv')
models = load_fitting_models(['H2O'])
fit = fit_stage(
    wave, flux,
    molecule='H2O',
    wavelength_range=(11.0, 19.0),
    fit_ranges=[(11.0, 12.0), (16.5, 18.5)],
    pretrained=models,
    h2o_components=2,
    component_names=['H2O_warm', 'H2O_hot'],
    T_bounds={'H2O_warm': (200, 600), 'H2O_hot': (600, 1100)},
    sigma=0.001,
    distance_pc=140.0,
)
print(fit['params'])
```

`fit_stage` returns fitted parameters, component spectra, the total model,
residuals, and uncertainty estimates. It accepts one molecule or a list for a
joint fit. `fit_ranges` accepts one `(low, high)` pair, a list of pairs, or a
molecule-to-ranges mapping. Every fit interval must lie within
`wavelength_range` and inside each selected checkpoint's trained domain.
`MODEL_SPECS` records each checkpoint's wavelength, temperature, column-density,
PCA, and network settings; the loader verifies them. Only requested models are
loaded.

## Ordered stages, saved results, and plots

```python
from pathlib import Path
from diskmelts import fit_observation, plot_saved_observation

source = Path('my_continuum_subtracted.csv')
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
        'name': 'carbon', 'mol': ['C2H2', 'HCN'],
        'wavelength_range': (11.5, 17.0),
        'fit_ranges': (12.9, 16.25),
    },
]
output = Path('results/my_source')
result = fit_observation(
    source, stages, name='my_source', output_dir=output,
    distance_pc=122.5, detection_screening=False,
)
figures = plot_saved_observation(
    source, output, stages, name='my_source',
    figure_dir='figures/my_source', active_stages=result['fits'],
    sigma_noise=result['sigma_noise'],
)
```

The stages run in order. Each stage fits the previous stage's residual and
writes parameter, spectrum, and running-residual CSVs. The plotter checks the
saved subtraction sequence and creates stage and combined figures. It can be
called later without loading checkpoints or rerunning optimization. Set
`output_dir` and `figure_dir` explicitly; otherwise fit CSVs go under the
current working directory. The source CSV may have a header row; the default
reader skips one row and takes the first two columns as wavelength in µm and
flux in Jy. Override `skiprows`, `wav_col`, or `flux_col` if needed.

For the repository's J16120505 example, run
`python examples/dev_v2_realobs.py`. The example uses the packaged API.

## Distance and area

The checkpoints predict flux for `A=1 au²` at 140 pc. Supply the known source
distance using `distance_pc` (default 140). The fitted physical area scales as
`A * (140 / distance_pc)**2` in the observed flux. Distance and area cannot both
be measured from one spectrum because only `A / distance_pc**2` is constrained.
When `sigma` is supplied, the model fitter scales it with the flux to the
140 pc reference. Fitted uncertainties are optimization summaries; bound hits
and degeneracies require caution.

## Optional direct-grid fit

The separate grid API requires local slab CSVs. They are **not** included in
the package and are **not** consulted by model fitting.

```python
from diskmelts import fit_observation_grids, plot_saved_observation

result = fit_observation_grids(
    source, stages, name='my_source',
    grid_root='Model_grids', output_dir='results/my_source_grids',
    distance_pc=122.5,
)
plot_saved_observation(
    source, 'results/my_source_grids', stages, name='my_source',
    figure_dir='figures/my_source_grids', active_stages=result['fits'],
)
```

`Model_grids/<molecule>/T{temperature}N{log_column}.csv` must contain `wave`
and `Line` columns and cover the fitted domain. The grid fitter interpolates
between T/logN points, refines the best solutions, and adds the full local grid
step in quadrature to the formal T/logN errors. For example, a 25 K step gives
`T_total_sigma = sqrt(T_fit_sigma**2 + 25**2)` K. Its covariance describes
one local mode, not a full posterior. See {doc}`training` for obtaining and
using optional grids.
