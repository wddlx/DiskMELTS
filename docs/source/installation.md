# Installation

## Requirements

| Package | Purpose |
|---|---|
| `torch` | Pretrained model inference and optional training |
| `numpy` | Array operations |
| `pandas` | CSV I/O |
| `scipy` | L-BFGS-B optimiser, NNLS, Sobol sampling |
| `scikit-learn` | `StandardScaler`, PCA |
| `matplotlib` | Plotting |

Python ≥ 3.9 is required.

## Conda (beginner-friendly)

We provide `environment.yaml` as a quick-start option. First, check that you have conda:

    conda --version

If you do not have conda, use a Python virtual environment instead.

Update conda to version 23 or above:

    conda update conda

Clone the repository and create the environment from the repository root:

```bash
git clone https://github.com/wddlx/DiskMELTS.git
cd DiskMELTS
conda env create --file=environment.yaml
```

Activate it:

    conda activate diskmelts

Verify everything is installed:

    conda list

## Custom environment

Feel free to use your own conda or virtual environment. Install the package with:

```bash
pip install -e .
```

This makes `import diskmelts` available from anywhere in your environment.

## Repository data layout

A fresh GitHub clone or built wheel contains everything required for v2 model
fitting. The wheel bundles 15 production checkpoint files (five H2O tiles and
one checkpoint for each of ten other molecules), plus the H2O manifest under
`diskmelts/models/Trained_model/`. Model fitting loads these files through
`load_fitting_models`; it does not read any slab-grid CSVs.

The repository additionally contains:

- an example observed spectrum under `Realobs_data/Consub_data/`
- `examples/dev_v2_realobs.py`
- `notebooks/Example_Fitting.ipynb` (model fitting, with an optional grid cell)

The large training inputs are intentionally not uploaded:

- `Model_grids/`
- `Pretrain_grid/`

These directories are ignored by Git. `Model_grids/` is required only for the
optional `fit_observation_grids` API or training. `Pretrain_grid/` is needed
only for training and related validation. Neither belongs in a fitting-only
GitHub upload or wheel. The repository's top-level `Trained_model/`, generated
`realobs_results/`, `figures/` outputs, and `docs/_build/` are also excluded
from the fitting-only upload. Four reference PDFs under
`figures/realobs_validations/` remain included. Normal v2 inference uses the
bundled checkpoints.

## Verify

Verify the package import:

```bash
python -c "import diskmelts; print(diskmelts.__version__ if hasattr(diskmelts, '__version__') else 'DiskMELTS import OK')"
```

To run the development test suite, install the optional development
dependencies first:

```bash
pip install -e ".[dev]"
pytest tests/
```

The test suite checks the bundled fitting assets and notebook paths. Its
scientific-optimizer tests use small synthetic inputs.

## Build the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -E -W -b html docs/source docs/_build/html
```

The build is self-contained and does not download external API inventories.
