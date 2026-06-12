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

If you don't have conda, google how to install it!

Update conda to version 23 or above:

    conda update conda

Press `y` to all prompts.

Clone the repository and create the environment from the repository root:

```bash
git clone https://github.com/wddlx/DiskMELTS.git
cd DiskMELTS
conda env create --file=environment.yaml
```

Press `y` to all prompts. This will download a number of packages.

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

A fresh GitHub clone contains everything required for fitting:

- self-contained pretrained checkpoints under `Trained_model/`
- an example observed spectrum under `Realobs_data/Consub_data/`
- `examples/dev_v1_realobs.py`
- `notebooks/Example_Fitting.ipynb`

The large training inputs are intentionally not uploaded:

- `Model_grids/`
- `Pretrain_grid/`

These directories are ignored by Git. They are only required when generating
pretraining tables, retraining checkpoints, or running the training-validation
workflow.

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

The test suite checks the committed fitting assets and notebook paths. Tests
that exercise the scientific optimizer use small synthetic inputs so they can
run in continuous integration.

## Build the documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -E -W -b html docs/source docs/_build/html
```

The build is self-contained and does not download external API inventories.
