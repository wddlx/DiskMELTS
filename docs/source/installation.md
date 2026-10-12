# Installation

DiskMELTS requires Python 3.9 or newer. The runtime dependencies are NumPy,
pandas, SciPy, scikit-learn, PyTorch, and matplotlib. Astropy is optional and
needed only for the FITS-reading helper.

## Install from a checkout

```bash
git clone https://github.com/wddlx/DiskMELTS.git
cd DiskMELTS
python -m pip install -e .
```

For FITS support, replace the last command with:

```bash
python -m pip install -e ".[io]"
```

Run installation commands in the environment used by your Python interpreter or
Jupyter kernel. Restart existing notebook kernels after upgrading the package.
Editable installation makes source changes available without reinstalling;
changes to dependencies or package metadata may still require reinstalling.

## Conda environment

Alternatively, from the repository root:

```bash
conda env create --file environment.yaml
conda activate diskmelts
```

This installs the local package in editable mode, its runtime dependencies,
Astropy, and Jupyter. Select the `diskmelts` environment when using notebooks.

## What is bundled?

The installed package contains:

- 15 production checkpoints: five H2O tiles and one checkpoint for each of ten
  other molecules, plus the H2O bank manifest;
- four continuum reference presets and the curated correction windows.

Ordinary model fitting and continuum subtraction need no `Model_grids/`,
`Pretrain_grid/`, or repository example directory. You supply your spectrum.
The checkout additionally contains example spectra, scripts, and notebooks.

The full slab grids are optional local data, excluded from Git and distributions.
`Model_grids/` is required for direct-grid fitting, retraining, or generating
new continuum reference windows. `Pretrain_grid/` is used by training and
related validation. The top-level `Trained_model/` holds optional local training
outputs; normal inference loads checkpoints inside the installed package.

## Verify and test

```bash
python -c "import diskmelts; print(diskmelts.consub.calculate.__name__)"
```

For development, from the repository root:

```bash
python -m pip install -e ".[dev,io]"
pytest
```

`pytest` runs the distributed tests in `tests/`. The private `0-Consub/`
directory is ignored by Git and is not required for installation or testing.
If you have that local directory, run its additional continuum tests explicitly:

```bash
pytest -q 0-Consub/consub_test.py
```

## Build the documentation

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -E -W -b html docs/source docs/_build/html
```

The build does not download external API inventories.
