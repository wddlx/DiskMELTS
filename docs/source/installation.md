# Installation

DiskMELTS requires Python 3.9 or newer; Python 3.10 and 3.11 are practical
installation targets. The runtime dependencies are NumPy, pandas, SciPy,
scikit-learn, PyTorch, and matplotlib. Astropy is optional and needed only for
the FITS-reading helper.

## Install from a checkout

Clone the repository and create a virtual environment with Python 3.11. The
following installs FITS support and JupyterLab for the supplied examples:

```bash
git clone https://github.com/wddlx/DiskMELTS.git
cd DiskMELTS
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[io,notebooks]"
.venv/bin/python -c "import diskmelts; print(diskmelts.__name__)"
```

Use `python3.10` if that is your native Python. On Apple Silicon, check
`python3.10 -c "import platform; print(platform.machine())"` (or use your
Python 3.11 command); it should print `arm64`. An existing Intel Conda does
not affect `.venv/bin/python` when the environment was created with native
Python. For core fitting and CSV/array inputs only, install `-e .` instead.
For FITS without notebooks, install `-e ".[io]"`.

On Windows PowerShell, run the equivalent commands from the repository root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[io,notebooks]"
.venv\Scripts\python.exe -c "import diskmelts; print(diskmelts.__name__)"
```

Use the virtual environment's Python for later scripts. If you prefer to type
`python` and `pytest` directly, activate it with `source .venv/bin/activate`
on macOS/Linux or `.venv\Scripts\Activate.ps1` in PowerShell. The `.venv/`
directory is ignored by Git.

Run installation commands in the environment used by your Python interpreter or
Jupyter kernel. Restart existing notebook kernels after upgrading the package.
Editable installation makes source changes available without reinstalling;
changes to dependencies or package metadata may still require reinstalling.

## Conda environment

The three YAML files keep Conda's solve small by requesting only Python and
pip. pip then installs the local package and its dependencies from
`pyproject.toml`. From the repository root, choose one file:

| File | Environment name | Installed features |
| --- | --- | --- |
| `environment.yaml` | `diskmelts` | Core fitting and continuum subtraction |
| `environment-notebooks.yaml` | `diskmelts-notebooks` | Core, FITS, JupyterLab |
| `environment-dev.yaml` | `diskmelts-dev` | Core, FITS, JupyterLab, tests and docs |

For most users of the supplied notebooks, the command is:

```bash
mamba env create --file environment-notebooks.yaml
conda activate diskmelts-notebooks
python -c "import diskmelts; print(diskmelts.__name__)"
```

With a current Conda release,
`conda env create --file environment-notebooks.yaml` works too. Substitute
another filename and its environment name from the table when you need a
different feature set. All
three files use conda-forge only and install DiskMELTS in editable mode.
Select the matching environment as your Jupyter kernel when using notebooks.

On Apple Silicon, use a native `osx-arm64` Conda installation such as Miniforge.
Older Intel (`osx-64`) Anaconda installations run through Rosetta and may solve
slowly, even with the smaller YAML files. Miniforge includes Mamba and has a
native Apple Silicon installer. If you already have native Python 3.10 or
3.11, the virtual environment commands above need no Conda installation.

Check the active Conda platform with `conda info` (look for `platform: osx-arm64`)
before creating the environment. An existing Intel Conda installation can be
kept separately; the new installation does not need to replace it.

If an existing Intel macOS environment reports that PyTorch was compiled with
NumPy 1.x but NumPy 2 is installed, update the editable package from the
repository root:

```bash
conda run -n diskmelts-notebooks python -m pip install -e ".[io,notebooks]"
```

DiskMELTS now requires NumPy 1.26 or newer but below 2 on Intel macOS, where the available PyTorch
2.2 wheels need NumPy 1.x. The same package metadata allows NumPy 2 on other
platforms with compatible PyTorch wheels. Restart the Jupyter kernel after
updating the environment.

On Linux and Windows, the core pip install uses the PyTorch wheel selected for
that platform. For a particular NVIDIA CUDA or AMD ROCm build, choose the
command for your system from the
[PyTorch installation selector](https://pytorch.org/get-started/locally/)
before installing DiskMELTS.

## What is bundled?

The installed package contains:

- 18 production checkpoints: five H2O tiles and one checkpoint for each of thirteen
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

`pytest` runs the distributed tests in `tests/`, including the continuum
API and bundled-reference checks. The private `0-Consub/`
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
