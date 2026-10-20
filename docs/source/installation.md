# Installation

DiskMELTS is available on [PyPI](https://pypi.org/project/diskmelts/). You do
not need to clone the repository to fit spectra or subtract a continuum. The
package requires Python 3.9 or newer. Python 3.11 or 3.12 is a practical
choice for a new installation.

## Install in a new Python environment

First, check which Python you have. On macOS or Linux run `python3 --version`;
on Windows run `py --version`. If the command is missing, install Python from
[python.org](https://www.python.org/downloads/) or use the Conda instructions
below. If you have several versions, use an installed one throughout the steps
below. For example, replace `python3` with `python3.11` or `python3.12` if that
is the version available on your machine. A command such as
`python3.11` only works when that specific interpreter is installed and on your
PATH.

### macOS or Linux

```bash
python3 -m venv diskmelts-env
source diskmelts-env/bin/activate
python -m pip install diskmelts
```

### Windows PowerShell

```powershell
py -3 -m venv diskmelts-env
.\diskmelts-env\Scripts\Activate.ps1
python -m pip install diskmelts
```

If Windows has several Python versions, use `py -0p` to list them and replace
`py -3` with a specific installed version, such as `py -3.11`.

The environment is active when its name appears in your terminal prompt. Use
`python` for later commands and scripts in that terminal. Run `deactivate` to
leave the environment.

### If you already use Conda or Mamba

Create a fresh environment without a repository checkout:

```bash
conda create -n diskmelts python=3.11 pip
conda activate diskmelts
python -m pip install diskmelts
```

Use `mamba create` instead of `conda create` if you prefer Mamba. After
activation, `python` refers to the environment's Python. Run `conda deactivate`
to leave it. If you already have an environment where you want DiskMELTS,
activate that environment and run `python -m pip install diskmelts` there.

## Check the installation

With the environment active, run:

```bash
python --version
python -c "import diskmelts; print('DiskMELTS is ready')"
```

The second command should print `DiskMELTS is ready`.
Use the same environment in your terminal, script, or Jupyter kernel. Restart
an existing notebook kernel after installing or upgrading the package.

## Choose optional features

The basic install handles CSV or NumPy spectra, continuum subtraction, and
fitting with the bundled production models. Add an extra only when you need it:

| Need | Command |
| --- | --- |
| Read FITS files with the DiskMELTS helper | `python -m pip install "diskmelts[io]"` |
| FITS support and JupyterLab | `python -m pip install "diskmelts[io,notebooks]"` |

On Windows PowerShell, the same quoted commands work. The `notebooks` extra
installs JupyterLab; the example notebook files are in the
[GitHub repository](https://github.com/wddlx/DiskMELTS/tree/main/notebooks)
and are not installed by pip.
For the fitting and training notebooks, use Python 3.11 or newer and also
install `scikit-learn==1.8.0` in the notebook environment, then restart its kernel;
the bundled PCA and scaler objects were saved with that version.

To update an existing installation later, run
`python -m pip install --upgrade diskmelts` in its active environment.

## Your first spectrum

Version 2.2.0 includes 44 production checkpoints: 21 trained with sigma=0 and
23 trained with sigma=2, plus continuum reference data. Choose the new models
with `load_fitting_models('C6H6', sigma=2)` or `fit_observation(..., sigma=2)`.
You supply a spectrum. The {doc}`quick start <quickstart>` shows how to load a
CSV, subtract a continuum if needed, fit a molecule, and use a
[downloadable example spectrum](https://github.com/wddlx/DiskMELTS/blob/main/Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv).
Ordinary fitting needs neither the repository nor the full slab grids.

Full `Model_grids/` data are separate and needed only for direct-grid fitting
or training. For the current grids, set `grid_root='Model_grids/sigma0'`
when using the grid API.
See {doc}`quickstart` and {doc}`training` for those workflows.

## For contributors: install from a checkout

Clone the repository when you want to edit the code or run its example
notebooks and tests. The supplied `environment.yaml` creates the core Conda
environment and installs DiskMELTS in editable mode:

```bash
git clone https://github.com/wddlx/DiskMELTS.git
cd DiskMELTS
conda env create -f environment.yaml
conda activate diskmelts
```

Use `mamba env create -f environment.yaml` if you prefer Mamba. For notebooks
and FITS support, use `environment-notebooks.yaml`; for tests and docs, use
`environment-dev.yaml`. Each file defines its own environment and installs
the checkout in editable mode. If you prefer a Python venv, activate one and
run `python -m pip install -e ".[dev,io,notebooks]"`, then `pytest`.

On Linux or Windows, pip selects the PyTorch wheel for the platform. If you
need a particular CUDA or ROCm build, follow the
[PyTorch installation selector](https://pytorch.org/get-started/locally/)
before installing DiskMELTS. On Intel macOS, DiskMELTS constrains NumPy below
2 for compatibility with the available PyTorch wheels.
