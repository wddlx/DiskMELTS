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
git clone https://github.com/YOUR_USERNAME/DiskMELTS.git
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

## Verify

```bash
pytest tests/
```
