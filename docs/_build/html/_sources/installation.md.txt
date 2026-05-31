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

## Install from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/YOUR_USERNAME/DiskMELTS.git
cd DiskMELTS
pip install -e .
```

This makes `import diskmelts` available from anywhere in your environment.

## Conda environment (recommended)

If you use Conda, create the project environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate diskmelts
```

The environment file installs DiskMELTS in editable mode with `pip install -e .`.

## Verify

```bash
python test.py
```
