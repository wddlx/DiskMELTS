# Installation

## Requirements

| Package | Purpose |
|---|---|
| `torch` | MLP training and inference |
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

If you use Conda, create a dedicated environment first:

```bash
conda create -n diskmelts python=3.11
conda activate diskmelts
pip install -e .
```

## Verify

```python
import diskmelts
print(diskmelts.__version__)   # should print 0.1.0
```
