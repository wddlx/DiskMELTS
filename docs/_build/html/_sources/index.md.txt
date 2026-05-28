# DiskMELTS

**Machine-learning Enabled Line fitting Tool for Spectra**

DiskMELTS is a neural surrogate-assisted framework for retrieving molecular gas
parameters — temperature ($T$), column density ($\log N$), and emitting area ($A$) —
from JWST mid-IR spectra (11–19 µm) of protoplanetary disks.

---

## How it works

Each molecule is represented by two small MLPs trained on slab-model grids:

- `net_shape` : $(T, \log N)$ → peak-normalised spectral shape (via PCA)
- `net_peak`  : $(T, \log N)$ → $\log_{10}(\text{peak flux})$

Retrieval is done by optimisation through the forward models, not by inverting
them, which keeps the problem well-posed for any combination of molecules.

---

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/trainmodel
api/fitting
api/validation
api/plotting
```
