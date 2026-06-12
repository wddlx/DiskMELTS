# DiskMELTS

**Machine-learning Enabled Line fitting Tool for Spectra**

DiskMELTS is a neural surrogate-assisted fitting package for retrieving
molecular gas parameters — temperature ($T$), column density ($\log N$), and
emitting area ($A$) — from JWST mid-infrared spectra of protoplanetary disks.

Most users should start with the pretrained models committed to the repository.
The checkpoints are self-contained and do not require the large local training
grids or pretraining CSV files.

---

## How it works

Each pretrained molecule is represented by two small MLPs trained on slab-model
grids:

- `net_shape` : $(T, \log N)$ → peak-normalised spectral shape (via PCA)
- `net_peak`  : $(T, \log N)$ → $\log_{10}(\text{peak flux})$

Retrieval is done by optimization through the forward models, with the linear
area scaling solved analytically during the fit.

---

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
training
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/trainmodel
api/fitting
api/validation
api/plotting
```
