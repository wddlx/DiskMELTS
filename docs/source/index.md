# DiskMELTS

**Machine-learning Enabled Line fitting Tool for Spectra**

DiskMELTS subtracts spectral continua and uses neural surrogates to retrieve
molecular gas parameters — temperature ($T$), column density ($\log N$), and
emitting area ($A$) — from JWST mid-infrared spectra of protoplanetary disks.

Most users should start with the pretrained models bundled in the installed
package. The checkpoints are self-contained and require no large local training
grids or pretraining CSV files.

For raw spectra, start with {doc}`continuum` and `diskmelts.consub.calculate`.
For spectra that are already continuum-subtracted, start with {doc}`quickstart`.

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
continuum
training
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/trainmodel
api/fitting
api/v2
api/consub
api/validation
api/plotting
```
