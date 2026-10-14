# DiskMELTS

**Machine-learning Enabled Line fitting Tool for Spectra**

DiskMELTS subtracts spectral continua and uses neural surrogates to retrieve
molecular gas parameters — temperature ($T$), column density ($\log N$), and
emitting area ($A$) — from JWST mid-infrared spectra of protoplanetary disks.

Most users can start with the pretrained models bundled in the installed
package. The checkpoints require no local training grids.

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} Raw spectrum → continuum subtraction
:link: notebooks/Example_Consub
:link-type: doc

Work through the Upper Sco J16120505 spectrum. See calibration pixels,
excluded wavelengths, and regions where the continuum has weak support.
:::

:::{grid-item-card} Continuum-subtracted spectrum → molecular fit
:link: notebooks/Example_Fitting
:link-type: doc

Fit the bundled models in stages and inspect each component and the
combined residual.
:::

::::

Start with {doc}`installation`, browse the {doc}`notebook gallery <notebooks/index>`,
or read the detailed {doc}`continuum` and {doc}`quickstart` guides.

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
continuum
notebooks/index
quickstart
training
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/consub
api/fitting
api/v2
api/trainmodel
api/validation
api/plotting
```
