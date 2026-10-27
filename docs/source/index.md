# DiskMELTS

**Machine-learning Enabled Line fitting Tool for Spectra**

DiskMELTS subtracts spectral continua and uses neural surrogates to retrieve
molecular gas parameters — temperature ($T$), column density ($\log N$), and
emitting area ($A$) — from JWST mid-infrared spectra of protoplanetary disks.

Most users can start with the pretrained models bundled in the installed
package. The checkpoints require no local training grids.

Install from [PyPI](https://pypi.org/project/diskmelts/) with Python 3.9 or
newer:

```bash
python -m pip install diskmelts
```

Then use {doc}`Your first spectrum <quickstart>` to load a spectrum and fit a
molecule. The guide includes a downloadable sample CSV, so a checkout is not
needed.

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

Start with {doc}`installation`, follow {doc}`Your first spectrum <quickstart>`,
or browse the {doc}`notebook tutorials <notebooks/index>` for detailed examples.

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
notebooks/index
continuum
generate_spectra
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
api/uncertainty
```
