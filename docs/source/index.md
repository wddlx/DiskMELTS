# DiskMELTS

**Machine-learning Enabled Line fitting Tool for Spectra**

DiskMELTS subtracts spectral continua and uses neural surrogates, or optional
local grids of slab models, to retrieve molecular gas parameters — temperature
($T$), column density ($\log N$), and emitting area ($A$) — from JWST
mid-infrared spectra of protoplanetary disks.

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

(fitting-approaches)=
### Surrogate or grid fitting?

DiskMELTS fits with either the bundled neural surrogates or a local grid of
precomputed slab spectra (`fit_observation_grids`). Both avoid recomputing
slab models during a fit, which makes classic slab fitting slow.

| Approach | Disk space | Fit time | Accuracy |
| --- | --- | --- | --- |
| Classic slab fitting | small (line lists) | slow: every trial spectrum is recomputed | exact slab model |
| Grid fitting | large: about 40 GB per broadening for the 18 species | fast: interpolates stored spectra | exact at grid points, interpolated between them |
| Surrogate fitting (default) | 41 MB for all 46 checkpoints | fast | median spectral error 0.01–0.26% of the peak (0.66% for the 19–25 µm water tile) |

On held-out grid spectra, the surrogates recover $T$ with an RMSE of
2–24 K and $\log N$ within 0.08 dex
(see {ref}`bundled-checkpoints-2-3-0`). Grid fitting needs the slab grids on
local disk; it can also reach beyond the surrogates' trained domain, for
example optically thick components with $\log N > 19.5$. See
{doc}`quickstart` for both.

(available-molecules)=
### Bundled molecules

Every molecule has checkpoints for two slab grids: `sigma=0` (default,
thermal line broadening only) and `sigma=2` (2 km/s turbulent broadening).

| Molecule | Band (µm) | T (K) | logN (cm⁻²) |
|---|---|---|---|
| H2O | 4.9–25.0 (five overlapping tiles) | 100–1400 | 13–19 |
| C2H2 | 6–9 and 11.5–17 | 100–1400 | 13–19.5 |
| HCN, CO2, 13C12CH2, 13CO2 | 11.5–17 | 100–1400 | 13–19.5 |
| HC3N, CO17O, CO18O | 13–18 | 100–1400 | 13–19.5 |
| C4H2 | 14–18 | 100–1400 | 13–19.5 |
| C3H4, CH3 | 15–18 | 100–1400 | 13–19.5 |
| C6H6 | 13–17 | 100–675 | 13–19.5 |
| C2H6 | 9–15 | 100–1400 | 13–19.5 |
| C2H4 | 8–14 | 100–1400 | 13–19.5 |
| CH4 | 5–9 | 100–1400 | 13–19.5 |
| C2HD | 11.5–20.5 | 100–1400 | 13–19.5 |
| OH | 15–26 | 100–1400 | 13–19.5 |

A separate 100–6000 K, 4.9–27 µm OH model is used only to compare OH line
patterns with residuals ({doc}`unfitted_lines`). `MODEL_SPECS` and
`SIGMA2_MODEL_SPECS` record the exact domains, PCA sizes, and network layers;
{doc}`training` gives the training recipe and held-out accuracy.

---

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
quickstart
notebooks/index
continuum
generate_spectra
unfitted_lines
training
changelog
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
api/deprecated
```
