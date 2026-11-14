---
title: DiskMELTS, a Machine-learning Enabled Line fitting Tool for Spectra.
tags:
  - Python
  - astronomy
authors:
  - name: Chengyan Xie
    orcid: 0000-0001-8184-5547
    affiliation: 1
  - name: Dingshan Deng
    orcid: 0000-0003-0777-7392
    affiliation: 1
  - name: Ilaria Pascucci
    orcid: 0000-0001-7962-1683
    affiliation: 1
affiliations:
  - name: Lunar and Planetary Laboratory, the University of Arizona, Tucson, AZ 85721, USA
    index: 1
date: 12 October 2026
bibliography: paper.bib
---

# Summary

The James Webb Space Telescope (JWST) is delivering detailed mid-infrared spectra of planet-forming disks around young stars. These spectra contain emission from water and other molecules in the inner disks, allowing astronomers to study the regions where planets form. However, interpreting them with traditional physical models can be computationally expensive. Machine-learning Enabled Line fitting Tool for (Disk) Spectra (`DiskMELTS`) is a Python package for continuum subtraction and rapid molecular fitting, either with pretrained neural networks or with precomputed model grids. It estimates molecular temperatures, column densities, and emitting areas, and provides tools to inspect and subtract fitted components.

# Statement of Need

The JWST Mid-Infrared Instrument (MIRI) Medium Resolution Spectrograph has opened a new window onto molecular gas in protoplanetary disks, delivering high-quality mid-infrared spectra for large samples of young disks in nearby star-forming regions [@Arulanantham25; @Henning24]. These spectra contain rich emission from H$_2$O, C$_2$H$_2$, HCN, CO$_2$, and other species across the $5$--$28\,\mu$m range, tracing the chemistry and physical conditions of the warm inner disk atmosphere.

The most common way to extract physical parameters from these spectra is fitting local thermodynamic equilibrium (LTE) slab models to the observed molecular emission [@Liu19; @Xie23; @Romero24; @Arulanantham25; @Long25]. The key parameters for each molecule are the excitation temperature $T$, molecular column density $N$, and emitting area $A$. Even accelerated slab-model fitting can require tens of minutes to several hours per molecule when the $(T,N,A)$ parameter space is explored thoroughly [@Romero24]. For hydrocarbon-rich disks, where many species must be fitted together, a single spectrum can take several days on a high-performance computing cluster [@Long25]. For JWST surveys of tens to hundreds of disks, this cost becomes a major bottleneck.

`DiskMELTS` removes this bottleneck by never computing slab models during a fit. The target users are astronomers who need to analyze mid-infrared molecular spectra efficiently and reproducibly. The package is particularly well suited for (1) characterizing molecular emission across large samples of disks, (2) isolating and subtracting water or other molecular emission before measuring atomic or ionic line fluxes, and (3) obtaining quick parameter estimates to guide detailed follow-up analysis.

# State of the field

Several tools fit molecular emission from protoplanetary disks. CLIcK [@Liu19] fits slab models over a parameter grid but recomputes the models for every fit, and IRIS [@Romero24] implements JAX-accelerated line-by-line radiative transfer with Bayesian retrieval. DuCKLinG [@Kaeufer24] interpolates precomputed slab grids for 11 molecules and fits them together with a dust model by nested sampling, which makes it the tool of choice for modeling gas and dust jointly. `DiskMELTS` instead focuses on the gas lines. It first subtracts the continuum, with line-free windows that users can choose, adjust, and inspect in diagnostic plots, and then fits the molecular emission. This keeps the gas fit independent of the dust model, whose broad features can be degenerate between dust species, although the result then depends on where the continuum is placed. `DiskMELTS` is installed with `pip`, includes its trained models, and fits a spectrum in a few lines of Python.

Fitting approaches trade accuracy against disk space and run time (Table 1). Classic slab fitting stores only line lists but recomputes the spectrum at every trial parameter, so it is accurate but slow. `DiskMELTS` offers two approaches that never compute slab models during a fit: fitting precomputed grids, as DuCKLinG also does, and, new in `DiskMELTS`, fitting neural surrogates trained on those grids. Fitting a precomputed grid is fast and as accurate as the slab model at the grid points, but the grids are large: ours take about 40 GB per line broadening for 18 species, and finer or wider grids reach hundreds of GB. Neural surrogates trained on these grids are both small and fast, at the cost of a small, measured loss of accuracy. Both approaches share one interface, so users can fit with the bundled surrogates and check selected results against the grids.

: Fitting approaches. Surrogate errors are medians over held-out grid spectra, relative to each spectrum's peak. []{label="tab:approaches"}

| Approach | Disk space | Fit time | Accuracy |
|:---------|:-----------|:---------|:---------|
| Classic slab fitting | small (line lists) | slow: recomputed at every step | exact slab model |
| Grid fitting | tens to hundreds of GB | fast: interpolated | exact at grid points |
| Neural surrogates | 41 MB (46 checkpoints) | fast | 0.01--0.26% spectral error |

# Software design

`DiskMELTS` has four parts: continuum subtraction (`consub`), surrogate fitting (`v2_fitting`), direct grid fitting (`v2_grid_fitting`), and plotting of saved results (`v2_read_plot`). Lower-level modules support retrieval, retraining, and validation. Spectra are given as CSV files or arrays. The exact settings, such as continuum presets, model domains, and network sizes, are described in the online documentation (<https://diskmelts.readthedocs.io>).

**Continuum subtraction.** `consub.calculate` fits a robust baseline through line-free windows, taken from presets for water-rich and carbon-rich disks or supplied by the user, and returns the continuum-subtracted spectrum. A diagnostic plot shows the calibration pixels and poorly anchored regions.

**Slab models.** The training and fitting grids are LTE isothermal slab spectra with optical depth summed over overlapping lines, following IRIS [@Romero24], with line data from HITRAN 2020 [@HITRAN22]. A slab of emitting area $A$ at the reference distance of 140 pc emits

$$F(\lambda) = A \cdot B_\nu(T)\bigl(1 - e^{-\tau(\lambda;\,T,\,N)}\bigr),$$

where $B_\nu$ is the Planck function and $\tau$ is the optical depth. Two grid sets differ in line width: thermal broadening only (`sigma=0`, the default) or an added $2\,\mathrm{km\,s^{-1}}$ turbulent broadening (`sigma=2`).

**Neural surrogates.** For each molecule and wavelength window, two multilayer perceptrons map $(T,\log_{10}N)$ to the peak-normalized spectral shape, in principal-component coefficients [@scikitlearn11; @Halko11], and to the peak flux. The model spectrum is

$$\hat{F}(\lambda;\,T,\,N,\,A) = A \cdot 10^{\hat{p}(T,\,N)} \cdot \hat{s}(\lambda;\,T,\,N).$$

The networks are trained with Adam [@Kingma14]. The package bundles 46 checkpoints, covering 18 species from 4.9 to 26 $\mu$m at both line broadenings, mostly for $T=100$--$1400$ K and $\log_{10}(N/\mathrm{cm^{-2}})=13$--$19.5$. Each checkpoint stores its weights, principal components, scalers, and wavelength axis, so fitting needs no grids.

**Grid fitting.** `fit_observation_grids` reads a local grid of slab spectra, interpolates in $(T,\log_{10}N)$, and searches with differential evolution followed by local refinement. It accepts the same fitting stages as the surrogates and can reach beyond their trained domain, for example optically thick components with $\log_{10}N>19.5$.

**Retrieval.** Surrogate fits draw quasi-random Sobol samples in $(T,\log_{10}N)$ [@SOBOL1967] and refine the best candidates with L-BFGS-B. For each trial, the emitting areas of all components are solved with non-negative least squares, which removes $A$ from the nonlinear search. Isotopologues can share a temperature, or be constrained to be cooler than the main species, and can share an emitting area. `fit_observation` fits ordered stages on successive residuals, for example water first and then carbon-bearing molecules. For both fitting methods, error bars are posterior 16--84% ranges from `emcee` sampling [@emcee13], with the noise inflated for correlated pixels and model mismatch.

**Inspecting residuals.** Fit figures can mark lines that are not modeled, such as H I, H$_2$, HD, and atomic fine-structure lines, from a bundled catalogue. Because mid-infrared OH emission is usually not in LTE, OH is not fitted; instead, LTE OH spectra from a separate 100--6000 K surrogate can be overlaid on the residual for comparison.

# Research impact statement

`DiskMELTS` is in use for JWST/MIRI disk studies. Xie et al. (in prep.) use it to analyze the long-lived "Peter Pan" disks together with complete samples of M-star disks in younger star-forming regions and in Upper Scorpius, fitting all spectra homogeneously with the same continuum and molecular models. Such homogeneous fits across tens of disks are where the speed of the surrogates matters most.

The software is ready for community use. It is distributed on PyPI with its trained models, documented online with tutorials, and checked by continuous integration (the full test suite on Linux and installation tests on macOS and Windows). The repository provides public JWST/MIRI spectra of disks in Upper Scorpius and younger regions, example notebooks that run from installation to fitted results, and workflows that compare fits with published slab-model results.

Benchmarks on grid spectra withheld from training quantify the cost of the surrogates. The median spectral error of each checkpoint is 0.01--0.26% of the spectrum's peak (0.66% for the 19--25 $\mu$m water window), and fits to these spectra recover $T$ with a root-mean-square error of 2--24 K and $\log_{10}N$ within 0.08 dex. The errors are largest for cold, optically thick emission, where $N$ is poorly constrained by any slab fit; there the same fit can be repeated with the grids. The reported error bars do not include systematic errors from the continuum or the slab-model assumptions. With local grids, users can retrain and validate the surrogates for new molecules, domains, or line widths.

# AI usage disclosure

Generative AI tools (Codex 6.1 and Claude Opus 5.5) were used to assist with code review and refactoring, test and documentation updates, and language editing of this paper. All AI-assisted changes were reviewed by the authors and checked with the automated test suite, and the fitting results of the package were checked and cross-matched with published slab-model results. The authors retain responsibility for the software and manuscript and for the correctness of all scientific claims.

# Acknowledgements

This material is based upon work supported by the National Aeronautics and Space Administration (NASA) under Agreement No. 80NSSC21K0593 for the program "Alien Earths".

# References
