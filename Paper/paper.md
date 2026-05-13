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
date: 12 May 2026
bibliography: paper.bib
---

# Statement of Need

The James Webb Space Telescope (JWST) Mid-Infrared Instrument (MIRI) has opened a new window onto the molecular gas reservoir of protoplanetary disks, delivering high-quality mid-infrared spectra for hundreds of sources in nearby star-forming regions [@pontoppidan_jwst_2024; @banzatti_jwst_2023]. Extracting physical parameters — column density $N$, excitation temperature $T$, and emitting area $A$ — from these spectra requires fitting local thermodynamic equilibrium (LTE) slab models to the observed emission features. Several tools are already available to the community for this task. CLIcK [@liu_click_2019] provides a grid-based slab model fitting approach. IRIS [@munoz-romero_iris_2024] is an open-source LTE slab model code that uses JAX-accelerated line-by-line radiative transfer and supports Bayesian retrieval via nested sampling with `dynesty`. 

These tools represent the current state of the art, yet even the fastest of them typically require on the order of one hour to several days of computation time per source when a thorough sampling of the $(N, T, A)$ parameter space is needed. For large JWST surveys targeting tens to hundreds of disks, this computational cost becomes a significant bottleneck, especially when multiple molecules must be fitted and subtracted sequentially.

`DiskMELTS` addresses this bottleneck by replacing the radiative-transfer forward model with a pretrained neural network surrogate, reducing per-source fitting time to under one minute: more than 1000 times faster than traditional approaches, while maintaining retrieval accuracy comparable to full posterior sampling. The package is particularly well suited for researchers who wish to (1) rapidly characterize molecular abundances across a large statistical sample of disks, (2) isolate and subtract water or other molecular line contamination before measuring atomic or ionic line fluxes, or (3) obtain quick parameter estimates to guide more detailed follow-up analysis.

# Background and Methods

## Slab model

`DiskMELTS` trains its surrogate models on LTE isothermal slab spectra that include optical depth effects following IRIS [@munoz-romero_iris_2024]. Each spectral line is described by its line-center optical depth
    $\tau_{0} = \frac{\sqrt{\ln 2}}{4\pi\sqrt{\pi}} \frac{A_{ul}\,N_{\rm mol}\,c^3}{\Delta v\,\nu_{ul}^3} \left(x_l\frac{g_u}{g_l}-x_u\right),$
where $A_{ul}$ is the Einstein $A$ coefficient, $N_{\rm mol}$ is the molecular column density, $c$ is the speed of light, $\Delta v$ is the intrinsic line width (FWHM), $\nu_{ul}$ is the line frequency, $g_u$ and $g_l$ are the upper- and lower-level degeneracies. The fractional level populations $x_i = g_i \exp(-E_i/kT_{\rm ex})/Q(T_{\rm ex})$, with $E_i$ the level energy, $k$ Boltzmann's constant, and $Q(T_{\rm ex})$ the partition function evaluated at the excitation temperature $T_{\rm ex}$. The emergent flux density from a slab of emitting area $A$ at distance is
    $F(\lambda) = A \cdot B_\nu(T_{\rm ex})\bigl(1 - e^{-\tau(\lambda)}\bigr)$
where $B_\nu$ is the Planck function. All molecular line parameters ($A_{ul}$, $E_i$, $g_i$, $\nu_{ul}$) are taken from the HITRAN 2020 database [@gordon_hitran2020_2022], and the intrinsic line width is fixed to pure thermal broadening at the excitation temperature. The three free parameters per molecule are therefore $T_{\rm ex}$, $N_{\rm mol}$, and $A$. Precomputed slab model grids are generated over a regular $(T,\,\log_{10} N)$ grid spanning $T = 100$–$1400$ K and $\log_{10}(N\,/\,{\rm cm}^{-2}) = 13$–$19$.

## Neural surrogate forward model

For each molecule, `DiskMELTS` trains two lightweight multilayer perceptrons (MLPs) that together emulate the slab model forward evaluation at negligible cost:

- **`net_shape`**: maps $(T,\,\log_{10} N)$ to the first $n_{\rm PCA}$ principal-component coefficients of the peak-normalised spectral shape.
- **`net_peak`**: maps $(T,\,\log_{10} N)$ to $\log_{10}$ of the peak flux density (in Jy) at $A = 1$.

Both networks take standardised inputs and are trained on the precomputed grid spectra using mean-squared-error loss with early stopping. The spectral shape is compressed with principal component analysis (PCA) before training ($n_{\rm PCA} = 21$ for H$_2$O; 15 for other molecules), which reduces output dimensionality and regularises the learning problem. The full flux prediction at arbitrary $(T,\,\log_{10} N,\,A)$ is then reconstructed as
$\hat{F}(\lambda;\,T,\,\log_{10} N,\,A) = A \cdot 10^{\hat{p}(T,\,\log_{10} N)} \cdot \hat{s}(\lambda;\,T,\,\log_{10} N),$
where $\hat{p}$ and $\hat{s}$ are the outputs of `net_peak` and the PCA-reconstructed `net_shape`, respectively.

## Parameter retrieval

Retrieval follows a global-to-local search strategy. A large set of $(T,\,\log_{10} N)$ candidates is drawn using Sobol quasi-random sequences [@sobol_1967], and the linear amplitude $A$ for each candidate is solved analytically with non-negative least squares (NNLS). The top distinct candidates by loss are then refined with L-BFGS-B local optimisation. Uncertainty estimates are derived from the spread of the top-ranked solutions, optionally weighted by their likelihood under the assumed noise model.

For observed spectra, `DiskMELTS` applies a sequential fit-and-subtract strategy: H$_2$O is fitted first in a wavelength region where it dominates ($11$–$12\,\mu$m and $16.5$–$18.5\,\mu$m), the best-fit H$_2$O model is subtracted from the observation, and the carbon-bearing molecules (C$_2$H$_2$, HCN, CO$_2$) are then jointly fitted on the residual in the $12$–$16\,\mu$m region. A $3\sigma$ detection screen is applied before each stage so that non-detected molecules are automatically excluded. Two-component H$_2$O fits (warm + hot) are also supported through the same API.

`DiskMELTS` is not limited to the four molecules included in the default distribution. The `Trainmodel` module exposes the full training pipeline: grid loading, pretrain CSV generation, and two-MLP training, so that users with their own slab model grids can train surrogate models for any molecule and integrate them directly into the fitting workflow.

# Acknowledgements


# References
