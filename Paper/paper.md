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

The James Webb Space Telescope (JWST) Mid-Infrared Instrument (MIRI) Medium Resolution Spectrograph has opened a new window onto molecular gas in protoplanetary disks, delivering high-quality mid-infrared spectra for large samples of young disks in nearby star-forming regions [@Pontoppidan24; @Banzatti23]. These spectra contain rich emission from H$_2$O, C$_2$H$_2$, HCN, CO$_2$, and other species across the $5$--$28\,\mu$m range, tracing the chemistry and physical conditions of the warm inner disk atmosphere.

Extracting physical parameters from these spectra commonly requires fitting local thermodynamic equilibrium (LTE) slab models to the observed molecular emission. The key parameters for each molecule are the excitation temperature $T$, molecular column density $N$, and emitting area scaling factor $A$. Several tools are already available for this task. CLIcK [@Liu19] provides a grid-based slab-model fitting approach, while IRIS [@Romero24] implements JAX-accelerated line-by-line radiative transfer and supports Bayesian retrieval with `dynesty`.

These tools represent the current state of the art, but even accelerated slab-model fitting can require tens of minutes to several hours per source when the $(T, N, A)$ parameter space is explored thoroughly, and potentially days for full Bayesian posterior sampling. For JWST surveys targeting tens to hundreds of disks, this cost becomes a major bottleneck, especially when multiple molecules must be fitted and subtracted sequentially.

`DiskMELTS` addresses this bottleneck by replacing the radiative-transfer forward model with pretrained neural-network surrogates. A complete retrieval for one molecule in one source typically runs in under one minute on a standard laptop CPU, more than three orders of magnitude faster than conventional approaches while preserving accuracy comparable to classical LTE slab-model fits. The default distribution provides surrogate models for H$_2$O over the $11$--$19\,\mu$m MIRI range, the main C-bearing molecules (C$_2$H$_2$, HCN, CO$_2$) and two main isotopes ($^{13}CCH_2$, $^{13}CO_2$) over the $12$--$16.5\,\mu$m MIRI range, covering $T = 100$--$1400$ K and $\log_{10}(N/{\mathrm{cm}}^{-2}) = 13$--$19$.

The package is particularly well suited for researchers who wish to (1) rapidly characterize molecular abundances across large statistical samples of disks, (2) isolate and subtract water or other molecular line contamination before measuring atomic or ionic line fluxes, or (3) obtain quick parameter estimates to guide more detailed follow-up analysis.

# Background and Methods

`DiskMELTS` is organized into four Python modules: `trainmodel` for surrogate training and slab-grid utilities, `validation` for synthetic recovery tests, `fitting` for spectral retrieval, and `plotting` for diagnostic figures. The default workflow starts from precomputed LTE slab spectra, trains compact neural surrogates for individual molecules, and then uses those surrogates in a global-to-local optimization procedure for observed spectra.

## Training set: Slab model

`DiskMELTS` trains its surrogate models on LTE isothermal slab spectra with optical-depth effects, following the same physical assumptions used by IRIS [@Romero24]. Because nearby molecular transitions can overlap in wavelength, the optical depth is summed over all contributing lines,

$$\tau(\lambda) = \sum_i \tau_{0,i}\exp[-(\lambda-\lambda_{0,i})^2/(2\sigma_\lambda^2)]$$

where $\lambda_{0,i}$ is the rest wavelength of line $i$, $\sigma_\lambda$ is the intrinsic line width, and $\tau_{0,i}$ is the line-center optical depth. For each transition,

$$\tau_{0} = \frac{\sqrt{\ln 2}}{4\pi\sqrt{\pi}} \frac{A_{ul}\,N_{\mathrm{mol}}\,c^3}{\Delta v\,\nu_{ul}^3} (x_l g_u/g_l - x_u)$$

where $A_{ul}$ is the Einstein $A$ coefficient, $N_{\mathrm{mol}}$ is the molecular column density, $c$ is the speed of light, $\Delta v$ is the intrinsic line width (FWHM), $\nu_{ul}$ is the line frequency, $g_u$ and $g_l$ are the upper- and lower-level degeneracies. The fractional level populations $x_i = g_i \exp(-E_i/kT_{\mathrm{ex}})/Q(T_{\mathrm{ex}})$, with $E_i$ the level energy, $k$ Boltzmann's constant, and $Q(T_{\mathrm{ex}})$ the partition function evaluated at the excitation temperature $T_{\mathrm{ex}}$. For the fixed reference distance used by the grid, the emergent flux density from a slab of emitting area $A$ is

$$F(\lambda) = A \cdot B_\nu(T_{\mathrm{ex}})\bigl(1 - e^{-\tau(\lambda)}\bigr)$$

where $B_\nu$ is the Planck function. All molecular line parameters ($A_{ul}$, $E_i$, $g_i$, $\nu_{ul}$) are taken from the HITRAN 2020 database [@HITRAN22]. In the default grids, the intrinsic line width is fixed to pure thermal broadening at the excitation temperature. The three free parameters per molecule are therefore $T_{\mathrm{ex}}$, $N_{\mathrm{mol}}$, and $A$.

Because $A$ is a linear scaling factor, the default slab-model grids are precomputed only over $(T,\,\log_{10} N)$. They span $T = 100$--$1400$ K in 25 K steps and $\log_{10}(N/{\mathrm{cm}}^{-2}) = 13.0$--$19.0$ in 0.25 dex steps. The default wavelength grid covers $11$--$19\,\mu$m at approximately 3000 wavelength points, and the reference distance is fixed to 140 pc. The range was chosen to include the main MIRI molecular features while avoiding shorter-wavelength hot-water lines that may be affected by non-LTE excitation [@Banzatti23]. Longer-wavelength or otherwise customized grids can be trained separately.

Each precomputed grid point is stored as a CSV file named by its temperature and column density, and the training utilities parse those filenames to recover $(T,\log_{10}N)$. For C-bearing molecules, only the $12$--$16.5\,\mu$m region is passed to the training pipeline, excluding long-wavelength channels with little signal for these species. H$_2$O uses the full $11$--$19\,\mu$m range.

## Neural surrogate forward model

For each molecule, `DiskMELTS` trains two lightweight multilayer perceptrons (MLPs) that together emulate the slab model forward evaluation at negligible cost:

- **`net_shape`**: maps $(T,\,\log_{10} N)$ to the first $n_{\mathrm{PCA}}$ principal-component coefficients of the peak-normalized spectral shape.
- **`net_peak`**: maps $(T,\,\log_{10} N)$ to $\log_{10}$ of the peak flux density (in Jy) at $A = 1$.

The training set is generated by evaluating each grid spectrum at $A=1$, peak-normalizing the spectrum so that its maximum absolute flux is unity, and storing both the normalized shape and the original peak flux. This separates the nonlinear spectral-shape problem from the absolute-amplitude problem.

Both networks take standardized inputs and use three fully connected hidden layers with widths 64, 128, and 64 and ReLU activations. The spectral shape is compressed with principal component analysis (PCA) before training, reducing output dimensionality while retaining more than 99.9% of the variance [@scikitlearn11; @Halko11]. The default models use $n_{\mathrm{PCA}} = 21$ for H$_2$O and $n_{\mathrm{PCA}} = 15$ for C$_2$H$_2$, HCN, and CO$_2$. The networks are trained with mean-squared-error loss using Adam [@Kingma14], a learning rate of $10^{-4}$, a batch size of 128, learning-rate reduction on validation-loss plateaus, and early stopping.

The full flux prediction at arbitrary $(T,\,\log_{10} N,\,A)$ is reconstructed as

$$\hat{F}(\lambda;\,T,\,\log_{10} N,\,A) = A \cdot 10^{\hat{p}(T,\,\log_{10} N)} \cdot \hat{s}(\lambda;\,T,\,\log_{10} N)$$

where $\hat{p}$ and $\hat{s}$ are the outputs of `net_peak` and the PCA-reconstructed `net_shape`, respectively.

## Parameter retrieval

Retrieval follows a global-to-local search strategy. By default, $20{,}000$ candidate $(T,\,\log_{10} N)$ pairs per molecule are drawn using scrambled Sobol quasi-random sequences [@SOBOL1967]. For each candidate, the linear amplitude $A$ is solved analytically with non-negative least squares (NNLS). When multiple molecules are fitted simultaneously, the amplitudes of all components are solved together in a single NNLS step using the combined model matrix.

The top distinct candidates are then refined with L-BFGS-B local optimization, again solving amplitudes by NNLS at each function evaluation. The best refined solution provides the point estimate. Uncertainty estimates are derived from the spread of the top-ranked distinct refined solutions, optionally weighted by their likelihood under the assumed noise model.

For observed spectra, `DiskMELTS` applies a sequential fit-and-subtract strategy: H$_2$O is fitted first in wavelength regions where it dominates ($11$--$12\,\mu$m and $16.5$--$18.5\,\mu$m), the best-fit H$_2$O model is subtracted from the observation, and the carbon-bearing molecules (C$_2$H$_2$, HCN, CO$_2$) are then jointly fitted on the residual in the $12$--$16.5\,\mu$m region. A $3\sigma$ detection screen can be applied before each stage so that non-detected molecules are automatically excluded. The sequential order and wavelength masks can be adjusted for individual spectra.

Two-component H$_2$O fits are also supported through the same API. In this mode, the model includes independent warm and hot components with separate $(T,N,A)$ parameters. An ordering constraint, $T_{\mathrm{warm}} \leq T_{\mathrm{hot}}$, reduces label switching, and optional Gaussian temperature penalties can regularize the decomposition when the data do not uniquely separate the two components.

## Validation

The trained surrogates are validated against held-out slab-model grid points and against real JWST/MIRI spectra. On validation spectra withheld from training, `DiskMELTS` recovers the input physical parameters with $R^2 > 0.99$ for all default molecules. The package also computes the integrated fractional spectral mismatch,

$$\Delta F/F = \sum_\lambda |F^{\mathrm{true}}_\lambda-\hat{F}_\lambda| / \sum_\lambda |F^{\mathrm{true}}_\lambda|$$

to quantify flux-level agreement between the surrogate and the original slab model. The trained surrogates achieve a maximum $\Delta F / F \lesssim 5\%$ for each of the molecule. 

Applied to real JWST/MIRI spectra, `DiskMELTS` returns molecular parameters broadly consistent with classical LTE slab-model fitting. As in standard slab retrievals, $N$ and $A$ can be strongly degenerate, especially for those optically thin emissions, while the product $N \times A$, which traces the total number of emitting molecules, is more robustly recovered. For blended C$_2$H$_2$ and HCN emission, multiple parameter combinations can produce comparably good spectra, so the returned parameter uncertainties should be interpreted as including both slab-model degeneracy and surrogate-model approximation error.

`DiskMELTS` is not limited to the four molecules and 2 isotopes included in the default distribution. The `trainmodel` module exposes the full training pipeline: grid loading, pretrain CSV generation, and two-MLP training, so that users with their own slab model grids can train surrogate models for any molecule and integrate them directly into the fitting workflow.

# Acknowledgements


# References
