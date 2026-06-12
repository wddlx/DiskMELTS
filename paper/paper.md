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

# Summary

The James Webb Space Telescope (JWST) is delivering detailed infrared spectra of planet-forming disks around young stars. These spectra contain emission from water and other molecules, allowing astronomers to study the regions where planets form. However, interpreting them with traditional physical models can be computationally expensive. Machine-learning Enabled Line fitting Tool for (Disk) Spectra (`DiskMELTS`) is a Python package that uses pretrained neural networks to fit molecular emission rapidly and estimate molecular temperatures, column densities, and emitting areas. It is designed for large JWST surveys, removal of molecular contamination, and rapid estimates that guide detailed follow-up analysis.

# Statement of Need

The JWST Mid-Infrared Instrument (MIRI) Medium Resolution Spectrograph has opened a new window onto molecular gas in protoplanetary disks, delivering high-quality mid-infrared spectra for large samples of young disks in nearby star-forming regions [@Arulanantham25; @Henning24]. These spectra contain rich emission from H$_2$O, C$_2$H$_2$, HCN, CO$_2$, and other species across the $5$--$28\,\mu$m range, tracing the chemistry and physical conditions of the warm inner disk atmosphere.

Extracting physical parameters from these spectra commonly requires fitting local thermodynamic equilibrium (LTE) slab models to the observed molecular emission. The key parameters for each molecule are the excitation temperature $T$, molecular column density $N$, and emitting area $A$. Even accelerated slab-model fitting can require tens of minutes to several hours per molecule when the $(T,N,A)$ parameter space is explored thoroughly. For JWST surveys targeting tens to hundreds of disks, this cost becomes a major bottleneck, especially when multiple molecules must be fitted and subtracted sequentially.

`DiskMELTS` addresses this bottleneck by replacing the radiative-transfer forward model with pretrained neural-network surrogates. A complete retrieval for one molecule in one source typically runs in under one minute on a standard laptop CPU, more than three orders of magnitude faster than conventional approaches while preserving accuracy comparable to classical LTE slab-model fits. The target users are astronomers who need to analyze continuum-subtracted mid-infrared molecular spectra efficiently and reproducibly.

The package is particularly well suited for (1) rapidly characterizing molecular abundances across large statistical samples of disks, (2) isolating and subtracting water or other molecular line contamination before measuring atomic or ionic line fluxes, or (3) obtaining quick parameter estimates to guide more detailed follow-up analysis.

# State of the field

Several tools are already available for fitting molecular emission from protoplanetary disks. CLIcK [@Liu19] provides a grid-based slab-model fitting approach, IRIS [@Romero24] implements JAX-accelerated line-by-line radiative transfer and supports Bayesian retrieval with `dynesty`, and DuCKLinG [@Kaeufer24] combines slab-model fitting with dust models based on precomputed dust and gas models, also with Bayesian retrieval and linear regression. These tools represent the current state of the art for direct physical evaluation and comprehensive modeling.

`DiskMELTS` is complementary to these packages. Its unique contribution is to separate expensive physical model generation from repeated observational fitting through reusable neural-network surrogates. This prioritizes rapid analysis of continuum-subtracted spectra across large samples, while the existing tools prioritize direct radiative-transfer evaluation or joint gas-and-dust modeling. A focused package also provides a clear route for distributing pretrained models and training additional molecules.

# Software design

The `DiskMELTS` framework is implemented across four Python modules for continuum-subtracted spectra [^1]. The core workflow relies on `fitting` to execute retrievals and `plotting` to generate diagnostic figures. For custom models, `trainmodel` provides surrogate-training and slab-grid utilities, while `validation` performs synthetic recovery tests. Precomputed LTE slab spectra train compact neural surrogates that are deployed within a global-to-local optimization routine.

[^1]: Continuum subtraction tools are not currently implemented in the package.

## Training set: Slab model

`DiskMELTS` trains its surrogate models on LTE isothermal slab spectra with optical-depth effects, following the same physical assumptions used by IRIS [@Romero24]. Because nearby molecular transitions can overlap in wavelength, the optical depth is summed over all contributing lines,

$$\tau(\lambda) = \sum_i \tau_{0,i}\exp[-(\lambda-\lambda_{0,i})^2/(2\sigma_\lambda^2)]$$

where $\lambda_{0,i}$ is the rest wavelength of line $i$, $\sigma_\lambda$ is the intrinsic line width, and $\tau_{0,i}$ is the line-center optical depth. For each transition,

$$\tau_{0} = \frac{\sqrt{\ln 2}}{4\pi\sqrt{\pi}} \frac{A_{ul}\,N_{\mathrm{mol}}\,c^3}{\Delta v\,\nu_{ul}^3} (x_l g_u/g_l - x_u)$$

where $A_{ul}$ is the Einstein $A$ coefficient, $N_{\mathrm{mol}}$ is the molecular column density, $c$ is the speed of light, $\Delta v$ is the intrinsic line width, and $\nu_{ul}$ is the line frequency. The fractional level populations are evaluated at the excitation temperature $T_{\mathrm{ex}}$. For the fixed reference distance used by the grid, the emergent flux density from a slab of emitting area $A$ is

$$F(\lambda) = A \cdot B_\nu(T_{\mathrm{ex}})\bigl(1 - e^{-\tau(\lambda)}\bigr)$$

where $B_\nu$ is the Planck function. All molecular line parameters are taken from the HITRAN 2020 database [@HITRAN22]. In the default grids, the intrinsic line width is fixed to pure thermal broadening at the excitation temperature. The three free parameters per molecule are therefore $T_{\mathrm{ex}}$, $N_{\mathrm{mol}}$, and $A$.

Because $A$ is a linear scaling factor, the default slab-model grids are precomputed only over $(T,\log_{10}N)$. They span $T=100$--$1400$ K in 25 K steps and $\log_{10}(N/\mathrm{cm}^{-2})=13.0$--19.0 in 0.125 dex steps. The wavelength grids cover $11$--$19\,\mu$m for H$_2$O and $11$--$17.5\,\mu$m for the carbon-bearing molecules. The reference distance is fixed to 140 pc. Longer-wavelength or otherwise customized grids can be trained separately.

The spectra are compressed using principal component analysis (PCA). The current models use 21 components for H$_2$O and 15 components for the other molecules:

| Molecule | Components | Retained variance | Wavelength range ($\mu$m) |
|:---------|-----------:|------------------:|---------------------------:|
| H$_2$O | 21 | 99.981% | 11--19 |
| C$_2$H$_2$ | 15 | 99.991% | 11--17.5 |
| $^{13}$C$^{12}$CH$_2$ | 15 | 99.964% | 11--17.5 |
| HCN | 15 | 99.979% | 11--17.5 |
| CO$_2$ | 15 | 99.988% | 11--17.5 |
| $^{13}$CO$_2$ | 15 | 99.981% | 11--17.5 |

## Neural surrogate forward model

For each molecule, `DiskMELTS` trains two lightweight multilayer perceptrons (MLPs) that together emulate the slab-model forward evaluation at negligible cost:

- **`net_shape`** maps $(T,\log_{10}N)$ to the first $n_{\mathrm{PCA}}$ principal-component coefficients of the peak-normalized spectral shape.
- **`net_peak`** maps $(T,\log_{10}N)$ to $\log_{10}$ of the peak flux density at $A=1$.

The training set is generated by evaluating each grid spectrum at $A=1$, peak-normalizing the spectrum so that its maximum absolute flux is unity, and storing both the normalized shape and the original peak flux. This separates the nonlinear spectral-shape problem from the absolute-amplitude problem.

Both networks take standardized inputs and use three fully connected hidden layers with widths 64, 128, and 64 and ReLU activations. The spectral shape is compressed with PCA before training, reducing output dimensionality while retaining more than 99.9% of the variance [@scikitlearn11; @Halko11]. The networks are trained with mean-squared-error loss using Adam [@Kingma14], a learning rate of $10^{-4}$, a batch size of 128, learning-rate reduction on validation-loss plateaus, and early stopping.

The full flux prediction at arbitrary $(T,\log_{10}N,A)$ is reconstructed as

$$\hat{F}(\lambda;\,T,\,\log_{10} N,\,A) = A \cdot 10^{\hat{p}(T,\,\log_{10} N)} \cdot \hat{s}(\lambda;\,T,\,\log_{10} N)$$

where $\hat{p}$ and $\hat{s}$ are the outputs of `net_peak` and the PCA-reconstructed `net_shape`, respectively. Separating spectral shape and peak flux reduces the dynamic range learned by each network. PCA introduces a small approximation but greatly reduces the network output size, while retaining $A$ as a linear parameter reduces the nonlinear search dimension.

## Parameter retrieval

Retrieval follows a global-to-local search strategy. By default, $20{,}000$ candidate $(T,\log_{10}N)$ pairs per molecule are drawn using scrambled Sobol quasi-random sequences [@SOBOL1967]. For each candidate, the linear amplitude $A$ is solved analytically with non-negative least squares (NNLS). When multiple molecules are fitted simultaneously, the amplitudes of all components are solved together in a single NNLS step using the combined model matrix.

The top distinct candidates are then refined with L-BFGS-B local optimization, again solving amplitudes by NNLS at each function evaluation. The best refined solution provides the point estimate. This combines the broad coverage of a global search with the efficiency of local optimization.

For observed spectra, `DiskMELTS` applies a sequential fit-and-subtract strategy: H$_2$O is fitted first in wavelength regions where it dominates ($11$--$12\,\mu$m and $16.5$--$18.5\,\mu$m), the best-fit H$_2$O model is subtracted from the observation, and the carbon-bearing molecules are then jointly fitted on the residual. A $3\sigma$ detection screen can be applied before each stage so that non-detected molecules are automatically excluded. The sequential order and wavelength masks can be adjusted for individual spectra. This approach is faster than a fully joint nonlinear retrieval, although results can depend on the fitting order and the quality of continuum subtraction.

Each distributed checkpoint contains the neural-network weights, PCA transformation, parameter scalers, and wavelength grid. Therefore, fitting real observations does not require the original pretraining CSV files. Those files and the full slab-model grids are needed only for retraining and complete validation.

# Research impact statement

The trained surrogates are validated against held-out slab-model grid points and against real JWST/MIRI spectra. On validation spectra withheld from training, `DiskMELTS` recovers the input physical parameters with $R^2>0.99$ for all default molecules. The package also computes the integrated fractional spectral mismatch,

$$\Delta F/F = \sum_\lambda |F^{\mathrm{true}}_\lambda-\hat{F}_\lambda| / \sum_\lambda |F^{\mathrm{true}}_\lambda|$$

to quantify flux-level agreement between the surrogate and the original slab model. The trained surrogates achieve a maximum $\Delta F/F\lesssim5\%$ for each molecule.

When applied to real JWST/MIRI spectra, `DiskMELTS` yields molecular parameters consistent with classical LTE slab-model fitting, typically agreeing within $\sim0.5$ dex for $N\times A$ and $<150$ K for $T$ relative to literature benchmarks [@Romero24; Xie et al. in prep.; Raul et al. in prep.]. As with standard slab retrievals, $N$ and $A$ can exhibit strong degeneracies, particularly for optically thin emission; however, their product $N\times A$, which traces the total number of emitting molecules, is more robustly recovered. For blended C$_2$H$_2$ and HCN emission, multiple parameter combinations can produce comparably good fits.

The repository provides pretrained models for four molecules and two isotopes, real-observation examples, validation workflows, and automated tests. The tracked checkpoints and example spectrum allow `realobs.py` and the fitting notebook to run without the larger training datasets. With the complete local grids, users can run `pt_validation.py` and the training-validation notebook, retrain the surrogates, and extend the package to new molecules or parameter ranges.

# AI usage disclosure

Generative AI tools (Codex 5.5 and Claude Opus 4.7) were used to assist with code review and refactoring, test and documentation updates, and language editing of this paper. The authors reviewed all suggested changes, verified software behavior with the automated tests and example workflows, checked numerical statements against the stored models and validation outputs, and retain responsibility for the correctness of the software and manuscript.

# References
