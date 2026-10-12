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

The James Webb Space Telescope (JWST) is delivering detailed infrared spectra of planet-forming disks around young stars. These spectra contain emission from water and other molecules, allowing astronomers to study the regions where planets form. However, interpreting them with traditional physical models can be computationally expensive. Machine-learning Enabled Line fitting Tool for (Disk) Spectra (`DiskMELTS`) is a Python package for continuum subtraction and rapid molecular fitting with pretrained neural networks. It estimates molecular temperatures, column densities, and emitting areas, and provides tools to inspect and subtract fitted components.

# Statement of Need

The JWST Mid-Infrared Instrument (MIRI) Medium Resolution Spectrograph has opened a new window onto molecular gas in protoplanetary disks, delivering high-quality mid-infrared spectra for large samples of young disks in nearby star-forming regions [@Arulanantham25; @Henning24]. These spectra contain rich emission from H$_2$O, C$_2$H$_2$, HCN, CO$_2$, and other species across the $5$--$28\,\mu$m range, tracing the chemistry and physical conditions of the warm inner disk atmosphere.

Extracting physical parameters from these spectra commonly requires fitting local thermodynamic equilibrium (LTE) slab models to the observed molecular emission. The key parameters for each molecule are the excitation temperature $T$, molecular column density $N$, and emitting area $A$. Even accelerated slab-model fitting can require tens of minutes to several hours per molecule when the $(T,N,A)$ parameter space is explored thoroughly. For JWST surveys targeting tens to hundreds of disks, this cost becomes a major bottleneck, especially when multiple molecules must be fitted and subtracted sequentially.

`DiskMELTS` addresses this bottleneck by using pretrained neural-network surrogates in repeated forward evaluations. It also provides continuum estimation from supplied spectra and reference windows. The target users are astronomers who need to prepare and analyze mid-infrared molecular spectra efficiently and reproducibly. Fit speed and accuracy depend on the selected molecules, wavelength coverage, search settings, and quality of the continuum estimate.

The package is particularly well suited for (1) rapidly characterizing molecular abundances across large statistical samples of disks, (2) isolating and subtracting water or other molecular line contamination before measuring atomic or ionic line fluxes, or (3) obtaining quick parameter estimates to guide more detailed follow-up analysis.

# State of the field

Several tools are already available for fitting molecular emission from protoplanetary disks. CLIcK [@Liu19] provides a grid-based slab-model fitting approach, IRIS [@Romero24] implements JAX-accelerated line-by-line radiative transfer and supports Bayesian retrieval with `dynesty`, and DuCKLinG [@Kaeufer24] combines slab-model fitting with dust models based on precomputed dust and gas models, also with Bayesian retrieval and linear regression. These tools represent the current state of the art for direct physical evaluation and comprehensive modeling.

`DiskMELTS` separates expensive physical model generation from repeated observational fitting through reusable neural-network surrogates. It includes an optional direct-grid fitting path for comparison, along with continuum subtraction as a preparation step. The bundled checkpoints make the surrogate fitting workflow usable without distributing the full slab grids.

# Software design

The package exposes `diskmelts.consub.calculate()` and `diskmelts.consub.plot()` for continuum subtraction and inspection. The `v2_fitting` workflow loads bundled molecular checkpoints, fits one or more components, and saves ordered fit-and-subtract results. `v2_grid_fitting` offers optional direct fitting from local slab grids; `v2_read_plot` reads and plots saved results. Lower-level `fitting`, `trainmodel`, `validation`, and `plotting` modules support retrieval, retraining, synthetic tests, and diagnostic figures. Precomputed LTE slab spectra train compact neural surrogates that are deployed within a global-to-local optimization routine.

## Continuum subtraction

`calculate(wav, flux, error, preset, exclude)` returns wavelength, continuum-subtracted flux, error, fitted continuum, and the baseline before a correction from line-free windows. The shipped `water`, `ttauri`, `carbon_rich`, and `carbon_extended` presets provide candidate windows and curated reference anchors. The water template uses $T=800$ K and $\log_{10}N=18$; the carbon templates use $T=400$ K and $\log_{10}N=16$. Carbon-rich presets also exclude $12.5$--$16.5\,\mu$m. Users can instead supply their own window CSV with `preset=None`, add excluded intervals, or adjust the automatic smoothing widths. A robust local baseline with line rejection is the default; an asymmetric reweighted penalized least-squares option is also available. These windows are candidates, so the fitted continuum should be inspected around broad blends and poorly anchored regions. Returned errors reflect input measurement noise or local estimates, not a full continuum uncertainty budget.

## Training set: Slab model

`DiskMELTS` trains its surrogate models on LTE isothermal slab spectra with optical-depth effects, following the same physical assumptions used by IRIS [@Romero24]. Because nearby molecular transitions can overlap in wavelength, the optical depth is summed over all contributing lines,

$$\tau(\lambda) = \sum_i \tau_{0,i}\exp[-(\lambda-\lambda_{0,i})^2/(2\sigma_\lambda^2)]$$

where $\lambda_{0,i}$ is the rest wavelength of line $i$, $\sigma_\lambda$ is the intrinsic line width, and $\tau_{0,i}$ is the line-center optical depth. For each transition,

$$\tau_{0} = \frac{\sqrt{\ln 2}}{4\pi\sqrt{\pi}} \frac{A_{ul}\,N_{\mathrm{mol}}\,c^3}{\Delta v\,\nu_{ul}^3} (x_l g_u/g_l - x_u)$$

where $A_{ul}$ is the Einstein $A$ coefficient, $N_{\mathrm{mol}}$ is the molecular column density, $c$ is the speed of light, $\Delta v$ is the intrinsic line width, and $\nu_{ul}$ is the line frequency. The fractional level populations are evaluated at the excitation temperature $T_{\mathrm{ex}}$. For the fixed reference distance used by the grid, the emergent flux density from a slab of emitting area $A$ is

$$F(\lambda) = A \cdot B_\nu(T_{\mathrm{ex}})\bigl(1 - e^{-\tau(\lambda)}\bigr)$$

where $B_\nu$ is the Planck function. All molecular line parameters are taken from the HITRAN 2020 database [@HITRAN22]. In the default grids, the intrinsic line width is fixed to pure thermal broadening at the excitation temperature. The three free parameters per molecule are therefore $T_{\mathrm{ex}}$, $N_{\mathrm{mol}}$, and $A$.

Because $A$ is a linear scaling factor, the deployed surrogate grids are restricted over $(T,\log_{10}N)$. The trained domain is $T=100$--$1400$ K, with $\log_{10}(N/\mathrm{cm}^{-2})=13.0$--19.0 for H$_2$O and 13.0--19.5 for the other molecules. The reference distance is 140 pc. At a supplied source distance $d$, the model flux scales as $A(140\,\mathrm{pc}/d)^2$. Custom grids can be trained separately.

The bundled release has five overlapping H$_2$O wavelength checkpoints and one checkpoint for each of ten other molecules. Each checkpoint uses 30 principal components of the normalized spectral shape:

| Molecule | Wavelength coverage ($\mu$m) |
|:---------|:-----------------------------|
| H$_2$O | 4.9--7.4, 7.2--11.2, 9.9--15.1, 13.5--19.2, 19.0--25.0 |
| C$_2$H$_2$, HCN, CO$_2$, $^{13}$C$^{12}$CH$_2$, $^{13}$CO$_2$ | 11.5--17.0 |
| C$_4$H$_2$ | 14.0--18.0 |
| HC$_3$N | 13.0--18.0 |
| C$_2$H$_6$ | 9.0--15.0 |
| C$_2$H$_4$ | 8.0--14.0 |
| CH$_4$ | 5.0--9.0 |

## Neural surrogate forward model

For each molecule, `DiskMELTS` trains two multilayer perceptrons (MLPs) that together emulate the slab-model forward evaluation:

- **`net_shape`** maps $(T,\log_{10}N)$ to the first $n_{\mathrm{PCA}}$ principal-component coefficients of the peak-normalized spectral shape.
- **`net_peak`** maps $(T,\log_{10}N)$ to $\log_{10}$ of the peak flux density at $A=1$.

The training set is generated by evaluating each grid spectrum at $A=1$, peak-normalizing the spectrum so that its maximum absolute flux is unity, and storing both the normalized shape and the original peak flux. This separates the nonlinear spectral-shape problem from the absolute-amplitude problem.

Both networks take standardized inputs and use three fully connected hidden layers with widths 64, 128, and 64 and ReLU activations. PCA reduces the spectral output dimensionality [@scikitlearn11; @Halko11]. The training implementation supports Adam [@Kingma14], validation-loss scheduling, and early stopping. Complete $(T,\log_{10}N)$ grid spectra are held out before fitting the scalers and PCA transform or training the networks; this supports testing on unseen grid points.

The full flux prediction at arbitrary $(T,\log_{10}N,A)$ is reconstructed as

$$\hat{F}(\lambda;\,T,\,\log_{10} N,\,A) = A \cdot 10^{\hat{p}(T,\,\log_{10} N)} \cdot \hat{s}(\lambda;\,T,\,\log_{10} N)$$

where $\hat{p}$ and $\hat{s}$ are the outputs of `net_peak` and the PCA-reconstructed `net_shape`, respectively. Separating spectral shape and peak flux reduces the dynamic range learned by each network. PCA introduces a small approximation but greatly reduces the network output size, while retaining $A$ as a linear parameter reduces the nonlinear search dimension.

## Parameter retrieval

Retrieval follows a global-to-local search strategy. By default, $20{,}000$ candidate $(T,\log_{10}N)$ pairs per molecule are drawn using scrambled Sobol quasi-random sequences [@SOBOL1967]. For each candidate, the linear amplitude $A$ is solved analytically with non-negative least squares (NNLS). When multiple molecules are fitted simultaneously, the amplitudes of all components are solved together in a single NNLS step using the combined model matrix.

The top distinct candidates are then refined with L-BFGS-B local optimization, again solving amplitudes by NNLS at each function evaluation. The best refined solution provides the point estimate. This combines the broad coverage of a global search with the efficiency of local optimization.

For observed spectra, `fit_observation` runs user-defined stages on successive residuals. A common configuration fits H$_2$O first in $11$--$12\,\mu$m and $16.5$--$18.5\,\mu$m windows, then fits carbon-bearing molecules on the residual. A configurable detection screen can skip undetected stage molecules. The order, molecules, and wavelength masks can be adjusted for individual spectra. Results can depend on fitting order and continuum quality.

Each distributed checkpoint contains the neural-network weights, PCA transformation, parameter scalers, and wavelength grid. Therefore, fitting real observations does not require the original pretraining CSV files. Those files and the full slab-model grids are needed only for retraining and complete validation.

# Research impact statement

The package provides held-out slab-grid recovery tests and workflows for comparisons with observed JWST/MIRI spectra. A reconstruction diagnostic is the integrated fractional spectral mismatch,

$$\Delta F/F = \sum_\lambda |F^{\mathrm{true}}_\lambda-\hat{F}_\lambda| / \sum_\lambda |F^{\mathrm{true}}_\lambda|$$

which quantifies flux-level agreement between the surrogate and the original slab model. Its value depends on the checkpoint, position in the trained domain, and test set. Users can run the validation workflow with local grids and examine the saved per-spectrum results.

Comparisons with published LTE slab-model results can be generated from user-supplied tables. As with standard slab retrievals, $N$ and $A$ can exhibit strong degeneracies, particularly for optically thin emission; their product $N\times A$ traces the total number of emitting molecules. For blended C$_2$H$_2$ and HCN emission, multiple parameter combinations can produce comparably good fits. The fitting uncertainty summaries do not include systematic errors from the continuum or slab-model assumptions.

The repository provides checkpoints for 11 species, real-observation examples, validation workflows, and automated tests. The bundled checkpoints and tracked example spectrum allow `examples/dev_v2_realobs.py` and the fitting notebook to run without the larger training datasets. The continuum package bundles its own reference tables and does not need the local `0-Consub/` experiments. With complete local grids, users can run `examples/dev_v2_pt_validation.py` and the training-validation notebook to retrain and validate the surrogates or generate new continuum windows.

# AI usage disclosure

Generative AI tools (Codex 5.5 and Claude Opus 4.7) were used to assist with code review and refactoring, test and documentation updates, and language editing of this paper. The authors retain responsibility for reviewing the software and manuscript and for the correctness of all scientific claims.

# References
