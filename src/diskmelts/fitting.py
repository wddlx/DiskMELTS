"""
diskmelts/fitting.py — Spectral fitting with pretrained forward surrogate models.

This module provides all fitting and model-loading utilities.

Forward model convention:
    flux = A * peak(T, logN) * shape(T, logN)

Global-to-local search strategy in fit_molecules:
1. Draw Sobol or Latin-hypercube samples over all nonlinear parameters (T, logN).
2. For each nonlinear trial, solve all linear amplitudes A together with NNLS.
3. Keep the best distinct candidates and refine them with L-BFGS-B.

Public functions
----------------
    load_models          : load per-molecule .pt checkpoints from disk
    generate_spectrum    : evaluate (T, logN, A) → flux via the two-MLP forward model
    fit_nested           : single-molecule retrieval via random-restart L-BFGS-B + NNLS
    fit_molecules        : multi-molecule global Sobol search + L-BFGS-B refinement
    load_observed_spectrum: load a continuum-subtracted spectrum from CSV
    detect_stage_molecules: screen for molecular detections above a noise threshold
    save_fit_outputs     : write best-fit CSV, top-list CSV, and spectrum to disk
    save_running_spectrum: write residual / running-model spectrum to disk
    print_fit_params     : pretty-print best-fit parameters and uncertainties
    save_fitted_comparison: upsert best-fit parameters into a comparison CSV
"""

import os
import numpy as np
import pandas as pd
import torch
import scipy.optimize
from scipy.stats import qmc

from diskmelts.trainmodel import pretrain_forward_model   # load-only (n_epochs=0)


# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

_DEFAULT_T_BOUNDS    = (200.0, 1400.0)
_DEFAULT_LOGN_BOUNDS = (14.0,  19.0)
_DEFAULT_LOGA_BOUNDS = None


# ---------------------------------------------------------------------------
# Internal spectral helpers (forward model evaluation)
# ---------------------------------------------------------------------------

def _wav_mask(wav, ranges):
    """Boolean mask for wavelengths within any of the given (lo, hi) ranges."""
    if isinstance(ranges[0], (int, float)):
        ranges = [ranges]
    mask = np.zeros(len(wav), dtype=bool)
    for lo, hi in ranges:
        mask |= (wav >= lo) & (wav <= hi)
    return mask


def _mol_flux_on_wav(pretrained_mol, T, logN, A, wav_grid):
    """
    Evaluate one molecule's two-MLP forward model at (T, logN, A).

    The model is evaluated on its native wavelength axis then interpolated onto
    wav_grid, so wav_grid can have any sampling.

    Args:
        pretrained_mol (dict): single-molecule entry from load_models
        T (float): temperature in K
        logN (float): log10 column density
        A (float): linear area-scaling factor
        wav_grid (np.ndarray): output wavelength grid in µm (any sampling)

    Returns:
        (np.ndarray): model flux on wav_grid (Jy)
    """
    net_shape   = pretrained_mol['net_shape']
    net_peak    = pretrained_mol['net_peak']
    xp_sc       = pretrained_mol['xp_sc']
    yp_sc_shape = pretrained_mol['yp_sc_shape']
    yp_sc_peak  = pretrained_mol['yp_sc_peak']
    pca         = pretrained_mol.get('pca', None)
    wav_net     = pretrained_mol['wav']
    device      = next(net_shape.parameters()).device

    T_s    = float((T    - xp_sc.mean_[0]) / xp_sc.scale_[0])
    logN_s = float((logN - xp_sc.mean_[1]) / xp_sc.scale_[1])
    x_s    = torch.tensor([[T_s, logN_s]], dtype=torch.float32, device=device)

    net_shape.eval()
    net_peak.eval()
    with torch.no_grad():
        z_shape_s = net_shape(x_s)[0].cpu().numpy()
        z_peak_s  = net_peak(x_s)[0].cpu().numpy()

    shape  = z_shape_s * yp_sc_shape.scale_ + yp_sc_shape.mean_
    log10p = float((z_peak_s * yp_sc_peak.scale_ + yp_sc_peak.mean_)[0])
    if pca is not None:
        shape = pca.inverse_transform(shape[np.newaxis])[0]

    flux_net = A * (10.0 ** log10p) * np.maximum(shape, 0.0)
    return np.interp(wav_grid, wav_net, flux_net, left=0.0, right=0.0)


# ---------------------------------------------------------------------------
# Public: load models and generate spectra
# ---------------------------------------------------------------------------

def load_models(model_paths, pretrain_csv_paths, wav_ranges, n_pca=None, device=None):
    """
    Load pretrained per-molecule forward models from disk.

    Args:
        model_paths (dict): mol -> path to the .pt checkpoint file
        pretrain_csv_paths (dict): mol -> path to the pretrain CSV used during
                                   training (needed to reconstruct scalers and
                                   wavelength axis)
        wav_ranges (dict): mol -> (lo_µm, hi_µm); must match wav_range used
                           in trainmodel.pretrain_forward_model
        n_pca (int or dict or None): number of PCA components; must match what
                                     was used during training. Pass a dict
                                     {mol: n} for per-molecule values, or a
                                     single int for all molecules.
        device (torch.device or None): auto-selects CUDA if available

    Returns:
        (dict): mol -> pretrained dict with keys
                'net_shape', 'net_peak', 'xp_sc', 'yp_sc_shape',
                'yp_sc_peak', 'pca', 'wav'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained = {}
    for mol in model_paths:
        mol_n_pca = n_pca[mol] if isinstance(n_pca, dict) else n_pca
        kw = {} if mol_n_pca is None else {'n_pca': mol_n_pca}
        pretrained[mol] = pretrain_forward_model(
            mol=mol,
            pretrain_csv=pretrain_csv_paths[mol],
            wav_range=wav_ranges[mol],
            device=device,
            n_epochs=0,
            model_path=model_paths[mol],
            seed=42,
            **kw,
        )
        print(f'  Loaded {mol} from {model_paths[mol]}')
    return pretrained


def generate_spectrum(T, logN, A, pretrained_mol, obs_wav=None):
    """
    Generate a model spectrum for one molecule using the two-MLP forward model.

    Args:
        T (float): temperature in K
        logN (float): log10 column density
        A (float): linear area-scaling factor
        pretrained_mol (dict): single-molecule pretrained dict, e.g. pretrained['H2O']
        obs_wav (np.ndarray or None): output wavelength grid in µm; if None,
                                      returns flux on the model's native grid

    Returns:
        (np.ndarray): model flux in Jy on obs_wav (or native grid if None)
    """
    if obs_wav is None:
        obs_wav = pretrained_mol['wav']
    return _mol_flux_on_wav(pretrained_mol, T, logN, A, obs_wav)


def fit_nested(
    obs_wav,
    obs_flux,
    mol,
    pretrained,
    fit_ranges,
    T_bounds=None,
    logN_bounds=None,
    loga_bounds=None,
    n_restarts=20,
    n_top=1,
    seed=None,
):
    """
    Retrieve (T, logN, A) for one molecule via nested L-BFGS-B + NNLS optimisation.

    Outer loop: scipy L-BFGS-B over (T, logN).
    Inner loop: scipy NNLS solves for A (enforces A ≥ 0).

    Args:
        obs_wav (np.ndarray): observed wavelength axis in µm
        obs_flux (np.ndarray): observed flux in Jy
        mol (str): molecule name, e.g. 'H2O', 'C2H2'
        pretrained (dict): output of load_models
        fit_ranges: (lo, hi) or [(lo1,hi1), (lo2,hi2)] µm
        T_bounds (tuple or None): (T_min, T_max) K (default (200, 1400))
        logN_bounds (tuple or None): (logN_min, logN_max) (default (14, 19))
        loga_bounds (tuple or None): (log10A_min, log10A_max) (default None = no upper limit)
        n_restarts (int): number of random L-BFGS-B restarts (default 20)
        n_top (int): if > 1, include top-n restart solutions in 'top_list'
        seed (int or None): random seed

    Returns:
        (dict) with keys:
            'params'     : {'T', 'logN', 'A'} — best-fit parameters
            'model_flux' : np.ndarray on obs_wav — model spectrum
            'residual'   : np.ndarray on obs_wav — obs_flux minus model_flux
            'top_list'   : list of dicts (only present when n_top > 1)
    """
    obs_wav  = np.asarray(obs_wav,  dtype=np.float64)
    obs_flux = np.asarray(obs_flux, dtype=np.float64)

    if T_bounds    is None: T_bounds    = _DEFAULT_T_BOUNDS
    if logN_bounds is None: logN_bounds = _DEFAULT_LOGN_BOUNDS

    if loga_bounds is not None:
        _A_lo = 10.0 ** loga_bounds[0]
        _A_hi = 10.0 ** loga_bounds[1]
    else:
        _A_lo, _A_hi = 0.0, np.inf

    rng = np.random.default_rng(seed)

    fit_mask = _wav_mask(obs_wav, fit_ranges)
    wav_fit  = obs_wav[fit_mask]
    obs_fit  = obs_flux[fit_mask]

    def _nnls_step(x0):
        spec = _mol_flux_on_wav(pretrained[mol], x0[0], x0[1], 1.0, wav_fit)
        if np.isfinite(_A_hi):
            res = scipy.optimize.lsq_linear(spec[:, None], obs_fit,
                                            bounds=([_A_lo], [_A_hi]))
            A = float(res.x[0])
        else:
            A_vec, _ = scipy.optimize.nnls(spec[:, None], obs_fit)
            A        = float(A_vec[0])
        resid = obs_fit - spec * A
        return float(np.sum(resid ** 2)), A

    bounds  = [T_bounds, logN_bounds]
    all_res = []
    for _ in range(n_restarts):
        x0 = np.array([rng.uniform(*T_bounds), rng.uniform(*logN_bounds)])
        opt = scipy.optimize.minimize(
            lambda p: _nnls_step(p)[0], x0, method='L-BFGS-B', bounds=bounds)
        loss, A = _nnls_step(opt.x)
        all_res.append((loss, opt.x.copy(), A))
    all_res.sort(key=lambda t: t[0])

    _, best_x, best_A = all_res[0]
    params     = {'T': float(best_x[0]), 'logN': float(best_x[1]), 'A': float(best_A)}
    model_flux = _mol_flux_on_wav(pretrained[mol], params['T'], params['logN'],
                                   params['A'], obs_wav)
    result = {
        'mol':        mol,
        'params':     params,
        'model_flux': model_flux,
        'residual':   obs_flux - model_flux,
    }
    if n_top > 1:
        top_list = []
        for rank, (loss, x, A) in enumerate(all_res[:n_top]):
            top_list.append({
                'rank': rank + 1, 'mse_loss': loss,
                'T': float(x[0]), 'logN': float(x[1]), 'A': float(A),
            })
        result['top_list'] = top_list
    return result


def _as_list(x):
    """Return x as a list, treating strings as one item."""
    if isinstance(x, str):
        return [x]
    return list(x)


def _range_list(ranges):
    """Normalize a single (lo, hi) range or a list of ranges."""
    if isinstance(ranges[0], (int, float)):
        return [tuple(ranges)]
    return [tuple(r) for r in ranges]


def _union_ranges(ranges):
    """Deduplicate wavelength ranges while preserving order."""
    out = []
    seen = set()
    for r in ranges:
        key = tuple(r)
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def _build_components(molecules, h2o_components=1, component_names=None):
    """
    Expand molecule names into fitted components.

    Each component has a unique label and a model_mol key.  For two-component
    water, both components use model_mol='H2O' but have labels H2O_1/H2O_2.
    """
    molecules = _as_list(molecules)
    components = []
    for mol in molecules:
        if mol == 'H2O' and h2o_components == 2:
            components.extend([
                {'label': 'H2O_1', 'model_mol': 'H2O'},
                {'label': 'H2O_2', 'model_mol': 'H2O'},
            ])
        else:
            components.append({'label': mol, 'model_mol': mol})

    if component_names is not None:
        component_names = _as_list(component_names)
        if len(component_names) != len(components):
            raise ValueError('component_names must match the number of fitted components')
        for comp, label in zip(components, component_names):
            comp['label'] = label
    return components


def _fit_ranges_for_components(fit_ranges, components):
    """
    Build a joint fitting range from either a shared range/list or mol -> ranges.
    """
    if isinstance(fit_ranges, dict):
        ranges = []
        for comp in components:
            label = comp['label']
            mol = comp['model_mol']
            if label in fit_ranges:
                ranges.extend(_range_list(fit_ranges[label]))
            elif mol in fit_ranges:
                ranges.extend(_range_list(fit_ranges[mol]))
            else:
                raise KeyError(f'fit_ranges has no entry for {label!r} or {mol!r}')
        return _union_ranges(ranges)
    return _range_list(fit_ranges)


def _draw_unit_samples(n_samples, n_dim, method='sobol', seed=None):
    """Draw approximately space-filling samples in [0, 1]^n_dim."""
    method = method.lower()
    if method == 'sobol':
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        sampler = qmc.Sobol(d=n_dim, scramble=True, seed=seed)
        return sampler.random_base2(m)[:n_samples]
    if method in ('latin', 'lhs', 'latin_hypercube'):
        sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
        return sampler.random(n_samples)
    if method == 'random':
        return np.random.default_rng(seed).random((n_samples, n_dim))
    raise ValueError("sample_method must be 'sobol', 'latin_hypercube', or 'random'")


def _scale_unit_samples(unit, bounds):
    """Scale unit-cube samples to physical bounds."""
    lows = np.array([b[0] for b in bounds], dtype=np.float64)
    highs = np.array([b[1] for b in bounds], dtype=np.float64)
    return lows + unit * (highs - lows)


def _is_ordered_water(params, components):
    """Require increasing T for duplicated H2O components to reduce label switching."""
    h2o_idx = [i for i, c in enumerate(components) if c['model_mol'] == 'H2O']
    if len(h2o_idx) < 2:
        return True
    temps = [params[2 * i] for i in h2o_idx]
    return all(t1 <= t2 for t1, t2 in zip(temps[:-1], temps[1:]))


def _distinct_solutions(solutions, n_keep, components, min_distance=0.03):
    """
    Keep top solutions that are separated in normalized nonlinear parameter space.
    """
    if n_keep <= 0:
        return []
    kept = []
    for sol in solutions:
        if not kept:
            kept.append(sol)
            continue
        p = sol['x_scaled']
        distances = [np.linalg.norm(p - k['x_scaled']) / np.sqrt(len(p)) for k in kept]
        if min(distances) >= min_distance:
            kept.append(sol)
        if len(kept) >= n_keep:
            break
    return kept


def _weighted_quantile(values, quantiles, weights=None):
    """Compute weighted quantiles for one-dimensional values."""
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if weights is None:
        return np.quantile(values, quantiles)

    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0 or weights.size == 0 or np.sum(weights) <= 0:
        return np.full(len(quantiles), np.nan)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf = cdf / cdf[-1]
    return np.interp(quantiles, cdf, values)


def _top_list_uncertainty(top_list, sigma=None):
    """
    Summarize parameter uncertainty from retained top solutions.

    If sigma is provided, solutions are weighted as exp(-0.5 * delta_sse/sigma^2).
    Otherwise the 16/50/84 percentiles are unweighted.  This is an exploration
    uncertainty from the global search, not a full posterior sample.
    """
    if not top_list:
        return {}

    if sigma is not None and sigma > 0:
        losses = np.array([s['mse_loss'] for s in top_list], dtype=np.float64)
        delta = losses - np.nanmin(losses)
        weights = np.exp(-0.5 * delta / sigma ** 2)
    else:
        weights = None

    labels = list(top_list[0]['params'].keys())
    uncertainty = {}
    for label in labels:
        uncertainty[label] = {}
        for key in ('T', 'logN', 'log10A'):
            vals = [s['params'][label][key] for s in top_list if label in s['params']]
            q16, q50, q84 = _weighted_quantile(vals, [0.16, 0.50, 0.84], weights)
            uncertainty[label][key] = {
                'q16': float(q16),
                'q50': float(q50),
                'q84': float(q84),
                'minus': float(q50 - q16),
                'plus': float(q84 - q50),
            }
    return uncertainty


def fit_molecules(
    obs_wav,
    obs_flux,
    mol,
    pretrained,
    fit_ranges,
    h2o_components=1,
    component_names=None,
    T_bounds=None,
    logN_bounds=None,
    loga_bounds=None,
    n_samples=20000,
    sample_method='sobol',
    n_refine=32,
    n_top=10,
    distinct_min_distance=0.03,
    sigma=None,
    seed=None,
    verbose=True,
    T_prior=None,
    T_prior_log10=False,
):
    """
    Fit one or more selected molecules with global screening + local optimisation.

    Nonlinear parameters are (T, logN) for every fitted component.  Linear
    amplitudes A are solved together at every trial.  This is usually better
    than pure random L-BFGS-B restarts because the initial candidates cover the
    allowed parameter volume more evenly.

    Args:
        obs_wav (np.ndarray): observed wavelength axis in µm.
        obs_flux (np.ndarray): observed flux in Jy.
        mol (str or list[str]): molecule(s) to fit, e.g. 'H2O' or
            ['C2H2', 'HCN', 'CO2'].
        pretrained (dict): mol -> pretrained surrogate dict from load_models.
        fit_ranges: shared (lo, hi), shared list of ranges, or dict mapping
            molecule/component labels to ranges.  Dict ranges are unioned for a
            joint fit.
        h2o_components (int): 1 or 2.  If 2 and molecules contains 'H2O', two
            H2O components are fitted using the same pretrained['H2O'] model.
        component_names (list[str] or None): optional custom labels for fitted
            components.  Useful for names like ['H2O_warm', 'H2O_hot'].
        T_bounds (tuple or dict or None): default (200, 1400).  A dict can use
            molecule names or component labels.
        logN_bounds (tuple or dict or None): default (14, 19).  A dict can use
            molecule names or component labels.
        loga_bounds (tuple or None): bounds on log10(A).  None means A >= 0
            with no upper bound; otherwise bounded least squares is used.
        n_samples (int): number of global screening samples.
        sample_method (str): 'sobol', 'latin_hypercube', or 'random'.
        n_refine (int): number of distinct global candidates to refine locally.
        n_top (int): number of ranked refined solutions to return.
        distinct_min_distance (float): normalized distance used to avoid
            refining many nearly identical global candidates.
        sigma (float or None): optional noise std in Jy.  If provided,
            top-solution uncertainties are likelihood-weighted by SSE.  If
            None, unweighted top-solution percentiles are returned.
        seed (int or None): random seed.
        verbose (bool): print progress and best losses.
        T_prior (dict or None): Gaussian prior on temperature for each
            component, keyed by component label.  Each value is a tuple
            (T_center, T_std) in Kelvin.  Example::

                T_prior = {
                    'H2O_warm': (400.0, 100.0),
                    'H2O_hot':  (800.0, 100.0),
                }

            The prior is added to the SSE as a penalty term scaled by
            ``sigma**2``, so both data and prior are in the same Jy² units:

                loss = SSE + sigma² × Σ ((T_i − T0_i) / σ_Ti)²

            ``sigma`` must be provided when ``T_prior`` is set; if it is None
            the prior term is skipped with a warning.
        T_prior_log10 (bool): if True, the prior is applied in log10(T) space
            and the values in ``T_prior`` are interpreted as (log10_T_center,
            log10_T_std).  Example::

                T_prior = {
                    'H2O_warm': (2.6, 0.2),   # log10(T_warm) ~ N(2.6, 0.2)
                    'H2O_hot':  (2.9, 0.1),   # log10(T_hot)  ~ N(2.9, 0.1)
                }
                T_prior_log10 = True

    Returns:
        dict with keys:
            'components'       : list of component metadata.
            'params'           : label -> {'T', 'logN', 'A', 'log10A', 'model_mol'}.
            'model_flux'       : total model flux on obs_wav.
            'component_fluxes' : label -> model flux on obs_wav.
            'residual'         : obs_flux - model_flux.
            'fit_ranges'       : actual joint ranges used.
            'top_list'         : ranked refined solutions.
            'global_top'       : best unrefined global candidates.
            'uncertainty'      : label -> parameter percentile summary.
    """
    if h2o_components not in (1, 2):
        raise ValueError('h2o_components must be 1 or 2')

    obs_wav = np.asarray(obs_wav, dtype=np.float64)
    obs_flux = np.asarray(obs_flux, dtype=np.float64)

    if T_bounds is None:
        T_bounds = _DEFAULT_T_BOUNDS
    if logN_bounds is None:
        logN_bounds = _DEFAULT_LOGN_BOUNDS
    if loga_bounds is None:
        loga_bounds = _DEFAULT_LOGA_BOUNDS

    components = _build_components(mol, h2o_components, component_names)
    for comp in components:
        if comp['model_mol'] not in pretrained:
            raise KeyError(f"pretrained has no model for {comp['model_mol']!r}")

    joint_ranges = _fit_ranges_for_components(fit_ranges, components)
    fit_mask = _wav_mask(obs_wav, joint_ranges)
    wav_fit = obs_wav[fit_mask]
    obs_fit = obs_flux[fit_mask]
    if len(wav_fit) == 0:
        raise ValueError('fit_ranges select zero observed wavelength pixels')

    def _bounds_for(comp, bound_spec, default):
        if isinstance(bound_spec, dict):
            if comp['label'] in bound_spec:
                return tuple(bound_spec[comp['label']])
            if comp['model_mol'] in bound_spec:
                return tuple(bound_spec[comp['model_mol']])
            return default
        return tuple(bound_spec)

    nonlinear_bounds = []
    for comp in components:
        nonlinear_bounds.append(_bounds_for(comp, T_bounds, _DEFAULT_T_BOUNDS))
        nonlinear_bounds.append(_bounds_for(comp, logN_bounds, _DEFAULT_LOGN_BOUNDS))

    if loga_bounds is None:
        A_bounds = (0.0, np.inf)
    else:
        A_bounds = (10.0 ** float(loga_bounds[0]), 10.0 ** float(loga_bounds[1]))

    def _matrix_for(params):
        cols = []
        for i, comp in enumerate(components):
            T = params[2 * i]
            logN = params[2 * i + 1]
            cols.append(_mol_flux_on_wav(pretrained[comp['model_mol']], T, logN, 1.0, wav_fit))
        return np.column_stack(cols)

    def _solve_amplitudes(params):
        if not _is_ordered_water(params, components):
            return np.inf, np.zeros(len(components)), None
        M = _matrix_for(params)
        if np.isfinite(A_bounds[1]):
            res = scipy.optimize.lsq_linear(
                M,
                obs_fit,
                bounds=(np.full(len(components), A_bounds[0]),
                        np.full(len(components), A_bounds[1])),
            )
            A = np.asarray(res.x, dtype=np.float64)
        else:
            A, _ = scipy.optimize.nnls(M, obs_fit)
        residual = obs_fit - M @ A
        loss = float(np.sum(residual ** 2))
        return loss, A, M

    if T_prior is not None and (sigma is None or sigma <= 0):
        print('  Warning: T_prior supplied but sigma is None or ≤ 0 — prior skipped.')
    _prior_scale = (sigma ** 2
                    if (T_prior is not None and sigma is not None and sigma > 0)
                    else 0.0)

    def _prior_penalty(x):
        """Sum of (T_i − T0_i)² / σ_Ti² for components with a T prior.

        When T_prior_log10 is True the penalty is evaluated in log10(T) space
        and T_prior values are (log10_T_center, log10_T_std).
        """
        if T_prior is None or _prior_scale == 0.0:
            return 0.0
        penalty = 0.0
        for i, comp in enumerate(components):
            label = comp['label']
            if label in T_prior:
                center, std = T_prior[label]
                T_val = x[2 * i]
                t = np.log10(max(T_val, 1e-10)) if T_prior_log10 else T_val
                penalty += ((t - center) / std) ** 2
        return penalty

    def _penalized_loss(x):
        """SSE + Gaussian prior penalty, both in Jy²."""
        sse, A, M = _solve_amplitudes(x)
        return sse + _prior_scale * _prior_penalty(x), A, M

    def _objective(x):
        loss, _, _ = _penalized_loss(x)
        return loss

    n_dim = 2 * len(components)
    unit = _draw_unit_samples(n_samples, n_dim, method=sample_method, seed=seed)
    samples = _scale_unit_samples(unit, nonlinear_bounds)

    if verbose:
        labels = ', '.join(c['label'] for c in components)
        print(f'Global nested fit: {labels}')
        print(f'  components={len(components)}  nonlinear_dim={n_dim}  samples={n_samples}')
        print(f'  fit pixels={len(wav_fit)}  ranges={joint_ranges}')

    global_results = []
    for i, x in enumerate(samples):
        loss, A, _ = _penalized_loss(x)
        global_results.append({
            'loss': loss,
            'x': x.copy(),
            'x_scaled': unit[i].copy(),
            'A': A.copy(),
        })

    global_results.sort(key=lambda r: r['loss'])
    global_candidates = _distinct_solutions(
        global_results,
        n_keep=max(n_refine, n_top),
        components=components,
        min_distance=distinct_min_distance,
    )
    refine_starts = global_candidates[:n_refine]

    refined = []
    for cand in refine_starts:
        opt = scipy.optimize.minimize(
            _objective,
            cand['x'],
            method='L-BFGS-B',
            bounds=nonlinear_bounds,
        )
        loss, A, _ = _penalized_loss(opt.x)
        scaled = np.array([
            (opt.x[j] - nonlinear_bounds[j][0]) /
            (nonlinear_bounds[j][1] - nonlinear_bounds[j][0])
            for j in range(n_dim)
        ])
        refined.append({
            'loss': loss,
            'x': opt.x.copy(),
            'x_scaled': scaled,
            'A': A.copy(),
            'success': bool(opt.success),
            'message': str(opt.message),
            'start_loss': cand['loss'],
        })

    refined.sort(key=lambda r: r['loss'])
    top_refined = _distinct_solutions(
        refined,
        n_keep=n_top,
        components=components,
        min_distance=distinct_min_distance,
    )
    if not top_refined:
        raise RuntimeError('no valid fitted solution found')

    best = top_refined[0]

    def _decode_solution(sol, rank=None):
        params = {}
        for i, comp in enumerate(components):
            A = float(sol['A'][i])
            params[comp['label']] = {
                'T': float(sol['x'][2 * i]),
                'logN': float(sol['x'][2 * i + 1]),
                'A': A,
                'log10A': float(np.log10(max(A, 1e-300))),
                'model_mol': comp['model_mol'],
            }
        out = {
            'mse_loss': float(sol['loss']),
            'params': params,
            'success': sol.get('success', None),
            'message': sol.get('message', ''),
            'start_loss': float(sol.get('start_loss', np.nan)),
        }
        if rank is not None:
            out['rank'] = rank
        return out

    best_params = _decode_solution(best)['params']
    component_fluxes = {}
    total_flux = np.zeros_like(obs_wav, dtype=np.float64)
    for label, p in best_params.items():
        flux = _mol_flux_on_wav(
            pretrained[p['model_mol']],
            p['T'],
            p['logN'],
            p['A'],
            obs_wav,
        )
        component_fluxes[label] = flux
        total_flux += flux

    top_list = [_decode_solution(sol, rank=i + 1) for i, sol in enumerate(top_refined)]
    global_top = [_decode_solution(sol, rank=i + 1)
                  for i, sol in enumerate(global_candidates[:n_top])]
    uncertainty = _top_list_uncertainty(top_list, sigma=sigma)

    if verbose:
        print(f'  best SSE={best["loss"]:.6g}')
        for label, p in best_params.items():
            print(f'  {label:8s} T={p["T"]:.1f}  logN={p["logN"]:.3f}  '
                  f'log10A={p["log10A"]:.3f}')

    return {
        'mol': mol,
        'components': components,
        'params': best_params,
        'model_flux': total_flux,
        'component_fluxes': component_fluxes,
        'residual': obs_flux - total_flux,
        'fit_ranges': joint_ranges,
        'top_list': top_list,
        'global_top': global_top,
        'uncertainty': uncertainty,
    }


def load_observed_spectrum(path, skiprows=1, delimiter=',', wav_col=0, flux_col=1):
    """
    Load an observed spectrum and remove NaN/blank flux pixels.

    Args:
        path (str): input CSV or text file.
        skiprows (int): number of header rows to skip.
        delimiter (str): file delimiter.
        wav_col (int): wavelength column index.
        flux_col (int): flux column index.

    Returns:
        tuple[np.ndarray, np.ndarray]: wavelength and flux arrays.
    """
    data = np.loadtxt(path, skiprows=skiprows, delimiter=delimiter, dtype='str')
    wav = np.array([np.float64(row[wav_col]) for row in data])
    flux = np.array([
        np.float64(row[flux_col]) if row[flux_col] != '' else np.nan
        for row in data
    ])
    valid = np.isfinite(wav) & np.isfinite(flux)
    return wav[valid], flux[valid]


def save_fit_outputs(result, output_dir, output_prefix):
    """
    Save fitted spectrum, best parameters, and top solutions to CSV files.

    Args:
        result (dict): result returned by fit_molecules, with obs_wav/obs_flux
            added by the caller.
        output_dir (str): directory for output CSV files.
        output_prefix (str): filename prefix.

    Returns:
        dict: output kind -> saved path.
    """
    os.makedirs(output_dir, exist_ok=True)

    spectrum_df = pd.DataFrame({
        'wave': result['obs_wav'],
        'flux': result['obs_flux'],
        'model': result['model_flux'],
        'residual': result['residual'],
    })
    for label, flux in result['component_fluxes'].items():
        spectrum_df[f'{label}_model'] = flux
    spectrum_path = os.path.join(output_dir, f'{output_prefix}_fit_spectrum.csv')
    spectrum_df.to_csv(spectrum_path, index=False)

    param_rows = []
    for label, p in result['params'].items():
        u = result.get('uncertainty', {}).get(label, {})
        u_T = u.get('T', {})
        u_logN = u.get('logN', {})
        u_log10A = u.get('log10A', {})
        param_rows.append({
            'component': label,
            'model_mol': p['model_mol'],
            'T': p['T'],
            'T_minus': u_T.get('minus', np.nan),
            'T_plus': u_T.get('plus', np.nan),
            'logN': p['logN'],
            'logN_minus': u_logN.get('minus', np.nan),
            'logN_plus': u_logN.get('plus', np.nan),
            'A': p['A'],
            'log10A': p['log10A'],
            'log10A_minus': u_log10A.get('minus', np.nan),
            'log10A_plus': u_log10A.get('plus', np.nan),
        })
    params_path = os.path.join(output_dir, f'{output_prefix}_fit_params.csv')
    pd.DataFrame(param_rows).to_csv(params_path, index=False)

    top_rows = []
    for sol in result['top_list']:
        row = {'rank': sol['rank'], 'mse_loss': sol['mse_loss']}
        for label, p in sol['params'].items():
            row[f'{label}_T'] = p['T']
            row[f'{label}_logN'] = p['logN']
            row[f'{label}_log10A'] = p['log10A']
        top_rows.append(row)
    top_path = os.path.join(output_dir, f'{output_prefix}_fit_top.csv')
    pd.DataFrame(top_rows).to_csv(top_path, index=False)

    return {
        'spectrum': spectrum_path,
        'params': params_path,
        'top': top_path,
    }


def detect_stage_molecules(obs_wav, obs_flux, stage, sigma_noise,
                            screening=True, sigma_factor=3.0):
    """
    Return the subset of molecules in stage['mol'] that pass a peak detection test.

    For single-molecule stages (detect_peaks is a list of (lo, hi) ranges): the
    stage passes if the maximum flux in any listed window exceeds the threshold.
    For multi-molecule stages (detect_peaks is a dict {mol: (lo, hi)}): each
    molecule is checked independently; only detected ones are returned.
    If screening is False, or detect_peaks is absent/None, all molecules are
    returned without checking.

    Args:
        obs_wav (np.ndarray): wavelength axis in µm (typically the running residual).
        obs_flux (np.ndarray): flux to check for detections.
        stage (dict): stage config dict with keys 'mol' and optionally 'detect_peaks'.
        sigma_noise (float or None): per-pixel noise std in Jy; if None, screening
            is skipped regardless of the screening flag.
        screening (bool): master toggle; pass DETECTION_SCREENING from the script.
        sigma_factor (float): detection threshold = sigma_factor × sigma_noise.

    Returns:
        list[str]: molecule names that passed detection (subset of stage['mol']).
    """
    mols         = _as_list(stage['mol'])
    detect_peaks = stage.get('detect_peaks')

    if not screening or detect_peaks is None or sigma_noise is None:
        return mols

    thr = sigma_factor * sigma_noise

    if isinstance(detect_peaks, dict):
        detected = []
        for mol in mols:
            if mol not in detect_peaks:
                detected.append(mol)
                continue
            lo, hi = detect_peaks[mol]
            mask   = (obs_wav >= lo) & (obs_wav <= hi)
            if mask.any() and np.nanmax(obs_flux[mask]) > thr:
                detected.append(mol)
            else:
                print(f'  [{mol}] not detected (peak < {sigma_factor:.1f}σ) — excluded')
        return detected
    else:
        for lo, hi in detect_peaks:
            mask = (obs_wav >= lo) & (obs_wav <= hi)
            if mask.any() and np.nanmax(obs_flux[mask]) > thr:
                return mols
        mol_str = ', '.join(mols)
        print(f'  [{mol_str}] not detected (all peaks < {sigma_factor:.1f}σ) — stage skipped')
        return []


def save_running_spectrum(obs_wav, residual, cumulative_model, output_dir, prefix):
    """
    Save the running residual and cumulative model after a fitting stage.

    Args:
        obs_wav (np.ndarray): wavelength axis in µm.
        residual (np.ndarray): current running residual (obs minus all stages so far).
        cumulative_model (np.ndarray): sum of all stage model fluxes fitted so far.
        output_dir (str): output directory (created if missing).
        prefix (str): filename prefix; file is saved as {prefix}_running.csv.

    Returns:
        str: path to the saved CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f'{prefix}_running.csv')
    pd.DataFrame({
        'wave':             obs_wav,
        'residual':         residual,
        'cumulative_model': cumulative_model,
    }).to_csv(path, index=False)
    return path


def print_fit_params(fit):
    """
    Print best-fit parameters with ± uncertainties for all components in a fit result.

    Args:
        fit (dict): result dict returned by fit_molecules.
    """
    for label, p in fit['params'].items():
        u        = fit.get('uncertainty', {}).get(label, {})
        u_T      = u.get('T',      {})
        u_logN   = u.get('logN',   {})
        u_log10A = u.get('log10A', {})
        print(
            f'    {label:14s} '
            f'T={p["T"]:.1f} (+{u_T.get("plus", float("nan")):.1f}/'
            f'-{u_T.get("minus", float("nan")):.1f}) K  '
            f'logN={p["logN"]:.3f} (+{u_logN.get("plus", float("nan")):.3f}/'
            f'-{u_logN.get("minus", float("nan")):.3f})  '
            f'log10A={p["log10A"]:.3f} (+{u_log10A.get("plus", float("nan")):.3f}/'
            f'-{u_log10A.get("minus", float("nan")):.3f})'
        )


def save_fitted_comparison(csv_path, source_name, all_fits):
    """
    Upsert one row of best-fit parameters into a comparison CSV.

    The CSV schema matches Parameter_Comparison.csv (log10 units for N and A,
    K for T).  If the file does not exist it is created.  If *source_name* is
    already present its values are updated in-place; otherwise a new row is
    appended.

    Args:
        csv_path (str): path to the comparison CSV.
        source_name (str): identifier for this source (used as the key).
        all_fits (dict): stage-name → fit-result dict from the fitting loop.
    """
    all_params = {}
    all_unc    = {}
    for fit in all_fits.values():
        all_params.update(fit['params'])
        all_unc.update(fit.get('uncertainty', {}))

    def _v(comp, key):
        if comp not in all_params:
            return ''
        return round(all_params[comp][key], 4)

    def _e(comp, key, which):
        if comp not in all_unc:
            return ''
        v = all_unc[comp].get(key, {}).get(which, float('nan'))
        return round(v, 4) if np.isfinite(v) else ''

    row = {
        'Source':               source_name,
        'H2O_N_warm':           _v('H2O_warm', 'logN'),
        'H2O_N_warm_err_low':   _e('H2O_warm', 'logN',   'minus'),
        'H2O_N_warm_err_high':  _e('H2O_warm', 'logN',   'plus'),
        'H2O_N_hot':            _v('H2O_hot',  'logN'),
        'H2O_N_hot_err_low':    _e('H2O_hot',  'logN',   'minus'),
        'H2O_N_hot_err_high':   _e('H2O_hot',  'logN',   'plus'),
        'H2O_T_warm':           _v('H2O_warm', 'T'),
        'H2O_T_warm_err_low':   _e('H2O_warm', 'T',      'minus'),
        'H2O_T_warm_err_high':  _e('H2O_warm', 'T',      'plus'),
        'H2O_T_hot':            _v('H2O_hot',  'T'),
        'H2O_T_hot_err_low':    _e('H2O_hot',  'T',      'minus'),
        'H2O_T_hot_err_high':   _e('H2O_hot',  'T',      'plus'),
        'H2O_A_warm':           _v('H2O_warm', 'log10A'),
        'H2O_A_warm_err_low':   _e('H2O_warm', 'log10A', 'minus'),
        'H2O_A_warm_err_high':  _e('H2O_warm', 'log10A', 'plus'),
        'H2O_A_hot':            _v('H2O_hot',  'log10A'),
        'H2O_A_hot_err_low':    _e('H2O_hot',  'log10A', 'minus'),
        'H2O_A_hot_err_high':   _e('H2O_hot',  'log10A', 'plus'),
        'C2H2_N':               _v('C2H2', 'logN'),
        'C2H2_N_err_low':       _e('C2H2', 'logN',   'minus'),
        'C2H2_N_err_high':      _e('C2H2', 'logN',   'plus'),
        'C2H2_T':               _v('C2H2', 'T'),
        'C2H2_T_err_low':       _e('C2H2', 'T',      'minus'),
        'C2H2_T_err_high':      _e('C2H2', 'T',      'plus'),
        'C2H2_A':               _v('C2H2', 'log10A'),
        'C2H2_A_err_low':       _e('C2H2', 'log10A', 'minus'),
        'C2H2_A_err_high':      _e('C2H2', 'log10A', 'plus'),
        'HCN_N':                _v('HCN', 'logN'),
        'HCN_N_err_low':        _e('HCN', 'logN',   'minus'),
        'HCN_N_err_high':       _e('HCN', 'logN',   'plus'),
        'HCN_T':                _v('HCN', 'T'),
        'HCN_T_err_low':        _e('HCN', 'T',      'minus'),
        'HCN_T_err_high':       _e('HCN', 'T',      'plus'),
        'HCN_A':                _v('HCN', 'log10A'),
        'HCN_A_err_low':        _e('HCN', 'log10A', 'minus'),
        'HCN_A_err_high':       _e('HCN', 'log10A', 'plus'),
        'CO2_N':                _v('CO2', 'logN'),
        'CO2_N_err_low':        _e('CO2', 'logN',   'minus'),
        'CO2_N_err_high':       _e('CO2', 'logN',   'plus'),
        'CO2_T':                _v('CO2', 'T'),
        'CO2_T_err_low':        _e('CO2', 'T',      'minus'),
        'CO2_T_err_high':       _e('CO2', 'T',      'plus'),
        'CO2_A':                _v('CO2', 'log10A'),
        'CO2_A_err_low':        _e('CO2', 'log10A', 'minus'),
        'CO2_A_err_high':       _e('CO2', 'log10A', 'plus'),
    }

    if os.path.exists(csv_path):
        df   = pd.read_csv(csv_path)
        mask = df['Source'] == source_name
        if mask.any():
            for col, val in row.items():
                if col in df.columns:
                    df.loc[mask, col] = val
            print(f'  Updated existing row for {source_name!r} in {os.path.basename(csv_path)}')
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            print(f'  Appended new row for {source_name!r} to {os.path.basename(csv_path)}')
    else:
        df = pd.DataFrame([row])
        print(f'  Created {os.path.basename(csv_path)} with first row for {source_name!r}')

    df.to_csv(csv_path, index=False)
