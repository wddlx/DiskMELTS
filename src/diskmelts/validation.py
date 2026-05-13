"""
diskmelts/validation.py — Validation functions for pretrained forward models.

Two public functions, both operating on a single molecule per call:

    validate_nt   : T and logN retrieval on pretrain grid spectra (A=1 fixed).
                    Tests model accuracy without the confound of A scaling.
                    Input comes from the pretrain CSV produced by
                    generate_pre_training_set.

    validate_full : T, logN, and A retrieval on synthetic spectra with random A.
                    Tests the full fitting pipeline as it is used on observations.
                    Spectra are generated in-memory directly from the model grid.

Both functions return dicts of true and predicted parameter arrays for downstream
analysis and plotting (see diskmelts/plotting.py :: plot_validation).

Typical workflow
----------------
    from diskmelts.fitting import load_models
    from diskmelts.trainmodel import load_model_grid
    from diskmelts.validation import validate_nt, validate_full
    from diskmelts.plotting import plot_validation

    pretrained = load_models(model_paths, pretrain_csv_paths, wav_ranges)
    models_h2o = load_model_grid('Model_grids/H2O')

    nt   = validate_nt('H2O', pretrained,
                       pretrain_csv='Pretrain_grid/pretrain_H2O_11to19.csv',
                       fit_ranges=[(11.0, 12.0), (17.0, 18.5)])
    full = validate_full('H2O', models_h2o, pretrained,
                         fit_ranges=[(11.0, 12.0), (17.0, 18.5)])

    plot_validation('H2O', nt,   save_path='figures/val_H2O_nt.png')
    plot_validation('H2O', full, save_path='figures/val_H2O_full.png')
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from diskmelts.fitting import fit_nested


# ---------------------------------------------------------------------------
# Internal helper — gradient-based NT-only fitter
# ---------------------------------------------------------------------------

def _fit_only_nt(
    obs_norm_flux,
    obs_log10_peak,
    wav_fit,
    pretrained_mol,
    T_bounds,
    logN_bounds,
    n_starts=20,
    n_steps=300,
    lr=0.03,
):
    """
    Recover T and logN only by jointly matching the peak-normalised spectral
    shape and the log10(peak flux) through the two-MLP forward model.

    Uses multi-start Adam gradient descent operating in standardised parameter
    space — no A is fitted.  wav_fit must be a subset of pretrained_mol['wav'].

    Args:
        obs_norm_flux (np.ndarray): peak-normalised observed flux on wav_fit
        obs_log10_peak (float): true log10(peak flux) of the spectrum
        wav_fit (np.ndarray): wavelength grid for fitting (subset of model wav)
        pretrained_mol (dict): single-molecule entry, e.g. pretrained['H2O']
        T_bounds (tuple): (T_min, T_max) K
        logN_bounds (tuple): (logN_min, logN_max)
        n_starts (int): number of random starting points (default 20)
        n_steps (int): Adam steps per start (default 300)
        lr (float): Adam learning rate (default 0.03)

    Returns:
        (T_phys, logN_phys) as floats
    """
    net_shape   = pretrained_mol['net_shape']
    net_peak    = pretrained_mol['net_peak']
    xp_sc       = pretrained_mol['xp_sc']
    yp_sc_shape = pretrained_mol['yp_sc_shape']
    yp_sc_peak  = pretrained_mol['yp_sc_peak']
    pca         = pretrained_mol.get('pca', None)
    wav         = pretrained_mol['wav']

    device   = next(net_shape.parameters()).device
    fit_mask = np.isin(wav, wav_fit)   # wav_fit must be on the model grid

    xp_mean = torch.tensor(xp_sc.mean_,  dtype=torch.float32, device=device)
    xp_std  = torch.tensor(xp_sc.scale_, dtype=torch.float32, device=device)

    yp_mean_shape = torch.tensor(yp_sc_shape.mean_,  dtype=torch.float32, device=device)
    yp_std_shape  = torch.tensor(yp_sc_shape.scale_, dtype=torch.float32, device=device)
    yp_mean_peak  = torch.tensor(yp_sc_peak.mean_,   dtype=torch.float32, device=device)
    yp_std_peak   = torch.tensor(yp_sc_peak.scale_,  dtype=torch.float32, device=device)

    if pca is not None:
        pca_comp = torch.tensor(pca.components_, dtype=torch.float32, device=device)
        pca_mean = torch.tensor(pca.mean_,       dtype=torch.float32, device=device)
    else:
        pca_comp = pca_mean = None

    for p in net_shape.parameters(): p.requires_grad_(False)
    for p in net_peak.parameters():  p.requires_grad_(False)

    obs_norm_t   = torch.from_numpy(obs_norm_flux.astype(np.float32)).to(device)
    obs_norm_rep = obs_norm_t.unsqueeze(0).expand(n_starts, -1)
    obs_peak_t   = torch.full((n_starts, 1), float(obs_log10_peak),
                              dtype=torch.float32, device=device)

    T_init    = torch.rand(n_starts) * (T_bounds[1]    - T_bounds[0])    + T_bounds[0]
    logN_init = torch.rand(n_starts) * (logN_bounds[1] - logN_bounds[0]) + logN_bounds[0]
    T_s    = (T_init    - xp_mean[0].cpu()) / xp_std[0].cpu()
    logN_s = (logN_init - xp_mean[1].cpu()) / xp_std[1].cpu()

    T_min_s    = (T_bounds[0]    - xp_mean[0]) / xp_std[0]
    T_max_s    = (T_bounds[1]    - xp_mean[0]) / xp_std[0]
    logN_min_s = (logN_bounds[0] - xp_mean[1]) / xp_std[1]
    logN_max_s = (logN_bounds[1] - xp_mean[1]) / xp_std[1]

    params_s = nn.Parameter(torch.stack([T_s, logN_s], dim=1).to(device))
    opt      = torch.optim.Adam([params_s], lr=lr)

    def _forward(T_logN_s):
        z_shape_s = net_shape(T_logN_s)
        z_shape   = z_shape_s * yp_std_shape + yp_mean_shape
        norm_spec  = (z_shape @ pca_comp + pca_mean) if pca is not None else z_shape
        norm_spec  = torch.clamp(norm_spec[:, fit_mask], min=0.0)
        z_peak_s   = net_peak(T_logN_s)
        log10_peak = z_peak_s * yp_std_peak + yp_mean_peak
        return norm_spec, log10_peak

    for _ in range(n_steps):
        opt.zero_grad()
        pred_norm, pred_log10_peak = _forward(params_s)
        loss = (torch.nn.functional.mse_loss(pred_norm, obs_norm_rep) +
                torch.nn.functional.mse_loss(pred_log10_peak, obs_peak_t))
        loss.backward()
        opt.step()
        with torch.no_grad():
            params_s[:, 0].clamp_(T_min_s, T_max_s)
            params_s[:, 1].clamp_(logN_min_s, logN_max_s)

    with torch.no_grad():
        pred_norm, pred_log10_peak = _forward(params_s)
        per_start_loss = (
            ((pred_norm - obs_norm_rep) ** 2).mean(dim=1) +
            ((pred_log10_peak - obs_peak_t) ** 2).squeeze(1)
        )
        best_s = params_s[per_start_loss.argmin()].cpu()

    for p in net_shape.parameters(): p.requires_grad_(True)
    for p in net_peak.parameters():  p.requires_grad_(True)

    T_phys    = (best_s[0] * xp_std[0].cpu() + xp_mean[0].cpu()).item()
    logN_phys = (best_s[1] * xp_std[1].cpu() + xp_mean[1].cpu()).item()
    return T_phys, logN_phys


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_nt(
    mol,
    pretrained,
    pretrain_csv,
    fit_ranges,
    n_samples=None,
    frac=0.05,
    T_bounds=(200, 1400),
    logN_bounds=(14.0, 19.0),
    n_starts=20,
    n_steps=300,
    lr=0.03,
    seed=42,
):
    """
    Validate T and logN retrieval on pretrain grid spectra with A=1.

    Uses the same gradient-based Adam optimizer as the NT-only fitter: jointly
    matches the peak-normalised spectral shape (net_shape output) and the
    log10(peak flux) (net_peak output).  No A is fitted.  This works directly
    in the two-MLP representation space and is much more accurate for NT-only
    retrieval than using fit_nested on physical flux.

    Call once per molecule.

    Args:
        mol (str): molecule name, e.g. 'H2O', 'C2H2'
        pretrained (dict): output of load_models (keyed by mol name)
        pretrain_csv (str): path to the pretrain CSV from generate_pre_training_set
        fit_ranges: (lo, hi) or [(lo1,hi1), ...] µm; must align with the model
                    wavelength grid (pretrained[mol]['wav'])
        n_samples (int or None): rows to validate; if None uses frac (default None)
        frac (float): fraction of rows when n_samples is None (default 0.05)
        T_bounds (tuple): (T_min, T_max) K (default (200, 1400))
        logN_bounds (tuple): (logN_min, logN_max) (default (14, 19))
        n_starts (int): random starting points for Adam optimizer (default 20)
        n_steps (int): Adam gradient steps per start (default 300)
        lr (float): Adam learning rate (default 0.03)
        seed (int): random seed for row sampling (default 42)

    Returns:
        (dict) with keys:
            'T_true'    : np.ndarray — grid temperatures (K)
            'T_pred'    : np.ndarray — recovered temperatures (K)
            'logN_true' : np.ndarray — grid log10 column densities
            'logN_pred' : np.ndarray — recovered log10 column densities
    """
    df = pd.read_csv(pretrain_csv)
    flux_cols = [c for c in df.columns if c.startswith('wav_')]
    wav_pre   = np.array([float(c[4:]) for c in flux_cols])

    df = df[
        (df[f'{mol}_T']    >= T_bounds[0])    & (df[f'{mol}_T']    <= T_bounds[1]) &
        (df[f'{mol}_logN'] >= logN_bounds[0]) & (df[f'{mol}_logN'] <= logN_bounds[1])
    ].reset_index(drop=True)

    n = int(len(df) * frac) if n_samples is None else min(n_samples, len(df))
    df_sample = df.sample(n=n, random_state=seed).reset_index(drop=True)

    wav_net = pretrained[mol]['wav']
    ranges  = [fit_ranges] if isinstance(fit_ranges[0], (int, float)) else list(fit_ranges)
    fit_mask_net = np.zeros(len(wav_net), dtype=bool)
    for lo, hi in ranges:
        fit_mask_net |= (wav_net >= lo) & (wav_net <= hi)
    wav_fit = wav_net[fit_mask_net]

    T_true_list, logN_true_list = [], []
    T_pred_list, logN_pred_list = [], []

    print(f'[{mol}] NT validation: {len(df_sample)} spectra ...')
    for i, (_, row) in enumerate(df_sample.iterrows()):
        T_t    = float(row[f'{mol}_T'])
        logN_t = float(row[f'{mol}_logN'])
        log10p = float(row[f'{mol}_log10_peak'])

        norm_flux    = row[flux_cols].to_numpy(dtype=np.float32)
        obs_norm_fit = np.interp(wav_fit, wav_pre, norm_flux)

        T_p, logN_p = _fit_only_nt(
            obs_norm_fit, log10p, wav_fit, pretrained[mol],
            T_bounds=T_bounds, logN_bounds=logN_bounds,
            n_starts=n_starts, n_steps=n_steps, lr=lr,
        )

        T_true_list.append(T_t);     logN_true_list.append(logN_t)
        T_pred_list.append(T_p);     logN_pred_list.append(logN_p)

        if (i + 1) % 10 == 0 or (i + 1) == len(df_sample):
            print(f'  {i + 1}/{len(df_sample)}')

    return {
        'T_true':    np.array(T_true_list,    dtype=np.float64),
        'T_pred':    np.array(T_pred_list,    dtype=np.float64),
        'logN_true': np.array(logN_true_list, dtype=np.float64),
        'logN_pred': np.array(logN_pred_list, dtype=np.float64),
    }


def validate_nt_holdout(
    mol,
    pretrained,
    models,
    holdout_keys,
    fit_ranges,
    T_bounds=(200, 1400),
    logN_bounds=(14.0, 19.0),
    n_starts=20,
    n_steps=300,
    lr=0.03,
):
    """
    Validate T and logN retrieval on grid points that were withheld from the
    pretrain CSV (unseen during training).

    For each holdout (T, logN) key the physical flux is taken directly from
    the model grid, peak-normalised, and passed to the Adam-based NT optimizer.
    This gives an honest estimate of generalisation to unseen grid points.

    Args:
        mol (str): molecule name, e.g. 'H2O', 'C2H2'
        pretrained (dict): output of load_models (keyed by mol name)
        models (dict): output of load_model_grid for this molecule,
                       keyed by (T, logN) with 'wavelength' and 'flux' arrays
        holdout_keys (list): list of (T, logN) tuples that were excluded from
                             the pretrain CSV
        fit_ranges: (lo, hi) or [(lo1,hi1), ...] µm; must align with the model
                    wavelength grid (pretrained[mol]['wav'])
        T_bounds (tuple): (T_min, T_max) K (default (200, 1400))
        logN_bounds (tuple): (logN_min, logN_max) (default (14, 19))
        n_starts (int): random starting points for Adam optimizer (default 20)
        n_steps (int): Adam gradient steps per start (default 300)
        lr (float): Adam learning rate (default 0.03)

    Returns:
        (dict) with keys:
            'T_true'    : np.ndarray — grid temperatures (K)
            'T_pred'    : np.ndarray — recovered temperatures (K)
            'logN_true' : np.ndarray — grid log10 column densities
            'logN_pred' : np.ndarray — recovered log10 column densities
    """
    wav_net = pretrained[mol]['wav']
    ranges  = [fit_ranges] if isinstance(fit_ranges[0], (int, float)) else list(fit_ranges)
    fit_mask = np.zeros(len(wav_net), dtype=bool)
    for lo, hi in ranges:
        fit_mask |= (wav_net >= lo) & (wav_net <= hi)
    wav_fit = wav_net[fit_mask]

    T_true_list, logN_true_list = [], []
    T_pred_list, logN_pred_list = [], []

    print(f'[{mol}] NT holdout validation: {len(holdout_keys)} spectra ...')
    for i, key in enumerate(holdout_keys):
        T_t, logN_t = key
        wav_model  = models[key]['wavelength']
        flux_model = models[key]['flux']

        flux_interp = np.interp(wav_net, wav_model, flux_model)
        peak = np.abs(flux_interp).max()
        if peak <= 0:
            continue

        norm_flux    = flux_interp / peak
        log10_peak   = np.log10(float(peak))
        obs_norm_fit = norm_flux[fit_mask]

        T_p, logN_p = _fit_only_nt(
            obs_norm_fit, log10_peak, wav_fit, pretrained[mol],
            T_bounds=T_bounds, logN_bounds=logN_bounds,
            n_starts=n_starts, n_steps=n_steps, lr=lr,
        )

        T_true_list.append(float(T_t))
        logN_true_list.append(float(logN_t))
        T_pred_list.append(T_p)
        logN_pred_list.append(logN_p)

        if (i + 1) % 10 == 0 or (i + 1) == len(holdout_keys):
            print(f'  {i + 1}/{len(holdout_keys)}')

    return {
        'T_true':    np.array(T_true_list,    dtype=np.float64),
        'T_pred':    np.array(T_pred_list,    dtype=np.float64),
        'logN_true': np.array(logN_true_list, dtype=np.float64),
        'logN_pred': np.array(logN_pred_list, dtype=np.float64),
    }


def validate_full(
    mol,
    models,
    pretrained,
    fit_ranges,
    n_samples=30,
    log_A_range=(-2.0, 2.0),
    noise_scale=0.001,
    T_bounds=(200, 1400),
    logN_bounds=(14.0, 19.0),
    loga_bounds=(-2.0, 2.0),
    n_restarts=50,
    seed=42,
):
    """
    Validate T, logN, and A retrieval on synthetic spectra with random A.

    Randomly draws (T, logN) grid points and A values, constructs the
    corresponding physical spectrum with optional noise, then runs fit_nested
    to recover all three parameters.  This tests the full fitting pipeline.

    Call once per molecule.

    Args:
        mol (str): molecule name, e.g. 'H2O', 'C2H2'
        models (dict): output of load_model_grid for this molecule,
                       keyed by (T, logN) with 'wavelength' and 'flux' arrays
        pretrained (dict): output of load_models (keyed by mol name)
        fit_ranges: (lo, hi) or [(lo1,hi1), ...] µm passed to fit_nested
        n_samples (int): number of synthetic spectra to generate (default 30)
        log_A_range (tuple): (log10A_min, log10A_max) for uniform A draw
        noise_scale (float): Gaussian noise std as fraction of peak flux;
                             0.0 for noise-free spectra (default 0.001)
        T_bounds (tuple): (T_min, T_max) K passed to fit_nested
        logN_bounds (tuple): (logN_min, logN_max) passed to fit_nested
        loga_bounds (tuple): (log10A_min, log10A_max) for fitting search range
                             (default (-2, 2)); independent of log_A_range
        n_restarts (int): random restarts per spectrum in fit_nested (default 50)
        seed (int): random seed (default 42)

    Returns:
        (dict) with keys:
            'T_true'       : np.ndarray — true temperatures (K)
            'T_pred'       : np.ndarray — recovered temperatures (K)
            'logN_true'    : np.ndarray — true log10 column densities
            'logN_pred'    : np.ndarray — recovered log10 column densities
            'log10A_true'  : np.ndarray — true log10 scaling factors
            'log10A_pred'  : np.ndarray — recovered log10 scaling factors
            'log10NA_true' : np.ndarray — logN + log10A (true)
            'log10NA_pred' : np.ndarray — logN + log10A (predicted)
    """
    rng  = np.random.default_rng(seed)
    keys = list(models.keys())
    wav_grid = next(iter(models.values()))['wavelength']

    T_true_list,    logN_true_list,    A_true_list    = [], [], []
    T_pred_list,    logN_pred_list,    A_pred_list    = [], [], []

    print(f'[{mol}] Full validation: {n_samples} spectra ...')
    for i in range(n_samples):
        idx      = rng.integers(len(keys))
        T_t, logN_t = keys[idx]
        A_t      = 10.0 ** rng.uniform(*log_A_range)

        flux = A_t * models[keys[idx]]['flux'].copy()
        if noise_scale > 0:
            peak = np.abs(flux).max()
            if peak > 0:
                flux += rng.normal(0, noise_scale * peak, size=len(flux))

        result = fit_nested(
            wav_grid, flux, mol, pretrained,
            fit_ranges=fit_ranges,
            T_bounds=T_bounds, logN_bounds=logN_bounds,
            loga_bounds=loga_bounds,
            n_restarts=n_restarts, seed=i,
        )

        T_true_list.append(float(T_t))
        logN_true_list.append(float(logN_t))
        A_true_list.append(float(A_t))
        T_pred_list.append(float(result['params']['T']))
        logN_pred_list.append(float(result['params']['logN']))
        A_pred_list.append(float(result['params']['A']))

        if (i + 1) % 10 == 0 or (i + 1) == n_samples:
            print(f'  {i + 1}/{n_samples}')

    logN_true = np.array(logN_true_list, dtype=np.float64)
    logN_pred = np.array(logN_pred_list, dtype=np.float64)
    A_true    = np.array(A_true_list,    dtype=np.float64)
    A_pred    = np.array(A_pred_list,    dtype=np.float64)

    log10A_true = np.log10(np.maximum(A_true, 1e-10))
    log10A_pred = np.log10(np.maximum(A_pred, 1e-10))

    return {
        'T_true':        np.array(T_true_list, dtype=np.float64),
        'T_pred':        np.array(T_pred_list, dtype=np.float64),
        'logN_true':     logN_true,
        'logN_pred':     logN_pred,
        'log10A_true':   log10A_true,
        'log10A_pred':   log10A_pred,
        'log10NA_true':  logN_true  + log10A_true,
        'log10NA_pred':  logN_pred  + log10A_pred,
    }
