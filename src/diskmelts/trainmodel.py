"""
diskmelts/trainmodel.py — Training utilities for per-molecule forward surrogate models.

Provides public functions:
    load_model_grid          : load T{T}N{logN}.csv files from a Model_grids/ directory
    generate_pre_training_set: enumerate every (T, logN) grid point → per-molecule CSV
    pretrain_forward_model   : train (or load) the two-MLP forward model per molecule
                                  net_shape : (T, logN) → n_pca PCA coefficients
                                  net_peak  : (T, logN) → log10(peak flux)
    train_model_bank         : train overlapping wavelength/T/logN checkpoint tiles
    select_grid_holdout      : reproducibly withhold complete slab-grid models
    tune_pca_components      : choose PCA size from unseen-spectrum reconstruction

Run as a script to generate pretrain CSVs, train all four molecules, and save
loss-curve diagnostics:
    python src/diskmelts/trainmodel.py

Edit the CONFIGURATION block inside if __name__ == '__main__' to change paths,
which molecules to train, architecture, PCA size, learning rate, etc.
"""

import copy
import json
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ===========================================================================
# Grid loading
# ===========================================================================

def load_model_grid(grid_dir, T_range=None, logN_range=None, wav_range=None):
    """
    Load all slab-model CSV files from a directory.

    Files must be named T{T}N{logN}.csv with columns 'wave' (µm) and 'Line' (Jy).

    Args:
        grid_dir (str): path to the directory containing the CSV files
        T_range (tuple or None): optional inclusive temperature bounds; files
            outside the bounds are not read.
        logN_range (tuple or None): optional inclusive column-density bounds;
            files outside the bounds are not read.
        wav_range (tuple or None): optional inclusive wavelength bounds; flux
            arrays are cropped while loading.

    Returns:
        (dict): keyed by (T, logN) tuples; each value is a dict with
                'wavelength' (np.ndarray) and 'flux' (np.ndarray)
    """
    models  = {}
    pattern = re.compile(r'T(\d+)N([\d.]+)\.csv')
    for fname in sorted(os.listdir(grid_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        T    = int(m.group(1))
        logN = float(m.group(2))
        if T_range is not None and not T_range[0] <= T <= T_range[1]:
            continue
        if logN_range is not None and not logN_range[0] <= logN <= logN_range[1]:
            continue
        df   = pd.read_csv(os.path.join(grid_dir, fname))
        if wav_range is not None:
            df = df[(df['wave'] >= wav_range[0]) & (df['wave'] <= wav_range[1])]
        models[(T, logN)] = {
            'wavelength': df['wave'].to_numpy(),
            'flux':       df['Line'].to_numpy(),
        }
    return models


def select_grid_holdout(
    models,
    fraction=0.1,
    seed=42,
    T_range=None,
    logN_range=None,
    preserve_boundaries=True,
):
    """Select complete ``(T, logN)`` grid points for unseen validation.

    Unlike the internal neural-network validation split, these keys are meant
    to be passed to :func:`generate_pre_training_set` as ``holdout_keys`` so
    their spectra never enter PCA fitting, scaler fitting, or MLP training.

    Args:
        models (dict): output of :func:`load_model_grid`.
        fraction (float): fraction of eligible grid points to withhold.
        seed (int): random seed.
        T_range (tuple or None): optional inclusive temperature bounds.
        logN_range (tuple or None): optional inclusive column-density bounds.
        preserve_boundaries (bool): keep the outer T/logN rows in training so
            the checkpoint retains full interpolation coverage.

    Returns:
        list: sorted ``(T, logN)`` holdout keys.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f'fraction must be between 0 and 1, got {fraction}')

    keys = [
        tuple(key) for key in models
        if (T_range is None or T_range[0] <= key[0] <= T_range[1])
        and (logN_range is None or logN_range[0] <= key[1] <= logN_range[1])
    ]
    if len(keys) < 3:
        raise ValueError('fewer than three grid points are eligible for holdout')

    candidates = keys
    if preserve_boundaries:
        temperatures = [key[0] for key in keys]
        columns = [key[1] for key in keys]
        T_min, T_max = min(temperatures), max(temperatures)
        N_min, N_max = min(columns), max(columns)
        candidates = [
            key for key in keys
            if key[0] not in (T_min, T_max) and key[1] not in (N_min, N_max)
        ]
    n_holdout = max(1, int(round(fraction * len(keys))))
    if n_holdout >= len(candidates):
        raise ValueError(
            f'cannot select {n_holdout} holdouts from {len(candidates)} '
            'non-boundary candidates'
        )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(candidates), size=n_holdout, replace=False)
    return sorted(candidates[index] for index in indices)


def tune_pca_components(
    models,
    holdout_keys,
    wav_range,
    candidates,
    seed=42,
    tolerance=0.05,
):
    """Choose PCA size using spectra excluded from surrogate training.

    One PCA decomposition is fitted to the non-holdout, peak-normalised model
    spectra. Each candidate is scored by its RMS reconstruction error on the
    unseen holdout spectra. The smallest candidate within ``tolerance`` of the
    best score is selected, avoiding unnecessary MLP output dimensions when
    the reconstruction improvement has already plateaued.

    Returns:
        dict: ``selected``, per-candidate ``rmse``, and explained variance.
    """
    from sklearn.decomposition import PCA as _PCA

    candidate_values = sorted({int(value) for value in candidates if int(value) > 0})
    if not candidate_values:
        raise ValueError('candidates must contain at least one positive integer')
    if tolerance < 0:
        raise ValueError('tolerance must be non-negative')

    holdout_set = set(map(tuple, holdout_keys))
    train_keys = [tuple(key) for key in models if tuple(key) not in holdout_set]
    valid_keys = [tuple(key) for key in models if tuple(key) in holdout_set]
    if not train_keys or not valid_keys:
        raise ValueError('PCA tuning requires both training and holdout spectra')

    reference_wav = np.asarray(models[train_keys[0]]['wavelength'])
    wav_mask = (reference_wav >= wav_range[0]) & (reference_wav <= wav_range[1])
    wav = reference_wav[wav_mask]
    if len(wav) < 2:
        raise ValueError(f'wav_range={wav_range} selects fewer than two channels')

    def _normalised_matrix(keys):
        spectra = []
        for key in keys:
            item = models[key]
            flux = np.interp(wav, item['wavelength'], item['flux']).astype(np.float32)
            peak = float(np.max(np.abs(flux)))
            spectra.append(flux / peak if peak > 0 else flux)
        return np.asarray(spectra, dtype=np.float32)

    train_flux = _normalised_matrix(train_keys)
    valid_flux = _normalised_matrix(valid_keys)
    max_components = min(max(candidate_values), len(train_flux) - 1, train_flux.shape[1])
    usable = [value for value in candidate_values if value <= max_components]
    if not usable:
        raise ValueError(
            f'all PCA candidates exceed usable maximum {max_components}'
        )

    pca = _PCA(n_components=max(usable), svd_solver='randomized', random_state=seed)
    pca.fit(train_flux)
    valid_coeff = pca.transform(valid_flux)
    scores = {}
    explained = {}
    for n_components in usable:
        reconstructed = (
            valid_coeff[:, :n_components] @ pca.components_[:n_components]
            + pca.mean_
        )
        scores[n_components] = float(np.sqrt(np.mean((reconstructed - valid_flux) ** 2)))
        explained[n_components] = float(
            np.sum(pca.explained_variance_ratio_[:n_components])
        )

    best_rmse = min(scores.values())
    selected = min(
        value for value in usable if scores[value] <= best_rmse * (1.0 + tolerance)
    )
    return {
        'selected': selected,
        'rmse': scores,
        'explained_variance': explained,
        'n_train': len(train_keys),
        'n_holdout': len(valid_keys),
        'wav_range': tuple(map(float, wav_range)),
    }


# ===========================================================================
# Dataset generation
# ===========================================================================

def generate_pre_training_set(
    mol,
    models,
    output_path,
    wav_out=None,
    noise_scale=0.0,
    n_noise=1,
    seed=None,
    holdout_keys=None,
):
    """
    Build a pre-training CSV by enumerating every (T, logN) grid point for
    one molecule.

    A = 1 is fixed; spectra are peak-normalised so net_shape always predicts
    the unit-peak spectral shape.  Optionally repeats each grid point n_noise
    times with independent noise realisations.  Call once per molecule.

    Args:
        mol (str): molecule name used as column prefix (e.g. 'H2O', 'C2H2')
        models (dict): output of load_model_grid for this molecule,
                       keyed by (T, logN) with 'wavelength' and 'flux' arrays
        output_path (str): destination CSV path
        wav_out (np.ndarray or None): common wavelength grid in µm; defaults
                                      to the wavelength axis of the first model
        noise_scale (float): Gaussian noise std as a fraction of each model's
                             peak flux magnitude; 0.0 for noise-free output (default 0.0)
        n_noise (int): number of independent noise realisations per grid point;
                       total rows = n_grid_points × n_noise (default 1)
        seed (int or None): random seed for reproducibility
        holdout_keys (iterable or None): (T, logN) tuples to exclude from the
                                         CSV so they can serve as unseen test
                                         points in validate_nt_holdout; the
                                         excluded count is printed (default None)

    Returns:
        (pd.DataFrame): the saved DataFrame
    """
    rng = np.random.default_rng(seed)

    if wav_out is None:
        wav_out = next(iter(models.values()))['wavelength']
    wav_cols = [f'wav_{w:.6f}' for w in wav_out]

    holdout_set = set(map(tuple, holdout_keys)) if holdout_keys is not None else set()
    keys   = [k for k in models.keys() if tuple(k) not in holdout_set]
    if holdout_set:
        n_excluded = len(models) - len(keys)
        print(f'  [{mol}] {n_excluded} grid points excluded as holdout '
              f'({len(keys)} kept for pretraining)')
    fluxes = np.stack([
        np.interp(wav_out, models[k]['wavelength'], models[k]['flux'])
        for k in keys
    ])

    rows = []
    for _ in range(n_noise):
        for idx, (T, logN) in enumerate(keys):
            flux = fluxes[idx].copy()
            peak = np.abs(flux).max()
            if peak > 0:
                if noise_scale > 0:
                    flux += rng.normal(0, noise_scale * peak, size=len(flux))
                norm_flux  = flux / peak
                log10_peak = np.log10(float(peak))
            else:
                norm_flux  = flux          # all zeros
                log10_peak = -30.0         # sentinel for dark models
            row = {
                f'{mol}_T':          T,
                f'{mol}_logN':       logN,
                f'{mol}_A':          1.0,
                f'{mol}_log10_peak': log10_peak,
            }
            row.update(dict(zip(wav_cols, norm_flux)))
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'[pretrain CSV] {len(df)} rows ({len(keys)} grid pts × {n_noise} noise) '
          f'→ {output_path}')
    return df


# ===========================================================================
# MLP architecture
# ===========================================================================

class MLP(nn.Module):
    """
    Simple fully-connected MLP: Linear → ReLU per hidden layer, then Linear output.

    Args:
        n_in   (int):   number of input features
        n_out  (int):   number of output features
        hidden (tuple): sizes of hidden layers (default (64, 128, 64))
    """
    def __init__(self, n_in, n_out, hidden=(64, 128, 64)):
        super().__init__()
        layers = []
        prev   = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===========================================================================
# Forward-model pretraining
# ===========================================================================

def pretrain_forward_model(
    mol,
    pretrain_csv=None,
    wav_range=(11.0, 19.0),
    device=None,
    hidden=(64, 128, 64),
    n_epochs=5000,
    batch_size=128,
    lr=1e-4,
    weight_decay=0.0,
    model_path=None,
    seed=42,
    n_pca=15,
    early_stopping_patience=500,
    T_range=None,
    logN_range=None,
    peak_normalization='csv',
    shape_loss='standardized',
    scheduler_patience=10,
    min_lr=0.0,
    temperature_transform='linear',
    shape_edge_ranges=None,
    shape_edge_weight=10.0,
):
    """
    Train (or load) the two-MLP forward model for one molecule.

    Two sub-models are trained independently from the same pretrain CSV.
    ``net_shape`` maps (T, logN) to n_pca PCA coefficients of the
    peak-normalised spectral shape; ``net_peak`` maps (T, logN) to
    log10(peak flux).

    If model_path already exists the checkpoint is loaded and training is
    skipped entirely.  After training the checkpoint is saved to model_path.

    Args:
        mol (str): molecule name matching column prefix in pretrain_csv
                   (e.g. 'H2O', 'C2H2')
        pretrain_csv (str or None): path to the CSV produced by
                                    generate_pre_training_set. May be None when
                                    loading a self-contained checkpoint.
        wav_range (tuple): (min_wav, max_wav) µm to select training channels;
                           use the molecule's actual emission range to drop
                           zero-only channels before PCA
                           (e.g. (12.0, 16.5) for C2H2/HCN/CO2, (11.0, 19.0)
                           for H2O)
        device (torch.device or None): defaults to CUDA if available
        hidden (tuple): hidden layer sizes for both MLPs (default (64, 128, 64))
        n_epochs (int): maximum training epochs (default 5000)
        batch_size (int): mini-batch size (default 128)
        lr (float): Adam learning rate (default 1e-4)
        weight_decay (float): Adam L2 regularisation (default 0.0)
        model_path (str or None): .pt checkpoint path; skip training if it
                                  exists, save there after training
        seed (int): random seed for train/val split (default 42)
        n_pca (int): number of PCA components for spectral shape compression;
                     set to None or 0 to disable PCA (default 15)
        early_stopping_patience (int): stop each sub-model when its validation
                                       loss does not improve for this many epochs;
                                       0 disables early stopping (default 500)
        T_range (tuple or None): (T_min, T_max) K; if set, only rows in this
                                  temperature range are used; useful for
                                  hot/warm two-component splits (default None)
        logN_range (tuple or None): (logN_min, logN_max); if set, only rows in
                                     this column-density range are used
        peak_normalization (str): 'csv' preserves the input representation;
            'window' renormalizes selected channels and adjusts log-peak so
            their product remains the physical spectrum. Empty windows have
            no defined log-peak and are excluded from the peak-head loss.
            Use 'window' for H2O.
        shape_loss (str): 'standardized' weights coefficients equally;
            'spectral' minimizes spectral MSE; 'spectral_edge' additionally
            weights selected wavelength intervals. Inference is unchanged.
        scheduler_patience (int): epochs without improvement before reducing LR.
        min_lr (float): lower learning-rate bound for both network heads.
        temperature_transform (str): 'linear' or 'log10' before standardization.
            Stored in the checkpoint; inference still accepts temperature in K.
        shape_edge_ranges (list): wavelength intervals emphasized by the edge loss.
        shape_edge_weight (float): extra weight applied within those intervals.

    Returns:
        (dict): with keys
            net_shape          (nn.Module)      : shape MLP in eval mode
            net_peak           (nn.Module)      : peak MLP in eval mode
            xp_sc              (StandardScaler) : fitted on transformed inputs
            yp_sc_shape        (StandardScaler) : fitted on PCA coefficients
            yp_sc_peak         (StandardScaler) : fitted on log10(peak) values
            pca                (PCA or None)    : fitted sklearn PCA, or None
            train_losses_shape (list)           : per-epoch train loss, shape MLP
            val_losses_shape   (list)           : per-epoch val   loss, shape MLP
            train_losses_peak  (list)           : per-epoch train loss, peak  MLP
            val_losses_peak    (list)           : per-epoch val   loss, peak  MLP
            wav                (np.ndarray)     : wavelength axis for wav_range
            X_pre_v            (np.ndarray)     : validation (T, logN), physical
            Y_pre_v            (np.ndarray)     : validation spectra (peak-norm.)
            X_pre_v_t          (torch.Tensor)   : standardised val inputs on device
            log10p_v           (np.ndarray)     : true log10(peak) for val samples
    """
    from sklearn.decomposition import PCA as _PCA

    if peak_normalization not in ('csv', 'window'):
        raise ValueError("peak_normalization must be 'csv' or 'window'")
    if shape_loss not in ('standardized', 'spectral', 'spectral_edge'):
        raise ValueError("shape_loss must be 'standardized', 'spectral', or 'spectral_edge'")
    if temperature_transform not in ('linear', 'log10'):
        raise ValueError("temperature_transform must be 'linear' or 'log10'")

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Self-contained checkpoints can be loaded without the training CSV.
    if pretrain_csv is None:
        if not model_path or not os.path.exists(model_path):
            raise ValueError(
                'pretrain_csv is required when training or when model_path '
                'does not point to an existing checkpoint'
            )
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        missing = [key for key in ('xp_sc', 'wav') if key not in ckpt]
        if missing:
            raise ValueError(
                f'Checkpoint {model_path!r} is missing {missing}; pass its '
                'matching pretrain_csv and wav_range to load this legacy checkpoint'
            )

        state_shape = ckpt['model_state_shape']
        wkeys = sorted(k for k in state_shape if k.endswith('.weight'))
        hidden_ckpt = tuple(state_shape[k].shape[0] for k in wkeys[:-1])
        net_shape = MLP(
            n_in=2, n_out=state_shape[wkeys[-1]].shape[0], hidden=hidden_ckpt
        ).to(device)
        net_shape.load_state_dict(state_shape)
        net_shape.eval()

        state_peak = ckpt['model_state_peak']
        wkeys_p = sorted(k for k in state_peak if k.endswith('.weight'))
        hidden_p = tuple(state_peak[k].shape[0] for k in wkeys_p[:-1])
        net_peak = MLP(n_in=2, n_out=1, hidden=hidden_p).to(device)
        net_peak.load_state_dict(state_peak)
        net_peak.eval()

        print(f'  [{mol}] loaded from {model_path}')
        return dict(
            net_shape=net_shape,
            net_peak=net_peak,
            xp_sc=ckpt['xp_sc'],
            yp_sc_shape=ckpt['yp_sc_shape'],
            yp_sc_peak=ckpt['yp_sc_peak'],
            pca=ckpt.get('pca', None),
            train_losses_shape=ckpt['train_losses_shape'],
            val_losses_shape=ckpt['val_losses_shape'],
            train_losses_peak=ckpt['train_losses_peak'],
            val_losses_peak=ckpt['val_losses_peak'],
            wav=np.asarray(ckpt['wav']),
            X_pre_v=np.empty((0, 2), dtype=np.float32),
            Y_pre_v=np.empty((0, len(ckpt['wav'])), dtype=np.float32),
            X_pre_v_t=torch.empty((0, 2), dtype=torch.float32, device=device),
            log10p_v=np.empty(0, dtype=np.float32),
            T_range=ckpt.get('T_range', None),
            logN_range=ckpt.get('logN_range', None),
            wav_range=ckpt.get('wav_range', None),
            model_path=model_path,
            peak_normalization=ckpt.get('peak_normalization', 'csv'),
            shape_loss=ckpt.get('shape_loss', 'standardized'),
            temperature_transform=ckpt.get('temperature_transform', 'linear'),
            peak_loss_excludes_dark=ckpt.get('peak_loss_excludes_dark', False),
            n_dark_training=ckpt.get('n_dark_training', 0),
            n_dark_validation=ckpt.get('n_dark_validation', 0),
            shape_edge_ranges=ckpt.get('shape_edge_ranges', []),
            shape_edge_weight=ckpt.get('shape_edge_weight', 1.0),
        )

    # --- Load pretrain CSV and select wavelength channels ---
    df_pre        = pd.read_csv(pretrain_csv)
    pre_flux_cols = [c for c in df_pre.columns
                     if c.startswith('wav_')
                     and wav_range[0] <= float(c[4:]) <= wav_range[1]]
    if not pre_flux_cols:
        raise ValueError(
            f'[{mol}] wav_range={wav_range} selects zero spectral columns '
            f'from {pretrain_csv}'
        )
    wav    = np.array([float(c[4:]) for c in pre_flux_cols])
    n_spec = len(pre_flux_cols)

    X_pre      = df_pre[[f'{mol}_T', f'{mol}_logN']].to_numpy(dtype=np.float32)
    Y_pre      = df_pre[pre_flux_cols].to_numpy(dtype=np.float32)
    log10_peak = df_pre[f'{mol}_log10_peak'].to_numpy(dtype=np.float32)
    if peak_normalization == 'window':
        window_peak = np.max(np.abs(Y_pre), axis=1)
        safe_peak = np.where(window_peak > 0, window_peak, 1.0)
        Y_pre = Y_pre / safe_peak[:, None]
        log10_peak = log10_peak + np.log10(safe_peak)

    # Optional temperature filter (e.g. for hot/warm two-component training)
    if T_range is not None:
        mask       = (X_pre[:, 0] >= T_range[0]) & (X_pre[:, 0] <= T_range[1])
        X_pre      = X_pre[mask]
        Y_pre      = Y_pre[mask]
        log10_peak = log10_peak[mask]
        print(f'  [{mol}] T_range={T_range}: {mask.sum()}/{len(mask)} samples kept')

    if logN_range is not None:
        mask       = (X_pre[:, 1] >= logN_range[0]) & (X_pre[:, 1] <= logN_range[1])
        X_pre      = X_pre[mask]
        Y_pre      = Y_pre[mask]
        log10_peak = log10_peak[mask]
        print(f'  [{mol}] logN_range={logN_range}: {mask.sum()}/{len(mask)} samples kept')

    if len(X_pre) < 2:
        raise ValueError(
            f'[{mol}] fewer than two samples remain after applying '
            f'T_range={T_range} and logN_range={logN_range}'
        )
    n_validation = max(1, int(np.ceil(0.1 * len(X_pre))))
    max_pca = min(len(X_pre) - n_validation, len(pre_flux_cols))
    if n_pca and n_pca > max_pca:
        raise ValueError(
            f'[{mol}] n_pca={n_pca} exceeds the usable limit '
            f'{max_pca} for '
            f'wav_range={wav_range}, T_range={T_range}, '
            f'logN_range={logN_range}'
        )

    X_tr, X_v, Y_tr, Y_v, p_tr, p_v = train_test_split(
        X_pre, Y_pre, log10_peak, test_size=0.1, random_state=seed)

    X_tr_input, X_v_input = X_tr.copy(), X_v.copy()
    if temperature_transform == 'log10':
        if np.any(X_pre[:, 0] <= 0):
            raise ValueError('log10 temperature requires positive temperatures')
        X_tr_input[:, 0] = np.log10(X_tr_input[:, 0])
        X_v_input[:, 0] = np.log10(X_v_input[:, 0])
    xp_sc = StandardScaler().fit(X_tr_input)

    # --- PCA compression on peak-normalised spectral shapes ---
    if n_pca and n_pca < n_spec:
        pca   = _PCA(n_components=n_pca, random_state=seed)
        Z_tr  = pca.fit_transform(Y_tr)
        Z_v   = pca.transform(Y_v)
        cumvar = pca.explained_variance_ratio_.cumsum()
        print(f'  [{mol}] wav_range={wav_range}  n_spec={n_spec} '
              f'→ PCA n={n_pca}  cumul. var={cumvar[-1]*100:.4f}%')
    else:
        pca  = None
        Z_tr = Y_tr
        Z_v  = Y_v

    n_shape     = Z_tr.shape[1]
    yp_sc_shape = StandardScaler().fit(Z_tr)
    peak_tr_valid = np.any(Y_tr != 0, axis=1) if peak_normalization == 'window' else np.ones(len(Y_tr), dtype=bool)
    peak_v_valid = np.any(Y_v != 0, axis=1) if peak_normalization == 'window' else np.ones(len(Y_v), dtype=bool)
    if not peak_tr_valid.any() or not peak_v_valid.any():
        raise ValueError('training and validation splits must each contain a nonzero spectrum')
    if not peak_tr_valid.all() or not peak_v_valid.all():
        print(f'  [{mol}] excluding {(~peak_tr_valid).sum()} training and '
              f'{(~peak_v_valid).sum()} validation empty windows from peak loss')
    yp_sc_peak = StandardScaler().fit(p_tr[peak_tr_valid].reshape(-1, 1))
    peak_v_valid_t = torch.as_tensor(peak_v_valid, dtype=torch.bool, device=device)

    X_v_t  = torch.from_numpy(xp_sc.transform(X_v_input).astype(np.float32)).to(device)
    Z_v_t  = torch.from_numpy(yp_sc_shape.transform(Z_v).astype(np.float32)).to(device)
    p_v_t  = torch.from_numpy(yp_sc_peak.transform(p_v.reshape(-1, 1)).astype(np.float32)).to(device)

    # --- Load from checkpoint if it exists ---
    if model_path and os.path.exists(model_path):
        ckpt        = torch.load(model_path, map_location=device, weights_only=False)
        expected_metadata = {
            'wav_range': tuple(map(float, wav_range)),
            'T_range': None if T_range is None else tuple(map(float, T_range)),
            'logN_range': None if logN_range is None else tuple(map(float, logN_range)),
            'n_pca': None if not n_pca or n_pca >= n_spec else int(n_pca),
            'hidden': tuple(hidden),
            'peak_normalization': peak_normalization,
            'shape_loss': shape_loss,
            'temperature_transform': temperature_transform,
            'peak_loss_excludes_dark': peak_normalization == 'window',
            'shape_edge_ranges': [] if shape_edge_ranges is None else
                [tuple(map(float, r)) for r in shape_edge_ranges],
            'shape_edge_weight': float(shape_edge_weight) if shape_loss == 'spectral_edge' else 1.0,
        }
        incompatible = []
        for key, expected in expected_metadata.items():
            if key not in ckpt:
                defaults = {'peak_normalization': 'csv', 'shape_loss': 'standardized',
                            'temperature_transform': 'linear',
                            'peak_loss_excludes_dark': False,
                            'shape_edge_ranges': [], 'shape_edge_weight': 1.0}
                if key not in defaults:
                    continue
                stored = defaults[key]
            else:
                stored = ckpt[key]
            if key == 'shape_edge_ranges':
                matches = np.allclose(np.asarray(stored, dtype=float).reshape(-1, 2),
                                      np.asarray(expected, dtype=float).reshape(-1, 2))
            elif isinstance(expected, tuple):
                matches = stored is not None and np.allclose(stored, expected)
            else:
                matches = stored == expected
            if not matches:
                incompatible.append(f'{key}: stored={stored}, requested={expected}')
        if incompatible:
            details = '; '.join(incompatible)
            raise ValueError(
                f'checkpoint {model_path!r} is incompatible with this training '
                f'configuration ({details}). Move the old checkpoint or choose '
                'a new model_path before retraining.'
            )
        xp_sc       = ckpt.get('xp_sc', xp_sc)
        pca         = ckpt.get('pca', None)
        yp_sc_shape = ckpt['yp_sc_shape']
        yp_sc_peak  = ckpt['yp_sc_peak']
        wav         = np.asarray(ckpt.get('wav', wav))

        # Rebuild validation targets with the checkpoint preprocessing rather
        # than the newly fitted temporary scalers/PCA.
        Z_v_ckpt = pca.transform(Y_v) if pca is not None else Y_v
        X_v_t = torch.from_numpy(
            xp_sc.transform(X_v_input).astype(np.float32)
        ).to(device)
        Z_v_t = torch.from_numpy(
            yp_sc_shape.transform(Z_v_ckpt).astype(np.float32)
        ).to(device)
        p_v_t = torch.from_numpy(
            yp_sc_peak.transform(p_v.reshape(-1, 1)).astype(np.float32)
        ).to(device)

        state_shape  = ckpt['model_state_shape']
        wkeys        = sorted(k for k in state_shape if k.endswith('.weight'))
        hidden_ckpt  = tuple(state_shape[k].shape[0] for k in wkeys[:-1])
        n_out_shape  = state_shape[wkeys[-1]].shape[0]
        net_shape    = MLP(n_in=2, n_out=n_out_shape, hidden=hidden_ckpt).to(device)
        net_shape.load_state_dict(state_shape)
        net_shape.eval()

        state_peak   = ckpt['model_state_peak']
        wkeys_p      = sorted(k for k in state_peak if k.endswith('.weight'))
        hidden_p     = tuple(state_peak[k].shape[0] for k in wkeys_p[:-1])
        net_peak     = MLP(n_in=2, n_out=1, hidden=hidden_p).to(device)
        net_peak.load_state_dict(state_peak)
        net_peak.eval()

        train_losses_shape = ckpt['train_losses_shape']
        val_losses_shape   = ckpt['val_losses_shape']
        train_losses_peak  = ckpt['train_losses_peak']
        val_losses_peak    = ckpt['val_losses_peak']
        print(f'  [{mol}] loaded from {model_path}')

    else:
        # --- Train from scratch ---
        torch.manual_seed(seed)
        net_shape = MLP(n_in=2, n_out=n_shape, hidden=hidden).to(device)
        net_peak  = MLP(n_in=2, n_out=1,        hidden=hidden).to(device)

        X_tr_s = torch.from_numpy(xp_sc.transform(X_tr_input).astype(np.float32))
        Z_tr_s = torch.from_numpy(yp_sc_shape.transform(Z_tr).astype(np.float32))
        p_tr_s = torch.from_numpy(yp_sc_peak.transform(p_tr.reshape(-1, 1)).astype(np.float32))

        loader = DataLoader(
            TensorDataset(X_tr_s, Z_tr_s, p_tr_s, torch.as_tensor(peak_tr_valid)),
            batch_size=batch_size, shuffle=True,
        )
        criterion   = nn.MSELoss()
        # PCA components are orthonormal: weighting squared standardized
        # coefficient errors by coefficient variance gives spectral MSE,
        # up to a constant. No large decoded-spectrum tensor is needed.
        shape_weights = np.asarray(yp_sc_shape.scale_) ** 2
        if shape_loss == 'standardized':
            shape_weights = np.ones_like(shape_weights)
        if shape_loss == 'spectral_edge':
            if pca is None:
                raise ValueError('spectral_edge currently requires PCA compression')
            if not shape_edge_ranges:
                raise ValueError("shape_edge_ranges is required for shape_loss='spectral_edge'")
            channel_weights = np.ones(n_spec, dtype=np.float64)
            edge_mask = np.zeros(n_spec, dtype=bool)
            for edge_lo, edge_hi in shape_edge_ranges:
                edge_mask |= (wav >= edge_lo) & (wav <= edge_hi)
            if not np.any(edge_mask):
                raise ValueError(f'shape_edge_ranges select no channels in {wav_range}')
            channel_weights[edge_mask] *= float(shape_edge_weight)
            projection = pca.components_ * channel_weights[None, :]
            weight_matrix = projection @ pca.components_.T
            weight_matrix *= np.asarray(yp_sc_shape.scale_)[:, None]
            weight_matrix *= np.asarray(yp_sc_shape.scale_)[None, :]
            weight_matrix /= np.sum(channel_weights)
            weight_matrix = torch.as_tensor(weight_matrix, dtype=torch.float32, device=device)
        else:
            shape_weights = torch.as_tensor(
                shape_weights / np.mean(shape_weights), dtype=torch.float32, device=device
            )

        def shape_criterion(prediction, target):
            if shape_loss == 'spectral_edge':
                delta = (prediction - target) * torch.as_tensor(
                    yp_sc_shape.scale_, dtype=torch.float32, device=device
                )
                return torch.einsum('bi,ij,bj->', delta, weight_matrix, delta) / len(delta)
            return ((prediction - target).square() * shape_weights).mean()

        opt_shape   = optim.Adam(net_shape.parameters(), lr=lr, weight_decay=weight_decay)
        opt_peak    = optim.Adam(net_peak.parameters(),  lr=lr, weight_decay=weight_decay)
        sched_shape = optim.lr_scheduler.ReduceLROnPlateau(
            opt_shape, patience=scheduler_patience, factor=0.5, min_lr=min_lr)
        sched_peak = optim.lr_scheduler.ReduceLROnPlateau(
            opt_peak, patience=scheduler_patience, factor=0.5, min_lr=min_lr)

        train_losses_shape, val_losses_shape = [], []
        train_losses_peak,  val_losses_peak  = [], []
        best_val_shape, best_val_peak = float('inf'), float('inf')
        best_state_shape, best_state_peak = None, None
        es_ctr_shape,  es_ctr_peak   = 0, 0
        stopped_shape, stopped_peak  = False, False

        print(f'  [{mol}] training for up to {n_epochs} epochs '
              f'(patience={early_stopping_patience}) ...')
        for epoch in range(1, n_epochs + 1):
            net_shape.train()
            net_peak.train()
            bl_shape, bl_peak = [], []
            for xb, zb, pb, peak_valid in loader:
                xb, zb, pb = xb.to(device), zb.to(device), pb.to(device)
                peak_valid = peak_valid.to(device)
                if not stopped_shape:
                    opt_shape.zero_grad()
                    loss_s = shape_criterion(net_shape(xb), zb)
                    loss_s.backward()
                    opt_shape.step()
                    bl_shape.append(loss_s.item())
                if not stopped_peak and peak_valid.any():
                    opt_peak.zero_grad()
                    loss_p = criterion(net_peak(xb[peak_valid]), pb[peak_valid])
                    loss_p.backward()
                    opt_peak.step()
                    bl_peak.append(loss_p.item())

            net_shape.eval()
            net_peak.eval()
            with torch.no_grad():
                vl_shape = shape_criterion(net_shape(X_v_t), Z_v_t).item()
                vl_peak = criterion(net_peak(X_v_t[peak_v_valid_t]), p_v_t[peak_v_valid_t]).item()
            sched_shape.step(vl_shape)
            sched_peak.step(vl_peak)
            if bl_shape:
                train_losses_shape.append(float(np.mean(bl_shape)))
            if bl_peak:
                train_losses_peak.append(float(np.mean(bl_peak)))
            val_losses_shape.append(vl_shape)
            val_losses_peak.append(vl_peak)

            if epoch % 100 == 0:
                lr_s = opt_shape.param_groups[0]['lr']
                lr_p = opt_peak.param_groups[0]['lr']
                ts   = train_losses_shape[-1] if train_losses_shape else float('nan')
                tp   = train_losses_peak[-1]  if train_losses_peak  else float('nan')
                print(f'    ep {epoch:5d}  '
                      f'shape tr={ts:.4f} val={vl_shape:.4f} lr={lr_s:.1e}  |  '
                      f'peak  tr={tp:.4f} val={vl_peak:.4f} lr={lr_p:.1e}')

            if early_stopping_patience > 0:
                if not stopped_shape:
                    if vl_shape < best_val_shape:
                        best_val_shape = vl_shape
                        best_state_shape = copy.deepcopy(net_shape.state_dict())
                        es_ctr_shape   = 0
                    else:
                        es_ctr_shape  += 1
                    if es_ctr_shape >= early_stopping_patience:
                        print(f'    shape: early stop ep={epoch}  best_val={best_val_shape:.4f}')
                        stopped_shape = True
                if not stopped_peak:
                    if vl_peak < best_val_peak:
                        best_val_peak = vl_peak
                        best_state_peak = copy.deepcopy(net_peak.state_dict())
                        es_ctr_peak   = 0
                    else:
                        es_ctr_peak  += 1
                    if es_ctr_peak >= early_stopping_patience:
                        print(f'    peak:  early stop ep={epoch}  best_val={best_val_peak:.4f}')
                        stopped_peak = True
                if stopped_shape and stopped_peak:
                    break

        # Early stopping should return the best validation epoch rather than
        # whichever weights happened to be present when patience expired.
        if best_state_shape is not None:
            net_shape.load_state_dict(best_state_shape)
        if best_state_peak is not None:
            net_peak.load_state_dict(best_state_peak)
        net_shape.eval()
        net_peak.eval()

        # Save checkpoint
        if model_path:
            os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
            torch.save({
                'model_state_shape':  net_shape.state_dict(),
                'model_state_peak':   net_peak.state_dict(),
                'train_losses_shape': train_losses_shape,
                'val_losses_shape':   val_losses_shape,
                'train_losses_peak':  train_losses_peak,
                'val_losses_peak':    val_losses_peak,
                'pca':                pca,
                'xp_sc':              xp_sc,
                'yp_sc_shape':        yp_sc_shape,
                'yp_sc_peak':         yp_sc_peak,
                'wav':                wav,
                'mol':                mol,
                'wav_range':          tuple(wav_range),
                'T_range':            None if T_range is None else tuple(T_range),
                'logN_range':         None if logN_range is None else tuple(logN_range),
                'n_pca':              None if pca is None else pca.n_components_,
                'hidden':             hidden,
                'peak_normalization': peak_normalization,
                'shape_loss':         shape_loss,
                'scheduler_patience': scheduler_patience,
                'min_lr':             min_lr,
                'seed':               seed,
                'temperature_transform': temperature_transform,
                'peak_loss_excludes_dark': peak_normalization == 'window',
                'shape_edge_ranges': [] if shape_edge_ranges is None else
                    [tuple(map(float, r)) for r in shape_edge_ranges],
                'shape_edge_weight': float(shape_edge_weight) if shape_loss == 'spectral_edge' else 1.0,
                'n_dark_training': int((~peak_tr_valid).sum()),
                'n_dark_validation': int((~peak_v_valid).sum()),
            }, model_path)
            print(f'  [{mol}] saved → {model_path}')

    return dict(
        net_shape=net_shape,          net_peak=net_peak,
        xp_sc=xp_sc,
        yp_sc_shape=yp_sc_shape,      yp_sc_peak=yp_sc_peak,
        pca=pca,
        train_losses_shape=train_losses_shape, val_losses_shape=val_losses_shape,
        train_losses_peak=train_losses_peak,   val_losses_peak=val_losses_peak,
        wav=wav,
        X_pre_v=X_v,    Y_pre_v=Y_v,
        X_pre_v_t=X_v_t, log10p_v=p_v,
        T_range=None if T_range is None else tuple(T_range),
        logN_range=None if logN_range is None else tuple(logN_range),
        wav_range=tuple(wav_range),
        model_path=model_path,
        peak_normalization=peak_normalization,
        shape_loss=shape_loss,
        temperature_transform=temperature_transform,
        peak_loss_excludes_dark=peak_normalization == 'window',
        n_dark_training=int((~peak_tr_valid).sum()),
        n_dark_validation=int((~peak_v_valid).sum()),
        shape_edge_ranges=[] if shape_edge_ranges is None else [tuple(map(float, r)) for r in shape_edge_ranges],
        shape_edge_weight=float(shape_edge_weight) if shape_loss == 'spectral_edge' else 1.0,
    )


def _range_token(bounds):
    """Filesystem-safe token for a numeric two-value range."""
    def _number(value):
        return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')
    return f'{_number(bounds[0])}to{_number(bounds[1])}'


def train_model_bank(
    mol,
    pretrain_csv,
    model_dir,
    wavelength_ranges,
    logN_ranges,
    T_ranges,
    device=None,
    hidden=(64, 128, 64),
    n_epochs=5000,
    batch_size=128,
    lr=1e-4,
    weight_decay=0.0,
    seed=42,
    n_pca=15,
    early_stopping_patience=500,
    manifest_path=None,
    peak_normalization='csv',
    shape_loss='standardized',
    shape_edge_weight=10.0,
    scheduler_patience=10,
    min_lr=0.0,
    temperature_transform='linear',
):
    """Train a bank of overlapping forward-surrogate tiles for one molecule.

    Every Cartesian-product combination of wavelength, logN, and temperature
    ranges is trained from one full pretraining CSV.  Checkpoints are
    self-contained and a JSON manifest records their coverage for fitting.

    Existing checkpoint files are loaded rather than retrained.  Use a new
    ``model_dir`` or remove an obsolete checkpoint intentionally when changing
    training data or hyperparameters.

    ``n_pca`` may be one integer shared by every wavelength range or a sequence
    with one value per wavelength range. The latter supports independently
    tuned spectral complexity while retaining shared physical coverage.
    ``peak_normalization``, ``shape_loss``, ``scheduler_patience``,
    ``min_lr`` and ``temperature_transform`` are forwarded to every checkpoint.
    A per-wavelength ``shape_loss`` sequence can assign ``spectral_edge`` to
    tiles whose overlaps need extra accuracy. Their complete overlaps with
    adjacent windows receive ``shape_edge_weight`` additional emphasis.

    Returns:
        dict: manifest data plus in-memory ``models`` and ``model_bank``.
    """
    wavelength_ranges = [tuple(map(float, item)) for item in wavelength_ranges]
    logN_ranges = [tuple(map(float, item)) for item in logN_ranges]
    T_ranges = [tuple(map(float, item)) for item in T_ranges]
    if np.isscalar(n_pca):
        n_pca_by_wavelength = [int(n_pca)] * len(wavelength_ranges)
    else:
        n_pca_by_wavelength = [int(value) for value in n_pca]
        if len(n_pca_by_wavelength) != len(wavelength_ranges):
            raise ValueError(
                'n_pca sequence must have one value per wavelength range: '
                f'{len(n_pca_by_wavelength)} != {len(wavelength_ranges)}'
            )
    if isinstance(shape_loss, str):
        shape_loss_by_wavelength = [shape_loss] * len(wavelength_ranges)
    else:
        shape_loss_by_wavelength = list(shape_loss)
        if len(shape_loss_by_wavelength) != len(wavelength_ranges):
            raise ValueError('shape_loss sequence must match wavelength_ranges')
    for label, ranges in (
        ('wavelength', wavelength_ranges),
        ('logN', logN_ranges),
        ('temperature', T_ranges),
    ):
        if not ranges or any(lo >= hi for lo, hi in ranges):
            raise ValueError(f'invalid {label} ranges: {ranges}')
        ordered = sorted(ranges)
        covered_through = ordered[0][1]
        for current in ordered[1:]:
            if current[0] > covered_through:
                raise ValueError(
                    f'{label} ranges leave an uncovered gap between '
                    f'{covered_through:g} and {current[0]:g}'
                )
            covered_through = max(covered_through, current[1])

    model_dir = os.path.abspath(os.fspath(model_dir))
    os.makedirs(model_dir, exist_ok=True)
    if manifest_path is None:
        manifest_path = os.path.join(model_dir, f'{mol}_model_bank.json')
    manifest_path = os.path.abspath(os.fspath(manifest_path))

    tiles = []
    models = []
    total = len(wavelength_ranges) * len(logN_ranges) * len(T_ranges)
    tile_number = 0
    for wav_index, wav_range in enumerate(wavelength_ranges):
        tile_n_pca = n_pca_by_wavelength[wav_index]
        tile_shape_loss = shape_loss_by_wavelength[wav_index]
        edge_ranges = []
        if tile_shape_loss == 'spectral_edge':
            for other_index, other_range in enumerate(wavelength_ranges):
                if other_index != wav_index:
                    overlap = (max(wav_range[0], other_range[0]),
                               min(wav_range[1], other_range[1]))
                    if overlap[0] < overlap[1]:
                        edge_ranges.append(overlap)
        for logN_range in logN_ranges:
            for T_range in T_ranges:
                tile_number += 1
                filename = (
                    f'net_{mol}_w{_range_token(wav_range)}_'
                    f'n{_range_token(logN_range)}_t{_range_token(T_range)}.pt'
                )
                model_path = os.path.join(model_dir, filename)
                print(f'\n[{mol}] tile {tile_number}/{total}: {filename}')
                trained = pretrain_forward_model(
                    mol=mol,
                    pretrain_csv=pretrain_csv,
                    wav_range=wav_range,
                    device=device,
                    hidden=hidden,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    weight_decay=weight_decay,
                    model_path=model_path,
                    seed=seed,
                    n_pca=tile_n_pca,
                    early_stopping_patience=early_stopping_patience,
                    T_range=T_range,
                    logN_range=logN_range,
                    peak_normalization=peak_normalization,
                    shape_loss=tile_shape_loss,
                    scheduler_patience=scheduler_patience,
                    min_lr=min_lr,
                    temperature_transform=temperature_transform,
                    shape_edge_ranges=edge_ranges,
                    shape_edge_weight=shape_edge_weight,
                )
                tiles.append({
                    'path': os.path.relpath(model_path, os.path.dirname(manifest_path)),
                    'wav_range': list(wav_range),
                    'logN_range': list(logN_range),
                    'T_range': list(T_range),
                    'n_pca': tile_n_pca,
                })
                models.append(trained)

    manifest = {
        'schema_version': 1,
        'mol': mol,
        'pretrain_csv': os.path.abspath(os.fspath(pretrain_csv)),
        'n_pca': (
            n_pca_by_wavelength[0]
            if len(set(n_pca_by_wavelength)) == 1
            else n_pca_by_wavelength
        ),
        'hidden': list(hidden),
        'peak_normalization': peak_normalization,
        'shape_loss': (shape_loss_by_wavelength[0]
                       if len(set(shape_loss_by_wavelength)) == 1
                       else shape_loss_by_wavelength),
        'shape_edge_weight': shape_edge_weight,
        'temperature_transform': temperature_transform,
        'wavelength_ranges': [list(item) for item in wavelength_ranges],
        'logN_ranges': [list(item) for item in logN_ranges],
        'T_ranges': [list(item) for item in T_ranges],
        'tiles': tiles,
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    print(f'\n[{mol}] model-bank manifest saved → {manifest_path}')

    in_memory_tiles = [
        {
            'model': model,
            'path': os.path.abspath(os.path.join(os.path.dirname(manifest_path), spec['path'])),
            'wav_range': tuple(spec['wav_range']),
            'T_range': tuple(spec['T_range']),
            'logN_range': tuple(spec['logN_range']),
        }
        for spec, model in zip(tiles, models)
    ]
    model_bank = {
        'is_model_bank': True,
        'mol': mol,
        'manifest_path': manifest_path,
        'tiles': in_memory_tiles,
        'wav_ranges': wavelength_ranges,
        'T_ranges': T_ranges,
        'logN_ranges': logN_ranges,
    }
    return {
        **manifest,
        'manifest_path': manifest_path,
        'models': models,
        'model_bank': model_bank,
    }


# ===========================================================================
# Script entry point — edit CONFIGURATION block to customise
# ===========================================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # -----------------------------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------------------------

    # -- Paths ---------------------------------------------------------------
    _here        = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR     = os.path.join(_here, '..', '..')              # project root
    GRID_ROOT    = os.path.join(ROOT_DIR, 'Model_grids')
    DATA_DIR     = os.path.join(ROOT_DIR, 'Pretrain_grid')
    MODEL_DIR    = os.path.join(ROOT_DIR, 'Trained_model')
    FIG_DIR      = os.path.join(ROOT_DIR, 'figures', 'trainmodel')
    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR,   exist_ok=True)

    # -- Which molecules to train --------------------------------------------
    MOLECULES = ['H2O', 'C2H2', 'HCN', 'CO2']

    WAV_RANGES = {
        'H2O':  (11.0, 19.0),
        'C2H2': (12.0, 16.5),
        'HCN':  (12.0, 16.5),
        'CO2':  (12.0, 16.5),
    }

    N_PCA = {
        'H2O':  21,
        'C2H2': 15,
        'HCN':  15,
        'CO2':  15,
    }

    HIDDEN = {
        'H2O':  (64, 128, 64),
        'C2H2': (64, 128, 64),
        'HCN':  (64, 128, 64),
        'CO2':  (64, 128, 64),
    }

    N_EPOCHS               = 5000
    BATCH_SIZE             = 128
    LR                     = 1e-4
    WEIGHT_DECAY           = 0.0
    EARLY_STOPPING_PATIENCE = 500
    SEED                   = 42

    NOISE_SCALE_PRETRAIN = 0.0
    N_NOISE_PRETRAIN     = 1

    H2O_MODE      = 'single'
    H2O_HOT_T_MIN = 800

    # -----------------------------------------------------------------------
    # Load model grids
    # -----------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    all_models = {mol: load_model_grid(os.path.join(GRID_ROOT, mol))
                  for mol in MOLECULES}
    wav_full = next(iter(all_models['H2O'].values()))['wavelength']
    for mol, models in all_models.items():
        print(f'  {mol}: {len(models)} grid points')

    # -----------------------------------------------------------------------
    # Generate per-molecule pretrain CSVs (skip if already present)
    # -----------------------------------------------------------------------
    print('\n--- Generating pretrain CSVs ---')
    PRETRAIN_CSVS = {
        mol: os.path.join(DATA_DIR, f'pretrain_{mol}_11to19_filtered.csv')
        for mol in MOLECULES
    }
    for mol in MOLECULES:
        path = PRETRAIN_CSVS[mol]
        if not os.path.exists(path):
            generate_pre_training_set(
                mol, all_models[mol],
                output_path=path,
                wav_out=wav_full,
                noise_scale=NOISE_SCALE_PRETRAIN,
                n_noise=N_NOISE_PRETRAIN,
                seed=SEED,
            )
        else:
            print(f'  [{mol}] CSV exists, skipping → {path}')

    # -----------------------------------------------------------------------
    # Pretrain forward models
    # -----------------------------------------------------------------------
    print('\n--- Pretraining forward models ---')
    pretrained = {}

    _h2o_kw = dict(
        mol='H2O',
        pretrain_csv=PRETRAIN_CSVS['H2O'],
        wav_range=WAV_RANGES['H2O'],
        n_pca=N_PCA['H2O'],
        device=device,
        hidden=HIDDEN['H2O'],
        n_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        seed=SEED,
    )
    if H2O_MODE == 'two_component':
        print(f'\n[H2O] two-component mode  (hot T>={H2O_HOT_T_MIN} K / warm T<{H2O_HOT_T_MIN} K)')
        pretrained['H2O_hot'] = pretrain_forward_model(
            **_h2o_kw,
            T_range=(H2O_HOT_T_MIN, 99999),
            model_path=os.path.join(MODEL_DIR, 'net_H2O_hot_forward_11to19.pt'),
        )
        pretrained['H2O_warm'] = pretrain_forward_model(
            **_h2o_kw,
            T_range=(0, H2O_HOT_T_MIN - 1),
            model_path=os.path.join(MODEL_DIR, 'net_H2O_warm_forward_11to19.pt'),
        )
    else:
        print('\n[H2O] single-component mode')
        pretrained['H2O'] = pretrain_forward_model(
            **_h2o_kw,
            model_path=os.path.join(MODEL_DIR, 'net_H2O_forward_11to19.pt'),
        )

    for mol in ['C2H2', 'HCN', 'CO2']:
        print(f'\n[{mol}]')
        pretrained[mol] = pretrain_forward_model(
            mol=mol,
            pretrain_csv=PRETRAIN_CSVS[mol],
            wav_range=WAV_RANGES[mol],
            n_pca=N_PCA[mol],
            device=device,
            hidden=HIDDEN[mol],
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            seed=SEED,
            model_path=os.path.join(MODEL_DIR, f'net_{mol}_forward_11to19.pt'),
        )

    # -----------------------------------------------------------------------
    # Diagnostics: PCA explained variance
    # -----------------------------------------------------------------------
    print('\n=== PCA explained variance ===')
    for mol, res in pretrained.items():
        if res.get('pca') is not None:
            cv = res['pca'].explained_variance_ratio_.cumsum()
            print(f'  {mol}: {len(cv)} components → cumul. var = {cv[-1]*100:.4f}%')

    # -----------------------------------------------------------------------
    # Diagnostics: loss curves and validation R²
    # -----------------------------------------------------------------------
    print('\n=== Validation R² (physical flux) ===')
    for mol, res in pretrained.items():
        if res['train_losses_shape'] or res['train_losses_peak']:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
            ax1.semilogy(res['train_losses_shape'], label='Train')
            ax1.semilogy(res['val_losses_shape'],   label='Val', ls='--')
            ax1.set_title(f'{mol} — shape  (net_shape)')
            ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE loss'); ax1.legend()
            ax2.semilogy(res['train_losses_peak'],  label='Train')
            ax2.semilogy(res['val_losses_peak'],    label='Val', ls='--')
            ax2.set_title(f'{mol} — peak  (net_peak)')
            ax2.set_xlabel('Epoch'); ax2.set_ylabel('MSE loss'); ax2.legend()
            plt.suptitle(f'Pretrain loss — {mol}')
            plt.tight_layout()
            plt.savefig(os.path.join(FIG_DIR, f'loss_{mol}.png'), dpi=200)
            plt.close()

        net_shape   = res['net_shape']
        net_peak    = res['net_peak']
        xp_sc       = res['xp_sc']
        yp_sc_shape = res['yp_sc_shape']
        yp_sc_peak  = res['yp_sc_peak']
        pca         = res.get('pca')
        X_v         = res['X_pre_v']
        Y_v         = res['Y_pre_v']
        X_v_t       = res['X_pre_v_t']
        p_v         = res['log10p_v']

        net_shape.eval(); net_peak.eval()
        with torch.no_grad():
            z_s_s = net_shape(X_v_t).cpu().numpy()
            z_p_s = net_peak(X_v_t).cpu().numpy()

        z_shape      = yp_sc_shape.inverse_transform(z_s_s)
        log10p_pred  = yp_sc_peak.inverse_transform(z_p_s)[:, 0]
        Y_pred_norm  = pca.inverse_transform(z_shape) if pca is not None else z_shape

        Y_true_phys = Y_v         * (10 ** p_v)[:, None]
        Y_pred_phys = np.maximum(Y_pred_norm, 0.0) * (10 ** log10p_pred)[:, None]

        yt = Y_true_phys.ravel()
        yp = Y_pred_phys.ravel()
        r2 = 1 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
        print(f'  {mol}: R² = {r2:.4f}')

        wav = res['wav']
        fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
        for k, ax in enumerate(axes):
            ax.plot(wav, Y_true_phys[k], 'k-',  lw=0.8, label='True')
            ax.plot(wav, Y_pred_phys[k], '--',  lw=0.8, label='Pred')
            ax.set_title(f'T={X_v[k,0]:.0f} K  logN={X_v[k,1]:.1f}', fontsize=7)
            ax.set_xlabel('Wavelength (µm)')
            if k == 0:
                ax.set_ylabel('Flux (Jy)')
        axes[-1].legend(fontsize=7)
        plt.suptitle(f'{mol} val  R²={r2:.3f}', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'val_{mol}.png'), dpi=200,
                    bbox_inches='tight')
        plt.close()

    print(f'\nFigures saved to {FIG_DIR}')
    print('Done.')
