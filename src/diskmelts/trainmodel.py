"""
diskmelts/trainmodel.py — Training utilities for per-molecule forward surrogate models.

Provides three public functions:
    load_model_grid          : load T{T}N{logN}.csv files from a Model_grids/ directory
    generate_pre_training_set: enumerate every (T, logN) grid point → per-molecule CSV
    pretrain_forward_model   : train (or load) the two-MLP forward model per molecule
                                  net_shape : (T, logN) → n_pca PCA coefficients
                                  net_peak  : (T, logN) → log10(peak flux)

Run as a script to generate pretrain CSVs, train all four molecules, and save
loss-curve diagnostics:
    python src/diskmelts/trainmodel.py

Edit the CONFIGURATION block inside if __name__ == '__main__' to change paths,
which molecules to train, architecture, PCA size, learning rate, etc.
"""

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

def load_model_grid(grid_dir):
    """
    Load all slab-model CSV files from a directory.

    Files must be named T{T}N{logN}.csv with columns 'wave' (µm) and 'Line' (Jy).

    Args:
        grid_dir (str): path to the directory containing the CSV files

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
        df   = pd.read_csv(os.path.join(grid_dir, fname))
        models[(T, logN)] = {
            'wavelength': df['wave'].to_numpy(),
            'flux':       df['Line'].to_numpy(),
        }
    return models


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
                             peak |flux|; 0.0 for noise-free output (default 0.0)
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
    pretrain_csv,
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
):
    """
    Train (or load) the two-MLP forward model for one molecule.

    Two sub-models are trained independently from the same pretrain CSV:
        net_shape : (T, logN) → n_pca PCA coefficients of the peak-normalised
                    spectral shape
        net_peak  : (T, logN) → log10(peak flux)  [scalar]

    If model_path already exists the checkpoint is loaded and training is
    skipped entirely.  After training the checkpoint is saved to model_path.

    Args:
        mol (str): molecule name matching column prefix in pretrain_csv
                   (e.g. 'H2O', 'C2H2')
        pretrain_csv (str): path to the CSV produced by generate_pre_training_set
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

    Returns:
        (dict): with keys
            net_shape          (nn.Module)      : shape MLP in eval mode
            net_peak           (nn.Module)      : peak MLP in eval mode
            xp_sc              (StandardScaler) : fitted on (T, logN) inputs
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

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load pretrain CSV and select wavelength channels ---
    df_pre        = pd.read_csv(pretrain_csv)
    pre_flux_cols = [c for c in df_pre.columns
                     if c.startswith('wav_')
                     and wav_range[0] <= float(c[4:]) <= wav_range[1]]
    wav    = np.array([float(c[4:]) for c in pre_flux_cols])
    n_spec = len(pre_flux_cols)

    X_pre      = df_pre[[f'{mol}_T', f'{mol}_logN']].to_numpy(dtype=np.float32)
    Y_pre      = df_pre[pre_flux_cols].to_numpy(dtype=np.float32)
    log10_peak = df_pre[f'{mol}_log10_peak'].to_numpy(dtype=np.float32)

    # Optional temperature filter (e.g. for hot/warm two-component training)
    if T_range is not None:
        mask       = (X_pre[:, 0] >= T_range[0]) & (X_pre[:, 0] <= T_range[1])
        X_pre      = X_pre[mask]
        Y_pre      = Y_pre[mask]
        log10_peak = log10_peak[mask]
        print(f'  [{mol}] T_range={T_range}: {mask.sum()}/{len(mask)} samples kept')

    X_tr, X_v, Y_tr, Y_v, p_tr, p_v = train_test_split(
        X_pre, Y_pre, log10_peak, test_size=0.1, random_state=seed)

    xp_sc = StandardScaler().fit(X_tr)

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
    yp_sc_peak  = StandardScaler().fit(p_tr.reshape(-1, 1))

    X_v_t  = torch.from_numpy(xp_sc.transform(X_v).astype(np.float32)).to(device)
    Z_v_t  = torch.from_numpy(yp_sc_shape.transform(Z_v).astype(np.float32)).to(device)
    p_v_t  = torch.from_numpy(yp_sc_peak.transform(p_v.reshape(-1, 1)).astype(np.float32)).to(device)

    # --- Load from checkpoint if it exists ---
    if model_path and os.path.exists(model_path):
        ckpt        = torch.load(model_path, map_location=device, weights_only=False)
        pca         = ckpt.get('pca', None)
        yp_sc_shape = ckpt['yp_sc_shape']
        yp_sc_peak  = ckpt['yp_sc_peak']

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
        net_shape = MLP(n_in=2, n_out=n_shape, hidden=hidden).to(device)
        net_peak  = MLP(n_in=2, n_out=1,        hidden=hidden).to(device)

        X_tr_s = torch.from_numpy(xp_sc.transform(X_tr).astype(np.float32))
        Z_tr_s = torch.from_numpy(yp_sc_shape.transform(Z_tr).astype(np.float32))
        p_tr_s = torch.from_numpy(yp_sc_peak.transform(p_tr.reshape(-1, 1)).astype(np.float32))

        loader = DataLoader(
            TensorDataset(X_tr_s, Z_tr_s, p_tr_s),
            batch_size=batch_size, shuffle=True,
        )
        criterion   = nn.MSELoss()
        opt_shape   = optim.Adam(net_shape.parameters(), lr=lr, weight_decay=weight_decay)
        opt_peak    = optim.Adam(net_peak.parameters(),  lr=lr, weight_decay=weight_decay)
        sched_shape = optim.lr_scheduler.ReduceLROnPlateau(opt_shape, patience=10, factor=0.5)
        sched_peak  = optim.lr_scheduler.ReduceLROnPlateau(opt_peak,  patience=10, factor=0.5)

        train_losses_shape, val_losses_shape = [], []
        train_losses_peak,  val_losses_peak  = [], []
        best_val_shape, best_val_peak = float('inf'), float('inf')
        es_ctr_shape,  es_ctr_peak   = 0, 0
        stopped_shape, stopped_peak  = False, False

        print(f'  [{mol}] training for up to {n_epochs} epochs '
              f'(patience={early_stopping_patience}) ...')
        for epoch in range(1, n_epochs + 1):
            net_shape.train()
            net_peak.train()
            bl_shape, bl_peak = [], []
            for xb, zb, pb in loader:
                xb, zb, pb = xb.to(device), zb.to(device), pb.to(device)
                if not stopped_shape:
                    opt_shape.zero_grad()
                    loss_s = criterion(net_shape(xb), zb)
                    loss_s.backward()
                    opt_shape.step()
                    bl_shape.append(loss_s.item())
                if not stopped_peak:
                    opt_peak.zero_grad()
                    loss_p = criterion(net_peak(xb), pb)
                    loss_p.backward()
                    opt_peak.step()
                    bl_peak.append(loss_p.item())

            net_shape.eval()
            net_peak.eval()
            with torch.no_grad():
                vl_shape = criterion(net_shape(X_v_t), Z_v_t).item()
                vl_peak  = criterion(net_peak(X_v_t),  p_v_t).item()
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
                        es_ctr_shape   = 0
                    else:
                        es_ctr_shape  += 1
                    if es_ctr_shape >= early_stopping_patience:
                        print(f'    shape: early stop ep={epoch}  best_val={best_val_shape:.4f}')
                        stopped_shape = True
                if not stopped_peak:
                    if vl_peak < best_val_peak:
                        best_val_peak = vl_peak
                        es_ctr_peak   = 0
                    else:
                        es_ctr_peak  += 1
                    if es_ctr_peak >= early_stopping_patience:
                        print(f'    peak:  early stop ep={epoch}  best_val={best_val_peak:.4f}')
                        stopped_peak = True
                if stopped_shape and stopped_peak:
                    break

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
                'yp_sc_shape':        yp_sc_shape,
                'yp_sc_peak':         yp_sc_peak,
                'hidden':             hidden,
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
    )


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
        'H2O':  22,
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
