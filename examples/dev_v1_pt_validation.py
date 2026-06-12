"""
examples/dev_v1_pt_validation.py
End-to-end pipeline for one molecule:
  generate pretrain CSV → train forward model → NT validation → full validation

Edit the CONFIGURATION block below, then run from the repository root:
    python examples/dev_v1_pt_validation.py

NT validation   : recovers T and logN from grid spectra at A=1 (no A fitting).
Full validation : recovers T, logN, and A from synthetic spectra with random A.
"""

import os
import numpy as np
import torch

from diskmelts.trainmodel import load_model_grid, generate_pre_training_set, pretrain_forward_model
from diskmelts.fitting import load_models
from diskmelts.validation import validate_nt, validate_nt_holdout  # validate_full unused (step 6 off)
from diskmelts.plotting import plot_validation_split  # plot_validation unused (step 6 off)

# ===========================================================================
# CONFIGURATION — change MOL and GRID_DIR; everything else auto-adjusts
# ===========================================================================

MOL = os.environ.get('DISKMELTS_MOL', 'H2O')

# Repository root = parent of examples/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input: local model grid directory (ignored by Git)
GRID_DIR = os.path.join(BASE_DIR, 'Model_grids', MOL)

# Output paths — auto-named from MOL; change the base dirs if needed
PRETRAIN_CSV = os.path.join(BASE_DIR, 'Pretrain_grid', f'pretrain_{MOL}_11to19.csv')
MODEL_PATH   = os.path.join(BASE_DIR, 'Trained_model', f'net_{MOL}_forward_11to19.pt')
FIG_DIR      = os.path.join(BASE_DIR, 'figures', 'validation')

# Per-molecule defaults — WAV_RANGE, FIT_RANGES, N_PCA differ by molecule
#
#   MOL     WAV_RANGE          FIT_RANGES                          N_PCA
#   H2O     (11.0, 19.0)       [(11.0,12.0), (17.0,18.5)]         21
#   C2H2    (11.0, 17.5)       (12.0, 16.5)                       15
#   HCN     (11.0, 17.5)       (12.0, 17.0)                       15
#   CO2     (11.0, 17.5)       (12.0, 17.0)                       15
#
_MOL_DEFAULTS = {
    'H2O':      dict(wav_range=(11.0, 19.0), fit_ranges=[(11.0, 12.0), (16.5, 18.5)], n_pca=21),
    'C2H2':     dict(wav_range=(11.0, 17.5), fit_ranges=(12.0, 16.5),                 n_pca=15),
    '13C12CH2': dict(wav_range=(11.0, 17.5), fit_ranges=(12.0, 16.5),                 n_pca=15),
    'HCN':      dict(wav_range=(11.0, 17.5), fit_ranges=(12.0, 17.0),                 n_pca=15),
    'CO2':      dict(wav_range=(11.0, 17.5), fit_ranges=(12.0, 17.0),                 n_pca=15),
    '13CO2':    dict(wav_range=(11.0, 17.5), fit_ranges=(12.0, 17.0),                 n_pca=15),
}
WAV_RANGE  = _MOL_DEFAULTS[MOL]['wav_range']
FIT_RANGES = _MOL_DEFAULTS[MOL]['fit_ranges']
N_PCA      = _MOL_DEFAULTS[MOL]['n_pca']

# Model architecture (same for all molecules)
HIDDEN = (64, 128, 64)   # hidden layer sizes for net_shape and net_peak

# Training hyperparameters
N_EPOCHS = int(os.environ.get('DISKMELTS_N_EPOCHS', '5000'))
BATCH    = 128
LR       = 1e-4
PATIENCE = 500   # early-stopping patience (epochs without improvement)

# Pretrain CSV options
NOISE_SCALE = 0.0   # noise as fraction of peak flux; 0.0 = noise-free
N_NOISE     = 1     # noise realisations per grid point

# Fitted parameter bounds — change these to restrict T, logN, or A search range
T_BOUNDS    = (100, 1400)    # (T_min, T_max) in K
LOGN_BOUNDS = (13.0, 19.0)   # (logN_min, logN_max) in log10(cm^-2)
LOGA_BOUNDS = (-2.0, 2.0)    # (log10A_min, log10A_max) for full validation fitting

# Holdout: grid points excluded from pretrain CSV and used as unseen validation
HOLDOUT_N = int(os.environ.get('DISKMELTS_HOLDOUT_N', '30'))

# NT validation (T and logN only, A = 1) — gradient-based Adam optimizer
VAL_NT_FRAC   = float(os.environ.get('DISKMELTS_VAL_NT_FRAC', '0.02'))
VAL_NT_STARTS = int(os.environ.get('DISKMELTS_VAL_NT_STARTS', '100'))
VAL_NT_STEPS  = int(os.environ.get('DISKMELTS_VAL_NT_STEPS', '300'))
VAL_NT_LR     = 0.03   # Adam learning rate

# Full validation (T, logN, A with random A)
VAL_FULL_N        = 30     # number of synthetic spectra
VAL_FULL_NOISE    = 0.001  # noise fraction for synthetic spectra
VAL_FULL_RESTARTS = 100    # fit_nested random restarts per spectrum

# Full validation fitter: 'fit_nested' (random restarts) or 'fit_molecules' (Sobol global search)
FITTER = 'fit_molecules'

# fit_molecules settings (only used when FITTER='fit_molecules')
VAL_NEW_N_SAMPLES = 20000  # Sobol global screening samples
VAL_NEW_N_REFINE  = 32     # L-BFGS-B refinement candidates
VAL_NEW_N_TOP     = 10     # top solutions to retain

SEED = 42

# ===========================================================================

if not os.path.isdir(GRID_DIR):
    raise FileNotFoundError(
        f'Model grid not found: {GRID_DIR}\n'
        'Model_grids/ is intentionally ignored by Git. Copy the full local '
        'model grids into the repository before running training validation.'
    )

os.makedirs(os.path.dirname(PRETRAIN_CSV), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  Molecule: {MOL}')

# ---------------------------------------------------------------------------
# Step 1 — Load model grid
# ---------------------------------------------------------------------------
print('\n[1] Loading model grid ...')
models   = load_model_grid(GRID_DIR)
wav_full = next(iter(models.values()))['wavelength']
print(f'    {len(models)} grid points, wav [{wav_full.min():.1f}, {wav_full.max():.1f}] µm')

# ---------------------------------------------------------------------------
# Step 1b — Select holdout grid points (excluded from pretrain CSV)
# ---------------------------------------------------------------------------
print(f'\n[1b] Selecting {HOLDOUT_N} holdout grid points ...')
_all_keys    = list(models.keys())
_rng_holdout = np.random.default_rng(SEED)
_holdout_idx = _rng_holdout.choice(len(_all_keys), size=min(HOLDOUT_N, len(_all_keys)),
                                   replace=False)
holdout_keys     = [_all_keys[i] for i in sorted(_holdout_idx)]
holdout_keys_set = set(map(tuple, holdout_keys))
print(f'    {len(holdout_keys)} holdout keys (T range {min(k[0] for k in holdout_keys)}'
      f'–{max(k[0] for k in holdout_keys)} K, '
      f'logN range {min(k[1] for k in holdout_keys):.2f}'
      f'–{max(k[1] for k in holdout_keys):.2f})')

# ---------------------------------------------------------------------------
# Step 2 — Generate pretrain CSV (skipped if the file already exists)
# ---------------------------------------------------------------------------
print('\n[2] Pretrain CSV ...')
if not os.path.exists(PRETRAIN_CSV):
    generate_pre_training_set(
        MOL, models,
        output_path=PRETRAIN_CSV,
        wav_out=wav_full,
        noise_scale=NOISE_SCALE,
        n_noise=N_NOISE,
        seed=SEED,
        holdout_keys=holdout_keys_set,
    )
else:
    print(f'    Exists, skipping → {PRETRAIN_CSV}')

# ---------------------------------------------------------------------------
# Step 3 — Train (or load) the forward model
# ---------------------------------------------------------------------------
print('\n[3] Training forward model ...')
pretrain_forward_model(
    mol=MOL,
    pretrain_csv=PRETRAIN_CSV,
    wav_range=WAV_RANGE,
    device=device,
    hidden=HIDDEN,
    n_pca=N_PCA,
    n_epochs=N_EPOCHS,
    batch_size=BATCH,
    lr=LR,
    early_stopping_patience=PATIENCE,
    model_path=MODEL_PATH,
    seed=SEED,
)

# ---------------------------------------------------------------------------
# Step 4 — Load model via the standard fitting API
# ---------------------------------------------------------------------------
print('\n[4] Loading model for fitting ...')
pretrained = load_models(
    model_paths={MOL: MODEL_PATH},
    device=device,
)

# ---------------------------------------------------------------------------
# Step 5 — NT validation: in-pretrain set + holdout set, plotted separately
# ---------------------------------------------------------------------------
print('\n[5a] NT validation — in-pretrain grid points ...')
nt_pretrain = validate_nt(
    MOL, pretrained, PRETRAIN_CSV, FIT_RANGES,
    frac=VAL_NT_FRAC,
    T_bounds=T_BOUNDS,
    logN_bounds=LOGN_BOUNDS,
    n_starts=VAL_NT_STARTS,
    n_steps=VAL_NT_STEPS,
    lr=VAL_NT_LR,
    seed=SEED,
)

print('\n[5b] NT validation — holdout (unseen) grid points ...')
nt_holdout = validate_nt_holdout(
    MOL, pretrained, models, holdout_keys, FIT_RANGES,
    T_bounds=T_BOUNDS,
    logN_bounds=LOGN_BOUNDS,
    n_starts=VAL_NT_STARTS,
    n_steps=VAL_NT_STEPS,
    lr=VAL_NT_LR,
)

plot_validation_split(MOL, nt_pretrain, nt_holdout,
                      save_path=os.path.join(FIG_DIR, f'val_nt_split_{MOL}.png'))

# ---------------------------------------------------------------------------
# Step 6 — Full validation: recover T, logN, A from synthetic spectra
#           (commented out; run NT split validation above instead)
# ---------------------------------------------------------------------------
# print(f'\n[6] Full validation (T, logN, A)  fitter={FITTER!r} ...')
#
# if FITTER == 'fit_molecules':
#     from diskmelts.fitting import fit_molecules
#
#     rng = np.random.default_rng(SEED)
#     keys_list = list(models.keys())
#     wav_grid  = next(iter(models.values()))['wavelength']
#
#     T_true_list, logN_true_list, A_true_list = [], [], []
#     T_pred_list, logN_pred_list, A_pred_list = [], [], []
#
#     print(f'  Sobol samples={VAL_NEW_N_SAMPLES}  refine={VAL_NEW_N_REFINE}  top={VAL_NEW_N_TOP}')
#     for i in range(VAL_FULL_N):
#         idx         = rng.integers(len(keys_list))
#         T_t, logN_t = keys_list[idx]
#         A_t         = 10.0 ** rng.uniform(*LOGA_BOUNDS)
#
#         flux = A_t * models[keys_list[idx]]['flux'].copy()
#         if VAL_FULL_NOISE > 0:
#             peak = np.abs(flux).max()
#             if peak > 0:
#                 flux += rng.normal(0, VAL_FULL_NOISE * peak, size=len(flux))
#
#         result = fit_molecules(
#             obs_wav=wav_grid, obs_flux=flux,
#             mol=MOL, pretrained=pretrained,
#             fit_ranges=FIT_RANGES,
#             T_bounds=T_BOUNDS, logN_bounds=LOGN_BOUNDS,
#             loga_bounds=LOGA_BOUNDS,
#             n_samples=VAL_NEW_N_SAMPLES,
#             n_refine=VAL_NEW_N_REFINE,
#             n_top=VAL_NEW_N_TOP,
#             seed=i, verbose=False,
#         )
#
#         T_true_list.append(float(T_t))
#         logN_true_list.append(float(logN_t))
#         A_true_list.append(float(A_t))
#         T_pred_list.append(float(result['params'][MOL]['T']))
#         logN_pred_list.append(float(result['params'][MOL]['logN']))
#         A_pred_list.append(float(result['params'][MOL]['A']))
#
#         if (i + 1) % 10 == 0 or (i + 1) == VAL_FULL_N:
#             print(f'  {i + 1}/{VAL_FULL_N}')
#
#     logN_true   = np.array(logN_true_list, dtype=np.float64)
#     logN_pred   = np.array(logN_pred_list, dtype=np.float64)
#     A_true      = np.array(A_true_list,    dtype=np.float64)
#     A_pred      = np.array(A_pred_list,    dtype=np.float64)
#     log10A_true = np.log10(np.maximum(A_true, 1e-10))
#     log10A_pred = np.log10(np.maximum(A_pred, 1e-10))
#
#     full_results = {
#         'T_true':        np.array(T_true_list, dtype=np.float64),
#         'T_pred':        np.array(T_pred_list, dtype=np.float64),
#         'logN_true':     logN_true,
#         'logN_pred':     logN_pred,
#         'log10A_true':   log10A_true,
#         'log10A_pred':   log10A_pred,
#         'log10NA_true':  logN_true  + log10A_true,
#         'log10NA_pred':  logN_pred  + log10A_pred,
#     }
#     from diskmelts.plotting import plot_validation
#     plot_validation(MOL, full_results,
#                     save_path=os.path.join(FIG_DIR, f'val_full_new_{MOL}.png'))
#
# else:  # 'fit_nested'
#     from diskmelts.validation import validate_full
#     full_results = validate_full(
#         MOL, models, pretrained, FIT_RANGES,
#         n_samples=VAL_FULL_N,
#         noise_scale=VAL_FULL_NOISE,
#         T_bounds=T_BOUNDS,
#         logN_bounds=LOGN_BOUNDS,
#         loga_bounds=LOGA_BOUNDS,
#         n_restarts=VAL_FULL_RESTARTS,
#         seed=SEED,
#     )
#     from diskmelts.plotting import plot_validation
#     plot_validation(MOL, full_results,
#                     save_path=os.path.join(FIG_DIR, f'val_full_{MOL}.png'))

print(f'\nFigures saved to {FIG_DIR}')
print('Done.')
