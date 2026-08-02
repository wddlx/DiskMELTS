"""Train and validate segmented DiskMELTS model banks.

Set the three ``run_*`` switches below, then run from the repository root:

    python examples/dev_v2_pt_validation.py
"""

import os

import numpy as np
import torch

from diskmelts.fitting import generate_spectrum
from diskmelts.plotting import plot_bank_reconstruction, plot_validation
from diskmelts.trainmodel import (
    generate_pre_training_set,
    load_model_grid,
    train_model_bank,
)
from diskmelts.validation import validate_nt


# ---------------------------------------------------------------------------
# Run controls — edit these three switches
# ---------------------------------------------------------------------------

run_H2O = False
run_Cmol = True
run_rarer = True


# ---------------------------------------------------------------------------
# Molecule segmentation
# ---------------------------------------------------------------------------

H2O_CONFIG = {
    'wavelength_ranges': [
        (4.9, 7.4),
        (7.2, 11.2),
        (9.9, 15.1),
        (13.5, 19.2),
        (19.0, 25.0),
    ],
    'logN_ranges': [(13.0, 19.0), (18.5, 22.0), (21.5, 24.0)],
    'T_ranges': [(50.0, 700.0), (600.0, 1300.0), (1200.0, 2000.0)],
    'n_pca': 21,
}

CARBON_CONFIG = {
    'wavelength_ranges': [(11.5, 17.0)],
    'logN_ranges': [(13.0, 19.5)],
    'T_ranges': [(200.0, 800.0), (600.0, 1500.0), (1200.0, 2000.0)],
    'n_pca': 15,
}

CARBON_MOLECULES = ['HCN', 'C2H2', 'CO2', '13C12CH2', '13CO2']

RARER_MOLECULES = []
# Reserved for later configurations, for example:
# RARER_MOLECULES = ['C4H2', 'HC3N', 'C2H6']

MODEL_CONFIGS = {'H2O': H2O_CONFIG}
MODEL_CONFIGS.update({mol: dict(CARBON_CONFIG) for mol in CARBON_MOLECULES})

MOLECULES = []
if run_H2O:
    MOLECULES.append('H2O')
if run_Cmol:
    MOLECULES.extend(CARBON_MOLECULES)
if run_rarer:
    MOLECULES.extend(RARER_MOLECULES)

if not MOLECULES:
    raise ValueError('No molecules selected: enable at least one run_* switch')

unknown = [mol for mol in MOLECULES if mol not in MODEL_CONFIGS]
if unknown:
    raise ValueError(
        f'No segmented training configuration for {unknown}. '
        f'Available molecules: {list(MODEL_CONFIGS)}'
    )


# ---------------------------------------------------------------------------
# Shared paths and original training hyperparameters
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIDDEN = (64, 128, 64)
N_EPOCHS = int(os.environ.get('DISKMELTS_N_EPOCHS', '5000'))
BATCH = 128
LR = 1.0e-4
PATIENCE = 500
NOISE_SCALE = 0.0
N_NOISE = 1
SEED = 42

# Tile retrieval validation uses each tile's withheld 10% split.
RUN_VALIDATION = os.environ.get('DISKMELTS_VALIDATE', '1') != '0'
VAL_N = int(os.environ.get('DISKMELTS_VAL_N', '10'))
VAL_NT_STARTS = int(os.environ.get('DISKMELTS_VAL_NT_STARTS', '100'))
VAL_NT_STEPS = int(os.environ.get('DISKMELTS_VAL_NT_STEPS', '300'))
VAL_NT_LR = 0.03

# Assembled-bank validation samples 100 full grid spectra by default and saves
# their wavelength-dependent normalized residuals.
BANK_RECON_N = int(os.environ.get('DISKMELTS_BANK_RECON_N', '100'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}  |  Molecules: {MOLECULES}')


# ---------------------------------------------------------------------------
# Each molecule is generated, trained, and validated sequentially.
# ---------------------------------------------------------------------------

for molecule_number, MOL in enumerate(MOLECULES, start=1):
    config = MODEL_CONFIGS[MOL]
    WAVELENGTH_RANGES = config['wavelength_ranges']
    LOGN_RANGES = config['logN_ranges']
    T_RANGES = config['T_ranges']
    N_PCA = config['n_pca']
    expected_tiles = len(WAVELENGTH_RANGES) * len(LOGN_RANGES) * len(T_RANGES)

    GRID_DIR = os.path.join(BASE_DIR, 'Model_grids', MOL)
    PRETRAIN_CSV = os.path.join(
        BASE_DIR, 'Pretrain_grid', f'pretrain_{MOL}_full.csv'
    )
    MODEL_DIR = os.path.join(BASE_DIR, 'Trained_model', f'{MOL}_bank_v2')
    MANIFEST_PATH = os.path.join(MODEL_DIR, f'{MOL}_model_bank.json')
    FIG_DIR = os.path.join(BASE_DIR, 'figures', 'validation', f'{MOL}_bank_v2')

    print('\n' + '=' * 76)
    print(
        f'[{molecule_number}/{len(MOLECULES)}] {MOL}: '
        f'{expected_tiles} model-bank tiles'
    )
    print('=' * 76)

    if not os.path.isdir(GRID_DIR):
        raise FileNotFoundError(
            f'Model grid not found: {GRID_DIR}\n'
            'Model_grids/ is intentionally ignored by Git. Copy the full local '
            'model grids into the repository before running training validation.'
        )

    os.makedirs(os.path.dirname(PRETRAIN_CSV), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print('\n[1] Loading full model grid ...')
    models = load_model_grid(GRID_DIR)
    if not models:
        raise ValueError(f'No T{{T}}N{{logN}}.csv grid files found in {GRID_DIR}')
    wav_full = next(iter(models.values()))['wavelength']
    print(
        f'    {len(models)} grid points, '
        f'T=[{min(k[0] for k in models)}, {max(k[0] for k in models)}] K, '
        f'logN=[{min(k[1] for k in models):g}, {max(k[1] for k in models):g}], '
        f'wavelength=[{wav_full.min():.2f}, {wav_full.max():.2f}] um'
    )

    print('\n[2] Full pretraining CSV ...')
    if not os.path.exists(PRETRAIN_CSV):
        generate_pre_training_set(
            MOL,
            models,
            output_path=PRETRAIN_CSV,
            wav_out=wav_full,
            noise_scale=NOISE_SCALE,
            n_noise=N_NOISE,
            seed=SEED,
            holdout_keys=None,
        )
    else:
        print(f'    Exists, reusing → {PRETRAIN_CSV}')

    print(f'\n[3] Training/loading {expected_tiles} model-bank tiles ...')
    bank_training = train_model_bank(
        mol=MOL,
        pretrain_csv=PRETRAIN_CSV,
        model_dir=MODEL_DIR,
        wavelength_ranges=WAVELENGTH_RANGES,
        logN_ranges=LOGN_RANGES,
        T_ranges=T_RANGES,
        device=device,
        hidden=HIDDEN,
        n_epochs=N_EPOCHS,
        batch_size=BATCH,
        lr=LR,
        n_pca=N_PCA,
        early_stopping_patience=PATIENCE,
        seed=SEED,
        manifest_path=MANIFEST_PATH,
    )

    if RUN_VALIDATION:
        print(f'\n[4] Validating {expected_tiles} withheld tile splits ...')
        summaries = []
        for tile_index, (tile_spec, trained) in enumerate(
            zip(bank_training['tiles'], bank_training['models']), start=1
        ):
            wav_range = tuple(tile_spec['wav_range'])
            logN_range = tuple(tile_spec['logN_range'])
            T_range = tuple(tile_spec['T_range'])
            print(
                f'\n[{tile_index:02d}/{expected_tiles:02d}] wav={wav_range}, '
                f'logN={logN_range}, T={T_range}'
            )
            result = validate_nt(
                MOL,
                pretrained={MOL: trained},
                pretrain_csv=PRETRAIN_CSV,
                fit_ranges=wav_range,
                n_samples=VAL_N,
                T_bounds=T_range,
                logN_bounds=logN_range,
                n_starts=VAL_NT_STARTS,
                n_steps=VAL_NT_STEPS,
                lr=VAL_NT_LR,
                seed=SEED,
                validation_points=trained['X_pre_v'],
            )
            tag = os.path.splitext(os.path.basename(tile_spec['path']))[0]
            plot_validation(
                MOL,
                result,
                save_path=os.path.join(FIG_DIR, f'{tag}.png'),
            )
            summaries.append((
                tag,
                float(np.sqrt(np.mean((result['T_pred'] - result['T_true']) ** 2))),
                float(np.sqrt(
                    np.mean((result['logN_pred'] - result['logN_true']) ** 2)
                )),
            ))

        print('\nTile validation RMSE:')
        for tag, T_rmse, logN_rmse in summaries:
            print(f'  {tag}: T={T_rmse:.2f} K, logN={logN_rmse:.4f}')

        print('\n[5] Validating the assembled model-bank spectrum ...')
        covered_keys = [
            key for key in models
            if min(r[0] for r in T_RANGES) <= key[0] <= max(r[1] for r in T_RANGES)
            and min(r[0] for r in LOGN_RANGES) <= key[1] <= max(r[1] for r in LOGN_RANGES)
        ]
        sample_count = min(BANK_RECON_N, len(covered_keys))
        if sample_count < 1:
            raise ValueError('assembled-bank validation selected zero spectra')
        rng = np.random.default_rng(SEED)
        sample_indices = rng.choice(
            len(covered_keys), size=sample_count, replace=False
        )
        bank = bank_training['model_bank']
        comparison_wav = None
        true_fluxes = []
        predicted_fluxes = []
        relative_rmse = []

        for index in sample_indices:
            key = covered_keys[index]
            true_wav = models[key]['wavelength']
            mask = (true_wav >= min(r[0] for r in WAVELENGTH_RANGES)) & (
                true_wav <= max(r[1] for r in WAVELENGTH_RANGES)
            )
            selected_wav = true_wav[mask]
            true_flux = models[key]['flux'][mask]
            predicted_flux = generate_spectrum(
                T=key[0],
                logN=key[1],
                A=1.0,
                pretrained_mol=bank,
                obs_wav=selected_wav,
            )
            if comparison_wav is None:
                comparison_wav = selected_wav
            elif not np.array_equal(comparison_wav, selected_wav):
                raise ValueError('model-grid wavelength axes differ between spectra')

            peak = max(float(np.max(np.abs(true_flux))), 1.0e-30)
            relative_rmse.append(
                float(np.sqrt(np.mean((predicted_flux - true_flux) ** 2)) / peak)
            )
            true_fluxes.append(true_flux)
            predicted_fluxes.append(predicted_flux)

        plot_bank_reconstruction(
            comparison_wav,
            true_fluxes,
            predicted_fluxes,
            mol=MOL,
            save_path=os.path.join(FIG_DIR, f'{MOL}_bank_reconstruction.png'),
        )
        print(
            f'  {sample_count} spectra: median RMSE/peak='
            f'{np.median(relative_rmse):.4e}, max={np.max(relative_rmse):.4e}'
        )
    else:
        print('\n[4] Validation skipped (DISKMELTS_VALIDATE=0).')

    print(f'\nManifest: {MANIFEST_PATH}')
    print(f'Validation figures: {FIG_DIR}')

print('\nAll requested molecules completed.')
