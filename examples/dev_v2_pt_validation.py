"""Train and validate the maintained hybrid v2 surrogate design.

The expanded v2 IRIS grids are restricted to the science domain configured
below before any pretraining data are written. Complete randomly selected grid
models are withheld from PCA, scaler, and neural-network training and used for
unseen-model validation. H2O uses five overlapping wavelength checkpoints;
carbon-bearing molecules use one checkpoint each.

Run from the repository root:

    python examples/dev_v2_pt_validation.py

Set DISKMELTS_V2_MODE=water_compare to benchmark the original and corrected
H2O banks on matched holdouts. Set DISKMELTS_V2_MODE=water_seam_trial to
reproduce the rejected edge-weighted seam diagnostic. Both are opt-in.
"""

import copy
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diskmelts.fitting import generate_spectrum, load_models
from diskmelts.plotting import plot_bank_reconstruction, plot_validation
from diskmelts.trainmodel import (
    generate_pre_training_set,
    load_model_grid,
    pretrain_forward_model,
    select_grid_holdout,
    train_model_bank,
    tune_pca_components,
)
from diskmelts.validation import validate_full, validate_nt_holdout


V2_MODE = os.environ.get('DISKMELTS_V2_MODE', 'train')
if V2_MODE not in {'train', 'water_compare', 'water_seam_trial'}:
    raise ValueError(f'Unknown DISKMELTS_V2_MODE: {V2_MODE}')

# ---------------------------------------------------------------------------
# H2O diagnostic mode: water_compare
# ---------------------------------------------------------------------------
if V2_MODE == 'water_compare':
    ROOT = Path(__file__).resolve().parents[1]
    BANKS = os.environ.get(
        'DISKMELTS_H2O_COMPARISON_BANKS',
        'H2O_wavelength_bank_v2,H2O_wavelength_bank_v2_fixed',
    ).split(',')
    OUT = ROOT / 'figures' / 'validation' / 'water_bank_comparison'
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((ROOT / 'Pretrain_grid' /
        'holdout_H2O_wavelength_bank_T100to1400_N13to19.json').read_text())
    keys = [tuple(key) for key in metadata['keys']]
    np.random.default_rng(42).shuffle(keys)
    paths = {}
    for path in (ROOT / 'Model_grids' / 'H2O').glob('*.csv'):
        match = re.fullmatch(r'T(\d+)N([\d.]+)\.csv', path.name)
        if match:
            paths[(float(match[1]), float(match[2]))] = path
    models = {}
    for key in keys[:100]:
        frame = pd.read_csv(paths[key])
        frame = frame[(frame.wave >= 4.9) & (frame.wave <= 25.0)]
        models[key] = {'wavelength': frame.wave.to_numpy(), 'flux': frame.Line.to_numpy()}
    wav = next(iter(models.values()))['wavelength']

    for name in BANKS:
        bank = load_models({'H2O': str(ROOT / 'Trained_model' / name / 'H2O_model_bank.json')},
                           device='cpu')['H2O']
        tile_metrics = []
        for index, tile in enumerate(bank['tiles']):
            result = validate_nt_holdout(
                'H2O', {'H2O': tile['model']}, models, keys[:30], tile['wav_range'],
                T_bounds=(100,1400), logN_bounds=(13,19), n_starts=100, n_steps=300, seed=42,
            )
            np.savez(OUT / f'{name}_tile{index}_nt.npz', **result)
            tile_metrics.append({
                'wav_range': tile['wav_range'],
                'n_evaluated': len(result['T_true']),
                'T_rmse_K': float(np.sqrt(np.mean((result['T_pred']-result['T_true'])**2))),
                'logN_rmse': float(np.sqrt(np.mean((result['logN_pred']-result['logN_true'])**2))),
            })
            plot_validation('H2O', result, save_path=str(OUT / f'{name}_tile{index}_nt.png'))
            print(name, tile_metrics[-1], flush=True)
        full = validate_full(
            'H2O', {k: models[k] for k in keys[:10]}, {'H2O': bank}, (4.9,25.0),
            n_samples=10, noise_scale=0.0, T_bounds=(100,1400), logN_bounds=(13,19),
            n_restarts=50, seed=42,
        )
        np.savez(OUT / f'{name}_full.npz', **full)
        plot_validation('H2O', full, save_path=str(OUT / f'{name}_full.png'))
        residuals = []
        truths, predictions = [], []
        for key in keys[:100]:
            truth = models[key]['flux']
            prediction = generate_spectrum(*key, 1.0, bank, obs_wav=wav)
            truths.append(truth)
            predictions.append(prediction)
            residuals.append((prediction-truth)/max(abs(truth)))
        plot_bank_reconstruction(wav, np.asarray(truths), np.asarray(predictions),
            mol='H2O', save_path=str(OUT / f'{name}_reconstruction.png'))
        errors = np.sqrt(np.mean(np.asarray(residuals)**2, axis=1))
        summary = {
            'bank': name, 'tile_nt_metrics': tile_metrics,
            'full_T_rmse_K': float(np.sqrt(np.mean((full['T_pred']-full['T_true'])**2))),
            'full_log10NA_rmse': float(np.sqrt(np.mean((full['log10NA_pred']-full['log10NA_true'])**2))),
            'reconstruction_relative_rmse_median': float(np.median(errors)),
            'reconstruction_relative_rmse_max': float(np.max(errors)),
            'seed': 42, 'n_nt': 30, 'n_full': 10, 'n_reconstruction': 100,
            'n_nt_starts': 100, 'n_nt_steps': 300, 'n_full_restarts': 50,
        }
        (OUT / f'{name}_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary, indent=2), flush=True)
    sys.exit(0)


# ---------------------------------------------------------------------------
# H2O diagnostic mode: water_seam_trial
# ---------------------------------------------------------------------------
if V2_MODE == 'water_seam_trial':
    ROOT = Path(__file__).resolve().parents[1]
    BASE_BANK = ROOT / 'Trained_model' / 'H2O_wavelength_bank_v2_fixed' / 'H2O_model_bank.json'
    OUT_DIR = ROOT / 'Trained_model' / 'H2O_wavelength_bank_v2_seam_trial'
    FIG_DIR = ROOT / 'figures' / 'validation' / 'H2O_wavelength_bank_v2_seam_trial'
    PRETRAIN = ROOT / 'Pretrain_grid' / 'pretrain_H2O_wavelength_bank_T100to1400_N13to19.csv'
    HOLDOUT = ROOT / 'Pretrain_grid' / 'holdout_H2O_wavelength_bank_T100to1400_N13to19.json'
    EDGE_RANGE = (19.0, 19.2)
    SEED = 42

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with BASE_BANK.open(encoding='utf-8') as stream:
        manifest = json.load(stream)
    edge_spec = manifest['tiles'][-1]
    edge_model_path = OUT_DIR / 'net_H2O_w19to25_edge.pt'

    pretrain_forward_model(
        mol='H2O', pretrain_csv=PRETRAIN, wav_range=tuple(edge_spec['wav_range']),
        T_range=tuple(edge_spec['T_range']), logN_range=tuple(edge_spec['logN_range']),
        device='cpu', hidden=(64, 128, 64), n_epochs=5000, batch_size=128,
        lr=1e-4, seed=SEED, n_pca=30, early_stopping_patience=500,
        peak_normalization='window', shape_loss='spectral_edge',
        scheduler_patience=100, min_lr=1e-6, temperature_transform='log10',
        shape_edge_ranges=[EDGE_RANGE], shape_edge_weight=20.0,
        model_path=edge_model_path,
    )
    manifest = copy.deepcopy(manifest)
    manifest['shape_loss'] = ['spectral', 'spectral', 'spectral', 'spectral', 'spectral_edge']
    manifest['shape_edge_weight'] = 20.0
    for index, spec in enumerate(manifest['tiles'][:-1]):
        old_model = (BASE_BANK.parent / spec['path']).resolve()
        local_model = OUT_DIR / old_model.name
        shutil.copy2(old_model, local_model)
        spec['path'] = local_model.name
    manifest['tiles'][-1]['path'] = os.path.relpath(edge_model_path, OUT_DIR)
    manifest_path = OUT_DIR / 'H2O_model_bank.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    bank = load_models({'H2O': str(manifest_path)}, device='cpu')['H2O']
    holdout_keys = [tuple(row) for row in json.loads(HOLDOUT.read_text(encoding='utf-8'))['keys']]
    np.random.default_rng(SEED).shuffle(holdout_keys)
    paths = {}
    for path in (ROOT / 'Model_grids' / 'H2O').glob('*.csv'):
        match = re.fullmatch(r'T(\d+)N([\d.]+)\.csv', path.name)
        if match:
            paths[(float(match[1]), float(match[2]))] = path
    models = {}
    for key in holdout_keys[:100]:
        frame = pd.read_csv(paths[key])
        models[key] = {'wavelength': frame.wave.to_numpy(), 'flux': frame.Line.to_numpy()}

    nt = validate_nt_holdout(
        'H2O', {'H2O': bank['tiles'][-1]['model']}, models, holdout_keys[:30],
        tuple(edge_spec['wav_range']), T_bounds=(100, 1400), logN_bounds=(13, 19),
        n_starts=100, n_steps=300, seed=SEED,
    )
    plot_validation('H2O', nt, save_path=str(FIG_DIR / 'tile4_nt.png'))
    full = validate_full(
        'H2O', {key: models[key] for key in holdout_keys[:10]}, {'H2O': bank},
        (4.9, 25.0), n_samples=10, noise_scale=0.0, T_bounds=(100, 1400),
        logN_bounds=(13, 19), n_restarts=50, seed=SEED,
    )
    plot_validation('H2O', full, save_path=str(FIG_DIR / 'full.png'))
    np.savez(FIG_DIR / 'tile4_nt.npz', **nt)
    np.savez(FIG_DIR / 'full.npz', **full)

    seam_rows = []
    reconstruction_rows = []
    reconstruction_truths, reconstruction_predictions = [], []
    for key in holdout_keys[:100]:
        frame = pd.read_csv(paths[key])
        wav = frame.wave.to_numpy()
        flux = frame.Line.to_numpy()
        keep = (wav >= EDGE_RANGE[0]) & (wav <= EDGE_RANGE[1])
        truth = flux[keep]
        first = generate_spectrum(*key, 1.0, bank['tiles'][-2]['model'], obs_wav=wav[keep])
        last = generate_spectrum(*key, 1.0, bank['tiles'][-1]['model'], obs_wav=wav[keep])
        peak = max(float(np.max(np.abs(flux))), 1e-30)
        seam_rows.append({
            'T': key[0], 'logN': key[1],
            'truth_rms_over_full_peak': float(np.sqrt(np.mean(truth**2)) / peak),
            'previous_tile_error_over_full_peak': float(np.sqrt(np.mean((first-truth)**2)) / peak),
            'refined_tile_error_over_full_peak': float(np.sqrt(np.mean((last-truth)**2)) / peak),
            'tile_disagreement_over_full_peak': float(np.sqrt(np.mean((first-last)**2)) / peak),
        })
        prediction = generate_spectrum(*key, 1.0, bank, obs_wav=wav)
        reconstruction_truths.append(flux)
        reconstruction_predictions.append(prediction)
        reconstruction_rows.append(float(np.sqrt(np.mean(((prediction-flux)/peak)**2)))
                                   if peak > 0 else float('nan'))
    plot_bank_reconstruction(
        wav, np.asarray(reconstruction_truths), np.asarray(reconstruction_predictions),
        mol='H2O', save_path=str(FIG_DIR / 'reconstruction.png'),
    )
    summary = {
        'manifest': str(manifest_path), 'shape_edge_range_um': list(EDGE_RANGE),
        'shape_edge_weight': 20.0, 'seed': SEED, 'n_edge_nt': len(nt['T_true']),
        'edge_nt_T_rmse_K': float(np.sqrt(np.mean((nt['T_pred']-nt['T_true'])**2))),
        'edge_nt_logN_rmse': float(np.sqrt(np.mean((nt['logN_pred']-nt['logN_true'])**2))),
        'full_T_rmse_K': float(np.sqrt(np.mean((full['T_pred']-full['T_true'])**2))),
        'full_log10NA_rmse': float(np.sqrt(np.mean((full['log10NA_pred']-full['log10NA_true'])**2))),
        'reconstruction_relative_rmse_median': float(np.nanmedian(reconstruction_rows)),
        'reconstruction_relative_rmse_max': float(np.nanmax(reconstruction_rows)),
        'seam_material_cases': [row for row in seam_rows if row['truth_rms_over_full_peak'] > 1e-3],
        'seam_median_previous_error_pct_peak': 100*np.median([row['previous_tile_error_over_full_peak'] for row in seam_rows if row['truth_rms_over_full_peak'] > 1e-3]),
        'seam_median_refined_error_pct_peak': 100*np.median([row['refined_tile_error_over_full_peak'] for row in seam_rows if row['truth_rms_over_full_peak'] > 1e-3]),
        'seam_median_disagreement_pct_peak': 100*np.median([row['tile_disagreement_over_full_peak'] for row in seam_rows if row['truth_rms_over_full_peak'] > 1e-3]),
    }
    (FIG_DIR / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    sys.exit(0)

# ---------------------------------------------------------------------------
# Run controls
# ---------------------------------------------------------------------------

run_H2O = False
run_Cmol = True
run_rarer = True


# ---------------------------------------------------------------------------
# Hybrid design: wavelength-bank H2O, single-model carbon species
# ---------------------------------------------------------------------------

H2O_CONFIG = {
    'wavelength_ranges': (
        (4.9, 7.4),
        (7.2, 11.2),
        (9.9, 15.1),
        (13.5, 19.2),
        (19.0, 25.0),
    ),
    'logN_range': (13.0, 19.0),
    'T_range': (100.0, 1400.0),
    'n_pca': 21,
    'pca_candidates': (10, 15, 21, 30),
}

CARBON_CONFIG = {
    'wav_range': (11.5, 17.0),
    'logN_range': (13.0, 19.5),
    'T_range': (100.0, 1400.0),
    'n_pca': 15,
    'pca_candidates': (10, 15, 20, 25, 30),
}

CARBON_MOLECULES = ['HCN', 'C2H2', 'CO2', '13C12CH2', '13CO2']
RARER_CONFIGS = {
    'C4H2': {'wav_range': (14.0, 18.0)},
    'HC3N': {'wav_range': (13.0, 18.0)},
    'C2H6': {'wav_range': (9.0, 15.0)},
    'C2H4': {'wav_range': (8.0, 14.0)},
    'CH4': {'wav_range': (5.0, 9.0)},
}

MODEL_CONFIGS = {'H2O': H2O_CONFIG}
MODEL_CONFIGS.update({mol: dict(CARBON_CONFIG) for mol in CARBON_MOLECULES})
for _mol, _specific in RARER_CONFIGS.items():
    MODEL_CONFIGS[_mol] = {**CARBON_CONFIG, **_specific}

MOLECULES = []
if run_H2O:
    MOLECULES.append('H2O')
if run_Cmol:
    MOLECULES.extend(CARBON_MOLECULES)
if run_rarer:
    MOLECULES.extend(RARER_CONFIGS)
requested_molecules = os.environ.get('DISKMELTS_MOLECULES')
if requested_molecules:
    requested = [item.strip() for item in requested_molecules.split(',') if item.strip()]
    unknown = [item for item in requested if item not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f'Unknown DISKMELTS_MOLECULES entries: {unknown}')
    MOLECULES = requested
if not MOLECULES:
    raise ValueError('No molecules selected: enable at least one run_* switch')


# ---------------------------------------------------------------------------
# Training and validation controls
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIDDEN = (64, 128, 64)
N_EPOCHS = int(os.environ.get('DISKMELTS_N_EPOCHS', '5000'))
BATCH = 128
LR = 1.0e-4
PATIENCE = 500
SEED = 42

# This must remain between 5% and 10% for the maintained scientific workflow.
HOLDOUT_FRACTION = float(os.environ.get('DISKMELTS_HOLDOUT_FRACTION', '0.10'))
if not 0.05 <= HOLDOUT_FRACTION <= 0.10:
    raise ValueError('DISKMELTS_HOLDOUT_FRACTION must be between 0.05 and 0.10')

TUNE_PCA = os.environ.get('DISKMELTS_TUNE_PCA', '1') != '0'
PCA_TOLERANCE = float(os.environ.get('DISKMELTS_PCA_TOLERANCE', '0.05'))
RUN_VALIDATION = os.environ.get('DISKMELTS_VALIDATE', '1') != '0'
COMPARE_V1 = os.environ.get('DISKMELTS_COMPARE_V1', '1') != '0'
V1_BENCHMARK_TOLERANCE = float(
    os.environ.get('DISKMELTS_V1_BENCHMARK_TOLERANCE', '0.10')
)
VAL_N = int(os.environ.get('DISKMELTS_VAL_N', '30'))
VAL_NT_STARTS = int(os.environ.get('DISKMELTS_VAL_NT_STARTS', '100'))
VAL_NT_STEPS = int(os.environ.get('DISKMELTS_VAL_NT_STEPS', '300'))
VAL_FULL_N = int(os.environ.get('DISKMELTS_VAL_FULL_N', '10'))
VAL_FULL_RESTARTS = int(os.environ.get('DISKMELTS_VAL_FULL_RESTARTS', '50'))
RECON_N = int(os.environ.get('DISKMELTS_RECON_N', '100'))

if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f'Device: {device}  |  Molecules: {MOLECULES}')


# ---------------------------------------------------------------------------
# Train and validate molecules sequentially to bound memory use.
# ---------------------------------------------------------------------------

for molecule_number, MOL in enumerate(MOLECULES, start=1):
    config = MODEL_CONFIGS[MOL]
    if MOL == 'H2O':
        wavelength_ranges = [tuple(item) for item in config['wavelength_ranges']]
        wav_range = (
            min(item[0] for item in wavelength_ranges),
            max(item[1] for item in wavelength_ranges),
        )
        T_range = tuple(config['T_range'])
        logN_range = tuple(config['logN_range'])
        grid_dir = os.path.join(BASE_DIR, 'Model_grids', MOL)
        pretrain_csv = os.path.join(
            BASE_DIR, 'Pretrain_grid', 'pretrain_H2O_wavelength_bank_T100to1400_N13to19.csv'
        )
        holdout_path = os.path.join(
            BASE_DIR, 'Pretrain_grid', 'holdout_H2O_wavelength_bank_T100to1400_N13to19.json'
        )
        model_dir = os.path.join(BASE_DIR, 'Trained_model', 'H2O_wavelength_bank_v2_fixed')
        manifest_path = os.path.join(model_dir, 'H2O_model_bank.json')
        figure_dir = os.path.join(
            BASE_DIR, 'figures', 'validation', 'H2O_wavelength_bank_v2_fixed'
        )

        print('\n' + '=' * 76)
        print(
            f'[{molecule_number}/{len(MOLECULES)}] H2O: '
            f'{len(wavelength_ranges)} wavelength models, '
            f'T={T_range}, logN={logN_range}'
        )
        print('=' * 76)
        if not os.path.isdir(grid_dir):
            raise FileNotFoundError(
                f'Model grid not found: {grid_dir}\n'
                'Model_grids/ is intentionally ignored by Git. Copy the full local '
                'model grids into the repository before running training validation.'
            )
        os.makedirs(os.path.dirname(pretrain_csv), exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(figure_dir, exist_ok=True)

        print('\n[1] Loading and restricting H2O model grid ...')
        models = load_model_grid(
            grid_dir,
            T_range=T_range,
            logN_range=logN_range,
            wav_range=wav_range,
        )
        if not models:
            raise ValueError('H2O: requested T/logN domain selects zero grid models')
        wav_out = next(iter(models.values()))['wavelength']
        holdout_keys = select_grid_holdout(
            models,
            fraction=HOLDOUT_FRACTION,
            seed=SEED,
            preserve_boundaries=True,
        )
        holdout_metadata = {
            'mol': MOL,
            'seed': SEED,
            'fraction': HOLDOUT_FRACTION,
            'wavelength_ranges': [list(item) for item in wavelength_ranges],
            'T_range': list(T_range),
            'logN_range': list(logN_range),
            'keys': [list(key) for key in holdout_keys],
        }
        print(
            f'    retained {len(models)} grid points, {len(wav_out)} wavelengths, '
            f'{len(holdout_keys)} unseen models'
        )

        print('\n[2] Shared full-range pretraining CSV ...')
        reuse_csv = False
        if os.path.isfile(pretrain_csv) and os.path.isfile(holdout_path):
            with open(holdout_path, encoding='utf-8') as stream:
                reuse_csv = json.load(stream) == holdout_metadata
        if reuse_csv:
            print(f'    Exact configuration exists, reusing -> {pretrain_csv}')
        else:
            generate_pre_training_set(
                MOL,
                models,
                output_path=pretrain_csv,
                wav_out=wav_out,
                noise_scale=0.0,
                n_noise=1,
                seed=SEED,
                holdout_keys=holdout_keys,
            )
            with open(holdout_path, 'w', encoding='utf-8') as stream:
                json.dump(holdout_metadata, stream, indent=2)
                stream.write('\n')

        pca_tuning = []
        selected_pca = []
        print('\n[3] Selecting PCA size independently for each wavelength model ...')
        for tile_range in wavelength_ranges:
            if TUNE_PCA:
                tuning = tune_pca_components(
                    models,
                    holdout_keys=holdout_keys,
                    wav_range=tile_range,
                    candidates=config['pca_candidates'],
                    seed=SEED,
                    tolerance=PCA_TOLERANCE,
                )
                selected = tuning['selected']
                pca_tuning.append(tuning)
                print(
                    f'    wav={tile_range}: PCA={selected}, '
                    f'holdout RMSE={tuning["rmse"][selected]:.6g}'
                )
            else:
                selected = int(config['n_pca'])
                pca_tuning.append(None)
                print(f'    wav={tile_range}: PCA={selected} (tuning disabled)')
            selected_pca.append(selected)

        print('\n[4] Training/loading five H2O wavelength models ...')
        bank_training = train_model_bank(
            mol=MOL,
            pretrain_csv=pretrain_csv,
            model_dir=model_dir,
            wavelength_ranges=wavelength_ranges,
            logN_ranges=[logN_range],
            T_ranges=[T_range],
            device=device,
            hidden=HIDDEN,
            n_epochs=N_EPOCHS,
            batch_size=BATCH,
            lr=LR,
            seed=SEED,
            n_pca=selected_pca,
            early_stopping_patience=PATIENCE,
            manifest_path=manifest_path,
            peak_normalization='window',
            shape_loss='spectral',
            scheduler_patience=100,
            min_lr=1e-6,
            temperature_transform='log10',
        )
        if not RUN_VALIDATION:
            continue

        rng = np.random.default_rng(SEED)
        shuffled_holdouts = list(holdout_keys)
        rng.shuffle(shuffled_holdouts)
        nt_keys = shuffled_holdouts[:min(VAL_N, len(shuffled_holdouts))]
        tile_nt_metrics = []
        print('\n[5] Per-wavelength unseen T/logN validation ...')
        for tile_spec, tile_model in zip(
            bank_training['tiles'], bank_training['models']
        ):
            tile_range = tuple(tile_spec['wav_range'])
            tile_result = validate_nt_holdout(
                MOL,
                pretrained={MOL: tile_model},
                models=models,
                holdout_keys=nt_keys,
                fit_ranges=tile_range,
                T_bounds=T_range,
                logN_bounds=logN_range,
                n_starts=VAL_NT_STARTS,
                n_steps=VAL_NT_STEPS,
                lr=0.03,
            )
            tile_tag = os.path.splitext(os.path.basename(tile_spec['path']))[0]
            np.savez(os.path.join(figure_dir, f'{tile_tag}_holdout_nt.npz'), **tile_result)
            plot_validation(
                MOL,
                tile_result,
                save_path=os.path.join(figure_dir, f'{tile_tag}_holdout_nt.png'),
            )
            tile_nt_metrics.append({
                'wav_range': list(tile_range),
                'n_pca': tile_spec['n_pca'],
                'n_evaluated': len(tile_result['T_true']),
                'n_dark_training': tile_model.get('n_dark_training', 0),
                'n_dark_validation': tile_model.get('n_dark_validation', 0),
                'T_rmse_K': float(np.sqrt(np.mean(
                    (tile_result['T_pred'] - tile_result['T_true']) ** 2
                ))),
                'logN_rmse': float(np.sqrt(np.mean(
                    (tile_result['logN_pred'] - tile_result['logN_true']) ** 2
                ))),
            })

        full_keys = shuffled_holdouts[:min(VAL_FULL_N, len(shuffled_holdouts))]
        print(f'\n[6] Full assembled-bank fitting ({len(full_keys)} spectra) ...')
        full_result = validate_full(
            MOL,
            models={key: models[key] for key in full_keys},
            pretrained={MOL: bank_training['model_bank']},
            fit_ranges=wav_range,
            n_samples=len(full_keys),
            noise_scale=0.0,
            T_bounds=T_range,
            logN_bounds=logN_range,
            n_restarts=VAL_FULL_RESTARTS,
            seed=SEED,
        )
        plot_validation(
            MOL,
            full_result,
            save_path=os.path.join(figure_dir, 'H2O_bank_holdout_full.png'),
        )
        np.savez(os.path.join(figure_dir, 'H2O_bank_holdout_full.npz'), **full_result)

        recon_keys = shuffled_holdouts[:min(RECON_N, len(shuffled_holdouts))]
        true_fluxes = []
        predicted_fluxes = []
        relative_rmse = []
        for key in recon_keys:
            truth = np.interp(wav_out, models[key]['wavelength'], models[key]['flux'])
            prediction = generate_spectrum(
                key[0], key[1], 1.0, bank_training['model_bank'], obs_wav=wav_out
            )
            peak = max(float(np.max(np.abs(truth))), 1e-30)
            true_fluxes.append(truth)
            predicted_fluxes.append(prediction)
            relative_rmse.append(float(np.sqrt(np.mean(
                ((prediction - truth) / peak) ** 2
            ))))
        plot_bank_reconstruction(
            wav_out,
            np.asarray(true_fluxes),
            np.asarray(predicted_fluxes),
            mol=MOL,
            save_path=os.path.join(figure_dir, 'H2O_bank_holdout_reconstruction.png'),
        )

        v1_summary = None
        legacy_path = os.path.join(
            BASE_DIR, 'Trained_model', 'net_H2O_forward_11to19.pt'
        )
        if COMPARE_V1 and os.path.isfile(legacy_path):
            legacy = load_models({MOL: legacy_path}, device=device)
            legacy_wav = legacy[MOL]['wav']
            comparison_range = (
                max(wav_range[0], float(np.min(legacy_wav))),
                min(wav_range[1], float(np.max(legacy_wav))),
            )
            common_models = {key: models[key] for key in full_keys}
            v2_common = validate_full(
                MOL, common_models, {MOL: bank_training['model_bank']},
                fit_ranges=comparison_range,
                n_samples=len(full_keys), noise_scale=0.0,
                T_bounds=T_range, logN_bounds=logN_range,
                n_restarts=VAL_FULL_RESTARTS, seed=SEED,
            )
            v1_common = validate_full(
                MOL, common_models, legacy,
                fit_ranges=comparison_range,
                n_samples=len(full_keys), noise_scale=0.0,
                T_bounds=T_range, logN_bounds=logN_range,
                n_restarts=VAL_FULL_RESTARTS, seed=SEED,
            )
            v2_T = float(np.sqrt(np.mean(
                (v2_common['T_pred'] - v2_common['T_true']) ** 2
            )))
            v1_T = float(np.sqrt(np.mean(
                (v1_common['T_pred'] - v1_common['T_true']) ** 2
            )))
            v2_NA = float(np.sqrt(np.mean(
                (v2_common['log10NA_pred'] - v2_common['log10NA_true']) ** 2
            )))
            v1_NA = float(np.sqrt(np.mean(
                (v1_common['log10NA_pred'] - v1_common['log10NA_true']) ** 2
            )))
            v1_summary = {
                'checkpoint': legacy_path,
                'wav_range': list(comparison_range),
                'v2_full_T_rmse_K': v2_T,
                'v1_full_T_rmse_K': v1_T,
                'v2_full_log10NA_rmse': v2_NA,
                'v1_full_log10NA_rmse': v1_NA,
                'allowed_relative_degradation': V1_BENCHMARK_TOLERANCE,
                'T_rmse_ratio_v2_over_v1': v2_T / max(v1_T, 1e-30),
                'log10NA_rmse_ratio_v2_over_v1': v2_NA / max(v1_NA, 1e-30),
                'passes_selected_metrics': (
                    v2_T <= (1.0 + V1_BENCHMARK_TOLERANCE) * v1_T
                    and v2_NA <= (1.0 + V1_BENCHMARK_TOLERANCE) * v1_NA
                ),
            }

        summary = {
            'mol': MOL,
            'manifest': manifest_path,
            'wavelength_ranges': [list(item) for item in wavelength_ranges],
            'T_range': list(T_range),
            'logN_range': list(logN_range),
            'n_pca': selected_pca,
            'peak_normalization': 'window',
            'shape_loss': 'spectral',
            'temperature_transform': 'log10',
            'validation_seed': SEED,
            'n_nt_validation': len(nt_keys),
            'n_full_validation': len(full_keys),
            'pca_tuning': pca_tuning,
            'holdout_fraction': HOLDOUT_FRACTION,
            'n_grid': len(models),
            'n_holdout': len(holdout_keys),
            'tile_nt_metrics': tile_nt_metrics,
            'full_T_rmse_K': float(np.sqrt(np.mean(
                (full_result['T_pred'] - full_result['T_true']) ** 2
            ))),
            'full_logN_rmse': float(np.sqrt(np.mean(
                (full_result['logN_pred'] - full_result['logN_true']) ** 2
            ))),
            'full_log10A_rmse': float(np.sqrt(np.mean(
                (full_result['log10A_pred'] - full_result['log10A_true']) ** 2
            ))),
            'full_log10NA_rmse': float(np.sqrt(np.mean(
                (full_result['log10NA_pred'] - full_result['log10NA_true']) ** 2
            ))),
            'reconstruction_relative_rmse_median': float(np.median(relative_rmse)),
            'reconstruction_relative_rmse_max': float(np.max(relative_rmse)),
            'v1_benchmark': v1_summary,
        }
        summary_path = os.path.join(
            figure_dir, 'H2O_bank_validation_summary.json'
        )
        with open(summary_path, 'w', encoding='utf-8') as stream:
            json.dump(summary, stream, indent=2)
            stream.write('\n')
        print(json.dumps(summary, indent=2))
        continue

    wav_range = tuple(config['wav_range'])
    T_range = tuple(config['T_range'])
    logN_range = tuple(config['logN_range'])

    grid_dir = os.path.join(BASE_DIR, 'Model_grids', MOL)
    data_tag = 'single_T100to1400_N13to19p5' if MOL != 'H2O' else 'single_T100to1400_N13to19'
    pretrain_csv = os.path.join(BASE_DIR, 'Pretrain_grid', f'pretrain_{MOL}_{data_tag}.csv')
    holdout_path = os.path.join(BASE_DIR, 'Pretrain_grid', f'holdout_{MOL}_{data_tag}.json')
    model_dir = os.path.join(BASE_DIR, 'Trained_model', f'{MOL}_single_v2')
    model_path = os.path.join(model_dir, f'net_{MOL}_forward.pt')
    figure_dir = os.path.join(BASE_DIR, 'figures', 'validation', f'{MOL}_single_v2')

    print('\n' + '=' * 76)
    print(
        f'[{molecule_number}/{len(MOLECULES)}] {MOL}: one checkpoint, '
        f'wav={wav_range}, T={T_range}, logN={logN_range}'
    )
    print('=' * 76)

    if not os.path.isdir(grid_dir):
        raise FileNotFoundError(
            f'Model grid not found: {grid_dir}\n'
            'Model_grids/ is intentionally ignored by Git. Copy the full local '
            'model grids into the repository before running training validation.'
        )
    os.makedirs(os.path.dirname(pretrain_csv), exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print('\n[1] Loading and restricting model grid ...')
    models = load_model_grid(
        grid_dir,
        T_range=T_range,
        logN_range=logN_range,
        wav_range=wav_range,
    )
    if not models:
        raise ValueError(f'{MOL}: requested T/logN domain selects zero grid models')
    reference_wav = next(iter(models.values()))['wavelength']
    wav_out = reference_wav[
        (reference_wav >= wav_range[0]) & (reference_wav <= wav_range[1])
    ]
    print(f'    retained {len(models)} grid points and {len(wav_out)} wavelengths')

    holdout_keys = select_grid_holdout(
        models,
        fraction=HOLDOUT_FRACTION,
        seed=SEED,
        preserve_boundaries=True,
    )
    holdout_metadata = {
        'mol': MOL,
        'seed': SEED,
        'fraction': HOLDOUT_FRACTION,
        'wav_range': list(wav_range),
        'T_range': list(T_range),
        'logN_range': list(logN_range),
        'keys': [list(key) for key in holdout_keys],
    }

    print(f'\n[2] Pretraining CSV ({len(holdout_keys)} unseen grid models) ...')
    reuse_csv = False
    if os.path.isfile(pretrain_csv) and os.path.isfile(holdout_path):
        with open(holdout_path, encoding='utf-8') as stream:
            reuse_csv = json.load(stream) == holdout_metadata
    if reuse_csv:
        print(f'    Exact configuration exists, reusing -> {pretrain_csv}')
    else:
        generate_pre_training_set(
            MOL,
            models,
            output_path=pretrain_csv,
            wav_out=wav_out,
            noise_scale=0.0,
            n_noise=1,
            seed=SEED,
            holdout_keys=holdout_keys,
        )
        with open(holdout_path, 'w', encoding='utf-8') as stream:
            json.dump(holdout_metadata, stream, indent=2)
            stream.write('\n')

    selected_n_pca = int(config['n_pca'])
    hidden = tuple(config.get('hidden', HIDDEN))
    pca_tuning = None
    if TUNE_PCA:
        print('\n[3] Selecting PCA size on unseen spectra ...')
        pca_tuning = tune_pca_components(
            models,
            holdout_keys=holdout_keys,
            wav_range=wav_range,
            candidates=config['pca_candidates'],
            seed=SEED,
            tolerance=PCA_TOLERANCE,
        )
        selected_n_pca = pca_tuning['selected']
        for count, rmse in pca_tuning['rmse'].items():
            variance = pca_tuning['explained_variance'][count]
            print(f'    PCA={count:2d}: holdout RMSE={rmse:.6g}, variance={variance:.7f}')
        print(f'    selected PCA={selected_n_pca}')
    else:
        print(f'\n[3] PCA tuning disabled; using PCA={selected_n_pca}')

    print('\n[4] Training/loading single checkpoint ...')
    trained = pretrain_forward_model(
        mol=MOL,
        pretrain_csv=pretrain_csv,
        wav_range=wav_range,
        device=device,
        hidden=hidden,
        n_epochs=N_EPOCHS,
        batch_size=BATCH,
        lr=LR,
        model_path=model_path,
        seed=SEED,
        n_pca=selected_n_pca,
        early_stopping_patience=PATIENCE,
        T_range=T_range,
        logN_range=logN_range,
    )

    if not RUN_VALIDATION:
        continue

    rng = np.random.default_rng(SEED)
    shuffled_holdouts = list(holdout_keys)
    rng.shuffle(shuffled_holdouts)
    nt_keys = shuffled_holdouts[:min(VAL_N, len(shuffled_holdouts))]

    print(f'\n[5] Unseen-model T/logN retrieval ({len(nt_keys)} spectra) ...')
    nt_result = validate_nt_holdout(
        MOL,
        pretrained={MOL: trained},
        models=models,
        holdout_keys=nt_keys,
        fit_ranges=wav_range,
        T_bounds=T_range,
        logN_bounds=logN_range,
        n_starts=VAL_NT_STARTS,
        n_steps=VAL_NT_STEPS,
        lr=0.03,
    )
    plot_validation(
        MOL,
        nt_result,
        save_path=os.path.join(figure_dir, f'{MOL}_holdout_nt.png'),
    )

    full_keys = shuffled_holdouts[:min(VAL_FULL_N, len(shuffled_holdouts))]
    print(f'\n[6] Full T/logN/A fitting ({len(full_keys)} unseen spectra) ...')
    full_result = validate_full(
        MOL,
        models={key: models[key] for key in full_keys},
        pretrained={MOL: trained},
        fit_ranges=wav_range,
        n_samples=len(full_keys),
        noise_scale=0.0,
        T_bounds=T_range,
        logN_bounds=logN_range,
        n_restarts=VAL_FULL_RESTARTS,
        seed=SEED,
    )
    plot_validation(
        MOL,
        full_result,
        save_path=os.path.join(figure_dir, f'{MOL}_holdout_full.png'),
    )

    recon_keys = shuffled_holdouts[:min(RECON_N, len(shuffled_holdouts))]
    true_fluxes = []
    predicted_fluxes = []
    relative_rmse = []
    for key in recon_keys:
        truth = np.interp(wav_out, models[key]['wavelength'], models[key]['flux'])
        prediction = generate_spectrum(
            key[0], key[1], 1.0, trained, obs_wav=wav_out
        )
        peak = max(float(np.max(np.abs(truth))), 1e-30)
        true_fluxes.append(truth)
        predicted_fluxes.append(prediction)
        relative_rmse.append(float(np.sqrt(np.mean(((prediction - truth) / peak) ** 2))))
    plot_bank_reconstruction(
        wav_out,
        np.asarray(true_fluxes),
        np.asarray(predicted_fluxes),
        mol=MOL,
        save_path=os.path.join(figure_dir, f'{MOL}_holdout_reconstruction.png'),
    )

    v1_summary = None
    legacy_path = os.path.join(BASE_DIR, 'Trained_model', f'net_{MOL}_forward_11to19.pt')
    if COMPARE_V1 and os.path.isfile(legacy_path):
        legacy = load_models({MOL: legacy_path}, device=device)
        legacy_wav = legacy[MOL]['wav']
        comparison_range = (
            max(wav_range[0], float(np.min(legacy_wav))),
            min(wav_range[1], float(np.max(legacy_wav))),
        )
        comparison_keys = [key for key in shuffled_holdouts if key[1] <= 19.0]
        legacy_nt_keys = comparison_keys[:min(VAL_N, len(comparison_keys))]
        legacy_full_keys = comparison_keys[:min(VAL_FULL_N, len(comparison_keys))]
        print(
            f'\n[7] V1 benchmark on common range {comparison_range} '
            f'({len(legacy_nt_keys)} unseen spectra) ...'
        )
        common_nt = validate_nt_holdout(
            MOL,
            pretrained={MOL: trained},
            models=models,
            holdout_keys=legacy_nt_keys,
            fit_ranges=comparison_range,
            T_bounds=(100.0, 1400.0),
            logN_bounds=(13.0, 19.0),
            n_starts=VAL_NT_STARTS,
            n_steps=VAL_NT_STEPS,
            lr=0.03,
        )
        common_full = validate_full(
            MOL,
            models={key: models[key] for key in legacy_full_keys},
            pretrained={MOL: trained},
            fit_ranges=comparison_range,
            n_samples=len(legacy_full_keys),
            noise_scale=0.0,
            T_bounds=(100.0, 1400.0),
            logN_bounds=(13.0, 19.0),
            n_restarts=VAL_FULL_RESTARTS,
            seed=SEED,
        )
        legacy_nt = validate_nt_holdout(
            MOL,
            pretrained=legacy,
            models=models,
            holdout_keys=legacy_nt_keys,
            fit_ranges=comparison_range,
            T_bounds=(100.0, 1400.0),
            logN_bounds=(13.0, 19.0),
            n_starts=VAL_NT_STARTS,
            n_steps=VAL_NT_STEPS,
            lr=0.03,
        )
        legacy_full = validate_full(
            MOL,
            models={key: models[key] for key in legacy_full_keys},
            pretrained=legacy,
            fit_ranges=comparison_range,
            n_samples=len(legacy_full_keys),
            noise_scale=0.0,
            T_bounds=(100.0, 1400.0),
            logN_bounds=(13.0, 19.0),
            n_restarts=VAL_FULL_RESTARTS,
            seed=SEED,
        )
        v2_common_metrics = {
            'T_rmse_K': float(np.sqrt(np.mean((common_nt['T_pred'] - common_nt['T_true']) ** 2))),
            'logN_rmse': float(np.sqrt(np.mean((common_nt['logN_pred'] - common_nt['logN_true']) ** 2))),
            'full_T_rmse_K': float(np.sqrt(np.mean((common_full['T_pred'] - common_full['T_true']) ** 2))),
            'full_logN_rmse': float(np.sqrt(np.mean((common_full['logN_pred'] - common_full['logN_true']) ** 2))),
            'full_log10A_rmse': float(np.sqrt(np.mean((common_full['log10A_pred'] - common_full['log10A_true']) ** 2))),
            'full_log10NA_rmse': float(np.sqrt(np.mean((common_full['log10NA_pred'] - common_full['log10NA_true']) ** 2))),
        }
        v1_metrics = {
            'wav_range': list(comparison_range),
            'T_rmse_K': float(np.sqrt(np.mean((legacy_nt['T_pred'] - legacy_nt['T_true']) ** 2))),
            'logN_rmse': float(np.sqrt(np.mean((legacy_nt['logN_pred'] - legacy_nt['logN_true']) ** 2))),
            'full_T_rmse_K': float(np.sqrt(np.mean((legacy_full['T_pred'] - legacy_full['T_true']) ** 2))),
            'full_logN_rmse': float(np.sqrt(np.mean((legacy_full['logN_pred'] - legacy_full['logN_true']) ** 2))),
            'full_log10A_rmse': float(np.sqrt(np.mean((legacy_full['log10A_pred'] - legacy_full['log10A_true']) ** 2))),
            'full_log10NA_rmse': float(np.sqrt(np.mean((legacy_full['log10NA_pred'] - legacy_full['log10NA_true']) ** 2))),
        }
        benchmark_metrics = ('T_rmse_K', 'logN_rmse', 'full_T_rmse_K', 'full_log10NA_rmse')
        metric_ratios = {
            metric: v2_common_metrics[metric] / max(v1_metrics[metric], 1e-30)
            for metric in benchmark_metrics
        }
        v1_summary = {
            'checkpoint': legacy_path,
            'wav_range': list(comparison_range),
            'v2_common_holdout': v2_common_metrics,
            'v1_common_holdout': v1_metrics,
            'allowed_relative_degradation': V1_BENCHMARK_TOLERANCE,
            'v2_to_v1_rmse_ratio': metric_ratios,
            'passes_selected_metrics': all(
                metric_ratios[metric] <= 1.0 + V1_BENCHMARK_TOLERANCE
                for metric in benchmark_metrics
            ),
        }

    summary = {
        'mol': MOL,
        'checkpoint': model_path,
        'n_pca': selected_n_pca,
        'pca_tuning': pca_tuning,
        'holdout_fraction': HOLDOUT_FRACTION,
        'n_grid': len(models),
        'n_holdout': len(holdout_keys),
        'T_rmse_K': float(np.sqrt(np.mean((nt_result['T_pred'] - nt_result['T_true']) ** 2))),
        'logN_rmse': float(np.sqrt(np.mean((nt_result['logN_pred'] - nt_result['logN_true']) ** 2))),
        'full_T_rmse_K': float(np.sqrt(np.mean((full_result['T_pred'] - full_result['T_true']) ** 2))),
        'full_logN_rmse': float(np.sqrt(np.mean((full_result['logN_pred'] - full_result['logN_true']) ** 2))),
        'full_log10A_rmse': float(np.sqrt(np.mean((full_result['log10A_pred'] - full_result['log10A_true']) ** 2))),
        'full_log10NA_rmse': float(np.sqrt(np.mean((full_result['log10NA_pred'] - full_result['log10NA_true']) ** 2))),
        'reconstruction_relative_rmse_median': float(np.median(relative_rmse)),
        'reconstruction_relative_rmse_max': float(np.max(relative_rmse)),
        'v1_benchmark': v1_summary,
    }
    with open(os.path.join(figure_dir, f'{MOL}_validation_summary.json'), 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))

print('\nDone.')
