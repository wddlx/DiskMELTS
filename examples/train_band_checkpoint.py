"""Train an additional single-molecule wavelength checkpoint from local grids.

Example, from the repository root::

    python examples/train_band_checkpoint.py C2H2 6 9

The default output lives in a new directory under the ignored top-level
Trained_model/ directory, separate from production checkpoints.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from diskmelts.fitting import generate_spectrum
from diskmelts.plotting import plot_validation
from diskmelts.trainmodel import (
    generate_pre_training_set,
    grid_sha256,
    load_model_grid,
    pretrain_forward_model,
    select_grid_holdout,
    tune_pca_components,
)
from diskmelts.validation import validate_full, validate_nt_holdout


ROOT = Path(__file__).resolve().parents[1]
T_RANGE = (100.0, 1400.0)
LOGN_RANGE = (13.0, 19.5)
SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('molecule')
    parser.add_argument('wave_min', type=float)
    parser.add_argument('wave_max', type=float)
    parser.add_argument('--t-min', type=float, default=T_RANGE[0])
    parser.add_argument('--t-max', type=float, default=T_RANGE[1])
    parser.add_argument('--sigma', type=int, choices=(0, 2), default=0,
                        help='Grid broadening directory (default: 0)')
    parser.add_argument('--tag', help='Output name under Trained_model/')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--pca-candidates', type=int, nargs='+',
                        default=[10, 15, 20, 25, 30],
                        help='PCA sizes to compare on held-out grid spectra')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--tune-only', action='store_true',
                        help='Audit grids and PCA sizes without training')
    parser.add_argument('--validate-retrieval', action='store_true',
                        help='Check T/logN on 30 holdouts and full fits on 10')
    args = parser.parse_args()

    wav_range = (args.wave_min, args.wave_max)
    t_range = (args.t_min, args.t_max)
    if not 0 < wav_range[0] < wav_range[1]:
        parser.error('wavelength limits must be positive and increasing')
    if not 0 < t_range[0] < t_range[1]:
        parser.error('temperature limits must be positive and increasing')
    if args.epochs < 1 or args.patience < 1:
        parser.error('epochs and patience must be positive')
    if any(value < 1 for value in args.pca_candidates):
        parser.error('PCA candidates must be positive')
    tag = args.tag or f'{args.molecule}_w{args.wave_min:g}to{args.wave_max:g}_v2'
    if args.tag is None and t_range != T_RANGE:
        tag += f'_T{args.t_min:g}to{args.t_max:g}'
    if tag in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        parser.error('tag must be a simple directory name')

    grid_dir = ROOT / 'Model_grids' / f'sigma{args.sigma}' / args.molecule
    if not grid_dir.is_dir():
        raise FileNotFoundError(f'Missing grid directory: {grid_dir}')
    models = load_model_grid(
        grid_dir, T_range=t_range, logN_range=LOGN_RANGE,
        wav_range=wav_range,
    )
    if not models:
        raise ValueError(f'{args.molecule}: no grid spectra in the requested domain')
    wav = next(iter(models.values()))['wavelength']
    if len(wav) < 2:
        raise ValueError(f'{args.molecule}: wavelength window has fewer than two channels')
    if any(not np.array_equal(item['wavelength'], wav) for item in models.values()):
        raise ValueError(f'{args.molecule}: grid wavelength axes differ')

    peak_flux = np.array([np.max(np.abs(item['flux'])) for item in models.values()])
    peak_wav = np.array([
        wav[np.argmax(np.abs(item['flux']))] for item in models.values()
    ])
    positive = peak_flux > 0
    print(f'{args.molecule} {wav_range} µm: {len(models)} grid spectra; '
          f'{int(positive.sum())} have signal and {int((~positive).sum())} are dark')
    if not positive.any():
        raise ValueError('All spectra are zero in this wavelength window; no peak model can be trained')
    quantiles = np.quantile(peak_wav[positive], [0, 0.05, 0.5, 0.95, 1])
    print('Nonzero peak wavelengths (min, 5%, median, 95%, max): '
          + ', '.join(f'{value:.4f}' for value in quantiles) + ' µm')
    if args.audit_only:
        return

    holdout_keys = select_grid_holdout(
        models, fraction=0.1, seed=SEED, preserve_boundaries=True,
    )
    fingerprint = grid_sha256(models)
    output_subdir = () if args.sigma == 0 else (f'sigma{args.sigma}',)
    pretrain_dir = ROOT / 'Pretrain_grid' / Path(*output_subdir)
    model_dir = ROOT / 'Trained_model' / Path(*output_subdir) / tag
    figure_dir = ROOT / 'figures' / 'validation' / Path(*output_subdir) / tag
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pretrain_csv = pretrain_dir / f'pretrain_{tag}.csv'
    holdout_path = pretrain_dir / f'holdout_{tag}.json'
    model_path = model_dir / f'net_{args.molecule}_forward.pt'
    metadata = {
        'mol': args.molecule, 'grid_sigma': args.sigma,
        'wav_range': list(wav_range),
        'T_range': list(t_range), 'logN_range': list(LOGN_RANGE),
        'seed': SEED, 'holdout_keys': [list(key) for key in holdout_keys],
        'grid_sha256': fingerprint,
    }
    if not (pretrain_csv.is_file() and holdout_path.is_file() and
            json.loads(holdout_path.read_text()) == metadata):
        generate_pre_training_set(
            args.molecule, models, output_path=pretrain_csv, wav_out=wav,
            noise_scale=0.0, n_noise=1, seed=SEED, holdout_keys=holdout_keys,
        )
        holdout_path.write_text(json.dumps(metadata, indent=2) + '\n')

    tuning = tune_pca_components(
        models, holdout_keys=holdout_keys, wav_range=wav_range,
        candidates=args.pca_candidates, seed=SEED, tolerance=0.05,
    )
    print(f'Holdout PCA selection: {tuning["selected"]}; '
          f'RMSE by count: {tuning["rmse"]}')
    if args.tune_only:
        return
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    trained = pretrain_forward_model(
        mol=args.molecule, pretrain_csv=pretrain_csv, wav_range=wav_range,
        T_range=t_range, logN_range=LOGN_RANGE, device=device,
        hidden=(64, 128, 64), n_epochs=args.epochs, batch_size=128,
        lr=1e-4, seed=SEED, n_pca=tuning['selected'],
        early_stopping_patience=args.patience, model_path=model_path,
        peak_normalization='window', shape_loss='spectral',
        scheduler_patience=100, min_lr=1e-6,
        temperature_transform='log10', training_data_sha256=fingerprint,
    )

    errors = []
    for key in holdout_keys:
        truth = models[tuple(key)]['flux']
        peak = float(np.max(np.abs(truth)))
        if peak == 0:
            continue
        predicted = generate_spectrum(key[0], key[1], 1.0, trained, obs_wav=wav)
        errors.append(float(np.sqrt(np.mean(((predicted - truth) / peak) ** 2))))
    summary = {
        **metadata, 'checkpoint': str(model_path), 'device': str(device),
        'n_pca': tuning['selected'], 'pca_holdout_rmse': tuning['rmse'],
        'n_grid': len(models), 'n_nonzero': int(positive.sum()),
        'peak_wavelength_quantiles_um': quantiles.tolist(),
        'n_reconstruction_holdout': len(errors),
        'reconstruction_relative_rmse_median': float(np.median(errors)),
        'reconstruction_relative_rmse_max': float(np.max(errors)),
    }
    if args.validate_retrieval:
        shuffled = list(holdout_keys)
        np.random.default_rng(SEED).shuffle(shuffled)
        nt_keys = shuffled[:30]
        full_keys = shuffled[:10]
        nt = validate_nt_holdout(
            args.molecule, {args.molecule: trained}, models, nt_keys,
            wav_range, T_bounds=t_range, logN_bounds=LOGN_RANGE,
            n_starts=100, n_steps=300, lr=0.03,
        )
        plot_validation(
            args.molecule, nt,
            save_path=str(figure_dir / f'{args.molecule}_holdout_nt.png'),
        )
        full = validate_full(
            args.molecule, {key: models[key] for key in full_keys},
            {args.molecule: trained}, wav_range, n_samples=len(full_keys),
            noise_scale=0.0, T_bounds=t_range, logN_bounds=LOGN_RANGE,
            n_restarts=50, seed=SEED,
        )
        plot_validation(
            args.molecule, full,
            save_path=str(figure_dir / f'{args.molecule}_holdout_full.png'),
        )
        summary['retrieval_validation'] = {
            'n_nt': len(nt['T_true']), 'n_full': len(full['T_true']),
            'T_rmse_K': float(np.sqrt(np.mean((nt['T_pred'] - nt['T_true']) ** 2))),
            'logN_rmse': float(np.sqrt(np.mean((nt['logN_pred'] - nt['logN_true']) ** 2))),
            'full_T_rmse_K': float(np.sqrt(np.mean((full['T_pred'] - full['T_true']) ** 2))),
            'full_log10NA_rmse': float(np.sqrt(np.mean(
                (full['log10NA_pred'] - full['log10NA_true']) ** 2))),
        }
    summary_path = figure_dir / 'training_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Saved checkpoint: {model_path}')
    print(f'Saved validation summary: {summary_path}')
    print(f'Holdout relative RMSE: median={np.median(errors):.4g}, '
          f'max={np.max(errors):.4g}')


if __name__ == '__main__':
    main()
