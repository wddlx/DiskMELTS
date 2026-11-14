"""Train the broad-temperature OH wavelength bank (100-6000 K) from local grids.

At high temperature OH emits from 5 to 27 µm, so this model covers the whole
grid wavelength axis with three overlapping wavelength tiles. It uses the same
training recipe as the bundled checkpoints. Example, from the repository root::

    python examples/train_oh_broadT_bank.py --sigma 0
    python examples/train_oh_broadT_bank.py --sigma 2

Outputs go to the ignored top-level Trained_model/ directory; copy the
validated bank into src/diskmelts/models/Trained_model/sigma{0,2}/ by hand.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diskmelts.fitting import generate_spectrum
from diskmelts.trainmodel import (
    generate_pre_training_set,
    grid_sha256,
    load_model_grid,
    select_grid_holdout,
    train_model_bank,
    tune_pca_components,
)
from diskmelts.validation import validate_nt_holdout


ROOT = Path(__file__).resolve().parents[1]
MOL = 'OH'
TAG = 'OH_broadT_bank_v2'
T_RANGE = (100.0, 6000.0)
LOGN_RANGE = (13.0, 19.5)
WAVELENGTH_RANGES = ((4.9, 10.5), (9.5, 16.0), (15.0, 27.0))
SEED = 42


def _relative_rmse(predicted, truth, mask=None):
    if mask is not None:
        predicted, truth = predicted[mask], truth[mask]
    peak = float(np.max(np.abs(truth)))
    if peak == 0:
        return None
    return float(np.sqrt(np.mean(((predicted - truth) / peak) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sigma', type=int, choices=(0, 2), default=0)
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--pca-candidates', type=int, nargs='+',
                        default=[30, 50, 75, 100])
    parser.add_argument('--tag', default=TAG,
                        help='Output directory name for checkpoints and figures')
    parser.add_argument('--n-nt', type=int, default=30,
                        help='Holdouts used for the T/logN retrieval check')
    args = parser.parse_args()

    grid_dir = ROOT / 'Model_grids' / f'sigma{args.sigma}' / MOL
    wav_span = (WAVELENGTH_RANGES[0][0], WAVELENGTH_RANGES[-1][1])
    models = load_model_grid(grid_dir, T_range=T_RANGE, logN_range=LOGN_RANGE,
                             wav_range=wav_span)
    wav = next(iter(models.values()))['wavelength']
    if any(not np.array_equal(item['wavelength'], wav) for item in models.values()):
        raise ValueError('grid wavelength axes differ')
    temperatures = sorted({key[0] for key in models})
    print(f'{MOL} sigma={args.sigma}: {len(models)} grid spectra, '
          f'{len(temperatures)} temperatures {temperatures[0]}-{temperatures[-1]} K, '
          f'{len(wav)} channels {wav[0]:.3f}-{wav[-1]:.3f} µm')

    holdout_keys = select_grid_holdout(models, fraction=0.1, seed=SEED,
                                       preserve_boundaries=True)
    fingerprint = grid_sha256(models)
    output_subdir = () if args.sigma == 0 else (f'sigma{args.sigma}',)
    pretrain_dir = ROOT / 'Pretrain_grid' / Path(*output_subdir)
    model_dir = ROOT / 'Trained_model' / Path(*output_subdir) / args.tag
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    pretrain_csv = pretrain_dir / f'pretrain_{TAG}.csv'
    holdout_path = pretrain_dir / f'holdout_{TAG}.json'
    metadata = {
        'mol': MOL, 'grid_sigma': args.sigma,
        'wavelength_ranges': [list(item) for item in WAVELENGTH_RANGES],
        'T_range': list(T_RANGE), 'logN_range': list(LOGN_RANGE),
        'seed': SEED, 'holdout_keys': [list(key) for key in holdout_keys],
        'grid_sha256': fingerprint,
    }
    if not (pretrain_csv.is_file() and holdout_path.is_file() and
            json.loads(holdout_path.read_text()) == metadata):
        generate_pre_training_set(
            MOL, models, output_path=pretrain_csv, wav_out=wav,
            noise_scale=0.0, n_noise=1, seed=SEED, holdout_keys=holdout_keys,
        )
        holdout_path.write_text(json.dumps(metadata, indent=2) + '\n')

    selected_pca, pca_rmse = [], []
    for wav_range in WAVELENGTH_RANGES:
        tuning = tune_pca_components(
            models, holdout_keys=holdout_keys, wav_range=wav_range,
            candidates=args.pca_candidates, seed=SEED, tolerance=0.05,
        )
        selected_pca.append(tuning['selected'])
        pca_rmse.append(tuning['rmse'])
        print(f'  wav={wav_range}: PCA={tuning["selected"]}; RMSE {tuning["rmse"]}')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    bank = train_model_bank(
        mol=MOL, pretrain_csv=pretrain_csv, model_dir=model_dir,
        wavelength_ranges=WAVELENGTH_RANGES, logN_ranges=[LOGN_RANGE],
        T_ranges=[T_RANGE], device=device, hidden=(64, 128, 64),
        n_epochs=args.epochs, batch_size=128, lr=1e-4, seed=SEED,
        n_pca=selected_pca, early_stopping_patience=args.patience,
        peak_normalization='window', shape_loss='spectral',
        scheduler_patience=100, min_lr=1e-6, temperature_transform='log10',
        grid_sigma=args.sigma,
    )
    model = bank['model_bank']
    # Package-ready manifest: named after the registry key OH_BROAD_T (the
    # loader requires that) and free of the local pretrain CSV path.
    manifest_path = Path(bank['manifest_path'])
    manifest = json.loads(manifest_path.read_text())
    manifest['mol'] = 'OH_broadT'
    manifest.pop('pretrain_csv', None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    # Reconstruction on unseen grid spectra: whole range and per tile, each
    # relative to the peak of the true spectrum in that range.
    rows = []
    for key in holdout_keys:
        truth = models[tuple(key)]['flux']
        predicted = generate_spectrum(key[0], key[1], 1.0, model, obs_wav=wav)
        row = {'T': key[0], 'logN': key[1],
               'full': _relative_rmse(predicted, truth),
               'peak_ratio': float(np.max(predicted) / np.max(truth))
               if np.max(truth) > 0 else None}
        for lo, hi in WAVELENGTH_RANGES:
            mask = (wav >= lo) & (wav <= hi)
            row[f'w{lo:g}-{hi:g}'] = _relative_rmse(predicted, truth, mask)
        rows.append(row)

    def _stats(name, subset=rows):
        values = np.array([row[name] for row in subset if row[name] is not None])
        return {'median': float(np.median(values)), 'p95': float(np.quantile(values, 0.95)),
                'max': float(np.max(values)), 'n': int(len(values))}

    summary = {
        **metadata, 'manifest': bank['manifest_path'],
        'device': str(device), 'n_pca': selected_pca, 'pca_holdout_rmse': pca_rmse,
        'n_grid': len(models), 'n_holdout': len(holdout_keys),
        'reconstruction_full': _stats('full'),
        'reconstruction_by_tile': {
            f'w{lo:g}-{hi:g}': _stats(f'w{lo:g}-{hi:g}') for lo, hi in WAVELENGTH_RANGES
        },
        'reconstruction_full_by_T': {
            label: _stats('full', [row for row in rows if lo <= row['T'] < hi])
            for label, lo, hi in (('100-1000', 100, 1000), ('1000-2000', 1000, 2000),
                                  ('2000-4000', 2000, 4000), ('4000-6000', 4000, 6001))
        },
    }
    ratios = np.array([row['peak_ratio'] for row in rows if row['peak_ratio'] is not None])
    summary['peak_ratio'] = {'median': float(np.median(ratios)),
                             'min': float(ratios.min()), 'max': float(ratios.max())}

    # T/logN retrieval, one tile at a time, on holdouts whose flux in that
    # tile reaches at least 5% of their whole-range peak.
    shuffled = list(holdout_keys)
    np.random.default_rng(SEED).shuffle(shuffled)
    summary['retrieval_nt_by_tile'] = {}
    for tile_spec, tile_model in zip(bank['tiles'], bank['models']):
        lo, hi = tile_spec['wav_range']
        mask = (wav >= lo) & (wav <= hi)
        bright = [key for key in shuffled
                  if np.max(models[tuple(key)]['flux'][mask])
                  >= 0.05 * np.max(models[tuple(key)]['flux'])][:args.n_nt]
        nt = validate_nt_holdout(
            MOL, {MOL: tile_model}, models, bright, (lo, hi),
            T_bounds=T_RANGE, logN_bounds=LOGN_RANGE,
            n_starts=100, n_steps=300, lr=0.03,
        )
        t_err = nt['T_pred'] - nt['T_true']
        summary['retrieval_nt_by_tile'][f'w{lo:g}-{hi:g}'] = {
            'n': int(len(t_err)),
            'T_true_range': [float(nt['T_true'].min()), float(nt['T_true'].max())],
            'T_rmse_K': float(np.sqrt(np.mean(t_err ** 2))),
            'T_rel_rmse': float(np.sqrt(np.mean((t_err / nt['T_true']) ** 2))),
            'logN_rmse': float(np.sqrt(np.mean(
                (nt['logN_pred'] - nt['logN_true']) ** 2))),
        }

    figure_dir = ROOT / 'figures' / 'validation' / Path(*output_subdir) / args.tag
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / 'holdout_reconstruction.json').write_text(json.dumps(rows, indent=1) + '\n')
    summary_path = figure_dir / 'training_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in (
        'n_pca', 'reconstruction_full', 'reconstruction_by_tile',
        'reconstruction_full_by_T', 'peak_ratio', 'retrieval_nt_by_tile')}, indent=2))
    print(f'Saved validation summary: {summary_path}')


if __name__ == '__main__':
    main()
