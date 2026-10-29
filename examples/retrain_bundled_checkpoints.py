"""Retrain every bundled surrogate into an isolated, resumable directory.

Run with ``PYTHONPATH=src python examples/retrain_bundled_checkpoints.py``.
Existing output checkpoints are reused, so an interrupted run can be resumed.
The bundled production checkpoints are never overwritten by this script.
"""

from pathlib import Path
import shutil

import torch

from diskmelts.trainmodel import pretrain_forward_model


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/diskmelts/models/Trained_model'
DEST = ROOT / 'Trained_model/retrained_2026_10_27'


def training_csv(sigma, mol, wav_range):
    base = ROOT / 'Pretrain_grid' / (f'sigma{sigma}' if sigma else '')
    if mol == 'H2O':
        stem = 'H2O_wavelength_bank_T100to1400_N13to19'
    elif mol == '13C12CH2' and sigma == 0:
        stem = '13C12CH2_single_T100to1400_N13to19p5_newgrid'
    elif mol == 'C2H2' and wav_range[1] <= 9:
        stem = 'C2H2_w6to9_v2'
    elif mol in {'OH', 'CH3', 'C3H4'}:
        stem = {'OH': 'OH_w15to26_v2', 'CH3': 'CH3_w15to18_v2',
                'C3H4': 'C3H4_w15to18_v2'}[mol]
    elif mol == 'C6H6':
        stem = 'C6H6_w13to17_v2_T100to675'
    else:
        stem = f'{mol}_single_T100to1400_N13to19p5'
    path = base / f'pretrain_{stem}.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    paths = sorted(SOURCE.glob('sigma*/*/*.pt'))
    if len(paths) != 46:
        raise RuntimeError(f'Expected 46 bundled checkpoints, found {len(paths)}')
    for number, source in enumerate(paths, 1):
        relative = source.relative_to(SOURCE)
        destination = DEST / relative
        checkpoint = torch.load(source, map_location='cpu', weights_only=True)
        mol = checkpoint['mol']
        sigma = int(relative.parts[0].removeprefix('sigma'))
        wav_range = tuple(checkpoint['wav_range'])
        csv = training_csv(sigma, mol, wav_range)
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f'[{number}/{len(paths)}] {relative}: {csv.name}', flush=True)
        pretrain_forward_model(
            mol=mol, pretrain_csv=csv, wav_range=wav_range,
            T_range=tuple(checkpoint['T_range']),
            logN_range=tuple(checkpoint['logN_range']),
            hidden=tuple(checkpoint['hidden']), n_pca=checkpoint['n_pca'],
            model_path=destination, device='cpu', n_epochs=5000,
            batch_size=128, lr=1e-4, early_stopping_patience=500,
            seed=checkpoint.get('seed', 42),
            peak_normalization=checkpoint.get('peak_normalization') or 'csv',
            shape_loss=checkpoint.get('shape_loss') or 'standardized',
            temperature_transform=checkpoint.get('temperature_transform') or 'linear',
            scheduler_patience=checkpoint.get('scheduler_patience') or 10,
            min_lr=checkpoint.get('min_lr') or 0.0,
            training_data_sha256=checkpoint.get('training_data_sha256'),
        )
        print(f'Completed {relative}', flush=True)
    for manifest in SOURCE.glob('sigma*/*/*.json'):
        destination = DEST / manifest.relative_to(SOURCE)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, destination)
    print(f'All {len(paths)} retrainings complete in {DEST}', flush=True)


if __name__ == '__main__':
    main()
