"""Train targeted spectral-loss candidates for weak retrained checkpoints."""

import sys

import pandas as pd
import torch

from diskmelts.trainmodel import pretrain_forward_model
from retrain_bundled_checkpoints import ROOT, SOURCE, training_csv


DEST = ROOT / 'Trained_model/tuned_2026_10_28'
COMPARISON = ROOT / 'Trained_model/retrained_2026_10_27/holdout_comparison.csv'


def train_candidate(source, *, edge=False, variant=None, hidden=None, n_pca=None,
                    n_epochs=5000, patience=500):
    relative = source.relative_to(SOURCE)
    destination = DEST / (variant or ('edge' if edge else 'spectral')) / relative
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    mol = checkpoint['mol']
    sigma = int(relative.parts[0].removeprefix('sigma'))
    wav_range = tuple(checkpoint['wav_range'])
    csv = training_csv(sigma, mol, wav_range)
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f'Training {relative} ({"edge" if edge else "spectral"})', flush=True)
    pretrain_forward_model(
        mol=mol, pretrain_csv=csv, wav_range=wav_range,
        T_range=tuple(checkpoint['T_range']),
        logN_range=tuple(checkpoint['logN_range']),
        hidden=tuple(hidden or checkpoint['hidden']), n_pca=n_pca or checkpoint['n_pca'],
        model_path=destination, device='cpu', n_epochs=n_epochs,
        batch_size=128, lr=1e-4, early_stopping_patience=patience,
        seed=42, peak_normalization='window',
        shape_loss='spectral_edge' if edge else 'spectral',
        temperature_transform='log10',
        scheduler_patience=100, min_lr=1e-6,
        shape_edge_ranges=[(19.0, 19.2)] if edge else None,
        shape_edge_weight=5.0,
        training_data_sha256=checkpoint.get('training_data_sha256'),
    )
    print(f'Completed {destination}', flush=True)


def main():
    # ``--all`` trains every bundled checkpoint with the tuned recipe; existing
    # candidates are loaded rather than retrained, so runs can be resumed.
    if '--all' in sys.argv:
        for source in sorted(SOURCE.glob('sigma*/*/*.pt')):
            train_candidate(source)
        return
    # Capacity variants for the cold, saturated corner where the tuned recipe
    # still loses logN accuracy.
    variants = {'wide': {}, 'wide_pca50': {'n_pca': 50}}
    for name in [a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--variant=')]:
        for relative in ('sigma2/CH4_single_v2/net_CH4_forward.pt',
                         'sigma2/HCN_single_v2/net_HCN_forward.pt',
                         'sigma2/CO18O_single_v2/net_CO18O_forward.pt',
                         'sigma0/C2H4_single_v2/net_C2H4_forward.pt'):
            train_candidate(SOURCE / relative, variant=name, hidden=(128, 256, 128),
                            n_epochs=10000, patience=1000, **variants[name])
        return
    comparison = pd.read_csv(COMPARISON)
    for relative in comparison.loc[comparison.median_ratio > 1.05, 'checkpoint']:
        train_candidate(SOURCE / relative)
    for sigma in (0, 2):
        source = SOURCE / f'sigma{sigma}/H2O_wavelength_bank_v2_fixed/' \
                 'net_H2O_w19to25_n13to19_t100to1400.pt'
        train_candidate(source, edge=True)


if __name__ == '__main__':
    main()
