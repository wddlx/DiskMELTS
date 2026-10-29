"""Compare bundled and retrained checkpoints on matched held-out grid models."""

import json

import numpy as np
import pandas as pd
import torch

from diskmelts.fitting import generate_spectrum, load_models
from retrain_bundled_checkpoints import ROOT, SOURCE, DEST, training_csv


def grid_file(sigma, mol, temperature, column):
    directory = ROOT / 'Model_grids' / f'sigma{sigma}' / mol
    for value in (str(column), f'{column:g}', f'{column:.1f}', f'{column:.3f}'):
        path = directory / f'T{temperature:g}N{value}.csv'
        if path.is_file():
            return path
    raise FileNotFoundError((sigma, mol, temperature, column))


def main():
    rows = []
    for source in sorted(SOURCE.glob('sigma*/*/*.pt')):
        relative = source.relative_to(SOURCE)
        retrained = DEST / relative
        if not retrained.is_file():
            raise FileNotFoundError(retrained)
        ckpt = torch.load(source, map_location='cpu', weights_only=True)
        mol = ckpt['mol']
        sigma = int(relative.parts[0].removeprefix('sigma'))
        csv = training_csv(sigma, mol, tuple(ckpt['wav_range']))
        holdout = csv.with_name(csv.name.replace('pretrain_', 'holdout_')).with_suffix('.json')
        metadata = json.loads(holdout.read_text())
        keys = metadata.get('keys', metadata.get('holdout_keys'))
        if not keys:
            raise ValueError(f'No held-out keys in {holdout}')
        chosen = np.random.default_rng(42).choice(len(keys), size=min(100, len(keys)), replace=False)
        old = load_models({mol: str(source)})[mol]
        new = load_models({mol: str(retrained)})[mol]
        errors = {'old': [], 'new': []}
        for index in chosen:
            temperature, column = map(float, keys[index])
            frame = pd.read_csv(grid_file(sigma, mol, temperature, column),
                                usecols=['wave', 'Line'])
            selected = frame[(frame.wave >= ckpt['wav_range'][0]) &
                             (frame.wave <= ckpt['wav_range'][1])]
            wave = selected.wave.to_numpy()
            truth = selected.Line.to_numpy()
            peak = np.max(np.abs(truth))
            if not np.isfinite(peak) or peak <= 0:
                continue
            for label, model in (('old', old), ('new', new)):
                predicted = generate_spectrum(temperature, column, 1.0, model, obs_wav=wave)
                errors[label].append(float(np.sqrt(np.mean(((predicted - truth) / peak) ** 2))))
        if not errors['old'] or len(errors['old']) != len(errors['new']):
            raise ValueError(f'No comparable held-out spectra for {relative}')
        row = {'checkpoint': str(relative), 'mol': mol, 'sigma': sigma,
               'n_holdout': len(errors['old']),
               'old_median': float(np.median(errors['old'])),
               'new_median': float(np.median(errors['new'])),
               'old_mean': float(np.mean(errors['old'])),
               'new_mean': float(np.mean(errors['new']))}
        row['median_ratio'] = row['new_median'] / max(row['old_median'], 1e-30)
        rows.append(row)
        print(f"{relative}: {row['old_median']:.5g} -> {row['new_median']:.5g} "
              f"({row['median_ratio']:.2f}x)", flush=True)
    output = DEST / 'holdout_comparison.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
