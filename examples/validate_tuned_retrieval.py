"""Compare parameter recovery for bundled and tuned models on unseen grids."""

import json

import numpy as np
import pandas as pd
import torch

from diskmelts.fitting import load_models
from diskmelts.validation import validate_nt_holdout
from compare_retrained_checkpoints import grid_file
from retrain_bundled_checkpoints import SOURCE, training_csv
from tune_retrained_checkpoints import DEST as TUNED


def main():
    rows = []
    for candidate in sorted((TUNED / 'spectral').glob('sigma*/*/*.pt')):
        relative = candidate.relative_to(TUNED / 'spectral')
        source = SOURCE / relative
        ckpt = torch.load(source, map_location='cpu', weights_only=True)
        mol = ckpt['mol']
        sigma = int(relative.parts[0].removeprefix('sigma'))
        csv = training_csv(sigma, mol, tuple(ckpt['wav_range']))
        holdout = csv.with_name(csv.name.replace('pretrain_', 'holdout_')).with_suffix('.json')
        metadata = json.loads(holdout.read_text())
        keys = metadata.get('keys', metadata.get('holdout_keys'))
        first = set(np.random.default_rng(42).choice(len(keys),
                                                    size=min(100, len(keys)), replace=False))
        selected = [tuple(map(float, keys[index]))
                    for index in np.random.default_rng(42).permutation(len(keys))
                    if index not in first][:20]
        models = {}
        for temperature, column in selected:
            frame = pd.read_csv(grid_file(sigma, mol, temperature, column),
                                usecols=['wave', 'Line'])
            models[(temperature, column)] = {
                'wavelength': frame.wave.to_numpy(),
                'flux': frame.Line.to_numpy(),
            }
        row = {'checkpoint': str(relative), 'n_holdout': len(selected)}
        for name, path in (('bundled', source), ('candidate', candidate)):
            loaded = load_models({mol: str(path)})
            result = validate_nt_holdout(
                mol, loaded, models, selected, tuple(ckpt['wav_range']),
                T_bounds=tuple(ckpt['T_range']),
                logN_bounds=tuple(ckpt['logN_range']),
                n_starts=20, n_steps=300, seed=42,
            )
            row[f'{name}_T_rmse'] = float(np.sqrt(np.mean(
                (result['T_pred'] - result['T_true']) ** 2)))
            row[f'{name}_logN_rmse'] = float(np.sqrt(np.mean(
                (result['logN_pred'] - result['logN_true']) ** 2)))
        rows.append(row)
        print(row, flush=True)
    output = TUNED / 'retrieval_comparison.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
