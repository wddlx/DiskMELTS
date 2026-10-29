"""Evaluate tuned candidates on held-out spectra outside the first audit sample."""

import json

import numpy as np
import pandas as pd
import torch

from diskmelts.fitting import generate_spectrum, load_models
from compare_retrained_checkpoints import grid_file
from retrain_bundled_checkpoints import SOURCE, DEST as RETRAINED, training_csv
from tune_retrained_checkpoints import DEST as TUNED


def evaluate(source, candidate, label):
    relative = source.relative_to(SOURCE)
    old_meta = torch.load(source, map_location='cpu', weights_only=True)
    mol = old_meta['mol']
    sigma = int(relative.parts[0].removeprefix('sigma'))
    csv = training_csv(sigma, mol, tuple(old_meta['wav_range']))
    holdout = csv.with_name(csv.name.replace('pretrain_', 'holdout_')).with_suffix('.json')
    metadata = json.loads(holdout.read_text())
    keys = metadata.get('keys', metadata.get('holdout_keys'))
    permutation = np.random.default_rng(42).permutation(len(keys))
    # The first audit selected 100 with a different RNG call. Explicitly
    # exclude those indices so model selection uses fresh holdout models.
    first = set(np.random.default_rng(42).choice(len(keys),
                                                size=min(100, len(keys)), replace=False))
    chosen = [index for index in permutation if index not in first][:100]
    if not chosen:
        raise ValueError(f'No fresh holdouts for {relative}')
    models = {name: load_models({mol: str(path)})[mol] for name, path in (
        ('bundled', source), ('retrained', RETRAINED / relative),
        ('candidate', candidate))}
    errors = {name: [] for name in models}
    seam_errors = {name: [] for name in models}
    for index in chosen:
        temperature, column = map(float, keys[index])
        frame = pd.read_csv(grid_file(sigma, mol, temperature, column),
                            usecols=['wave', 'Line'])
        frame = frame[(frame.wave >= old_meta['wav_range'][0]) &
                      (frame.wave <= old_meta['wav_range'][1])]
        wave, truth = frame.wave.to_numpy(), frame.Line.to_numpy()
        peak = np.max(np.abs(truth))
        if not np.isfinite(peak) or peak <= 0:
            continue
        seam = (wave >= 19.0) & (wave <= 19.2)
        for name, model in models.items():
            prediction = generate_spectrum(temperature, column, 1.0, model,
                                           obs_wav=wave)
            residual = (prediction - truth) / peak
            errors[name].append(float(np.sqrt(np.mean(residual ** 2))))
            if label == 'edge' and np.any(seam):
                seam_errors[name].append(float(np.sqrt(np.mean(residual[seam] ** 2))))
    row = {'checkpoint': str(relative), 'candidate_type': label,
           'n_holdout': len(errors['bundled'])}
    for name in models:
        row[f'{name}_median'] = float(np.median(errors[name]))
        row[f'{name}_mean'] = float(np.mean(errors[name]))
        if seam_errors[name]:
            row[f'{name}_seam_median'] = float(np.median(seam_errors[name]))
    return row


def main():
    rows = []
    for label in ('spectral', 'edge'):
        for candidate in sorted((TUNED / label).glob('sigma*/*/*.pt')):
            source = SOURCE / candidate.relative_to(TUNED / label)
            row = evaluate(source, candidate, label)
            rows.append(row)
            print(f"{row['checkpoint']}: bundled {row['bundled_median']:.5g}, "
                  f"candidate {row['candidate_median']:.5g}", flush=True)
    output = TUNED / 'independent_holdout_comparison.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
