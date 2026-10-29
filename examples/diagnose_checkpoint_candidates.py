"""Split surrogate error into shape and peak parts and re-test NT retrieval.

Run with ``PYTHONPATH=src:examples python examples/diagnose_checkpoint_candidates.py
[candidate_root ...]``. Each candidate root mirrors the bundled
``sigma*/<set>/<file>.pt`` layout; only checkpoints present there are tested.
For each one, the bundled and candidate models are compared on the same 100
held-out grid spectra (shape RMSE after normalizing both to their own window
peak, and absolute log10 window-peak error), then on 40 further held-out
spectra for T/logN retrieval with 64 starts and 600 Adam steps, and on the
held-out cold (T <= 400 K) saturated (logN >= 18.5) or faint (logN <= 13.5)
corner. ``--only=a,b`` restricts the run to checkpoint paths containing a or b.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diskmelts.fitting import generate_spectrum, load_models
from diskmelts.validation import validate_nt_holdout
from compare_retrained_checkpoints import grid_file
from retrain_bundled_checkpoints import ROOT, SOURCE, training_csv


def held_out_keys(sigma, mol, wav_range):
    csv = training_csv(sigma, mol, wav_range)
    holdout = csv.with_name(csv.name.replace('pretrain_', 'holdout_')).with_suffix('.json')
    metadata = json.loads(holdout.read_text())
    keys = [tuple(map(float, k)) for k in metadata.get('keys', metadata.get('holdout_keys'))]
    order = np.random.default_rng(7).permutation(len(keys))
    # Cold, saturated or very faint models are where peak errors map to the
    # largest logN errors; they are tested separately from the random set.
    corner = [k for k in keys if k[0] <= 400 and (k[1] >= 18.5 or k[1] <= 13.5)]
    return [keys[i] for i in order[:100]], [keys[i] for i in order[100:140]], corner


def main():
    roots = [Path(p) for p in sys.argv[1:] if not p.startswith('--')] \
        or [ROOT / 'Trained_model/tuned_2026_10_28/spectral']
    only = [a.split('=', 1)[1].split(',') for a in sys.argv[1:] if a.startswith('--only=')]
    only = only[0] if only else None
    rows = []
    for root in roots:
        for candidate in sorted(root.glob('sigma*/*/*.pt')):
            relative = candidate.relative_to(root)
            if only and not any(part in str(relative) for part in only):
                continue
            source = SOURCE / relative
            ckpt = torch.load(source, map_location='cpu', weights_only=True)
            mol = ckpt['mol']
            sigma = int(relative.parts[0].removeprefix('sigma'))
            wav_range = tuple(ckpt['wav_range'])
            spectral_keys, retrieval_keys, corner_keys = held_out_keys(sigma, mol, wav_range)
            grids = {}
            for key in set(spectral_keys + retrieval_keys + corner_keys):
                frame = pd.read_csv(grid_file(sigma, mol, *key), usecols=['wave', 'Line'])
                grids[key] = {'wavelength': frame.wave.to_numpy(), 'flux': frame.Line.to_numpy()}
            row = {'root': root.name, 'checkpoint': str(relative)}
            for name, path in (('bundled', source), ('candidate', candidate)):
                model = load_models({mol: str(path)})
                shape_err, peak_err = [], []
                for key in spectral_keys:
                    wave, flux = grids[key]['wavelength'], grids[key]['flux']
                    inside = (wave >= wav_range[0]) & (wave <= wav_range[1])
                    wave, truth = wave[inside], flux[inside]
                    peak = np.max(np.abs(truth))
                    if not np.isfinite(peak) or peak <= 0:
                        continue
                    predicted = generate_spectrum(*key, 1.0, model[mol], obs_wav=wave)
                    predicted_peak = max(np.max(np.abs(predicted)), 1e-300)
                    shape_err.append(np.sqrt(np.mean((predicted / predicted_peak - truth / peak) ** 2)))
                    peak_err.append(abs(np.log10(predicted_peak / peak)))
                errors = {}
                for label, keys in (('', retrieval_keys), ('corner_', corner_keys)):
                    result = validate_nt_holdout(
                        mol, model, grids, keys, wav_range,
                        T_bounds=tuple(ckpt['T_range']), logN_bounds=tuple(ckpt['logN_range']),
                        n_starts=64, n_steps=600, seed=11,
                    )
                    errors[label] = (np.abs(result['T_pred'] - result['T_true']),
                                     np.abs(result['logN_pred'] - result['logN_true']))
                dT, dN = errors['']
                cT, cN = errors['corner_']
                row.update({
                    f'{name}_shape_med': float(np.median(shape_err)),
                    f'{name}_peak_dex_med': float(np.median(peak_err)),
                    f'{name}_peak_dex_p90': float(np.percentile(peak_err, 90)),
                    f'{name}_T_med': float(np.median(dT)),
                    f'{name}_T_rmse': float(np.sqrt(np.mean(dT ** 2))),
                    f'{name}_logN_med': float(np.median(dN)),
                    f'{name}_logN_rmse': float(np.sqrt(np.mean(dN ** 2))),
                    f'{name}_corner_T_rmse': float(np.sqrt(np.mean(cT ** 2))),
                    f'{name}_corner_logN_rmse': float(np.sqrt(np.mean(cN ** 2))),
                    f'{name}_corner_logN_max': float(np.max(cN)),
                })
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    output = roots[0].parent / f'diagnostics_{"_".join(r.name for r in roots)}.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
