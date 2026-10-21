"""Fit J16120505 or another continuum-subtracted CSV with bundled models.

The default uses the repository's J16120505 spectrum. For a CSV elsewhere::

    python examples/dev_v2_realobs.py --input /path/to/spectrum.csv \
        --name my_source --distance-pc 140 --sigma 2

Adjust the stage windows below to match the observation. This script uses
installed checkpoints and does not require local slab grids.
"""

import argparse
import os
from pathlib import Path

from diskmelts import fit_observation, plot_saved_observation


ROOT = Path(__file__).resolve().parents[1]
H2O_COMPONENTS = int(os.environ.get('DISKMELTS_H2O_COMPONENTS', '2'))
DEFAULT_INPUT = ROOT / 'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv'
STAGES = [
    {
        'name': 'H2O', 'mol': 'H2O', 'wavelength_range': (11.0, 19.0),
        'fit_ranges': [(11.0, 12.0), (16.5, 18.5)],
        'h2o_components': H2O_COMPONENTS,
        'component_names': (['H2O_warm', 'H2O_hot']
                            if H2O_COMPONENTS == 2 else None),
        'detect_peaks': [(17.19, 17.25), (17.31, 17.33), (17.49, 17.51)],
    },
    {
        'name': 'C2H2_HCN', 'mol': ['C2H2', 'HCN'],
        'wavelength_range': (11.5, 17.0), 'fit_ranges': (12.0, 16.5),
        'detect_peaks': {'C2H2': (13.705, 13.715), 'HCN': (13.99, 14.05)},
    },
    {
        'name': 'CO2', 'mol': 'CO2',
        'wavelength_range': (11.5, 17.0), 'fit_ranges': (12.0, 16.5),
        'detect_peaks': [(14.935, 14.985)],
    },
]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT,
                        help='Continuum-subtracted CSV (default: repository J16120505)')
    parser.add_argument('--name', default='J16120505', help='Saved-output prefix')
    parser.add_argument('--distance-pc', type=float, default=122.5)
    parser.add_argument('--sigma', type=int, choices=(0, 2), default=0,
                        help='Checkpoint grid broadening')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--figure-dir', type=Path)
    args = parser.parse_args(argv)
    result_dir = args.output_dir or ROOT / 'realobs_results/v2' / args.name
    figure_dir = args.figure_dir or ROOT / 'figures/realobs/v2'
    fit = fit_observation(
        input_path=args.input, name=args.name, stages=STAGES,
        distance_pc=args.distance_pc, sigma=args.sigma,
        output_dir=result_dir,
        comparison_csv=result_dir / 'Fitted_Parameters.csv',
    )
    return plot_saved_observation(
        args.input, result_dir, STAGES, name=args.name,
        figure_dir=figure_dir, active_stages=fit['fits'],
        sigma_noise=fit['sigma_noise'],
    )


if __name__ == '__main__':
    main()
