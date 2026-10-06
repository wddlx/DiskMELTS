"""Fit J16120505, then read the saved results and plot them with the data.

Run from the repository root with the ``data_reduction`` environment::

    conda run -n data_reduction python examples/dev_v2_realobs.py

Edit the source and stage list below for another observation. Model paths,
trained domains, PCA sizes, and network layers are in
``diskmelts.v2_fitting.MODEL_SPECS``.
"""

import os
from pathlib import Path

from diskmelts import fit_observation, plot_saved_observation


ROOT = Path(__file__).resolve().parents[1]
H2O_COMPONENTS = int(os.environ.get('DISKMELTS_H2O_COMPONENTS', '2'))
INPUT_PATH = ROOT / 'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv'
RESULT_DIR = ROOT / 'realobs_results/v2/J16120505'
FIGURE_DIR = ROOT / 'figures/realobs/v2'
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

fit = fit_observation(
    input_path=INPUT_PATH,
    name='J16120505',
    stages=STAGES,
    distance_pc=122.5,
    output_dir=RESULT_DIR,
    comparison_csv=ROOT / 'realobs_results/v2/Fitted_Parameters.csv',
)
plot_saved_observation(
    INPUT_PATH, RESULT_DIR, STAGES, name='J16120505',
    figure_dir=FIGURE_DIR, active_stages=fit['fits'],
    sigma_noise=fit['sigma_noise'],
)
