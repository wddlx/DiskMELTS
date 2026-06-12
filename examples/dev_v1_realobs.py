"""
examples/dev_v1_realobs.py
Sequential multi-stage spectral fitting for one real continuum-subtracted spectrum.

Each stage fits one molecule or a group of molecules jointly on the running
residual from the previous stage, then subtracts its model before the next stage.
After every stage the running residual and cumulative model are saved.
A final combined plot shows all components on the original spectrum.

Edit the CONFIGURATION block, then run from the repository root:
    conda run -n data_reduction python examples/dev_v1_realobs.py
"""

import os
import numpy as np

from diskmelts.fitting import (
    load_models, load_observed_spectrum, fit_molecules,
    save_fit_outputs, detect_stage_molecules,
    save_running_spectrum, print_fit_params, save_fitted_comparison,
)
from diskmelts.plotting import plot_fit

# Project root = two levels up from this file (examples/ → repo root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===========================================================================
# CONFIGURATION
# ===========================================================================

INPUT_PATH = os.path.join(
    BASE_DIR, 'Realobs_data', 'Consub_data', 'j16120505_v9.0_contsub_RVcorr.csv'
)
NAME = 'J16120505'   # source name used in filenames and plot titles;
                     # set to None to fall back to the input filename stem

OUTPUT_DIR    = os.path.join(BASE_DIR, 'realobs_results')
OUTPUT_PREFIX = None   # None → resolved from NAME (or filename stem) at runtime
FIG_DIR       = os.path.join(BASE_DIR, 'figures', 'realobs')
FITTED_CSV    = os.path.join(BASE_DIR, 'Realobs_data', 'Fitted_Parameters.csv')

# Global fit settings shared across all stages
N_SAMPLES     = int(os.environ.get('DISKMELTS_N_SAMPLES', '20000'))
N_REFINE      = int(os.environ.get('DISKMELTS_N_REFINE', '32'))
N_TOP         = int(os.environ.get('DISKMELTS_N_TOP', '20'))
SAMPLE_METHOD = 'sobol'
SEED          = 42
H2O_COMPONENTS = int(os.environ.get('DISKMELTS_H2O_COMPONENTS', '2'))

# Input file format
SKIPROWS  = 1
DELIMITER = ','
WAV_COL   = 0
FLUX_COL  = 1

# Line-free windows for noise σ estimation (Jy).  Set to [] to skip.
LINE_FREE_WINDOWS = [(11.45, 11.55), (13.10, 13.20), (15.65, 15.72), (15.90, 15.91)]

# Detection screening: set True to skip molecules whose peak is below threshold
DETECTION_SCREENING    = True
DETECTION_SIGMA_FACTOR = float(
    os.environ.get('DISKMELTS_DETECTION_SIGMA_FACTOR', '3.0')
)

# ---------------------------------------------------------------------------
# STAGES — executed in order; each stage fits on the running residual.
#
# Required keys:
#   name          str          — used in output filenames
#   mol           str|list     — molecule(s) fitted jointly in this stage
#   fit_ranges    (lo,hi)|list — wavelength fitting window(s) in µm
#
# Optional keys (fall back to defaults if omitted):
#   h2o_components  int         — 2 for two-component H2O (default 1)
#   component_names list|None   — custom labels, e.g. ['H2O_warm', 'H2O_hot']
#   T_bounds        (min, max)  — temperature search range in K (default 100–1400)
#   logN_bounds     (min, max)  — log10 column density range (default 13–19)
#   loga_bounds     (min, max)  — log10 A range (default -2–2)
#   detect_peaks:
#     • list of (lo, hi) — for a single molecule: detected if ANY window > threshold
#     • dict {mol: (lo, hi)} — for a multi-mol stage: each molecule checked
#                               independently; only detected ones are fitted
#     • omit or None — always fit this stage regardless of screening flag
# ---------------------------------------------------------------------------

STAGES = [
    {
        'name':            'H2O',
        'mol':             'H2O',
        'fit_ranges':      [(11.0, 12.0), (16.5, 18.5)],
        'h2o_components':  H2O_COMPONENTS,
        'component_names': (
            ['H2O_warm', 'H2O_hot'] if H2O_COMPONENTS == 2 else None
        ),
        'T_bounds':        (200.0, 1300.0),
        'logN_bounds':     (14.0,  19.0),
        'loga_bounds':     (-2.0,  2.0),
        'detect_peaks':    [(17.19, 17.25), (17.31, 17.33), (17.49, 17.51)],
    },
    # Uncomment to add C-molecule stages:
    # {
    #     'name':         'C2H2_HCN',
    #     'mol':          ['C2H2', 'HCN'],
    #     'fit_ranges':   (12.0, 16.5),
    #     'T_bounds':     (200.0, 1300.0),
    #     'logN_bounds':  (14.0,  19.0),
    #     'loga_bounds':  (-2.0,  2.0),
    #     'detect_peaks': {
    #         'C2H2': (13.705, 13.715),
    #         'HCN':  (13.99,  14.05),
    #     },
    # },
    # {
    #     'name':         'CO2',
    #     'mol':          'CO2',
    #     'fit_ranges':   (12.0, 16.5),
    #     'T_bounds':     (200.0, 1300.0),
    #     'logN_bounds':  (14.0,  19.0),
    #     'loga_bounds':  (-2.0,  2.0),
    #     'detect_peaks': [(14.935, 14.985)],
    # },
]

# ---------------------------------------------------------------------------
# Pretrained model paths — one entry per molecule appearing in STAGES
# ---------------------------------------------------------------------------

MODEL_PATHS = {
    'H2O':  os.path.join(BASE_DIR, 'Trained_model', 'net_H2O_forward_11to19.pt'),
    'C2H2': os.path.join(BASE_DIR, 'Trained_model', 'net_C2H2_forward_11to19.pt'),
    'HCN':  os.path.join(BASE_DIR, 'Trained_model', 'net_HCN_forward_11to19.pt'),
    'CO2':  os.path.join(BASE_DIR, 'Trained_model', 'net_CO2_forward_11to19.pt'),
}
# ===========================================================================


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

required_paths = [INPUT_PATH, *MODEL_PATHS.values()]
missing_paths = [path for path in required_paths if not os.path.isfile(path)]
if missing_paths:
    missing = '\n'.join(f'  - {path}' for path in missing_paths)
    raise FileNotFoundError(
        'Required fitting files are missing. A fresh clone should include the '
        f'example spectrum and pretrained checkpoints:\n{missing}'
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

pretrained = load_models(
    model_paths=MODEL_PATHS,
)

obs_wav, obs_flux = load_observed_spectrum(
    INPUT_PATH,
    skiprows=SKIPROWS,
    delimiter=DELIMITER,
    wav_col=WAV_COL,
    flux_col=FLUX_COL,
)

if OUTPUT_PREFIX is None:
    OUTPUT_PREFIX = NAME if NAME else os.path.splitext(os.path.basename(INPUT_PATH))[0]

# Estimate noise σ from line-free windows
sigma_noise = None
if LINE_FREE_WINDOWS:
    _lf = np.concatenate([
        obs_flux[(obs_wav >= lo) & (obs_wav <= hi)]
        for lo, hi in LINE_FREE_WINDOWS
        if np.any((obs_wav >= lo) & (obs_wav <= hi))
    ])
    if len(_lf) > 0:
        sigma_noise = float(np.nanstd(_lf))
        print(f'Estimated σ = {sigma_noise:.5f} Jy  ({len(_lf)} line-free pixels)')

# ---------------------------------------------------------------------------
# Sequential stage fitting
# ---------------------------------------------------------------------------

residual         = obs_flux.copy()
cumulative_model = np.zeros_like(obs_flux)
all_fits         = {}   # stage name → fit result dict

for stage in STAGES:
    if not isinstance(stage, dict):
        continue
    stage_name = stage['name']
    print(f'\n{"=" * 60}')
    print(f'Stage: {stage_name}')
    print(f'{"=" * 60}')

    detected_mols = detect_stage_molecules(
        obs_wav, residual, stage, sigma_noise,
        screening=DETECTION_SCREENING,
        sigma_factor=DETECTION_SIGMA_FACTOR,
    )
    if not detected_mols:
        print(f'  Stage {stage_name!r} skipped.')
        continue

    mol_to_fit = detected_mols[0] if len(detected_mols) == 1 else detected_mols

    fit = fit_molecules(
        obs_wav=obs_wav,
        obs_flux=residual,
        mol=mol_to_fit,
        pretrained=pretrained,
        fit_ranges=stage['fit_ranges'],
        h2o_components=stage.get('h2o_components', 1),
        component_names=stage.get('component_names', None),
        T_bounds=stage.get('T_bounds',   (100.0, 1400.0)),
        logN_bounds=stage.get('logN_bounds', (13.0, 19.0)),
        loga_bounds=stage.get('loga_bounds', (-2.0,  2.0)),
        n_samples=N_SAMPLES,
        sample_method=SAMPLE_METHOD,
        n_refine=N_REFINE,
        n_top=N_TOP,
        sigma=sigma_noise,
        seed=SEED,
    )
    fit['obs_wav']  = obs_wav
    fit['obs_flux'] = residual.copy()

    residual         = residual - fit['model_flux']
    cumulative_model = cumulative_model + fit['model_flux']

    stage_prefix = f'{OUTPUT_PREFIX}_{stage_name}'
    saved        = save_fit_outputs(fit, OUTPUT_DIR, stage_prefix)

    running_path = save_running_spectrum(obs_wav, residual, cumulative_model,
                                         OUTPUT_DIR, stage_prefix)

    print('\n  Saved:')
    for k, p in saved.items():
        print(f'    {k:8s}: {p}')
    print(f'    running : {running_path}')

    plot_fit(
        obs_wav=obs_wav,
        obs_flux=fit['obs_flux'],
        results=fit,
        sigma=sigma_noise,
        fit_ranges=stage['fit_ranges'],
        wav_plot_range=(11.0, 19.0),
        name=stage_prefix,
        save_path=os.path.join(FIG_DIR, f'{stage_prefix}_fit.png'),
        params=fit['params'],
        uncertainty=fit.get('uncertainty'),
    )

    print(f'\n  Best-fit parameters ({stage_name}):')
    print_fit_params(fit)

    all_fits[stage_name] = fit

# ---------------------------------------------------------------------------
# Final combined plot — all components overlaid on the original spectrum
# ---------------------------------------------------------------------------

if all_fits:
    final_path = save_running_spectrum(obs_wav, residual, cumulative_model,
                                       OUTPUT_DIR, f'{OUTPUT_PREFIX}_final')
    print(f'\nFinal running spectrum → {final_path}')

    combined_params      = {}
    combined_uncertainty = {}
    for fit in all_fits.values():
        combined_params.update(fit['params'])
        combined_uncertainty.update(fit.get('uncertainty', {}))

    plot_fit(
        obs_wav=obs_wav,
        obs_flux=obs_flux,
        results=list(all_fits.values()),
        sigma=sigma_noise,
        fit_ranges=None,
        wav_plot_range=(11.0, 19.0),
        name=OUTPUT_PREFIX,
        save_path=os.path.join(FIG_DIR, f'{OUTPUT_PREFIX}_combined_fit.png'),
        params=combined_params,
        uncertainty=combined_uncertainty,
    )

    save_fitted_comparison(FITTED_CSV, OUTPUT_PREFIX, all_fits)

print('\nDone.')
