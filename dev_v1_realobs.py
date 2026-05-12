"""
dev_v1_realobs.py
Sequential multi-stage spectral fitting for one real continuum-subtracted spectrum.

Each stage fits one molecule or a group of molecules jointly on the running
residual from the previous stage, then subtracts its model before the next stage.
After every stage the running residual and cumulative model are saved.
A final combined plot shows all components on the original spectrum.

Edit the CONFIGURATION block, then run from the repository root:
    conda run -n data_reduction python Package/dev_v1_realobs.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Fitting import (load_models, load_observed_spectrum, fit_molecules,
                          save_fit_outputs, detect_stage_molecules,
                          save_running_spectrum, print_fit_params)
from Plotting       import plot_fit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===========================================================================
# CONFIGURATION
# ===========================================================================

INPUT_PATH = (
    'Realobs_data/Consub_data/j16142029_v9.0_contsub_RVcorr.csv'
)
NAME = 'J16142029'   # source name used in filenames and plot titles;
                     # set to None to fall back to the input filename stem

OUTPUT_DIR    = os.path.join(BASE_DIR, 'realobs_results')
OUTPUT_PREFIX = None   # None → resolved from NAME (or filename stem) at runtime
FIG_DIR       = os.path.join(BASE_DIR, 'figures', 'realobs')
FITTED_CSV    = os.path.join(BASE_DIR, 'Realobs_data', 'Fitted_Parameters.csv')

# Global fit settings shared across all stages
N_SAMPLES     = 20000
N_REFINE      = 32
N_TOP         = 20
SAMPLE_METHOD = 'sobol'
SEED          = 42

# Input file format
SKIPROWS  = 1
DELIMITER = ','
WAV_COL   = 0
FLUX_COL  = 1

# Line-free windows for noise σ estimation (Jy).  Set to [] to skip.
LINE_FREE_WINDOWS = [(11.45, 11.55), (13.10, 13.20), (15.65, 15.72), (15.90, 15.91)]

# Detection screening: set True to skip molecules whose peak is below threshold
DETECTION_SCREENING   = True
DETECTION_SIGMA_FACTOR = 3.0   # threshold = this × σ_noise

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
        'h2o_components':  2,
        'component_names': ['H2O_warm', 'H2O_hot'],
        'T_bounds':        (200.0, 1300.0),
        'logN_bounds':     (14.0,  19.0),
        'loga_bounds':     (-2.0,  2.0),
        'detect_peaks':    [(17.19, 17.25), (17.31, 17.33), (17.49, 17.51)],
    },
    '''
    {
        'name':         'C2H2_HCN',
        'mol':          ['C2H2', 'HCN'],
        'fit_ranges':   (12.0, 16.5),
        'T_bounds':     (200.0, 1300.0),
        'logN_bounds':  (14.0,  19.0),
        'loga_bounds':  (-2.0,  2.0),
        'detect_peaks': {
            'C2H2': (13.705, 13.715),
            'HCN':  (13.99,  14.05),
        },
    },
    {
        'name':         'CO2',
        'mol':          'CO2',
        'fit_ranges':   (12.0, 16.5),
        'T_bounds':     (200.0, 1300.0),
        'logN_bounds':  (14.0,  19.0),
        'loga_bounds':  (-2.0,  2.0),
        'detect_peaks': [(14.935, 14.985)],
    },
    '''
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
PRETRAIN_CSV_PATHS = {
    'H2O':  os.path.join(BASE_DIR, 'Pretrain_grid', 'pretrain_H2O_11to19.csv'),
    'C2H2': os.path.join(BASE_DIR, 'Pretrain_grid', 'pretrain_C2H2_11to19.csv'),
    'HCN':  os.path.join(BASE_DIR, 'Pretrain_grid', 'pretrain_HCN_11to19.csv'),
    'CO2':  os.path.join(BASE_DIR, 'Pretrain_grid', 'pretrain_CO2_11to19.csv'),
}
WAV_RANGES = {
    'H2O':  (11.0, 19.0),
    'C2H2': (11.0, 17.5),
    'HCN':  (11.0, 17.5),
    'CO2':  (11.0, 17.5),
}
N_PCA = {
    'H2O':  25,
    'C2H2': 15,
    'HCN':  15,
    'CO2':  15,
}

# ===========================================================================


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

pretrained = load_models(
    model_paths=MODEL_PATHS,
    pretrain_csv_paths=PRETRAIN_CSV_PATHS,
    wav_ranges=WAV_RANGES,
    n_pca=N_PCA,
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

    # Detection screening — may reduce or empty the molecule list
    detected_mols = detect_stage_molecules(
        obs_wav, residual, stage, sigma_noise,
        screening=DETECTION_SCREENING,
        sigma_factor=DETECTION_SIGMA_FACTOR,
    )
    if not detected_mols:
        print(f'  Stage {stage_name!r} skipped.')
        continue

    # Collapse single-element list back to a string for fit_molecules
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
    # Store the spectrum that was fitted so per-stage plots are self-consistent
    fit['obs_wav']  = obs_wav
    fit['obs_flux'] = residual.copy()

    # Update running state
    residual         = residual - fit['model_flux']
    cumulative_model = cumulative_model + fit['model_flux']

    # Save per-stage fit outputs (spectrum, params, top-list CSVs)
    stage_prefix = f'{OUTPUT_PREFIX}_{stage_name}'
    saved        = save_fit_outputs(fit, OUTPUT_DIR, stage_prefix)

    # Save running residual and cumulative model
    running_path = save_running_spectrum(obs_wav, residual, cumulative_model,
                                         OUTPUT_DIR, stage_prefix)

    print('\n  Saved:')
    for k, p in saved.items():
        print(f'    {k:8s}: {p}')
    print(f'    running : {running_path}')

    # Per-stage plot — shows the residual that was fitted vs the stage model
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

def _save_fitted_comparison(csv_path, source_name, all_fits):
    """
    Upsert one row of best-fit parameters into a comparison CSV.

    The CSV schema matches Parameter_Comparison.csv (log10 units for N and A,
    K for T).  If the file does not exist it is created.  If *source_name* is
    already present its values are updated in-place; otherwise a new row is
    appended.

    Args:
        csv_path (str): path to the comparison CSV.
        source_name (str): identifier for this source (used as the key).
        all_fits (dict): stage-name → fit-result dict from the fitting loop.
    """
    import pandas as pd

    # Merge params and uncertainty from every completed stage
    all_params = {}
    all_unc    = {}
    for fit in all_fits.values():
        all_params.update(fit['params'])
        all_unc.update(fit.get('uncertainty', {}))

    def _v(comp, key):
        """Best-fit value; '' if component was not fitted."""
        if comp not in all_params:
            return ''
        return round(all_params[comp][key], 4)

    def _e(comp, key, which):
        """Uncertainty (minus/plus); '' if absent or non-finite."""
        if comp not in all_unc:
            return ''
        v = all_unc[comp].get(key, {}).get(which, float('nan'))
        return round(v, 4) if np.isfinite(v) else ''

    # Build the flat row matching the CSV column order
    row = {
        'Source':               source_name,
        # ── H2O warm ──────────────────────────────────────────────────────
        'H2O_N_warm':           _v('H2O_warm', 'logN'),
        'H2O_N_warm_err_low':   _e('H2O_warm', 'logN',   'minus'),
        'H2O_N_warm_err_high':  _e('H2O_warm', 'logN',   'plus'),
        'H2O_N_hot':            _v('H2O_hot',  'logN'),
        'H2O_N_hot_err_low':    _e('H2O_hot',  'logN',   'minus'),
        'H2O_N_hot_err_high':   _e('H2O_hot',  'logN',   'plus'),
        'H2O_T_warm':           _v('H2O_warm', 'T'),
        'H2O_T_warm_err_low':   _e('H2O_warm', 'T',      'minus'),
        'H2O_T_warm_err_high':  _e('H2O_warm', 'T',      'plus'),
        'H2O_T_hot':            _v('H2O_hot',  'T'),
        'H2O_T_hot_err_low':    _e('H2O_hot',  'T',      'minus'),
        'H2O_T_hot_err_high':   _e('H2O_hot',  'T',      'plus'),
        'H2O_A_warm':           _v('H2O_warm', 'log10A'),
        'H2O_A_warm_err_low':   _e('H2O_warm', 'log10A', 'minus'),
        'H2O_A_warm_err_high':  _e('H2O_warm', 'log10A', 'plus'),
        'H2O_A_hot':            _v('H2O_hot',  'log10A'),
        'H2O_A_hot_err_low':    _e('H2O_hot',  'log10A', 'minus'),
        'H2O_A_hot_err_high':   _e('H2O_hot',  'log10A', 'plus'),
        # ── C2H2 ──────────────────────────────────────────────────────────
        'C2H2_N':               _v('C2H2', 'logN'),
        'C2H2_N_err_low':       _e('C2H2', 'logN',   'minus'),
        'C2H2_N_err_high':      _e('C2H2', 'logN',   'plus'),
        'C2H2_T':               _v('C2H2', 'T'),
        'C2H2_T_err_low':       _e('C2H2', 'T',      'minus'),
        'C2H2_T_err_high':      _e('C2H2', 'T',      'plus'),
        'C2H2_A':               _v('C2H2', 'log10A'),
        'C2H2_A_err_low':       _e('C2H2', 'log10A', 'minus'),
        'C2H2_A_err_high':      _e('C2H2', 'log10A', 'plus'),
        # ── HCN ───────────────────────────────────────────────────────────
        'HCN_N':                _v('HCN', 'logN'),
        'HCN_N_err_low':        _e('HCN', 'logN',   'minus'),
        'HCN_N_err_high':       _e('HCN', 'logN',   'plus'),
        'HCN_T':                _v('HCN', 'T'),
        'HCN_T_err_low':        _e('HCN', 'T',      'minus'),
        'HCN_T_err_high':       _e('HCN', 'T',      'plus'),
        'HCN_A':                _v('HCN', 'log10A'),
        'HCN_A_err_low':        _e('HCN', 'log10A', 'minus'),
        'HCN_A_err_high':       _e('HCN', 'log10A', 'plus'),
        # ── CO2 ───────────────────────────────────────────────────────────
        'CO2_N':                _v('CO2', 'logN'),
        'CO2_N_err_low':        _e('CO2', 'logN',   'minus'),
        'CO2_N_err_high':       _e('CO2', 'logN',   'plus'),
        'CO2_T':                _v('CO2', 'T'),
        'CO2_T_err_low':        _e('CO2', 'T',      'minus'),
        'CO2_T_err_high':       _e('CO2', 'T',      'plus'),
        'CO2_A':                _v('CO2', 'log10A'),
        'CO2_A_err_low':        _e('CO2', 'log10A', 'minus'),
        'CO2_A_err_high':       _e('CO2', 'log10A', 'plus'),
    }

    if os.path.exists(csv_path):
        df   = pd.read_csv(csv_path)
        mask = df['Source'] == source_name
        if mask.any():
            for col, val in row.items():
                if col in df.columns:
                    df.loc[mask, col] = val
            print(f'  Updated existing row for {source_name!r} in {os.path.basename(csv_path)}')
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            print(f'  Appended new row for {source_name!r} to {os.path.basename(csv_path)}')
    else:
        df = pd.DataFrame([row])
        print(f'  Created {os.path.basename(csv_path)} with first row for {source_name!r}')

    df.to_csv(csv_path, index=False)


if all_fits:
    # Save final running state
    final_path = save_running_spectrum(obs_wav, residual, cumulative_model,
                                       OUTPUT_DIR, f'{OUTPUT_PREFIX}_final')
    print(f'\nFinal running spectrum → {final_path}')

    # Merge params and uncertainty from all stages for the annotation box
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

    # Save / update fitted parameters in the comparison CSV
    _save_fitted_comparison(FITTED_CSV, OUTPUT_PREFIX, all_fits)

print('\nDone.')
