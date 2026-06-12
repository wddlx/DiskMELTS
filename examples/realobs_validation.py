"""
examples/realobs_validation.py
Multi-source validation: fits USco (U3034) and MINDS sources, then produces
parameter-comparison scatter plots against literature values.

Fitting routine per source (stages driven by Parameter_Comparison.csv):
  1.  H2O             — 1- or 2-component depending on literature presence
  2.  13C12CH2 + C2H2 + HCN  — C2H2/HCN included only if in literature ref
  3.  13CO2  + CO2            — only if CO2 present in literature ref

Saved outputs:
  - All stage model spectra / fit params → OUTPUT_DIR  (realobs_results/)
  - C2H2 / HCN / CO2 best-fit values     → FITTED_CSV  (Realobs_data/Fitted_U3034.csv)
  - Four comparison figures (H2O, C2H2, HCN, CO2) → FIG_DIR

Edit the CONFIGURATION block, then run from the repository root:
    conda run -n data_reduction python examples/realobs_validation.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from diskmelts.fitting import (
    load_models, load_observed_spectrum, fit_molecules,
    save_fit_outputs, detect_stage_molecules,
    save_running_spectrum, print_fit_params, save_fitted_comparison,
)
from diskmelts.plotting import plot_fit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===========================================================================
# CONFIGURATION
# ===========================================================================

PARAM_CSV  = os.path.normpath(os.path.join(
    BASE_DIR, '..', 'Molecular_Luminosity_estimation',
    'Package', 'Realobs_data', 'Parameter_Comparison.csv',
))
FITTED_CSV = os.path.join(BASE_DIR, 'Realobs_data', 'Fitted_U3034.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'realobs_results')
FIG_DIR    = os.path.join(BASE_DIR, 'figures', 'realobs_validation')

# Global fit settings
N_SAMPLES     = 20000
N_REFINE      = 32
N_TOP         = 20
SAMPLE_METHOD = 'sobol'
SEED          = 42

# Detection screening
DETECTION_SCREENING    = True
DETECTION_SIGMA_FACTOR = 3.0

# Input file format
SKIPROWS  = 1
DELIMITER = ','
WAV_COL   = 0
FLUX_COL  = 1

# Line-free windows for noise σ estimation (µm)
LINE_FREE_WINDOWS = [(11.45, 11.55), (13.10, 13.20), (15.65, 15.72), (15.90, 15.91)]

# ---------------------------------------------------------------------------
# Sources  — spectrum files from Molecular_Luminosity_estimation/Package/Realobs_data
# ---------------------------------------------------------------------------
_DIR_DATA = os.path.normpath(os.path.join(
    BASE_DIR, '..', 'Molecular_Luminosity_estimation',
    'Package', 'Realobs_data', 'Consub_data',
))

SOURCES = [
    {
        'name':  'J1605-2023',
        'label': 'USco2',
        'file':  os.path.join(_DIR_DATA, 'j1605-2023_contsub_v9.0_gpnoise.csv'),
    },
    {
        'name':  'J1614-2332',
        'label': 'USco5',
        'file':  os.path.join(_DIR_DATA, 'j1614-2332_contsub_v9.0_gpnoise.csv'),
    },
    {
        'name':  'J1622-2511',
        'label': 'USco8',
        'file':  os.path.join(_DIR_DATA, 'j1622-2511_contsub_v9.0_gpnoise.csv'),
    },
    {
        'name':  'J1608-1930',
        'label': 'USco9',
        'file':  os.path.join(_DIR_DATA, 'j1608-1930_contsub_v9.0_gpnoise.csv'),
    },
    {
        'name':  'J1609-1908',
        'label': 'USco10',
        'file':  os.path.join(_DIR_DATA, 'j1609-1908_contsub_v9.0_gpnoise.csv'),
    },
    # MINDS spectrum not yet available; will be skipped automatically
    # {
    #     'name':  'J04381486+2611399',
    #     'label': 'MINDS',
    #     'file':  os.path.join(_DIR_DATA, 'j04381486+2611399_contsub.csv'),
    # },
]

# ---------------------------------------------------------------------------
# Pretrained models
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    'H2O':      os.path.join(BASE_DIR, 'Trained_model', 'net_H2O_forward_11to19.pt'),
    '13C12CH2': os.path.join(BASE_DIR, 'Trained_model', 'net_13C12CH2_forward_11to19.pt'),
    'C2H2':     os.path.join(BASE_DIR, 'Trained_model', 'net_C2H2_forward_11to19.pt'),
    'HCN':      os.path.join(BASE_DIR, 'Trained_model', 'net_HCN_forward_11to19.pt'),
    '13CO2':    os.path.join(BASE_DIR, 'Trained_model', 'net_13CO2_forward_11to19.pt'),
    'CO2':      os.path.join(BASE_DIR, 'Trained_model', 'net_CO2_forward_11to19.pt'),
}
# ===========================================================================
# Helper functions
# ===========================================================================

def _has_value(val):
    """Return True if val is a non-empty, finite numeric value."""
    try:
        return np.isfinite(float(val))
    except (ValueError, TypeError):
        return False


def _safe_float(val):
    """Convert val to float; return NaN on failure or non-finite."""
    try:
        f = float(val)
        return f if np.isfinite(f) else np.nan
    except (ValueError, TypeError):
        return np.nan


def _build_stages(ref_row):
    """
    Build fitting stage list for one source from its Parameter_Comparison row.

    H2O: 2-component if both warm and hot N are present; 1-component (warm
    only) if just warm; skipped if neither.
    C-mol stage: only included when at least one of C2H2/HCN is in ref;
    13C12CH2 is always prepended.
    CO2 stage: only when CO2_N is in ref; 13CO2 always prepended.
    """
    has_h2o_warm = _has_value(ref_row.get('H2O_N_warm'))
    has_h2o_hot  = _has_value(ref_row.get('H2O_N_hot'))
    has_c2h2     = _has_value(ref_row.get('C2H2_N'))
    has_hcn      = _has_value(ref_row.get('HCN_N'))
    has_co2      = _has_value(ref_row.get('CO2_N'))

    stages = []

    # --- Stage 1 : H2O ---
    if has_h2o_warm or has_h2o_hot:
        n_comp = 2 if (has_h2o_warm and has_h2o_hot) else 1
        stages.append({
            'name':            'H2O',
            'mol':             'H2O',
            'fit_ranges':      [(11.0, 12.0), (16.5, 18.5)],
            'h2o_components':  n_comp,
            'component_names': ['H2O_warm', 'H2O_hot'] if n_comp == 2 else ['H2O_warm'],
            'T_bounds':        (200.0, 1300.0),
            'logN_bounds':     (14.0,  19.0),
            'loga_bounds':     (-2.0,  2.0),
            'detect_peaks':    [(17.19, 17.25), (17.31, 17.33), (17.49, 17.51)],
        })

    # --- Stage 2 : 13C12CH2 + C2H2 + HCN ---
    if has_c2h2 or has_hcn:
        mol_list = ['13C12CH2']
        detect   = {}
        if has_c2h2:
            mol_list.append('C2H2')
            detect['C2H2'] = (13.705, 13.715)
        if has_hcn:
            mol_list.append('HCN')
            detect['HCN'] = (13.99, 14.05)
        # 13C12CH2 not in detect dict → always kept by detect_stage_molecules
        stages.append({
            'name':         '13CCH2_C2H2_HCN',
            'mol':          mol_list,
            'fit_ranges':   (12.0, 16.5),
            'T_bounds':     (200.0, 1300.0),
            'logN_bounds':  (14.0,  19.0),
            'loga_bounds':  (-2.0,  2.0),
            'detect_peaks': detect,
        })

    # --- Stage 3 : 13CO2 + CO2 ---
    if has_co2:
        stages.append({
            'name':         '13CO2_CO2',
            'mol':          ['13CO2', 'CO2'],
            'fit_ranges':   (12.0, 16.5),
            'T_bounds':     (200.0, 1300.0),
            'logN_bounds':  (14.0,  19.0),
            'loga_bounds':  (-2.0,  2.0),
            'detect_peaks': {'CO2': (14.935, 14.985)},
        })

    return stages


def _fit_to_data(params, uncertainty, comp_key):
    """
    Extract {N, N_lo, N_hi, T, T_lo, T_hi, A, A_lo, A_hi} from fit_molecules
    result for component *comp_key*.  Returns None if the component is absent.
    """
    if comp_key not in params:
        return None
    p = params[comp_key]
    u = uncertainty.get(comp_key, {})
    return {
        'N':    p['logN'],
        'N_lo': u.get('logN',   {}).get('minus', np.nan),
        'N_hi': u.get('logN',   {}).get('plus',  np.nan),
        'T':    p['T'],
        'T_lo': u.get('T',      {}).get('minus', np.nan),
        'T_hi': u.get('T',      {}).get('plus',  np.nan),
        'A':    p['log10A'],
        'A_lo': u.get('log10A', {}).get('minus', np.nan),
        'A_hi': u.get('log10A', {}).get('plus',  np.nan),
    }


def _collect_ref(df, n_col, t_col, a_col):
    """
    Build {src: data_dict} from a DataFrame indexed by Source.
    Only rows where N, T and A are all finite are included.
    Error columns are inferred as n_col+'_err_low' etc.
    """
    out = {}
    for src, row in df.iterrows():
        N = _safe_float(row.get(n_col))
        T = _safe_float(row.get(t_col))
        A = _safe_float(row.get(a_col))
        if not (np.isfinite(N) and np.isfinite(T) and np.isfinite(A)):
            continue
        out[src] = {
            'N':    N,
            'N_lo': _safe_float(row.get(f'{n_col}_err_low')),
            'N_hi': _safe_float(row.get(f'{n_col}_err_high')),
            'T':    T,
            'T_lo': _safe_float(row.get(f'{t_col}_err_low')),
            'T_hi': _safe_float(row.get(f'{t_col}_err_high')),
            'A':    A,
            'A_lo': _safe_float(row.get(f'{a_col}_err_low')),
            'A_hi': _safe_float(row.get(f'{a_col}_err_high')),
        }
    return out


def _logNA(d):
    """Return (logNA, logNA_lo, logNA_hi) from a data dict."""
    NA    = d['N'] + d['A']
    NA_lo = (np.sqrt(d['N_lo']**2 + d['A_lo']**2)
             if (np.isfinite(d['N_lo']) and np.isfinite(d['A_lo'])) else np.nan)
    NA_hi = (np.sqrt(d['N_hi']**2 + d['A_hi']**2)
             if (np.isfinite(d['N_hi']) and np.isfinite(d['A_hi'])) else np.nan)
    return NA, NA_lo, NA_hi


def _draw_comparison_fig(save_path, fig_title, data_layers):
    """
    Create a 4-panel scatter comparison figure (T | logN | logA | logNA).

    Parameters
    ----------
    save_path : str
    fig_title : str
    data_layers : list of (ref_dict, fit_dict, color, marker, legend_label)
        ref_dict / fit_dict : {src_name: data_dict}
        data_dict keys : N, N_lo, N_hi, T, T_lo, T_hi, A, A_lo, A_hi
    """
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    ax_T, ax_N, ax_A, ax_NA = axes

    panel_axes  = [ax_T, ax_N, ax_A, ax_NA]
    panel_keys  = ['T',    'N',    'A',    'NA']
    panel_titles = [
        'Temperature (K)',
        r'$\log_{10}N$ (cm$^{-2}$)',
        r'$\log_{10}A$ (au$^2$)',
        r'$\log_{10}(N \cdot A)$',
    ]
    panel_xlabels = [
        'T (K)',
        r'$\log_{10}N$',
        r'$\log_{10}A$',
        r'$\log_{10}(N \cdot A)$',
    ]

    all_vals = {k: [] for k in panel_keys}

    for ref_dict, fit_dict, color, marker, leg_label in data_layers:
        first_on = {k: True for k in panel_keys}  # label only on first plotted point

        for src in list(ref_dict.keys()):
            if src not in fit_dict:
                continue
            r = ref_dict[src]
            f = fit_dict[src]

            r_NA, r_NA_lo, r_NA_hi = _logNA(r)
            f_NA, f_NA_lo, f_NA_hi = _logNA(f)

            vals_r = {'T': r['T'],  'N': r['N'],  'A': r['A'],  'NA': r_NA}
            vals_f = {'T': f['T'],  'N': f['N'],  'A': f['A'],  'NA': f_NA}
            errs_r = {
                'T':  (r['T_lo'],  r['T_hi']),
                'N':  (r['N_lo'],  r['N_hi']),
                'A':  (r['A_lo'],  r['A_hi']),
                'NA': (r_NA_lo,    r_NA_hi),
            }
            errs_f = {
                'T':  (f['T_lo'],  f['T_hi']),
                'N':  (f['N_lo'],  f['N_hi']),
                'A':  (f['A_lo'],  f['A_hi']),
                'NA': (f_NA_lo,    f_NA_hi),
            }

            for ax, key in zip(panel_axes, panel_keys):
                rv, fv = vals_r[key], vals_f[key]
                if not (np.isfinite(rv) and np.isfinite(fv)):
                    continue
                r_lo, r_hi = errs_r[key]
                f_lo, f_hi = errs_f[key]
                lbl = leg_label if first_on[key] else ''
                first_on[key] = False
                ax.errorbar(
                    rv, fv,
                    xerr=[[r_lo if np.isfinite(r_lo) else 0],
                           [r_hi if np.isfinite(r_hi) else 0]],
                    yerr=[[f_lo if np.isfinite(f_lo) else 0],
                           [f_hi if np.isfinite(f_hi) else 0]],
                    fmt=marker, color=color, ms=6, alpha=0.8,
                    elinewidth=0.8, capsize=2, label=lbl,
                )
                all_vals[key].extend([rv, fv])

    for ax, key, title, xlabel in zip(panel_axes, panel_keys, panel_titles, panel_xlabels):
        vals = [v for v in all_vals[key] if np.isfinite(v)]
        if vals:
            lo, hi = min(vals), max(vals)
            m = max((hi - lo) * 0.12, 0.5 if key != 'T' else 20)
            ax.plot([lo - m, hi + m], [lo - m, hi + m], 'k--', lw=0.8, zorder=0)
            ax.set_xlim(lo - m, hi + m)
            ax.set_ylim(lo - m, hi + m)
        ax.set_xlabel(f'Ref. {xlabel}', fontsize=9)
        ax.set_ylabel(f'DiskMELTS {xlabel}', fontsize=9)
        ax.set_title(title, fontsize=10)

    # Deduplicated legend on first axis
    seen, handles, labels = set(), [], []
    for ax in panel_axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l and l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    if handles:
        panel_axes[0].legend(handles, labels, fontsize=8, loc='upper left')

    fig.suptitle(fig_title, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved comparison figure: {save_path}')


# ===========================================================================
# Setup
# ===========================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

pretrained = load_models(
    model_paths=MODEL_PATHS,
)

# Load reference CSV (row 1 is the units row — skip it)
ref_df = pd.read_csv(PARAM_CSV, skiprows=[1], dtype=str)
ref_df['Source'] = ref_df['Source'].astype(str).str.strip()
ref_df = ref_df[ref_df['distance(pc)'].apply(_has_value)].copy()
ref_df = ref_df.set_index('Source')

# ===========================================================================
# Fitting loop
# ===========================================================================

# Collect H2O fit results in memory for later comparison plot
h2o_collected = {}   # src_name → {'params': ..., 'uncertainty': ...}

for src in SOURCES:
    src_name  = src['name']
    src_label = src['label']
    spec_file = src['file']

    print(f'\n{"=" * 70}')
    print(f'Source: {src_label}  ({src_name})')
    print(f'{"=" * 70}')

    if not os.path.isfile(spec_file):
        print(f'  Spectrum file not found — skipping:\n  {spec_file}')
        continue

    if src_name not in ref_df.index:
        print(f'  No reference row in Parameter_Comparison.csv — skipping.')
        continue

    ref_row = ref_df.loc[src_name].to_dict()
    stages  = _build_stages(ref_row)
    if not stages:
        print(f'  No fitting stages derived from reference — skipping.')
        continue

    obs_wav, obs_flux = load_observed_spectrum(
        spec_file, skiprows=SKIPROWS, delimiter=DELIMITER,
        wav_col=WAV_COL, flux_col=FLUX_COL,
    )

    # Noise σ from line-free windows
    sigma_noise = None
    lf_segs = [
        obs_flux[(obs_wav >= lo) & (obs_wav <= hi)]
        for lo, hi in LINE_FREE_WINDOWS
        if np.any((obs_wav >= lo) & (obs_wav <= hi))
    ]
    if lf_segs:
        lf_arr = np.concatenate(lf_segs)
        if len(lf_arr) > 0:
            sigma_noise = float(np.nanstd(lf_arr))
            print(f'  σ_noise = {sigma_noise:.5f} Jy  ({len(lf_arr)} line-free pixels)')

    residual         = obs_flux.copy()
    cumulative_model = np.zeros_like(obs_flux)
    all_fits         = {}

    for stage in stages:
        stage_name = stage['name']
        print(f'\n  --- Stage: {stage_name} ---')

        detected_mols = detect_stage_molecules(
            obs_wav, residual, stage, sigma_noise,
            screening=DETECTION_SCREENING,
            sigma_factor=DETECTION_SIGMA_FACTOR,
        )
        if not detected_mols:
            print(f'    Stage {stage_name!r} skipped (no detections).')
            continue

        mol_to_fit = detected_mols[0] if len(detected_mols) == 1 else detected_mols

        fit = fit_molecules(
            obs_wav=obs_wav,
            obs_flux=residual,
            mol=mol_to_fit,
            pretrained=pretrained,
            fit_ranges=stage['fit_ranges'],
            h2o_components=stage.get('h2o_components', 1),
            component_names=stage.get('component_names'),
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

        residual         -= fit['model_flux']
        cumulative_model += fit['model_flux']

        stage_prefix = f'{src_name}_{stage_name}'
        saved        = save_fit_outputs(fit, OUTPUT_DIR, stage_prefix)
        running_path = save_running_spectrum(
            obs_wav, residual, cumulative_model, OUTPUT_DIR, stage_prefix)

        print('    Saved:')
        for k, p in saved.items():
            print(f'      {k:8s}: {p}')
        print(f'      running : {running_path}')

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

        print(f'    Best-fit ({stage_name}):')
        print_fit_params(fit)

        all_fits[stage_name] = fit

    if not all_fits:
        continue

    # Combined plot (all components on original spectrum)
    save_running_spectrum(obs_wav, residual, cumulative_model,
                          OUTPUT_DIR, f'{src_name}_final')
    combined_params = {}
    combined_unc    = {}
    for fit in all_fits.values():
        combined_params.update(fit['params'])
        combined_unc.update(fit.get('uncertainty', {}))

    plot_fit(
        obs_wav=obs_wav,
        obs_flux=obs_flux,
        results=list(all_fits.values()),
        sigma=sigma_noise,
        fit_ranges=None,
        wav_plot_range=(11.0, 19.0),
        name=src_name,
        save_path=os.path.join(FIG_DIR, f'{src_name}_combined_fit.png'),
        params=combined_params,
        uncertainty=combined_unc,
    )

    # Store H2O results in memory for comparison plot
    if 'H2O' in all_fits:
        h2o_collected[src_name] = {
            'params':      all_fits['H2O']['params'],
            'uncertainty': all_fits['H2O'].get('uncertainty', {}),
        }

    # Save only C-mol parameters to the comparison CSV
    cmol_fits = {
        k: v for k, v in all_fits.items()
        if k in ('13CCH2_C2H2_HCN', '13CO2_CO2')
    }
    if cmol_fits:
        save_fitted_comparison(FITTED_CSV, src_name, cmol_fits)

print('\nAll sources processed.')

# ===========================================================================
# Comparison plots
# ===========================================================================

print('\nBuilding comparison figures ...')

# --- Reference data from Parameter_Comparison.csv ---
ref_h2o_warm = _collect_ref(ref_df, 'H2O_N_warm', 'H2O_T_warm', 'H2O_A_warm')
ref_h2o_hot  = _collect_ref(ref_df, 'H2O_N_hot',  'H2O_T_hot',  'H2O_A_hot')
ref_c2h2     = _collect_ref(ref_df, 'C2H2_N',     'C2H2_T',     'C2H2_A')
ref_hcn      = _collect_ref(ref_df, 'HCN_N',      'HCN_T',      'HCN_A')
ref_co2      = _collect_ref(ref_df, 'CO2_N',      'CO2_T',      'CO2_A')

# --- Fitted H2O data from in-memory collection ---
fitted_h2o_warm = {}
fitted_h2o_hot  = {}
for sname, hd in h2o_collected.items():
    d_warm = _fit_to_data(hd['params'], hd['uncertainty'], 'H2O_warm')
    d_hot  = _fit_to_data(hd['params'], hd['uncertainty'], 'H2O_hot')
    if d_warm is not None:
        fitted_h2o_warm[sname] = d_warm
    if d_hot is not None:
        fitted_h2o_hot[sname] = d_hot

# --- Fitted C-mol data from FITTED_CSV ---
fitted_c2h2 = {}
fitted_hcn  = {}
fitted_co2  = {}
if os.path.exists(FITTED_CSV):
    fitted_df = pd.read_csv(FITTED_CSV, dtype=str)
    fitted_df['Source'] = fitted_df['Source'].astype(str).str.strip()
    fitted_df = fitted_df.set_index('Source')
    fitted_c2h2 = _collect_ref(fitted_df, 'C2H2_N', 'C2H2_T', 'C2H2_A')
    fitted_hcn  = _collect_ref(fitted_df, 'HCN_N',  'HCN_T',  'HCN_A')
    fitted_co2  = _collect_ref(fitted_df, 'CO2_N',  'CO2_T',  'CO2_A')
else:
    print(f'  FITTED_CSV not found ({FITTED_CSV}) — comparison plots will be empty.')

# --- H2O comparison figure ---
_draw_comparison_fig(
    save_path=os.path.join(FIG_DIR, 'comparison_H2O.png'),
    fig_title=r'H$_2$O parameter comparison',
    data_layers=[
        (ref_h2o_warm, fitted_h2o_warm, 'C9',  'o', r'warm H$_2$O'),
        (ref_h2o_hot,  fitted_h2o_hot,  'blue', '^', r'hot H$_2$O'),
    ],
)

# --- C2H2 comparison figure ---
_draw_comparison_fig(
    save_path=os.path.join(FIG_DIR, 'comparison_C2H2.png'),
    fig_title=r'C$_2$H$_2$ parameter comparison',
    data_layers=[
        (ref_c2h2, fitted_c2h2, 'C3', 'o', r'C$_2$H$_2$'),
    ],
)

# --- HCN comparison figure ---
_draw_comparison_fig(
    save_path=os.path.join(FIG_DIR, 'comparison_HCN.png'),
    fig_title='HCN parameter comparison',
    data_layers=[
        (ref_hcn, fitted_hcn, 'C4', 'o', 'HCN'),
    ],
)

# --- CO2 comparison figure ---
_draw_comparison_fig(
    save_path=os.path.join(FIG_DIR, 'comparison_CO2.png'),
    fig_title=r'CO$_2$ parameter comparison',
    data_layers=[
        (ref_co2, fitted_co2, 'C2', 'o', r'CO$_2$'),
    ],
)

print('\nDone.')
