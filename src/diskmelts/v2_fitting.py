"""Reusable v2 fitting workflow for continuum-subtracted observations.

``MODEL_SPECS`` records the grid sigma, trained domain, PCA size, and network layers for
each checkpoint. ``fit_stage`` fits one or several molecules over one or more
wavelength intervals. ``fit_observation`` runs stages on successive residuals
and writes their numerical results for later reading and plotting.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import torch

from diskmelts.uncertainty import default_mcmc_steps
from diskmelts.fitting import (
    _build_components,
    _check_bank_wavelength_coverage,
    _screened_group_options,
    detect_stage_molecules,
    fit_molecules,
    generate_spectrum,
    load_models,
    load_observed_spectrum,
    print_fit_params,
    save_fit_outputs,
    save_fitted_comparison,
    save_running_spectrum,
)


MODEL_ROOT = Path(__file__).resolve().parent / 'models'
WATER_RANGES = ((4.9, 7.4), (7.2, 11.2), (9.9, 15.1),
                (13.5, 19.2), (19.0, 25.0))
CARBON_RANGES = {
    'HCN': (11.5, 17.0), 'C2H2': (11.5, 17.0),
    'CO2': (11.5, 17.0), '13C12CH2': (11.5, 17.0),
    '13CO2': (11.5, 17.0), 'C4H2': (14.0, 18.0),
    'HC3N': (13.0, 18.0), 'C2H6': (9.0, 15.0),
    'C2H4': (8.0, 14.0), 'CH4': (5.0, 9.0),
    'C2HD': (11.5, 20.5), 'CO17O': (13.0, 18.0),
    'CO18O': (13.0, 18.0), 'C6H6': (13.0, 17.0),
    'C3H4': (15.0, 18.0), 'CH3': (15.0, 18.0),
}
MODEL_SPECS = {
    'H2O': {
        'grid_sigma': 0,
        'path': 'Trained_model/sigma0/H2O_wavelength_bank_v2_fixed/H2O_model_bank.json',
        'wavelength_ranges': WATER_RANGES,
        'T_bounds': (100.0, 1400.0),
        'logN_bounds': (13.0, 19.0),
        'n_pca': 30,
        'hidden': (64, 128, 64),
    },
    **{
        mol: {
            'grid_sigma': 0,
            'path': f'Trained_model/sigma0/{mol}_single_v2/net_{mol}_forward.pt',
            'wavelength_ranges': (wav_range,),
            'T_bounds': (100.0, 675.0) if mol == 'C6H6' else (100.0, 1400.0),
            'logN_bounds': (13.0, 19.5),
            'n_pca': 75 if mol in {'C3H4', 'CH3'} else 30,
            'hidden': (64, 128, 64),
        }
        for mol, wav_range in CARBON_RANGES.items() if mol != 'C2H2'
    },
    'C2H2': {
        'grid_sigma': 0,
        'path': 'Trained_model/sigma0/C2H2_wavelength_bank_v2/C2H2_model_bank.json',
        'wavelength_ranges': ((6.0, 9.0), (11.5, 17.0)),
        'T_bounds': (100.0, 1400.0),
        'logN_bounds': (13.0, 19.5),
        'n_pca': 30,
        'hidden': (64, 128, 64),
    },
    'OH': {
        'grid_sigma': 0,
        'path': 'Trained_model/sigma0/OH_single_v2/net_OH_forward.pt',
        'wavelength_ranges': ((15.0, 26.0),),
        'T_bounds': (100.0, 1400.0),
        'logN_bounds': (13.0, 19.5),
        'n_pca': 75,
        'hidden': (64, 128, 64),
    },
}
SIGMA2_MODEL_SPECS = {
    'H2O': {
        **MODEL_SPECS['H2O'],
        'grid_sigma': 2,
        'path': 'Trained_model/sigma2/H2O_wavelength_bank_v2_fixed/H2O_model_bank.json',
    },
    **{
        mol: {
            **MODEL_SPECS[mol],
            'grid_sigma': 2,
            'path': f'Trained_model/sigma2/{mol}_single_v2/net_{mol}_forward.pt',
        }
        for mol in ('HCN', 'CO2', '13C12CH2', '13CO2',
                    'C4H2', 'HC3N', 'C2H6', 'C2H4', 'CH4',
                    'C2HD', 'CO17O', 'CO18O', 'C3H4', 'CH3', 'C6H6')
    },
    'C2H2': {
        **MODEL_SPECS['C2H2'],
        'grid_sigma': 2,
        'path': 'Trained_model/sigma2/C2H2_wavelength_bank_v2/C2H2_model_bank.json',
        'wavelength_ranges': ((6.0, 9.0), (11.5, 17.0)),
    },
    'OH': {
        **MODEL_SPECS['OH'],
        'grid_sigma': 2,
        'path': 'Trained_model/sigma2/OH_single_v2/net_OH_forward.pt',
    },
}


def _model_specs_for_sigma(sigma):
    if isinstance(sigma, bool) or sigma not in (0, 2):
        raise ValueError('sigma must be 0 or 2')
    return MODEL_SPECS if sigma == 0 else SIGMA2_MODEL_SPECS


def _ranges(value):
    """Return validated wavelength intervals from one interval or a sequence."""
    if len(value) == 2 and all(np.isscalar(item) for item in value):
        value = [value]
    ranges = [tuple(map(float, item)) for item in value]
    if not ranges or any(lo >= hi for lo, hi in ranges):
        raise ValueError('wavelength ranges must contain increasing (lo, hi) pairs')
    return ranges


def _fit_ranges(value):
    if isinstance(value, dict):
        return [item for ranges in value.values() for item in _ranges(ranges)]
    return _ranges(value)


def _validate_loaded_model(mol, model, spec):
    """Catch a checkpoint/registry mismatch before fitting observations."""
    if (model.get('grid_sigma') is not None and
            model['grid_sigma'] != spec['grid_sigma']):
        raise ValueError(f'{mol} checkpoint grid_sigma differs from MODEL_SPECS')
    if model.get('is_model_bank', False):
        entries = [(tuple(tile['wav_range']), tile['model']) for tile in model['tiles']]
    else:
        entries = [(tuple(model['wav_range']), model)]
    expected_ranges = [tuple(item) for item in spec['wavelength_ranges']]
    if [item[0] for item in entries] != expected_ranges:
        raise ValueError(f'{mol} checkpoint wavelength ranges differ from MODEL_SPECS')
    expected_pca = spec['n_pca']
    if np.isscalar(expected_pca):
        expected_pca = [int(expected_pca)] * len(entries)
    elif len(expected_pca) != len(entries):
        raise ValueError(f'{mol} checkpoint PCA counts differ from MODEL_SPECS')
    for (wav_range, checkpoint), expected_count in zip(entries, expected_pca):
        pca = checkpoint.get('pca')
        n_pca = None if pca is None else pca.n_components_
        shape_layers = [layer.out_features for layer in checkpoint['net_shape'].net
                        if isinstance(layer, torch.nn.Linear)]
        peak_layers = [layer.out_features for layer in checkpoint['net_peak'].net
                       if isinstance(layer, torch.nn.Linear)]
        shape_hidden = tuple(shape_layers[:-1])
        peak_hidden = tuple(peak_layers[:-1])
        if (n_pca != expected_count or shape_hidden != tuple(spec['hidden']) or
                peak_hidden != tuple(spec['hidden'])):
            raise ValueError(
                f'{mol} {wav_range} checkpoint has PCA={n_pca}, '
                f'shape layers={shape_hidden}, peak layers={peak_hidden}; '
                f"MODEL_SPECS expects PCA={expected_count}, layers={spec['hidden']}"
            )
        for key, expected in (('T_range', spec['T_bounds']),
                              ('logN_range', spec['logN_bounds'])):
            if checkpoint.get(key) is None or not np.allclose(checkpoint[key], expected):
                raise ValueError(f'{mol} {wav_range} checkpoint {key} differs from MODEL_SPECS')


def load_fitting_models(molecules, root=None, model_specs=None, *, sigma=0):
    """Load requested bundled v2 checkpoints, or checkpoints below ``root``.

    The bundled files contain networks, scalers, PCA, and wavelength axes.
    Neither slab grids nor pretraining tables are read during inference.
    ``root`` may point to an alternate directory containing ``Trained_model``.
    ``sigma`` selects the grid broadening (0 or 2). Observational noise is a
    separate fitting input.
    """
    root = MODEL_ROOT if root is None else Path(root)
    if model_specs is None:
        model_specs = _model_specs_for_sigma(sigma)
    else:
        _model_specs_for_sigma(sigma)
        if any(spec.get('grid_sigma', sigma) != sigma for spec in model_specs.values()):
            raise ValueError('model_specs grid_sigma must match sigma')
    molecules = list(dict.fromkeys([molecules] if isinstance(molecules, str) else molecules))
    unknown = [mol for mol in molecules if mol not in model_specs]
    if unknown:
        raise ValueError(f'No v2 model checkpoint for {unknown} at sigma={sigma}')
    paths = {mol: str(Path(root) / model_specs[mol]['path']) for mol in molecules}
    missing = [path for path in paths.values() if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError('Missing v2 fitting checkpoint(s):\n' + '\n'.join(missing))
    pretrained = load_models(
        paths,
        n_pca={mol: model_specs[mol]['n_pca'] for mol in molecules},
    )
    for mol in molecules:
        _validate_loaded_model(mol, pretrained[mol], model_specs[mol])
        pretrained[mol]['grid_sigma'] = sigma
    return pretrained


def synthesize_spectrum(molecule, T, logN, A=1.0, *, sigma=0,
                        wavelength=None, distance_pc=140.0, root=None):
    """Return wavelength (µm) and model flux (Jy) for one bundled molecule.

    ``logN`` is log10 column density in cm⁻² and ``A`` is emitting area in
    au². Checkpoints predict flux for A=1 au² at 140 pc; ``distance_pc``
    rescales that flux. With no ``wavelength``, use the checkpoint's native
    wavelength axis (the union of its tiles for a model bank).
    """
    specs = _model_specs_for_sigma(sigma)
    if molecule not in specs:
        raise ValueError(f'No v2 model checkpoint for {molecule!r} at sigma={sigma}')
    spec = specs[molecule]
    if not spec['T_bounds'][0] <= T <= spec['T_bounds'][1]:
        raise ValueError(f'T is outside the trained range {spec["T_bounds"]}')
    if not spec['logN_bounds'][0] <= logN <= spec['logN_bounds'][1]:
        raise ValueError(f'logN is outside the trained range {spec["logN_bounds"]}')
    if not np.isfinite(A) or A < 0:
        raise ValueError('A must be finite and nonnegative')
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError('distance_pc must be finite and positive')
    model = load_fitting_models(molecule, root=root, sigma=sigma)[molecule]
    if wavelength is None:
        if model.get('is_model_bank', False):
            wavelength = np.unique(np.concatenate([
                tile['model']['wav'] for tile in model['tiles']
            ]))
        else:
            wavelength = model['wav'].copy()
    else:
        wavelength = np.asarray(wavelength, dtype=float)
        if (wavelength.ndim != 1 or wavelength.size == 0 or
                not np.all(np.isfinite(wavelength)) or
                np.any(np.diff(wavelength) <= 0)):
            raise ValueError('wavelength must be a nonempty, increasing 1D array')
        _check_bank_wavelength_coverage(model, wavelength)
        if not model.get('is_model_bank', False):
            outside = (wavelength < model['wav'][0]) | (wavelength > model['wav'][-1])
            if np.any(outside):
                warnings.warn(
                    f'{int(outside.sum())} wavelength pixels lie outside the '
                    f'{molecule} model range {model["wav"][0]:.6g}-'
                    f'{model["wav"][-1]:.6g} um; their flux is set to zero',
                    stacklevel=2,
                )
    flux = generate_spectrum(T, logN, A, model, obs_wav=wavelength)
    return wavelength, flux * (140.0 / float(distance_pc)) ** 2


def _check_trained_bounds(molecules, options, model_specs):
    """Reject T/logN bounds outside a checkpoint's trained domain.

    A network evaluated outside its training grid extrapolates without any
    warning, so such bounds would return unphysical fits.
    """
    components = _build_components(molecules, options.get('h2o_components', 1),
                                   options.get('component_names'))
    for key in ('T_bounds', 'logN_bounds'):
        spec = options.get(key)
        for comp in components:
            label, mol = comp['label'], comp['model_mol']
            if isinstance(spec, dict):
                bounds = spec.get(label, spec.get(mol))
            else:
                bounds = spec
            if bounds is None:
                continue
            lo, hi = model_specs[mol][key]
            if bounds[0] < lo - 1e-9 or bounds[1] > hi + 1e-9:
                raise ValueError(
                    f'{key} {tuple(bounds)} for {label!r} exceed the {mol} model\'s '
                    f'trained range ({lo}, {hi}); narrow them, or use '
                    f'fit_observation_grids with local slab grids'
                )


def fit_stage(obs_wav, obs_flux, molecule, wavelength_range, fit_ranges,
              pretrained, *, model_specs=None, distance_pc=140.0,
              **options):
    """Fit molecules jointly within selected windows on the full observed grid.

    ``molecule`` accepts one name or a list. ``fit_ranges`` accepts one interval,
    several intervals, or a molecule-to-interval mapping. ``wavelength_range``
    sets the stage's allowed and plotted domain; the returned model still spans
    the full observation so it can be subtracted before the next stage.
    ``distance_pc`` converts the 140-pc checkpoint flux normalization to the
    supplied source distance, so the fitted A is an area in au².
    Other options go to ``fit_molecules``; for example ``errorbars=False``
    skips the MCMC error bars and returns the best fit only.
    """
    molecules = [molecule] if isinstance(molecule, str) else list(molecule)
    if not molecules or any(mol not in pretrained for mol in molecules):
        raise ValueError('every stage molecule must have a loaded checkpoint')
    if model_specs is None:
        sigmas = {pretrained[mol].get('grid_sigma', 0)
                  if isinstance(pretrained[mol], dict) else 0
                  for mol in molecules}
        if len(sigmas) != 1:
            raise ValueError('stage molecules must use the same grid sigma')
        model_specs = _model_specs_for_sigma(sigmas.pop())
    plot_ranges = _ranges(wavelength_range)
    selected_ranges = _fit_ranges(fit_ranges)
    if any(not any(a <= lo and hi <= b for a, b in plot_ranges)
           for lo, hi in selected_ranges):
        raise ValueError('all fit_ranges must lie within wavelength_range')
    T_bounds = {mol: model_specs[mol]['T_bounds'] for mol in molecules}
    logN_bounds = {mol: model_specs[mol]['logN_bounds'] for mol in molecules}
    options.setdefault('T_bounds', T_bounds)
    options.setdefault('logN_bounds', logN_bounds)
    options.setdefault('loga_bounds', (-2.0, 2.0))
    _check_trained_bounds(molecules, options, model_specs)
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError('distance_pc must be finite and positive')
    # Checkpoints predict flux at 140 pc for A=1 au². Fitting the observed
    # spectrum scaled to that reference distance returns a physical area.
    to_reference = (float(distance_pc) / 140.0) ** 2
    if options.get('sigma') is not None:
        options['sigma'] *= to_reference
    fit = fit_molecules(
        obs_wav=obs_wav, obs_flux=np.asarray(obs_flux) * to_reference,
        mol=molecule,
        pretrained=pretrained, fit_ranges=fit_ranges, **options,
    )
    fit['model_flux'] /= to_reference
    fit['residual'] /= to_reference
    for label in fit['component_fluxes']:
        fit['component_fluxes'][label] /= to_reference
    for solution in fit['top_list'] + fit.get('global_top', []):
        solution['mse_loss'] /= to_reference ** 2
    info = fit.get('uncertainty_info', {})
    for key in ('sigma_input', 'residual_rms', 'sigma_likelihood'):
        if info.get(key) is not None:
            info[key] /= to_reference
    fit['obs_wav'] = np.asarray(obs_wav)
    fit['obs_flux'] = np.asarray(obs_flux).copy()
    fit['wavelength_range'] = plot_ranges
    fit['distance_pc'] = float(distance_pc)
    return fit


def fit_observation(input_path=None, stages=None, *, name=None, root=None, sigma=0,
                    output_dir=None, comparison_csv=None,
                    distance_pc=140.0,
                    line_free_windows=((11.45, 11.55), (13.10, 13.20),
                                       (15.65, 15.72), (15.90, 15.91)),
                    detection_screening=True, detection_sigma_factor=None,
                    n_samples=None, n_refine=None, n_top=None, n_mcmc=None,
                    errorbars=True,
                    sample_method='sobol', seed=42, skiprows=1,
                    delimiter=',', wav_col=0, flux_col=1,
                    obs_wav=None, obs_flux=None, obs_err=None):
    """Fit ordered stages, subtracting each model and saving CSV products.

    Each stage specifies ``name``, ``mol``, ``wavelength_range``, and
    ``fit_ranges``. A stage may contain multiple molecules or fit windows.
    Optional stage keys include ``detect_peaks``, ``h2o_components``,
    ``component_names``, ``shared_area_groups``, ``isotopologue_groups``,
    ``Tshare``, ``Ashare``, and bounds passed to
    ``fit_stage``.
    ``sigma`` selects the bundled grid broadening (0 or 2), whereas estimated
    ``sigma_noise`` represents the observed flux uncertainty. Pass either
    ``input_path`` or in-memory ``obs_wav`` and ``obs_flux`` arrays. Optional
    ``obs_err`` supplies per-pixel measurement errors; their positive finite
    median is used as the scalar noise level for screening and uncertainty
    estimates. Spectral fitting itself remains unweighted by individual errors.
    With ``errorbars=True`` (default), each stage's error bars come from emcee
    sampling of its posterior, starting with ``n_mcmc`` likelihood evaluations
    (default 300000, or ``DISKMELTS_N_MCMC``); see ``diskmelts.uncertainty``.
    ``errorbars=False`` skips sampling and reports best fits only.
    """
    model_root = MODEL_ROOT if root is None else Path(root)
    model_specs = _model_specs_for_sigma(sigma)
    array_input = obs_wav is not None or obs_flux is not None or obs_err is not None
    if input_path is not None and array_input:
        raise ValueError('pass either input_path or obs_wav/obs_flux/obs_err, not both')
    if input_path is None:
        if obs_wav is None or obs_flux is None:
            raise ValueError('obs_wav and obs_flux are required without input_path')
        obs_wav = np.asarray(obs_wav, dtype=float)
        obs_flux = np.asarray(obs_flux, dtype=float)
        if (obs_wav.ndim != 1 or obs_flux.ndim != 1 or
                obs_wav.shape != obs_flux.shape or obs_wav.size == 0):
            raise ValueError('obs_wav and obs_flux must be nonempty, matching 1D arrays')
        if obs_err is not None:
            obs_err = np.asarray(obs_err, dtype=float)
            if obs_err.shape != obs_wav.shape:
                raise ValueError('obs_err must match obs_wav and obs_flux')
        valid = np.isfinite(obs_wav) & np.isfinite(obs_flux)
        obs_wav, obs_flux = obs_wav[valid], obs_flux[valid]
        if obs_err is not None:
            obs_err = obs_err[valid]
        if obs_wav.size == 0 or np.any(np.diff(obs_wav) <= 0):
            raise ValueError('usable obs_wav values must be increasing and unique')
        source_name = name or 'spectrum'
    else:
        input_path = Path(input_path)
        if not input_path.is_file():
            raise FileNotFoundError(f'Observed spectrum does not exist: {input_path}')
        obs_wav, obs_flux = load_observed_spectrum(
            input_path, skiprows=skiprows, delimiter=delimiter,
            wav_col=wav_col, flux_col=flux_col,
        )
        source_name = name or input_path.stem
    stages = list(stages) if stages is not None else []
    if not stages:
        raise ValueError('at least one fitting stage is required')
    names = [stage['name'] for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError('stage names must be unique')
    molecules = list(dict.fromkeys(
        mol for stage in stages
        for mol in ([stage['mol']] if isinstance(stage['mol'], str) else stage['mol'])
    ))
    pretrained = load_fitting_models(molecules, root=model_root, sigma=sigma)
    default_output_dir = Path.cwd() / 'realobs_results'
    if sigma == 2:
        default_output_dir /= 'sigma2'
    output_dir = Path(output_dir) if output_dir else default_output_dir
    comparison_csv = (Path(comparison_csv) if comparison_csv else
                      output_dir / 'Fitted_Parameters.csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(n_samples if n_samples is not None else
                    os.environ.get('DISKMELTS_N_SAMPLES', '20000'))
    n_refine = int(n_refine if n_refine is not None else
                   os.environ.get('DISKMELTS_N_REFINE', '32'))
    n_top = int(n_top if n_top is not None else
                os.environ.get('DISKMELTS_N_TOP', '20'))
    n_mcmc = default_mcmc_steps(n_mcmc)
    if detection_sigma_factor is None:
        detection_sigma_factor = float(os.environ.get('DISKMELTS_DETECTION_SIGMA_FACTOR', '3.0'))
    sigma_noise = None
    if obs_err is not None:
        usable_err = obs_err[np.isfinite(obs_err) & (obs_err > 0)]
        if usable_err.size == 0:
            raise ValueError('obs_err must contain a positive finite value')
        sigma_noise = float(np.median(usable_err))
        print(f'Using median supplied error σ = {sigma_noise:.5f} Jy')
    elif line_free_windows:
        chunks = [obs_flux[(obs_wav >= lo) & (obs_wav <= hi)]
                  for lo, hi in line_free_windows
                  if np.any((obs_wav >= lo) & (obs_wav <= hi))]
        if chunks:
            sample = np.concatenate(chunks)
            sigma_noise = float(np.nanstd(sample))
            print(f'Estimated σ = {sigma_noise:.5f} Jy ({len(sample)} line-free pixels)')

    residual = obs_flux.copy()
    cumulative_model = np.zeros_like(obs_flux)
    all_fits = {}
    products = {}
    for stage in stages:
        stage_name = stage['name']
        print(f'\nStage: {stage_name}')
        detected = detect_stage_molecules(
            obs_wav, residual, stage, sigma_noise,
            screening=detection_screening, sigma_factor=detection_sigma_factor,
        )
        if not detected:
            print(f'  Stage {stage_name!r} skipped.')
            continue
        selected = detected[0] if len(detected) == 1 else detected
        fit = fit_stage(
            obs_wav, residual, selected, stage['wavelength_range'],
            stage['fit_ranges'], pretrained, model_specs=model_specs,
            h2o_components=stage.get('h2o_components', 1),
            **_screened_group_options(stage, detected),
            T_bounds=stage.get('T_bounds', {mol: model_specs[mol]['T_bounds'] for mol in detected}),
            logN_bounds=stage.get('logN_bounds', {mol: model_specs[mol]['logN_bounds'] for mol in detected}),
            loga_bounds=stage.get('loga_bounds', (-2.0, 2.0)),
            n_samples=n_samples, sample_method=sample_method,
            n_refine=n_refine, n_top=n_top, n_mcmc=n_mcmc, errorbars=errorbars,
            sigma=sigma_noise,
            seed=seed, distance_pc=distance_pc,
        )
        residual -= fit['model_flux']
        cumulative_model += fit['model_flux']
        prefix = f'{source_name}_{stage_name}'
        saved = save_fit_outputs(fit, output_dir, prefix)
        saved['running'] = save_running_spectrum(
            obs_wav, residual, cumulative_model, output_dir, prefix,
        )
        print_fit_params(fit)
        all_fits[stage_name] = fit
        products[stage_name] = saved

    if all_fits:
        products['final_running'] = save_running_spectrum(
            obs_wav, residual, cumulative_model, output_dir,
            f'{source_name}_final',
        )
        comparison_csv.parent.mkdir(parents=True, exist_ok=True)
        save_fitted_comparison(comparison_csv, source_name, all_fits)
        products['comparison_csv'] = str(comparison_csv)
    print('\nDone.')
    return {
        'fits': all_fits, 'obs_wav': obs_wav, 'obs_flux': obs_flux,
        'obs_err': obs_err,
        'residual': residual, 'cumulative_model': cumulative_model,
        'sigma_noise': sigma_noise, 'grid_sigma': sigma, 'products': products,
    }
