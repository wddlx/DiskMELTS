"""Reusable v2 fitting workflow for continuum-subtracted observations.

``MODEL_SPECS`` records the trained domain, PCA size, and network layers for
each checkpoint. ``fit_stage`` fits one or several molecules over one or more
wavelength intervals. ``fit_observation`` runs stages on successive residuals
and writes their numerical results for later reading and plotting.
"""

import os
from pathlib import Path

import numpy as np
import torch

from diskmelts.fitting import (
    detect_stage_molecules,
    fit_molecules,
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
}
MODEL_SPECS = {
    'H2O': {
        'path': 'Trained_model/H2O_wavelength_bank_v2_fixed/H2O_model_bank.json',
        'wavelength_ranges': WATER_RANGES,
        'T_bounds': (100.0, 1400.0),
        'logN_bounds': (13.0, 19.0),
        'n_pca': 30,
        'hidden': (64, 128, 64),
    },
    **{
        mol: {
            'path': f'Trained_model/{mol}_single_v2/net_{mol}_forward.pt',
            'wavelength_ranges': (wav_range,),
            'T_bounds': (100.0, 1400.0),
            'logN_bounds': (13.0, 19.5),
            'n_pca': 30,
            'hidden': (64, 128, 64),
        }
        for mol, wav_range in CARBON_RANGES.items()
    },
}


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
    if model.get('is_model_bank', False):
        entries = [(tuple(tile['wav_range']), tile['model']) for tile in model['tiles']]
    else:
        entries = [(tuple(model['wav_range']), model)]
    expected_ranges = [tuple(item) for item in spec['wavelength_ranges']]
    if [item[0] for item in entries] != expected_ranges:
        raise ValueError(f'{mol} checkpoint wavelength ranges differ from MODEL_SPECS')
    for wav_range, checkpoint in entries:
        pca = checkpoint.get('pca')
        n_pca = None if pca is None else pca.n_components_
        shape_layers = [layer.out_features for layer in checkpoint['net_shape'].net
                        if isinstance(layer, torch.nn.Linear)]
        peak_layers = [layer.out_features for layer in checkpoint['net_peak'].net
                       if isinstance(layer, torch.nn.Linear)]
        shape_hidden = tuple(shape_layers[:-1])
        peak_hidden = tuple(peak_layers[:-1])
        if (n_pca != spec['n_pca'] or shape_hidden != tuple(spec['hidden']) or
                peak_hidden != tuple(spec['hidden'])):
            raise ValueError(
                f'{mol} {wav_range} checkpoint has PCA={n_pca}, '
                f'shape layers={shape_hidden}, peak layers={peak_hidden}; '
                f"MODEL_SPECS expects PCA={spec['n_pca']}, layers={spec['hidden']}"
            )
        for key, expected in (('T_range', spec['T_bounds']),
                              ('logN_range', spec['logN_bounds'])):
            if checkpoint.get(key) is None or not np.allclose(checkpoint[key], expected):
                raise ValueError(f'{mol} {wav_range} checkpoint {key} differs from MODEL_SPECS')


def load_fitting_models(molecules, root=None, model_specs=MODEL_SPECS):
    """Load requested bundled v2 checkpoints, or checkpoints below ``root``.

    The bundled files contain networks, scalers, PCA, and wavelength axes.
    Neither slab grids nor pretraining tables are read during inference.
    ``root`` may point to an alternate directory containing ``Trained_model``.
    """
    root = MODEL_ROOT if root is None else Path(root)
    molecules = list(dict.fromkeys([molecules] if isinstance(molecules, str) else molecules))
    unknown = [mol for mol in molecules if mol not in model_specs]
    if unknown:
        raise ValueError(f'No v2 model configuration for {unknown}')
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
    return pretrained


def fit_stage(obs_wav, obs_flux, molecule, wavelength_range, fit_ranges,
              pretrained, *, model_specs=MODEL_SPECS, distance_pc=140.0,
              **options):
    """Fit molecules jointly within selected windows on the full observed grid.

    ``molecule`` accepts one name or a list. ``fit_ranges`` accepts one interval,
    several intervals, or a molecule-to-interval mapping. ``wavelength_range``
    sets the stage's allowed and plotted domain; the returned model still spans
    the full observation so it can be subtracted before the next stage.
    ``distance_pc`` converts the 140-pc checkpoint flux normalization to the
    supplied source distance, so the fitted A is an area in au².
    """
    molecules = [molecule] if isinstance(molecule, str) else list(molecule)
    if not molecules or any(mol not in pretrained for mol in molecules):
        raise ValueError('every stage molecule must have a loaded checkpoint')
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
    for solution in fit['top_list']:
        solution['mse_loss'] /= to_reference ** 2
    fit['obs_wav'] = np.asarray(obs_wav)
    fit['obs_flux'] = np.asarray(obs_flux).copy()
    fit['wavelength_range'] = plot_ranges
    fit['distance_pc'] = float(distance_pc)
    return fit


def fit_observation(input_path, stages, *, name=None, root=None,
                    output_dir=None, comparison_csv=None,
                    distance_pc=140.0,
                    line_free_windows=((11.45, 11.55), (13.10, 13.20),
                                       (15.65, 15.72), (15.90, 15.91)),
                    detection_screening=True, detection_sigma_factor=None,
                    n_samples=None, n_refine=None, n_top=None,
                    sample_method='sobol', seed=42, skiprows=1,
                    delimiter=',', wav_col=0, flux_col=1):
    """Fit ordered stages, subtracting each model and saving CSV products.

    Each stage specifies ``name``, ``mol``, ``wavelength_range``, and
    ``fit_ranges``. A stage may contain multiple molecules or fit windows.
    Optional stage keys include ``detect_peaks``, ``h2o_components``,
    ``component_names``, and bounds passed to ``fit_stage``.
    """
    model_root = MODEL_ROOT if root is None else Path(root)
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f'Observed spectrum does not exist: {input_path}')
    stages = list(stages)
    if not stages:
        raise ValueError('at least one fitting stage is required')
    names = [stage['name'] for stage in stages]
    if len(names) != len(set(names)):
        raise ValueError('stage names must be unique')
    molecules = list(dict.fromkeys(
        mol for stage in stages
        for mol in ([stage['mol']] if isinstance(stage['mol'], str) else stage['mol'])
    ))
    pretrained = load_fitting_models(molecules, root=model_root)
    obs_wav, obs_flux = load_observed_spectrum(
        input_path, skiprows=skiprows, delimiter=delimiter,
        wav_col=wav_col, flux_col=flux_col,
    )
    source_name = name or input_path.stem
    output_dir = Path(output_dir) if output_dir else Path.cwd() / 'realobs_results'
    comparison_csv = (Path(comparison_csv) if comparison_csv else
                      output_dir / 'Fitted_Parameters.csv')
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(n_samples if n_samples is not None else
                    os.environ.get('DISKMELTS_N_SAMPLES', '20000'))
    n_refine = int(n_refine if n_refine is not None else
                   os.environ.get('DISKMELTS_N_REFINE', '32'))
    n_top = int(n_top if n_top is not None else
                os.environ.get('DISKMELTS_N_TOP', '20'))
    if detection_sigma_factor is None:
        detection_sigma_factor = float(os.environ.get('DISKMELTS_DETECTION_SIGMA_FACTOR', '3.0'))
    sigma_noise = None
    if line_free_windows:
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
            stage['fit_ranges'], pretrained,
            h2o_components=stage.get('h2o_components', 1),
            component_names=stage.get('component_names'),
            T_bounds=stage.get('T_bounds', {mol: MODEL_SPECS[mol]['T_bounds'] for mol in detected}),
            logN_bounds=stage.get('logN_bounds', {mol: MODEL_SPECS[mol]['logN_bounds'] for mol in detected}),
            loga_bounds=stage.get('loga_bounds', (-2.0, 2.0)),
            n_samples=n_samples, sample_method=sample_method,
            n_refine=n_refine, n_top=n_top, sigma=sigma_noise, seed=seed,
            distance_pc=distance_pc,
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
        'residual': residual, 'cumulative_model': cumulative_model,
        'sigma_noise': sigma_noise, 'products': products,
    }
