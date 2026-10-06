"""Fit observations directly with the v2 slab CSV grids.

This workflow reads ``Model_grids/<molecule>/T...N....csv`` and never loads a
trained surrogate. It interpolates flux linearly between adjacent T/logN grid
points, fits positive component amplitudes, and estimates local likelihood
errors. The reported T and logN errors include the *full local grid step* in
quadrature, as requested: ``sqrt(fit_sigma**2 + grid_step**2)``.

``fit_grid_stage`` fits one or several molecules. ``fit_observation_grids``
runs sequential stages and writes the same stage-spectrum/parameter/running
CSV format used by ``dev_v2_read_plot.py``. The local covariance describes one
mode near the optimum; it is not a substitute for a multimodal posterior.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, lsq_linear

from diskmelts.fitting import (
    detect_stage_molecules,
    load_observed_spectrum,
    print_fit_params,
    save_fit_outputs,
    save_running_spectrum,
)


GRID_ROOT = None
LINE_FREE_WINDOWS = ((11.45, 11.55), (13.10, 13.20),
                     (15.65, 15.72), (15.90, 15.91))
_GRID_NAME = re.compile(r'T(?P<T>\d+(?:\.\d+)?)N(?P<N>\d+(?:\.\d+)?)\.csv')


def _ranges(value):
    if len(value) == 2 and all(np.isscalar(item) for item in value):
        return [tuple(map(float, value))]
    return [tuple(map(float, pair)) for pair in value]


def _stage_ranges(fit_ranges, components):
    if not isinstance(fit_ranges, dict):
        return _ranges(fit_ranges)
    out = []
    for component in components:
        label, mol = component['label'], component['model_mol']
        if label in fit_ranges:
            out.extend(_ranges(fit_ranges[label]))
        elif mol in fit_ranges:
            out.extend(_ranges(fit_ranges[mol]))
        else:
            raise KeyError(f'fit_ranges has no entry for {label} or {mol}')
    return list(dict.fromkeys(out))


def _components(molecule, h2o_components, component_names):
    molecules = [molecule] if isinstance(molecule, str) else list(molecule)
    if not molecules or len(molecules) != len(set(molecules)):
        raise ValueError('molecule must contain distinct molecular names')
    if h2o_components not in (1, 2):
        raise ValueError('h2o_components must be 1 or 2')
    parts = []
    for mol in molecules:
        if mol == 'H2O' and h2o_components == 2:
            parts.extend([{'label': 'H2O_warm', 'model_mol': mol},
                          {'label': 'H2O_hot', 'model_mol': mol}])
        else:
            parts.append({'label': mol, 'model_mol': mol})
    if component_names is not None:
        if len(component_names) != len(parts):
            raise ValueError('component_names must match the fitted components')
        for part, label in zip(parts, component_names):
            part['label'] = label
    labels = [part['label'] for part in parts]
    if len(labels) != len(set(labels)):
        raise ValueError('component labels must be unique')
    return parts


def _axis_including_bounds(values, bounds):
    if len(values) < 2:
        raise ValueError('grid needs at least two points on each parameter axis')
    lo, hi = map(float, bounds)
    if not lo < hi or lo < values[0] or hi > values[-1]:
        raise ValueError(f'bounds {bounds} fall outside grid [{values[0]}, {values[-1]}]')
    first = max(0, int(np.searchsorted(values, lo, side='right')) - 1)
    last = min(len(values) - 1, int(np.searchsorted(values, hi, side='left')))
    if first == last:
        last = min(len(values) - 1, last + 1)
    if first == last:
        first = max(0, first - 1)
    return values[first:last + 1]


def _bracket(axis, value):
    if value < axis[0] - 1e-10 or value > axis[-1] + 1e-10:
        raise ValueError(f'{value} is outside the loaded grid axis')
    lower = int(np.clip(np.searchsorted(axis, value, side='right') - 1,
                        0, len(axis) - 2))
    fraction = (value - axis[lower]) / (axis[lower + 1] - axis[lower])
    return lower, float(np.clip(fraction, 0.0, 1.0))


def _local_step(axis, value):
    index, _ = _bracket(axis, value)
    return float(axis[index + 1] - axis[index])


@dataclass
class GridCube:
    molecule: str
    temperatures: np.ndarray
    log_columns: np.ndarray
    fit_wav: np.ndarray
    fit_flux: np.ndarray
    paths: dict

    def evaluate(self, temperature, log_column, obs_wav=None):
        """Bilinear flux interpolation on fit pixels or a requested axis."""
        i, ft = _bracket(self.temperatures, temperature)
        j, fn = _bracket(self.log_columns, log_column)
        weights = ((i, j, (1 - ft) * (1 - fn)),
                   (i + 1, j, ft * (1 - fn)),
                   (i, j + 1, (1 - ft) * fn),
                   (i + 1, j + 1, ft * fn))
        if obs_wav is None:
            return sum((weight * self.fit_flux[ti, nj] for ti, nj, weight in weights),
                       np.zeros_like(self.fit_wav))
        obs_wav = np.asarray(obs_wav, dtype=float)
        flux = np.zeros_like(obs_wav)
        for ti, nj, weight in weights:
            if weight == 0:
                continue
            frame = pd.read_csv(self.paths[(self.temperatures[ti], self.log_columns[nj])])
            flux += weight * np.interp(obs_wav, frame.wave, frame.Line,
                                       left=0.0, right=0.0)
        return flux


def load_grid_cube(molecule, fit_wav, *, grid_root=GRID_ROOT,
                   T_bounds=(100.0, 1400.0), logN_bounds=(13.0, 19.0)):
    """Load a complete rectangular T/logN subgrid on the selected fit pixels."""
    if grid_root is None:
        raise ValueError('grid_root is required for direct-grid fitting')
    grid_dir = Path(grid_root) / molecule
    if not grid_dir.is_dir():
        raise FileNotFoundError(f'Missing required slab grid directory: {grid_dir}')
    paths = {}
    for path in grid_dir.glob('*.csv'):
        match = _GRID_NAME.fullmatch(path.name)
        if match:
            key = (float(match['T']), float(match['N']))
            if key in paths:
                raise ValueError(f'Duplicate grid point {key} for {molecule}')
            paths[key] = path
    if not paths:
        raise ValueError(f'No T...N....csv slab grids in {grid_dir}')
    all_T = np.array(sorted({key[0] for key in paths}), dtype=float)
    all_N = np.array(sorted({key[1] for key in paths}), dtype=float)
    temperatures = _axis_including_bounds(all_T, T_bounds)
    log_columns = _axis_including_bounds(all_N, logN_bounds)
    missing = [(T, N) for T in temperatures for N in log_columns
               if (T, N) not in paths]
    if missing:
        raise ValueError(f'{molecule} grid is incomplete in the fitted domain; '
                         f'first missing point is {missing[0]}')
    fit_wav = np.asarray(fit_wav, dtype=float)
    if (fit_wav.ndim != 1 or not len(fit_wav) or
            not np.all(np.isfinite(fit_wav)) or
            np.any(np.diff(fit_wav) <= 0)):
        raise ValueError('fit_wav must be a nonempty increasing wavelength axis')
    cube = np.empty((len(temperatures), len(log_columns), len(fit_wav)),
                    dtype=np.float64)
    native_wav = None
    for ti, T in enumerate(temperatures):
        for nj, N in enumerate(log_columns):
            frame = pd.read_csv(paths[(T, N)], usecols=['wave', 'Line'])
            wav = frame.wave.to_numpy(dtype=float)
            line = frame.Line.to_numpy(dtype=float)
            if (len(wav) < 2 or not np.all(np.isfinite(wav)) or
                    not np.all(np.isfinite(line)) or np.any(np.diff(wav) <= 0)):
                raise ValueError(f'Invalid grid spectrum: {paths[(T, N)]}')
            if native_wav is None:
                native_wav = wav
                if fit_wav[0] < wav[0] or fit_wav[-1] > wav[-1]:
                    raise ValueError(f'{molecule} grid does not cover the fit wavelengths')
            elif not np.array_equal(wav, native_wav):
                raise ValueError(f'{molecule} grid wavelength axes differ between files')
            cube[ti, nj] = np.interp(fit_wav, wav, line)
    print(f'  Loaded {molecule}: {len(temperatures)} T × {len(log_columns)} logN '
          f'grids on {len(fit_wav)} fit pixels')
    return GridCube(molecule, temperatures, log_columns, fit_wav, cube, paths)


def combine_grid_uncertainty(fit_sigma, grid_step):
    """Add the local grid spacing in quadrature to a fitted 1σ error."""
    if not np.isfinite(fit_sigma) or fit_sigma < 0:
        return float('nan')
    if not np.isfinite(grid_step) or grid_step <= 0:
        raise ValueError('grid_step must be finite and positive')
    return float(np.hypot(fit_sigma, grid_step))


def _fit_covariance(result, scales, sigma_known, fixed=None):
    """Local Gaussian covariance from the weighted residual Jacobian."""
    if fixed is None:
        fixed = np.zeros(len(scales), dtype=bool)
    active = ~np.asarray(fixed, dtype=bool)
    jac_scaled = result.jac[:, active] * scales[None, active]
    _, singular, vh = np.linalg.svd(jac_scaled, full_matrices=False)
    threshold = singular[0] * max(jac_scaled.shape) * np.finfo(float).eps * 100
    valid = singular > threshold
    inverse = np.zeros_like(singular)
    inverse[valid] = 1.0 / singular[valid] ** 2
    cov_scaled = (vh.T * inverse) @ vh
    if not sigma_known:
        dof = max(1, len(result.fun) - int(active.sum()))
        scatter = float(np.sqrt(np.sum(result.fun ** 2) / dof))
        cov_scaled *= scatter ** 2
    active_scales = scales[active]
    cov = (active_scales[:, None] * cov_scaled) * active_scales[None, :]
    errors = np.zeros(len(scales))
    errors[active] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    if not np.all(valid):
        null_weight = np.sum(vh[~valid] ** 2, axis=0)
        active_errors = errors[active]
        active_errors[null_weight > 1e-4] = np.nan
        errors[active] = active_errors
    return errors


def fit_grid_stage(obs_wav, obs_flux, molecule, wavelength_range, fit_ranges, *,
                   grid_root=GRID_ROOT, h2o_components=1, component_names=None,
                   T_bounds=(100.0, 1400.0), logN_bounds=None,
                   loga_bounds=(-2.0, 2.0), sigma=None, distance_pc=140.0,
                   seed=42,
                   maxiter=100, popsize=8, n_refine=3):
    """Fit one or more slab-grid molecules with positive amplitudes.

    Differential evolution searches nonlinear T/logN space while amplitudes
    are solved by bounded least squares. A few best candidates are refined by
    weighted nonlinear least squares. Formal 1σ errors come from the local
    covariance; the T/logN errors then include their grid steps in quadrature.
    ``distance_pc`` converts the 140-pc grid flux normalization to the source
    distance and makes fitted amplitudes physical emitting areas in au².
    """
    if grid_root is None:
        raise ValueError('grid_root is required for direct-grid fitting')
    wav = np.asarray(obs_wav, dtype=float)
    flux = np.asarray(obs_flux, dtype=float)
    if (wav.ndim != 1 or flux.shape != wav.shape or
            not np.all(np.isfinite(wav)) or
            not np.all(np.isfinite(flux)) or np.any(np.diff(wav) <= 0)):
        raise ValueError('observed wavelength and flux must be aligned 1D arrays')
    if not np.isfinite(distance_pc) or distance_pc <= 0:
        raise ValueError('distance_pc must be finite and positive')
    to_reference = (float(distance_pc) / 140.0) ** 2
    parts = _components(molecule, h2o_components, component_names)
    plot_ranges = _ranges(wavelength_range)
    selected_ranges = _stage_ranges(fit_ranges, parts)
    if not selected_ranges or any(lo >= hi for lo, hi in selected_ranges):
        raise ValueError('fit_ranges must contain increasing intervals')
    if any(not any(a <= lo and hi <= b for a, b in plot_ranges)
           for lo, hi in selected_ranges):
        raise ValueError('fit_ranges must lie within wavelength_range')
    fit_mask = np.logical_or.reduce([(wav >= lo) & (wav <= hi)
                                     for lo, hi in selected_ranges])
    if not np.any(fit_mask):
        raise ValueError('fit_ranges select zero observed pixels')
    wav_fit, target = wav[fit_mask], flux[fit_mask] * to_reference
    if len(wav_fit) <= 3 * len(parts):
        raise ValueError('fit_ranges need more pixels than fitted parameters')
    if not np.all(np.isfinite(target)):
        raise ValueError('fit flux must be finite')
    if sigma is not None and (not np.isscalar(sigma) or
                              not np.isfinite(sigma) or sigma <= 0):
        raise ValueError('sigma must be a positive scalar in Jy')
    noise_scale = float(sigma) * to_reference if sigma is not None else max(
        float(np.std(target)), np.finfo(float).tiny,
    )
    lo_A, hi_A = 10.0 ** float(loga_bounds[0]), 10.0 ** float(loga_bounds[1])
    if not 0 < lo_A < hi_A:
        raise ValueError('loga_bounds must be increasing and finite')

    if n_refine < 1:
        raise ValueError('n_refine must be positive')

    def bound_for(spec, part, default):
        if spec is None:
            return default
        if isinstance(spec, dict):
            return tuple(spec.get(part['label'], spec.get(part['model_mol'], default)))
        return tuple(spec)

    def default_logN(part):
        return (13.0, 19.0) if part['model_mol'] == 'H2O' else (13.0, 19.5)

    bounds = [(bound_for(T_bounds, part, (100.0, 1400.0)),
               bound_for(logN_bounds, part, default_logN(part)))
              for part in parts]
    molecules = list(dict.fromkeys(part['model_mol'] for part in parts))
    libraries = {}
    for mol in molecules:
        local = [item for part, pair in zip(parts, bounds)
                 if part['model_mol'] == mol for item in pair]
        T_envelope = (min(item[0] for item in local[::2]),
                      max(item[1] for item in local[::2]))
        N_envelope = (min(item[0] for item in local[1::2]),
                      max(item[1] for item in local[1::2]))
        libraries[mol] = load_grid_cube(
            mol, wav_fit, grid_root=grid_root,
            T_bounds=T_envelope, logN_bounds=N_envelope,
        )

    nonlinear_bounds = [entry for pair in bounds for entry in pair]
    n_parts = len(parts)
    water_indices = [k for k, part in enumerate(parts)
                     if part['model_mol'] == 'H2O']
    ordered_water = h2o_components == 2 and len(water_indices) == 2

    def templates(nonlin):
        return np.column_stack([
            libraries[part['model_mol']].evaluate(nonlin[2 * k], nonlin[2 * k + 1])
            for k, part in enumerate(parts)
        ])

    def profile_amplitudes(nonlin):
        if (ordered_water and
                nonlin[2 * water_indices[0]] > nonlin[2 * water_indices[1]]):
            return np.inf, np.full(n_parts, lo_A)
        matrix = templates(nonlin)
        scale = max(float(np.max(np.abs(target))),
                    float(np.max(np.abs(matrix))), np.finfo(float).tiny)
        if n_parts == 1:
            column = matrix[:, 0] / scale
            denom = float(np.dot(column, column))
            amplitude = (float(np.clip(np.dot(column, target / scale) / denom,
                                       lo_A, hi_A)) if denom > 0 else lo_A)
            amplitudes = np.array([amplitude])
        else:
            solution = lsq_linear(matrix / scale, target / scale,
                                  bounds=(np.full(n_parts, lo_A),
                                          np.full(n_parts, hi_A)))
            amplitudes = solution.x
        residual = matrix @ amplitudes - target
        return float(np.dot(residual, residual)), amplitudes

    search = differential_evolution(
        lambda x: profile_amplitudes(x)[0] / noise_scale ** 2,
        nonlinear_bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        polish=False,
    )
    candidates = sorted((x for x in [search.x, *search.population]
                         if np.isfinite(profile_amplitudes(x)[0])),
                        key=lambda x: profile_amplitudes(x)[0])
    if not candidates:
        raise RuntimeError('grid search found no feasible component ordering')
    solution_bounds = [entry for pair in bounds
                       for entry in (pair[0], pair[1], loga_bounds)]
    lower = np.array([item[0] for item in solution_bounds], dtype=float)
    upper = np.array([item[1] for item in solution_bounds], dtype=float)
    scales = np.array([
        step for part in parts for step in (
            _local_step(libraries[part['model_mol']].temperatures,
                        np.mean(bound_for(T_bounds, part, (100.0, 1400.0)))),
            _local_step(libraries[part['model_mol']].log_columns,
                        np.mean(bound_for(logN_bounds, part, default_logN(part)))),
            0.5,
        )
    ])

    def residual_vector(x):
        if (ordered_water and
                x[3 * water_indices[0]] > x[3 * water_indices[1]]):
            # Keep warm/hot labels ordered during local refinement.
            return np.full_like(target, 1e6)
        model = np.zeros_like(target)
        for k, part in enumerate(parts):
            model += 10.0 ** x[3 * k + 2] * libraries[part['model_mol']].evaluate(
                x[3 * k], x[3 * k + 1],
            )
        return (model - target) / noise_scale

    best = None
    best_loss = np.inf
    seen = set()
    for candidate in candidates:
        key = tuple(np.round(candidate, 6))
        if key in seen:
            continue
        seen.add(key)
        _, amplitudes = profile_amplitudes(candidate)
        x0 = np.array([value for k in range(n_parts)
                       for value in (candidate[2 * k], candidate[2 * k + 1],
                                     np.log10(amplitudes[k]))])
        x0 = np.clip(x0, lower + 1e-10, upper - 1e-10)
        opt = least_squares(residual_vector, x0, bounds=(lower, upper),
                            x_scale=scales, max_nfev=600)
        loss = float(np.dot(opt.fun, opt.fun))
        if loss < best_loss:
            best, best_loss = opt, loss
        if len(seen) >= n_refine:
            break
    if best is None:
        raise RuntimeError('grid optimization returned no solution')

    fixed = upper - lower < 1e-4
    covariance_error = _fit_covariance(best, scales,
                                       sigma_known=sigma is not None,
                                       fixed=fixed)
    params, uncertainty, details, component_fluxes = {}, {}, {}, {}
    model_flux = np.zeros_like(flux)
    plot_mask = np.logical_or.reduce([(wav >= lo) & (wav <= hi)
                                      for lo, hi in plot_ranges])
    for k, part in enumerate(parts):
        label, mol = part['label'], part['model_mol']
        T, logN, logA = map(float, best.x[3 * k:3 * k + 3])
        A = 10.0 ** logA
        component = A / to_reference * libraries[mol].evaluate(T, logN, obs_wav=wav)
        component[~plot_mask] = 0.0
        component_fluxes[label] = component
        model_flux += component
        params[label] = {'T': T, 'logN': logN, 'A': A,
                         'log10A': logA, 'model_mol': mol}
        T_step = _local_step(libraries[mol].temperatures, T)
        N_step = _local_step(libraries[mol].log_columns, logN)
        T_fit = float(covariance_error[3 * k])
        N_fit = float(covariance_error[3 * k + 1])
        A_fit = float(covariance_error[3 * k + 2])
        T_total = combine_grid_uncertainty(T_fit, T_step)
        N_total = combine_grid_uncertainty(N_fit, N_step)
        uncertainty[label] = {
            'T': {'minus': T_total, 'plus': T_total},
            'logN': {'minus': N_total, 'plus': N_total},
            'log10A': {'minus': A_fit, 'plus': A_fit},
        }
        details[label] = {
            'T_fit_sigma': T_fit, 'T_grid_step': T_step, 'T_total_sigma': T_total,
            'logN_fit_sigma': N_fit, 'logN_grid_step': N_step,
            'logN_total_sigma': N_total, 'log10A_fit_sigma': A_fit,
            'T_at_bound': bool(np.isclose(T, bounds[k][0][0], rtol=0, atol=1e-5) or
                               np.isclose(T, bounds[k][0][1], rtol=0, atol=1e-5)),
            'logN_at_bound': bool(np.isclose(logN, bounds[k][1][0], rtol=0, atol=1e-6) or
                                  np.isclose(logN, bounds[k][1][1], rtol=0, atol=1e-6)),
            'log10A_at_bound': bool(np.isclose(logA, loga_bounds[0], rtol=0, atol=1e-6) or
                                    np.isclose(logA, loga_bounds[1], rtol=0, atol=1e-6)),
        }
    print(f'  Grid fit: {len(parts)} components, weighted SSE={best_loss:.4g}')
    return {
        'mol': molecule, 'components': parts, 'params': params,
        'uncertainty': uncertainty, 'uncertainty_details': details,
        'obs_wav': wav, 'obs_flux': flux.copy(), 'model_flux': model_flux,
        'component_fluxes': component_fluxes, 'residual': flux - model_flux,
        'fit_ranges': selected_ranges,
        'top_list': [{'rank': 1, 'mse_loss': best_loss * (noise_scale / to_reference) ** 2 / len(target),
                      'params': params}],
        'grid_root': str(grid_root), 'sigma_noise': sigma,
        'distance_pc': float(distance_pc),
    }


def fit_observation_grids(input_path, stages, *, name=None, grid_root=GRID_ROOT,
                          output_dir=None, line_free_windows=LINE_FREE_WINDOWS,
                          detection_screening=True, detection_sigma_factor=3.0,
                          sigma_noise=None, distance_pc=140.0,
                          seed=42, maxiter=100, popsize=8,
                          n_refine=3, skiprows=1, delimiter=',',
                          wav_col=0, flux_col=1):
    """Run ordered grid-only fits and save stage CSVs for dev_v2_read_plot."""
    if grid_root is None:
        raise ValueError('grid_root is required for direct-grid fitting')
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f'Observed spectrum does not exist: {input_path}')
    stages = list(stages)
    if not stages or len({stage['name'] for stage in stages}) != len(stages):
        raise ValueError('stages must be nonempty and have unique names')
    obs_wav, obs_flux = load_observed_spectrum(
        input_path, skiprows=skiprows, delimiter=delimiter,
        wav_col=wav_col, flux_col=flux_col,
    )
    source_name = name or input_path.stem
    output_dir = (Path(output_dir) if output_dir else
                  Path.cwd() / 'realobs_results' / 'grids' / source_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sigma_noise is None and line_free_windows:
        chunks = [obs_flux[(obs_wav >= lo) & (obs_wav <= hi)]
                  for lo, hi in line_free_windows
                  if np.any((obs_wav >= lo) & (obs_wav <= hi))]
        if chunks:
            sigma_noise = float(np.nanstd(np.concatenate(chunks)))
    if sigma_noise is not None and sigma_noise <= 0:
        raise ValueError('noise sigma must be positive to estimate fit errors')
    print(f'Grid fit for {source_name}: noise sigma={sigma_noise} Jy')
    residual = obs_flux.copy()
    cumulative = np.zeros_like(obs_flux)
    fits, products, summary_rows = {}, {}, []
    for stage in stages:
        detected = detect_stage_molecules(
            obs_wav, residual, stage, sigma_noise,
            screening=detection_screening,
            sigma_factor=detection_sigma_factor,
        )
        if not detected:
            print(f"  Grid stage {stage['name']} skipped")
            continue
        selected = detected[0] if len(detected) == 1 else detected
        print(f"Grid stage: {stage['name']}")
        fit = fit_grid_stage(
            obs_wav, residual, selected, stage['wavelength_range'],
            stage['fit_ranges'], grid_root=grid_root,
            h2o_components=stage.get('h2o_components', 1),
            component_names=stage.get('component_names'),
            T_bounds=stage.get('T_bounds', (100.0, 1400.0)),
            logN_bounds=stage.get('logN_bounds',
                                  {mol: ((13.0, 19.0) if mol == 'H2O'
                                         else (13.0, 19.5))
                                   for mol in ([selected] if isinstance(selected, str)
                                               else selected)}),
            loga_bounds=stage.get('loga_bounds', (-2.0, 2.0)),
            sigma=sigma_noise, distance_pc=distance_pc,
            seed=seed, maxiter=maxiter,
            popsize=popsize, n_refine=n_refine,
        )
        residual -= fit['model_flux']
        cumulative += fit['model_flux']
        prefix = f"{source_name}_{stage['name']}"
        saved = save_fit_outputs(fit, output_dir, prefix)
        saved['running'] = save_running_spectrum(
            obs_wav, residual, cumulative, output_dir, prefix,
        )
        error_rows = []
        for label, param in fit['params'].items():
            row = {'stage': stage['name'], 'component': label,
                   'model_mol': param['model_mol'],
                   'T': param['T'], 'logN': param['logN'],
                   'A': param['A'], 'log10A': param['log10A'],
                   **fit['uncertainty_details'][label]}
            error_rows.append(row)
            summary_rows.append(row)
        uncertainty_path = output_dir / f'{prefix}_grid_uncertainty.csv'
        pd.DataFrame(error_rows).to_csv(uncertainty_path, index=False)
        saved['grid_uncertainty'] = str(uncertainty_path)
        print_fit_params(fit)
        fits[stage['name']] = fit
        products[stage['name']] = saved
    if fits:
        products['final_running'] = save_running_spectrum(
            obs_wav, residual, cumulative, output_dir,
            f'{source_name}_final',
        )
        summary_path = output_dir / f'{source_name}_grid_summary.csv'
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        products['summary'] = str(summary_path)
    print('Grid fitting done.')
    return {'fits': fits, 'obs_wav': obs_wav, 'obs_flux': obs_flux,
            'residual': residual, 'cumulative_model': cumulative,
            'sigma_noise': sigma_noise, 'products': products}
