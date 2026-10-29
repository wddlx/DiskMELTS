"""Fit observations directly with the v2 slab CSV grids.

This workflow reads ``<grid_root>/<molecule>/T...N....csv`` (currently
``Model_grids/sigma0/<molecule>/``) and never loads a
trained surrogate. It interpolates flux linearly between adjacent T/logN grid
points, fits positive component amplitudes, and samples the posterior by
MCMC (``diskmelts.uncertainty``). The reported T and logN errors include the
*full local grid step* in quadrature: ``sqrt(fit_sigma**2 + grid_step**2)``.

``fit_grid_stage`` fits one or several molecules. ``fit_observation_grids``
runs sequential stages and writes the same stage-spectrum/parameter/running
CSV format used by ``diskmelts.v2_read_plot``. The sampler starts at the best
fit, so it describes that mode; a distant second mode would need its own run.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, least_squares, lsq_linear

from diskmelts.uncertainty import (
    default_mcmc_steps, likelihood_sigma, sample_posterior, summarize_parameter,
)
from diskmelts.fitting import (
    _estimate_noise,
    _read_observation,
    _screened_group_options,
    _tie_layout,
    LINE_FREE_WINDOWS,
    detect_stage_molecules,
    print_fit_params,
    save_fit_outputs,
    save_running_spectrum,
)


GRID_ROOT = None
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
    if not molecules:
        raise ValueError('molecule must contain at least one molecular name')
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


def _bracket_many(axis, values):
    """Vectorized _bracket for points already inside the loaded axis."""
    values = np.asarray(values, dtype=float)
    lower = np.clip(np.searchsorted(axis, values, side='right') - 1, 0, len(axis) - 2)
    fraction = (values - axis[lower]) / (axis[lower + 1] - axis[lower])
    return lower, np.clip(fraction, 0.0, 1.0)


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

    def evaluate_batch(self, temperatures, log_columns):
        """Bilinear flux on the fit pixels for many points, shape (n, n_pixels)."""
        i, ft = _bracket_many(self.temperatures, temperatures)
        j, fn = _bracket_many(self.log_columns, log_columns)
        f = self.fit_flux
        return (((1 - ft) * (1 - fn))[:, None] * f[i, j] +
                (ft * (1 - fn))[:, None] * f[i + 1, j] +
                ((1 - ft) * fn)[:, None] * f[i, j + 1] +
                (ft * fn)[:, None] * f[i + 1, j + 1])

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


def fit_grid_stage(obs_wav, obs_flux, molecule, wavelength_range, fit_ranges, *,
                   grid_root=GRID_ROOT, h2o_components=1, component_names=None,
                   T_bounds=(100.0, 1400.0), logN_bounds=None,
                   loga_bounds=(-2.0, 2.0), sigma=None, distance_pc=140.0,
                   isotopologue_groups=None, Tshare=None, Ashare=None,
                   shared_area_groups=None, seed=42,
                   maxiter=100, popsize=8, n_refine=3, n_mcmc=None,
                   errorbars=True):
    """Fit one or more slab-grid molecules with positive amplitudes.

    Differential evolution searches nonlinear T/logN space while amplitudes
    are solved by bounded least squares. A few best candidates are refined by
    weighted nonlinear least squares. Errors come from ``n_mcmc`` MCMC steps of
    the posterior on the interpolated grid (see ``diskmelts.uncertainty``);
    the T/logN intervals then include their grid steps in quadrature.
    ``errorbars=False`` skips sampling and reports best fits only.
    ``distance_pc`` converts the 140-pc grid flux normalization to the source
    distance and makes fitted amplitudes physical emitting areas in au².
    ``isotopologue_groups``, ``Tshare``, ``Ashare``, and ``shared_area_groups``
    tie temperatures and areas exactly as in ``fit_molecules``. Tied
    components share free coordinates, so their posterior samples respect
    the ties.
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
    if any(len(pair) != 2 or not np.all(np.isfinite(pair)) or pair[0] >= pair[1]
           for bound in bounds for pair in bound):
        raise ValueError('T and logN bounds must be finite and increasing')
    layout = _tie_layout(parts, bounds, isotopologue_groups, Tshare, Ashare,
                         shared_area_groups)
    decode = layout['decode']
    nonlinear_bounds = layout['nonlinear_bounds']
    groups = layout['area_group_members']
    area_bounds = [bound_for(loga_bounds, part, (-2.0, 2.0))
                   for part in parts]
    if any(len(pair) != 2 or not np.all(np.isfinite(pair)) or
           pair[0] >= pair[1] or not 0 < 10.0 ** pair[0] < 10.0 ** pair[1]
           for pair in area_bounds):
        raise ValueError('loga_bounds must be increasing and finite')
    # A shared area must satisfy every member's bounds.
    group_area_bounds = [(max(area_bounds[k][0] for k in members),
                          min(area_bounds[k][1] for k in members))
                         for members in groups]
    if any(lo >= hi for lo, hi in group_area_bounds):
        raise ValueError('shared area group has no overlap in loga_bounds')
    area_lower = np.array([10.0 ** lo for lo, _ in group_area_bounds])
    area_upper = np.array([10.0 ** hi for _, hi in group_area_bounds])
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

    n_parts = len(parts)
    n_free = len(nonlinear_bounds)
    water_indices = [k for k, part in enumerate(parts)
                     if part['model_mol'] == 'H2O']
    ordered_water = h2o_components == 2 and len(water_indices) == 2

    def water_misordered(physical):
        return (ordered_water and
                physical[2 * water_indices[0]] > physical[2 * water_indices[1]])

    def templates(physical):
        return np.column_stack([
            libraries[part['model_mol']].evaluate(physical[2 * k], physical[2 * k + 1])
            for k, part in enumerate(parts)
        ])

    def expand_areas(group_values):
        areas = np.empty(n_parts)
        for members, value in zip(groups, group_values):
            areas[members] = value
        return areas

    def profile_amplitudes(free):
        physical = decode(free)
        if water_misordered(physical):
            return np.inf, area_lower.copy()
        matrix = templates(physical)
        grouped = np.column_stack([matrix[:, members].sum(axis=1)
                                   for members in groups])
        scale = max(float(np.max(np.abs(target))),
                    float(np.max(np.abs(grouped))), np.finfo(float).tiny)
        if grouped.shape[1] == 1:
            column = grouped[:, 0] / scale
            denom = float(np.dot(column, column))
            amplitude = (float(np.clip(np.dot(column, target / scale) / denom,
                                       area_lower[0], area_upper[0]))
                         if denom > 0 else area_lower[0])
            amplitudes = np.array([amplitude])
        else:
            solution = lsq_linear(grouped / scale, target / scale,
                                  bounds=(area_lower, area_upper))
            amplitudes = solution.x
        residual = grouped @ amplitudes - target
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
    # Refinement coordinates: the free nonlinear ones, then one log10 area
    # per area group.
    solution_bounds = list(nonlinear_bounds) + group_area_bounds
    lower = np.array([item[0] for item in solution_bounds], dtype=float)
    upper = np.array([item[1] for item in solution_bounds], dtype=float)
    mid_physical = decode(np.mean([lower[:n_free], upper[:n_free]], axis=0))
    scales = np.full(len(solution_bounds), 0.5)
    for k, part in enumerate(parts):
        library = libraries[part['model_mol']]
        T_step = _local_step(library.temperatures, mid_physical[2 * k])
        N_step = _local_step(library.log_columns, mid_physical[2 * k + 1])
        scales[layout['column_indices'][k]] = N_step
        j = layout['temperature_indices'].get(k)
        if j is None:
            continue
        relation = layout['group_for'].get(k)
        if relation is None or relation[0] == k:
            scales[j] = T_step
        else:
            # A 'less' coordinate is a fraction of the allowed T interval.
            scales[j] = min(0.5, T_step / (bounds[k][0][1] - bounds[k][0][0]))

    def physical_vector(z):
        """Per-component (T, logN, log10A) from refinement coordinates."""
        physical = decode(z[:n_free])
        log_area = expand_areas(z[n_free:])
        return np.array([value for k in range(n_parts)
                         for value in (physical[2 * k], physical[2 * k + 1],
                                       log_area[k])])

    def residual_vector(z):
        physical = decode(z[:n_free])
        if water_misordered(physical):
            # Keep warm/hot labels ordered during local refinement.
            return np.full_like(target, 1e6)
        model = templates(physical) @ (10.0 ** expand_areas(z[n_free:]))
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
        z0 = np.concatenate([candidate, np.log10(amplitudes)])
        z0 = np.clip(z0, lower + 1e-10, upper - 1e-10)
        opt = least_squares(residual_vector, z0, bounds=(lower, upper),
                            x_scale=scales, max_nfev=600)
        loss = float(np.dot(opt.fun, opt.fun))
        if loss < best_loss:
            best, best_loss = opt, loss
        if len(seen) >= n_refine:
            break
    if best is None:
        raise RuntimeError('grid optimization returned no solution')

    best_physical = physical_vector(best.x)
    # Posterior sampling of the free coordinates (fixed ones stay at the best
    # fit); intervals are then widened by the local grid step.
    fixed = upper - lower < 1e-4
    free_idx = np.flatnonzero(~fixed)
    sigma_like, uncertainty_info = likelihood_sigma(
        best.fun * noise_scale, None if sigma is None else noise_scale,
        n_params=len(free_idx), flux_scale=np.max(np.abs(target)),
    )
    posterior = None
    n_mcmc = default_mcmc_steps(n_mcmc)
    if errorbars and n_mcmc > 0 and len(free_idx):
        def residual_free(values):
            z = best.x.copy()
            z[free_idx] = values
            return residual_vector(z) * (noise_scale / sigma_like)

        def residual_free_batch(values):
            Z = np.repeat(best.x[None, :], len(values), axis=0)
            Z[:, free_idx] = values
            physical = np.array([decode(z[:n_free]) for z in Z])
            areas = 10.0 ** np.array([expand_areas(z[n_free:]) for z in Z])
            model = np.zeros((len(Z), len(target)))
            for k, part in enumerate(parts):
                model += areas[:, k, None] * libraries[part['model_mol']].evaluate_batch(
                    physical[:, 2 * k], physical[:, 2 * k + 1])
            R = (model - target) / sigma_like
            misordered = np.array([water_misordered(p) for p in physical])
            R[misordered] = np.inf
            return R

        less_members = [(i, main) for i, (main, mode) in layout['group_for'].items()
                        if i != main and mode == 'less']
        def physical_prior(values):
            z = best.x.copy()
            z[free_idx] = values
            physical = decode(z[:n_free])
            correction = 0.0
            for i, main in less_members:
                lo, hi = bounds[i][0]
                width = min(hi, physical[2 * main]) - lo
                if width < 0:
                    return -np.inf
                correction += np.log(max(width, 1e-9 * (hi - lo)))
            return correction

        free_samples, sampler_info = sample_posterior(
            residual_free, best.x[free_idx], lower[free_idx], upper[free_idx],
            n_steps=n_mcmc, seed=seed, batch_residual_fn=residual_free_batch,
            extra_log_prob=physical_prior if less_members else None,
        )
        uncertainty_info.update(sampler_info, method='mcmc')
        full = np.repeat(best.x[None, :], len(free_samples), axis=0)
        full[:, free_idx] = free_samples
        posterior = np.array([physical_vector(z) for z in full])
        print(f"  MCMC: {sampler_info['n_steps']} steps, acceptance "
              f"{sampler_info['acceptance']:.2f}, R-hat {sampler_info['r_hat_max']:.3f}, "
              f"noise correlation length {uncertainty_info['correlation_length']:.1f} px")
        if not sampler_info['converged']:
            warnings.warn(f"grid MCMC has not converged (R-hat "
                          f"{sampler_info['r_hat_max']:.2f}) after {sampler_info['n_steps']} "
                          f"likelihood evaluations; increase n_mcmc", stacklevel=2)
    else:
        uncertainty_info['method'] = 'none'
    params, uncertainty, details, component_fluxes = {}, {}, {}, {}
    model_flux = np.zeros_like(flux)
    plot_mask = np.logical_or.reduce([(wav >= lo) & (wav <= hi)
                                      for lo, hi in plot_ranges])
    for k, part in enumerate(parts):
        label, mol = part['label'], part['model_mol']
        T, logN, logA = map(float, best_physical[3 * k:3 * k + 3])
        A = 10.0 ** logA
        component = A / to_reference * libraries[mol].evaluate(T, logN, obs_wav=wav)
        component[~plot_mask] = 0.0
        component_fluxes[label] = component
        model_flux += component
        params[label] = {'T': T, 'logN': logN, 'A': A,
                         'log10A': logA, 'model_mol': mol}
        steps = {'T': _local_step(libraries[mol].temperatures, T),
                 'logN': _local_step(libraries[mol].log_columns, logN)}
        limits = {'T': bounds[k][0], 'logN': bounds[k][1], 'log10A': area_bounds[k]}
        best_values = {'T': T, 'logN': logN, 'log10A': logA}
        uncertainty[label], details[label] = {}, {}
        for j, key in enumerate(('T', 'logN', 'log10A')):
            if posterior is None:
                summary = {'minus': np.nan, 'plus': np.nan, 'at_bound': False}
            else:
                summary = summarize_parameter(posterior[:, 3 * k + j],
                                              best_values[key], limits[key])
            fit_sigma = 0.5 * (summary.get('q84', np.nan) - summary.get('q16', np.nan))
            on_bound = bool(np.isclose(best_values[key], limits[key][0], rtol=0, atol=1e-6) or
                            np.isclose(best_values[key], limits[key][1], rtol=0, atol=1e-6))
            summary['at_bound'] = bool(summary['at_bound'] or on_bound)
            details[label][f'{key}_fit_sigma'] = fit_sigma
            if key in steps:
                # The grid step is added in quadrature to each side.
                step = steps[key]
                summary['minus'] = combine_grid_uncertainty(summary['minus'], step)
                summary['plus'] = combine_grid_uncertainty(summary['plus'], step)
                details[label][f'{key}_grid_step'] = step
                details[label][f'{key}_total_sigma'] = combine_grid_uncertainty(fit_sigma, step)
            details[label][f'{key}_at_bound'] = summary['at_bound']
            uncertainty[label][key] = summary
        if posterior is not None:
            uncertainty[label]['logNA'] = summarize_parameter(
                posterior[:, 3 * k + 1] + posterior[:, 3 * k + 2], logN + logA)
    print(f'  Grid fit: {len(parts)} components, weighted SSE={best_loss:.4g}')
    return {
        'mol': molecule, 'components': parts, 'params': params,
        'uncertainty': uncertainty, 'uncertainty_details': details,
        'uncertainty_info': uncertainty_info,
        'obs_wav': wav, 'obs_flux': flux.copy(), 'model_flux': model_flux,
        'component_fluxes': component_fluxes, 'residual': flux - model_flux,
        'fit_ranges': selected_ranges,
        'top_list': [{'rank': 1, 'mse_loss': best_loss * (noise_scale / to_reference) ** 2 / len(target),
                      'params': params}],
        'grid_root': str(grid_root), 'sigma_noise': sigma,
        'distance_pc': float(distance_pc),
        'isotopologue_groups': layout['iso_groups'], 'Tshare': layout['modes'],
        'Ashare': layout['area_flags'],
        'shared_area_groups': layout['all_area_groups'],
    }


def fit_observation_grids(input_path=None, stages=None, *, name=None, grid_root=GRID_ROOT,
                          output_dir=None, line_free_windows=LINE_FREE_WINDOWS,
                          detection_screening=True, detection_sigma_factor=3.0,
                          sigma_noise=None, distance_pc=140.0,
                          seed=42, maxiter=100, popsize=8,
                          n_refine=3, n_mcmc=None, errorbars=True,
                          skiprows=1, delimiter=',',
                          wav_col=0, flux_col=1,
                          obs_wav=None, obs_flux=None, obs_err=None):
    """Run ordered direct-grid stages and save the same CSVs as ``fit_observation``.

    This is the slab-grid counterpart of ``fit_observation``: it takes the
    same stage dictionaries and input options, but evaluates spectra by
    interpolating the local grids under ``grid_root`` instead of the bundled
    neural models. Give the spectrum as ``input_path`` (a CSV read with
    ``load_observed_spectrum``) or as ``obs_wav``/``obs_flux`` arrays with
    optional ``obs_err``. ``sigma_noise`` overrides the noise estimate; else
    the median of ``obs_err`` is used, else the scatter in
    ``line_free_windows``.

    ``errorbars`` (default True) computes MCMC error bars for each stage,
    starting with ``n_mcmc`` likelihood evaluations (default 300000 or
    ``DISKMELTS_N_MCMC``); ``errorbars=False`` reports best fits only.
    """
    if grid_root is None:
        raise ValueError('grid_root is required for direct-grid fitting')
    obs_wav, obs_flux, obs_err, source_name = _read_observation(
        input_path, obs_wav, obs_flux, obs_err, name=name, skiprows=skiprows,
        delimiter=delimiter, wav_col=wav_col, flux_col=flux_col,
    )
    stages = list(stages or ())
    if not stages or len({stage['name'] for stage in stages}) != len(stages):
        raise ValueError('stages must be nonempty and have unique names')
    output_dir = (Path(output_dir) if output_dir else
                  Path.cwd() / 'realobs_results' / 'grids' / source_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    if sigma_noise is None:
        sigma_noise = _estimate_noise(obs_wav, obs_flux, obs_err, line_free_windows)
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
            **_screened_group_options(stage, detected, _components),
            T_bounds=stage.get('T_bounds', (100.0, 1400.0)),
            logN_bounds=stage.get('logN_bounds',
                                  {mol: ((13.0, 19.0) if mol == 'H2O'
                                         else (13.0, 19.5))
                                   for mol in ([selected] if isinstance(selected, str)
                                               else selected)}),
            loga_bounds=stage.get('loga_bounds', (-2.0, 2.0)),
            sigma=sigma_noise, distance_pc=distance_pc,
            seed=seed, maxiter=maxiter,
            popsize=popsize, n_refine=n_refine, n_mcmc=n_mcmc, errorbars=errorbars,
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
            # Posterior medians as in the params CSV; best fits as *_best.
            u = fit['uncertainty'][label]
            median = {key: u[key].get('q50', np.nan) for key in ('T', 'logN', 'log10A')}
            median = {key: value if np.isfinite(value) else param[key]
                      for key, value in median.items()}
            row = {'stage': stage['name'], 'component': label,
                   'model_mol': param['model_mol'],
                   'T': median['T'], 'logN': median['logN'],
                   'A': 10.0 ** median['log10A'], 'log10A': median['log10A'],
                   'T_best': param['T'], 'logN_best': param['logN'],
                   'A_best': param['A'], 'log10A_best': param['log10A'],
                   **{f'{key}_{side}': u[key][side]
                      for key in ('T', 'logN', 'log10A') for side in ('minus', 'plus')},
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
            'obs_err': obs_err, 'residual': residual, 'cumulative_model': cumulative,
            'sigma_noise': sigma_noise, 'products': products}
