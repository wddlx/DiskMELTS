"""
diskmelts/uncertainty.py — Posterior parameter uncertainties by MCMC.

The fitters find a best fit by optimization; this module then samples the
likelihood around it, so every reported interval comes from the posterior
rather than from the spread of optimizer solutions.

Likelihood. Pixels are compared with a single noise level, inflated for two
effects that a per-pixel chi-square ignores:

* correlated noise — MIRI spectra are oversampled and resampled, so adjacent
  residuals are correlated. The integrated autocorrelation length ``tau`` of
  the best-fit residual (summed over lags of at most 10 pixels) counts how
  many pixels make one independent sample, and the noise is scaled by
  ``sqrt(tau)``;
* model mismatch — if the best-fit reduced chi-square exceeds 1, the noise is
  scaled by ``sqrt(chi2_red)`` (never shrunk below the supplied noise).

With no supplied noise, the best-fit residual RMS is used. Priors are flat
in temperature and logN inside the fit bounds and flat in log10(A) (an
unbounded area uses at least -6 to 6, widened to cover the best fit by 4 dex).
Isotopologue groups with ``Tshare='less'`` add a Jacobian term, so the prior
stays flat in physical temperature. The posterior is sampled with the emcee ensemble
sampler. The reported value is the posterior median and its error bars reach
the 16th and 84th percentiles. The best fit (the single best-fitting model,
which the plotted spectra use) is kept separately: in degenerate
temperature–column–area spaces it can lie well off the median, because most
of the probability sits elsewhere along the degeneracy. Parameters whose
interval reaches a bound, or whose best fit sits on one, are flagged
``at_bound``, because the bound rather than the data then limits them. Convergence is checked
with R-hat (comparing walkers): a run that has not reached 1.1 is doubled,
up to three times, and is flagged if it still has not converged.
"""

import logging
import os

import numpy as np


def default_mcmc_steps(n_mcmc=None):
    """Resolve the MCMC step count (argument, else DISKMELTS_N_MCMC, else 300000)."""
    if n_mcmc is not None:
        return int(n_mcmc)
    return int(os.environ.get('DISKMELTS_N_MCMC', '300000'))


def noise_correlation_length(residual, max_lag=10):
    """Integrated autocorrelation length of a residual, in pixels (>= 1).

    Only lags up to ``max_lag`` pixels are summed: that covers the pixel-scale
    correlation from oversampling and resampling, while slower residual
    structure from imperfect line fits is model mismatch, not noise, and is
    handled by the reduced chi-square inflation instead.
    """
    r = np.asarray(residual, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return 1.0
    r = r - r.mean()
    variance = float(np.dot(r, r))
    if variance <= 0:
        return 1.0
    tau = 1.0
    # Sum the initial positive autocorrelations (Geyer's initial sequence).
    for lag in range(1, min(max_lag, r.size // 4) + 1):
        rho = float(np.dot(r[:-lag], r[lag:])) / variance
        if rho <= 0:
            break
        tau += 2.0 * rho
    return tau


def likelihood_sigma(residual, sigma=None, n_params=0, flux_scale=0.0):
    """Return the likelihood noise level and a diagnostics dict.

    ``residual`` is the best-fit residual on the fitted pixels and ``sigma``
    the supplied per-pixel noise (or None to use the residual RMS). The noise
    is never taken below ``1e-5 * flux_scale`` (the peak fitted flux), about
    the surrogate networks' float32 precision, so noise-free synthetic spectra
    still give a well-defined posterior.
    """
    r = np.asarray(residual, dtype=np.float64)
    dof = max(1, r.size - int(n_params))
    rms = float(np.sqrt(np.sum(r ** 2) / dof))
    tau = noise_correlation_length(r)
    if sigma is not None and sigma > 0:
        chi2_red = (rms / sigma) ** 2
        base = float(sigma) * np.sqrt(max(1.0, chi2_red))
    else:
        chi2_red = float('nan')
        base = rms
    base = max(base, 1e-5 * float(flux_scale), np.finfo(float).tiny)
    sigma_like = base * np.sqrt(tau)
    return sigma_like, {
        'sigma_input': None if sigma is None else float(sigma),
        'residual_rms': rms,
        'chi2_red': chi2_red,
        'correlation_length': tau,
        'sigma_likelihood': float(sigma_like),
    }


def _laplace_unit_covariance(residual_fn, u0, span, lower):
    """Inverse Gauss-Newton Hessian of 0.5*|r|^2 in unit coordinates."""
    d = len(u0)
    step = 1e-4
    columns = []
    for j in range(d):
        hi = u0.copy()
        lo = u0.copy()
        hi[j] = min(1.0, u0[j] + step)
        lo[j] = max(0.0, u0[j] - step)
        width = hi[j] - lo[j]
        r_hi = residual_fn(lower + hi * span)
        r_lo = residual_fn(lower + lo * span)
        if width <= 0 or not (np.all(np.isfinite(r_hi)) and np.all(np.isfinite(r_lo))):
            columns.append(np.zeros_like(r_hi))
        else:
            columns.append((r_hi - r_lo) / width)
    J = np.column_stack(columns)
    cov = np.linalg.pinv(J.T @ J)
    # Keep the starting proposal sane even for unconstrained directions.
    variances = np.clip(np.diag(cov), 1e-10, 0.05)
    corr = cov / np.sqrt(np.outer(np.maximum(np.diag(cov), 1e-30),
                                  np.maximum(np.diag(cov), 1e-30)))
    corr = np.nan_to_num(np.clip(corr, -1, 1))
    np.fill_diagonal(corr, 1.0)
    out = corr * np.sqrt(np.outer(variances, variances))
    # Guard against a non-positive-definite estimate.
    eigvals, eigvecs = np.linalg.eigh(out)
    return (eigvecs * np.maximum(eigvals, 1e-12)) @ eigvecs.T


def sample_posterior(residual_fn, x0, lower, upper, *, n_steps=300000,
                     seed=None, burn_fraction=0.5, extra_log_prob=None,
                     batch_residual_fn=None, max_extensions=3, max_samples=40000):
    """Ensemble MCMC (emcee) sampling of exp(-0.5*|r(x)|^2) inside bounds.

    ``residual_fn(x)`` returns the noise-normalized residual vector (non-finite
    values mark a forbidden point). ``batch_residual_fn(X)``, if given, does the
    same for a 2D array of points and is used to evaluate all walkers at once.
    ``n_steps`` is the initial number of likelihood evaluations, spread over
    the walkers. If the walkers have not converged (R-hat >= 1.1), the run is
    doubled up to ``max_extensions`` times (8x the budget by default), so
    stages with many parameters get longer chains automatically. The first
    ``burn_fraction`` of each walker is discarded, and at most ``max_samples``
    evenly thinned samples are returned.
    Differential-evolution moves follow the long, curved ridges typical of
    temperature/column-density/area degeneracies.

    Returns ``(samples, info)`` with the retained samples in physical
    coordinates.
    """
    import emcee

    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    span = upper - lower
    d = len(lower)
    if np.any(span <= 0):
        raise ValueError('sample_posterior needs increasing bounds')
    rng = np.random.default_rng(seed)

    def _extra(x):
        return 0.0 if extra_log_prob is None else float(extra_log_prob(x))

    def log_prob_batch(U):
        U = np.atleast_2d(U)
        out = np.full(len(U), -np.inf)
        valid = np.all((U >= 0) & (U <= 1), axis=1)
        if not np.any(valid):
            return out
        X = lower + U[valid] * span
        if batch_residual_fn is not None:
            R = np.atleast_2d(batch_residual_fn(X))
        else:
            R = np.array([residual_fn(x) for x in X])
        finite = np.all(np.isfinite(R), axis=1)
        values = np.where(finite, -0.5 * np.sum(np.where(np.isfinite(R), R, 0.0) ** 2, axis=1),
                          -np.inf)
        if extra_log_prob is not None:
            values = values + np.array([_extra(x) for x in X])
        out[valid] = values
        return out

    u0 = np.clip((np.asarray(x0, dtype=np.float64) - lower) / span, 0.0, 1.0)
    if not np.isfinite(log_prob_batch(u0[None, :])[0]):
        raise ValueError('best fit has non-finite posterior probability')

    n_walkers = max(32, 4 * d)
    n_walkers += n_walkers % 2
    base_steps = max(200, int(np.ceil(int(n_steps) / n_walkers)))
    # Store a thinned chain so long runs stay small in memory.
    thin = max(1, (base_steps * 2 ** max_extensions) // 4000)

    # Start the walkers in a small ball shaped by the local covariance.
    cov = _laplace_unit_covariance(residual_fn, u0, span, lower)
    chol = np.linalg.cholesky(cov + 1e-14 * np.eye(d))
    p0 = np.empty((n_walkers, d))
    for k in range(n_walkers):
        for _ in range(100):
            trial = u0 + 0.1 * chol @ rng.standard_normal(d)
            trial = np.abs(trial)
            trial = 1.0 - np.abs(1.0 - trial)  # reflect into [0, 1]
            if np.isfinite(log_prob_batch(trial[None, :])[0]):
                break
        else:
            trial = np.clip(u0 + 1e-6 * rng.standard_normal(d), 0.0, 1.0)
        p0[k] = trial

    sampler = emcee.EnsembleSampler(
        n_walkers, d, log_prob_batch, vectorize=True,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
    )
    sampler.random_state = np.random.RandomState(
        int(rng.integers(0, 2 ** 31 - 1))).get_state()
    def advance(state, steps):
        return sampler.run_mcmc(state, max(1, int(np.ceil(steps / thin))),
                                thin_by=thin, progress=False)

    state = advance(p0, base_steps)
    total_steps, extensions = base_steps, 0
    while True:
        chain = sampler.get_chain()
        chain = chain[int(len(chain) * burn_fraction):]  # (steps, walkers, d)
        r_hat = _walker_r_hat(chain)
        if r_hat < 1.1 or extensions >= max_extensions:
            break
        state = advance(state, total_steps)  # double the run
        total_steps *= 2
        extensions += 1
    # integrated_time logs a "chain is too short" warning that is only a
    # diagnostic here; convergence is judged by R-hat.
    autocorr_logger = logging.getLogger('emcee.autocorr')
    previous_level = autocorr_logger.level
    autocorr_logger.setLevel(logging.ERROR)
    try:
        tau = float(np.max(emcee.autocorr.integrated_time(chain, quiet=True))) * thin
    except Exception:
        tau = float('nan')
    finally:
        autocorr_logger.setLevel(previous_level)
    kept = chain.reshape(-1, d)
    if len(kept) > max_samples:
        kept = kept[np.linspace(0, len(kept) - 1, max_samples).astype(int)]
    info = {
        'n_steps': int(total_steps * n_walkers),
        'n_walkers': int(n_walkers),
        'n_extensions': int(extensions),
        'n_kept': int(len(kept)),
        'acceptance': float(np.mean(sampler.acceptance_fraction)),
        'r_hat_max': r_hat,
        'autocorr_steps': tau,
        'converged': bool(r_hat < 1.1),
    }
    return lower + kept * span, info


def _walker_r_hat(chain):
    """Gelman-Rubin statistic treating each walker's retained path as a chain."""
    chain = np.asarray(chain, dtype=np.float64)
    n, m = chain.shape[:2]
    if n < 10 or m < 2:
        return float('nan')
    means = chain.mean(axis=0)                    # (walkers, d)
    within = chain.var(axis=0, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    variance = (n - 1) / n * within + between / n
    with np.errstate(divide='ignore', invalid='ignore'):
        r_hat = np.sqrt(np.where(within > 0, variance / within, np.inf))
    return float(np.nanmax(r_hat))


def split_r_hat(chain, n_segments=4):
    """Largest split-chain Gelman-Rubin statistic over parameters.

    The retained chain is cut into segments that are compared as if they were
    independent chains; values above about 1.1 mean the chain has not settled.
    """
    chain = np.asarray(chain, dtype=np.float64)
    n = len(chain) // n_segments
    if n < 10:
        return float('nan')
    segments = chain[:n * n_segments].reshape(n_segments, n, -1)
    means = segments.mean(axis=1)
    within = segments.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    variance = (n - 1) / n * within + between / n
    with np.errstate(divide='ignore', invalid='ignore'):
        r_hat = np.sqrt(np.where(within > 0, variance / within, np.inf))
    return float(np.nanmax(r_hat))


def summarize_parameter(samples, best, bounds=None):
    """Posterior median and 16-84% range, the best fit, and a bound flag.

    ``minus``/``plus`` are measured from the median, so they are zero only when
    the posterior itself has no width.
    """
    samples = np.asarray(samples, dtype=np.float64)
    q16, q50, q84 = (float(v) for v in np.percentile(samples, [16, 50, 84]))
    at_bound = False
    if bounds is not None and np.all(np.isfinite(bounds)):
        lo, hi = map(float, bounds)
        tolerance = 0.01 * (hi - lo)
        # The interval reaches a bound, or the best fit itself sits on one.
        at_bound = bool(q16 - lo <= tolerance or hi - q84 <= tolerance or
                        min(abs(best - lo), abs(hi - best)) <= 1e-6 * (hi - lo))
    return {
        'q16': q16, 'q50': q50, 'q84': q84,
        'minus': q50 - q16, 'plus': q84 - q50,
        'best': float(best), 'at_bound': at_bound,
    }
