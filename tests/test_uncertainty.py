"""Posterior error bars: sampler accuracy, noise handling, coverage, and outputs."""

import numpy as np
import pandas as pd
import pytest


pytest.importorskip('torch')
from diskmelts import fitting
from diskmelts.uncertainty import (
    _walker_r_hat,
    likelihood_sigma,
    noise_correlation_length,
    sample_posterior,
    split_r_hat,
    summarize_parameter,
)


def test_frozen_disagreeing_chains_are_not_converged():
    walkers = np.broadcast_to(np.arange(32)[None, :, None], (100, 32, 1))
    assert np.isinf(_walker_r_hat(walkers))
    segments = np.repeat(np.arange(4), 40)[:, None]
    assert np.isinf(split_r_hat(segments))


def test_sampler_recovers_a_known_gaussian():
    mu, sd = np.array([3.0, -1.0]), np.array([0.2, 0.05])
    samples, info = sample_posterior(lambda x: (x - mu) / sd, mu, mu - 10 * sd, mu + 10 * sd,
                                     n_steps=20000, seed=3)
    assert 0.1 < info['acceptance'] < 0.9 and info['converged']
    np.testing.assert_allclose(samples.mean(axis=0), mu, atol=0.2 * sd.max())
    np.testing.assert_allclose(samples.std(axis=0), sd, rtol=0.15)


def test_split_r_hat_flags_a_drifting_chain():
    rng = np.random.default_rng(4)
    settled = rng.normal(size=(8000, 2))
    drifting = settled + np.linspace(0, 5, 8000)[:, None]
    assert split_r_hat(settled) < 1.05
    assert split_r_hat(drifting) > 1.1


def test_correlation_length_detects_correlated_noise():
    rng = np.random.default_rng(0)
    white = rng.normal(size=4000)
    smooth = np.convolve(white, np.ones(5) / 5, mode='same')
    assert noise_correlation_length(white) < 1.3
    assert 3.0 < noise_correlation_length(smooth) < 8.0


def test_likelihood_sigma_inflates_for_misfit_but_never_shrinks():
    rng = np.random.default_rng(1)
    residual = rng.normal(scale=2.0, size=3000)
    inflated, info = likelihood_sigma(residual, sigma=1.0)
    assert info['chi2_red'] == pytest.approx(4.0, rel=0.1)
    assert inflated == pytest.approx(2.0 * np.sqrt(info['correlation_length']), rel=0.1)
    kept, info = likelihood_sigma(residual, sigma=5.0)
    assert kept == pytest.approx(5.0 * np.sqrt(info['correlation_length']))
    estimated, info = likelihood_sigma(residual, sigma=None)
    assert estimated == pytest.approx(2.0 * np.sqrt(info['correlation_length']), rel=0.1)


def test_summary_reports_median_range_and_flags_bounds():
    rng = np.random.default_rng(2)
    summary = summarize_parameter(rng.normal(10, 1, 20000), 10.2, (0, 20))
    assert summary['q50'] == pytest.approx(10, abs=0.05)
    assert summary['minus'] == pytest.approx(1, abs=0.05)
    assert summary['plus'] == pytest.approx(1, abs=0.05)
    assert summary['best'] == 10.2 and not summary['at_bound']
    # A best fit outside the bulk of the posterior still gets nonzero errors.
    piled = summarize_parameter(rng.uniform(0, 0.5, 20000), 0.0, (0, 20))
    assert piled['at_bound'] and piled['minus'] > 0.1 and piled['plus'] > 0.1


def _gaussian_line(monkeypatch):
    def fake_flux(model, T, logN, A, obs_wav):
        # Curve of growth: optical depth flattens the core, so logN sets the
        # line shape and all three parameters are identifiable.
        width = 0.02 + 0.06 * (T - 200.0) / 600.0
        tau = 10 ** (logN - 16.0) * np.exp(-0.5 * ((obs_wav - 13.7) / width) ** 2)
        return A * (1.0 - np.exp(-tau))

    monkeypatch.setattr(fitting, '_mol_flux_on_wav', fake_flux)
    return fake_flux


def test_error_bars_are_positive_and_calibrated(monkeypatch):
    fake_flux = _gaussian_line(monkeypatch)
    wave = np.linspace(13.3, 14.1, 160)
    truth = {'T': 450.0, 'logN': 16.5, 'log10A': np.log10(0.5)}
    noise = 0.01
    rng = np.random.default_rng(11)
    covered = dict.fromkeys(truth, 0)
    trials = 40
    for trial in range(trials):
        observed = fake_flux(None, truth['T'], truth['logN'], 0.5, wave)
        observed = observed + rng.normal(scale=noise, size=wave.size)
        fit = fitting.fit_molecules(
            wave, observed, 'HCN', {'HCN': {'mol': 'HCN'}}, (13.3, 14.1),
            T_bounds=(200, 800), logN_bounds=(14, 18), loga_bounds=(-2, 1),
            n_samples=64, n_refine=2, n_top=2, sigma=noise, seed=trial,
            n_mcmc=20000, verbose=False,
        )
        assert fit['uncertainty_info']['converged']
        for key, value in truth.items():
            u = fit['uncertainty']['HCN'][key]
            assert u['minus'] > 0 and u['plus'] > 0  # never a zero error bar
            covered[key] += u['q16'] <= value <= u['q84']
    # A 68% interval should contain the truth in about 27 of 40 trials.
    for key, count in covered.items():
        assert 19 <= count <= 35, (key, count)


def test_correlated_noise_widens_the_interval(monkeypatch):
    fake_flux = _gaussian_line(monkeypatch)
    wave = np.linspace(13.3, 14.1, 400)
    rng = np.random.default_rng(5)
    white = rng.normal(scale=0.01, size=wave.size)
    smooth = np.convolve(rng.normal(scale=0.01 * np.sqrt(7), size=wave.size),
                         np.ones(7) / 7, mode='same')
    widths = []
    for noise in (white, smooth):
        observed = fake_flux(None, 450.0, 16.5, 0.5, wave) + noise
        fit = fitting.fit_molecules(
            wave, observed, 'HCN', {'HCN': {'mol': 'HCN'}}, (13.3, 14.1),
            T_bounds=(200, 800), logN_bounds=(14, 18), loga_bounds=(-2, 1),
            n_samples=64, n_refine=2, n_top=2, sigma=0.01, seed=1,
            n_mcmc=20000, verbose=False,
        )
        u = fit['uncertainty']['HCN']['T']
        widths.append(u['q84'] - u['q16'])
        assert fit['uncertainty_info']['method'] == 'mcmc'
    assert widths[1] > 1.5 * widths[0]


def test_saved_parameters_include_posterior_columns(monkeypatch, tmp_path):
    fake_flux = _gaussian_line(monkeypatch)
    wave = np.linspace(13.3, 14.1, 160)
    observed = fake_flux(None, 450.0, 16.5, 0.5, wave)
    observed = observed + np.random.default_rng(0).normal(scale=0.01, size=wave.size)
    fit = fitting.fit_molecules(
        wave, observed, 'HCN', {'HCN': {'mol': 'HCN'}}, (13.3, 14.1),
        T_bounds=(200, 800), logN_bounds=(14, 18), loga_bounds=(-2, 1),
        n_samples=64, n_refine=2, n_top=2, sigma=0.01, seed=0, verbose=False,
    )
    fit.update(obs_wav=wave, obs_flux=observed)
    paths = fitting.save_fit_outputs(fit, tmp_path, 'demo')
    row = pd.read_csv(paths['params']).iloc[0]
    for key in ('T', 'logN', 'log10A'):
        assert row[f'{key}_q16'] <= row[f'{key}_q84']
        assert row[f'{key}_minus'] >= 0 and row[f'{key}_plus'] >= 0
        assert f'{key}_at_bound' in row
    assert row['uncertainty_method'] == 'mcmc'
    assert row['noise_correlation_length'] >= 1


def test_errorbars_false_returns_best_fit_only(monkeypatch, tmp_path):
    fake_flux = _gaussian_line(monkeypatch)
    wave = np.linspace(13.3, 14.1, 160)
    observed = fake_flux(None, 450.0, 16.5, 0.5, wave)
    observed = observed + np.random.default_rng(0).normal(scale=0.01, size=wave.size)
    fit = fitting.fit_molecules(
        wave, observed, 'HCN', {'HCN': {'mol': 'HCN'}}, (13.3, 14.1),
        T_bounds=(200, 800), logN_bounds=(14, 18), loga_bounds=(-2, 1),
        n_samples=64, n_refine=2, n_top=2, sigma=0.01, seed=0,
        errorbars=False, verbose=False,
    )
    assert fit['uncertainty'] == {}
    assert fit['uncertainty_info']['method'] == 'none'
    fit.update(obs_wav=wave, obs_flux=observed)
    row = pd.read_csv(fitting.save_fit_outputs(fit, tmp_path, 'demo')['params']).iloc[0]
    assert row['T'] == pytest.approx(fit['params']['HCN']['T'])
    assert row['T'] == row['T_best'] and np.isnan(row['T_minus'])
    assert row['uncertainty_method'] == 'none'


@pytest.mark.parametrize('errorbars', [True, False])
def test_saved_results_read_back_and_plot_consistently(monkeypatch, tmp_path, errorbars):
    import matplotlib
    matplotlib.use('Agg')
    from diskmelts.plotting import plot_fit
    from diskmelts.v2_read_plot import read_stage_result

    fake_flux = _gaussian_line(monkeypatch)
    wave = np.linspace(13.3, 14.1, 160)
    observed = fake_flux(None, 450.0, 16.5, 0.5, wave)
    observed = observed + np.random.default_rng(3).normal(scale=0.01, size=wave.size)
    fit = fitting.fit_molecules(
        wave, observed, 'HCN', {'HCN': {'mol': 'HCN'}}, (13.3, 14.1),
        T_bounds=(200, 800), logN_bounds=(14, 18), loga_bounds=(-2, 1),
        n_samples=64, n_refine=2, n_top=2, sigma=0.01, seed=0,
        errorbars=errorbars, verbose=False,
    )
    fit.update(obs_wav=wave, obs_flux=observed)
    fitting.save_fit_outputs(fit, tmp_path, 'src_hcn')
    fitting.save_running_spectrum(wave, fit['residual'], fit['model_flux'], tmp_path, 'src_hcn')
    back = read_stage_result(tmp_path, 'src', 'hcn')

    # params are best fits, as in a live fit.
    for key in ('T', 'logN', 'log10A'):
        assert back['params']['HCN'][key] == pytest.approx(fit['params']['HCN'][key])
    u = back['uncertainty']['HCN']['T']
    if errorbars:
        assert u['q50'] == pytest.approx(fit['uncertainty']['HCN']['T']['q50'])
    else:
        assert 'q50' not in u

    figure = plot_fit(wave, observed, back, params=back['params'],
                      uncertainty=back['uncertainty'], show=False)
    text = '\n'.join(t.get_text() for ax in figure.axes for t in ax.texts)
    assert 'nan' not in text
    assert ('posterior median' in text) is errorbars
    if not errorbars:
        assert f"T={fit['params']['HCN']['T']:.0f} K" in text
