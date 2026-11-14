from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
from matplotlib import pyplot as plt  # noqa: E402

from diskmelts.lines import LINE_CATEGORIES, load_line_list, plot_line_markers  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RYDBERG_H = 109677.58  # cm^-1, hydrogen with reduced-mass correction


def test_catalogue_contents():
    lines = load_line_list()
    assert set(lines['category']) == set(LINE_CATEGORIES)
    assert {'HI', 'H2', 'HD', 'NeII', 'NeIII', 'ArII', 'ArIII'} <= set(lines['species'])
    assert 'OH' not in set(lines['species'])
    assert lines['wavelength_um'].is_monotonic_increasing
    assert lines['wavelength_um'].between(4.8, 28.6).all()
    assert not lines.duplicated(['species', 'transition']).any()
    assert lines[['label', 'reference', 'source_url']].notna().all().all()


@pytest.mark.parametrize('species, transition, wavelength', [
    ('H2', 'v=0-0 S(1)', 17.03483),
    ('H2', 'v=0-0 S(2)', 12.27861),
    ('HD', 'v=0-0 R(5)', 19.43100),
    ('NeII', '2P1/2-2P3/2', 12.81355),
    ('NeIII', '3P1-3P2', 15.5551),
    ('ArII', '2P1/2-2P3/2', 6.985274),
    ('ArIII', '3P1-3P2', 8.99138),
    ('HI', '7-6', 12.371898),
])
def test_reference_wavelengths(species, transition, wavelength):
    lines = load_line_list(species)
    row = lines[lines['transition'] == transition]
    assert len(row) == 1
    assert row['wavelength_um'].iloc[0] == pytest.approx(wavelength, abs=1e-6)


def test_hydrogen_wavelengths_follow_rydberg():
    lines = load_line_list('HI')
    for transition, wav in zip(lines['transition'], lines['wavelength_um']):
        upper, lower = map(int, transition.split('-'))
        assert 0 < upper - lower <= 3 and upper <= 14
        rydberg = 1e4 / (RYDBERG_H * (1 / lower**2 - 1 / upper**2))
        assert wav == pytest.approx(rydberg, rel=1e-5)


def test_filters():
    subset = load_line_list(['H2', 'NeII'], wav_range=(10, 20))
    assert set(subset['species']) == {'H2', 'NeII'}
    assert subset['wavelength_um'].between(10, 20).all()
    assert set(load_line_list(category='molecular')['species']) == {'H2', 'HD'}
    with pytest.raises(ValueError, match='Unknown species'):
        load_line_list('OH')
    with pytest.raises(ValueError, match='Unknown category'):
        load_line_list(category='ionic')


def test_plot_line_markers_uses_axis_range():
    fig, ax = plt.subplots()
    ax.set_xlim(12.0, 13.0)
    drawn = plot_line_markers(ax)
    assert ax.get_xlim() == (12.0, 13.0)
    assert set(drawn['species']) >= {'H2', 'HI', 'NeII'}
    assert len(ax.lines) == len(drawn)
    marker = ax.lines[0]
    assert marker.get_linestyle() == '--' and marker.get_alpha() == 0.6
    text = ' / '.join(t.get_text() for t in ax.texts)
    assert '[Ne II]' in text and 'H I 7-6' in text
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.set_xlim(12.0, 13.0)
    plot_line_markers(ax, species='NeII', label=False)
    assert len(ax.lines) == 1 and not ax.texts
    plt.close(fig)


def test_plot_fit_show_lines():
    from diskmelts.plotting import plot_fit

    wav = np.linspace(12.0, 16.0, 400)
    flux = np.exp(-0.5 * ((wav - 13.7) / 0.01) ** 2)
    result = {'model_flux': flux * 0.9, 'mol': 'C2H2'}
    plain = plot_fit(wav, flux, result, show=False)
    assert not plain.axes[0].texts
    marked = plot_fit(wav, flux, result, show=False, show_lines=True,
                      line_species=['H2', 'NeIII'])
    labels = {t.get_text() for t in marked.axes[0].texts}
    assert labels == {'H$_2$ S(2)', '[Ne III]'}
    assert len(marked.axes[1].lines) - len(plain.axes[1].lines) == 2
    plt.close('all')


def test_docs_table_matches_catalogue():
    page = (ROOT / 'docs/source/unfitted_lines.md').read_text()
    rows = re.findall(r'^\| `(\w+)` \| ([^|]+) \| ([\d.]+) \|', page, re.M)
    lines = load_line_list()
    documented = sorted((s, t.strip(), float(w)) for s, t, w in rows)
    expected = sorted(zip(lines['species'], lines['transition'], lines['wavelength_um']))
    assert documented == expected


@pytest.mark.parametrize('grid_sigma', (0, 2))
def test_broad_t_oh_model_covers_100_to_6000_k(grid_sigma):
    pytest.importorskip('torch')
    from diskmelts import OH_BROAD_T_SPECS, oh_model_flux, synthesize_spectrum

    spec = OH_BROAD_T_SPECS[grid_sigma]
    assert spec['T_bounds'] == (100.0, 6000.0)
    wav = np.linspace(4.0, 28.0, 3000)
    cold = oh_model_flux(300, wav, sigma=grid_sigma)
    hot = oh_model_flux(5000, wav, sigma=grid_sigma)
    assert np.isfinite(cold).all() and np.isfinite(hot).all()
    assert not cold[(wav < 4.9) | (wav > 27.0)].any()
    # Hot OH peaks at shorter wavelengths than cold OH.
    assert wav[np.argmax(hot)] < wav[np.argmax(cold)]
    assert hot[wav < 10].max() > 0.05 * hot.max()
    _, flux = synthesize_spectrum('OH_broadT', 3000, 14.0, sigma=grid_sigma,
                                  wavelength=wav[(wav > 5) & (wav < 26)])
    assert flux.max() > 0
    for bad in (50, 6500):
        with pytest.raises(ValueError, match='outside'):
            oh_model_flux(bad, wav, sigma=grid_sigma)


def test_plot_oh_model_scales_to_residual_peak():
    pytest.importorskip('torch')
    from diskmelts import plot_oh_model

    wav = np.linspace(9.0, 26.0, 4000)
    residual = 0.02 * np.sin(wav * 7.0) + 0.05 * np.exp(-0.5 * ((wav - 17) / 0.01) ** 2)
    peak = residual.max()
    fig, ax = plt.subplots()
    curves = plot_oh_model(ax, wav, residual, [800, 4000])
    assert set(curves) == {800.0, 4000.0}
    for curve in curves.values():
        assert np.isclose(curve.max(), 0.6 * peak)
        assert np.isclose(curve.min(), -0.4 * peak, atol=1e-3 * peak)
    assert len(ax.lines) == 2 and ax.get_legend() is not None
    plt.close(fig)


def test_plot_fit_draws_oh_model_in_residual_panel():
    pytest.importorskip('torch')
    from diskmelts.plotting import plot_fit

    wav = np.linspace(12.0, 26.0, 3000)
    flux = np.exp(-0.5 * ((wav - 13.7) / 0.01) ** 2)
    result = {'model_flux': flux * 0.9, 'mol': 'C2H2', 'grid_sigma': 2}
    plain = plot_fit(wav, flux, result, show=False)
    with_oh = plot_fit(wav, flux, result, show=False, oh_T=1500)
    assert len(with_oh.axes[1].lines) == len(plain.axes[1].lines) + 1
    assert 'OH LTE 1500 K' in with_oh.axes[1].get_legend().get_texts()[0].get_text()
    with pytest.raises(ValueError, match='outside'):
        plot_fit(wav, flux, result, show=False, oh_T=8000)
    plt.close('all')


def test_line_list_accepts_reversed_range():
    assert load_line_list(wav_range=(20, 10)).equals(load_line_list(wav_range=(10, 20)))


def test_check_oh_model_args_bounds():
    from diskmelts.lines import check_oh_model_args

    check_oh_model_args([100, 6000], 13.0, 0)
    check_oh_model_args(3000, 19.5, 2)
    for bad in (dict(T=[1000, 99]), dict(T=6001), dict(T=[]),
                dict(T=1000, logN=12.9), dict(T=1000, sigma=1),
                dict(T=1000, sigma=True), dict(T=float('nan'))):
        with pytest.raises(ValueError):
            check_oh_model_args(**bad)


def test_broad_t_oh_bank_is_not_a_fitting_model():
    pytest.importorskip('torch')
    from diskmelts import load_fitting_models, synthesize_spectrum

    with pytest.raises(ValueError, match='No v2 model checkpoint'):
        load_fitting_models('OH_broadT')
    wav, flux = synthesize_spectrum('OH_broadT', 3000, 14.0)
    assert wav.min() == pytest.approx(4.9, abs=0.01)
    assert wav.max() == pytest.approx(27.0, abs=0.05)
    assert np.isfinite(flux).all() and flux.max() > 0


def test_plot_oh_model_warns_when_nothing_to_draw():
    pytest.importorskip('torch')
    from diskmelts import plot_oh_model

    fig, ax = plt.subplots()
    wav = np.linspace(12.0, 20.0, 500)
    with pytest.warns(UserWarning, match='no positive peak'):
        assert plot_oh_model(ax, wav, -np.ones_like(wav), 1000) == {}
    outside = np.linspace(27.5, 28.0, 50)
    with pytest.warns(UserWarning, match='no OH emission'):
        assert plot_oh_model(ax, outside, np.ones_like(outside), 1000) == {}
    assert not ax.lines
    plt.close(fig)


def test_plot_fit_rejects_bad_oh_t_before_drawing():
    from diskmelts.plotting import plot_fit

    wav = np.linspace(12.0, 26.0, 300)
    flux = np.ones_like(wav)
    plt.close('all')
    with pytest.raises(ValueError, match='outside'):
        plot_fit(wav, flux, {'model_flux': flux}, show=False, oh_T=[1000, 8000])
    assert not plt.get_fignums()


def test_plot_fit_title_clears_line_names():
    from diskmelts.plotting import plot_fit

    wav = np.linspace(5.0, 26.0, 3000)
    flux = 1 + 0.1 * np.sin(wav * 9.0)
    fig = plot_fit(wav, flux, {'model_flux': 0.9 * flux}, show=False,
                   show_lines=True, name='source')
    fig.canvas.draw()
    ax = fig.axes[0]
    names = [t for t in ax.texts if not t.get_clip_on()]
    assert names
    title = ax.title.get_window_extent()
    assert title.y0 >= max(t.get_window_extent().y1 for t in names) - 1
    assert title.y1 <= fig.bbox.height
    plt.close(fig)
