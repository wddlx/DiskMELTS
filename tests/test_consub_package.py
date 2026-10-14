"""Release-facing continuum tests; independent of the ignored 0-Consub tree."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from diskmelts import consub


def _spectrum():
    wave = np.linspace(5.0, 25.0, 3000)
    continuum = 1.0 + 0.01 * (wave - 15.0)
    line = 0.2 * np.exp(-0.5 * ((wave - 12.5) / 0.06) ** 2)
    return wave, continuum + line, np.full(wave.shape, 0.01)


@pytest.mark.parametrize("preset", ["water", "ttauri", "carbon_rich", "carbon_extended"])
def test_bundled_presets_and_five_array_contract(preset, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not Path("0-Consub").exists()
    windows = consub.load_regions(preset)
    assert {"xmin", "xmax"} <= set(windows.columns)
    assert len(windows) > 0

    wave, flux, error = _spectrum()
    result = consub.calculate(wave, flux, error, preset=preset, exclude=[[12.4, 12.6]])
    assert len(result) == 5
    assert all(isinstance(column, np.ndarray) and column.shape == wave.shape
               for column in result)
    out_wave, residual, out_error, continuum, original = result
    np.testing.assert_array_equal(out_wave, wave)
    np.testing.assert_allclose(out_error, error)
    np.testing.assert_allclose(residual + continuum, flux, atol=1e-12)
    assert np.isfinite(original).all()


def test_custom_windows_and_no_reference_mode(tmp_path):
    wave, flux, error = _spectrum()
    csv = tmp_path / "windows.csv"
    pd.DataFrame({"xmin": [5.2, 8.2, 18.2],
                  "xmax": [6.2, 9.2, 19.2]}).to_csv(csv, index=False)
    custom = consub.calculate(wave, flux, error, preset=None, linefreefile=csv,
                              exclude=[[12.4, 12.6]])
    assert len(custom) == 5
    no_reference = consub.calculate(wave, flux, error, preset=None,
                                    linefreefile=None, exclude=[[12.4, 12.6]])
    np.testing.assert_allclose(no_reference[3], no_reference[4])
    with pytest.raises(ValueError, match="preset=None"):
        consub.calculate(wave, flux, error, preset="ttauri", linefreefile=csv)


def test_plot_optional_inputs_and_save_directory(tmp_path):
    import matplotlib.pyplot as plt

    wave, flux, error = _spectrum()
    result = consub.calculate(wave, flux, error, preset=None, linefreefile=None)
    fig = consub.plot(result[0], result[1], cont=result[3], ranges=(11, 14))
    assert not (tmp_path / "continuum.png").exists()
    plt.close(fig)

    directory = tmp_path / "plots"
    fig = consub.plot(*result, savedir=directory, ranges=[(5, 12), (12, 25)])
    assert (directory / "continuum.png").is_file()
    plt.close(fig)


def test_diagnostic_plot_marks_calibration_pixels_and_exclusions():
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    wave, flux, error = _spectrum()
    result = consub.subtract_continuum(
        wave, flux, error=error, preset="ttauri", exclude=[[12.4, 12.6]],
    )
    assert result.reference_mask.any()
    assert result.excluded.any()

    fig = consub.plot_consub(result, ranges=(11, 14))
    try:
        upper, lower = fig.axes
        calibration = next(collection for collection in upper.collections
                           if collection.get_label() == "Calibration pixels")
        plotted_reference = result.reference_mask.to_numpy() & (wave >= 11) & (wave <= 14)
        np.testing.assert_allclose(
            calibration.get_offsets()[:, 0],
            wave[plotted_reference],
        )
        assert any(patch.get_label() == "Excluded" for patch in upper.patches)
        assert any(patch.get_label() == "Weak continuum support"
                   for patch in upper.patches)
        assert any(np.allclose(patch.get_facecolor()[:3], to_rgba("lightcoral")[:3])
                   for patch in lower.patches)
    finally:
        plt.close(fig)
