from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parent.parent
FITTING_NOTEBOOK = ROOT / "notebooks" / "Example_Fitting.ipynb"
TRAINING_NOTEBOOK = ROOT / "notebooks" / "Example_Training_Validation.ipynb"
CONSUB_NOTEBOOK = ROOT / "notebooks" / "Example_Consub.ipynb"
SPECTRUM_NOTEBOOK = ROOT / "notebooks" / "Example_Generate_Spectra.ipynb"

def _notebook(path):
    return json.loads(path.read_text())


def _code_cell(notebook, index):
    return "".join(notebook["cells"][index]["source"])


@pytest.mark.parametrize("path", [FITTING_NOTEBOOK, TRAINING_NOTEBOOK,
                                  CONSUB_NOTEBOOK, SPECTRUM_NOTEBOOK])
def test_notebook_code_cells_compile(path):
    for index, cell in enumerate(_notebook(path)["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("path,minimum", [(CONSUB_NOTEBOOK, 1),
                                          (FITTING_NOTEBOOK, 4),
                                          (SPECTRUM_NOTEBOOK, 2)])
def test_notebooks_keep_saved_figures_for_docs(path, minimum):
    # The docs never execute notebooks, so these saved figures are what readers see.
    figures = sum("image/png" in output.get("data", {})
                  for cell in _notebook(path)["cells"] if cell["cell_type"] == "code"
                  for output in cell.get("outputs", []))
    assert figures >= minimum
    assert "/Users/" not in path.read_text()


@pytest.mark.parametrize("start_dir", [ROOT, ROOT / "notebooks"])
def test_consub_notebook_reads_public_usco_csv(monkeypatch, start_dir):
    raw_dir = ROOT / "Realobs_data" / "Raw_data"
    assert len(list(raw_dir.glob("*_1d_v9.0.csv"))) == 14
    namespace = {}
    monkeypatch.chdir(start_dir)
    exec(_code_cell(_notebook(CONSUB_NOTEBOOK), 1), namespace)

    assert namespace["RAW_CSV"] == raw_dir / "j16120505_1d_v9.0.csv"
    assert len(namespace["wav"]) == len(namespace["flux"]) == len(namespace["error"])
    assert len(namespace["wav"]) > 10_000


@pytest.mark.parametrize("start_dir", [ROOT, ROOT / "notebooks"])
def test_fitting_notebook_accepts_standalone_paths(monkeypatch, start_dir):
    pytest.importorskip("torch")
    notebook = _notebook(FITTING_NOTEBOOK)
    namespace = {}

    monkeypatch.chdir(start_dir)
    exec(_code_cell(notebook, 1), namespace)
    exec(_code_cell(notebook, 2), namespace)

    assert "BASE_DIR" not in namespace
    assert namespace["SOURCE"] == Path('my_spectrum.csv')
    assert namespace["OBS_WAV"] is None
    assert namespace["OBS_FLUX"] is None
    assert namespace["OBS_ERR"] is None
    assert namespace["DISTANCE_PC"] == pytest.approx(140.0)
    assert namespace["OUTPUT_DIR"] == Path('results/my_source')
    assert [stage['name'] for stage in namespace['STAGES']] == [
        'water', 'carbon', 'CO2']
    assert 'MODEL_PATHS' not in namespace


def test_fitting_notebook_config_runs_outside_checkout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    notebook = _notebook(FITTING_NOTEBOOK)
    namespace = {}
    exec(_code_cell(notebook, 1), namespace)
    exec(_code_cell(notebook, 2), namespace)
    assert namespace['SOURCE'] == Path('my_spectrum.csv')
    assert namespace['OUTPUT_DIR'] == Path('results/my_source')


@pytest.mark.parametrize('array_mode', [False, True])
def test_fitting_notebook_selects_csv_or_arrays(monkeypatch, tmp_path, array_mode):
    import numpy as np

    monkeypatch.chdir(tmp_path)
    notebook = _notebook(FITTING_NOTEBOOK)
    namespace = {}
    exec(_code_cell(notebook, 1), namespace)
    exec(_code_cell(notebook, 2), namespace)
    wave = np.array([13.8, 13.9])
    flux = np.array([0.1, 0.2])
    error = np.array([0.01, 0.01])
    source = tmp_path / 'my_spectrum.csv'
    source.write_text('wav,flux\n13.8,0.1\n13.9,0.2\n')
    if array_mode:
        namespace.update(SOURCE=None, OBS_WAV=wave, OBS_FLUX=flux,
                         OBS_ERR=error)
    called = {}

    def fake_fit_observation(**kwargs):
        called.update(kwargs)
        return {'fits': {}, 'obs_wav': wave, 'obs_flux': flux}

    namespace['fit_observation'] = fake_fit_observation
    namespace['OUTPUT_DIR'].mkdir(parents=True)
    fit_setup = _code_cell(notebook, 4).split("print('Fitted stages:")[0]
    exec(fit_setup, namespace)
    if array_mode:
        np.testing.assert_array_equal(called['obs_wav'], wave)
        np.testing.assert_array_equal(called['obs_flux'], flux)
        np.testing.assert_array_equal(called['obs_err'], error)
        assert set(namespace['PLOT_INPUT']) == {'obs_wav', 'obs_flux'}
        np.testing.assert_array_equal(namespace['PLOT_INPUT']['obs_wav'], wave)
    else:
        assert called['input_path'] == Path('my_spectrum.csv')
        assert namespace['PLOT_INPUT'] == {'input_path': Path('my_spectrum.csv')}


@pytest.mark.parametrize("start_dir", [ROOT, ROOT / "notebooks"])
@pytest.mark.parametrize("grid_sigma", [0, 2])
def test_training_notebook_resolves_local_data_locations(monkeypatch, start_dir,
                                                        grid_sigma):
    pytest.importorskip("torch")
    notebook = _notebook(TRAINING_NOTEBOOK)
    namespace = {}

    monkeypatch.chdir(start_dir)
    exec(_code_cell(notebook, 1), namespace)
    namespace['TRAIN_MOLECULE'] = 'CO2'
    source = _code_cell(notebook, 5).replace(
        'GRID_SIGMA = 0  #', f'GRID_SIGMA = {grid_sigma}  #')
    exec(source, namespace)

    assert namespace["BASE_DIR"] == start_dir
    assert namespace["GRID_DIR"] == start_dir / "Model_grids" / f"sigma{grid_sigma}" / namespace["TRAIN_MOLECULE"]
    output = start_dir / 'Trained_model'
    if grid_sigma == 2:
        output /= 'sigma2'
    assert namespace['LOCAL_MODEL'].parent == output / 'CO2_single_v2'
    assert namespace['RUN_TRAINING'] is False


def test_production_checkpoint_files_are_bundled():
    from diskmelts import MODEL_SPECS, SIGMA2_MODEL_SPECS
    from diskmelts.v2_fitting import MODEL_ROOT

    assert len(MODEL_SPECS) == 18
    for molecule, spec in MODEL_SPECS.items():
        assert spec['grid_sigma'] == 0
        path = MODEL_ROOT / spec['path']
        assert path.is_file(), f'missing bundled checkpoint for {molecule}: {path}'
    assert set(SIGMA2_MODEL_SPECS) == {
        'H2O', 'C2H2', 'HCN', 'CO2', '13C12CH2', '13CO2',
        'C4H2', 'HC3N', 'C2H6', 'C2H4', 'CH4', 'C2HD',
        'CO17O', 'CO18O', 'C3H4', 'CH3', 'C6H6', 'OH',
    }
    for molecule, spec in SIGMA2_MODEL_SPECS.items():
        assert spec['grid_sigma'] == 2
        assert (MODEL_ROOT / spec['path']).is_file(), molecule
    assert len(list(MODEL_ROOT.rglob('*.pt'))) == 46


def test_new_grid_checkpoints_load_and_generate_finite_spectra():
    import numpy as np
    from diskmelts import load_fitting_models
    from diskmelts.fitting import generate_spectrum

    molecules = ('13C12CH2', 'C2HD', 'CO17O', 'CO18O', 'C6H6', 'C3H4', 'CH3')
    models = load_fitting_models(molecules)
    for molecule in molecules:
        flux = generate_spectrum(400., 16., 1., models[molecule])
        assert flux.size == models[molecule]['wav'].size
        assert np.isfinite(flux).all()
        assert flux.max() > 0


def test_gitignore_separates_fitting_assets_from_training_data():
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    for pattern in ('/.ai/', '/0-ai/', '/Model_grids/', '/Pretrain_grid/', '/Trained_model/',
                    '/0-Consub/',
                    '/docs/_build/', '/realobs_results/', '/figures/*'):
        assert pattern in patterns
    for path in ('src/diskmelts/models/Trained_model/sigma0/C2H2_wavelength_bank_v2/net_C2H2_w11p5to17.pt',
                 'src/diskmelts/consub/regions/ttauri.csv',
                 'src/diskmelts/consub/cont_ranges.csv',
                 'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv',
        'Realobs_data/Raw_data/j16120505_1d_v9.0.csv',
                 'notebooks/Example_Fitting.ipynb',
                 'examples/synthetic_array_fit.py',
                 'figures/realobs_validations/H2O_parameter_comparison.pdf'):
        result = subprocess.run(['git', 'check-ignore', '--no-index', '-q', path],
                                cwd=ROOT)
        assert result.returncode == 1, f'important file is ignored: {path}'


def test_git_index_contains_fitting_distribution_only():
    """Catch an omitted checkpoint or accidentally tracked generated tree."""
    if not (ROOT / '.git').exists():
        pytest.skip('Git index is unavailable in an installed source archive')
    tracked = set(subprocess.check_output(
        ['git', 'ls-files', '--cached'], cwd=ROOT, text=True).splitlines())
    from diskmelts import MODEL_SPECS, SIGMA2_MODEL_SPECS

    for spec in (*MODEL_SPECS.values(), *SIGMA2_MODEL_SPECS.values()):
        path = f"src/diskmelts/models/{spec['path']}"
        assert (ROOT / path).is_file()
        assert path in tracked
        if path.endswith('.json'):
            manifest = json.loads((ROOT / path).read_text())
            for tile in manifest['tiles']:
                tile_path = str(Path(path).parent / tile['path'])
                assert tile_path in tracked
                assert (ROOT / tile_path).is_file()
        assert subprocess.run(['git', 'check-ignore', '--no-index', '-q', path],
                              cwd=ROOT).returncode == 1
    from diskmelts.consub._core import HERE, PRESETS

    for preset in PRESETS:
        assert f'src/diskmelts/consub/regions/{preset}.csv' in tracked
        assert f'src/diskmelts/consub/regions/{preset}.json' in tracked
        assert (HERE / 'regions' / f'{preset}.csv').is_file()
    required = {
        'src/diskmelts/v2_fitting.py', 'src/diskmelts/v2_grid_fitting.py',
        'src/diskmelts/v2_read_plot.py', 'src/diskmelts/v2_paper_comparison.py',
        'notebooks/Example_Fitting.ipynb',
        'notebooks/Example_Training_Validation.ipynb',
        'docs/source/api/v2.md', 'examples/dev_v2_realobs.py',
        'examples/dev_v2_testsWithRO.py',
        'Realobs_data/literature_usco_table2.csv',
        'Realobs_data/literature_water_two_component.csv',
        'src/diskmelts/consub/__init__.py',
        'src/diskmelts/consub/_api.py',
        'src/diskmelts/consub/_core.py',
        'src/diskmelts/consub/cont_ranges.csv',
        'src/diskmelts/consub/cont_ranges_PPan.csv',
        'docs/source/continuum.md',
        'docs/source/api/consub.md',
    }
    assert required <= tracked
    assert not any(path.startswith((
        'Trained_model/', '0-Consub/', 'docs/_build/', 'realobs_results/',
        'figures/validation/', 'figures/realobs/', 'Model_grids/',
        'Pretrain_grid/',
    )) for path in tracked)


def test_realobs_example_uses_committed_spectrum():
    source = (ROOT / "examples" / "dev_v2_realobs.py").read_text()
    assert "j16120505_v9.0_contsub_RVcorr.csv" in source
    assert "j16142029_v9.0_contsub_RVcorr.csv" not in source


def test_synthetic_array_example_runs_without_repo_data(tmp_path):
    pytest.importorskip('torch')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src')
    env['MPLCONFIGDIR'] = str(tmp_path / 'matplotlib')
    output_dir = tmp_path / 'fits'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'examples' / 'synthetic_array_fit.py'),
         '--output-dir', str(output_dir), '--n-samples', '64'],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert 'Fitted HCN:' in result.stdout
    assert (output_dir / 'synthetic_hcn_hcn_running.csv').is_file()


def test_realobs_example_starts_from_github_assets(tmp_path):
    pytest.importorskip("torch")
    env = os.environ.copy()
    env["DISKMELTS_DETECTION_SIGMA_FACTOR"] = "1000000000"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["PYTHONPATH"] = str(ROOT / "src")
    # Exercise the unchanged script in an isolated layout so the smoke test
    # cannot overwrite a user's scientific fit products in the checkout.
    example_dir = tmp_path / "examples"
    example_dir.mkdir()
    script = example_dir / "dev_v2_realobs.py"
    shutil.copy(ROOT / "examples" / script.name, script)
    data_dir = tmp_path / "Realobs_data" / "Consub_data"
    data_dir.mkdir(parents=True)
    spectrum = "j16120505_v9.0_contsub_RVcorr.csv"
    shutil.copy(ROOT / "Realobs_data" / "Consub_data" / spectrum, data_dir / spectrum)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "Loaded H2O" in result.stdout
    assert "Done." in result.stdout


def test_training_example_explains_missing_local_grids(tmp_path):
    pytest.importorskip("torch")
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    script = examples_dir / "dev_v2_pt_validation.py"
    shutil.copy(ROOT / "examples" / "dev_v2_pt_validation.py", script)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Model_grids/ is intentionally ignored by Git" in result.stderr


def test_package_and_docs_versions_match():
    import re

    package = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M)
    docs = re.search(r"^release\s*=\s*'([^']+)'", (ROOT / "docs/source/conf.py").read_text(), re.M)
    assert package and docs and package.group(1) == docs.group(1)
    changelog = (ROOT / "docs/source/changelog.md").read_text()
    assert f"## {package.group(1)}" in changelog


def test_every_public_function_is_in_the_api_docs():
    pytest.importorskip("torch")
    import re

    import diskmelts

    api = "\n".join(p.read_text() for p in (ROOT / "docs/source/api").glob("*.md"))
    documented = set(re.findall(r"auto(?:function|class):: [\w.]*?\.(\w+)\n", api + "\n"))
    public = {name for name in diskmelts.__all__
              if callable(getattr(diskmelts, name)) and name != "consub"}
    assert not public - documented, sorted(public - documented)
    for name in ("fit_nested", "validate_nt", "plot_validation_split"):
        assert name in (ROOT / "docs/source/api/deprecated.md").read_text()


def test_observation_functions_accept_arrays_or_csv(tmp_path):
    pytest.importorskip("torch")
    import numpy as np

    from diskmelts.fitting import _read_observation

    csv = tmp_path / "spec.csv"
    csv.write_text("wav,flux\n13.8,0.1\n13.9,nan\n14.0,0.3\n")
    wav, flux, err, name = _read_observation(csv)
    assert name == "spec" and err is None and list(flux) == [0.1, 0.3]
    wav2, flux2, err2, name2 = _read_observation(
        obs_wav=[13.8, 13.9, 14.0], obs_flux=[0.1, np.nan, 0.3],
        obs_err=[0.01, 0.02, 0.03], name="arr")
    assert name2 == "arr" and list(err2) == [0.01, 0.03]
    np.testing.assert_array_equal(wav2, wav)
    with pytest.raises(ValueError, match="not both"):
        _read_observation(csv, obs_wav=wav, obs_flux=flux)
    import inspect

    from diskmelts import fit_observation, fit_observation_grids, plot_saved_observation
    for function in (fit_observation, fit_observation_grids, plot_saved_observation):
        parameters = inspect.signature(function).parameters
        assert {"input_path", "obs_wav", "obs_flux"} <= set(parameters)
