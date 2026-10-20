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

def _notebook(path):
    return json.loads(path.read_text())


def _code_cell(notebook, index):
    return "".join(notebook["cells"][index]["source"])


@pytest.mark.parametrize("path", [FITTING_NOTEBOOK, TRAINING_NOTEBOOK, CONSUB_NOTEBOOK])
def test_notebook_code_cells_compile(path):
    for index, cell in enumerate(_notebook(path)["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("path", [FITTING_NOTEBOOK, TRAINING_NOTEBOOK])
def test_notebook_rejects_incompatible_sklearn(path, monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    monkeypatch.setattr(sklearn, "__version__", "0.0")
    with pytest.raises(RuntimeError, match="require scikit-learn 1.8.0"):
        exec(_code_cell(_notebook(path), 1), {})


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
def test_fitting_notebook_resolves_committed_assets(monkeypatch, start_dir):
    pytest.importorskip("torch")
    sklearn = pytest.importorskip("sklearn")
    # This test checks paths and stage settings; the version guard is tested above.
    monkeypatch.setattr(sklearn, "__version__", "1.8.0")
    notebook = _notebook(FITTING_NOTEBOOK)
    namespace = {}

    monkeypatch.chdir(start_dir)
    exec(_code_cell(notebook, 1), namespace)
    exec(_code_cell(notebook, 2), namespace)

    assert namespace["BASE_DIR"] == ROOT
    assert namespace["SOURCE"].is_file()
    assert namespace["SOURCE"].name == "j16120505_v9.0_contsub_RVcorr.csv"
    assert namespace["DISTANCE_PC"] == pytest.approx(122.5)
    assert [stage['name'] for stage in namespace['STAGES']] == [
        'water', 'carbon', 'CO2']
    assert 'MODEL_PATHS' not in namespace


@pytest.mark.parametrize("start_dir", [ROOT, ROOT / "notebooks"])
@pytest.mark.parametrize("grid_sigma", [0, 2])
def test_training_notebook_resolves_local_data_locations(monkeypatch, start_dir,
                                                        grid_sigma):
    pytest.importorskip("torch")
    sklearn = pytest.importorskip("sklearn")
    monkeypatch.setattr(sklearn, "__version__", "1.8.0")
    notebook = _notebook(TRAINING_NOTEBOOK)
    namespace = {}

    monkeypatch.chdir(start_dir)
    exec(_code_cell(notebook, 1), namespace)
    namespace['TRAIN_MOLECULE'] = 'CO2'
    source = _code_cell(notebook, 5).replace(
        'GRID_SIGMA = 0  #', f'GRID_SIGMA = {grid_sigma}  #')
    exec(source, namespace)

    assert namespace["BASE_DIR"] == ROOT
    assert namespace["GRID_DIR"] == ROOT / "Model_grids" / f"sigma{grid_sigma}" / namespace["TRAIN_MOLECULE"]
    output = ROOT / 'Trained_model'
    if grid_sigma == 2:
        output /= 'sigma2'
    assert namespace['LOCAL_MODEL'].parent == output / 'CO2_single_v2'
    assert namespace['RUN_TRAINING'] is False


def test_production_checkpoint_files_are_bundled():
    from diskmelts import MODEL_SPECS, SIGMA2_MODEL_SPECS
    from diskmelts.v2_fitting import MODEL_ROOT

    assert len(MODEL_SPECS) == 17
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
    assert len(list(MODEL_ROOT.rglob('*.pt'))) == 44


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

    for pattern in ('/Model_grids/', '/Pretrain_grid/', '/Trained_model/',
                    '/0-Consub/',
                    '/docs/_build/', '/realobs_results/', '/figures/*'):
        assert pattern in patterns
    for path in ('src/diskmelts/models/Trained_model/sigma0/C2H2_single_v2/net_C2H2_forward.pt',
                 'src/diskmelts/consub/regions/ttauri.csv',
                 'src/diskmelts/consub/cont_ranges.csv',
                 'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv',
        'Realobs_data/Raw_data/j16120505_1d_v9.0.csv',
                 'notebooks/Example_Fitting.ipynb',
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
