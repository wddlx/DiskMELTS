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

def _notebook(path):
    return json.loads(path.read_text())


def _code_cell(notebook, index):
    return "".join(notebook["cells"][index]["source"])


@pytest.mark.parametrize("path", [FITTING_NOTEBOOK, TRAINING_NOTEBOOK])
def test_notebook_code_cells_compile(path):
    for index, cell in enumerate(_notebook(path)["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")


@pytest.mark.parametrize("start_dir", [ROOT, ROOT / "notebooks"])
def test_fitting_notebook_resolves_committed_assets(monkeypatch, start_dir):
    pytest.importorskip("torch")
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
def test_training_notebook_resolves_local_data_locations(monkeypatch, start_dir):
    pytest.importorskip("torch")
    notebook = _notebook(TRAINING_NOTEBOOK)
    namespace = {}

    monkeypatch.chdir(start_dir)
    exec(_code_cell(notebook, 1), namespace)
    namespace['TRAIN_MOLECULE'] = 'CO2'
    exec(_code_cell(notebook, 5), namespace)

    assert namespace["BASE_DIR"] == ROOT
    assert namespace["GRID_DIR"] == ROOT / "Model_grids" / namespace["TRAIN_MOLECULE"]
    assert namespace['LOCAL_MODEL'].parent == ROOT / 'Trained_model/CO2_single_v2'
    assert namespace['RUN_TRAINING'] is False


def test_production_checkpoint_files_are_bundled():
    from diskmelts import MODEL_SPECS
    from diskmelts.v2_fitting import MODEL_ROOT

    assert len(MODEL_SPECS) == 11
    for molecule, spec in MODEL_SPECS.items():
        path = MODEL_ROOT / spec['path']
        assert path.is_file(), f'missing bundled checkpoint for {molecule}: {path}'
    assert len(list(MODEL_ROOT.rglob('*.pt'))) == 15


def test_gitignore_separates_fitting_assets_from_training_data():
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    for pattern in ('/Model_grids/', '/Pretrain_grid/', '/Trained_model/',
                    '/docs/_build/', '/realobs_results/', '/figures/*'):
        assert pattern in patterns
    for path in ('src/diskmelts/models/Trained_model/C2H2_single_v2/net_C2H2_forward.pt',
                 'Realobs_data/Consub_data/j16120505_v9.0_contsub_RVcorr.csv',
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
    from diskmelts import MODEL_SPECS

    for spec in MODEL_SPECS.values():
        assert f"src/diskmelts/models/{spec['path']}" in tracked
    required = {
        'src/diskmelts/v2_fitting.py', 'src/diskmelts/v2_grid_fitting.py',
        'src/diskmelts/v2_read_plot.py', 'src/diskmelts/v2_paper_comparison.py',
        'notebooks/Example_Fitting.ipynb',
        'notebooks/Example_Training_Validation.ipynb',
        'docs/source/api/v2.md', 'examples/dev_v2_realobs.py',
        'examples/dev_v2_testsWithRO.py',
        'Realobs_data/literature_usco_table2.csv',
        'Realobs_data/literature_water_two_component.csv',
    }
    assert required <= tracked
    assert not any(path.startswith((
        'Trained_model/', 'docs/_build/', 'realobs_results/',
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

    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "dev_v2_realobs.py")],
        cwd=ROOT,
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
