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

MODEL_FILES = {
    "H2O": ("net_H2O_forward_11to19.pt", 21),
    "C2H2": ("net_C2H2_forward_11to19.pt", 15),
    "13C12CH2": ("net_13C12CH2_forward_11to19.pt", 15),
    "HCN": ("net_HCN_forward_11to19.pt", 15),
    "CO2": ("net_CO2_forward_11to19.pt", 15),
    "13CO2": ("net_13CO2_forward_11to19.pt", 15),
}


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
    exec(_code_cell(notebook, 0), namespace)
    exec(_code_cell(notebook, 1), namespace)

    assert namespace["BASE_DIR"] == ROOT
    assert namespace["INPUT_PATH"].is_file()
    assert namespace["INPUT_PATH"].name == "j16120505_v9.0_contsub_RVcorr.csv"
    assert all(path.is_file() for path in namespace["MODEL_PATHS"].values())


@pytest.mark.parametrize("start_dir", [ROOT, ROOT / "notebooks"])
def test_training_notebook_resolves_local_data_locations(monkeypatch, start_dir):
    pytest.importorskip("torch")
    notebook = _notebook(TRAINING_NOTEBOOK)
    namespace = {}

    monkeypatch.chdir(start_dir)
    exec(_code_cell(notebook, 0), namespace)
    exec(_code_cell(notebook, 2), namespace)

    assert namespace["BASE_DIR"] == ROOT
    assert namespace["GRID_DIR"] == ROOT / "Model_grids" / namespace["MOL"]
    assert namespace["PRETRAIN_CSV"].parent == ROOT / "Pretrain_grid"
    assert namespace["MODEL_PATH"].parent == ROOT / "Trained_model"


def test_committed_checkpoints_are_self_contained():
    torch = pytest.importorskip("torch")

    for molecule, (filename, expected_n_pca) in MODEL_FILES.items():
        path = ROOT / "Trained_model" / filename
        assert path.is_file(), f"missing committed checkpoint for {molecule}: {path}"

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint["mol"] == molecule
        assert checkpoint["n_pca"] == expected_n_pca
        assert checkpoint["pca"].n_components_ == expected_n_pca
        assert len(checkpoint["wav"]) > 0
        assert hasattr(checkpoint["xp_sc"], "mean_")


def test_gitignore_separates_fitting_assets_from_training_data():
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "Model_grids/" in patterns
    assert "Pretrain_grid/" in patterns
    assert "Trained_model/" not in patterns
    assert "Realobs_data/Consub_data/" not in patterns


def test_realobs_example_uses_committed_spectrum():
    source = (ROOT / "examples" / "dev_v1_realobs.py").read_text()
    assert "j16120505_v9.0_contsub_RVcorr.csv" in source
    assert "j16142029_v9.0_contsub_RVcorr.csv" not in source


def test_realobs_example_starts_from_github_assets(tmp_path):
    pytest.importorskip("torch")
    env = os.environ.copy()
    env["DISKMELTS_DETECTION_SIGMA_FACTOR"] = "1000000000"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")

    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "dev_v1_realobs.py")],
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
    script = examples_dir / "dev_v1_pt_validation.py"
    shutil.copy(ROOT / "examples" / "dev_v1_pt_validation.py", script)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert "Model_grids/ is intentionally ignored by Git" in result.stderr
