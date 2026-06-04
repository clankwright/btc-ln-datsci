"""Tests for notebooks/03_routing_efficiency.ipynb (Phase 3.4)."""
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "03_routing_efficiency.ipynb"


def test_notebook_03_exists():
    assert NOTEBOOK_PATH.exists(), (
        "notebooks/03_routing_efficiency.ipynb must exist; create it as Phase 3.4"
    )


def test_notebook_03_valid_nbformat():
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    assert nb.nbformat == 4
    assert len(nb.cells) > 0, "Notebook must have at least one cell"


def test_notebook_03_has_required_sections():
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    required = [
        "Routing",
        "fee",
        "k-shortest",
        "CLTV",
    ]
    for section in required:
        assert section.lower() in md_text.lower(), (
            f"Notebook 03 missing required content containing '{section}'"
        )


@pytest.mark.notebook
def test_notebook_03_executes_cleanly(tmp_path, monkeypatch):
    """Execute notebook 03 against the committed real-data fixture; assert no cell errors."""
    import shutil
    from pathlib import Path

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    sample = Path(__file__).parent / "fixtures" / "describegraph_sample.json"
    shutil.copy(sample, tmp_path / "describegraph.json")
    monkeypatch.setenv("LNGRAPH_SNAPSHOT_DIR", str(tmp_path))

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
    nb_out, _ = ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_PATH.parent)}})

    cell_errors = [
        {"cell": i, "ename": out.get("ename"), "evalue": out.get("evalue")}
        for i, cell in enumerate(nb_out.cells)
        if cell.cell_type == "code"
        for out in cell.outputs
        if out.output_type == "error"
    ]
    assert not cell_errors, f"Notebook cells raised errors: {cell_errors}"
