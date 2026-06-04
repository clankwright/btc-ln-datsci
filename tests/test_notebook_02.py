"""Tests for notebooks/02_centrality.ipynb (Phase 2.2)."""
from pathlib import Path

import pytest

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "02_centrality.ipynb"


def test_notebook_02_exists():
    assert NOTEBOOK_PATH.exists(), (
        "notebooks/02_centrality.ipynb must exist; create it as Phase 2.2"
    )


def test_notebook_02_valid_nbformat():
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    assert nb.nbformat == 4
    assert len(nb.cells) > 0, "Notebook must have at least one cell"


def test_notebook_02_has_required_sections():
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    required = [
        "Degree Centrality",
        "Betweenness",
        "Closeness",
        "k-sampling",
        "igraph",
    ]
    for section in required:
        assert section in md_text, (
            f"Notebook missing required section containing '{section}'"
        )


@pytest.mark.notebook
def test_notebook_02_executes_cleanly(tmp_path, monkeypatch):
    """Execute notebook 02 against the committed real-data fixture; assert no cell errors.

    Marked @pytest.mark.notebook and excluded from the default run.
    Run explicitly with: pytest -m notebook
    """
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
