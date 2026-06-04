"""Tests for notebooks/04_liquidity_partial_observability.ipynb (Phase 3.4)."""
from pathlib import Path

import pytest

NOTEBOOK_PATH = (
    Path(__file__).parent.parent / "notebooks" / "04_liquidity_partial_observability.ipynb"
)


def test_notebook_04_exists():
    assert NOTEBOOK_PATH.exists(), (
        "notebooks/04_liquidity_partial_observability.ipynb must exist; create it as Phase 3.4"
    )


def test_notebook_04_valid_nbformat():
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    assert nb.nbformat == 4
    assert len(nb.cells) > 0, "Notebook must have at least one cell"


def test_notebook_04_has_required_sections():
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    required = [
        "Success Probability",
        "Monte-Carlo",
        "success-optimal",
    ]
    for section in required:
        assert section.lower() in md_text.lower(), (
            f"Notebook 04 missing required content containing '{section}'"
        )


def test_nb4_cell11_bottleneck_scoped_to_route_edge():
    """Cell 11 hop_caps loop must use G.edges(u, v, data=True), not G.edges(u, data=True).

    G.edges(u, data=True) returns ALL outgoing edges from u, so max() picks the
    largest capacity of any outgoing channel — not the capacity of the u→v hop.
    On a hub with channels of varying size this overstates the bottleneck and
    makes the success-probability sweep degenerate (all zeros for the upper half).
    """
    import nbformat

    nb = nbformat.read(str(NOTEBOOK_PATH), as_version=4)
    cell_src = nb.cells[11].source

    assert "G[u][v].items()" in cell_src, (
        "Cell 11 bottleneck loop must use G[u][v].items() "
        "to scope hop capacity to the specific u→v route edge"
    )
    assert "G.edges(u, data=True)" not in cell_src, (
        "Cell 11 must not use G.edges(u, data=True) — iterates all outgoing edges "
        "from u, overstating the bottleneck when u is a hub with unequal capacities"
    )


@pytest.mark.notebook
def test_notebook_04_executes_cleanly(tmp_path, monkeypatch):
    """Execute notebook 04 against the committed real-data fixture; assert no cell errors."""
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
