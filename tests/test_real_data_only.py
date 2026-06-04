"""Tests for 4.5+4.6: notebooks are real-data-only, no IS_TEST_MODE, committed fixture."""
import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "describegraph_sample.json"

NB_PATHS = [
    NOTEBOOKS_DIR / "01_crawl_and_topology.ipynb",
    NOTEBOOKS_DIR / "02_centrality.ipynb",
    NOTEBOOKS_DIR / "03_routing_efficiency.ipynb",
    NOTEBOOKS_DIR / "04_liquidity_partial_observability.ipynb",
]


@pytest.mark.parametrize("nb_path", NB_PATHS, ids=[p.stem for p in NB_PATHS])
def test_no_IS_TEST_MODE_in_notebook(nb_path):
    """IS_TEST_MODE must not appear in any cell of any notebook (4.5)."""
    nb = nbformat.read(str(nb_path), as_version=4)
    violations = [
        (i, cell.source[:80])
        for i, cell in enumerate(nb.cells)
        if "IS_TEST_MODE" in cell.source
    ]
    assert not violations, (
        f"{nb_path.name}: IS_TEST_MODE found in cells {violations} — "
        "remove all IS_TEST_MODE branches (spec 4.5)"
    )


@pytest.mark.parametrize("nb_path", NB_PATHS, ids=[p.stem for p in NB_PATHS])
def test_no_synthetic_fallback_in_notebook(nb_path):
    """Notebooks must not construct inline synthetic node/channel lists (4.5)."""
    nb = nbformat.read(str(nb_path), as_version=4)
    violations = []
    for i, cell in enumerate(nb.cells):
        src = cell.source
        if ('nodes = [{"pub_key"' in src
                or 'channels = [' in src and '"node1_pub"' in src
                or '"pub{' in src
                or '"node{i' in src and '"pub_key"' in src):
            violations.append((i, src[:120]))
    assert not violations, (
        f"{nb_path.name}: synthetic graph builder found in cells {violations} — "
        "remove inline node/channel constructors (spec 4.5)"
    )


@pytest.mark.notebook
@pytest.mark.parametrize("nb_path", NB_PATHS, ids=[p.stem for p in NB_PATHS])
def test_notebook_raises_on_missing_snapshot(nb_path, tmp_path, monkeypatch):
    """Notebook load cell must fail loudly (FileNotFoundError or RuntimeError) when snapshot absent (4.5).

    LNGRAPH_SNAPSHOT_DIR is pointed at an empty tmp dir so no snapshot exists.
    """
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert.preprocessors.execute import CellExecutionError

    monkeypatch.setenv("LNGRAPH_SNAPSHOT_DIR", str(tmp_path))

    nb = nbformat.read(str(nb_path), as_version=4)
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    with pytest.raises(CellExecutionError):
        ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})


def test_fixture_file_exists():
    """tests/fixtures/describegraph_sample.json must exist (4.6)."""
    assert FIXTURE_PATH.exists(), (
        "tests/fixtures/describegraph_sample.json missing — "
        "extract a real subgraph sample (spec 4.6)"
    )


def test_fixture_has_real_data():
    """Fixture must have real node pub_keys (66-char hex), not synthetic 'node0Xaaa...' format (4.6)."""
    with open(FIXTURE_PATH) as f:
        dg = json.load(f)
    assert len(dg["nodes"]) >= 20, "Fixture must have at least 20 nodes"
    assert len(dg["edges"]) >= 20, "Fixture must have at least 20 channels"
    for n in dg["nodes"]:
        key = n["pub_key"]
        assert len(key) == 66 and all(c in "0123456789abcdef" for c in key), (
            f"pub_key {key!r} is not a real 66-char hex key — fixture must use real data"
        )


def test_fixture_graph_is_connected():
    """Fixture graph must be weakly connected (4.6)."""
    import networkx as nx

    with open(FIXTURE_PATH) as f:
        dg = json.load(f)
    G = nx.MultiDiGraph()
    for n in dg["nodes"]:
        G.add_node(n["pub_key"])
    for e in dg["edges"]:
        G.add_edge(e["node1_pub"], e["node2_pub"])
    assert nx.is_weakly_connected(G), (
        "Fixture graph must be weakly connected — extract a connected subgraph"
    )


def test_fixture_has_valid_describegraph_schema():
    """Fixture must conform to describegraph schema (nodes + edges with required fields) (4.6)."""
    with open(FIXTURE_PATH) as f:
        dg = json.load(f)
    assert "nodes" in dg and "edges" in dg
    for n in dg["nodes"]:
        assert "pub_key" in n
    for e in dg["edges"]:
        for field in ("channel_id", "capacity", "node1_pub", "node2_pub"):
            assert field in e, f"Edge missing field {field!r}"
