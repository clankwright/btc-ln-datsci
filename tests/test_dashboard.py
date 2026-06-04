"""Tests for dashboard/app.py and dashboard/helpers.py (Phase 4.3)."""
import json
from pathlib import Path

import pytest

DASHBOARD_APP = Path(__file__).parent.parent / "dashboard" / "app.py"
DASHBOARD_HELPERS = Path(__file__).parent.parent / "dashboard" / "helpers.py"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "describegraph_sample.json"


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

def test_dashboard_app_exists():
    assert DASHBOARD_APP.exists(), "dashboard/app.py must exist (Phase 4.3)"


def test_dashboard_helpers_exists():
    assert DASHBOARD_HELPERS.exists(), "dashboard/helpers.py must exist (Phase 4.3)"


def test_dashboard_app_has_node_inspector():
    text = DASHBOARD_APP.read_text()
    assert "Node Inspector" in text


def test_dashboard_app_has_route_explorer():
    text = DASHBOARD_APP.read_text()
    assert "Route Explorer" in text


def test_dashboard_helpers_importable():
    from dashboard.helpers import (  # noqa: F401
        build_node_options,
        format_route_table,
        get_node_centrality_row,
        summarize_route,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_graph():
    from lngraph.ingest import ingest
    data = json.loads(FIXTURE_PATH.read_text())
    return ingest(data)


# ---------------------------------------------------------------------------
# build_node_options
# ---------------------------------------------------------------------------

def test_build_node_options_length(sample_graph):
    from dashboard.helpers import build_node_options
    opts = build_node_options(sample_graph)
    assert len(opts) == sample_graph.number_of_nodes()


def test_build_node_options_sorted(sample_graph):
    from dashboard.helpers import build_node_options
    opts = build_node_options(sample_graph)
    labels = [lbl.lower() for _, lbl in opts]
    assert labels == sorted(labels)


def test_build_node_options_all_ids_in_graph(sample_graph):
    from dashboard.helpers import build_node_options
    opts = build_node_options(sample_graph)
    for node_id, _ in opts:
        assert node_id in sample_graph


def test_build_node_options_alias_in_label(sample_graph):
    from dashboard.helpers import build_node_options
    opts = build_node_options(sample_graph)
    # The sample fixture has nodes with aliases (e.g. "bfx-lnd1")
    alias_nodes = [
        nid for nid in sample_graph
        if (sample_graph.nodes[nid].get("alias") or "").strip()
    ]
    assert alias_nodes, "Fixture should have at least one aliased node"
    opt_map = dict(opts)
    for nid in alias_nodes[:3]:
        alias = (sample_graph.nodes[nid].get("alias") or "").strip()
        assert alias in opt_map[nid], f"Alias '{alias}' should appear in label"


# ---------------------------------------------------------------------------
# get_node_centrality_row
# ---------------------------------------------------------------------------

def test_get_node_centrality_row_keys(sample_graph):
    from dashboard.helpers import get_node_centrality_row
    from lngraph.centrality import (
        betweenness_sampled,
        closeness_centrality,
        degree_centrality,
        eigenvector_centrality,
    )
    deg = degree_centrality(sample_graph)
    close = closeness_centrality(sample_graph)
    eig = eigenvector_centrality(sample_graph)
    btwn = betweenness_sampled(sample_graph, k=10)
    node_id = next(iter(sample_graph.nodes()))
    row = get_node_centrality_row(node_id, deg, close, eig, btwn)
    for key in ("degree", "capacity_weighted", "closeness", "eigenvector", "betweenness"):
        assert key in row, f"Expected key '{key}' in centrality row"


def test_get_node_centrality_row_values_are_floats(sample_graph):
    from dashboard.helpers import get_node_centrality_row
    from lngraph.centrality import (
        betweenness_sampled,
        closeness_centrality,
        degree_centrality,
        eigenvector_centrality,
    )
    deg = degree_centrality(sample_graph)
    close = closeness_centrality(sample_graph)
    eig = eigenvector_centrality(sample_graph)
    btwn = betweenness_sampled(sample_graph, k=10)
    node_id = next(iter(sample_graph.nodes()))
    row = get_node_centrality_row(node_id, deg, close, eig, btwn)
    for key, val in row.items():
        assert isinstance(val, float), f"Expected float for '{key}', got {type(val)}"


# ---------------------------------------------------------------------------
# format_route_table
# ---------------------------------------------------------------------------

def test_format_route_table_columns_empty():
    from dashboard.helpers import format_route_table
    df = format_route_table({})
    for col in ("hop", "from", "to", "channel_id", "fee_msat", "cltv_delta"):
        assert col in df.columns


def test_format_route_table_row_count():
    from dashboard.helpers import format_route_table
    route = {
        "hops": 2,
        "total_fee_msat": 10.0,
        "total_cltv": 80,
        "per_hop": [
            {"from": "aaa", "to": "bbb", "channel_id": "c1", "fee_msat": 5.0, "cltv_delta": 40},
            {"from": "bbb", "to": "ccc", "channel_id": "c2", "fee_msat": 5.0, "cltv_delta": 40},
        ],
    }
    df = format_route_table(route)
    assert len(df) == 2


def test_format_route_table_hop_column():
    from dashboard.helpers import format_route_table
    route = {
        "per_hop": [
            {"from": "a", "to": "b", "channel_id": "", "fee_msat": 1.0, "cltv_delta": 10},
        ],
    }
    df = format_route_table(route)
    assert list(df["hop"]) == [1]


# ---------------------------------------------------------------------------
# summarize_route
# ---------------------------------------------------------------------------

def test_summarize_route_keys():
    from dashboard.helpers import summarize_route
    route = {"hops": 2, "total_fee_msat": 10.0, "total_cltv": 80, "per_hop": []}
    s = summarize_route(route)
    for key in ("hops", "total_fee_msat", "total_cltv", "success_probability"):
        assert key in s


def test_summarize_route_success_prob_passthrough():
    from dashboard.helpers import summarize_route
    route = {"hops": 1, "total_fee_msat": 5.0, "total_cltv": 40, "per_hop": []}
    s = summarize_route(route, success_probability=0.75)
    assert s["success_probability"] == pytest.approx(0.75)


def test_summarize_route_no_success_prob():
    from dashboard.helpers import summarize_route
    route = {"hops": 1, "total_fee_msat": 5.0, "total_cltv": 40, "per_hop": []}
    s = summarize_route(route)
    assert s["success_probability"] is None
