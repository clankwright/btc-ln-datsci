"""Tests for the describegraph ingester (Phase 1.2).

`ingest()` parses the canonical describegraph schema into a
networkx.MultiDiGraph carrying two directed edges per channel, each with its
own directional fee policy. This is the single internal contract every
analysis module downstream consumes.
"""
import json

import networkx as nx
import pytest

from lngraph import ingest


def test_ingest_returns_multidigraph(synthetic_describegraph):
    G = ingest.ingest(synthetic_describegraph)
    assert isinstance(G, nx.MultiDiGraph)


def test_ingest_node_count(synthetic_describegraph):
    G = ingest.ingest(synthetic_describegraph)
    assert G.number_of_nodes() == 5


def test_ingest_edge_count(synthetic_describegraph):
    # 4 channels × 2 directed edges
    G = ingest.ingest(synthetic_describegraph)
    assert G.number_of_edges() == 8


def test_ingest_edge_attributes(synthetic_describegraph):
    G = ingest.ingest(synthetic_describegraph)
    for _u, _v, data in G.edges(data=True):
        assert "channel_id" in data
        assert "capacity" in data
        assert isinstance(data["capacity"], int)
        assert "policy" in data


def test_ingest_directional_policies(synthetic_describegraph):
    G = ingest.ingest(synthetic_describegraph)
    ch = synthetic_describegraph["edges"][0]
    n1, n2 = ch["node1_pub"], ch["node2_pub"]
    # forward edge carries node1_policy; reverse carries node2_policy
    fwd = list(G.get_edge_data(n1, n2).values())[0]
    rev = list(G.get_edge_data(n2, n1).values())[0]
    assert fwd["policy"] == ch["node1_policy"]
    assert rev["policy"] == ch["node2_policy"]


def test_ingest_node_alias_preserved(synthetic_describegraph):
    G = ingest.ingest(synthetic_describegraph)
    hub = synthetic_describegraph["nodes"][0]
    assert G.nodes[hub["pub_key"]]["alias"] == hub["alias"]


def test_ingest_empty_graph():
    G = ingest.ingest({"nodes": [], "edges": []})
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() == 0
    assert G.number_of_edges() == 0


def test_ingest_adds_nodes_referenced_only_by_edges():
    # a channel may reference a peer absent from the nodes list (crawl boundary)
    dg = {
        "nodes": [{"pub_key": "a" * 66, "alias": "A"}],
        "edges": [{
            "channel_id": "1x1x0", "chan_point": "x:0", "capacity": "1000",
            "node1_pub": "a" * 66, "node2_pub": "b" * 66,
            "node1_policy": {"disabled": False},
            "node2_policy": {"disabled": False},
        }],
    }
    G = ingest.ingest(dg)
    assert ("b" * 66) in G
    assert G.number_of_nodes() == 2


def test_ingest_handles_missing_policy():
    dg = {
        "nodes": [{"pub_key": "a" * 66, "alias": "A"},
                  {"pub_key": "b" * 66, "alias": "B"}],
        "edges": [{
            "channel_id": "1x1x0", "chan_point": "x:0", "capacity": "1000",
            "node1_pub": "a" * 66, "node2_pub": "b" * 66,
            "node1_policy": None, "node2_policy": None,
        }],
    }
    G = ingest.ingest(dg)  # must not raise
    assert G.number_of_edges() == 2


def test_ingest_file_round_trips(synthetic_describegraph, tmp_path):
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(synthetic_describegraph))
    G = ingest.ingest_file(path)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 8


# ---------------------------------------------------------------------------
# end-to-end: crawl (mocked HTTP) → ingest. Validates the schema contract
# between the two modules shipped this cycle.
# ---------------------------------------------------------------------------

def test_crawl_then_ingest_round_trip(mock_http, tmp_path):
    from lngraph import fetch_mempool
    import lngraph.fetch_mempool as fm

    # neutralize sleeps for the live crawl
    fm.time.sleep = lambda *_a, **_k: None

    pub0 = "node00" + "a" * 64
    dg = fetch_mempool.crawl(
        seeds=[pub0], max_nodes=1, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap",
    )
    G = ingest.ingest(dg)
    assert isinstance(G, nx.MultiDiGraph)
    # one channel → two directed edges
    assert G.number_of_edges() == 2
    assert G.number_of_nodes() == 2
