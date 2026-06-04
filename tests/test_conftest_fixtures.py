"""Tests for the conftest synthetic fixtures (Phase 0.2)."""
import networkx as nx


def test_synthetic_describegraph_structure(synthetic_describegraph):
    dg = synthetic_describegraph
    assert "nodes" in dg and "edges" in dg
    assert len(dg["nodes"]) == 5
    assert len(dg["edges"]) == 4


def test_synthetic_describegraph_channel_fields(synthetic_describegraph):
    ch = synthetic_describegraph["edges"][0]
    required = {"channel_id", "chan_point", "capacity", "node1_pub", "node2_pub",
                "node1_policy", "node2_policy"}
    assert required.issubset(ch.keys())


def test_synthetic_graph_is_multidigraph(synthetic_graph):
    assert isinstance(synthetic_graph, nx.MultiDiGraph)


def test_synthetic_graph_node_count(synthetic_graph):
    assert synthetic_graph.number_of_nodes() == 5


def test_synthetic_graph_edge_count(synthetic_graph):
    # 4 channels × 2 directed edges each
    assert synthetic_graph.number_of_edges() == 8


def test_synthetic_graph_hub_degree(synthetic_graph):
    hub = list(synthetic_graph.nodes)[0]
    # Hub has 4 out-edges and 4 in-edges
    assert synthetic_graph.out_degree(hub) == 4
    assert synthetic_graph.in_degree(hub) == 4


def test_synthetic_graph_leaf_degree(synthetic_graph):
    hub = list(synthetic_graph.nodes)[0]
    for leaf in list(synthetic_graph.nodes)[1:]:
        assert synthetic_graph.out_degree(leaf) == 1
        assert synthetic_graph.in_degree(leaf) == 1


def test_synthetic_graph_capacity_varies(synthetic_graph):
    capacities = {
        d["capacity"]
        for _, _, d in synthetic_graph.edges(data=True)
    }
    assert len(capacities) > 1, "Edge capacities should differ across channels"
