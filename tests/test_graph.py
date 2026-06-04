"""Tests for graph.py — Phase 1.3.

Covers: simplify(), largest_connected_component(), to_igraph().
All tests run against the 5-node/4-channel synthetic fixture from conftest.
"""
import pytest
import networkx as nx

from lngraph import graph


# ---------------------------------------------------------------------------
# simplify()
# ---------------------------------------------------------------------------

def test_simplify_returns_digraph(synthetic_graph):
    G = graph.simplify(synthetic_graph)
    assert isinstance(G, nx.DiGraph)


def test_simplify_not_multidigraph(synthetic_graph):
    G = graph.simplify(synthetic_graph)
    assert not isinstance(G, nx.MultiDiGraph)


def test_simplify_preserves_node_count(synthetic_graph):
    G = graph.simplify(synthetic_graph)
    assert G.number_of_nodes() == synthetic_graph.number_of_nodes()


def test_simplify_edge_count_no_parallel(synthetic_graph):
    # synthetic graph has no parallel channels; 4 channels × 2 directions = 8 edges
    G = graph.simplify(synthetic_graph)
    assert G.number_of_edges() == 8


def test_simplify_parallel_channels_merged():
    # Two parallel channels between A→B; simplify must keep ONE edge per direction.
    MDG = nx.MultiDiGraph()
    MDG.add_node("A", alias="Alice")
    MDG.add_node("B", alias="Bob")
    MDG.add_edge("A", "B", channel_id="1x1x0", capacity=100_000)
    MDG.add_edge("A", "B", channel_id="1x1x1", capacity=200_000)
    MDG.add_edge("B", "A", channel_id="1x1x0", capacity=100_000)
    MDG.add_edge("B", "A", channel_id="1x1x1", capacity=200_000)
    G = graph.simplify(MDG)
    assert G.number_of_edges() == 2  # one A→B, one B→A


def test_simplify_capacity_summed_on_parallel():
    # Parallel channels should have their capacities summed on the merged edge.
    MDG = nx.MultiDiGraph()
    MDG.add_node("A")
    MDG.add_node("B")
    MDG.add_edge("A", "B", channel_id="1x1x0", capacity=100_000)
    MDG.add_edge("A", "B", channel_id="1x1x1", capacity=200_000)
    G = graph.simplify(MDG)
    assert G["A"]["B"]["capacity"] == 300_000


def test_simplify_preserves_node_attributes(synthetic_graph):
    G = graph.simplify(synthetic_graph)
    hub = list(synthetic_graph.nodes(data=True))[0]
    pub, attrs = hub
    assert G.nodes[pub]["alias"] == attrs["alias"]


def test_simplify_single_node_no_edges():
    MDG = nx.MultiDiGraph()
    MDG.add_node("A", alias="solo")
    G = graph.simplify(MDG)
    assert G.number_of_nodes() == 1
    assert G.number_of_edges() == 0


# ---------------------------------------------------------------------------
# largest_connected_component()
# ---------------------------------------------------------------------------

def test_lcc_returns_multidigraph(synthetic_graph):
    sub = graph.largest_connected_component(synthetic_graph)
    assert isinstance(sub, nx.MultiDiGraph)


def test_lcc_fully_connected_returns_all_nodes(synthetic_graph):
    # Synthetic graph is fully (weakly) connected; LCC == whole graph.
    sub = graph.largest_connected_component(synthetic_graph)
    assert sub.number_of_nodes() == synthetic_graph.number_of_nodes()


def test_lcc_returns_largest_component():
    # Build two disconnected components: 3-node chain and 1 isolated node.
    MDG = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        MDG.add_node(n)
    MDG.add_edge("A", "B", capacity=1000)
    MDG.add_edge("B", "A", capacity=1000)
    MDG.add_edge("B", "C", capacity=1000)
    MDG.add_edge("C", "B", capacity=1000)
    # D is isolated
    sub = graph.largest_connected_component(MDG)
    assert sub.number_of_nodes() == 3
    assert "D" not in sub


def test_lcc_empty_graph():
    MDG = nx.MultiDiGraph()
    sub = graph.largest_connected_component(MDG)
    assert isinstance(sub, nx.MultiDiGraph)
    assert sub.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# to_igraph()
# ---------------------------------------------------------------------------

def test_to_igraph_returns_igraph_graph(synthetic_graph):
    ig = graph.to_igraph(synthetic_graph)
    import igraph
    assert isinstance(ig, igraph.Graph)


def test_to_igraph_vertex_count(synthetic_graph):
    ig = graph.to_igraph(synthetic_graph)
    assert ig.vcount() == synthetic_graph.number_of_nodes()


def test_to_igraph_edge_count(synthetic_graph):
    ig = graph.to_igraph(synthetic_graph)
    assert ig.ecount() == synthetic_graph.number_of_edges()


def test_to_igraph_directed(synthetic_graph):
    ig = graph.to_igraph(synthetic_graph)
    assert ig.is_directed()


def test_to_igraph_preserves_pubkey_names(synthetic_graph):
    ig = graph.to_igraph(synthetic_graph)
    vertex_names = set(ig.vs["name"])
    assert vertex_names == set(synthetic_graph.nodes())


def test_to_igraph_capacity_attribute(synthetic_graph):
    ig = graph.to_igraph(synthetic_graph)
    assert "capacity" in ig.es.attribute_names()
    assert all(c > 0 for c in ig.es["capacity"])


def test_to_igraph_single_node():
    MDG = nx.MultiDiGraph()
    MDG.add_node("X", alias="solo")
    ig = graph.to_igraph(MDG)
    assert ig.vcount() == 1
    assert ig.ecount() == 0
