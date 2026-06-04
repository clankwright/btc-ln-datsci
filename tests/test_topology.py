"""Tests for topology.py — Phase 1.4.

Covers: degree_distribution(), capacity_distribution(), gini(),
component_stats(), clustering_stats(), sampled_diameter(), coverage_report().

All numeric tests run against the 5-node/4-channel synthetic star graph
(hub=node00, 4 leaves) where values are hand-checkable.

Synthetic graph capacities (sats): 1_000_000, 2_000_000, 3_000_000, 4_000_000
Total capacity: 10_000_000 sats
Gini of [1,2,3,4]: 0.25  (derived: Σ|xi-xj| / (2n²*mean) = 10/(2*16*2.5) = 0.125...
  wait let me recalculate: n=4, mean=2.5, Σ|xi-xj|= 2*(|1-2|+|1-3|+|1-4|+|2-3|+|2-4|+|3-4|)
  = 2*(1+2+3+1+2+1) = 2*10 = 20; Gini = 20/(2*4*4*2.5) = 20/80 = 0.25)
"""
import random
from unittest.mock import patch

import pytest
import networkx as nx

from lngraph import topology


# ---------------------------------------------------------------------------
# gini()
# ---------------------------------------------------------------------------

def test_gini_perfect_equality():
    # All equal → Gini = 0
    assert topology.gini([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-9)


def test_gini_max_inequality():
    # One unit at one node, zero at all others (closest achievable with >0)
    # gini([1, 0, 0, 0]) = 0.75 for 4 elements
    assert topology.gini([1, 0, 0, 0]) == pytest.approx(0.75, abs=1e-6)


def test_gini_synthetic_capacities():
    # Capacities 1,2,3,4 (units) → Gini = 0.25
    assert topology.gini([1_000_000, 2_000_000, 3_000_000, 4_000_000]) == pytest.approx(0.25, abs=1e-6)


def test_gini_single_value():
    assert topology.gini([42]) == pytest.approx(0.0, abs=1e-9)


def test_gini_raises_on_empty():
    with pytest.raises(ValueError):
        topology.gini([])


def test_gini_all_zeros():
    # All zeros: perfectly equal (zero distribution), Gini = 0
    assert topology.gini([0, 0, 0]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# degree_distribution()
# ---------------------------------------------------------------------------

def test_degree_distribution_returns_dict(synthetic_graph):
    result = topology.degree_distribution(synthetic_graph)
    assert isinstance(result, dict)


def test_degree_distribution_keys(synthetic_graph):
    result = topology.degree_distribution(synthetic_graph)
    for key in ("in_degrees", "out_degrees", "total_degrees", "mean_in", "mean_out",
                "max_in", "max_out", "min_in", "min_out"):
        assert key in result, f"missing key: {key}"


def test_degree_distribution_hub_max_degree(synthetic_graph):
    # Hub (node00) has 4 in-edges and 4 out-edges; all leaves have 1 each.
    result = topology.degree_distribution(synthetic_graph)
    assert result["max_in"] == 4
    assert result["max_out"] == 4
    assert result["min_in"] == 1
    assert result["min_out"] == 1


def test_degree_distribution_mean(synthetic_graph):
    # 5 nodes: hub has 4, 4 leaves have 1 each → mean in-degree = (4+1+1+1+1)/5 = 1.6
    result = topology.degree_distribution(synthetic_graph)
    assert result["mean_in"] == pytest.approx(1.6, abs=1e-9)
    assert result["mean_out"] == pytest.approx(1.6, abs=1e-9)


def test_degree_distribution_total_degrees_count(synthetic_graph):
    # Returns a list/sequence of length == number of nodes
    result = topology.degree_distribution(synthetic_graph)
    assert len(result["total_degrees"]) == synthetic_graph.number_of_nodes()


# ---------------------------------------------------------------------------
# capacity_distribution()
# ---------------------------------------------------------------------------

def test_capacity_distribution_returns_dict(synthetic_graph):
    result = topology.capacity_distribution(synthetic_graph)
    assert isinstance(result, dict)


def test_capacity_distribution_keys(synthetic_graph):
    result = topology.capacity_distribution(synthetic_graph)
    for key in ("capacities", "total_capacity", "mean_capacity", "median_capacity",
                "max_capacity", "min_capacity", "gini_capacity"):
        assert key in result, f"missing key: {key}"


def test_capacity_distribution_channel_count(synthetic_graph):
    # 4 unique channels (not 8 directed edges)
    result = topology.capacity_distribution(synthetic_graph)
    assert len(result["capacities"]) == 4


def test_capacity_distribution_total(synthetic_graph):
    result = topology.capacity_distribution(synthetic_graph)
    assert result["total_capacity"] == 10_000_000


def test_capacity_distribution_gini(synthetic_graph):
    result = topology.capacity_distribution(synthetic_graph)
    assert result["gini_capacity"] == pytest.approx(0.25, abs=1e-6)


def test_capacity_distribution_max_min(synthetic_graph):
    result = topology.capacity_distribution(synthetic_graph)
    assert result["max_capacity"] == 4_000_000
    assert result["min_capacity"] == 1_000_000


# ---------------------------------------------------------------------------
# component_stats()
# ---------------------------------------------------------------------------

def test_component_stats_returns_dict(synthetic_graph):
    result = topology.component_stats(synthetic_graph)
    assert isinstance(result, dict)


def test_component_stats_keys(synthetic_graph):
    result = topology.component_stats(synthetic_graph)
    for key in ("num_weakly_connected", "num_strongly_connected",
                "largest_wcc_size", "largest_scc_size", "fraction_in_lcc"):
        assert key in result, f"missing key: {key}"


def test_component_stats_single_component(synthetic_graph):
    # Synthetic graph is fully connected
    result = topology.component_stats(synthetic_graph)
    assert result["num_weakly_connected"] == 1
    assert result["largest_wcc_size"] == 5
    assert result["fraction_in_lcc"] == pytest.approx(1.0)


def test_component_stats_disconnected():
    MDG = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        MDG.add_node(n)
    MDG.add_edge("A", "B", capacity=1000)
    MDG.add_edge("B", "A", capacity=1000)
    # C and D are isolated
    result = topology.component_stats(MDG)
    assert result["num_weakly_connected"] == 3
    assert result["largest_wcc_size"] == 2
    assert result["fraction_in_lcc"] == pytest.approx(0.5)


def test_component_stats_strongly_connected_star(synthetic_graph):
    # Star is strongly connected (hub↔leaves bidirectional)
    result = topology.component_stats(synthetic_graph)
    assert result["num_strongly_connected"] == 1
    assert result["largest_scc_size"] == 5


# ---------------------------------------------------------------------------
# clustering_stats()
# ---------------------------------------------------------------------------

def test_clustering_stats_returns_dict(synthetic_graph):
    result = topology.clustering_stats(synthetic_graph)
    assert isinstance(result, dict)


def test_clustering_stats_keys(synthetic_graph):
    result = topology.clustering_stats(synthetic_graph)
    for key in ("average_clustering", "transitivity"):
        assert key in result, f"missing key: {key}"


def test_clustering_stats_pure_star_zero(synthetic_graph):
    # A pure star has 0 triangles → clustering = 0
    result = topology.clustering_stats(synthetic_graph)
    assert result["average_clustering"] == pytest.approx(0.0, abs=1e-9)


def test_clustering_stats_complete_graph():
    # Complete undirected 4-node graph → all clustering = 1
    MDG = nx.MultiDiGraph()
    nodes = ["A", "B", "C", "D"]
    for n in nodes:
        MDG.add_node(n)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            MDG.add_edge(u, v, capacity=1000)
            MDG.add_edge(v, u, capacity=1000)
    result = topology.clustering_stats(MDG)
    assert result["average_clustering"] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# sampled_diameter()
# ---------------------------------------------------------------------------

def test_sampled_diameter_returns_dict(synthetic_graph):
    result = topology.sampled_diameter(synthetic_graph, k=4, seed=0)
    assert isinstance(result, dict)


def test_sampled_diameter_keys(synthetic_graph):
    result = topology.sampled_diameter(synthetic_graph, k=4, seed=0)
    for key in ("estimated_diameter", "avg_path_length", "sample_size"):
        assert key in result, f"missing key: {key}"


def test_sampled_diameter_star_diameter(synthetic_graph):
    # In star: leaf-to-leaf shortest path = 2 hops; hub-to-leaf = 1 hop
    result = topology.sampled_diameter(synthetic_graph, k=10, seed=42)
    assert result["estimated_diameter"] == 2


def test_sampled_diameter_star_avg_path(synthetic_graph):
    # All hub-leaf pairs: distance 1 (4 pairs each direction = 8)
    # All leaf-leaf pairs: distance 2 (4*3=12 ordered pairs via hub)
    # Avg = (8*1 + 12*2) / 20 = (8+24)/20 = 1.6
    result = topology.sampled_diameter(synthetic_graph, k=20, seed=42)
    assert result["avg_path_length"] == pytest.approx(1.6, abs=0.1)


def test_sampled_diameter_uses_lcc(synthetic_graph):
    # Disconnected graph: function must restrict to LCC to avoid inf paths
    MDG = nx.MultiDiGraph()
    MDG.add_node("X")  # isolated
    for n in synthetic_graph.nodes():
        MDG.add_node(n)
    for u, v, d in synthetic_graph.edges(data=True):
        MDG.add_edge(u, v, **d)
    result = topology.sampled_diameter(MDG, k=4, seed=0)
    # Should not raise; should work on the LCC (5-node component)
    assert result["estimated_diameter"] >= 1


def test_sampled_diameter_sample_size_capped(synthetic_graph):
    # k larger than node count: sample_size ≤ n*(n-1)
    result = topology.sampled_diameter(synthetic_graph, k=1000, seed=0)
    n = synthetic_graph.number_of_nodes()
    assert result["sample_size"] <= n * (n - 1)


def test_sampled_diameter_no_quadratic_pairs_list():
    # sampled_diameter must NOT build an O(V²) all-pairs list before sampling.
    # It must call rng.sample(nodes, 2) k times rather than
    # rng.sample(all_pairs, k) where all_pairs has n*(n-1) elements.
    G = nx.MultiDiGraph()
    node_ids = [str(i) for i in range(20)]
    for u in node_ids:
        for v in node_ids:
            if u != v:
                G.add_edge(u, v, capacity=1_000_000)
    n = len(node_ids)

    max_population_len = 0
    original_sample = random.Random.sample

    def tracking_sample(self, population, k):
        nonlocal max_population_len
        if hasattr(population, "__len__"):
            max_population_len = max(max_population_len, len(population))
        return original_sample(self, population, k)

    with patch.object(random.Random, "sample", tracking_sample):
        result = topology.sampled_diameter(G, k=5, seed=0)

    assert result["sample_size"] == 5
    # Must never build the O(V²) pairs list — population passed to sample must be
    # the node list (len == n), not the pairs list (len == n*(n-1) == 380).
    assert max_population_len <= n, (
        f"sampled_diameter built an O(V²) population of length {max_population_len}; "
        f"expected ≤ n={n}"
    )


# ---------------------------------------------------------------------------
# coverage_report()
# ---------------------------------------------------------------------------

MOCK_STATISTICS = {
    "latest": {
        "node_count": 17508,
        "channel_count": 41343,
        "total_capacity": 5_200_000_000,
    }
}


def test_coverage_report_returns_dict(synthetic_graph):
    result = topology.coverage_report(synthetic_graph, MOCK_STATISTICS)
    assert isinstance(result, dict)


def test_coverage_report_keys(synthetic_graph):
    result = topology.coverage_report(synthetic_graph, MOCK_STATISTICS)
    for key in ("crawled_nodes", "crawled_channels", "network_nodes",
                "network_channels", "node_coverage_pct", "channel_coverage_pct"):
        assert key in result, f"missing key: {key}"


def test_coverage_report_crawled_counts(synthetic_graph):
    result = topology.coverage_report(synthetic_graph, MOCK_STATISTICS)
    assert result["crawled_nodes"] == 5
    assert result["crawled_channels"] == 4  # 4 unique channels, not 8 directed edges


def test_coverage_report_network_totals(synthetic_graph):
    result = topology.coverage_report(synthetic_graph, MOCK_STATISTICS)
    assert result["network_nodes"] == 17508
    assert result["network_channels"] == 41343


def test_coverage_report_coverage_pct(synthetic_graph):
    result = topology.coverage_report(synthetic_graph, MOCK_STATISTICS)
    assert result["node_coverage_pct"] == pytest.approx(5 / 17508 * 100, rel=1e-4)
    assert result["channel_coverage_pct"] == pytest.approx(4 / 41343 * 100, rel=1e-4)


def test_coverage_report_missing_statistics(synthetic_graph):
    # Should raise if statistics data doesn't have the expected keys
    with pytest.raises((KeyError, ValueError)):
        topology.coverage_report(synthetic_graph, {})
