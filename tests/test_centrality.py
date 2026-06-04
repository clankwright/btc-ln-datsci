"""Tests for centrality.py — Phase 2.1.

Covers: degree_centrality(), closeness_centrality(), eigenvector_centrality(),
betweenness_sampled(), betweenness_exact(), betweenness_fee().

All numeric tests run against the 5-node/4-channel synthetic star fixture.

Hand-checkable values for the simplified DiGraph (5-node star, hub=node00):
  Synthetic capacities: leaf01=1M, leaf02=2M, leaf03=3M, leaf04=4M sats
  Policy: fee_base_msat=1000, fee_rate_milli_msat=500 on all edges

  degree centrality (normalized by n-1=4):
    hub in/out = 4/4 = 1.0; each leaf = 1/4 = 0.25
  total degree centrality (avg of in+out, normalized by n-1):
    hub = (1.0 + 1.0) / 2 = 1.0; each leaf = 0.25
  capacity-weighted degree (normalized by max node total-capacity=20M):
    hub  = (4*(1M+2M+3M+4M) both dirs) = 20M → 1.0
    leaf01 = (1M in + 1M out) = 2M  → 0.1
    leaf02 = 4M → 0.2, leaf03 = 6M → 0.3, leaf04 = 8M → 0.4
  closeness centrality (wf_improved, directed, from hub):
    hub = 1.0  (reaches 4 leaves in 1 hop)
    leaf = 4/7 ≈ 0.571 (hub 1 hop, other leaves 2 hops)
  betweenness (directed, normalized by (n-1)*(n-2)=12):
    hub = 12/12 = 1.0 (all 12 ordered leaf-pair paths pass through hub)
    each leaf = 0.0
  fee-based betweenness: same ordering as hop-count (uniform policy)
"""
import pytest
import networkx as nx

from lngraph import centrality


# ---------------------------------------------------------------------------
# degree_centrality()
# ---------------------------------------------------------------------------

def test_degree_centrality_returns_expected_keys(synthetic_graph):
    result = centrality.degree_centrality(synthetic_graph)
    for key in ("in_degree", "out_degree", "total_degree", "capacity_weighted"):
        assert key in result, f"missing key: {key}"


def test_degree_centrality_hub_in_degree_is_one(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.degree_centrality(synthetic_graph)
    assert result["in_degree"][hub] == pytest.approx(1.0, abs=1e-9)


def test_degree_centrality_hub_out_degree_is_one(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.degree_centrality(synthetic_graph)
    assert result["out_degree"][hub] == pytest.approx(1.0, abs=1e-9)


def test_degree_centrality_leaf_in_degree(synthetic_graph):
    leaf = list(synthetic_graph.nodes())[1]
    result = centrality.degree_centrality(synthetic_graph)
    assert result["in_degree"][leaf] == pytest.approx(0.25, abs=1e-9)


def test_degree_centrality_hub_max_total(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.degree_centrality(synthetic_graph)
    leaves = list(synthetic_graph.nodes())[1:]
    assert all(
        result["total_degree"][hub] > result["total_degree"][leaf]
        for leaf in leaves
    )


def test_degree_centrality_hub_capacity_weighted_is_one(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.degree_centrality(synthetic_graph)
    assert result["capacity_weighted"][hub] == pytest.approx(1.0, abs=1e-9)


def test_degree_centrality_leaf01_capacity_weighted(synthetic_graph):
    # leaf01 has capacity 1M in+out = 2M; max is hub 20M → 0.1
    nodes = list(synthetic_graph.nodes())
    leaf01 = nodes[1]
    result = centrality.degree_centrality(synthetic_graph)
    assert result["capacity_weighted"][leaf01] == pytest.approx(0.1, abs=1e-6)


def test_degree_centrality_leaves_capacity_weighted_ordered(synthetic_graph):
    # leaf capacities increase: leaf01 < leaf02 < leaf03 < leaf04
    nodes = list(synthetic_graph.nodes())
    result = centrality.degree_centrality(synthetic_graph)
    caps = [result["capacity_weighted"][nodes[i]] for i in range(1, 5)]
    assert caps == sorted(caps)


def test_degree_centrality_empty_graph():
    G = nx.MultiDiGraph()
    result = centrality.degree_centrality(G)
    for key in ("in_degree", "out_degree", "total_degree", "capacity_weighted"):
        assert result[key] == {}


def test_degree_centrality_single_node():
    # networkx in_degree_centrality returns 1 (by convention) for a single-node graph
    G = nx.MultiDiGraph()
    G.add_node("X", alias="solo")
    result = centrality.degree_centrality(G)
    assert "X" in result["in_degree"]
    assert isinstance(result["in_degree"]["X"], (int, float))


# ---------------------------------------------------------------------------
# closeness_centrality()
# ---------------------------------------------------------------------------

def test_closeness_centrality_returns_dict_of_floats(synthetic_graph):
    result = centrality.closeness_centrality(synthetic_graph)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_closeness_centrality_hub_is_max(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.closeness_centrality(synthetic_graph)
    assert result[hub] == max(result.values())


def test_closeness_centrality_hub_is_one(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.closeness_centrality(synthetic_graph)
    assert result[hub] == pytest.approx(1.0, abs=1e-6)


def test_closeness_centrality_leaf_value(synthetic_graph):
    # leaf closeness = 4/7 ≈ 0.5714 (hub at 1 hop, 3 other leaves at 2 hops)
    leaf = list(synthetic_graph.nodes())[1]
    result = centrality.closeness_centrality(synthetic_graph)
    assert result[leaf] == pytest.approx(4 / 7, abs=1e-4)


def test_closeness_centrality_all_in_unit_interval(synthetic_graph):
    result = centrality.closeness_centrality(synthetic_graph)
    assert all(0.0 <= v <= 1.0 for v in result.values())


def test_closeness_centrality_empty_graph():
    G = nx.MultiDiGraph()
    assert centrality.closeness_centrality(G) == {}


# ---------------------------------------------------------------------------
# eigenvector_centrality()
# ---------------------------------------------------------------------------

def test_eigenvector_centrality_returns_dict_of_floats(synthetic_graph):
    result = centrality.eigenvector_centrality(synthetic_graph)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_eigenvector_centrality_hub_is_max(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.eigenvector_centrality(synthetic_graph)
    assert result[hub] == max(result.values())


def test_eigenvector_centrality_leaves_equal(synthetic_graph):
    # All leaves have the same degree structure → equal eigenvector centrality
    nodes = list(synthetic_graph.nodes())
    result = centrality.eigenvector_centrality(synthetic_graph)
    leaf_vals = [result[nodes[i]] for i in range(1, 5)]
    assert max(leaf_vals) == pytest.approx(min(leaf_vals), abs=1e-6)


def test_eigenvector_centrality_all_positive(synthetic_graph):
    result = centrality.eigenvector_centrality(synthetic_graph)
    assert all(v > 0 for v in result.values())


def test_eigenvector_centrality_empty_graph():
    G = nx.MultiDiGraph()
    assert centrality.eigenvector_centrality(G) == {}


# ---------------------------------------------------------------------------
# betweenness_sampled()
# ---------------------------------------------------------------------------

def test_betweenness_sampled_returns_dict_of_floats(synthetic_graph):
    result = centrality.betweenness_sampled(synthetic_graph)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_betweenness_sampled_hub_is_one(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.betweenness_sampled(synthetic_graph)
    assert result[hub] == pytest.approx(1.0, abs=1e-6)


def test_betweenness_sampled_leaves_are_zero(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    result = centrality.betweenness_sampled(synthetic_graph)
    for leaf in nodes[1:]:
        assert result[leaf] == pytest.approx(0.0, abs=1e-6)


def test_betweenness_sampled_all_in_unit_interval(synthetic_graph):
    result = centrality.betweenness_sampled(synthetic_graph)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in result.values())


def test_betweenness_sampled_reproducible_with_seed(synthetic_graph):
    r1 = centrality.betweenness_sampled(synthetic_graph, k=3, seed=7)
    r2 = centrality.betweenness_sampled(synthetic_graph, k=3, seed=7)
    assert r1 == r2


def test_betweenness_sampled_empty_graph():
    G = nx.MultiDiGraph()
    assert centrality.betweenness_sampled(G) == {}


# ---------------------------------------------------------------------------
# betweenness_exact()
# ---------------------------------------------------------------------------

def test_betweenness_exact_returns_dict_of_floats(synthetic_graph):
    result = centrality.betweenness_exact(synthetic_graph)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_betweenness_exact_hub_is_one(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.betweenness_exact(synthetic_graph)
    assert result[hub] == pytest.approx(1.0, abs=1e-6)


def test_betweenness_exact_leaves_are_zero(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    result = centrality.betweenness_exact(synthetic_graph)
    for leaf in nodes[1:]:
        assert result[leaf] == pytest.approx(0.0, abs=1e-6)


def test_betweenness_exact_matches_sampled_on_small_graph(synthetic_graph):
    # On the 5-node star (exact and sampled k=n both compute every pivot)
    exact = centrality.betweenness_exact(synthetic_graph)
    sampled = centrality.betweenness_sampled(synthetic_graph)
    for node in exact:
        assert exact[node] == pytest.approx(sampled[node], abs=1e-5)


def test_betweenness_exact_all_in_unit_interval(synthetic_graph):
    result = centrality.betweenness_exact(synthetic_graph)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in result.values())


def test_betweenness_exact_empty_graph():
    G = nx.MultiDiGraph()
    assert centrality.betweenness_exact(G) == {}


# ---------------------------------------------------------------------------
# betweenness_fee()
# ---------------------------------------------------------------------------

def test_betweenness_fee_returns_dict_of_floats(synthetic_graph):
    result = centrality.betweenness_fee(synthetic_graph)
    assert isinstance(result, dict)
    assert all(isinstance(v, float) for v in result.values())


def test_betweenness_fee_hub_is_max(synthetic_graph):
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.betweenness_fee(synthetic_graph)
    assert result[hub] == max(result.values())


def test_betweenness_fee_hub_is_one_uniform_policy(synthetic_graph):
    # Uniform fee policy → fee cost proportional to hop count → same result as hop betweenness
    hub = list(synthetic_graph.nodes())[0]
    result = centrality.betweenness_fee(synthetic_graph)
    assert result[hub] == pytest.approx(1.0, abs=1e-6)


def test_betweenness_fee_leaves_are_zero(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    result = centrality.betweenness_fee(synthetic_graph)
    for leaf in nodes[1:]:
        assert result[leaf] == pytest.approx(0.0, abs=1e-6)


def test_betweenness_fee_reproducible_with_seed(synthetic_graph):
    r1 = centrality.betweenness_fee(synthetic_graph, seed=42)
    r2 = centrality.betweenness_fee(synthetic_graph, seed=42)
    assert r1 == r2


def test_betweenness_fee_all_in_unit_interval(synthetic_graph):
    result = centrality.betweenness_fee(synthetic_graph)
    assert all(0.0 <= v <= 1.0 + 1e-9 for v in result.values())


def test_betweenness_fee_empty_graph():
    G = nx.MultiDiGraph()
    assert centrality.betweenness_fee(G) == {}


def test_betweenness_fee_disabled_edges_dropped():
    # A graph where all edges from leaf01 are disabled — those paths should be excluded
    G = nx.MultiDiGraph()
    G.add_node("hub")
    G.add_node("leaf01")
    G.add_node("leaf02")
    disabled_policy = {"fee_base_msat": "1000", "fee_rate_milli_msat": "500",
                       "time_lock_delta": 40, "disabled": True}
    active_policy = {"fee_base_msat": "1000", "fee_rate_milli_msat": "500",
                     "time_lock_delta": 40, "disabled": False}
    # hub ↔ leaf02 (active), hub → leaf01 and leaf01 → hub (disabled)
    G.add_edge("hub", "leaf01", capacity=1_000_000, policy=disabled_policy)
    G.add_edge("leaf01", "hub", capacity=1_000_000, policy=disabled_policy)
    G.add_edge("hub", "leaf02", capacity=2_000_000, policy=active_policy)
    G.add_edge("leaf02", "hub", capacity=2_000_000, policy=active_policy)
    result = centrality.betweenness_fee(G)
    # leaf01 is isolated (all its edges disabled) → betweenness 0
    assert result["leaf01"] == pytest.approx(0.0, abs=1e-9)


def test_betweenness_fee_nonuniform_fees_differ_from_hop_count():
    """Asymmetric fees route all paths through the cheap node, not equally like hop count.

    Graph: S→A (cheap), A→T (cheap), S→B (expensive), B→T (cheap).
    Both paths S→A→T and S→B→T are 2 hops (equal under hop count), but only
    S→A→T is cheapest under fee routing, so betweenness_fee ranks A above B
    while betweenness_sampled ranks them equally.
    """
    G = nx.MultiDiGraph()
    cheap = {"fee_base_msat": "1", "fee_rate_milli_msat": "0", "disabled": False}
    expensive = {"fee_base_msat": "1000", "fee_rate_milli_msat": "0", "disabled": False}
    G.add_edge("S", "A", capacity=1_000_000, policy=cheap)
    G.add_edge("A", "T", capacity=1_000_000, policy=cheap)
    G.add_edge("S", "B", capacity=1_000_000, policy=expensive)
    G.add_edge("B", "T", capacity=1_000_000, policy=cheap)

    fee_result = centrality.betweenness_fee(G)
    hop_result = centrality.betweenness_sampled(G)

    # Fee routing: S always takes S→A→T (cheapest) → A has betweenness, B has none
    assert fee_result["A"] > fee_result["B"]
    # Hop count: both paths are 2 hops → A and B ranked equally
    assert hop_result["A"] == pytest.approx(hop_result["B"], abs=1e-6)
    # Fee and hop results differ for A
    assert not (fee_result["A"] == pytest.approx(hop_result["A"], abs=1e-3))


def test_betweenness_fee_none_policy_equals_hop_count():
    """betweenness_fee on an all-None-policy graph equals betweenness_sampled.

    When no channel has a fee policy, every hop should carry an equal positive
    unit cost (epsilon), making fee-based Dijkstra behave identically to
    unweighted BFS. Tests the fix for the all-zero-weight Dijkstra miscount.

    Graph: bidirectional 5-node path A↔B↔C↔D↔E, all policy=None.
    Hand-checkable hop-count betweenness (norm=(n-1)*(n-2)=12):
      A=E=0, B=D=6/12=0.5, C=8/12≈0.667.
    """
    G = nx.MultiDiGraph()
    for u, v in [("A","B"),("B","A"),("B","C"),("C","B"),
                 ("C","D"),("D","C"),("D","E"),("E","D")]:
        G.add_edge(u, v, capacity=1_000_000, policy=None)

    fee_result = centrality.betweenness_fee(G)
    hop_result = centrality.betweenness_sampled(G)

    for node in fee_result:
        assert fee_result[node] == pytest.approx(hop_result[node], abs=1e-6), (
            f"Node {node}: fee={fee_result[node]:.4f} vs hop={hop_result[node]:.4f}"
        )
