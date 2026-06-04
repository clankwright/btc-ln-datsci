"""Tests for routing.py — Phase 3.2.

Covers: route_cost(), find_route(), find_k_routes(), fee_vs_hops().

Hand-checkable values for the 5-node synthetic star (hub=node00):
  Policy on all edges: fee_base_msat=1000, fee_rate_milli_msat=500, time_lock_delta=40
  amount_msat=100_000

  Per-edge fee_msat (no epsilon, just LN fee):
    1000 + 100_000 * 500 / 1_000_000 = 1000 + 50 = 1050 msat

  1-hop route (e.g. hub → leaf01):
    total_fee_msat = 1050, total_cltv = 40, hops = 1

  2-hop route (e.g. leaf01 → hub → leaf02):
    total_fee_msat = 2100, total_cltv = 80, hops = 2
"""
import pytest
import networkx as nx

from lngraph import routing


# ---------------------------------------------------------------------------
# Helpers / extra fixtures
# ---------------------------------------------------------------------------

def _make_path_graph():
    """A→B→C→D→E linear path graph with known fee policies.

    Useful for testing multi-hop routes and k-shortest comparisons where
    the simple star topology offers only one route between any two leaves.

    Policy: fee_base=500, fee_rate=200, cltv_delta=20 on all edges.
    """
    G = nx.MultiDiGraph()
    nodes = ["A", "B", "C", "D", "E"]
    for n in nodes:
        G.add_node(n)
    policy = {
        "fee_base_msat": "500",
        "fee_rate_milli_msat": "200",
        "time_lock_delta": 20,
        "disabled": False,
    }
    for u, v in zip(nodes[:-1], nodes[1:]):
        # forward direction
        G.add_edge(u, v, channel_id=f"{u}{v}", capacity=1_000_000, policy=dict(policy))
        # reverse direction
        G.add_edge(v, u, channel_id=f"{u}{v}", capacity=1_000_000, policy=dict(policy))
    return G


def _make_diamond_graph():
    """Diamond graph: A→B→D and A→C→D (two distinct paths of equal length).

    B-path policy: fee_base=1000, fee_rate=0, cltv=10
    C-path policy: fee_base=2000, fee_rate=0, cltv=5
    Used for k-shortest path ordering tests.
    """
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n)

    cheap = {"fee_base_msat": "1000", "fee_rate_milli_msat": "0", "time_lock_delta": 10, "disabled": False}
    expensive = {"fee_base_msat": "2000", "fee_rate_milli_msat": "0", "time_lock_delta": 5, "disabled": False}

    G.add_edge("A", "B", channel_id="AB", capacity=1_000_000, policy=cheap)
    G.add_edge("B", "D", channel_id="BD", capacity=1_000_000, policy=cheap)
    G.add_edge("A", "C", channel_id="AC", capacity=1_000_000, policy=expensive)
    G.add_edge("C", "D", channel_id="CD", capacity=1_000_000, policy=expensive)
    return G


# ---------------------------------------------------------------------------
# route_cost()
# ---------------------------------------------------------------------------

def test_route_cost_no_policy():
    """No policy → epsilon only."""
    cost = routing.route_cost(None, 100_000, cltv_risk_weight=0.0)
    assert cost == pytest.approx(routing._EPSILON)


def test_route_cost_base_fee_only():
    policy = {"fee_base_msat": "2000", "fee_rate_milli_msat": "0", "time_lock_delta": 0}
    cost = routing.route_cost(policy, 100_000, cltv_risk_weight=0.0)
    assert cost == pytest.approx(routing._EPSILON + 2000)


def test_route_cost_proportional_fee():
    policy = {"fee_base_msat": "0", "fee_rate_milli_msat": "1000", "time_lock_delta": 0}
    # 100_000 * 1000 / 1_000_000 = 100 msat
    cost = routing.route_cost(policy, 100_000, cltv_risk_weight=0.0)
    assert cost == pytest.approx(routing._EPSILON + 100.0)


def test_route_cost_combined_fee():
    policy = {"fee_base_msat": "1000", "fee_rate_milli_msat": "500", "time_lock_delta": 0}
    # 1000 + 100_000*500/1_000_000 = 1000 + 50 = 1050
    cost = routing.route_cost(policy, 100_000, cltv_risk_weight=0.0)
    assert cost == pytest.approx(routing._EPSILON + 1050.0)


def test_route_cost_cltv_risk_added():
    policy = {"fee_base_msat": "0", "fee_rate_milli_msat": "0", "time_lock_delta": 40}
    cost_zero = routing.route_cost(policy, 100_000, cltv_risk_weight=0.0)
    cost_one = routing.route_cost(policy, 100_000, cltv_risk_weight=1.0)
    assert cost_one == pytest.approx(cost_zero + 40.0)


def test_route_cost_empty_policy_dict():
    """Empty dict → all fields default to 0 → epsilon only (no cltv risk)."""
    cost = routing.route_cost({}, 100_000, cltv_risk_weight=1.0)
    assert cost == pytest.approx(routing._EPSILON)


# ---------------------------------------------------------------------------
# find_route() — basic structure
# ---------------------------------------------------------------------------

def test_find_route_result_keys(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf)
    assert result is not None
    for key in ("path", "hops", "total_fee_msat", "total_cltv", "per_hop"):
        assert key in result, f"missing key: {key}"


def test_find_route_returns_none_no_path():
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    assert routing.find_route(G, "A", "B") is None


def test_find_route_returns_none_missing_node():
    G = nx.MultiDiGraph()
    G.add_node("A")
    assert routing.find_route(G, "A", "MISSING") is None


# ---------------------------------------------------------------------------
# find_route() — 1-hop route (hub → leaf01)
# ---------------------------------------------------------------------------

def test_find_route_one_hop_hops(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    assert result["hops"] == 1


def test_find_route_one_hop_path_length(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    assert len(result["path"]) == 2
    assert result["path"][0] == hub
    assert result["path"][-1] == leaf


def test_find_route_one_hop_total_fee(synthetic_graph):
    # fee = fee_base + amount*rate/1e6 = 1000 + 100_000*500/1_000_000 = 1050 msat
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    assert result["total_fee_msat"] == pytest.approx(1050.0)


def test_find_route_one_hop_total_cltv(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    assert result["total_cltv"] == 40


def test_find_route_one_hop_per_hop_count(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    assert len(result["per_hop"]) == 1


def test_find_route_one_hop_per_hop_structure(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    hop = result["per_hop"][0]
    for key in ("from", "to", "channel_id", "fee_msat", "cltv_delta"):
        assert key in hop, f"per_hop entry missing key: {key}"
    assert hop["from"] == hub
    assert hop["to"] == leaf


# ---------------------------------------------------------------------------
# find_route() — 2-hop route (leaf01 → hub → leaf02)
# ---------------------------------------------------------------------------

def test_find_route_two_hop_hops(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    leaf01, leaf02 = nodes[1], nodes[2]
    result = routing.find_route(synthetic_graph, leaf01, leaf02, amount_msat=100_000)
    assert result is not None
    assert result["hops"] == 2


def test_find_route_two_hop_total_fee(synthetic_graph):
    # 2 hops × 1050 msat each = 2100 msat
    nodes = list(synthetic_graph.nodes())
    leaf01, leaf02 = nodes[1], nodes[2]
    result = routing.find_route(synthetic_graph, leaf01, leaf02, amount_msat=100_000)
    assert result["total_fee_msat"] == pytest.approx(2100.0)


def test_find_route_two_hop_total_cltv(synthetic_graph):
    # 2 hops × 40 cltv each = 80
    nodes = list(synthetic_graph.nodes())
    leaf01, leaf02 = nodes[1], nodes[2]
    result = routing.find_route(synthetic_graph, leaf01, leaf02, amount_msat=100_000)
    assert result["total_cltv"] == 80


def test_find_route_two_hop_per_hop_count(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    leaf01, leaf02 = nodes[1], nodes[2]
    result = routing.find_route(synthetic_graph, leaf01, leaf02, amount_msat=100_000)
    assert len(result["per_hop"]) == 2


# ---------------------------------------------------------------------------
# find_route() — disabled edges excluded
# ---------------------------------------------------------------------------

def test_find_route_disabled_direct_falls_back_to_none():
    """If the only path is disabled, find_route returns None."""
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B", channel_id="AB", capacity=1_000_000,
               policy={"fee_base_msat": "0", "fee_rate_milli_msat": "0",
                        "time_lock_delta": 0, "disabled": True})
    assert routing.find_route(G, "A", "B") is None


def test_find_route_disabled_edge_not_used(synthetic_graph):
    """Disabling hub→leaf02 forces leaf01→leaf02 route to use a different leaf (no path)."""
    nodes = list(synthetic_graph.nodes())
    hub, leaf01, leaf02 = nodes[0], nodes[1], nodes[2]
    # Disable hub→leaf02
    for u, v, k, data in synthetic_graph.edges(keys=True, data=True):
        if u == hub and v == leaf02:
            synthetic_graph[u][v][k]["policy"] = dict(data.get("policy") or {})
            synthetic_graph[u][v][k]["policy"]["disabled"] = True
    result = routing.find_route(synthetic_graph, leaf01, leaf02, amount_msat=100_000)
    assert result is None


# ---------------------------------------------------------------------------
# find_route() — path graph (multi-hop verification)
# ---------------------------------------------------------------------------

def test_find_route_path_graph_four_hops():
    G = _make_path_graph()
    result = routing.find_route(G, "A", "E", amount_msat=100_000)
    assert result is not None
    assert result["hops"] == 4
    assert result["path"] == ["A", "B", "C", "D", "E"]


def test_find_route_path_graph_fee_calculation():
    G = _make_path_graph()
    # policy: fee_base=500, fee_rate=200; 100_000*200/1_000_000=20; fee/hop=520
    result = routing.find_route(G, "A", "E", amount_msat=100_000)
    assert result["total_fee_msat"] == pytest.approx(4 * 520.0)


def test_find_route_path_graph_cltv():
    G = _make_path_graph()
    result = routing.find_route(G, "A", "E", amount_msat=100_000)
    # 4 hops × cltv_delta=20 = 80
    assert result["total_cltv"] == 4 * 20


# ---------------------------------------------------------------------------
# find_k_routes()
# ---------------------------------------------------------------------------

def test_find_k_routes_returns_list(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_k_routes(synthetic_graph, hub, leaf, k=3)
    assert isinstance(result, list)


def test_find_k_routes_one_path_star(synthetic_graph):
    """In the star graph, only 1 simple path exists between hub and leaf."""
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.find_k_routes(synthetic_graph, hub, leaf, k=5)
    assert len(result) >= 1


def test_find_k_routes_no_path_returns_empty():
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    result = routing.find_k_routes(G, "A", "B", k=3)
    assert result == []


def test_find_k_routes_result_keys(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    routes = routing.find_k_routes(synthetic_graph, hub, leaf, k=1)
    assert len(routes) >= 1
    for key in ("path", "hops", "total_fee_msat", "total_cltv", "per_hop"):
        assert key in routes[0], f"missing key: {key}"


def test_find_k_routes_diamond_two_paths():
    """Diamond graph has exactly 2 simple paths; k=5 returns 2."""
    G = _make_diamond_graph()
    routes = routing.find_k_routes(G, "A", "D", k=5)
    assert len(routes) == 2


def test_find_k_routes_diamond_ordered_by_cost():
    """Cheaper path (via B: base=1000 each leg) before expensive (via C: base=2000 each leg)."""
    G = _make_diamond_graph()
    routes = routing.find_k_routes(G, "A", "D", k=2)
    assert routes[0]["total_fee_msat"] <= routes[1]["total_fee_msat"]


def test_find_k_routes_diamond_first_path_cheaper():
    """Via B: 2×1000=2000 msat; via C: 2×2000=4000 msat."""
    G = _make_diamond_graph()
    routes = routing.find_k_routes(G, "A", "D", k=2, amount_msat=100_000)
    assert routes[0]["total_fee_msat"] == pytest.approx(2000.0)
    assert routes[1]["total_fee_msat"] == pytest.approx(4000.0)


# ---------------------------------------------------------------------------
# fee_vs_hops()
# ---------------------------------------------------------------------------

def test_fee_vs_hops_returns_list(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = routing.fee_vs_hops(synthetic_graph, hub, leaf)
    assert isinstance(result, list)


def test_fee_vs_hops_no_path_returns_empty():
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    result = routing.fee_vs_hops(G, "A", "B")
    assert result == []


def test_fee_vs_hops_result_structure(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    routes = routing.fee_vs_hops(synthetic_graph, hub, leaf, k=3)
    assert len(routes) >= 1
    for key in ("hops", "total_fee_msat", "total_cltv", "path"):
        assert key in routes[0], f"missing key: {key}"


def test_fee_vs_hops_diamond_sorted_by_fee():
    """fee_vs_hops returns routes sorted by fee (cheap first)."""
    G = _make_diamond_graph()
    routes = routing.fee_vs_hops(G, "A", "D", k=5, amount_msat=100_000)
    fees = [r["total_fee_msat"] for r in routes]
    assert fees == sorted(fees)


def _make_fee_hops_tradeoff_graph():
    """Two routes A→E where Dijkstra-cost order differs from fee order.

    Direct A→E:          1 hop,  fee_base=5 msat,  Dijkstra cost=6  (eps=1)
    Indirect A→B→C→D→E:  4 hops, fee_base=1 each,  Dijkstra cost=8  (4×2)

    find_k_routes picks [direct, indirect] by cost (6 < 8).
    fee_vs_hops must return [indirect, direct] by fee (4 < 5).
    """
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D", "E"]:
        G.add_node(n)
    high = {"fee_base_msat": "5", "fee_rate_milli_msat": "0", "time_lock_delta": 0, "disabled": False}
    low = {"fee_base_msat": "1", "fee_rate_milli_msat": "0", "time_lock_delta": 0, "disabled": False}
    G.add_edge("A", "E", channel_id="AE_direct", capacity=1_000_000, policy=high)
    for u, v in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
        G.add_edge(u, v, channel_id=f"{u}{v}", capacity=1_000_000, policy=dict(low))
    return G


def test_fee_vs_hops_cost_order_differs_from_fee_order():
    """fee_vs_hops sorts by total_fee_msat, not Dijkstra cost.

    1-hop direct route has fee=5 and cost=6; 4-hop indirect has fee=4 and cost=8.
    find_k_routes returns [direct, indirect] (by cost), but fee_vs_hops must
    return [indirect, direct] (by fee), so routes[0] is the minimum-fee route.
    """
    G = _make_fee_hops_tradeoff_graph()
    routes = routing.fee_vs_hops(G, "A", "E", k=5, amount_msat=0)
    assert len(routes) == 2
    # indirect (4 hops, fee=4) must sort before direct (1 hop, fee=5)
    assert routes[0]["hops"] == 4
    assert routes[0]["total_fee_msat"] == pytest.approx(4.0)
    assert routes[1]["hops"] == 1
    assert routes[1]["total_fee_msat"] == pytest.approx(5.0)
    fees = [r["total_fee_msat"] for r in routes]
    assert fees == sorted(fees)


# ---------------------------------------------------------------------------
# _build_routing_graph() — parallel-channel collapse (regression coverage)
# ---------------------------------------------------------------------------

def test_build_routing_graph_parallel_cheapest_wins():
    """Parallel A→B channels: cheapest wins regardless of insertion order.

    Expensive channel is added first so the cost-comparison (not first-seen)
    drives the selection. Removing the `cost < best[key]['cost']` guard would
    keep the expensive one and produce fee=2000 instead of 500.
    """
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B", channel_id="AB_expensive", capacity=1_000_000,
               policy={"fee_base_msat": "2000", "fee_rate_milli_msat": "0",
                       "time_lock_delta": 0, "disabled": False})
    G.add_edge("A", "B", channel_id="AB_cheap", capacity=1_000_000,
               policy={"fee_base_msat": "500", "fee_rate_milli_msat": "0",
                       "time_lock_delta": 0, "disabled": False})
    result = routing.find_route(G, "A", "B", amount_msat=100_000)
    assert result is not None
    assert result["total_fee_msat"] == pytest.approx(500.0)


def test_build_routing_graph_disabled_cheap_parallel_uses_expensive():
    """Disabled cheaper parallel channel is excluded; enabled expensive one is used.

    Removing the `disabled` skip would allow the cheap channel through and
    produce fee=500 instead of 2000, failing this assertion.
    """
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B", channel_id="AB_cheap", capacity=1_000_000,
               policy={"fee_base_msat": "500", "fee_rate_milli_msat": "0",
                       "time_lock_delta": 0, "disabled": True})
    G.add_edge("A", "B", channel_id="AB_expensive", capacity=1_000_000,
               policy={"fee_base_msat": "2000", "fee_rate_milli_msat": "0",
                       "time_lock_delta": 0, "disabled": False})
    result = routing.find_route(G, "A", "B", amount_msat=100_000)
    assert result is not None
    assert result["total_fee_msat"] == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# require_policy: unknown-policy (un-enriched) channels are non-routable (3.12)
# ---------------------------------------------------------------------------


def _make_mixed_enrichment_graph():
    """A→B→D enriched (real policy); A→C→D un-enriched (policy=None).

    Mirrors a partially-enriched snapshot: the un-enriched A→C→D path is one
    hop cheaper to *cost* (epsilon only) but carries unknown fees, while the
    enriched A→B→D path has real, non-zero fees.
    """
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n)
    enriched = {"fee_base_msat": "1000", "fee_rate_milli_msat": "500",
                "time_lock_delta": 40, "disabled": False}
    # enriched path A→B→D
    G.add_edge("A", "B", channel_id="AB", capacity=1_000_000, policy=dict(enriched))
    G.add_edge("B", "D", channel_id="BD", capacity=1_000_000, policy=dict(enriched))
    # un-enriched path A→C→D (no policy yet)
    G.add_edge("A", "C", channel_id="AC", capacity=1_000_000, policy=None)
    G.add_edge("C", "D", channel_id="CD", capacity=1_000_000, policy=None)
    return G


def test_require_policy_excludes_unenriched_edges():
    """Default require_policy=True routes only over enriched channels."""
    G = _make_mixed_enrichment_graph()
    r = routing.find_route(G, "A", "D", amount_msat=100_000)
    assert r is not None
    # must take the enriched A→B→D path, never the un-enriched A→C→D
    assert r["path"] == ["A", "B", "D"]
    assert "C" not in r["path"]
    assert r["total_fee_msat"] > 0  # real fees, not a spurious 0


def test_require_policy_false_allows_unenriched_path():
    """require_policy=False lets Dijkstra use the cheaper un-enriched path."""
    G = _make_mixed_enrichment_graph()
    r = routing.find_route(G, "A", "D", amount_msat=100_000, require_policy=False)
    assert r is not None
    # un-enriched edges cost epsilon only → the C-path is cheaper → fee 0
    assert r["path"] == ["A", "C", "D"]
    assert r["total_fee_msat"] == 0.0


def test_require_policy_no_route_when_only_unenriched():
    """With every edge un-enriched, require_policy=True finds no route."""
    G = nx.MultiDiGraph()
    for n in ["A", "B"]:
        G.add_node(n)
    G.add_edge("A", "B", channel_id="AB", capacity=1_000_000, policy=None)
    assert routing.find_route(G, "A", "B") is None
    # but is routable when the policy requirement is relaxed
    assert routing.find_route(G, "A", "B", require_policy=False) is not None


def test_find_k_routes_respects_require_policy():
    """find_k_routes also excludes un-enriched edges by default."""
    G = _make_mixed_enrichment_graph()
    routes = routing.find_k_routes(G, "A", "D", k=5)
    assert len(routes) >= 1
    for r in routes:
        assert "C" not in r["path"]  # never the un-enriched path
