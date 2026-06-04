"""Tests for liquidity.py — Phase 3.3.

Covers: success_probability(), path_success_probability(),
        success_optimal_route(), monte_carlo_success_rate().

Hand-checkable values:

  Two-hop graph A→B→C:
    AB capacity = 2 sats = 2_000 msat; BC capacity = 4 sats = 4_000 msat
    amount_msat = 1_000
    P(AB) = (2_000 - 1_000) / 2_000 = 0.5
    P(BC) = (4_000 - 1_000) / 4_000 = 0.75
    path P = 0.5 × 0.75 = 0.375

  Success-vs-fee diamond A→D via B or via C:
    amount_msat = 1_000
    Via B: each hop capacity=2_000 msat → P/hop=0.5, path P=0.25
    Via C: each hop capacity=4_000 msat → P/hop=0.75, path P=0.5625
    B legs: fee_base=100 msat each, C legs: fee_base=200 msat each
    Fee-optimal picks via B (total fee=200). Success-optimal picks via C (P=0.5625).
"""
import pytest
import networkx as nx

from lngraph import liquidity


# ---------------------------------------------------------------------------
# Helper graphs
# ---------------------------------------------------------------------------

def _make_two_hop_graph():
    """A→B→C. AB=2 sat (2_000 msat), BC=4 sat (4_000 msat)."""
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C"]:
        G.add_node(n)
    G.add_edge("A", "B", channel_id="AB", capacity=2,
               policy={"disabled": False, "fee_base_msat": "0", "fee_rate_milli_msat": "0"})
    G.add_edge("B", "C", channel_id="BC", capacity=4,
               policy={"disabled": False, "fee_base_msat": "0", "fee_rate_milli_msat": "0"})
    return G


def _make_success_vs_fee_diamond():
    """A→D via two paths:
    - Via B: each hop capacity=2 sat (2_000 msat), fee_base=100
    - Via C: each hop capacity=4 sat (4_000 msat), fee_base=200
    amount_msat=1_000.
    Fee-optimal=via B (cheaper). Success-optimal=via C (higher capacity → higher P).
    """
    G = nx.MultiDiGraph()
    for n in ["A", "B", "C", "D"]:
        G.add_node(n)
    cheap = {"disabled": False, "fee_base_msat": "100", "fee_rate_milli_msat": "0", "time_lock_delta": 10}
    expensive = {"disabled": False, "fee_base_msat": "200", "fee_rate_milli_msat": "0", "time_lock_delta": 5}
    G.add_edge("A", "B", channel_id="AB", capacity=2, policy=cheap)
    G.add_edge("B", "D", channel_id="BD", capacity=2, policy=cheap)
    G.add_edge("A", "C", channel_id="AC", capacity=4, policy=expensive)
    G.add_edge("C", "D", channel_id="CD", capacity=4, policy=expensive)
    return G


def _make_single_hop_graph(capacity_sat: int):
    """A→B with a single channel of given capacity in sats."""
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    G.add_edge("A", "B", channel_id="AB", capacity=capacity_sat,
               policy={"disabled": False})
    return G


# ---------------------------------------------------------------------------
# success_probability()
# ---------------------------------------------------------------------------

def test_success_prob_zero_amount():
    """Amount=0 → probability=1.0 (payment always succeeds)."""
    assert liquidity.success_probability(1_000_000, 0) == pytest.approx(1.0)


def test_success_prob_full_amount():
    """Amount=capacity → probability=0.0."""
    assert liquidity.success_probability(1_000, 1_000) == pytest.approx(0.0)


def test_success_prob_half_amount():
    """Amount=capacity/2 → probability=0.5."""
    assert liquidity.success_probability(2_000, 1_000) == pytest.approx(0.5)


def test_success_prob_over_capacity_clamped():
    """Amount > capacity → probability=0.0 (clamped, not negative)."""
    assert liquidity.success_probability(1_000, 1_500) == pytest.approx(0.0)


def test_success_prob_zero_capacity_returns_zero():
    """Capacity=0 edge case → 0.0."""
    assert liquidity.success_probability(0, 100) == pytest.approx(0.0)


def test_success_prob_known_value():
    """amount=1_000, capacity=4_000 → (4_000-1_000)/4_000 = 0.75."""
    assert liquidity.success_probability(4_000, 1_000) == pytest.approx(0.75)


def test_success_prob_tiny_amount():
    """amount=1, capacity=1_000_000 → ≈ 1.0 (not exactly 1.0 but very close)."""
    p = liquidity.success_probability(1_000_000, 1)
    assert 0.999 < p < 1.0


# ---------------------------------------------------------------------------
# path_success_probability()
# ---------------------------------------------------------------------------

def test_path_success_empty_path_returns_one():
    """Single-node path (len=1) → empty product → 1.0."""
    G = _make_two_hop_graph()
    assert liquidity.path_success_probability(G, ["A"], 1_000) == pytest.approx(1.0)


def test_path_success_single_hop_matches_channel():
    """Single-hop A→B: P = (2_000 - 1_000) / 2_000 = 0.5."""
    G = _make_two_hop_graph()
    assert liquidity.path_success_probability(G, ["A", "B"], 1_000) == pytest.approx(0.5)


def test_path_success_two_hops_product():
    """A→B→C: 0.5 × 0.75 = 0.375."""
    G = _make_two_hop_graph()
    p = liquidity.path_success_probability(G, ["A", "B", "C"], 1_000)
    assert p == pytest.approx(0.375)


def test_path_success_zero_if_over_capacity_any_hop():
    """If any hop has amount ≥ capacity → total probability = 0.0."""
    G = _make_two_hop_graph()
    # AB capacity = 2_000 msat; amount = 2_500 > capacity → P(AB) = 0
    p = liquidity.path_success_probability(G, ["A", "B", "C"], 2_500)
    assert p == pytest.approx(0.0)


def test_path_success_parallel_picks_highest_capacity():
    """With two parallel A→B channels (cap=1 sat and cap=4 sat), path_success picks cap=4."""
    G = nx.MultiDiGraph()
    for n in ["A", "B"]:
        G.add_node(n)
    G.add_edge("A", "B", channel_id="small", capacity=1,
               policy={"disabled": False})
    G.add_edge("A", "B", channel_id="large", capacity=4,
               policy={"disabled": False})
    # With cap=4_000 msat, amount=1_000: P = 0.75
    p = liquidity.path_success_probability(G, ["A", "B"], 1_000)
    assert p == pytest.approx(0.75)


def test_path_success_disabled_channel_not_used():
    """Disabled large channel falls back to small; P = (1_000-500)/1_000 = 0.5."""
    G = nx.MultiDiGraph()
    for n in ["A", "B"]:
        G.add_node(n)
    G.add_edge("A", "B", channel_id="large", capacity=100,
               policy={"disabled": True})   # disabled — should be ignored
    G.add_edge("A", "B", channel_id="small", capacity=1,
               policy={"disabled": False})  # cap=1_000 msat
    # Only small channel is active: P = (1_000-500)/1_000 = 0.5
    p = liquidity.path_success_probability(G, ["A", "B"], 500)
    assert p == pytest.approx(0.5)


def test_path_success_no_edge_returns_zero():
    """Hop with no edge in graph → capacity=0 → P=0.0."""
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    # No edge added between A and B
    p = liquidity.path_success_probability(G, ["A", "B"], 100)
    assert p == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# success_optimal_route()
# ---------------------------------------------------------------------------

def test_success_optimal_returns_none_disconnected():
    """Disconnected graph → None."""
    G = nx.MultiDiGraph()
    G.add_node("A")
    G.add_node("B")
    result = liquidity.success_optimal_route(G, "A", "B", amount_msat=1_000)
    assert result is None


def test_success_optimal_returns_none_missing_node():
    G = nx.MultiDiGraph()
    G.add_node("A")
    result = liquidity.success_optimal_route(G, "A", "MISSING", amount_msat=1_000)
    assert result is None


def test_success_optimal_result_keys():
    """Route dict contains required keys."""
    G = _make_single_hop_graph(100)
    result = liquidity.success_optimal_route(G, "A", "B", amount_msat=1_000)
    assert result is not None
    for key in ("path", "hops", "success_probability", "per_hop"):
        assert key in result, f"missing key: {key}"


def test_success_optimal_single_hop_path(synthetic_graph):
    nodes = list(synthetic_graph.nodes())
    hub, leaf = nodes[0], nodes[1]
    result = liquidity.success_optimal_route(synthetic_graph, hub, leaf, amount_msat=100_000)
    assert result is not None
    assert result["hops"] == 1
    assert len(result["path"]) == 2


def test_success_optimal_success_prob_in_range():
    G = _make_two_hop_graph()
    result = liquidity.success_optimal_route(G, "A", "C", amount_msat=1_000)
    assert result is not None
    assert 0.0 <= result["success_probability"] <= 1.0


def test_success_optimal_prefers_high_capacity():
    """Success-optimal picks the high-capacity path (via C), not fee-optimal (via B)."""
    G = _make_success_vs_fee_diamond()
    result = liquidity.success_optimal_route(G, "A", "D", amount_msat=1_000)
    assert result is not None
    # via C: path A→C→D; via B: path A→B→D
    # success-optimal must pick via C (P=0.5625 > 0.25 via B)
    assert "C" in result["path"], (
        f"Expected success-optimal to route via C (high capacity), got {result['path']}"
    )


def test_success_optimal_success_prob_via_c_is_0_5625():
    """success_probability of the success-optimal route = 0.75^2 = 0.5625."""
    G = _make_success_vs_fee_diamond()
    result = liquidity.success_optimal_route(G, "A", "D", amount_msat=1_000)
    assert result is not None
    assert result["success_probability"] == pytest.approx(0.5625)


def test_success_optimal_per_hop_structure():
    """Each per_hop entry has: from, to, capacity_msat, success_prob."""
    G = _make_two_hop_graph()
    result = liquidity.success_optimal_route(G, "A", "C", amount_msat=1_000)
    assert result is not None
    for hop in result["per_hop"]:
        for key in ("from", "to", "capacity_msat", "success_prob"):
            assert key in hop, f"per_hop entry missing key: {key}"


def test_success_optimal_per_hop_success_prob_product():
    """Product of per_hop success_prob matches route success_probability."""
    G = _make_two_hop_graph()
    result = liquidity.success_optimal_route(G, "A", "C", amount_msat=1_000)
    assert result is not None
    product = 1.0
    for hop in result["per_hop"]:
        product *= hop["success_prob"]
    assert product == pytest.approx(result["success_probability"])


# ---------------------------------------------------------------------------
# monte_carlo_success_rate()
# ---------------------------------------------------------------------------

def test_monte_carlo_impossible_route_returns_zero():
    """amount > capacity → every trial fails → rate = 0.0."""
    # Single hop A→B, capacity=1 sat = 1_000 msat, amount=2_000 msat
    G = _make_single_hop_graph(1)
    rate = liquidity.monte_carlo_success_rate(G, ["A", "B"], amount_msat=2_000, n_trials=500, seed=0)
    assert rate == pytest.approx(0.0)


def test_monte_carlo_trivial_route_near_one():
    """amount << capacity → nearly every trial succeeds."""
    # 1 sat = 1_000 msat capacity; amount = 1 msat → P ≈ 0.999
    G = _make_single_hop_graph(1_000_000)
    rate = liquidity.monte_carlo_success_rate(G, ["A", "B"], amount_msat=1, n_trials=10_000, seed=42)
    assert rate > 0.99


def test_monte_carlo_returns_float_in_unit_interval():
    G = _make_two_hop_graph()
    rate = liquidity.monte_carlo_success_rate(G, ["A", "B", "C"], amount_msat=1_000, n_trials=200, seed=7)
    assert isinstance(rate, float)
    assert 0.0 <= rate <= 1.0


def test_monte_carlo_deterministic_same_seed():
    G = _make_two_hop_graph()
    r1 = liquidity.monte_carlo_success_rate(G, ["A", "B", "C"], amount_msat=1_000, n_trials=500, seed=99)
    r2 = liquidity.monte_carlo_success_rate(G, ["A", "B", "C"], amount_msat=1_000, n_trials=500, seed=99)
    assert r1 == r2


def test_monte_carlo_different_seeds_can_differ():
    """Different seeds produce potentially different results (proves it's not constant)."""
    G = _make_two_hop_graph()
    # Use a small n_trials so variance is high enough to make seeds differ
    rates = {
        liquidity.monte_carlo_success_rate(G, ["A", "B", "C"], amount_msat=1_000, n_trials=10, seed=s)
        for s in range(20)
    }
    assert len(rates) > 1, "Different seeds should produce at least some variation"


def test_monte_carlo_single_node_path_returns_one():
    """Trivially empty path (single node) → no hops → 100% success."""
    G = _make_single_hop_graph(1_000)
    rate = liquidity.monte_carlo_success_rate(G, ["A"], amount_msat=1_000, n_trials=10, seed=0)
    assert rate == pytest.approx(1.0)


def test_monte_carlo_matches_analytic_single_hop():
    """Empirical rate approximates success_probability for large n_trials.

    Single hop A→B, capacity=4_000 msat (4 sats), amount=1_000 msat.
    Analytic P = (4_000-1_000)/4_000 = 0.75. With n=10_000 empirical ≈ 0.75 ± 0.01.
    """
    G = _make_single_hop_graph(4)  # 4 sats = 4_000 msat
    rate = liquidity.monte_carlo_success_rate(G, ["A", "B"], amount_msat=1_000, n_trials=10_000, seed=42)
    assert abs(rate - 0.75) < 0.02, f"Expected ~0.75, got {rate}"


def test_monte_carlo_two_hop_matches_analytic():
    """A→B→C: analytic P = 0.375. Empirical ≈ 0.375 ± 0.02 with n=10_000."""
    G = _make_two_hop_graph()
    rate = liquidity.monte_carlo_success_rate(G, ["A", "B", "C"], amount_msat=1_000, n_trials=10_000, seed=42)
    assert abs(rate - 0.375) < 0.02, f"Expected ~0.375, got {rate}"
