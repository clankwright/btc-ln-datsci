"""Liquidity / partial-observability module (Phase 3.3).

Models the public-only-capacity view of channel liquidity using a uniform
prior on the channel balance distribution:

    P(success | capacity_msat, amount_msat) = max(0, (capacity − amount) / capacity)

This is the simplest coherent model of routing success when balances are
unobservable. It degrades gracefully: impossible payments (amount > capacity)
return 0; trivial payments (amount ≈ 0) return ≈ 1.

Success-optimal routing minimises −log(p) per hop, which Dijkstra minimises
as a sum, equivalent to maximising the product of per-hop probabilities.

The Monte-Carlo simulator samples channel balances uniformly from
[0, capacity_msat) and returns the fraction of trials where every hop
on the path succeeds.
"""
from __future__ import annotations

import math
import random

import networkx as nx

_LOG_FLOOR = 1e-10  # minimum probability before log to avoid −∞ edge weights


def success_probability(capacity_msat: int, amount_msat: int) -> float:
    """Uniform-prior success probability for one channel hop.

    P = max(0, (capacity − amount) / capacity)

    Capacity and amount must both be in millisatoshis.
    """
    if capacity_msat <= 0:
        return 0.0
    return max(0.0, (capacity_msat - amount_msat) / capacity_msat)


def _best_capacity_msat(G: nx.MultiDiGraph, u: str, v: str) -> int:
    """Highest capacity in msat among non-disabled edges from u to v.

    Returns 0 if no non-disabled edge exists between u and v.
    Graph stores capacity in satoshis; this returns millisatoshis.
    """
    if v not in G[u]:
        return 0
    best = 0
    for _, data in G[u][v].items():
        policy = data.get("policy") or {}
        if policy.get("disabled", False):
            continue
        cap_sat = data.get("capacity", 0)
        best = max(best, int(cap_sat) * 1000)
    return best


def path_success_probability(
    G: nx.MultiDiGraph,
    path: list[str],
    amount_msat: int,
) -> float:
    """Product of per-hop success_probability over a node path.

    For each hop (u, v) picks the highest-capacity non-disabled channel to
    give the best probability estimate. Returns 1.0 for paths of length ≤ 1
    (trivially succeed with no hops).
    """
    prob = 1.0
    for u, v in zip(path[:-1], path[1:]):
        cap_msat = _best_capacity_msat(G, u, v)
        prob *= success_probability(cap_msat, amount_msat)
        if prob == 0.0:
            break
    return prob


def _build_success_graph(G: nx.MultiDiGraph, amount_msat: int) -> nx.DiGraph:
    """Build a DiGraph weighted by −log(success_probability) for Dijkstra.

    For each ordered pair (u, v) keeps the non-disabled channel with the
    highest capacity (maximises success probability, minimises −log weight).
    The weight floor of −log(_LOG_FLOOR) prevents infinite weights from
    zero-probability hops while still ranking them last.
    """
    best: dict[tuple[str, str], int] = {}  # (u, v) → best capacity_msat

    for u, v, data in G.edges(data=True):
        policy = data.get("policy") or {}
        if policy.get("disabled", False):
            continue
        cap_msat = int(data.get("capacity", 0)) * 1000
        key = (u, v)
        if key not in best or cap_msat > best[key]:
            best[key] = cap_msat

    SG = nx.DiGraph()
    SG.add_nodes_from(G.nodes(data=True))
    for (u, v), cap_msat in best.items():
        p = success_probability(cap_msat, amount_msat)
        weight = -math.log(max(p, _LOG_FLOOR))
        SG.add_edge(u, v, weight=weight, capacity_msat=cap_msat, success_prob=p)
    return SG


def success_optimal_route(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
    amount_msat: int = 100_000,
) -> dict | None:
    """Find the route maximising path success probability via Dijkstra on −log(p).

    Returns a dict:
        path                list of node pub_keys, source first
        hops                number of hops
        success_probability product of per-hop success probabilities
        per_hop             list of per-hop dicts:
                              from, to, capacity_msat, success_prob
    Returns None when no route exists or source/target is unknown.
    """
    SG = _build_success_graph(G, amount_msat)
    try:
        path = nx.dijkstra_path(SG, source, target, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    per_hop = []
    total_p = 1.0
    for u, v in zip(path[:-1], path[1:]):
        d = SG[u][v]
        per_hop.append({
            "from": u,
            "to": v,
            "capacity_msat": d["capacity_msat"],
            "success_prob": d["success_prob"],
        })
        total_p *= d["success_prob"]

    return {
        "path": path,
        "hops": len(path) - 1,
        "success_probability": total_p,
        "per_hop": per_hop,
    }


def monte_carlo_success_rate(
    G: nx.MultiDiGraph,
    path: list[str],
    amount_msat: int,
    n_trials: int = 1_000,
    seed: int | None = None,
) -> float:
    """Empirical success rate by sampling channel balances uniformly.

    For each trial: sample a balance uniformly from [0, capacity_msat) for each
    hop; the trial succeeds when every hop's sampled balance ≥ amount_msat.
    Returns the fraction of successful trials ∈ [0.0, 1.0].

    Converges to path_success_probability() as n_trials → ∞.
    """
    if len(path) <= 1:
        return 1.0

    rng = random.Random(seed)

    hop_caps = [_best_capacity_msat(G, u, v) for u, v in zip(path[:-1], path[1:])]

    successes = 0
    for _ in range(n_trials):
        ok = True
        for cap_msat in hop_caps:
            if cap_msat <= 0 or rng.uniform(0.0, cap_msat) < amount_msat:
                ok = False
                break
        if ok:
            successes += 1

    return successes / n_trials
