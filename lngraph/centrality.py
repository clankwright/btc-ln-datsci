"""Centrality metrics for the Lightning Network channel graph (Phase 2.1).

All public functions accept a networkx.MultiDiGraph (as produced by ingest.ingest)
and internally simplify to a DiGraph before running centrality algorithms.
"""
from __future__ import annotations

import networkx as nx

from lngraph.graph import simplify, to_igraph


def degree_centrality(G: nx.MultiDiGraph) -> dict:
    """Degree centrality variants for every node.

    Returns a dict with four sub-dicts:
      in_degree      — nx.in_degree_centrality (normalized by n-1)
      out_degree     — nx.out_degree_centrality (normalized by n-1)
      total_degree   — average of in and out degree centrality
      capacity_weighted — sum of capacities of all incident edges,
                          normalized so the node with the most capacity = 1.0
    """
    DG = simplify(G)
    n = DG.number_of_nodes()
    if n == 0:
        return {"in_degree": {}, "out_degree": {}, "total_degree": {}, "capacity_weighted": {}}

    in_deg = nx.in_degree_centrality(DG)
    out_deg = nx.out_degree_centrality(DG)

    if n > 1:
        total_deg = {v: (in_deg[v] + out_deg[v]) / 2.0 for v in DG}
    else:
        total_deg = {v: 0.0 for v in DG}

    cap_sum: dict[str, float] = {}
    for v in DG:
        in_cap = sum(d.get("capacity", 0) for _, _, d in DG.in_edges(v, data=True))
        out_cap = sum(d.get("capacity", 0) for _, _, d in DG.out_edges(v, data=True))
        cap_sum[v] = float(in_cap + out_cap)

    max_cap = max(cap_sum.values()) if cap_sum else 0.0
    if max_cap == 0.0:
        cap_weighted = {v: 0.0 for v in DG}
    else:
        cap_weighted = {v: cap_sum[v] / max_cap for v in DG}

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": total_deg,
        "capacity_weighted": cap_weighted,
    }


def closeness_centrality(G: nx.MultiDiGraph) -> dict:
    """Closeness centrality for every node (Wasserman-Faust improved, directed).

    Uses out-closeness: distance from each node to all reachable successors.
    Returns {node: float}.
    """
    DG = simplify(G)
    if DG.number_of_nodes() == 0:
        return {}
    return nx.closeness_centrality(DG)


def eigenvector_centrality(G: nx.MultiDiGraph, max_iter: int = 1000) -> dict:
    """Eigenvector centrality (right eigenvector, unweighted) for every node.

    Falls back to the numpy-based solver if power iteration does not converge.
    Returns {node: float}.
    """
    DG = simplify(G)
    if DG.number_of_nodes() == 0:
        return {}
    try:
        result = nx.eigenvector_centrality(DG, max_iter=max_iter)
    except nx.PowerIterationFailedConvergence:
        result = nx.eigenvector_centrality_numpy(DG)
    return {v: float(c) for v, c in result.items()}


def betweenness_sampled(G: nx.MultiDiGraph, k: int = 500, seed: int = 42) -> dict:
    """Approximate betweenness centrality via k-pivot sampling.

    k pivots are drawn at random (seed-reproducible); k is automatically
    capped at the number of nodes so the call never raises ValueError.
    Returns {node: float} normalized to [0, 1].
    """
    DG = simplify(G)
    n = DG.number_of_nodes()
    if n == 0:
        return {}
    k_actual = min(k, n)
    return nx.betweenness_centrality(DG, k=k_actual, seed=seed, normalized=True)


def betweenness_exact(G: nx.MultiDiGraph) -> dict:
    """Exact betweenness centrality via igraph's C core.

    Normalized the same way as networkx: divide by (n-1)*(n-2) for directed
    graphs (n >= 3). igraph's built-in normalization uses n*(n-1), which
    differs from networkx's convention; we re-apply the correct factor from
    the raw (un-normalized) counts.
    Returns {node: float}.
    """
    DG = simplify(G)
    n = DG.number_of_nodes()
    if n == 0:
        return {}
    ig = to_igraph(DG)
    raw = ig.betweenness(directed=True, normalized=False)
    norm = float((n - 1) * (n - 2)) if n > 2 else 1.0
    return {ig.vs[i]["name"]: float(raw[i] / norm) for i in range(ig.vcount())}


_EPSILON = 1.0  # msat; one positive unit per hop so Dijkstra never ties on zero-weight paths


def _fee_cost(policy: dict | None, amount_msat: int) -> float:
    """LN routing fee cost: epsilon + base_fee + amount * fee_rate_ppm / 1e6.

    _EPSILON ensures every hop carries a positive distance even when fees are
    absent or zero, so Dijkstra counts shortest paths correctly. When all fees
    are uniform/absent the result reduces to hop-count betweenness.
    """
    if not policy:
        return _EPSILON
    base = int(policy.get("fee_base_msat", 0))
    rate = int(policy.get("fee_rate_milli_msat", 0))
    return float(_EPSILON + base + amount_msat * rate / 1_000_000)


def betweenness_fee(
    G: nx.MultiDiGraph,
    k: int = 500,
    seed: int = 42,
    amount_msat: int = 100_000,
) -> dict:
    """Betweenness centrality weighted by the real LN fee-based routing cost.

    Cost per hop = fee_base_msat + amount_msat * fee_rate_milli_msat / 1e6.
    Disabled edges (policy.disabled=True) are excluded entirely.
    Returns {node: float} normalized to [0, 1].
    """
    DG = simplify(G)
    n = DG.number_of_nodes()
    if n == 0:
        return {}

    FG: nx.DiGraph = nx.DiGraph()
    FG.add_nodes_from(DG.nodes(data=True))
    for u, v, data in DG.edges(data=True):
        policy = data.get("policy") or {}
        if policy.get("disabled", False):
            continue
        weight = _fee_cost(policy, amount_msat)
        FG.add_edge(u, v, fee_weight=weight)

    if FG.number_of_edges() == 0:
        return {v: 0.0 for v in FG}

    k_actual = min(k, FG.number_of_nodes())
    return nx.betweenness_centrality(FG, k=k_actual, seed=seed, weight="fee_weight", normalized=True)
