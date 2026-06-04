"""Routing module for the Lightning Network channel graph (Phase 3.2).

Models routing cost as the real LN fee function:
    fee_base_msat + amount_msat * fee_rate_milli_msat / 1_000_000 + cltv_risk(cltv_delta)

Disabled channels are excluded. Parallel channels between the same node pair
are collapsed to the cheapest non-disabled one. Returns per-hop breakdowns
alongside path-level summary stats.
"""
from __future__ import annotations

import networkx as nx

_EPSILON = 1.0  # msat; ensures every edge carries positive weight for Dijkstra


def route_cost(
    policy: dict | None,
    amount_msat: int,
    cltv_risk_weight: float = 1.0,
) -> float:
    """Per-edge LN routing cost used for Dijkstra optimization.

    cost = epsilon + fee_base_msat + amount_msat * fee_rate_ppm / 1e6
                   + cltv_risk_weight * cltv_delta

    `_EPSILON` keeps all edge weights positive even when fees are zero so
    Dijkstra never treats zero-fee paths as having equal cost to disconnected
    node pairs. The caller separates `fee_msat` (what you pay) from `cost`
    (what Dijkstra minimizes).
    """
    if not policy:
        return float(_EPSILON)
    base = int(policy.get("fee_base_msat", 0))
    rate = int(policy.get("fee_rate_milli_msat", 0))
    cltv = int(policy.get("time_lock_delta", 0))
    return float(_EPSILON + base + amount_msat * rate / 1_000_000 + cltv_risk_weight * cltv)


def _fee_msat(policy: dict | None, amount_msat: int) -> float:
    """Actual LN routing fee (without epsilon or cltv risk)."""
    if not policy:
        return 0.0
    base = int(policy.get("fee_base_msat", 0))
    rate = int(policy.get("fee_rate_milli_msat", 0))
    return float(base + amount_msat * rate / 1_000_000)


def _build_routing_graph(
    G: nx.MultiDiGraph,
    amount_msat: int,
    cltv_risk_weight: float,
    require_policy: bool = True,
) -> nx.DiGraph:
    """Build a weighted DiGraph from the MultiDiGraph for Dijkstra routing.

    For each ordered node pair (u, v), keep only the cheapest non-disabled
    channel. Each edge carries three attributes:
      - cost       — Dijkstra weight (epsilon + fee + cltv risk)
      - fee_msat   — actual LN fee paid for this hop
      - cltv_delta — timelock added by this hop
      - channel_id — channel short ID

    When ``require_policy`` is True (the default), channels with no per-direction
    policy (``policy is None``, i.e. not yet enriched via Phase 3.1) are excluded
    entirely. Unknown policy is NOT the same as a free channel: on a partially
    enriched snapshot, costing an un-enriched edge at ``_EPSILON`` makes it look
    nearly free and biases Dijkstra into routing through it, so routes report a
    spurious 0 fee. Excluding them confines routing to the enriched subgraph,
    where the reported fees are real. Pass ``require_policy=False`` to route over
    the full graph (un-enriched edges costed at ``_EPSILON``).
    """
    best: dict[tuple[str, str], dict] = {}

    for u, v, data in G.edges(data=True):
        raw_policy = data.get("policy")
        if require_policy and not raw_policy:
            continue
        policy = raw_policy or {}
        if policy.get("disabled", False):
            continue
        cost = route_cost(policy, amount_msat, cltv_risk_weight)
        key = (u, v)
        if key not in best or cost < best[key]["cost"]:
            best[key] = {
                "cost": cost,
                "fee_msat": _fee_msat(policy, amount_msat),
                "cltv_delta": int(policy.get("time_lock_delta", 0)),
                "channel_id": data.get("channel_id", ""),
                "capacity": data.get("capacity", 0),
            }

    RG = nx.DiGraph()
    RG.add_nodes_from(G.nodes(data=True))
    for (u, v), attrs in best.items():
        RG.add_edge(u, v, **attrs)
    return RG


def _route_from_path(RG: nx.DiGraph, path: list[str]) -> dict:
    """Build a route dict from a node path and a weighted routing DiGraph."""
    per_hop = []
    for u, v in zip(path[:-1], path[1:]):
        d = RG[u][v]
        per_hop.append({
            "from": u,
            "to": v,
            "channel_id": d.get("channel_id", ""),
            "fee_msat": d["fee_msat"],
            "cltv_delta": d["cltv_delta"],
        })
    return {
        "path": path,
        "hops": len(path) - 1,
        "total_fee_msat": sum(h["fee_msat"] for h in per_hop),
        "total_cltv": sum(h["cltv_delta"] for h in per_hop),
        "per_hop": per_hop,
    }


def find_route(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
    amount_msat: int = 100_000,
    cltv_risk_weight: float = 1.0,
    require_policy: bool = True,
) -> dict | None:
    """Find the cheapest route from source to target using Dijkstra.

    Cost per edge = epsilon + fee_base_msat + amount_msat * ppm / 1e6
                  + cltv_risk_weight * cltv_delta.
    Disabled channels are excluded. Parallel channels between the same node
    pair are collapsed to the cheapest non-disabled one.

    Returns a dict:
        path           list of node pub_keys, source first
        hops           number of hops (len(path) - 1)
        total_fee_msat sum of per-hop LN fees along the route
        total_cltv     sum of per-hop CLTV deltas along the route
        per_hop        list of per-hop dicts:
                         from, to, channel_id, fee_msat, cltv_delta

    Returns None when no route exists or source/target is unknown.

    By default (``require_policy=True``) only channels with a known per-direction
    policy are routable; pass ``require_policy=False`` to route over the full
    graph (un-enriched edges costed at ``_EPSILON``).
    """
    RG = _build_routing_graph(G, amount_msat, cltv_risk_weight, require_policy)
    try:
        path = nx.dijkstra_path(RG, source, target, weight="cost")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    return _route_from_path(RG, path)


def find_k_routes(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
    k: int = 3,
    amount_msat: int = 100_000,
    cltv_risk_weight: float = 1.0,
    require_policy: bool = True,
) -> list[dict]:
    """Find up to k simple routes ordered by total cost (Yen's algorithm).

    Uses nx.shortest_simple_paths which implements Yen's k-shortest simple
    paths algorithm on weighted graphs. Returns an empty list when no route
    exists.

    Each returned route has the same structure as find_route(). By default
    (``require_policy=True``) only channels with a known policy are routable.
    """
    RG = _build_routing_graph(G, amount_msat, cltv_risk_weight, require_policy)
    routes = []
    try:
        for path in nx.shortest_simple_paths(RG, source, target, weight="cost"):
            routes.append(_route_from_path(RG, path))
            if len(routes) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return routes


def fee_vs_hops(
    G: nx.MultiDiGraph,
    source: str,
    target: str,
    amount_msat: int = 100_000,
    k: int = 10,
    require_policy: bool = True,
) -> list[dict]:
    """Return up to k alternative routes sorted by total_fee_msat (CLTV risk excluded).

    The top-k selection uses Dijkstra cost (fee + epsilon*hops, cltv_risk=0) to
    find candidate routes, but the returned list is sorted by total_fee_msat so
    routes[0] is always the minimum-fee route. This matters when a longer route
    costs less in fees than a shorter one — the epsilon-per-hop penalty in the
    Dijkstra cost can rank a low-fee multi-hop route after a high-fee 1-hop route.

    Each route dict has: hops, total_fee_msat, total_cltv, path.
    Returns an empty list when no route exists.
    """
    routes = find_k_routes(
        G, source, target, k=k,
        amount_msat=amount_msat,
        cltv_risk_weight=0.0,
        require_policy=require_policy,
    )
    result = [
        {
            "hops": r["hops"],
            "total_fee_msat": r["total_fee_msat"],
            "total_cltv": r["total_cltv"],
            "path": r["path"],
        }
        for r in routes
    ]
    result.sort(key=lambda x: x["total_fee_msat"])
    return result
