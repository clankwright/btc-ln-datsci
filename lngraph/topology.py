"""Topology metrics (Phase 1.4).

Compute standard network-science statistics on the Lightning channel graph.
All functions accept a networkx.MultiDiGraph (as produced by ingest.ingest).
"""
from __future__ import annotations

import random
import statistics
from typing import Sequence

import networkx as nx


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of inequality over a non-negative sequence.

    Returns 0 for perfect equality, 1 for maximum concentration.
    Raises ValueError on an empty sequence.
    """
    vals = list(values)
    if not vals:
        raise ValueError("gini() requires at least one value")
    n = len(vals)
    if n == 1:
        return 0.0
    total = sum(vals)
    if total == 0:
        return 0.0
    sorted_v = sorted(vals)
    # Gini = (2 * Σ_{i=1}^{n} i*x_i) / (n * Σx) - (n+1)/n   (sorted ascending, 1-indexed)
    cum_sum = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return (2.0 * cum_sum) / (n * total) - (n + 1) / n


def degree_distribution(G: nx.MultiDiGraph) -> dict:
    """In-degree, out-degree, and total-degree statistics."""
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [i + o for i, o in zip(in_degrees, out_degrees)]

    def _stats(seq):
        if not seq:
            return 0.0, 0, 0
        return statistics.mean(seq), max(seq), min(seq)

    mean_in, max_in, min_in = _stats(in_degrees)
    mean_out, max_out, min_out = _stats(out_degrees)

    return {
        "in_degrees": in_degrees,
        "out_degrees": out_degrees,
        "total_degrees": total_degrees,
        "mean_in": mean_in,
        "mean_out": mean_out,
        "max_in": max_in,
        "max_out": max_out,
        "min_in": min_in,
        "min_out": min_out,
    }


def capacity_distribution(G: nx.MultiDiGraph) -> dict:
    """Per-channel capacity statistics (unique channels, not directed edges).

    The MultiDiGraph has two directed edges per channel; we deduplicate by
    channel_id so each physical channel is counted once.
    """
    seen: set[str] = set()
    capacities: list[int] = []

    for _u, _v, data in G.edges(data=True):
        cid = data.get("channel_id", id(data))
        if cid not in seen:
            seen.add(cid)
            capacities.append(int(data.get("capacity", 0)))

    if not capacities:
        return {
            "capacities": [],
            "total_capacity": 0,
            "mean_capacity": 0.0,
            "median_capacity": 0.0,
            "max_capacity": 0,
            "min_capacity": 0,
            "gini_capacity": 0.0,
        }

    return {
        "capacities": capacities,
        "total_capacity": sum(capacities),
        "mean_capacity": statistics.mean(capacities),
        "median_capacity": statistics.median(capacities),
        "max_capacity": max(capacities),
        "min_capacity": min(capacities),
        "gini_capacity": gini(capacities),
    }


def component_stats(G: nx.MultiDiGraph) -> dict:
    """Weakly and strongly connected component statistics."""
    if G.number_of_nodes() == 0:
        return {
            "num_weakly_connected": 0,
            "num_strongly_connected": 0,
            "largest_wcc_size": 0,
            "largest_scc_size": 0,
            "fraction_in_lcc": 0.0,
        }

    wccs = list(nx.weakly_connected_components(G))
    sccs = list(nx.strongly_connected_components(G))

    largest_wcc = max(wccs, key=len)
    largest_scc = max(sccs, key=len)

    return {
        "num_weakly_connected": len(wccs),
        "num_strongly_connected": len(sccs),
        "largest_wcc_size": len(largest_wcc),
        "largest_scc_size": len(largest_scc),
        "fraction_in_lcc": len(largest_wcc) / G.number_of_nodes(),
    }


def clustering_stats(G: nx.MultiDiGraph) -> dict:
    """Average clustering coefficient and transitivity on the underlying undirected graph.

    LN channels are bidirectional in practice; undirected clustering gives the
    standard triangle-based measure used in LN topology papers.
    """
    # nx.average_clustering and transitivity require a simple Graph, not MultiGraph.
    UG = nx.Graph(G.to_undirected())
    avg_clustering = nx.average_clustering(UG)
    transitivity = nx.transitivity(UG)

    return {
        "average_clustering": avg_clustering,
        "transitivity": transitivity,
    }


def sampled_diameter(G: nx.MultiDiGraph, k: int = 500, seed: int = 42) -> dict:
    """Estimate diameter and average path length by sampling k random pairs.

    Restricts to the largest weakly connected component so unreachable pairs
    (between disconnected components) don't produce infinite paths.
    """
    from lngraph.graph import largest_connected_component

    lcc = largest_connected_component(G)
    if lcc.number_of_nodes() < 2:
        return {"estimated_diameter": 0, "avg_path_length": 0.0, "sample_size": 0}

    nodes = list(lcc.nodes())
    n = len(nodes)

    rng = random.Random(seed)
    k_actual = min(k, n * (n - 1))
    sampled = [rng.sample(nodes, 2) for _ in range(k_actual)]

    lengths = []
    for u, v in sampled:
        try:
            lengths.append(nx.shortest_path_length(lcc, u, v))
        except nx.NetworkXNoPath:
            pass

    if not lengths:
        return {"estimated_diameter": 0, "avg_path_length": 0.0, "sample_size": k_actual}

    return {
        "estimated_diameter": max(lengths),
        "avg_path_length": statistics.mean(lengths),
        "sample_size": k_actual,
    }


def coverage_report(G: nx.MultiDiGraph, statistics_data: dict) -> dict:
    """Compare the crawled subgraph against the /statistics/latest totals.

    ``statistics_data`` must be the dict returned by fetch_mempool.fetch_statistics(),
    i.e. ``{"latest": {"node_count": ..., "channel_count": ..., ...}}``.
    """
    latest = statistics_data["latest"]
    network_nodes = latest["node_count"]
    network_channels = latest["channel_count"]

    seen_channels: set = set()
    for _u, _v, data in G.edges(data=True):
        cid = data.get("channel_id", id(data))
        seen_channels.add(cid)

    crawled_nodes = G.number_of_nodes()
    crawled_channels = len(seen_channels)

    return {
        "crawled_nodes": crawled_nodes,
        "crawled_channels": crawled_channels,
        "network_nodes": network_nodes,
        "network_channels": network_channels,
        "node_coverage_pct": crawled_nodes / network_nodes * 100,
        "channel_coverage_pct": crawled_channels / network_channels * 100,
    }
