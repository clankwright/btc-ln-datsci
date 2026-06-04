"""Graph core utilities (Phase 1.3).

Functions that operate on the raw MultiDiGraph produced by ingest.py and
yield derived graph objects consumed by topology, centrality, and routing
modules.
"""
from __future__ import annotations

import networkx as nx


def simplify(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse a MultiDiGraph into a DiGraph for simple-graph algorithms.

    Parallel edges (multiple channels between the same ordered pair) are merged
    into a single edge whose `capacity` is the sum of component capacities.
    Node attributes are preserved unchanged.
    """
    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        cap = data.get("capacity", 0)
        if DG.has_edge(u, v):
            DG[u][v]["capacity"] += cap
        else:
            DG.add_edge(u, v, **data)

    return DG


def largest_connected_component(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return the subgraph induced by the largest weakly connected component."""
    if G.number_of_nodes() == 0:
        return G.__class__()

    wcc = max(nx.weakly_connected_components(G), key=len)
    return G.subgraph(wcc).copy()


def to_igraph(G: nx.MultiDiGraph | nx.DiGraph):
    """Convert a networkx directed graph to an igraph.Graph.

    Vertex attribute ``name`` is set to the node identifier (pub_key string).
    Edge attribute ``capacity`` is carried over when present.
    """
    import igraph

    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}

    edges = []
    capacities = []
    for u, v, data in G.edges(data=True):
        edges.append((node_idx[u], node_idx[v]))
        capacities.append(data.get("capacity", 0))

    ig = igraph.Graph(n=len(nodes), edges=edges, directed=True)
    ig.vs["name"] = nodes
    ig.es["capacity"] = capacities
    return ig
