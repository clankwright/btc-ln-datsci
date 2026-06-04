"""describegraph → networkx.MultiDiGraph ingester (the internal contract).

Every data source (the mempool crawler today; an LND ``describegraph.json`` or
historical gossip replay tomorrow) normalizes to the canonical describegraph
schema, and every analysis module reads the ``MultiDiGraph`` this module
produces. Keeping that boundary single and source-agnostic is the whole point.

Schema (describegraph)::

    {
      "nodes": [{"pub_key": str, "alias": str}, ...],
      "edges": [{
        "channel_id": str, "chan_point": str, "capacity": str(int),
        "node1_pub": str, "node2_pub": str,
        "node1_policy": {...} | None, "node2_policy": {...} | None,
      }, ...]
    }

Each channel becomes **two** directed edges — ``node1_pub → node2_pub`` carrying
``node1_policy`` and the reverse carrying ``node2_policy`` — because LN fee
policy is directional. A ``MultiDiGraph`` is used so parallel channels between
the same pair of nodes are preserved as distinct edges.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx


def ingest(describegraph: dict) -> nx.MultiDiGraph:
    """Build a ``MultiDiGraph`` from a describegraph dict.

    Nodes referenced only by an edge (e.g. peers at the crawl boundary that
    were discovered but never fetched) are added with an empty alias.
    """
    G = nx.MultiDiGraph()

    for node in describegraph.get("nodes", []):
        G.add_node(node["pub_key"], alias=node.get("alias", ""))

    for ch in describegraph.get("edges", []):
        n1, n2 = ch["node1_pub"], ch["node2_pub"]
        if n1 not in G:
            G.add_node(n1, alias="")
        if n2 not in G:
            G.add_node(n2, alias="")
        capacity = int(ch["capacity"])
        G.add_edge(
            n1, n2,
            channel_id=ch["channel_id"],
            capacity=capacity,
            policy=ch.get("node1_policy"),
        )
        G.add_edge(
            n2, n1,
            channel_id=ch["channel_id"],
            capacity=capacity,
            policy=ch.get("node2_policy"),
        )

    return G


def ingest_file(path: str | Path) -> nx.MultiDiGraph:
    """Load a describegraph JSON snapshot from ``path`` and ingest it."""
    data = json.loads(Path(path).read_text())
    return ingest(data)
