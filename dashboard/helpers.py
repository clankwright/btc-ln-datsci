"""Pure-Python helpers for the lngraph Streamlit dashboard.

No Streamlit imports — these functions are unit-testable without
starting a server or mocking the Streamlit library.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_node_options(G: nx.MultiDiGraph) -> list[tuple[str, str]]:
    """Return (node_id, display_label) pairs sorted by display_label (case-insensitive).

    Labels prefer the node alias when present; fall back to the first 16 chars of
    the pubkey followed by an ellipsis.
    """
    options: list[tuple[str, str]] = []
    for node_id in G.nodes():
        alias = (G.nodes[node_id].get("alias") or "").strip()
        if alias:
            label = f"{alias} ({node_id[:12]}…)"
        else:
            label = node_id[:16] + "…"
        options.append((node_id, label))
    return sorted(options, key=lambda x: x[1].lower())


def get_node_centrality_row(
    node_id: str,
    deg: dict,
    close: dict,
    eig: dict,
    btwn: dict,
) -> dict:
    """Extract per-node centrality values across four pre-computed metric dicts.

    Returns a flat dict with float values for: degree, capacity_weighted,
    closeness, eigenvector, and betweenness.  Missing entries default to 0.0.
    """
    return {
        "degree": float(deg.get("total_degree", {}).get(node_id, 0.0)),
        "capacity_weighted": float(deg.get("capacity_weighted", {}).get(node_id, 0.0)),
        "closeness": float(close.get(node_id, 0.0)),
        "eigenvector": float(eig.get(node_id, 0.0)),
        "betweenness": float(btwn.get(node_id, 0.0)),
    }


def format_route_table(route: dict) -> pd.DataFrame:
    """Convert a route's per_hop list to a display DataFrame.

    Columns: hop, from, to, channel_id, fee_msat, cltv_delta.
    Returns an empty DataFrame (with those columns) when route has no hops.
    """
    cols = ["hop", "from", "to", "channel_id", "fee_msat", "cltv_delta"]
    rows = []
    for i, hop in enumerate(route.get("per_hop", []), start=1):
        src = hop["from"]
        dst = hop["to"]
        rows.append({
            "hop": i,
            "from": src[:16] + "…" if len(src) > 16 else src,
            "to": dst[:16] + "…" if len(dst) > 16 else dst,
            "channel_id": hop.get("channel_id", ""),
            "fee_msat": round(float(hop.get("fee_msat", 0.0)), 3),
            "cltv_delta": int(hop.get("cltv_delta", 0)),
        })
    if rows:
        return pd.DataFrame(rows, columns=cols)
    return pd.DataFrame(columns=cols)


def summarize_route(route: dict, success_probability: float | None = None) -> dict:
    """Flatten top-level route fields into a dict suitable for st.metric display.

    success_probability is passed through as-is (caller computes it separately).
    """
    return {
        "hops": route.get("hops", 0),
        "total_fee_msat": round(float(route.get("total_fee_msat", 0.0)), 3),
        "total_cltv": route.get("total_cltv", 0),
        "success_probability": success_probability,
    }
