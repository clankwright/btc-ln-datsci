"""lngraph Streamlit dashboard (Phase 4.3).

Two tabs:
  Node Inspector  — pick a node → centrality metrics + neighbourhood graph.
  Route Explorer  — pick source + target + amount → fee-optimal and
                    success-optimal routes with per-hop fee/CLTV breakdown.

Run:
    make dashboard
    # or: .venv/bin/streamlit run dashboard/app.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend; must precede any pyplot import

import pandas as pd
import streamlit as st

from lngraph.centrality import (
    betweenness_sampled,
    closeness_centrality,
    degree_centrality,
    eigenvector_centrality,
)
from lngraph.config import SNAPSHOT_DIR
from lngraph.ingest import ingest_file
from lngraph.liquidity import path_success_probability, success_optimal_route
from lngraph.routing import find_route
from lngraph.viz import neighborhood_graph_plot
from dashboard.helpers import (
    build_node_options,
    format_route_table,
    get_node_centrality_row,
    summarize_route,
)

SNAPSHOT_PATH = SNAPSHOT_DIR / "describegraph.json"

st.set_page_config(
    page_title="lngraph — Lightning Network Analysis",
    page_icon="⚡",
    layout="wide",
)


@st.cache_resource
def load_graph():
    if not SNAPSHOT_PATH.exists():
        return None
    return ingest_file(SNAPSHOT_PATH)


@st.cache_data
def precompute_centrality(_G):
    """Compute all centrality dicts once; results cached across Streamlit re-runs.

    Uses sampled betweenness (k=100) for speed on large graphs. The underscore
    prefix on _G tells Streamlit not to hash the graph object itself.
    """
    deg = degree_centrality(_G)
    close = closeness_centrality(_G)
    eig = eigenvector_centrality(_G)
    btwn = betweenness_sampled(_G, k=100)
    return deg, close, eig, btwn


def main() -> None:
    st.title("lngraph — Lightning Network Graph Analysis")
    st.caption(
        "Data-science platform for the public Lightning Network channel graph. "
        "Data: mempool.space public API. No own node required."
    )

    G = load_graph()
    if G is None:
        st.error(
            "No snapshot found at `data/snapshots/describegraph.json`. "
            "Run `make crawl` first to download the mainnet graph."
        )
        st.stop()

    n_nodes = G.number_of_nodes()
    n_channels = G.number_of_edges() // 2
    st.info(
        f"Loaded graph: **{n_nodes:,} nodes** · **{n_channels:,} channels** "
        f"(each channel stored as 2 directed edges)"
    )

    node_tab, route_tab = st.tabs(["Node Inspector", "Route Explorer"])

    node_options = build_node_options(G)
    option_ids = [nid for nid, _ in node_options]
    option_labels = [lbl for _, lbl in node_options]

    # ------------------------------------------------------------------
    # Tab 1: Node Inspector
    # ------------------------------------------------------------------
    with node_tab:
        st.header("Node Inspector")

        ctrl_col, graph_col = st.columns([1, 2])

        with ctrl_col:
            node_idx = st.selectbox(
                "Select node",
                range(len(option_ids)),
                format_func=lambda i: option_labels[i],
                key="node_select",
            )
            node_id = option_ids[node_idx]
            alias = (G.nodes[node_id].get("alias") or "").strip()

            radius = st.number_input(
                "Neighbourhood radius",
                min_value=1,
                max_value=3,
                value=1,
                step=1,
                key="radius_input",
            )

            st.markdown("---")
            st.markdown(f"**Alias:** {alias or '*(none)*'}")
            st.markdown(f"**Pubkey:**")
            st.code(node_id, language=None)

            in_deg = G.in_degree(node_id)
            out_deg = G.out_degree(node_id)
            st.metric("Channels (in)", in_deg)
            st.metric("Channels (out)", out_deg)

        with graph_col:
            st.subheader("Centrality")
            with st.spinner("Computing centrality (sampled betweenness k=100)…"):
                deg, close, eig, btwn = precompute_centrality(G)
            row = get_node_centrality_row(node_id, deg, close, eig, btwn)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Degree", f"{row['degree']:.4f}")
            m2.metric("Cap-weighted deg", f"{row['capacity_weighted']:.4f}")
            m3.metric("Closeness", f"{row['closeness']:.4f}")
            m4.metric("Eigenvector", f"{row['eigenvector']:.2e}")
            m5.metric("Betweenness", f"{row['betweenness']:.2e}")

            st.subheader("Neighbourhood Graph")
            fig = neighborhood_graph_plot(G, node_id, radius=int(radius))
            st.pyplot(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # Tab 2: Route Explorer
    # ------------------------------------------------------------------
    with route_tab:
        st.header("Route Explorer")

        rc1, rc2, rc3 = st.columns([2, 2, 1])
        with rc1:
            src_idx = st.selectbox(
                "Source node",
                range(len(option_ids)),
                format_func=lambda i: option_labels[i],
                key="src_select",
            )
        with rc2:
            tgt_idx = st.selectbox(
                "Target node",
                range(len(option_ids)),
                index=min(1, len(option_ids) - 1),
                format_func=lambda i: option_labels[i],
                key="tgt_select",
            )
        with rc3:
            amount_msat = st.number_input(
                "Amount (msat)",
                min_value=1_000,
                max_value=10_000_000_000,
                value=1_000_000,
                step=100_000,
                key="amount_input",
            )

        src_id = option_ids[src_idx]
        tgt_id = option_ids[tgt_idx]

        if src_id == tgt_id:
            st.warning("Source and target must be different nodes.")
            return

        amt = int(amount_msat)
        fee_route = find_route(G, src_id, tgt_id, amount_msat=amt)
        succ_route = success_optimal_route(G, src_id, tgt_id, amount_msat=amt)

        if fee_route is None and succ_route is None:
            st.error(
                "No route found between the selected nodes. "
                "Try a different pair or run `make enrich` to populate fee policies."
            )
            return

        fee_col, succ_col = st.columns(2)

        with fee_col:
            st.subheader("Fee-Optimal Route")
            if fee_route:
                fee_prob = path_success_probability(G, fee_route["path"], amt)
                s = summarize_route(fee_route, success_probability=fee_prob)
                fa, fb, fc, fd = st.columns(4)
                fa.metric("Hops", s["hops"])
                fb.metric("Total fee (msat)", f"{s['total_fee_msat']:.1f}")
                fc.metric("Total CLTV", s["total_cltv"])
                fd.metric("Success prob", f"{fee_prob:.1%}")
                st.dataframe(format_route_table(fee_route), use_container_width=True)
            else:
                st.info("No fee-optimal route found.")

        with succ_col:
            st.subheader("Success-Optimal Route")
            if succ_route:
                s2 = summarize_route(
                    succ_route,
                    success_probability=succ_route["success_probability"],
                )
                sa, sb, sc, sd = st.columns(4)
                sa.metric("Hops", s2["hops"])
                sb.metric("Success prob", f"{s2['success_probability']:.1%}")
                sc.metric("Total CLTV", "—")
                sd.metric("Total fee", "—")

                hop_rows = [
                    {
                        "hop": i + 1,
                        "from": h["from"][:16] + "…",
                        "to": h["to"][:16] + "…",
                        "capacity_msat": h["capacity_msat"],
                        "success_prob": f"{h['success_prob']:.1%}",
                    }
                    for i, h in enumerate(succ_route.get("per_hop", []))
                ]
                if hop_rows:
                    st.dataframe(pd.DataFrame(hop_rows), use_container_width=True)
            else:
                st.info("No success-optimal route found.")


if __name__ == "__main__":
    main()
