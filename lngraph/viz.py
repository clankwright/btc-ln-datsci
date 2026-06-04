"""Visualization helpers for lngraph notebooks and analysis (Phase 4.2).

All public functions accept plain Python data (lists, dicts, networkx graphs)
and return a ``matplotlib.figure.Figure``.  Callers control whether to display
(``plt.show()``) or save (``fig.savefig(...)``).

Conventions applied across every figure:
  - Count axes use log y-scale so heavy tails are visible.
  - Figure labels use ``Fig<nb>.<idx>`` prefixes (passed via ``fig_label``).
  - Open/edge-only markers on log-log scatters so points don't obscure fits.
"""
from __future__ import annotations

from typing import Callable

import matplotlib.pyplot as plt
import matplotlib.figure
import networkx as nx
import numpy as np


def degree_distribution_plot(
    degrees: list[int],
    fig_label: str = "Degree distribution",
) -> matplotlib.figure.Figure:
    """2-panel: histogram (linear-x, log-count) + log-log scatter with power-law fit.

    Mirrors the pattern in notebook 01, cell 9.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    if degrees:
        axes[0].hist(
            degrees,
            bins=min(50, max(5, len(set(degrees)))),
            color="steelblue",
            edgecolor="white",
            linewidth=0.4,
        )
    axes[0].set_xlabel("Total degree (in + out)")
    axes[0].set_ylabel("Node count")
    axes[0].set_yscale("log")
    axes[0].set_title("Degree distribution (linear x, log count)")

    if len(degrees) >= 3:
        unique_deg, counts = np.unique(degrees, return_counts=True)
        axes[1].scatter(
            unique_deg,
            counts,
            facecolors="none",
            edgecolors="steelblue",
            alpha=0.6,
            s=12,
            linewidths=0.8,
            zorder=3,
        )
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Degree (log)")
        axes[1].set_ylabel("Count (log)")
        if len(unique_deg) >= 5:
            mask = (unique_deg > 0) & (counts > 0)
            from scipy import stats as sp_stats
            slope, intercept, r, _, _ = sp_stats.linregress(
                np.log10(unique_deg[mask]), np.log10(counts[mask])
            )
            x_fit = np.linspace(unique_deg[mask].min(), unique_deg[mask].max(), 100)
            axes[1].plot(
                x_fit,
                10 ** (intercept + slope * np.log10(x_fit)),
                "r--",
                lw=1.5,
                zorder=2,
                label=f"slope={slope:.2f}  R²={r**2:.2f}",
            )
            axes[1].legend(fontsize=9)
            axes[1].set_title(f"Degree (log-log)   γ ≈ {-slope:.2f}")
        else:
            axes[1].set_title("Degree (log-log)")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Too few unique degrees\nfor log-log plot",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=11,
        )
        axes[1].set_title("Log-log (insufficient data)")

    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = max(degrees) if degrees else 0
    n = len(degrees)
    fig.suptitle(
        f"{fig_label}: mean={mean_deg:.1f}   max={max_deg}   n={n:,} nodes",
        y=1.02,
        fontsize=11,
    )
    plt.tight_layout()
    return fig


def capacity_distribution_plot(
    capacities_sat: list[int],
    fig_label: str = "Capacity distribution",
) -> matplotlib.figure.Figure:
    """2-panel: histogram (linear-x, log-count) + log-x histogram (log-x, log-count).

    Mirrors the pattern in notebook 01, cell 11.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    caps_btc = [c / 1e8 for c in capacities_sat if c > 0]

    if caps_btc:
        axes[0].hist(
            [c / 1e8 for c in capacities_sat],
            bins=min(60, max(5, len(caps_btc))),
            color="darkorange",
            edgecolor="white",
            linewidth=0.4,
        )
    axes[0].set_xlabel("Capacity (BTC)")
    axes[0].set_ylabel("Channel count")
    axes[0].set_yscale("log")
    axes[0].set_title("Capacity distribution (linear x, log count)")

    if caps_btc:
        log_caps = [np.log10(c) for c in caps_btc]
        axes[1].hist(
            log_caps,
            bins=min(40, max(5, len(log_caps))),
            color="darkorange",
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1].set_xlabel("log₁₀(Capacity / BTC)")
        axes[1].set_ylabel("Channel count")
        axes[1].set_yscale("log")
        axes[1].set_title("Capacity distribution (log-x, log count)")
    else:
        axes[1].text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=11,
        )
        axes[1].set_title("Capacity (log-x, log count)")

    fig.suptitle(fig_label, y=1.02, fontsize=11)
    plt.tight_layout()
    return fig


def lorenz_curve_plot(
    values: list[float],
    label: str = "channels",
    fig_label: str = "Lorenz curve",
) -> matplotlib.figure.Figure:
    """Single-panel Lorenz curve with Gini annotation.

    Mirrors the pattern in notebook 01, cell 13.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect equality")

    if len(values) >= 2:
        from lngraph.topology import gini as _gini
        g = _gini(values)
        vals = np.array(sorted(values), dtype=float)
        n = len(vals)
        cum_pop = np.concatenate([[0], np.arange(1, n + 1) / n])
        cum_val = np.concatenate([[0], np.cumsum(vals) / vals.sum()])
        ax.plot(
            cum_pop,
            cum_val,
            color="darkorange",
            lw=2,
            label=f"{label}  Gini={g:.3f}",
        )
        ax.fill_between(cum_pop, cum_pop, cum_val, alpha=0.15, color="darkorange")
        ax.set_title(f"{fig_label}   Gini={g:.3f}")
    else:
        ax.set_title(fig_label)

    ax.set_xlabel("Cumulative share of channels (sorted by value)")
    ax.set_ylabel("Cumulative share of total")
    ax.legend()
    plt.tight_layout()
    return fig


def centrality_rank_plot(
    centrality_dict: dict,
    metric_name: str = "centrality",
    fig_label: str = "",
) -> matplotlib.figure.Figure:
    """Sorted rank vs centrality value for a single centrality dict.

    Useful for a quick "who are the top nodes?" scan.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    if centrality_dict:
        vals = sorted(centrality_dict.values(), reverse=True)
        ax.plot(range(1, len(vals) + 1), vals, "o-", markersize=4, color="steelblue")

    ax.set_xlabel("Rank")
    ax.set_ylabel(metric_name)
    title = fig_label if fig_label else f"{metric_name} by rank"
    ax.set_title(title)
    plt.tight_layout()
    return fig


def centrality_scatter(
    x_dict: dict,
    y_dict: dict,
    x_label: str = "x",
    y_label: str = "y",
    fig_label: str = "",
) -> matplotlib.figure.Figure:
    """Scatter plot of two centrality dicts.

    Only nodes present in both dicts are plotted; extra nodes are silently
    dropped so callers don't need to align them first.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    common = sorted(set(x_dict) & set(y_dict))
    if common:
        xs = [x_dict[n] for n in common]
        ys = [y_dict[n] for n in common]
        # Open/edge-only markers (matches nb1 style) so overlapping points
        # don't obscure one another or any overlaid reference line.
        ax.scatter(xs, ys, facecolors="none", edgecolors="steelblue",
                   alpha=0.6, s=30, linewidths=0.8, zorder=3)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title = fig_label if fig_label else f"{x_label} vs {y_label}"
    ax.set_title(title)
    plt.tight_layout()
    return fig


def betweenness_convergence_plot(
    k_values: list[int],
    correlations: list[float],
    threshold: float = 0.95,
    fig_label: str = "Betweenness k-sampling convergence",
) -> matplotlib.figure.Figure:
    """k-pivot count vs Spearman ρ convergence curve.

    Mirrors the pattern in notebook 02, cell 14.  A horizontal dashed
    reference line is always drawn at ``threshold`` (default 0.95).
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    if k_values and correlations:
        ax.plot(k_values, correlations, "o-", color="steelblue", markersize=5)

    ax.axhline(threshold, color="gray", lw=1, linestyle="--", label=f"r = {threshold}")
    ax.set_xlabel("k (number of pivot nodes)")
    ax.set_ylabel("Spearman rank correlation with exact")
    ax.set_title(fig_label)
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    plt.tight_layout()
    return fig


def neighborhood_graph_plot(
    G: nx.MultiDiGraph,
    center: str,
    radius: int = 1,
    alias_map: dict | None = None,
    fig_label: str = "",
    seed: int = 42,
) -> matplotlib.figure.Figure:
    """Ego subgraph around ``center`` node, spring-layout.

    Parameters
    ----------
    G:
        Full MultiDiGraph (as produced by ingest.ingest).
    center:
        Node ID of the focal node.  Must exist in G; raises ValueError if not.
    radius:
        Number of hops to include.  0 → centre only; 1 → direct neighbours.
    alias_map:
        Optional {node_id: label} for display.  Falls back to first 8 chars.
    fig_label:
        Text for the figure suptitle / title.
    seed:
        Random seed for spring layout reproducibility.
    """
    if center not in G:
        raise ValueError(f"center node {center!r} not found in graph")

    # Build the ego subgraph (undirected horizon so both in/out neighbours
    # at each radius step are included, matching the LN bidirectional view).
    ego = nx.ego_graph(G.to_undirected(), center, radius=radius, undirected=True)
    sub = G.subgraph(ego.nodes()).copy()

    label_fn: Callable[[str], str]
    if alias_map:
        label_fn = lambda n: alias_map.get(n, n[:8])
    else:
        label_fn = lambda n: G.nodes[n].get("alias", n[:8]) if n in G.nodes else n[:8]

    labels = {n: label_fn(n) for n in sub.nodes()}

    # Colour: centre in orange, neighbours in blue
    node_colors = [
        "darkorange" if n == center else "steelblue" for n in sub.nodes()
    ]

    pos = nx.spring_layout(sub, seed=seed, k=1.5)

    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx(
        sub,
        pos=pos,
        labels=labels,
        node_color=node_colors,
        node_size=600,
        font_size=8,
        arrows=True,
        ax=ax,
    )
    title = fig_label if fig_label else f"Neighbourhood of {label_fn(center)} (radius={radius})"
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    return fig
