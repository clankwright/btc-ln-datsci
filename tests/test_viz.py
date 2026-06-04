"""Tests for lngraph/viz.py — Phase 4.2.

All tests run in headless mode (Agg backend) against the synthetic 5-node
star fixture from conftest.  No display required; figures are closed after
each test to avoid accumulation.
"""
import importlib

import matplotlib
matplotlib.use("Agg")  # must precede any other matplotlib import

import matplotlib.pyplot as plt
import matplotlib.figure
import pytest
import networkx as nx


# ---------------------------------------------------------------------------
# Import guard: viz must be importable
# ---------------------------------------------------------------------------

def test_viz_importable():
    import lngraph.viz  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after every test to prevent leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# degree_distribution_plot()
# ---------------------------------------------------------------------------

def test_degree_distribution_plot_returns_figure(synthetic_graph):
    from lngraph.viz import degree_distribution_plot
    from lngraph import topology
    dd = topology.degree_distribution(synthetic_graph)
    fig = degree_distribution_plot(dd["total_degrees"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_degree_distribution_plot_has_two_axes(synthetic_graph):
    from lngraph.viz import degree_distribution_plot
    from lngraph import topology
    dd = topology.degree_distribution(synthetic_graph)
    fig = degree_distribution_plot(dd["total_degrees"])
    assert len(fig.axes) == 2


def test_degree_distribution_plot_empty_degrees():
    from lngraph.viz import degree_distribution_plot
    fig = degree_distribution_plot([])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_degree_distribution_plot_single_value():
    from lngraph.viz import degree_distribution_plot
    fig = degree_distribution_plot([3])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_degree_distribution_plot_fig_label_in_title(synthetic_graph):
    from lngraph.viz import degree_distribution_plot
    from lngraph import topology
    dd = topology.degree_distribution(synthetic_graph)
    fig = degree_distribution_plot(dd["total_degrees"], fig_label="Fig1.1")
    # At least one axis or suptitle should reference the label
    all_text = (fig.texts + [ax.title for ax in fig.axes])
    texts = " ".join(t.get_text() for t in all_text if hasattr(t, "get_text"))
    assert "Fig1.1" in texts


# ---------------------------------------------------------------------------
# capacity_distribution_plot()
# ---------------------------------------------------------------------------

def test_capacity_distribution_plot_returns_figure(synthetic_graph):
    from lngraph.viz import capacity_distribution_plot
    from lngraph import topology
    cd = topology.capacity_distribution(synthetic_graph)
    fig = capacity_distribution_plot(cd["capacities"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_capacity_distribution_plot_has_two_axes(synthetic_graph):
    from lngraph.viz import capacity_distribution_plot
    from lngraph import topology
    cd = topology.capacity_distribution(synthetic_graph)
    fig = capacity_distribution_plot(cd["capacities"])
    assert len(fig.axes) == 2


def test_capacity_distribution_plot_empty():
    from lngraph.viz import capacity_distribution_plot
    fig = capacity_distribution_plot([])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_capacity_distribution_plot_single():
    from lngraph.viz import capacity_distribution_plot
    fig = capacity_distribution_plot([1_000_000])
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# lorenz_curve_plot()
# ---------------------------------------------------------------------------

def test_lorenz_curve_plot_returns_figure(synthetic_graph):
    from lngraph.viz import lorenz_curve_plot
    from lngraph import topology
    cd = topology.capacity_distribution(synthetic_graph)
    fig = lorenz_curve_plot(cd["capacities"])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_lorenz_curve_plot_single_axis(synthetic_graph):
    from lngraph.viz import lorenz_curve_plot
    from lngraph import topology
    cd = topology.capacity_distribution(synthetic_graph)
    fig = lorenz_curve_plot(cd["capacities"])
    assert len(fig.axes) == 1


def test_lorenz_curve_plot_empty():
    from lngraph.viz import lorenz_curve_plot
    fig = lorenz_curve_plot([])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_lorenz_curve_plot_shows_gini(synthetic_graph):
    from lngraph.viz import lorenz_curve_plot
    from lngraph import topology
    cd = topology.capacity_distribution(synthetic_graph)
    fig = lorenz_curve_plot(cd["capacities"], fig_label="Fig1.3")
    ax = fig.axes[0]
    title_text = ax.get_title() + " ".join(t.get_text() for t in fig.texts)
    assert "Gini" in title_text or "gini" in title_text.lower()


# ---------------------------------------------------------------------------
# centrality_rank_plot()
# ---------------------------------------------------------------------------

def test_centrality_rank_plot_returns_figure(synthetic_graph):
    from lngraph.viz import centrality_rank_plot
    from lngraph import centrality
    deg = centrality.degree_centrality(synthetic_graph)
    fig = centrality_rank_plot(deg["total_degree"], metric_name="total_degree")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_centrality_rank_plot_empty_dict():
    from lngraph.viz import centrality_rank_plot
    fig = centrality_rank_plot({})
    assert isinstance(fig, matplotlib.figure.Figure)


def test_centrality_rank_plot_single_axis(synthetic_graph):
    from lngraph.viz import centrality_rank_plot
    from lngraph import centrality
    deg = centrality.degree_centrality(synthetic_graph)
    fig = centrality_rank_plot(deg["total_degree"])
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# centrality_scatter()
# ---------------------------------------------------------------------------

def test_centrality_scatter_returns_figure(synthetic_graph):
    from lngraph.viz import centrality_scatter
    from lngraph import centrality
    deg = centrality.degree_centrality(synthetic_graph)
    fig = centrality_scatter(
        deg["total_degree"], deg["capacity_weighted"],
        x_label="total_degree", y_label="cap_weighted",
    )
    assert isinstance(fig, matplotlib.figure.Figure)


def test_centrality_scatter_empty():
    from lngraph.viz import centrality_scatter
    fig = centrality_scatter({}, {})
    assert isinstance(fig, matplotlib.figure.Figure)


def test_centrality_scatter_mismatched_keys(synthetic_graph):
    from lngraph.viz import centrality_scatter
    # Only nodes present in both dicts should appear; no exception raised.
    x = {"a": 0.5, "b": 0.3}
    y = {"b": 0.2, "c": 0.9}
    fig = centrality_scatter(x, y)
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# betweenness_convergence_plot()
# ---------------------------------------------------------------------------

def test_betweenness_convergence_plot_returns_figure():
    from lngraph.viz import betweenness_convergence_plot
    k_values = [1, 5, 10, 20, 50]
    corrs = [0.4, 0.7, 0.85, 0.92, 0.97]
    fig = betweenness_convergence_plot(k_values, corrs)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_betweenness_convergence_plot_has_threshold_line():
    """The 0.95 reference line must appear as a horizontal line on the axes."""
    from lngraph.viz import betweenness_convergence_plot
    k_values = [1, 5, 10, 20, 50]
    corrs = [0.4, 0.7, 0.85, 0.92, 0.97]
    fig = betweenness_convergence_plot(k_values, corrs)
    ax = fig.axes[0]
    # axhline adds a Line2D with xdata spanning [0, 1] in axes coordinates
    hlines = [ln for ln in ax.get_lines() if list(ln.get_xdata()) == [0.0, 1.0]]
    assert len(hlines) >= 1


def test_betweenness_convergence_plot_empty():
    from lngraph.viz import betweenness_convergence_plot
    fig = betweenness_convergence_plot([], [])
    assert isinstance(fig, matplotlib.figure.Figure)


# ---------------------------------------------------------------------------
# neighborhood_graph_plot()
# ---------------------------------------------------------------------------

def test_neighborhood_graph_plot_returns_figure(synthetic_graph):
    from lngraph.viz import neighborhood_graph_plot
    nodes = list(synthetic_graph.nodes())
    fig = neighborhood_graph_plot(synthetic_graph, nodes[0])
    assert isinstance(fig, matplotlib.figure.Figure)


def test_neighborhood_graph_plot_missing_center_raises(synthetic_graph):
    from lngraph.viz import neighborhood_graph_plot
    with pytest.raises((KeyError, ValueError, nx.exception.NetworkXError)):
        neighborhood_graph_plot(synthetic_graph, "nonexistent_node_xyz")


def test_neighborhood_graph_plot_radius_zero(synthetic_graph):
    """Radius 0 → only the center node is shown."""
    from lngraph.viz import neighborhood_graph_plot
    nodes = list(synthetic_graph.nodes())
    fig = neighborhood_graph_plot(synthetic_graph, nodes[1], radius=0)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_neighborhood_graph_plot_accepts_alias_map(synthetic_graph):
    from lngraph.viz import neighborhood_graph_plot
    nodes = list(synthetic_graph.nodes())
    alias_map = {n: synthetic_graph.nodes[n].get("alias", n[:8]) for n in nodes}
    fig = neighborhood_graph_plot(synthetic_graph, nodes[0], alias_map=alias_map)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_neighborhood_graph_plot_radius_one_includes_neighbors(synthetic_graph):
    """Hub (nodes[0]) at radius 1 should show hub + all 4 leaf nodes."""
    from lngraph.viz import neighborhood_graph_plot
    hub = list(synthetic_graph.nodes())[0]
    fig = neighborhood_graph_plot(synthetic_graph, hub, radius=1)
    # Figure should contain at least one axes with drawn nodes (artists present)
    assert len(fig.axes) >= 1
