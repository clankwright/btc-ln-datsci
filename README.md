# lngraph — Lightning Network Graph Analysis

A data-science platform for analyzing the public Lightning Network channel graph, focused on **network centrality and routing efficiency** on real mainnet data.

Ingests the real mainnet graph from the [mempool.space](https://mempool.space) public API, builds a `networkx` model, and computes the metrics a routing and centrality researcher cares about.

## What it does

- BFS crawl of the LN channel graph from high-connectivity seed hubs (~17 500 nodes, ~41 000 channels as of May 2026)
- Policy enrichment: per-direction fee rates, base fees, CLTV deltas from the channel-detail endpoint
- Centrality: degree, closeness, eigenvector, betweenness (sampled + igraph-exact, fee-weighted)
- Routing: Dijkstra pathfinding with real fee+CLTV cost function, k-shortest paths, fee-vs-hops analysis
- Liquidity / partial observability: uniform-prior balance model, success-probability routing, Monte Carlo validation
- Jupyter notebooks telling the full analysis story (01 topology → 02 centrality → 03 routing → 04 liquidity)

## Notebooks

The analysis is told as four notebooks, meant to be read in order:

1. [01 — Crawl & Topology](notebooks/01_crawl_and_topology.ipynb) — what the data is and how it was collected; connected components, degree and capacity distributions, capacity inequality (Gini), small-world properties.
2. [02 — Centrality](notebooks/02_centrality.ipynb) — degree, closeness, eigenvector, and betweenness (k-pivot sampled vs igraph-exact, plus fee-weighted), and which nodes lead each measure.
3. [03 — Routing Efficiency](notebooks/03_routing_efficiency.ipynb) — the real fee + CLTV cost function, k-shortest alternative paths, the fee-vs-hops trade-off, and the network-wide fee landscape.
4. [04 — Liquidity & Partial Observability](notebooks/04_liquidity_partial_observability.ipynb) — the uniform-prior success-probability model, success- vs fee-optimal routing, Monte-Carlo validation, and why multi-path payments improve reliability.

## Data source limits

mempool.space provides keyless public access, rate-limited to roughly 1.5 req/s. The crawl is resumable across sessions and backs off on HTTP 429. The snapshot covers public gossip data only — channel *capacity* is public; per-direction *balances* are not (the partial-observability theme of notebook 04).

## Quickstart

```bash
# 1. Install
make install            # creates .venv, installs deps + package

# 2. Crawl the mainnet graph (resumable; run again to grow coverage)
make crawl              # BFS from seed hubs → data/snapshots/describegraph.json

# 3. Enrich with per-direction fee policy
make enrich             # hits /channels/{id} for each channel (bounded by ENRICH_MAX_CHANNELS)

# 4. Run the notebooks
make lab                # opens JupyterLab; run 01 → 02 → 03 → 04 in order

# 5. Run tests
make test               # default suite (excludes live + notebook markers)
make smoke              # opt-in: live API smoke tests (require network)
pytest -m notebook      # opt-in: notebook execution tests (require Jupyter kernel)
```

## Partial-observability caveat

The routing and liquidity analysis (notebooks 03 + 04) uses a uniform-prior model: absent any balance information, each channel is modelled as equally likely to hold any split. This matches the baseline used in academic LN routing research. A Bayesian update from observed payment failures would tighten these estimates but is out of scope.

## Project layout

```
lngraph/          Python package (fetch_mempool, ingest, graph, topology,
                  centrality, routing, liquidity)
notebooks/        Jupyter analysis notebooks (01–04)
tests/            pytest suite — unit tests on synthetic fixtures + opt-in live/notebook tests
data/cache/       Raw HTTP response cache (gitignored)
data/snapshots/   Assembled describegraph JSON (gitignored)
tests/fixtures/   Committed real-data subgraph for notebook CI tests
```

## References

- [mempool.space](https://mempool.space) — public Lightning Network API and data source
- [networkx](https://networkx.org) and [python-igraph](https://igraph.org/python/) — graph algorithms

## License

MIT — see [LICENSE](LICENSE).
