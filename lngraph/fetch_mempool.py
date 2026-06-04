"""Resumable BFS crawler for the mempool.space Lightning API.

Assembles a real mainnet subgraph by breadth-first search from a set of
top-connectivity seed hubs. For each node it pages through the public
``/channels`` endpoint, discovers peer pubkeys, and normalizes every channel
into the canonical ``describegraph`` schema (the internal contract every
analysis module consumes). Raw HTTP responses are cached on disk so a crawl is
cheap to re-run, and the BFS frontier is persisted so a crawl can resume across
sessions. Transient failures (HTTP 429 / 5xx) are retried with exponential
backoff.

mempool.space publishes no API key and no hard rate limit, but empirically
tolerates ~1.5 req/s; the crawler self-throttles to that budget.

CLI::

    python -m lngraph.fetch_mempool            # crawl from the seed hubs
    python -m lngraph.fetch_mempool --stats    # print /statistics/latest

Everything is mockable: pass ``cache_dir`` / ``snapshot_dir`` to redirect disk
writes, and the network layer goes through plain ``requests`` so tests can
patch it with ``requests_mock``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import requests

from lngraph import config

# Name of the resumable crawl-state file written under the cache dir.
STATE_FILENAME = "crawl_state.json"

# Module-level monotonic timestamp of the last HTTP call, for rate limiting.
_last_call = [0.0]


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_path(url: str, cache_dir: Path) -> Path:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{digest}.json"


def _cache_read(url: str, cache_dir: Path) -> Any | None:
    path = _cache_path(url, cache_dir)
    if path.exists():
        return json.loads(path.read_text())
    return None


def _cache_write(url: str, data: Any, cache_dir: Path) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(url, cache_dir).write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# HTTP with rate limiting + exponential backoff
# ---------------------------------------------------------------------------

def _rate_limit() -> None:
    rate = config.RATE_LIMIT_CALLS_PER_SEC
    if rate <= 0:
        return
    min_interval = 1.0 / rate
    elapsed = time.monotonic() - _last_call[0]
    wait = min_interval - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _get(url: str, cache_dir: Path | None = None,
         session: requests.Session | None = None) -> Any:
    """GET ``url`` returning parsed JSON, using the disk cache and backing off
    on 429/5xx. Raises after ``BACKOFF_MAX_RETRIES`` exhausted retries."""
    cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    cached = _cache_read(url, cache_dir)
    if cached is not None:
        return cached

    getter = session.get if session is not None else requests.get
    last_exc: Exception | None = None
    for attempt in range(config.BACKOFF_MAX_RETRIES + 1):
        _rate_limit()
        resp = getter(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            _cache_write(url, data, cache_dir)
            return data
        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            last_exc = requests.HTTPError(
                f"{resp.status_code} for {url}", response=resp)
            delay = min(
                config.BACKOFF_BASE_DELAY * (2 ** attempt),
                config.BACKOFF_MAX_DELAY,
            )
            time.sleep(delay)
            continue
        resp.raise_for_status()
    raise RuntimeError(
        f"giving up on {url} after {config.BACKOFF_MAX_RETRIES} retries"
    ) from last_exc


# ---------------------------------------------------------------------------
# Endpoint wrappers
# ---------------------------------------------------------------------------

def fetch_statistics(cache_dir: Path | None = None,
                     session: requests.Session | None = None) -> dict:
    """Return the ``latest`` block of ``/v1/lightning/statistics/latest``."""
    url = f"{config.MEMPOOL_BASE_URL}/v1/lightning/statistics/latest"
    data = _get(url, cache_dir=cache_dir, session=session)
    return data.get("latest", data)


def fetch_node_channels(pubkey: str, cache_dir: Path | None = None,
                        session: requests.Session | None = None) -> list[dict]:
    """Page through every open channel of ``pubkey`` and return the raw channel
    dicts (mempool list schema, not yet normalized).

    The endpoint requires ``status=open`` — omitting it returns HTTP 400 — and
    returns a bare JSON array. ``index`` is an **item offset** (page size 10),
    so we advance by the number of channels returned, not by 1. Pagination ends
    on an empty page or one that introduces no new channel ids (a defensive
    guard against a repeated tail window)."""
    channels: list[dict] = []
    seen: set[str] = set()
    index = 0
    for _ in range(config.CRAWL_MAX_PAGES_PER_NODE):
        url = (
            f"{config.MEMPOOL_BASE_URL}/v1/lightning/channels"
            f"?public_key={pubkey}&status=open&index={index}"
        )
        page = _get(url, cache_dir=cache_dir, session=session)
        # Real schema is a bare array; tolerate a {"channels": [...]} wrapper too.
        page_channels = page if isinstance(page, list) else page.get("channels", [])
        if not page_channels:
            break
        fresh = [c for c in page_channels if str(c.get("id")) not in seen]
        if not fresh:
            break
        seen.update(str(c.get("id")) for c in fresh)
        channels.extend(fresh)
        index += len(page_channels)
    return channels


# ---------------------------------------------------------------------------
# Normalization: mempool channel JSON → describegraph edge
# ---------------------------------------------------------------------------

def normalize_channel(raw: dict, source_pub: str) -> dict:
    """Convert one mempool channel-list element to a describegraph edge dict.

    The list endpoint returns the *counterparty* (``node``) only; the queried
    pubkey (``source_pub``) is the other end, so ``node1_pub`` is the source and
    ``node2_pub`` is the counterparty. This endpoint carries no per-direction
    policy — both ``node*_policy`` fields stay ``None`` until the Phase 3.1
    enrichment pass fills them from ``/channels/{short_id}``. ``short_id`` is
    retained on the edge so that pass can address each channel.
    """
    counterparty = raw.get("node", {}) or {}
    return {
        "channel_id": str(raw["id"]),
        "short_id": str(raw.get("short_id", "")),
        "chan_point": "",
        "capacity": str(raw["capacity"]),
        "node1_pub": source_pub,
        "node2_pub": counterparty.get("public_key", ""),
        "node1_policy": None,
        "node2_policy": None,
    }


# ---------------------------------------------------------------------------
# Channel detail: per-direction fee policy (Phase 3.1)
# ---------------------------------------------------------------------------

def fetch_channel_detail(channel_id: str, cache_dir: Path | None = None,
                         session: requests.Session | None = None) -> dict:
    """Return the raw channel-detail dict for the numeric ``channel_id``.

    Hits ``/v1/lightning/channels/{channel_id}`` and returns the parsed JSON.
    Uses the same disk-cache + backoff infrastructure as the other endpoints.

    The detail response embeds per-direction fee policy directly in the
    ``node_left`` and ``node_right`` objects (fields: ``base_fee_mtokens``,
    ``fee_rate`` ppm, ``cltv_delta``, ``min_htlc_mtokens``, ``max_htlc_mtokens``,
    ``is_disabled`` as 0/1 int). ``enrich_policies`` extracts those fields and
    normalizes them into the describegraph ``node*_policy`` schema.

    Note: the endpoint requires the *numeric* channel id (``edge["channel_id"]``),
    not the human-readable short channel id (``edge["short_id"]``). The two are
    different: ``775673x1007x1`` vs ``852861482917888001``.
    """
    url = f"{config.MEMPOOL_BASE_URL}/v1/lightning/channels/{channel_id}"
    return _get(url, cache_dir=cache_dir, session=session)


def _normalize_policy(raw: dict | None) -> dict | None:
    """Convert one mempool ``node_left`` / ``node_right`` block → describegraph policy.

    The real channel-detail endpoint embeds policy fields directly in the node
    object alongside metadata (alias, channels count, lat/lon, etc.). Only the
    policy fields are extracted; extras are ignored.

    Mempool field names and types → describegraph schema:
      ``base_fee_mtokens`` (int)  → ``fee_base_msat`` (str)
      ``fee_rate`` (int ppm)      → ``fee_rate_milli_msat`` (str)
      ``cltv_delta`` (int)        → ``time_lock_delta`` (int)
      ``min_htlc_mtokens`` (int)  → ``min_htlc`` (str)
      ``max_htlc_mtokens`` (int)  → ``max_htlc_msat`` (str)
      ``is_disabled`` (0 or 1)    → ``disabled`` (bool)
    """
    if raw is None:
        return None
    # A node object with NO announced channel_update for this direction comes
    # back with null policy fields. `cltv_delta` is mandatory in a real
    # channel_update, so its absence marks an unannounced/unknown direction:
    # return None (no policy), NOT a zero-fee policy. Coercing nulls to 0 below
    # would otherwise mint a phantom {fee 0, cltv 0} policy — indistinguishable
    # from a genuine zero-fee channel — and since `require_policy` accepts any
    # non-None dict and Dijkstra always prefers the cheapest edge, those phantom
    # edges capture every route and produce spurious 0-fee / 0-CLTV paths.
    if raw.get("cltv_delta") is None:
        return None
    # Use `raw.get(field) or default` rather than `raw.get(field, default)`
    # because dict.get returns None (not the default) when the key is present
    # with a JSON null value — and int(None) / str(None) would crash or silently
    # produce "None" strings that break downstream int() coercions. (Genuine
    # zero-fee channels keep fee_base/fee_rate == 0 here; only cltv_delta being
    # null marks "no policy" above.)
    return {
        "fee_base_msat": str(raw.get("base_fee_mtokens") or 0),
        "fee_rate_milli_msat": str(raw.get("fee_rate") or 0),
        "time_lock_delta": int(raw.get("cltv_delta") or 0),
        "min_htlc": str(raw.get("min_htlc_mtokens") or 0),
        "max_htlc_msat": str(raw.get("max_htlc_mtokens") or 0),
        "disabled": bool(raw.get("is_disabled") or 0),
    }


def enrich_policies(describegraph: dict, max_enrich: int | None = None,
                    cache_dir: Path | None = None,
                    snapshot_dir: Path | None = None,
                    session: requests.Session | None = None,
                    write_snapshot: bool = False) -> dict:
    """Fill ``node1_policy`` / ``node2_policy`` on every edge by hitting
    ``/v1/lightning/channels/{channel_id}`` for per-direction fee policy.

    Each edge's numeric ``channel_id`` is used to address the detail endpoint.
    Edges without a ``channel_id`` are skipped. ``max_enrich`` (default
    ``config.ENRICH_MAX_CHANNELS``) caps the total HTTP requests so the pass
    can be bounded to a sampled subgraph. Individual fetch errors are swallowed
    so a transient failure does not abort the whole pass.

    Returns the same ``describegraph`` dict (modified in-place) and optionally
    rewrites the snapshot at ``snapshot_dir/describegraph.json``.
    """
    max_enrich = max_enrich if max_enrich is not None else config.ENRICH_MAX_CHANNELS
    cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    snapshot_dir = (
        Path(snapshot_dir) if snapshot_dir is not None else config.SNAPSHOT_DIR
    )

    enriched = 0
    for edge in describegraph.get("edges", []):
        if enriched >= max_enrich:
            break
        channel_id = edge.get("channel_id", "")
        if not channel_id:
            continue
        try:
            detail = fetch_channel_detail(
                channel_id, cache_dir=cache_dir, session=session)
        except Exception:
            continue

        # Policy fields are embedded in node_left / node_right; match by pubkey.
        node_left = detail.get("node_left") or {}
        node_right = detail.get("node_right") or {}
        by_pub = {}
        if node_left.get("public_key"):
            by_pub[node_left["public_key"]] = node_left
        if node_right.get("public_key"):
            by_pub[node_right["public_key"]] = node_right

        edge["node1_policy"] = _normalize_policy(by_pub.get(edge.get("node1_pub")))
        edge["node2_policy"] = _normalize_policy(by_pub.get(edge.get("node2_pub")))
        enriched += 1

    if write_snapshot:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        out = snapshot_dir / "describegraph.json"
        out.write_text(json.dumps(describegraph))

    return describegraph


def _alias_of(raw_node: dict | None) -> str:
    return (raw_node or {}).get("alias", "") or ""


# ---------------------------------------------------------------------------
# Resumable crawl state
# ---------------------------------------------------------------------------

def _load_state(cache_dir: Path) -> dict | None:
    path = Path(cache_dir) / STATE_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_state(cache_dir: Path, frontier: Iterable[str], visited: set[str],
                nodes: dict[str, str], channels: dict[str, dict]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "frontier": list(frontier),
        "visited": sorted(visited),
        "nodes": nodes,
        "channels": channels,
    }
    (cache_dir / STATE_FILENAME).write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# BFS crawl
# ---------------------------------------------------------------------------

def crawl(seeds: list[str] | None = None, max_nodes: int | None = None,
          cache_dir: Path | None = None, snapshot_dir: Path | None = None,
          session: requests.Session | None = None, resume: bool = True,
          write_snapshot: bool = True,
          progress_fn=None) -> dict:
    """BFS the mempool graph from ``seeds`` and return a describegraph dict.

    Stops once ``max_nodes`` distinct nodes have been fetched or the frontier
    drains. Persists a resumable state file after every node so an interrupted
    crawl continues from where it stopped (``resume=True``).

    ``progress_fn``, if provided, is called after each node is processed with
    ``(visited_count, frontier_len, channels_count)`` so callers can report
    progress without needing to parse stdout.
    """
    seeds = list(seeds) if seeds is not None else list(config.SEED_PUBKEYS)
    max_nodes = max_nodes if max_nodes is not None else config.CRAWL_MAX_NODES
    cache_dir = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    snapshot_dir = (
        Path(snapshot_dir) if snapshot_dir is not None else config.SNAPSHOT_DIR
    )

    nodes: dict[str, str] = {}      # pubkey -> alias
    channels: dict[str, dict] = {}  # channel_id -> normalized edge
    visited: set[str] = set()
    frontier: deque[str] = deque(seeds)

    if resume:
        state = _load_state(cache_dir)
        if state is not None:
            frontier = deque(state.get("frontier", seeds))
            visited = set(state.get("visited", []))
            nodes = dict(state.get("nodes", {}))
            channels = dict(state.get("channels", {}))

    while frontier and len(visited) < max_nodes:
        pub = frontier.popleft()
        if pub in visited:
            continue
        visited.add(pub)
        nodes.setdefault(pub, "")

        for raw in fetch_node_channels(pub, cache_dir=cache_dir, session=session):
            edge = normalize_channel(raw, source_pub=pub)
            channels[edge["channel_id"]] = edge
            # The list element carries only the counterparty `node`; the queried
            # node (`pub`) is the other end and is already in `nodes`/`visited`.
            peer = edge["node2_pub"]
            if not peer:
                continue
            alias = _alias_of(raw.get("node"))
            if alias or peer not in nodes:
                nodes[peer] = alias or nodes.get(peer, "")
            if peer not in visited and peer not in frontier:
                frontier.append(peer)

        _save_state(cache_dir, frontier, visited, nodes, channels)
        if progress_fn is not None:
            progress_fn(len(visited), len(frontier), len(channels))

    describegraph = {
        "nodes": [{"pub_key": k, "alias": v} for k, v in nodes.items()],
        "edges": list(channels.values()),
    }

    if write_snapshot:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        out = snapshot_dir / "describegraph.json"
        out.write_text(json.dumps(describegraph))

    return describegraph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", action="store_true",
                        help="print /statistics/latest and exit")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any persisted frontier and start fresh")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="override CRAWL_MAX_NODES for this run")
    parser.add_argument("--enrich", action="store_true",
                        help="after crawling, enrich all edges with per-direction "
                             "fee policy from /channels/{short_id} "
                             "(bounded by ENRICH_MAX_CHANNELS)")
    args = parser.parse_args(argv)

    if args.stats:
        stats = fetch_statistics()
        print(json.dumps(stats, indent=2))
        return

    def _print_progress(visited, frontier, channels):
        if visited % 100 == 0 or frontier == 0:
            print(f"  progress: {visited} nodes visited, "
                  f"{frontier} in frontier, {channels} channels")

    dg = crawl(max_nodes=args.max_nodes, resume=not args.no_resume,
               progress_fn=_print_progress)
    print(f"crawled {len(dg['nodes'])} nodes / {len(dg['edges'])} channels")

    # Coverage report: compare against the known network totals.
    try:
        stats = fetch_statistics()
        net_nodes = stats.get("node_count", 0)
        net_channels = stats.get("channel_count", 0)
        if net_nodes and net_channels:
            crawled_nodes = len(dg["nodes"])
            crawled_channels = len(dg["edges"])
            node_pct = crawled_nodes / net_nodes * 100
            chan_pct = crawled_channels / net_channels * 100
            print(
                f"coverage: {node_pct:.1f}% nodes "
                f"({crawled_nodes}/{net_nodes}), "
                f"{chan_pct:.1f}% channels "
                f"({crawled_channels}/{net_channels})"
            )
    except Exception:
        pass  # coverage report is non-critical; network call may fail

    if args.enrich:
        dg = enrich_policies(dg, write_snapshot=True)
        n_enriched = sum(
            1 for e in dg["edges"]
            if e.get("node1_policy") or e.get("node2_policy")
        )
        print(f"enriched {n_enriched}/{len(dg['edges'])} channels with policy data")


if __name__ == "__main__":
    main()
