"""Tests for the resumable BFS mempool crawler (Phase 1.1).

The crawler walks the mempool.space Lightning API breadth-first from seed
hubs, normalizes each channel into the canonical describegraph schema, caches
raw responses on disk, backs off on 429/5xx, and persists a resumable frontier.

The real ``/v1/lightning/channels?public_key=X&status=open&index=N`` endpoint
returns a **bare JSON array** of channel objects, each carrying a single
counterparty ``node`` (NOT ``node1``/``node2``) and **no** per-direction
``policies`` array; ``index`` is an item offset (page size 10), so the crawler
advances by the page length, not by 1. These tests encode that real shape — see
the ``@pytest.mark.live`` smoke tests at the bottom, which assert the same shape
against the live API and are excluded from the default run.

All offline HTTP is mocked via the `requests_mock` fixture — no live calls. The
`no_sleep` autouse fixture neutralizes rate-limit and backoff delays so the
suite stays fast.
"""
import json

import pytest

from lngraph import config, fetch_mempool


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make rate-limit + backoff sleeps instantaneous."""
    monkeypatch.setattr(fetch_mempool.time, "sleep", lambda *_a, **_k: None)


def _pub(n):
    return f"node{n:02d}" + "a" * 64


def _raw_channel(chan_id, peer_pub, capacity, *, short_id=None, fee_rate=500,
                 alias=None, status=1):
    """Build a mempool channel-list element (the real source schema).

    The element carries only the *counterparty* (`node`); the queried pubkey is
    implicit in the request URL and is supplied to ``normalize_channel`` as
    ``source_pub``. There is no ``policies`` array on this endpoint.
    """
    return {
        "status": status,
        "closing_reason": None,
        "closing_date": None,
        "capacity": capacity,
        "short_id": short_id if short_id is not None else str(chan_id),
        "id": chan_id,
        "fee_rate": fee_rate,
        "node": {
            "alias": alias if alias is not None else f"alias-{str(peer_pub)[:6]}",
            "public_key": peer_pub,
            "channels": 5,
            "capacity": str(capacity * 3),
        },
    }


def _channels_url(pub, index):
    return (
        f"{config.MEMPOOL_BASE_URL}/v1/lightning/channels"
        f"?public_key={pub}&status=open&index={index}"
    )


def _register_node(requests_mock, pub, channels):
    """Register `channels` (a bare array) at offset 0 plus an empty terminator.

    `index` is an item offset, so the crawler's next request after a page of
    length N is at offset N — that page is the empty terminator.
    """
    requests_mock.get(_channels_url(pub, 0), json=channels)
    requests_mock.get(_channels_url(pub, len(channels)), json=[])


# ---------------------------------------------------------------------------
# fetch_statistics
# ---------------------------------------------------------------------------

def test_fetch_statistics_returns_latest(mock_http, tmp_path):
    stats = fetch_mempool.fetch_statistics(cache_dir=tmp_path)
    assert stats["node_count"] == 17508
    assert stats["channel_count"] == 41343


# ---------------------------------------------------------------------------
# fetch_node_channels — real endpoint contract
# ---------------------------------------------------------------------------

def test_fetch_node_channels_sends_status_open(requests_mock, tmp_path):
    # Regression for 1.6 defect (a): omitting status=open returns HTTP 400.
    _register_node(requests_mock, _pub(0),
                   [_raw_channel("800001x1x0", _pub(1), 1_000_000)])
    fetch_mempool.fetch_node_channels(_pub(0), cache_dir=tmp_path / "cache")
    assert any("status=open" in req.url for req in requests_mock.request_history)


def test_fetch_node_channels_pages_by_item_offset(requests_mock, tmp_path):
    # Regression for 1.6 defect (c): index is an item offset (page size 10),
    # so the crawler must advance by len(page), not by 1. Two full pages of 10
    # then a short page, then the empty terminator.
    page0 = [_raw_channel(f"p0-{i}", _pub(100 + i), 1_000_000) for i in range(10)]
    page1 = [_raw_channel(f"p1-{i}", _pub(200 + i), 1_000_000) for i in range(10)]
    page2 = [_raw_channel(f"p2-{i}", _pub(300 + i), 1_000_000) for i in range(3)]
    requests_mock.get(_channels_url(_pub(0), 0), json=page0)
    requests_mock.get(_channels_url(_pub(0), 10), json=page1)
    requests_mock.get(_channels_url(_pub(0), 20), json=page2)
    requests_mock.get(_channels_url(_pub(0), 23), json=[])

    chans = fetch_mempool.fetch_node_channels(_pub(0), cache_dir=tmp_path / "cache")
    assert len(chans) == 23
    # the offsets actually requested are 0, 10, 20, 23 — never 1
    offsets = {req.qs.get("index", ["?"])[0] for req in requests_mock.request_history}
    assert offsets == {"0", "10", "20", "23"}


# ---------------------------------------------------------------------------
# normalize_channel — mempool list schema → describegraph schema
# ---------------------------------------------------------------------------

def test_normalize_channel_has_describegraph_keys():
    raw = _raw_channel("800001x1x0", _pub(1), 1_000_000)
    edge = fetch_mempool.normalize_channel(raw, source_pub=_pub(0))
    required = {"channel_id", "chan_point", "capacity", "node1_pub",
                "node2_pub", "node1_policy", "node2_policy"}
    assert required.issubset(edge.keys())
    assert edge["channel_id"] == "800001x1x0"
    # describegraph stores capacity as a string
    assert edge["capacity"] == "1000000"


def test_normalize_channel_node1_is_source_node2_is_counterparty():
    # The list endpoint gives only the counterparty `node`; the queried pubkey
    # is node1, the counterparty is node2.
    raw = _raw_channel("800001x1x0", _pub(7), 1_000_000)
    edge = fetch_mempool.normalize_channel(raw, source_pub=_pub(0))
    assert edge["node1_pub"] == _pub(0)
    assert edge["node2_pub"] == _pub(7)


def test_normalize_channel_defers_policy_to_enrichment():
    # The list endpoint carries no per-direction policy; both policies are None
    # until the Phase 3.1 channel-detail enrichment pass fills them in.
    raw = _raw_channel("800001x1x0", _pub(1), 1_000_000)
    edge = fetch_mempool.normalize_channel(raw, source_pub=_pub(0))
    assert edge["node1_policy"] is None
    assert edge["node2_policy"] is None


def test_normalize_channel_preserves_short_id():
    # short_id is retained so Phase 3.1 can hit /channels/{short_id} for policy.
    raw = _raw_channel(892785849701564416, _pub(1), 500_000_000,
                       short_id="811984x2037x0")
    edge = fetch_mempool.normalize_channel(raw, source_pub=_pub(0))
    assert edge["short_id"] == "811984x2037x0"
    # channel_id is stringified even when the API returns it as an int
    assert edge["channel_id"] == "892785849701564416"


# ---------------------------------------------------------------------------
# crawl — BFS assembly
# ---------------------------------------------------------------------------

def test_crawl_single_node_produces_describegraph(mock_http, tmp_path):
    dg = fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=1, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap",
    )
    assert "nodes" in dg and "edges" in dg
    assert len(dg["edges"]) == 1
    pubs = {n["pub_key"] for n in dg["nodes"]}
    # the hub and its discovered peer both appear as nodes
    assert _pub(0) in pubs and _pub(1) in pubs


def test_crawl_records_peer_alias(mock_http, tmp_path):
    dg = fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=1, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap",
    )
    aliases = {n["pub_key"]: n["alias"] for n in dg["nodes"]}
    # the counterparty's alias (from the channel `node` block) is captured
    assert aliases[_pub(1)] == "peer-one"


def test_crawl_writes_snapshot(mock_http, tmp_path):
    snap_dir = tmp_path / "snap"
    fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=1, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=snap_dir,
    )
    snaps = list(snap_dir.glob("*.json"))
    assert len(snaps) == 1
    written = json.loads(snaps[0].read_text())
    assert "nodes" in written and "edges" in written


def test_crawl_full_bfs_discovers_peers(requests_mock, tmp_path):
    # node0 -- node1 -- node2 chain; BFS should reach all three
    _register_node(requests_mock, _pub(0),
                   [_raw_channel("800001x1x0", _pub(1), 1_000_000)])
    _register_node(requests_mock, _pub(1),
                   [_raw_channel("800002x1x0", _pub(2), 2_000_000)])
    _register_node(requests_mock, _pub(2), [])

    dg = fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=100, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap",
    )
    pubs = {n["pub_key"] for n in dg["nodes"]}
    assert {_pub(0), _pub(1), _pub(2)}.issubset(pubs)
    assert len(dg["edges"]) == 2


def test_crawl_respects_max_nodes(requests_mock, tmp_path):
    _register_node(requests_mock, _pub(0),
                   [_raw_channel("800001x1x0", _pub(1), 1_000_000)])
    _register_node(requests_mock, _pub(1),
                   [_raw_channel("800002x1x0", _pub(2), 2_000_000)])
    _register_node(requests_mock, _pub(2), [])

    dg = fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=1, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap",
    )
    # only node0 was visited (fetched); node1's channel was never requested
    assert len(dg["edges"]) == 1


def test_crawl_max_nodes_ceiling_supports_full_network():
    # CRAWL_MAX_NODES must be large enough to grow toward the real LN graph
    # ceiling (~17 500 nodes in May 2026).  5 000 was the development cap; 4.1
    # raises it so resumed crawls actually grow past what they already have.
    assert config.CRAWL_MAX_NODES >= 15_000, (
        f"CRAWL_MAX_NODES={config.CRAWL_MAX_NODES} is below 15 000; "
        "a resumed crawl with 8 000+ nodes already visited would stall immediately."
    )


def test_crawl_accepts_progress_fn(requests_mock, tmp_path):
    # crawl() must accept a progress_fn callback and invoke it during the BFS.
    _register_node(requests_mock, _pub(0), [_raw_channel("c001", _pub(1), 1_000_000)])
    _register_node(requests_mock, _pub(1), [_raw_channel("c002", _pub(2), 2_000_000)])
    _register_node(requests_mock, _pub(2), [])

    calls = []
    fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=5, resume=False,
        cache_dir=tmp_path / "cache", snapshot_dir=tmp_path / "snap",
        write_snapshot=False,
        progress_fn=lambda visited, frontier, channels: calls.append((visited, frontier, channels)),
    )
    assert len(calls) > 0, "progress_fn was never called"
    assert all(isinstance(c[0], int) and c[0] >= 0 for c in calls), \
        "first arg (visited count) must be a non-negative int"


# ---------------------------------------------------------------------------
# disk cache
# ---------------------------------------------------------------------------

def test_crawl_disk_cache_avoids_repeat_http(mock_http, tmp_path):
    cache_dir = tmp_path / "cache"
    kwargs = dict(seeds=[_pub(0)], max_nodes=1, resume=False,
                  cache_dir=cache_dir, snapshot_dir=tmp_path / "snap")
    fetch_mempool.crawl(**kwargs)
    calls_after_first = mock_http.call_count
    assert calls_after_first > 0
    # second crawl reads cached responses → no new HTTP calls
    fetch_mempool.crawl(**kwargs)
    assert mock_http.call_count == calls_after_first


# ---------------------------------------------------------------------------
# resumable frontier
# ---------------------------------------------------------------------------

def test_crawl_persists_resumable_state(requests_mock, tmp_path):
    _register_node(requests_mock, _pub(0),
                   [_raw_channel("800001x1x0", _pub(1), 1_000_000)])
    _register_node(requests_mock, _pub(1),
                   [_raw_channel("800002x1x0", _pub(2), 2_000_000)])
    _register_node(requests_mock, _pub(2), [])
    cache_dir = tmp_path / "cache"

    fetch_mempool.crawl(
        seeds=[_pub(0)], max_nodes=1, resume=False,
        cache_dir=cache_dir, snapshot_dir=tmp_path / "snap",
    )
    state_path = cache_dir / fetch_mempool.STATE_FILENAME
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    # node0 visited; node1 discovered and still queued in the frontier
    assert _pub(0) in state["visited"]
    assert _pub(1) in state["frontier"]


def test_crawl_resumes_from_persisted_frontier(requests_mock, tmp_path):
    _register_node(requests_mock, _pub(0),
                   [_raw_channel("800001x1x0", _pub(1), 1_000_000)])
    _register_node(requests_mock, _pub(1),
                   [_raw_channel("800002x1x0", _pub(2), 2_000_000)])
    _register_node(requests_mock, _pub(2), [])
    cache_dir = tmp_path / "cache"
    snap_dir = tmp_path / "snap"

    # first pass: stop after one node, leaving node1 in the frontier
    fetch_mempool.crawl(seeds=[_pub(0)], max_nodes=1, resume=False,
                        cache_dir=cache_dir, snapshot_dir=snap_dir)
    # resume: continues from the saved frontier and reaches node1
    dg = fetch_mempool.crawl(seeds=[_pub(0)], max_nodes=10, resume=True,
                             cache_dir=cache_dir, snapshot_dir=snap_dir)
    pubs = {n["pub_key"] for n in dg["nodes"]}
    assert {_pub(0), _pub(1), _pub(2)}.issubset(pubs)
    assert len(dg["edges"]) == 2


# ---------------------------------------------------------------------------
# backoff on 429 / 5xx
# ---------------------------------------------------------------------------

def test_get_retries_on_429_then_succeeds(requests_mock, tmp_path):
    url = f"{config.MEMPOOL_BASE_URL}/v1/lightning/statistics/latest"
    requests_mock.get(url, [
        {"status_code": 429},
        {"status_code": 429},
        {"json": {"latest": {"node_count": 7}}, "status_code": 200},
    ])
    data = fetch_mempool._get(url, cache_dir=tmp_path)
    assert data["latest"]["node_count"] == 7


def test_get_gives_up_after_max_retries(requests_mock, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKOFF_MAX_RETRIES", 2)
    url = f"{config.MEMPOOL_BASE_URL}/v1/lightning/statistics/latest"
    requests_mock.get(url, status_code=429)
    with pytest.raises(Exception):
        fetch_mempool._get(url, cache_dir=tmp_path)


# ---------------------------------------------------------------------------
# fetch_channel_detail (Phase 3.1)
# ---------------------------------------------------------------------------

def _channel_detail_url(channel_id):
    return f"{config.MEMPOOL_BASE_URL}/v1/lightning/channels/{channel_id}"


def _raw_channel_detail(channel_id, node1_pub, node2_pub, capacity=1_000_000):
    """Minimal channel-detail response matching the real mempool API shape.

    Policy fields are embedded directly in ``node_left`` and ``node_right``
    (no separate ``policies`` array). Values are ints, not strings. The
    endpoint is addressed by the *numeric* channel id, not the short_id.
    """
    return {
        "id": channel_id,
        "short_id": "800001x1x0",
        "capacity": capacity,
        "status": 1,
        "node_left": {
            "alias": "Node0",
            "public_key": node1_pub,
            "base_fee_mtokens": 1000,
            "fee_rate": 500,
            "cltv_delta": 40,
            "min_htlc_mtokens": 1000,
            "max_htlc_mtokens": capacity * 1000,
            "is_disabled": 0,
        },
        "node_right": {
            "alias": "Node1",
            "public_key": node2_pub,
            "base_fee_mtokens": 2000,
            "fee_rate": 1000,
            "cltv_delta": 144,
            "min_htlc_mtokens": 1000,
            "max_htlc_mtokens": capacity * 1000,
            "is_disabled": 1,
        },
    }


# Numeric channel id used in offline tests (matches the real API id format).
_TEST_CHANNEL_ID = "892785849701564416"


def _dg_with_channel_id():
    """A two-node describegraph whose single edge has a numeric channel_id."""
    n1, n2 = _pub(0), _pub(1)
    return {
        "nodes": [
            {"pub_key": n1, "alias": "Node0"},
            {"pub_key": n2, "alias": "Node1"},
        ],
        "edges": [
            {
                "channel_id": _TEST_CHANNEL_ID,
                "short_id": "800001x1x0",
                "chan_point": "",
                "capacity": "1000000",
                "node1_pub": n1,
                "node2_pub": n2,
                "node1_policy": None,
                "node2_policy": None,
            }
        ],
    }


def test_fetch_channel_detail_returns_parsed_json(requests_mock, tmp_path):
    detail = _raw_channel_detail(_TEST_CHANNEL_ID, _pub(0), _pub(1))
    requests_mock.get(_channel_detail_url(_TEST_CHANNEL_ID), json=detail)
    result = fetch_mempool.fetch_channel_detail(
        _TEST_CHANNEL_ID, cache_dir=tmp_path / "cache")
    assert result["id"] == _TEST_CHANNEL_ID
    assert "node_left" in result and "node_right" in result


def test_fetch_channel_detail_caches_response(requests_mock, tmp_path):
    detail = _raw_channel_detail(_TEST_CHANNEL_ID, _pub(0), _pub(1))
    requests_mock.get(_channel_detail_url(_TEST_CHANNEL_ID), json=detail)
    cache_dir = tmp_path / "cache"
    fetch_mempool.fetch_channel_detail(_TEST_CHANNEL_ID, cache_dir=cache_dir)
    first_count = requests_mock.call_count
    fetch_mempool.fetch_channel_detail(_TEST_CHANNEL_ID, cache_dir=cache_dir)
    assert requests_mock.call_count == first_count


# ---------------------------------------------------------------------------
# _normalize_policy (Phase 3.1)
# ---------------------------------------------------------------------------

def test_normalize_policy_maps_fields_to_describegraph_schema():
    # Real API delivers ints, not strings, for msat/ppm values.
    raw = {
        "public_key": _pub(0),
        "base_fee_mtokens": 1000,
        "fee_rate": 500,
        "cltv_delta": 40,
        "min_htlc_mtokens": 1000,
        "max_htlc_mtokens": 500000000000,
        "is_disabled": 0,
    }
    p = fetch_mempool._normalize_policy(raw)
    assert p["fee_base_msat"] == "1000"
    assert p["fee_rate_milli_msat"] == "500"
    assert p["time_lock_delta"] == 40
    assert p["min_htlc"] == "1000"
    assert p["max_htlc_msat"] == "500000000000"
    assert p["disabled"] is False


def test_normalize_policy_handles_none_input():
    assert fetch_mempool._normalize_policy(None) is None


def test_normalize_policy_disabled_true():
    # Real API uses 0/1 int for is_disabled; 1 → disabled=True.
    raw = {
        "public_key": _pub(0),
        "base_fee_mtokens": 0,
        "fee_rate": 0,
        "cltv_delta": 0,
        "min_htlc_mtokens": 1,
        "max_htlc_mtokens": 1000000,
        "is_disabled": 1,
    }
    assert fetch_mempool._normalize_policy(raw)["disabled"] is True


def test_normalize_policy_absent_max_htlc_mtokens_yields_string_zero():
    # When max_htlc_mtokens is absent, max_htlc_msat must default to "0" (not
    # "") so Phase 3.2 consumers can safely call int(policy["max_htlc_msat"])
    # without raising ValueError.
    raw = {
        "public_key": _pub(0),
        "base_fee_mtokens": 1000,
        "fee_rate": 500,
        "cltv_delta": 40,
        "min_htlc_mtokens": 1000,
        # max_htlc_mtokens intentionally absent
        "is_disabled": 0,
    }
    p = fetch_mempool._normalize_policy(raw)
    assert p["max_htlc_msat"] == "0", (
        f"Expected '0' but got {p['max_htlc_msat']!r}; "
        "int() on an empty string raises ValueError"
    )
    assert int(p["max_htlc_msat"]) == 0


def test_normalize_policy_null_cltv_means_no_policy():
    # A node object for a direction with NO announced channel_update comes back
    # with null policy fields. `cltv_delta` is mandatory in a real channel_update,
    # so a null cltv_delta marks an unannounced/unknown direction: return None
    # (no policy), NOT a phantom zero-fee policy. Coercing nulls to 0 instead
    # mints {fee 0, cltv 0} edges that `require_policy` accepts and Dijkstra
    # always prefers, producing spurious 0-fee / 0-CLTV routes (the 3.13 bug).
    raw = {
        "public_key": _pub(0),
        "base_fee_mtokens": None,
        "fee_rate": None,
        "cltv_delta": None,
        "min_htlc_mtokens": None,
        "max_htlc_mtokens": None,
        "is_disabled": None,
    }
    # Must not raise (the 3.9 crash protection still holds) AND must return None.
    assert fetch_mempool._normalize_policy(raw) is None


def test_normalize_policy_zero_fee_with_real_cltv_is_kept():
    # A genuine zero-fee channel (base 0, rate 0) WITH a real cltv_delta is a
    # valid announced policy and must be kept with its zeros preserved — only a
    # null cltv_delta marks "no policy".
    raw = {
        "public_key": _pub(0),
        "base_fee_mtokens": 0,
        "fee_rate": 0,
        "cltv_delta": 40,
        "min_htlc_mtokens": 1000,
        "max_htlc_mtokens": None,
        "is_disabled": False,
    }
    p = fetch_mempool._normalize_policy(raw)
    assert p is not None
    assert p["fee_base_msat"] == "0"
    assert p["fee_rate_milli_msat"] == "0"
    assert p["time_lock_delta"] == 40
    assert p["max_htlc_msat"] == "0"  # null max_htlc still coerces to "0"
    # All string fields remain int-parseable so routing consumers don't raise.
    for field in ("fee_base_msat", "fee_rate_milli_msat", "min_htlc", "max_htlc_msat"):
        int(p[field])


# ---------------------------------------------------------------------------
# enrich_policies (Phase 3.1)
# ---------------------------------------------------------------------------

def test_enrich_policies_fills_node1_and_node2_policy(requests_mock, tmp_path):
    dg = _dg_with_channel_id()
    edge = dg["edges"][0]
    requests_mock.get(
        _channel_detail_url(edge["channel_id"]),
        json=_raw_channel_detail(edge["channel_id"], edge["node1_pub"], edge["node2_pub"]),
    )
    result = fetch_mempool.enrich_policies(dg, cache_dir=tmp_path / "cache")
    p1 = result["edges"][0]["node1_policy"]
    p2 = result["edges"][0]["node2_policy"]
    assert p1 is not None and p2 is not None
    assert p1["fee_rate_milli_msat"] == "500"
    assert p2["fee_rate_milli_msat"] == "1000"
    assert p2["disabled"] is True


def test_enrich_policies_skips_edge_without_channel_id(requests_mock, tmp_path):
    dg = _dg_with_channel_id()
    dg["edges"][0]["channel_id"] = ""
    result = fetch_mempool.enrich_policies(dg, cache_dir=tmp_path / "cache")
    assert result["edges"][0]["node1_policy"] is None


def test_enrich_policies_respects_max_enrich_zero(requests_mock, tmp_path):
    dg = _dg_with_channel_id()
    edge = dg["edges"][0]
    requests_mock.get(
        _channel_detail_url(edge["channel_id"]),
        json=_raw_channel_detail(edge["channel_id"], edge["node1_pub"], edge["node2_pub"]),
    )
    result = fetch_mempool.enrich_policies(
        dg, max_enrich=0, cache_dir=tmp_path / "cache")
    assert result["edges"][0]["node1_policy"] is None


def test_enrich_policies_writes_snapshot_when_requested(requests_mock, tmp_path):
    dg = _dg_with_channel_id()
    edge = dg["edges"][0]
    requests_mock.get(
        _channel_detail_url(edge["channel_id"]),
        json=_raw_channel_detail(edge["channel_id"], edge["node1_pub"], edge["node2_pub"]),
    )
    snap_dir = tmp_path / "snap"
    fetch_mempool.enrich_policies(
        dg, cache_dir=tmp_path / "cache",
        snapshot_dir=snap_dir, write_snapshot=True,
    )
    written = json.loads((snap_dir / "describegraph.json").read_text())
    assert written["edges"][0]["node1_policy"] is not None


def test_enrich_policies_swallows_fetch_error(requests_mock, tmp_path):
    """A 404 on a channel detail leaves that edge's policy None; no exception raised."""
    dg = _dg_with_channel_id()
    requests_mock.get(
        _channel_detail_url(dg["edges"][0]["channel_id"]), status_code=404)
    result = fetch_mempool.enrich_policies(dg, cache_dir=tmp_path / "cache")
    assert result["edges"][0]["node1_policy"] is None


# ---------------------------------------------------------------------------
# Live smoke tests (Phase 1.7) — opt-in, excluded from the default run.
# Run with: `pytest -m live` or `make smoke`. These hit the real
# mempool.space API and assert the response shape the crawler depends on, so
# schema drift (the exact failure that produced the 1.6 blocker) is caught.
# ---------------------------------------------------------------------------

# A well-known, long-lived, high-connectivity hub (ACINQ) used as the live
# crawl probe. Public mainnet node key — not a secret.
_LIVE_PROBE_PUBKEY = (
    "03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f"
)


@pytest.mark.live
def test_live_statistics_shape(tmp_path):
    stats = fetch_mempool.fetch_statistics(cache_dir=tmp_path / "cache")
    assert isinstance(stats.get("node_count"), int) and stats["node_count"] > 0
    assert isinstance(stats.get("channel_count"), int) and stats["channel_count"] > 0


@pytest.mark.live
def test_live_node_channels_shape(tmp_path):
    raws = fetch_mempool.fetch_node_channels(
        _LIVE_PROBE_PUBKEY, cache_dir=tmp_path / "cache")
    assert raws, "expected the live probe node to have open channels"
    sample = raws[0]
    # the real list shape: a single counterparty `node`, no node1/node2/policies
    assert "node1" not in sample and "policies" not in sample
    assert "public_key" in sample["node"]
    assert "id" in sample and "capacity" in sample


@pytest.mark.live
def test_live_normalize_against_real_channel(tmp_path):
    raws = fetch_mempool.fetch_node_channels(
        _LIVE_PROBE_PUBKEY, cache_dir=tmp_path / "cache")
    edge = fetch_mempool.normalize_channel(raws[0], source_pub=_LIVE_PROBE_PUBKEY)
    assert edge["node1_pub"] == _LIVE_PROBE_PUBKEY
    assert len(edge["node2_pub"]) == 66  # a real 33-byte compressed pubkey, hex
    assert edge["capacity"].isdigit()


@pytest.mark.live
def test_live_channel_detail_shape(tmp_path):
    """Validate the real /v1/lightning/channels/{channel_id} response shape.

    The endpoint uses the numeric channel id (the ``id`` field from the list
    endpoint), NOT the human-readable short_id. Policy fields are embedded in
    ``node_left`` and ``node_right`` objects (no separate ``policies`` array).
    """
    raws = fetch_mempool.fetch_node_channels(
        _LIVE_PROBE_PUBKEY, cache_dir=tmp_path / "cache")
    assert raws, "probe node has no open channels"
    channel_id = str(raws[0].get("id", ""))
    assert channel_id, "expected numeric 'id' in channel list response"

    detail = fetch_mempool.fetch_channel_detail(
        channel_id, cache_dir=tmp_path / "cache")
    # Validate the node_left / node_right shape that _normalize_policy reads.
    assert "node_left" in detail, (
        f"expected 'node_left' key; got: {list(detail.keys())}")
    assert "node_right" in detail, (
        f"expected 'node_right' key; got: {list(detail.keys())}")
    for side in (detail["node_left"], detail["node_right"]):
        assert "public_key" in side
        assert "base_fee_mtokens" in side
        assert "fee_rate" in side
        assert "cltv_delta" in side
        assert "is_disabled" in side


@pytest.mark.live
def test_live_enrich_policies_fills_one_channel(tmp_path):
    """End-to-end: crawl one seed, enrich the first channel, assert policies set."""
    raws = fetch_mempool.fetch_node_channels(
        _LIVE_PROBE_PUBKEY, cache_dir=tmp_path / "cache")
    assert raws, "probe node has no open channels"
    edge = fetch_mempool.normalize_channel(raws[0], source_pub=_LIVE_PROBE_PUBKEY)
    dg = {
        "nodes": [{"pub_key": _LIVE_PROBE_PUBKEY, "alias": "ACINQ"},
                  {"pub_key": edge["node2_pub"], "alias": "peer"}],
        "edges": [edge],
    }
    result = fetch_mempool.enrich_policies(
        dg, max_enrich=1, cache_dir=tmp_path / "cache")
    enriched = result["edges"][0]
    assert enriched["node1_policy"] is not None or enriched["node2_policy"] is not None, (
        "expected at least one policy to be filled after enrichment"
    )


@pytest.mark.live
def test_live_enrich_policies_policy_fields_are_int_compatible(tmp_path):
    """Regression for 3.9: null policy fields from the live API must be
    coerced to 0, not left as None, so int() on any field never raises.
    Enriches one real channel and asserts every policy string field is
    safely int-parseable — proving the null-coercion fix holds on live data."""
    raws = fetch_mempool.fetch_node_channels(
        _LIVE_PROBE_PUBKEY, cache_dir=tmp_path / "cache")
    assert raws, "probe node has no open channels"
    edge = fetch_mempool.normalize_channel(raws[0], source_pub=_LIVE_PROBE_PUBKEY)
    dg = {
        "nodes": [{"pub_key": _LIVE_PROBE_PUBKEY, "alias": "ACINQ"},
                  {"pub_key": edge["node2_pub"], "alias": "peer"}],
        "edges": [edge],
    }
    result = fetch_mempool.enrich_policies(
        dg, max_enrich=1, cache_dir=tmp_path / "cache")
    enriched = result["edges"][0]
    for side_key in ("node1_policy", "node2_policy"):
        policy = enriched.get(side_key)
        if policy is None:
            continue
        # All string fields must parse as integers without raising.
        for field in ("fee_base_msat", "fee_rate_milli_msat",
                      "min_htlc", "max_htlc_msat"):
            assert policy[field] != "None", (
                f"{side_key}[{field!r}] == 'None' — null was not coerced"
            )
            int(policy[field])  # raises ValueError/TypeError if null leaked
        assert isinstance(policy["time_lock_delta"], int)
        assert isinstance(policy["disabled"], bool)
