"""Shared fixtures for lngraph tests.

Synthetic graph: 5-node star + one extra edge. Node 0 is the hub.
Hand-checkable values:
  - degree centrality of node 0: (4 out-edges + 4 in-edges) in a 5-node graph
  - closeness centrality of node 0: highest (shortest paths to all others)
  - nodes 1-4 are leaves (degree 2: one in + one out to hub)
"""

# Synthetic fixtures in this file are unit-test correctness oracles only —
# hand-checkable values for centrality/routing/liquidity assertions. They must
# never be used as notebook or analysis inputs; notebooks load only
# tests/fixtures/describegraph_sample.json (real data) or data/snapshots/*.json.
SYNTHETIC_DATA_POLICY = (
    "Synthetic fixtures are unit-test correctness oracles only (hand-checkable "
    "centrality/routing/liquidity values). Never feed synthetic data to notebooks "
    "or analysis — use tests/fixtures/describegraph_sample.json (real subgraph) "
    "or data/snapshots/describegraph.json."
)
import json
import pytest
import networkx as nx
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Synthetic describegraph fixture
# ---------------------------------------------------------------------------

SYNTHETIC_NODES = [
    {"pub_key": f"node{i:02d}" + "a" * 64, "alias": f"Node{i}"}
    for i in range(5)
]

# Hub (node00) ↔ each leaf; directed edges both ways per channel
SYNTHETIC_CHANNELS = []
for leaf_idx in range(1, 5):
    short_id = f"80000{leaf_idx}x1x0"
    capacity = 1_000_000 * leaf_idx  # vary capacity for Gini tests
    node1 = SYNTHETIC_NODES[0]["pub_key"]
    node2 = SYNTHETIC_NODES[leaf_idx]["pub_key"]
    SYNTHETIC_CHANNELS.append({
        "channel_id": short_id,
        "chan_point": f"abc{leaf_idx}:0",
        "capacity": str(capacity),
        "node1_pub": node1,
        "node2_pub": node2,
        "node1_policy": {
            "fee_base_msat": "1000",
            "fee_rate_milli_msat": "500",
            "time_lock_delta": 40,
            "min_htlc": "1000",
            "max_htlc_msat": str(capacity * 1000 // 2),
            "disabled": False,
        },
        "node2_policy": {
            "fee_base_msat": "1000",
            "fee_rate_milli_msat": "500",
            "time_lock_delta": 40,
            "min_htlc": "1000",
            "max_htlc_msat": str(capacity * 1000 // 2),
            "disabled": False,
        },
    })

SYNTHETIC_DESCRIBEGRAPH = {
    "nodes": SYNTHETIC_NODES,
    "edges": SYNTHETIC_CHANNELS,
}


@pytest.fixture
def synthetic_describegraph():
    """Return a small deterministic describegraph dict (5 nodes, 4 channels)."""
    return SYNTHETIC_DESCRIBEGRAPH


@pytest.fixture
def synthetic_graph():
    """Return a networkx.MultiDiGraph built from the synthetic describegraph."""
    G = nx.MultiDiGraph()
    for n in SYNTHETIC_NODES:
        G.add_node(n["pub_key"], alias=n["alias"])
    for ch in SYNTHETIC_CHANNELS:
        cap = int(ch["capacity"])
        G.add_edge(
            ch["node1_pub"], ch["node2_pub"],
            channel_id=ch["channel_id"],
            capacity=cap,
            policy=ch["node1_policy"],
        )
        G.add_edge(
            ch["node2_pub"], ch["node1_pub"],
            channel_id=ch["channel_id"],
            capacity=cap,
            policy=ch["node2_policy"],
        )
    return G


# ---------------------------------------------------------------------------
# Mocked-HTTP fixture for fetch_mempool tests
# ---------------------------------------------------------------------------

MOCK_STATISTICS = {
    "latest": {
        "id": 1,
        "added": "2026-06-01T00:00:00.000Z",
        "channel_count": 41343,
        "node_count": 17508,
        "total_capacity": 5_200_000_000,
        "avg_capacity": 125_750,
        "avg_fee_rate": 500,
        "avg_base_fee_mtokens": "1000",
        "med_capacity": 50_000,
    }
}

# The real /v1/lightning/channels?public_key=X&status=open endpoint returns a
# BARE ARRAY of channel objects. Each object carries only the *counterparty*
# `node` (the queried pubkey X is implicit), an integer `capacity`, a `short_id`
# and `id`, a single `fee_rate`, and NO per-direction `policies` array. The hub
# (node00) is the queried node here, so the channel's `node` is the leaf peer.
MOCK_CHANNELS_PAGE = [
    {
        "status": 1,
        "closing_reason": None,
        "closing_date": None,
        "capacity": int(SYNTHETIC_CHANNELS[0]["capacity"]),
        "short_id": SYNTHETIC_CHANNELS[0]["channel_id"],
        "id": SYNTHETIC_CHANNELS[0]["channel_id"],
        "fee_rate": 500,
        "node": {
            "alias": "peer-one",
            "public_key": SYNTHETIC_CHANNELS[0]["node2_pub"],
            "channels": 5,
            "capacity": str(int(SYNTHETIC_CHANNELS[0]["capacity"]) * 3),
        },
    }
]


@pytest.fixture
def mock_http(requests_mock):
    """Patch HTTP calls to mempool.space with deterministic stub data."""
    from lngraph.config import MEMPOOL_BASE_URL
    requests_mock.get(
        f"{MEMPOOL_BASE_URL}/v1/lightning/statistics/latest",
        json=MOCK_STATISTICS,
    )
    node_pub = SYNTHETIC_NODES[0]["pub_key"]
    requests_mock.get(
        f"{MEMPOOL_BASE_URL}/v1/lightning/channels"
        f"?public_key={node_pub}&status=open&index=0",
        json=MOCK_CHANNELS_PAGE,
    )
    # `index` is an item offset; after a 1-channel page the next offset is 1,
    # which returns the empty terminator that ends pagination.
    requests_mock.get(
        f"{MEMPOOL_BASE_URL}/v1/lightning/channels"
        f"?public_key={node_pub}&status=open&index=1",
        json=[],
    )
    return requests_mock
