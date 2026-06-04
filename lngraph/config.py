"""Runtime configuration for lngraph.

All tunable knobs in one place. Downstream modules import from here;
tests may monkeypatch individual values.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
# Repo-root-relative data dirs. The crawler caches raw HTTP responses under
# CACHE_DIR (also the home of the resumable crawl-state file) and writes the
# assembled describegraph snapshot under SNAPSHOT_DIR. Both are gitignored.

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# --------------------------------------------------------------------------
# Data source
# --------------------------------------------------------------------------

MEMPOOL_BASE_URL = "https://mempool.space/api"

# --------------------------------------------------------------------------
# Crawl parameters
# --------------------------------------------------------------------------

# Hard ceiling on the BFS frontier size for one crawl session.
# The real LN graph has ~17 500 nodes (May 2026); set to 17 500 so a resumed
# crawl can grow toward full coverage without stalling.
CRAWL_MAX_NODES = 17_500

# Maximum paginated-channel-list responses to request per pubkey.
# mempool returns ~100 channels per page; a hub with 1 000 channels
# needs up to 10 pages.
CRAWL_MAX_PAGES_PER_NODE = 20

# --------------------------------------------------------------------------
# Rate-limit / backoff
# --------------------------------------------------------------------------

# Sustained call rate budget.  mempool.space has no published limit but
# empirically tolerates ~2 req/s without triggering 429s.
RATE_LIMIT_CALLS_PER_SEC: float = 1.5

# Exponential backoff: base delay (seconds) and ceiling.
BACKOFF_BASE_DELAY: float = 2.0
BACKOFF_MAX_DELAY: float = 120.0

# Number of times to retry a 429/5xx before giving up on a single URL.
BACKOFF_MAX_RETRIES: int = 6

# --------------------------------------------------------------------------
# Analysis parameters
# --------------------------------------------------------------------------

# Maximum number of channels to enrich with per-direction fee policy in one
# enrich_policies() pass. Each enriched channel requires one HTTP request to
# /v1/lightning/channels/{short_id}; the cache de-duplicates repeat calls.
ENRICH_MAX_CHANNELS: int = 5_000

# k-pivot sampling for approximate betweenness centrality.
# Higher k → more accurate but slower.  k=500 is a reasonable default
# for a ~5 000-node subgraph; increase to 1 000+ for publication quality.
SAMPLING_K: int = 500

# --------------------------------------------------------------------------
# Bootstrap seeds (top-connectivity hubs, May 2026 snapshot)
# --------------------------------------------------------------------------
# Public keys from the mempool "top nodes by connectivity" list.
# BFS starts from these to maximize early coverage.

SEED_PUBKEYS: list[str] = [
    # ACINQ (Phoenix / eclair)
    "03864ef025fde8fb587d989186ce6a4a186895ee44a926bfc370e2c366597a3f8f",
    # Wallet of Satoshi
    "035e4ff418fc8b5554c5d9eea66396c227bd429a3251c8cbc711002ba215bfc226",
    # Bitfinex
    "033d8656219478701227199cbd6f670335c8d408a92ae88b962c49d4dc0e83e025",
    # River Financial
    "03037dc08e9ac63b82581f79b662a4d0ceca8a8ca162b1af3551595b8f2d97b70a",
    # OpenNode
    "02d4531a2f2e6079eda1ebf38f0e7e0b6d6eba7a5b3be0e4d08e75b0f6a3ab8f29",
    # CoinGate
    "0242a4ae0c5bef18048fbecf995094b74bfb0f7391418d71ed394784373f41e4f3",
    # Muun wallet relay
    "038f8f113c580048d847d6949371726653e02b928196bad310e3eda39ff61723f6",
    # Boltz exchange
    "026165850492521f4ac8abd9bd8088123446d126f648ca35e60f88177dc149ceb2",
]
