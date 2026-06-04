"""Tests for Phase 0 scaffold: package structure and config values."""
import importlib
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def test_lngraph_importable():
    mod = importlib.import_module("lngraph")
    assert mod is not None


def test_config_has_required_attrs():
    from lngraph import config
    assert hasattr(config, "MEMPOOL_BASE_URL")
    assert hasattr(config, "CRAWL_MAX_NODES")
    assert hasattr(config, "RATE_LIMIT_CALLS_PER_SEC")
    assert hasattr(config, "BACKOFF_MAX_RETRIES")
    assert hasattr(config, "SAMPLING_K")
    assert hasattr(config, "SEED_PUBKEYS")


def test_config_mempool_url_is_https():
    from lngraph import config
    assert config.MEMPOOL_BASE_URL.startswith("https://")


def test_config_seed_pubkeys_nonempty():
    from lngraph import config
    assert len(config.SEED_PUBKEYS) > 0


def test_config_sampling_k_positive():
    from lngraph import config
    assert config.SAMPLING_K > 0


def test_config_crawl_max_nodes_positive():
    from lngraph import config
    assert config.CRAWL_MAX_NODES > 0


# ---------------------------------------------------------------------------
# README (Phase 4.4)
# ---------------------------------------------------------------------------

def test_readme_exists():
    assert (_PROJECT_ROOT / "README.md").exists(), \
        "README.md must exist at the project root (spec 4.4)"


def test_readme_has_required_content():
    text = (_PROJECT_ROOT / "README.md").read_text()
    text_lower = text.lower()
    assert "quickstart" in text_lower or "quick start" in text_lower or "make install" in text_lower, \
        "README.md must include a quickstart / install section"
    assert "mempool" in text_lower, \
        "README.md must describe the mempool.space data source"
    assert "lightning" in text_lower, \
        "README.md must mention Lightning Network context"
