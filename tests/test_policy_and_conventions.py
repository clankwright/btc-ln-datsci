"""Tests for project conventions: test markers and synthetic-data policy (4.7 + 4.8)."""
import importlib.util
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
DOCS_DIR = TESTS_DIR.parent / "docs"


def _load_file_module(name: str, path: Path):
    """Load a Python module from a file path without requiring a package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_notebook_raises_has_notebook_marker():
    """test_notebook_raises_on_missing_snapshot must carry @pytest.mark.notebook (4.7).

    Without this marker the test spawns Jupyter kernels in the default suite,
    contradicting pytest.ini's addopts exclusion of notebook-marked tests.
    """
    mod = _load_file_module("test_real_data_only", TESTS_DIR / "test_real_data_only.py")
    fn = getattr(mod, "test_notebook_raises_on_missing_snapshot", None)
    assert fn is not None, "test_notebook_raises_on_missing_snapshot not found in test_real_data_only.py"
    markers = {m.name for m in getattr(fn, "pytestmark", [])}
    assert "notebook" in markers, (
        "test_notebook_raises_on_missing_snapshot is missing @pytest.mark.notebook — "
        "it runs in the default suite and spawns Jupyter kernels (spec 4.7)"
    )


def test_conftest_has_synthetic_data_policy():
    """tests/conftest.py must expose SYNTHETIC_DATA_POLICY to document the oracle-only rule (4.8)."""
    cf = _load_file_module("conftest_module", TESTS_DIR / "conftest.py")
    assert hasattr(cf, "SYNTHETIC_DATA_POLICY"), (
        "tests/conftest.py is missing SYNTHETIC_DATA_POLICY constant — "
        "synthetic fixtures must be documented as unit-test oracles only (spec 4.8)"
    )
    policy = cf.SYNTHETIC_DATA_POLICY
    assert isinstance(policy, str) and len(policy) > 20, (
        "SYNTHETIC_DATA_POLICY must be a non-empty descriptive string"
    )


def test_spec_tests_line_mentions_synthetic_oracles():
    """SPEC.md architecture/stack 'Tests:' line must state synthetic fixtures are unit-test oracles (4.8)."""
    spec_text = (DOCS_DIR / "SPEC.md").read_text()
    # Find the Tests: bullet in the architecture section
    tests_line = next(
        (line for line in spec_text.splitlines() if line.strip().startswith("- Tests:")),
        None,
    )
    assert tests_line is not None, "SPEC.md missing '- Tests:' line in architecture section"
    assert "unit" in tests_line.lower() or "oracle" in tests_line.lower(), (
        "SPEC.md Tests line must mention 'unit' or 'oracle' to distinguish synthetic fixtures "
        "from notebook inputs (spec 4.8). Current line:\n  " + tests_line
    )
    assert "notebook" in tests_line.lower(), (
        "SPEC.md Tests line must mention notebook execution tests and real fixture (spec 4.8). "
        "Current line:\n  " + tests_line
    )
