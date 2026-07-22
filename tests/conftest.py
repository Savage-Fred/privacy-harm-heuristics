import asyncio
import inspect
import sys
from pathlib import Path

# Ensure project root (containing src/privacy_harm_heuristics) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():  # pragma: no cover
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Lightweight asyncio mark shim (since pytest-asyncio not installed)
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(items):  # type: ignore[override]
    """Wrap async test functions marked with @pytest.mark.asyncio so they run via event loop.

    This avoids introducing the pytest-asyncio dependency while preserving existing test style.
    """
    for item in items:
        # item.obj is the original test function; item.function is read-only in pytest>=8
        fn = getattr(item, "obj", None)
        if fn and "asyncio" in item.keywords and inspect.iscoroutinefunction(fn):
            original = fn

            def _sync_wrapper(*args, __orig=original, **kwargs):  # type: ignore
                return asyncio.run(__orig(*args, **kwargs))

            # Replace the underlying object
            item.obj = _sync_wrapper  # type: ignore[attr-defined]
