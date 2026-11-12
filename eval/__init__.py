# Make ``nexus_decision`` importable from ``src/`` even when the editable-install .pth file is
# skipped (macOS can flag it hidden, and Python then ignores it).
import sys as _sys
from pathlib import Path as _Path

_SRC = str(_Path(__file__).resolve().parents[1] / "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)
