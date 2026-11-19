"""Look-ahead suite tests: the mutants module and the registry test helpers are importable by name."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for extra in (HERE, HERE.parent / "registry"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))
