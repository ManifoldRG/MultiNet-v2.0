"""The solver/scorer path must not import the heavy interface stack.

Each check runs in a fresh interpreter (subprocess) because the rest of the
suite imports `interface`, which would pollute sys.modules within one process.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pulls_interface(module: str) -> bool:
    code = (
        f"import {module}, sys; "
        "hit = [m for m in sys.modules if m == 'interface' or m.startswith('interface.')]; "
        "print('IFACE' if hit else 'CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return "IFACE" in result.stdout


def test_scorer_import_is_interface_free():
    assert not _pulls_interface("scorer"), "import scorer pulled in interface"
