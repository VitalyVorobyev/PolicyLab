"""Temporary protocol files for testing the M4 freeze guard in both states.

The real protocol (docs/studies/gravity-atlas/protocol.md) is frozen (v1, D26), so the
`draft` behaviour of the guard is tested against a temporary file holding only a header
`Status` row; the guard reads nothing else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

FROZEN_CONFIG_SHA256: Final = "4cfc119543dd0c53d95750d24561badf1e440aea9170db76ffd165553bf3415a"
"""`config_sha256` recorded at the freeze (protocol section 1 header, decision D26)."""
FROZEN_GIT_COMMIT: Final = "90c57756d809d1bee0c238e7954c43fe147b055e"


def protocol_file(directory: Path, status: str) -> Path:
    """A minimal protocol whose header `Status` row says `status`."""
    path = directory / f"protocol-{status}.md"
    path.write_text(f"| | |\n|---|---|\n| Status | **`{status}`** (test) |\n", encoding="utf-8")
    return path
