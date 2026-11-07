#!/usr/bin/env python3
"""Thin CLI launcher kept at the repository root for the assignment command."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution without requiring an editable package installation.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from seated_posture_monitor.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
