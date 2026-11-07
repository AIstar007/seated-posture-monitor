"""Atomic JSON report writer adapter."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonReportWriter:
    def write(self, destination: Path, summary: dict[str, Any]) -> None:
        path = destination.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temporary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
