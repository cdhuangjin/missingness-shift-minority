"""Run-key based resume, checkpointing and failure recording."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResumeManager:
    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.results_dir / "progress.json"
        self.failed_path = self.results_dir / "failed_runs.csv"

    def _key_path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode()).hexdigest()
        return self.raw_dir / f"{digest}.json"

    def is_complete(self, key: str) -> bool:
        return self._key_path(key).exists()

    def save(self, key: str, data: dict[str, Any]) -> None:
        path = self._key_path(key)
        payload = {"run_key": key, "saved_at": datetime.now().isoformat(), "data": data}
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, key: str) -> dict[str, Any] | None:
        path = self._key_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def record_failure(self, key: str, error: str) -> None:
        row = {"run_key": key, "error": error, "timestamp": datetime.now().isoformat()}
        import csv
        import os

        write_header = not self.failed_path.exists()
        with open(self.failed_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def save_progress(
        self,
        completed: int,
        total: int,
        failed: int,
        last_key: str | None,
    ) -> None:
        payload = {
            "completed_runs": completed,
            "total_runs": total,
            "failed_runs": failed,
            "last_completed_key": last_key,
            "timestamp": datetime.now().isoformat(),
        }
        self.progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_progress(self) -> dict[str, Any]:
        if self.progress_path.exists():
            return json.loads(self.progress_path.read_text())
        return {}

    def completed_count(self) -> int:
        return sum(1 for _ in self.raw_dir.glob("*.json"))
