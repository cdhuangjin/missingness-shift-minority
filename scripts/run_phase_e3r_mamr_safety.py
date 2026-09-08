"""Thin wrapper for the Phase E3R MAMR Safety Repair Gate reproduction."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

sys.argv = [sys.argv[0], *sys.argv[1:]]
from scripts.reproduce_phase_e3r_mamr_safety import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
