"""Thin entry point for the Phase E3 MAMR Method Gate.

The implementation lives in ``reproduce_phase_e3_mamr.py``; this is provided for
parity with the prompt's ``run_phase_e3_mamr.py`` name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "reproduce_phase_e3_mamr", _ROOT / "scripts" / "reproduce_phase_e3_mamr.py"
)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
main = _mod.main


if __name__ == "__main__":
    raise SystemExit(main())
