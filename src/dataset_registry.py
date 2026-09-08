"""Frozen dataset registry for Phase E1/E2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .data_loader import load_dataset


def load_registry(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve()
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    data["_path"] = p
    return data


def registry_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data["datasets"])


def freeze_hash(data: dict[str, Any]) -> str:
    raw = Path(data["_path"]).read_bytes()
    return hashlib.sha1(raw).hexdigest()


def list_hash(data: dict[str, Any]) -> str:
    canonical = [
        {
            "name": d["name"],
            "source": d.get("source"),
            "uci_id": d.get("uci_id"),
            "zip_url": d.get("zip_url"),
            "drop_cols": d.get("drop_cols", []),
        }
        for d in registry_list(data)
    ]
    return hashlib.sha1(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def load_all(cfg: Config, data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_dir = Path(cfg.paths["raw_data"])
    processed_dir = Path(cfg.paths["processed_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return [
        load_dataset(cfg, ds_cfg, raw_dir, processed_dir)
        for ds_cfg in registry_list(data)
    ]
