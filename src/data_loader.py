"""Dataset loading and cataloguing.

Every dataset is an independent public UCI source. Raw files are cached under
``data/raw`` and a cleaned snapshot under ``data/processed`` so repeated runs are
stable; content hashes are recorded for the manifest.
"""

from __future__ import annotations

import hashlib
import io
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, resolve

_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _safe_col(name: str) -> str:
    """Turn an arbitrary feature name into a valid XGBoost/LightGBM identifier."""
    s = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not s or s[0].isdigit():
        s = "f_" + s
    if not s:
        s = "f"
    return s


def _md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _fetch_bytes(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req) as resp:
        raw = resp.read()
    return raw, _md5_bytes(raw)


def _read_uci(uid: int) -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo

    obj = fetch_ucirepo(id=uid)
    features = obj.data.features
    targets = obj.data.targets
    if targets is None or targets.shape[1] == 0:
        raise RuntimeError(f"UCI id {uid} returned no target column.")
    frame = features.copy()
    frame["__target__"] = targets.iloc[:, 0]
    return frame


def _read_uci_zip(url: str, csv_in_zip: str, sep: str) -> pd.DataFrame:
    raw, _ = _fetch_bytes(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        names = zf.namelist()
        if csv_in_zip in names:
            name = csv_in_zip
        else:
            cand = [n for n in names if n.endswith(csv_in_zip) or Path(n).name == csv_in_zip]
            name = cand[0] if cand else next((n for n in names if n.endswith(".csv")), None)
        if name is None:
            raise RuntimeError(f"No CSV found in zip; entries: {names}")
        with zf.open(name) as fh:
            frame = pd.read_csv(fh, sep=sep)
    return frame


def _encode_target(col: pd.Series) -> tuple[pd.Series, int, Any]:
    """Encode y so the minority class becomes 1 and the majority 0."""
    counts = col.value_counts()
    if counts.shape[0] != 2:
        raise ValueError(
            f"Target is not binary (found {counts.shape[0]} classes). "
            "Imbalanced benchmark requires a binary target."
        )
    ordered = counts.sort_values(ascending=False)
    majority = ordered.index[0]
    minority = ordered.index[1]
    mapping = {majority: 0, minority: 1}
    encoded = col.map(mapping).astype(int)
    return encoded, minority, majority


def _subset(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Deterministically subsample when fraction < 1.0."""
    if fraction >= 1.0:
        return frame
    y = frame["__target__"]
    counts = y.value_counts()
    minority = counts.idxmin()
    rng = np.random.default_rng(seed)
    keep_idx: list[int] = []
    for cls, grp in frame.groupby("__target__"):
        n_keep = max(2, int(round(grp.shape[0] * fraction)))
        n_keep = min(n_keep, grp.shape[0])
        sampled = grp.sample(n=n_keep, random_state=int(rng.integers(0, 2**31)))
        keep_idx.extend(sampled.index.tolist())
    return frame.loc[keep_idx].reset_index(drop=True)


def load_dataset(
    cfg: Config,
    ds_cfg: dict[str, Any],
    raw_dir: Path,
    processed_dir: Path,
) -> dict[str, Any]:
    """Return a canonical dataset bundle."""
    name = ds_cfg["name"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache = processed_dir / f"{name}.csv"
    if cache.exists():
        frame = pd.read_csv(cache)
    else:
        source = ds_cfg["source"]
        if source == "uci":
            frame = _read_uci(int(ds_cfg["uci_id"]))
        elif source == "uci_zip":
            zip_path = raw_dir / f"{name}.zip"
            if zip_path.exists() and zip_path.stat().st_size > 0:
                raw = zip_path.read_bytes()
            else:
                raw, _ = _fetch_bytes(ds_cfg["zip_url"])
                zip_path.write_bytes(raw)
            frame = _read_uci_zip(
                ds_cfg["zip_url"],
                ds_cfg["csv_in_zip"],
                ds_cfg.get("separator", ";"),
            )
        else:
            raise ValueError(f"Unknown dataset source: {source}")

        frame = _subset(frame, float(ds_cfg.get("subset_fraction", 1.0)), seed=42)
        frame.to_csv(cache, index=False)

    # Drop any metadata / non-predictive columns (e.g. the "year" column of the
    # Polish bankruptcy dataset).
    for col in ds_cfg.get("drop_cols", []):
        if col in frame.columns:
            frame = frame.drop(columns=[col])

    target_col = ds_cfg["target"]
    if target_col not in frame.columns and "__target__" in frame.columns:
        # UCI loader returns the target under a sentinel name.
        target_col = "__target__"
    if target_col not in frame.columns:
        raise KeyError(
            f"Target '{target_col}' not found in columns {list(frame.columns)}"
        )
    y_raw = frame[target_col]
    X = frame.drop(columns=[target_col])

    # Sanitize feature names: XGBoost/LightGBM reject names containing [], <, etc.
    clean = []
    seen: set[str] = set()
    for c in X.columns:
        base = _safe_col(c)
        col_name = base
        suffix = 1
        while col_name in seen:
            col_name = f"{base}_{suffix}"
            suffix += 1
        seen.add(col_name)
        clean.append(col_name)
    X.columns = clean

    natural_missing = int(X.isna().sum().sum())
    y, minority_label, majority_label = _encode_target(y_raw)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    counts = y.value_counts()
    imbalance = float(counts[0] / counts[1])
    source_info: dict[str, Any] = {
        k: ds_cfg[k] for k in ("source", "target", "positive_class") if k in ds_cfg
    }
    if "uci_id" in ds_cfg:
        source_info["uci_id"] = ds_cfg["uci_id"]

    return {
        "name": name,
        "X": X,
        "y": y,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "natural_missing_count": natural_missing,
        "imbalance_ratio": imbalance,
        "minority_label": minority_label,
        "majority_label": majority_label,
        "source_info": source_info,
    }


def load_all_datasets(cfg: Config, names: list[str] | None = None) -> list[dict[str, Any]]:
    raw_dir = resolve(cfg, "raw_data")
    processed_dir = resolve(cfg, "processed_data")
    out = []
    for ds_cfg in cfg.data["datasets"]:
        if names and ds_cfg["name"] not in names:
            continue
        out.append(load_dataset(cfg, ds_cfg, raw_dir, processed_dir))
    return out
