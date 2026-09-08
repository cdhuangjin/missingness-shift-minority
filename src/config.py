"""Configuration loading and small typed helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Mirror of configs/gate_a.yaml with defaults."""

    seeds: list[int] = field(default_factory=lambda: [42, 52, 62])
    models: list[str] = field(default_factory=lambda: ["xgboost", "logistic_regression"])
    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.60, "validation": 0.20, "test": 0.20}
    )
    data: dict[str, Any] = field(default_factory=dict)
    source_missingness: dict[str, Any] = field(
        default_factory=lambda: {"mechanism": "mcar", "rate": 0.05}
    )
    target_environments: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(
        default_factory=lambda: {"max_affected": 5, "rank_method": "mutual_information"}
    )
    pipelines: dict[str, Any] = field(
        default_factory=lambda: {
            "p0_models": ["xgboost", "logistic_regression"],
            "p1_models": ["xgboost"],
            "indicator_for": "topk_numeric",
        }
    )
    thresholds: dict[str, float] = field(default_factory=lambda: {"epsilon": 1.0e-12})
    gate: dict[str, Any] = field(
        default_factory=lambda: {
            "strong_mvg": 0.10,
            "moderate_mvg": 0.05,
            "strong_recall_drop": 0.10,
            "stable_auroc_drop": 0.03,
            "slope_ratio": 1.5,
            "min_seed_consistency": 2,
            "min_datasets": 2,
        }
    )
    paths: dict[str, str] = field(default_factory=dict)
    # Phase E1/E2 additions.
    mcar_rates: list[float] = field(default_factory=lambda: [0.05, 0.10, 0.20, 0.30, 0.40])
    mar: dict[str, float] = field(default_factory=dict)
    class_conditional: dict[str, dict[str, float]] = field(default_factory=dict)
    reverse: dict[str, dict[str, float]] = field(default_factory=dict)
    block: dict[str, float] = field(default_factory=dict)
    feature_specific: dict[str, Any] = field(default_factory=dict)
    feature_set_sensitivity: dict[str, Any] = field(default_factory=dict)
    feature_set_envs: list[str] = field(default_factory=list)
    core_mechanisms: list[str] = field(default_factory=list)
    mechanism_expansion: list[str] = field(default_factory=list)
    tier3: dict[str, Any] = field(default_factory=dict)
    statistics: dict[str, Any] = field(default_factory=lambda: {"confidence_level": 0.95, "bootstrap_iterations": 2000, "fdr_alpha": 0.05})
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        cfg = cls()
        for key in (
            "seeds",
            "models",
            "split",
            "data",
            "source_missingness",
            "target_environments",
            "features",
            "pipelines",
            "thresholds",
            "gate",
            "paths",
            "mcar_rates",
            "mar",
            "class_conditional",
            "reverse",
            "block",
            "feature_specific",
            "feature_set_sensitivity",
            "feature_set_envs",
            "core_mechanisms",
            "mechanism_expansion",
            "tier3",
            "statistics",
        ):
            if key in raw:
                setattr(cfg, key, raw[key])
        cfg.project_root = path.parent.parent
        # Resolve relative paths against the project root.
        resolved: dict[str, str] = {}
        for key, val in dict(cfg.paths).items():
            p = Path(val)
            resolved[key] = str(p if p.is_absolute() else (cfg.project_root / p))
        cfg.paths = resolved
        return cfg


def resolve(cfg: Config, key: str) -> Path:
    return Path(cfg.paths[key])
