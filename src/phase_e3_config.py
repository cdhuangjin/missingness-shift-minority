"""Phase E3 MAMR Method Gate configuration loader.

This is a dedicated loader so that Phase E1/E2 ``Config`` is untouched. It reads
``configs/phase_e3_mamr.yaml``, resolves relative paths against the project root
and exposes a small typed namespace for the runner, hyperparameter gate and
Method Gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AugmentationConfig:
    enabled: bool = True
    minority_only: bool = True
    copy_fraction: float = 0.5
    corruption_rate: float = 0.15
    minority_corruption_rate: float = 0.15
    majority_corruption_rate: float = 0.05
    mechanism_mix: dict[str, float] = field(
        default_factory=lambda: {"mcar": 0.4, "feature_specific": 0.3, "block": 0.3}
    )
    copy_fraction_candidates: list[float] = field(default_factory=lambda: [0.25, 0.5])
    corruption_rate_candidates: list[float] = field(
        default_factory=lambda: [0.10, 0.15, 0.20]
    )


@dataclass
class MAMRConfig:
    topk_features: int = 5
    lambda_: float = 1.0
    lambda_candidates: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 1.0, 2.0]
    )
    selected_lambda: float | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass
class PhaseE3Config:
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    environments: list[str] = field(default_factory=list)
    split: dict[str, float] = field(
        default_factory=lambda: {"train": 0.60, "validation": 0.20, "test": 0.20}
    )
    source_missingness: dict[str, Any] = field(
        default_factory=lambda: {"mechanism": "mcar", "rate": 0.05}
    )
    features: dict[str, Any] = field(
        default_factory=lambda: {"max_affected": 5, "rank_method": "mutual_information"}
    )
    mamr: MAMRConfig = field(default_factory=MAMRConfig)
    statistics: dict[str, Any] = field(
        default_factory=lambda: {
            "bootstrap_iterations": 2000,
            "confidence_level": 0.95,
            "fdr_alpha": 0.05,
        }
    )
    gate: dict[str, Any] = field(default_factory=dict)
    high_risk_mechanisms: list[str] = field(default_factory=list)
    control_mechanisms: list[str] = field(default_factory=list)
    paths: dict[str, Path] = field(default_factory=dict)
    project_root: Path = field(default_factory=Path.cwd)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PhaseE3Config":
        path = Path(path).resolve()
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = cls(raw=raw)
        project_root = path.parent.parent
        cfg.project_root = project_root

        for key in (
            "datasets",
            "models",
            "seeds",
            "methods",
            "environments",
            "split",
            "source_missingness",
            "features",
            "statistics",
            "gate",
            "high_risk_mechanisms",
            "control_mechanisms",
        ):
            if key in raw:
                setattr(cfg, key, raw[key])

        mamr_raw = raw.get("mamr", {})
        aug_raw = mamr_raw.get("augmentation", {})
        aug = AugmentationConfig(**aug_raw)
        cfg.mamr = MAMRConfig(
            topk_features=int(mamr_raw.get("topk_features", 5)),
            lambda_=float(mamr_raw.get("lambda", 1.0)),
            lambda_candidates=list(mamr_raw.get("lambda_candidates", [0.25, 0.5, 1.0, 2.0])),
            selected_lambda=(
                float(mamr_raw["selected_lambda"])
                if mamr_raw.get("selected_lambda") is not None
                else None
            ),
            augmentation=aug,
        )

        resolved: dict[str, Path] = {}
        for key, val in dict(raw.get("paths", {})).items():
            p = Path(val)
            resolved[key] = p if p.is_absolute() else (project_root / p)
        cfg.paths = resolved
        return cfg


def selected_params(cfg: PhaseE3Config) -> dict[str, Any]:
    """Return the active MAMR hyperparameters (post hyperparameter gate)."""
    aug = cfg.mamr.augmentation
    lam = (
        cfg.mamr.selected_lambda
        if cfg.mamr.selected_lambda is not None
        else cfg.mamr.lambda_
    )
    return {
        "lambda": float(lam),
        "copy_fraction": float(aug.copy_fraction),
        "corruption_rate": float(aug.corruption_rate),
        "minority_only": bool(aug.minority_only),
        "topk_features": int(cfg.mamr.topk_features),
    }


def update_selected(cfg: PhaseE3Config, lambda_: float, copy_fraction: float, corruption_rate: float) -> None:
    cfg.mamr.selected_lambda = float(lambda_)
    cfg.mamr.augmentation.copy_fraction = float(copy_fraction)
    cfg.mamr.augmentation.corruption_rate = float(corruption_rate)
    cfg.mamr.lambda_ = float(lambda_)

