"""Model construction for Gate A."""

from __future__ import annotations

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def make_xgboost(seed: int):
    if xgb is None:  # pragma: no cover
        raise RuntimeError("xgboost is not installed.")
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def make_logistic_regression(seed: int):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=seed,
    )


def make_random_forest(seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=seed,
        class_weight=None,
    )


def make_lightgbm(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("lightgbm is not installed.") from exc
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )


def make_model(name: str, seed: int):
    if name == "xgboost":
        return make_xgboost(seed)
    if name == "logistic_regression":
        return make_logistic_regression(seed)
    if name == "random_forest":
        return make_random_forest(seed)
    if name == "lightgbm":
        return make_lightgbm(seed)
    raise ValueError(f"Unknown model: {name}")
