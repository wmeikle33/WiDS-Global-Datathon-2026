from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump, load
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from .metrics import WiDS_metrics


def make_model(random_state: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )

    return XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.035,
        max_depth=3,
        min_child_weight=25,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_estimators=800,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
        tree_method='gpu_hist' if USE_GPU else 'hist',
        eval_metric='logloss',
    )

def train_one_horizon(
    df: pd.DataFrame,
    features: list[str],
    horizon: int,
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[CalibratedClassifierCV | None, dict[str, float] | None]:
    label_col = f"y_{horizon}"
    df_h = df[df[label_col].notna()].copy()

    if len(df_h) < 30:
        return None, None

    X = df_h[features]
    y = df_h[label_col]

    if y.nunique() < 2:
        return None, None

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    base_model = make_model(random_state=random_state)
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    pos_prob = model.predict_proba(X_val)[:, 1]
    metrics = WiDS_metrics(y_val, pos_prob)
    metrics["brier"] = brier_score_loss(y_val, pos_prob)

    return model, metrics


def train_eval_save(
    df: pd.DataFrame,
    features: list[str],
    horizons: list[int],
    model_path: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[int, dict[str, float]]:
    models_by_horizon: dict[int, CalibratedClassifierCV] = {}
    metrics_by_horizon: dict[int, dict[str, float]] = {}

    for horizon in horizons:
        model, metrics = train_one_horizon(
            df=df,
            features=features,
            horizon=horizon,
            random_state=random_state,
            test_size=test_size,
        )

        if model is None or metrics is None:
            continue

        models_by_horizon[horizon] = model
        metrics_by_horizon[horizon] = metrics

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(models_by_horizon, model_path)

    return metrics_by_horizon


def load_model(path: str):
    return load(path)
