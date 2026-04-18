from __future__ import annotations
from pathlib import Path
import pandas as pd
from joblib import dump, load
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from .features import split_features_label
from .metrics import WiDS_metrics


def build_pipeline(
    X: pd.DataFrame,
    model_name: str = "lgbm",
    random_state: int = 42,
) -> Pipeline:
    cal_models = {}
    val_score = {}
    
    model = lgb.LGBMClassifier(
        n_estimators=80,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )


    # Calibrate
    cal_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    cal_model.fit(X_df_train, y_df_train)
    cal_models[h] = cal_model
    models[h] = model

    # Validation scores using calibrated model
    y_pred = cal_model.predict_proba(X_val)[:, 1]  # ✅ use cal_model, not model
    brier = brier_score_loss(y_val, y_pred)
    val_score[h] = brier
    c_index = concordance_index(y_val, y_pred)

    X_df_test = df_test[features]
    preds = {}
    for h in horizons:
        if h in cal_models:
            preds[h] = cal_models[h].predict_proba(X_df_test)[:, 1]

    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("lgbm", classifier),
        ]
    )

def train_eval_save(
    df: pd.DataFrame,
    label: str,
    model_path: str,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict[str, float]:

    pipe = build_pipeline(X, model_name=model_name, random_state=random_state)
    metrics: dict[str, float] = {}

    stratify = y if y.nunique() <= 20 else None

    for h in horizons:
        X, y = split_features_label(df, h)
        
        df_h = df_train[df_train[f'y_{h}'].notna()].copy()
        
        if len(df_h) < 30:
            continue
    
        X = df_h[features]
        y = df_h[f'y_{h}']
        
        if y.nunique() < 2:
            continue
    
        X_df_train, X_val, y_df_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y

         pipe.fit(X_train, y_train)

        metrics = WiDS_metrics(y_val, pos_prob)
        )


    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_path)

    return metrics



def load_model(path: str):
    return load(path)
