from __future__ import annotations

from typing import Any

from sklearn.metrics import log_loss, roc_auc_score


def WiDS_metrics(y_true, y_prob) -> dict[str, float]:
    metrics = {"log_loss": float(log_loss(y_true, y_prob))}
    metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def metric_score(metric_fn: Any, y_true, y_pred):
    return metric_fn(y_true, y_pred)
