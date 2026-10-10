"""Statistical (classification) metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(g: pd.DataFrame) -> pd.Series:
    y, p, s = g["y_true"], g["pred"], g["score"]
    auc = roc_auc_score(y, s) if s.nunique() > 1 else 0.5  # constant score carries no ranking info
    return pd.Series({
        "accuracy": accuracy_score(y, p),
        "precision": precision_score(y, p, zero_division=0),
        "recall": recall_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "roc_auc": auc,
        "n": len(g),
    })


def metrics_table(preds: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    keys = ["model"] + (by or [])
    return preds.groupby(keys).apply(classification_metrics, include_groups=False)
