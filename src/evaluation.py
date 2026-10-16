"""Evaluation metrics and result-table helpers.

The metric suite is fixed by the course brief: Accuracy, Precision, Recall,
F1-score, ROC-AUC.  Average Precision (PR-AUC) is added as an extra metric —
it is the standard companion for imbalanced problems (both datasets here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    y_proba=None,
    prefix: str = "",
) -> dict[str, float]:
    """Compute the full metric suite for one set of predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["avg_precision"] = average_precision_score(y_true, y_proba)

    return {f"{prefix}{name}": float(value) for name, value in metrics.items()}


def confusion_matrix_df(y_true, y_pred, labels=(0, 1)) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows=true, cols=predicted)."""
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(labels)),
        index=[f"true_{lab}" for lab in labels],
        columns=[f"pred_{lab}" for lab in labels],
    )


def cv_results_to_frame(cv_results: dict, model_name: str) -> pd.DataFrame:
    """Convert sklearn ``cross_validate`` output into a tidy DataFrame.

    Columns: model, fold, accuracy, precision, recall, f1, roc_auc, avg_precision.
    """
    rows = []
    n_folds = len(cv_results["test_accuracy"])
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    for fold in range(n_folds):
        row = {"model": model_name, "fold": fold + 1}
        for name in metric_names:
            row[name] = float(cv_results[f"test_{name}"][fold])
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of each metric per model, sorted by the primary metric."""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    summary = cv_df.groupby("model")[metric_names].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.sort_values("roc_auc_mean", ascending=False).reset_index()


def metrics_table(rows: list[dict], sort_by: str = "roc_auc") -> pd.DataFrame:
    """Assemble a metrics DataFrame and rank it by ``sort_by``."""
    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return df
