"""Tests for metric computation and result helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    compute_metrics,
    confusion_matrix_df,
    cv_results_to_frame,
    metrics_table,
    summarise_cv,
)


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["avg_precision"] == 1.0


def test_compute_metrics_zero_division():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["precision"] == 0.0  # no warnings / no crash
    assert "roc_auc" not in metrics  # probabilities not provided


def test_compute_metrics_prefix():
    metrics = compute_metrics([0, 1], [0, 1], [0.2, 0.8], prefix="test_")
    assert "test_f1" in metrics


def test_confusion_matrix_df_labels():
    cm = confusion_matrix_df([0, 1, 1, 0], [0, 1, 0, 0])
    assert cm.loc["true_0", "pred_0"] == 2
    assert cm.loc["true_1", "pred_1"] == 1
    assert cm.loc["true_1", "pred_0"] == 1


def test_cv_results_to_frame_and_summary():
    cv_results = {
        "test_accuracy": np.array([0.8, 0.9]),
        "test_precision": np.array([0.7, 0.8]),
        "test_recall": np.array([0.6, 0.7]),
        "test_f1": np.array([0.65, 0.75]),
        "test_roc_auc": np.array([0.85, 0.95]),
        "test_avg_precision": np.array([0.55, 0.65]),
    }
    frame = cv_results_to_frame(cv_results, "logreg")
    assert len(frame) == 2
    assert frame["model"].nunique() == 1

    summary = summarise_cv(frame)
    assert summary.loc[0, "model"] == "logreg"
    assert summary.loc[0, "roc_auc_mean"] == pytest.approx(0.9)


def test_metrics_table_sorting():
    rows = [
        {"model": "a", "roc_auc": 0.7, "f1": 0.5},
        {"model": "b", "roc_auc": 0.9, "f1": 0.6},
    ]
    table = metrics_table(rows, sort_by="roc_auc")
    assert table.loc[0, "model"] == "b"


def test_compute_metrics_matches_sklearn():
    from sklearn.metrics import f1_score, roc_auc_score

    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    y_proba = rng.uniform(0, 1, 200)

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["f1"] == pytest.approx(f1_score(y_true, y_pred))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_proba))
