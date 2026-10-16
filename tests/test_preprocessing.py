"""Tests for the preprocessing pipeline.

The key property verified here is the course-defence requirement: the
transformer is fitted on train only and the *same* transformer is applied to
test data (no refit, no leakage, unknown categories tolerated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import build_preprocessor, infer_column_types, prepare_matrix
from src.training import make_pipeline, train_test_split_stratified


def test_infer_column_types(tiny_telco):
    numeric, categorical = infer_column_types(tiny_telco.drop(columns=["Churn"]))
    assert "MonthlyCharges" in numeric and "tenure" in numeric
    assert "Contract" in categorical and "gender" in categorical
    assert set(numeric) | set(categorical) == set(tiny_telco.columns) - {"Churn"}


def test_preprocessor_fitted_on_train_only(tiny_bank):
    """Transform fitted on train must work on test without any refit."""
    X = tiny_bank.drop(columns=["y"])
    y = tiny_bank["y"]
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    numeric, categorical = infer_column_types(X_train)
    preprocessor = build_preprocessor(numeric, categorical)

    train_matrix = preprocessor.fit_transform(X_train)
    test_matrix = preprocessor.transform(X_test)

    assert train_matrix.shape[1] == test_matrix.shape[1]
    assert train_matrix.shape[0] == len(X_train)
    assert not np.isnan(train_matrix).any()
    assert not np.isnan(test_matrix).any()


def test_preprocessor_handles_unknown_categories():
    """New categories at inference time must not break the transform."""
    train = pd.DataFrame({"cat": ["a", "b", "a"], "num": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"cat": ["a", "UNSEEN"], "num": [4.0, 5.0]})

    preprocessor = build_preprocessor(["num"], ["cat"])
    preprocessor.fit_transform(train)
    out = preprocessor.transform(test)

    assert out.shape[0] == 2
    assert not np.isnan(out).any()


def test_imputation_uses_train_statistics():
    """Median imputation must use the *train* median, not the test one."""
    train = pd.DataFrame({"num": [1.0, 2.0, 3.0, np.nan]})
    test = pd.DataFrame({"num": [np.nan, 100.0]})

    preprocessor = build_preprocessor(["num"], [], scale_numeric=False)
    preprocessor.fit_transform(train)
    out = preprocessor.transform(test)

    train_median = 2.0  # median of [1, 2, 3]
    assert out[0, 0] == pytest.approx(train_median)


def test_make_pipeline_end_to_end(tiny_telco):
    """Full preprocessing + model pipeline trains and predicts."""
    X = tiny_telco.drop(columns=["Churn"])
    y = tiny_telco["Churn"]
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)

    from sklearn.linear_model import LogisticRegression

    pipeline = make_pipeline(LogisticRegression(max_iter=500, random_state=0), X_train)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)

    assert len(preds) == len(X_test)
    assert proba.shape == (len(X_test), 2)
    assert set(np.unique(preds)) <= {0, 1}


def test_prepare_matrix_drops_bank_leakage(tiny_bank):
    X = tiny_bank.drop(columns=["y"])
    assert "duration" in X.columns
    cleaned = prepare_matrix(X, "bank")
    assert "duration" not in cleaned.columns

    telco_like = prepare_matrix(X, "telco")  # no leakage columns for telco
    assert "duration" in telco_like.columns


def test_preprocessor_raises_on_missing_columns():
    preprocessor = build_preprocessor(["num"], ["cat"])
    with pytest.raises(Exception):
        preprocessor.fit(pd.DataFrame({"num": [1.0]}))
