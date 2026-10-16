"""Training, cross-validation and hyper-parameter tuning.

Every candidate model is wrapped into its own sklearn ``Pipeline``:

    Pipeline(preprocessor -> model)

so the preprocessor is **always** fitted on the training folds only and the
exact same transformer is applied to validation/test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    N_ITER_SEARCH,
    RANDOM_STATE,
    SEARCH_CV,
    TEST_SIZE,
)
from src.evaluation import compute_metrics, cv_results_to_frame
from src.models import adapt_to_imbalance, get_models, get_search_space, imbalance_ratio
from src.preprocessing import build_preprocessor, infer_column_types

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "avg_precision": "average_precision",
}


def train_test_split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split (keeps the class balance in both parts)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def make_pipeline(model, X_train: pd.DataFrame) -> Pipeline:
    """Build preprocessor + model pipeline inferred from the training frame."""
    numeric, categorical = infer_column_types(X_train)
    preprocessor = build_preprocessor(numeric, categorical)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_models_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict | None = None,
    n_folds: int = CV_FOLDS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    """Cross-validate every candidate model and score it on the test set.

    Returns
    -------
    test_metrics : DataFrame — one row per model with test-set metrics;
    cv_metrics   : DataFrame — per-fold CV metrics for every model;
    fitted       : dict[model_name -> Pipeline] fitted on the full train set.
    """
    models = models if models is not None else get_models()
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    test_rows: list[dict] = []
    cv_frames: list[pd.DataFrame] = []
    fitted: dict[str, Pipeline] = {}

    for name, model in models.items():
        if verbose:
            print(f"[cv ] {name}: {n_folds}-fold стратифицированная кросс-валидация ...")
        model = adapt_to_imbalance(model, y_train)  # fair comparison: все модели знают о дисбалансе
        pipeline = make_pipeline(model, X_train)

        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            return_train_score=False,
        )
        cv_frames.append(cv_results_to_frame(cv_result, name))

        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        row = {"model": name, **compute_metrics(y_test, y_pred, y_proba)}
        test_rows.append(row)
        if verbose:
            print(
                f"[test] {name}: roc_auc={row['roc_auc']:.4f}  "
                f"f1={row['f1']:.4f}  accuracy={row['accuracy']:.4f}"
            )

    test_metrics = pd.DataFrame(test_rows).sort_values("roc_auc", ascending=False)
    test_metrics = test_metrics.reset_index(drop=True)
    cv_metrics = pd.concat(cv_frames, ignore_index=True)
    return test_metrics, cv_metrics, fitted


def tune_best_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = N_ITER_SEARCH,
    n_folds: int = SEARCH_CV,
    verbose: bool = True,
) -> RandomizedSearchCV:
    """Randomized-search hyper-parameters for the winning model family.

    The search runs on the *whole* preprocessing+model pipeline, refitting the
    best candidate on the full training set at the end (``refit=True``).
    """
    base_models = get_models()
    if model_name not in base_models:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступные: {sorted(base_models)}")

    base_model = adapt_to_imbalance(base_models[model_name], y_train)
    pipeline = make_pipeline(base_model, X_train)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=get_search_space(model_name, scale_pos_weight=imbalance_ratio(y_train)),
        n_iter=n_iter,
        scoring=SCORING,
        refit="roc_auc",
        cv=StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=False,
    )
    if verbose:
        print(f"[tune] {model_name}: RandomizedSearchCV (n_iter={n_iter}, cv={n_folds}) ...")
    search.fit(X_train, y_train)
    if verbose:
        print(f"[tune] лучший roc_auc(cv)={search.best_score_:.4f}")
        for key, value in sorted(search.best_params_.items()):
            print(f"       {key} = {value}")
    return search


def final_test_metrics(
    search_or_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "tuned",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Score a fitted estimator on the test set."""
    estimator = getattr(search_or_pipeline, "best_estimator_", search_or_pipeline)
    y_pred = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]
    metrics = {"model": model_name, **compute_metrics(y_test, y_pred, y_proba)}
    return metrics, y_pred, y_proba


def cross_val_scores_single(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = CV_FOLDS,
) -> pd.DataFrame:
    """CV metrics for a single unfitted estimator (helper for experiments)."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_result = cross_validate(clone(estimator), X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return cv_results_to_frame(cv_result, "experiment")


def optimise_decision_threshold(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    thresholds: np.ndarray | None = None,
    n_folds: int = CV_FOLDS,
) -> tuple[float, pd.DataFrame]:
    """Choose the decision threshold that maximises F1 on out-of-fold data.

    With imbalanced targets the default 0.5 threshold is rarely optimal for
    the business goal (missing a churner/subsciber is expensive).  Thresholds
    are selected on **out-of-fold training predictions only** — the test set
    stays untouched until the final report.

    Returns ``(best_threshold, curve)`` where ``curve`` has precision/recall/F1
    for every candidate threshold (for plotting and the report).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.model_selection import cross_val_predict

    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = cross_val_predict(
        clone(estimator), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    rows = []
    for thr in thresholds:
        pred = (oof_proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": precision_score(y_train, pred, zero_division=0),
                "recall": recall_score(y_train, pred, zero_division=0),
                "f1": f1_score(y_train, pred, zero_division=0),
            }
        )
    curve = pd.DataFrame(rows)
    best_threshold = float(curve.loc[curve["f1"].idxmax(), "threshold"])
    return best_threshold, curve
