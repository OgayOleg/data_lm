"""Model zoo: three families of classifiers required by the course brief.

1. Logistic Regression  — interpretable linear baseline;
2. Random Forest        — bagged trees, robust non-linear benchmark;
3. Gradient Boosting    — XGBoost and LightGBM implementations.

Both project datasets are imbalanced, so **every** model is made
imbalance-aware in the same way (``class_weight='balanced'`` for
scikit-learn models, ``scale_pos_weight`` for XGBoost — see
``adapt_to_imbalance``).  Otherwise the model comparison would be confounded
by different decision policies rather than by different learning algorithms.
"""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


def get_models() -> dict:
    """Return the candidate models with sensible, fixed starting defaults."""
    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        # Примечание: параметр `penalty` в sklearn >= 1.8 устарел — тип
        # регуляризации задаётся через `l1_ratio` (см. get_search_space).
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }


def imbalance_ratio(y) -> float:
    """Return neg/pos class ratio (used as ``scale_pos_weight`` for XGBoost)."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg / max(n_pos, 1))


def adapt_to_imbalance(model, y):
    """Make a model imbalance-aware using the *training* class ratio.

    scikit-learn models in the zoo already carry ``class_weight='balanced'``;
    XGBoost has no such argument, so it gets an equivalent
    ``scale_pos_weight`` computed from the training labels.
    """
    if isinstance(model, XGBClassifier):
        model.set_params(scale_pos_weight=imbalance_ratio(y))
    return model


def get_search_space(model_name: str | None = None, scale_pos_weight: float | None = None) -> dict:
    """Hyper-parameter search space for the final tuning stage.

    Each model family has its own space so that only *valid* combinations are
    sampled (e.g. ``penalty='l1'`` requires the ``liblinear``/``saga`` solver).
    """
    spaces = {
        "logreg": {
            # l1_ratio=0 -> L2-регуляризация, l1_ratio=1 -> L1 (отбор признаков).
            # liblinear поддерживает оба варианта и подходит для небольших таблиц.
            "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "model__l1_ratio": [0.0, 1.0],
            "model__solver": ["liblinear"],
        },
        "random_forest": {
            "model__n_estimators": [150, 300, 500],
            "model__max_depth": [None, 8, 12, 16],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5, 0.8],
        },
        "xgboost": {
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__min_child_weight": [1, 5, 10],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.5, 1.0, 3.0],
            # Imbalance handling stays in the search (1.0 = ignore imbalance).
            "model__scale_pos_weight": [1.0, scale_pos_weight or 1.0],
        },
        "lightgbm": {
            "model__n_estimators": [200, 400, 800],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 4, 6, 8],
            "model__min_child_samples": [10, 20, 50],
            "model__subsample": [0.7, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.1, 1.0],
            "model__reg_lambda": [0.0, 0.1, 1.0],
        },
    }
    if model_name is None:
        # Legacy behaviour: LightGBM space (the usual tuning target).
        return spaces["lightgbm"]
    if model_name not in spaces:
        raise ValueError(f"Нет пространства поиска для модели: {model_name}. Доступные: {sorted(spaces)}")
    return spaces[model_name]
