"""Preprocessing for modelling.

The central guarantee of the project: **everything that is learned from data**
(median/most-frequent imputation, scaling, one-hot encoding) is encapsulated
in a single sklearn ``ColumnTransformer`` that is *fitted on the training
data only* and then applied unchanged to validation/test data.  This is how
train/test consistency of the transformations is enforced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_column_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature names into numeric and categorical lists.

    Integer columns with few distinct values are treated as categorical only
    when their dtype is object/category/bool; numeric dtypes stay numeric so
    that ordinal information (age, tenure, ...) is preserved.
    """
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the reusable preprocessing transformer.

    * numeric: median imputation (+ optional standardisation);
    * categorical: most-frequent imputation + one-hot encoding with
      ``handle_unknown='ignore'`` (new categories at inference time do not
      break the pipeline).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """Readable feature names after transformation (one-hot expanded)."""
    return np.asarray(preprocessor.get_feature_names_out())


def prepare_matrix(
    X: pd.DataFrame,
    key: str,
    drop_leakage: bool = True,
) -> pd.DataFrame:
    """Light frame-level preparation shared by train and inference.

    Removes obvious leakage columns that must never reach the model
    (``duration`` in the bank dataset).  Row-wise feature engineering is
    applied beforehand in ``src/features.py``.
    """
    X = X.copy()
    leakage_columns = {"duration"} if (drop_leakage and key == "bank") else set()
    existing = [c for c in leakage_columns if c in X.columns]
    if existing:
        X = X.drop(columns=existing)
    return X
