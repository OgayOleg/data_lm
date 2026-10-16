"""Dataset-specific loading and cleaning.

Each dataset ships with a small, explicit cleaning contract:

* ``telco``  — coerce ``TotalCharges`` to numeric (11 blank values -> NaN),
  drop the ID column, map the target to {0, 1};
* ``bank``   — map the target to {0, 1}, replace the sentinel ``pdays=999``
  with NaN and flag never-contacted clients.

The actual missing-value imputation happens **inside** the sklearn
preprocessing pipeline (fitted on train only) — see ``src/preprocessing.py``.
"""

from __future__ import annotations

import pandas as pd

from src.config import DATASETS, RAW_DIR, DatasetSpec

TELCO_ID_COLUMN = "customerID"
BANK_PDAYS_SENTINEL = 999  # "клиент ранее не контактировали"


def load_raw(spec: DatasetSpec) -> pd.DataFrame:
    """Load the raw CSV exactly as published."""
    path = RAW_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"Файл {path} не найден. Сначала выполните: python -m src.download --dataset {spec.key}"
        )
    return pd.read_csv(path, sep=spec.sep)


def load_raw_by_key(key: str) -> pd.DataFrame:
    """Load the raw CSV for a registered dataset key ('telco' | 'bank')."""
    return load_raw(DATASETS[key])


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Telco Customer Churn frame (see module docstring)."""
    df = df.copy()
    if TELCO_ID_COLUMN in df.columns:
        df = df.drop(columns=[TELCO_ID_COLUMN])

    # TotalCharges is stored as a string; 11 customers with tenure == 0 have blanks.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # Binary target: Yes/No -> 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


def clean_bank(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the Bank Marketing frame (see module docstring)."""
    df = df.copy()

    # Binary target: yes/no -> 1/0
    df["y"] = (df["y"] == "yes").astype(int)

    # pdays == 999 means "the client was never contacted before" -> missing
    # value + a dedicated flag feature (engineered later in src/features.py).
    df["pdays"] = pd.to_numeric(
        df["pdays"].where(df["pdays"] != BANK_PDAYS_SENTINEL), errors="coerce"
    ).astype(float)
    return df


CLEANERS = {"telco": clean_telco, "bank": clean_bank}


def clean_dataset(key: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean a dataset by key; loads the raw frame when ``df`` is not given."""
    if key not in CLEANERS:
        raise ValueError(f"Неизвестный датасет: {key}. Доступные: {sorted(CLEANERS)}")
    if df is None:
        df = load_raw_by_key(key)
    return CLEANERS[key](df)


def split_features_target(df: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned frame into features ``X`` and binary target ``y``."""
    spec = DATASETS[key]
    if spec.target not in df.columns:
        raise KeyError(f"В датасете {key} нет целевой колонки '{spec.target}'")
    y = df[spec.target].astype(int)
    X = df.drop(columns=[spec.target])
    return X, y
