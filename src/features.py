"""Feature engineering for both datasets.

All transformers here are **stateless** and operate on a full frame
(row-wise), so they are applied identically to train and test.  Everything
that must be *fitted* (imputers, scalers, encoders) lives in the sklearn
ColumnTransformer from ``src/preprocessing.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DATASETS

# ---------------------------------------------------------------------------
# Telco
# ---------------------------------------------------------------------------
TELCO_ADDON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
TENURE_BINS = [-0.01, 1, 12, 24, 48, np.inf]
TENURE_LABELS = ["0-1", "1-12", "12-24", "24-48", "48+"]


def _yes_mask(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("yes")


def add_telco_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain features for the Telco churn dataset."""
    df = df.copy()

    # Сколько доп. услуг подключено (0..6) — сильный сигнал лояльности.
    df["num_addons"] = sum(_yes_mask(df[col]) for col in TELCO_ADDON_SERVICES).astype(int)

    # Группа срока жизни клиента (tenure в месяцах).
    df["tenure_group"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)

    # Средний месячный платёж за весь период (NaN для tenure == 0).
    tenure_safe = df["tenure"].replace(0, np.nan)
    df["avg_monthly_charge"] = df["TotalCharges"] / tenure_safe

    # Отношение текущего тарифа к среднему (рост цены = риск ухода).
    df["charge_growth"] = df["MonthlyCharges"] / df["avg_monthly_charge"]

    # Семейный статус и тип оплаты в бинарные флаги.
    df["has_family"] = (_yes_mask(df["Partner"]) | _yes_mask(df["Dependents"])).astype(int)
    df["is_electronic_check"] = (
        df["PaymentMethod"].astype(str).str.contains("Electronic check", case=False)
    ).astype(int)

    # Новые клиенты — отдельный сегмент риска.
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    return df


# ---------------------------------------------------------------------------
# Bank marketing
# ---------------------------------------------------------------------------
BANK_MONTH_SEASON = {
    "jan": "winter",
    "feb": "winter",
    "mar": "spring",
    "apr": "spring",
    "may": "spring",
    "jun": "summer",
    "jul": "summer",
    "aug": "summer",
    "sep": "autumn",
    "oct": "autumn",
    "nov": "autumn",
    "dec": "winter",
}
PDAYS_BINS = [-0.01, 7, 30, 90, np.inf]
PDAYS_LABELS = ["0-7d", "8-30d", "31-90d", "90d+"]


def add_bank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer domain features for the Bank Marketing dataset.

    Note: ``duration`` (call length) is *leaked* target information and is
    dropped here for modelling; the leakage itself is studied separately in
    the EDA notebook.
    """
    df = df.copy()

    # Контактовали ли раньше (previous > 0) — ключевой сигнал кампании.
    df["was_contacted_before"] = (df["previous"] > 0).astype(int)

    # Давность последнего контакта (NaN -> отдельная категория "never").
    df["pdays_recency"] = pd.cut(df["pdays"].astype(float), bins=PDAYS_BINS, labels=PDAYS_LABELS)
    df["pdays_recency"] = df["pdays_recency"].cat.add_categories("never").fillna("never")

    # Интенсивность текущей кампании (длинный хвост -> log1p).
    df["campaign_log"] = np.log1p(df["campaign"])

    # Сезонность звонка.
    df["season"] = df["month"].astype(str).str.lower().map(BANK_MONTH_SEASON).fillna("unknown")

    # Макроэкономический стресс: разница ключевой ставки и доверия потребителей.
    df["macro_stress"] = df["cons.conf.idx"] / df["emp.var.rate"].replace(0, np.nan)

    # Утечка: длительность звонка известна только постфактум — удаляем.
    df = df.drop(columns=["duration"])
    return df


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
FEATURE_ENGINEERING = {"telco": add_telco_features, "bank": add_bank_features}


def engineer_features(key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply the registered feature engineering for a dataset key."""
    if key not in FEATURE_ENGINEERING:
        raise ValueError(f"Неизвестный датасет: {key}. Доступные: {sorted(FEATURE_ENGINEERING)}")
    if key not in DATASETS:
        raise ValueError(f"Датасет {key} не зарегистрирован в config.DATASETS")
    return FEATURE_ENGINEERING[key](df)
