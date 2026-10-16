"""Shared fixtures: small synthetic frames that mirror real dataset schemas.

Tests never require the downloaded data — they stay fast and hermetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import RANDOM_STATE


@pytest.fixture()
def tiny_telco() -> pd.DataFrame:
    """A cleaned Telco-like frame (target already mapped to 0/1)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 60
    return pd.DataFrame(
        {
            "gender": rng.choice(["Female", "Male"], n),
            "SeniorCitizen": rng.integers(0, 2, n),
            "Partner": rng.choice(["Yes", "No"], n),
            "Dependents": rng.choice(["Yes", "No"], n),
            "tenure": rng.integers(0, 72, n),
            "PhoneService": rng.choice(["Yes", "No"], n),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
            "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
            "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
            "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
            "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
            "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
            "PaperlessBilling": rng.choice(["Yes", "No"], n),
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                n,
            ),
            "MonthlyCharges": rng.uniform(20, 120, n).round(2),
            "TotalCharges": rng.uniform(20, 8000, n).round(2),
            "Churn": rng.integers(0, 2, n),
        }
    )


@pytest.fixture()
def tiny_bank() -> pd.DataFrame:
    """A cleaned Bank-like frame (target already mapped to 0/1)."""
    rng = np.random.default_rng(RANDOM_STATE)
    n = 80
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "job": rng.choice(["admin.", "blue-collar", "technician", "services"], n),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["basic.4y", "basic.6y", "high.school", "university.degree"], n),
            "default": rng.choice(["no", "yes", "unknown"], n),
            "housing": rng.choice(["no", "yes", "unknown"], n),
            "loan": rng.choice(["no", "yes", "unknown"], n),
            "contact": rng.choice(["cellular", "telephone"], n),
            "month": rng.choice(["may", "jun", "jul", "aug", "oct", "nov"], n),
            "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "duration": rng.integers(30, 1200, n),
            "campaign": rng.integers(1, 10, n),
            "pdays": rng.integers(1, 40, n),
            "previous": rng.integers(0, 3, n),
            "poutcome": rng.choice(["success", "failure", "nonexistent"], n),
            "emp.var.rate": rng.uniform(-3, 1.5, n).round(3),
            "cons.price.idx": rng.uniform(92, 95, n).round(3),
            "cons.conf.idx": rng.uniform(-50, -25, n).round(1),
            "euribor3m": rng.uniform(0.6, 5.2, n).round(3),
            "nr.employed": rng.uniform(4950, 5250, n).round(1),
            "y": rng.integers(0, 2, n),
        }
    )


@pytest.fixture()
def tiny_bank_with_sentinel(tiny_bank: pd.DataFrame) -> pd.DataFrame:
    """Bank frame in 'raw' state: pdays=999 sentinel and yes/no target."""
    df = tiny_bank.copy()
    df.loc[df.index[:25], "pdays"] = 999
    df["y"] = df["y"].map({0: "no", 1: "yes"})
    return df
