"""Tests for cleaning and feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets import clean_bank, clean_telco, split_features_target
from src.features import (
    TELCO_ADDON_SERVICES,
    add_bank_features,
    add_telco_features,
    engineer_features,
)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def test_clean_telco_maps_target_and_coerces_totalcharges():
    df = pd.DataFrame(
        {
            "customerID": ["A", "B", "C"],
            "tenure": [1, 0, 12],
            "TotalCharges": ["100.5", " ", "200"],
            "Churn": ["Yes", "No", "Yes"],
            "gender": ["Female", "Male", "Female"],
        }
    )
    cleaned = clean_telco(df)

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [1, 0, 1]
    assert cleaned["TotalCharges"].isna().sum() == 1  # blank -> NaN
    assert cleaned.loc[0, "TotalCharges"] == pytest.approx(100.5)


def test_clean_bank_sentinel_and_target(tiny_bank_with_sentinel):
    cleaned = clean_bank(tiny_bank_with_sentinel)

    assert set(cleaned["y"].unique()) <= {0, 1}
    assert cleaned["pdays"].isna().sum() == 25  # 999 -> missing
    assert not (cleaned["pdays"] == 999).any()


def test_split_features_target(tiny_telco):
    X, y = split_features_target(tiny_telco, "telco")
    assert "Churn" not in X.columns
    assert len(X) == len(y) == 60
    assert set(y.unique()) <= {0, 1}


# ---------------------------------------------------------------------------
# Feature engineering — telco
# ---------------------------------------------------------------------------
def test_add_telco_features_num_addons(tiny_telco):
    df = tiny_telco.copy()
    for col in TELCO_ADDON_SERVICES:
        df[col] = "No"
    df.loc[0, ["OnlineSecurity", "TechSupport"]] = "Yes"

    featured = add_telco_features(df)
    assert featured.loc[0, "num_addons"] == 2
    assert (featured["num_addons"] <= len(TELCO_ADDON_SERVICES)).all()


def test_add_telco_features_tenure_groups_and_avg_charge(tiny_telco):
    df = tiny_telco.copy()
    df.loc[0, "tenure"] = 0
    df.loc[0, "TotalCharges"] = np.nan
    df.loc[1, "tenure"] = 30
    df.loc[1, "TotalCharges"] = 3000.0

    featured = add_telco_features(df)
    assert pd.isna(featured.loc[0, "avg_monthly_charge"])  # tenure == 0 -> NaN
    assert featured.loc[1, "avg_monthly_charge"] == pytest.approx(100.0)
    assert featured.loc[1, "tenure_group"] == "24-48"
    assert set(featured["tenure_group"].cat.categories) == {"0-1", "1-12", "12-24", "24-48", "48+"}


def test_add_telco_features_flags(tiny_telco):
    df = tiny_telco.copy()
    df.loc[0, ["Partner", "Dependents"]] = ["No", "No"]
    df.loc[0, "PaymentMethod"] = "Electronic check"
    df.loc[1, "PaymentMethod"] = "Mailed check"

    featured = add_telco_features(df)
    assert featured.loc[0, "has_family"] == 0
    assert featured.loc[0, "is_electronic_check"] == 1
    assert featured.loc[1, "is_electronic_check"] == 0


# ---------------------------------------------------------------------------
# Feature engineering — bank
# ---------------------------------------------------------------------------
def test_add_bank_features_drops_duration(tiny_bank):
    featured = add_bank_features(tiny_bank)
    assert "duration" not in featured.columns


def test_add_bank_features_contact_and_recency(tiny_bank):
    df = tiny_bank.copy()
    df["pdays"] = df["pdays"].astype("Float64")
    df.loc[0, "previous"] = 0
    df.loc[1, "previous"] = 2
    df.loc[1, "pdays"] = 5
    df.loc[2, "pdays"] = pd.NA  # never contacted

    featured = add_bank_features(df)
    assert featured.loc[0, "was_contacted_before"] == 0
    assert featured.loc[1, "was_contacted_before"] == 1
    assert featured.loc[1, "pdays_recency"] == "0-7d"
    assert featured.loc[2, "pdays_recency"] == "never"
    assert "campaign_log" in featured.columns
    assert "season" in featured.columns


def test_engineer_features_unknown_key_raises():
    with pytest.raises(ValueError):
        engineer_features("unknown_dataset", pd.DataFrame())
