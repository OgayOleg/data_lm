"""Central project configuration: paths, seeds and dataset registry.

All stages of the ML pipeline (download -> preprocessing -> training ->
evaluation) read their settings from this module, which makes the whole
project reproducible from a single place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Experiment settings (fixed seeds -> reproducible results)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_ITER_SEARCH = 12  # iterations for RandomizedSearchCV
SEARCH_CV = 3  # inner CV folds for hyper-parameter search

PRIMARY_METRIC = "roc_auc"  # model-selection metric
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]

FIGURE_DPI = 130


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Description of one binary-classification dataset."""

    key: str
    title: str
    filename: str
    target: str
    positive_label: object
    n_rows: int
    n_features_raw: int
    kaggle_url: str
    source_url: str
    source_sha256: str
    sep: str = ","
    task_summary: str = ""
    notes: tuple = field(default_factory=tuple)


DATASETS: dict[str, DatasetSpec] = {
    "telco": DatasetSpec(
        key="telco",
        title="Telco Customer Churn",
        filename="telco_customer_churn.csv",
        target="Churn",
        positive_label="Yes",
        n_rows=7043,
        n_features_raw=20,
        kaggle_url="https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
        # Public mirror of the original IBM / Kaggle CSV
        # (pinned to a git commit so the download is reproducible).
        source_url=(
            "https://api.github.com/repos/deaneeth/telco-churn-mlops-pipeline/"
            "contents/data/raw/Telco-Customer-Churn.csv"
            "?ref=42ccd979f1fca8084acf8f8dbc7ac8006077b175"
        ),
        source_sha256=("16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"),
        task_summary=(
            "Прогноз оттока клиентов телеком-компании: уйдёт ли клиент " "в ближайшее время (Churn = Yes/No)."
        ),
        notes=(
            "11 пустых значений TotalCharges (клиенты с tenure = 0).",
            "Дисбаланс классов ~73% / 27%.",
        ),
    ),
    "bank": DatasetSpec(
        key="bank",
        title="Bank Marketing (bank-additional-full)",
        filename="bank_marketing.csv",
        target="y",
        positive_label="yes",
        n_rows=41188,
        n_features_raw=20,
        kaggle_url="https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing",
        # Public mirror of the UCI "Bank Marketing" dataset
        # (bank-additional-full.csv, 41 188 записей).
        source_url=(
            "https://api.github.com/repos/davified/learn-scikit-learn/"
            "contents/data/bank-marketing-data/bank-additional-full.csv"
            "?ref=b397836fab1139bb89ddb1246e9807e9d4e73a2c"
        ),
        source_sha256=("74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"),
        sep=";",
        task_summary=(
            "Прогноз подписки клиента банка на срочный депозит "
            "по итогам маркетинговой кампании (y = yes/no)."
        ),
        notes=(
            "Явная утечка целевой переменной: признак duration "
            "(длительность звонка) известен только после контакта.",
            "Дисбаланс классов ~89% / 11%.",
            "Категория 'unknown' в нескольких признаках.",
        ),
    ),
}


def ensure_directories() -> None:
    """Create all output directories if they do not exist yet."""
    for path in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
