"""Shared plotting helpers.

All project figures use one consistent house style so the EDA notebooks, the
pipeline artefacts and the final report look like one product.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Вне Jupyter (пайплайн, CI, скрипты) нужен неинтерактивный бэкенд.
# Внутри ноутбука оставляем бэкенд по умолчанию, иначе plt.show()
# не отрисует графики в ячейках (inline-вывод).
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")  # headless-safe backend

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import FIGURE_DPI, FIGURES_DIR  # noqa: E402

PALETTE = {
    "primary": "#2b6cb0",
    "secondary": "#2f855a",
    "accent": "#c05621",
    "negative": "#718096",
    "positive": "#c53030",
}
MODEL_COLORS = ["#2b6cb0", "#2f855a", "#c05621", "#805ad5", "#c53030", "#2c7a7b"]

STYLE_SET_UP = False


def setup_style() -> None:
    """Apply the project-wide matplotlib/seaborn style (idempotent)."""
    global STYLE_SET_UP
    if STYLE_SET_UP:
        return
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5.2),
            "figure.dpi": 100,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    sns.set_palette([PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]])
    STYLE_SET_UP = True


def save_figure(fig: plt.Figure, name: str, figures_dir: Path | None = None) -> Path:
    """Save a figure under reports/figures (or a custom directory)."""
    setup_style()
    directory = Path(figures_dir) if figures_dir else FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_target_balance(y: pd.Series, title: str, positive_label: str = "1"):
    """Class-balance bar chart with percentages."""
    setup_style()
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["Negative (0)", f"Positive ({positive_label})"]
    colors = [PALETTE["negative"], PALETTE["positive"]]
    bars = ax.bar(labels, counts.values, color=colors, width=0.55)
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{value:,}\n({value / total:.1%})".replace(",", " "),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_title(title)
    ax.set_ylabel("Количество наблюдений")
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str, annot: bool = False):
    """Correlation heatmap for numeric columns.

    Cells without enough pairwise data (e.g. after sentinel -> NaN conversion)
    are shaded grey instead of rendering as blank white holes.
    """
    setup_style()
    numeric = df.select_dtypes(include="number")
    corr = numeric.corr(numeric_only=True)
    mask = corr.isna()

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    ax.set_facecolor("#e6e6e6")  # цвет для пар без данных
    sns.heatmap(
        corr,
        ax=ax,
        cmap=sns.diverging_palette(220, 10, as_cmap=True),
        center=0,
        square=True,
        linewidths=0.4,
        annot=annot,
        fmt=".2f",
        mask=mask,
        cbar_kws={"shrink": 0.8},
    )
    if mask.to_numpy().any():
        ax.set_title(title + "\n(серые ячейки — недостаточно парных наблюдений)")
    else:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc_curves(roc_data: list[dict], title: str):
    """Overlay ROC curves. roc_data items: {name, fpr, tpr, auc}."""
    from sklearn.metrics import auc  # local import keeps module import light

    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for idx, item in enumerate(roc_data):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(
            item["fpr"],
            item["tpr"],
            color=color,
            lw=2,
            label=f"{item['name']} (AUC = {item.get('auc', auc(item['fpr'], item['tpr'])):.3f})",
        )
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color=PALETTE["negative"], lw=1.2, label="Случайный классификатор"
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    return fig


def plot_pr_curves(pr_data: list[dict], title: str):
    """Overlay Precision-Recall curves. pr_data items: {name, precision, recall, ap}."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for idx, item in enumerate(pr_data):
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]
        ax.plot(
            item["recall"],
            item["precision"],
            color=color,
            lw=2,
            label=f"{item['name']} (AP = {item['ap']:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(matrices: list[dict], title: str, ncols: int = 2):
    """Grid of annotated confusion matrices. Items: {name, cm (ndarray)}."""
    setup_style()
    n = len(matrices)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for idx, item in enumerate(matrices):
        cm = np.asarray(item["cm"])
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["0 (pred)", "1 (pred)"],
            yticklabels=["0 (true)", "1 (true)"],
            ax=ax,
            annot_kws={"size": 12},
        )
        ax.set_title(item["name"])
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
    fig.suptitle(title, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_metric_comparison(df: pd.DataFrame, metric_col: str, title: str, ylabel: str | None = None):
    """Horizontal bar chart comparing models on one metric."""
    setup_style()
    data = df.sort_values(metric_col, ascending=True)
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(data) + 1.8))
    colors = MODEL_COLORS[: len(data)]
    bars = ax.barh(data["model"], data[metric_col], color=colors, height=0.6)
    for bar, value in zip(bars, data[metric_col]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, min(1.05, data[metric_col].max() * 1.12))
    ax.set_xlabel(ylabel or metric_col)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_feature_importance(names, values, title: str, top_n: int = 20, xlabel: str = "Важность"):
    """Horizontal bar chart of feature importances (top-N)."""
    setup_style()
    importance = pd.Series(np.asarray(values), index=np.asarray(names))
    importance = importance.sort_values(ascending=False).head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(8.6, 0.45 * len(importance) + 1.8))
    ax.barh(importance.index, importance.values, color=PALETTE["primary"], height=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig


def plot_threshold_curves(curve: pd.DataFrame, best_threshold: float, title: str):
    """Precision / Recall / F1 as functions of the decision threshold."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(curve["threshold"], curve["precision"], label="Precision", lw=2, color=PALETTE["primary"])
    ax.plot(curve["threshold"], curve["recall"], label="Recall", lw=2, color=PALETTE["accent"])
    ax.plot(curve["threshold"], curve["f1"], label="F1", lw=2.4, color=PALETTE["secondary"])
    ax.axvline(0.5, ls=":", color="gray", lw=1.4, label="порог по умолчанию 0.5")
    ax.axvline(
        best_threshold,
        ls="--",
        color=PALETTE["positive"],
        lw=1.6,
        label=f"оптимальный порог = {best_threshold:.2f}",
    )
    ax.set_xlabel("Порог вероятности")
    ax.set_ylabel("Значение метрики")
    ax.set_title(title)
    ax.set_xlim(curve["threshold"].min(), curve["threshold"].max())
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


def plot_cv_boxplot(cv_df: pd.DataFrame, metric_col: str, title: str):
    """Distribution of a CV metric across folds for every model."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    order = cv_df.groupby("model")[metric_col].median().sort_values(ascending=False).index.tolist()
    sns.boxplot(
        data=cv_df,
        x="model",
        y=metric_col,
        order=order,
        hue="model",
        palette=MODEL_COLORS[: len(order)],
        ax=ax,
        width=0.55,
        legend=False,
    )
    sns.stripplot(
        data=cv_df,
        x="model",
        y=metric_col,
        order=order,
        color="black",
        size=4,
        alpha=0.7,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Модель")
    ax.set_ylabel(metric_col)
    fig.tight_layout()
    return fig
