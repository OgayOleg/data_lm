"""End-to-end ML pipeline: one command reproduces every project artefact.

Stages (per dataset):
    1. load raw CSV          (python -m src.download)
    2. clean + feature eng.
    3. stratified train/test split
    4. train 4 candidate models with 5-fold CV
    5. evaluate on the held-out test set
    6. tune the best model family (RandomizedSearchCV)
    7. figures + metrics + saved model

Usage::

    python -m src.pipeline                 # both datasets
    python -m src.pipeline --dataset telco
    python -m src.pipeline --skip-tuning   # fast mode
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from src.config import (
    DATASETS,
    METRICS_DIR,
    MODELS_DIR,
    PRIMARY_METRIC,
    RANDOM_STATE,
    ensure_directories,
)
from src.datasets import clean_dataset, load_raw_by_key, split_features_target
from src.evaluation import compute_metrics, confusion_matrix_df
from src.features import engineer_features
from src.models import get_models
from src.plotting import (
    plot_confusion_matrices,
    plot_correlation_heatmap,
    plot_cv_boxplot,
    plot_feature_importance,
    plot_metric_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_target_balance,
    plot_threshold_curves,
    save_figure,
    setup_style,
)
from src.preprocessing import prepare_matrix
from src.training import (
    evaluate_models_with_cv,
    final_test_metrics,
    optimise_decision_threshold,
    train_test_split_stratified,
    tune_best_model,
)

setup_style()


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(key: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load -> clean -> engineer features -> model matrix (X, y)."""
    raw = load_raw_by_key(key)
    cleaned = clean_dataset(key, raw)
    featured = engineer_features(key, cleaned)
    X, y = split_features_target(featured, key)
    X = prepare_matrix(X, key)
    return X, y


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def compute_permutation_importance(
    estimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    scoring: str = PRIMARY_METRIC,
) -> pd.DataFrame:
    """Model-agnostic feature importance on the test set.

    Works for every model family (including Logistic Regression, which has no
    ``feature_importances_`` attribute). Returns a ranked DataFrame with the
    mean/std decrease of the scoring metric when a feature is shuffled.

    ``scoring`` defaults to ROC-AUC and **must not** be accuracy: models trained
    with balanced class weights over-predict the positive class (42% positives
    on Telco against a 26.5% base rate), so shuffling a feature can *raise*
    accuracy and produce meaningless negative importances.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        estimator,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring=scoring,
    )
    frame = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def make_dataset_figures(
    key: str,
    X: pd.DataFrame,
    y: pd.Series,
    test_metrics: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    fitted: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    importance: pd.DataFrame | None = None,
) -> list[Path]:
    """Generate the standard figure suite for one dataset."""
    spec = DATASETS[key]
    prefix = key
    paths: list[Path] = []

    # 1. Class balance
    fig = plot_target_balance(y, f"{spec.title}: баланс классов (target = {spec.target})")
    paths.append(save_figure(fig, f"{prefix}_target_balance.png"))

    # 2. Correlation heatmap of engineered numeric features
    fig = plot_correlation_heatmap(X, f"{spec.title}: корреляция числовых признаков")
    paths.append(save_figure(fig, f"{prefix}_correlation_heatmap.png"))

    # 3. ROC curves on test
    roc_data = []
    for name, model in fitted.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data.append({"name": name, "fpr": fpr, "tpr": tpr, "auc": float(np.trapezoid(tpr, fpr))})
    fig = plot_roc_curves(roc_data, f"{spec.title}: ROC-кривые (test)")
    paths.append(save_figure(fig, f"{prefix}_roc_curves.png"))

    # 4. PR curves on test
    pr_data = []
    for name, model in fitted.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_data.append(
            {
                "name": name,
                "precision": precision,
                "recall": recall,
                "ap": float(average_precision_score(y_test, y_proba)),
            }
        )
    fig = plot_pr_curves(pr_data, f"{spec.title}: Precision-Recall кривые (test)")
    paths.append(save_figure(fig, f"{prefix}_pr_curves.png"))

    # 5. Confusion matrices of all models
    matrices = []
    for name, model in fitted.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix_df(y_test, y_pred).values
        matrices.append({"name": name, "cm": cm})
    fig = plot_confusion_matrices(matrices, f"{spec.title}: матрицы ошибок (test)")
    paths.append(save_figure(fig, f"{prefix}_confusion_matrices.png"))

    # 6. CV boxplot (roc_auc)
    fig = plot_cv_boxplot(cv_metrics, "roc_auc", f"{spec.title}: ROC-AUC по фолдам кросс-валидации")
    paths.append(save_figure(fig, f"{prefix}_cv_roc_auc_boxplot.png"))

    # 7. Test metrics comparison
    fig = plot_metric_comparison(
        test_metrics,
        PRIMARY_METRIC,
        f"{spec.title}: сравнение моделей по ROC-AUC (test)",
        ylabel="ROC-AUC (test)",
    )
    paths.append(save_figure(fig, f"{prefix}_model_comparison.png"))

    # 8. Feature importance (permutation) for the final model
    final_model_label = test_metrics.iloc[0]["model"]
    if importance is not None and not importance.empty:
        fig = plot_feature_importance(
            importance["feature"],
            importance["importance_mean"],
            f"{spec.title}: permutation importance (модель — {final_model_label})",
            xlabel=f"Среднее падение {PRIMARY_METRIC.upper()} при перемешивании признака",
        )
        paths.append(save_figure(fig, f"{prefix}_feature_importance.png"))

    return paths


# ---------------------------------------------------------------------------
# Duration-leakage experiment (bank only)
# ---------------------------------------------------------------------------
def run_duration_leakage_experiment() -> pd.DataFrame:
    """Hypothesis check: how much does the leaked 'duration' feature inflate metrics?"""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    from src.training import make_pipeline

    print("[exp ] bank: эксперимент с утечкой признака duration ...")
    raw = load_raw_by_key("bank")
    cleaned = clean_dataset("bank", raw)
    featured = engineer_features("bank", cleaned)
    X_base, y = split_features_target(featured, "bank")
    X_base = prepare_matrix(X_base, "bank", drop_leakage=True)

    # Re-add the leaked feature (it was removed inside prepare_matrix).
    X_leak = X_base.copy()
    X_leak["duration"] = pd.to_numeric(clean_dataset("bank", raw)["duration"], errors="coerce")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for label, X in [("без duration (честная модель)", X_base), ("с duration (утечка)", X_leak)]:
        pipeline = make_pipeline(
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            X,
        )
        auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"experiment": label, "roc_auc_mean": float(auc.mean()), "roc_auc_std": float(auc.std())})
        print(f"[exp ] {label}: roc_auc = {auc.mean():.4f} ± {auc.std():.4f}")

    result = pd.DataFrame(rows)
    result.to_csv(METRICS_DIR / "bank_duration_leakage.csv", index=False)

    fig = plot_metric_comparison(
        result.rename(columns={"experiment": "model", "roc_auc_mean": "roc_auc"}),
        "roc_auc",
        "Bank Marketing: эффект утечки признака duration (LightGBM, CV=5)",
        ylabel="ROC-AUC (CV)",
    )
    save_figure(fig, "bank_duration_leakage.png")
    return result


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------
def run_dataset(key: str, skip_tuning: bool = False) -> dict:
    """Run the full pipeline for one dataset and save all artefacts."""
    spec = DATASETS[key]
    ensure_directories()
    print(f"\n{'=' * 70}\n=== Датасет: {spec.title} ({key})\n{'=' * 70}")

    X, y = prepare_dataset(key)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
    print(
        f"[split] train = {X_train.shape}, test = {X_test.shape}, "
        f"positive rate: train={y_train.mean():.3f}, test={y_test.mean():.3f}"
    )

    # --- Stage 4-5: candidates + CV + test -------------------------------
    test_metrics, cv_metrics, fitted = evaluate_models_with_cv(
        X_train, y_train, X_test, y_test, models=get_models()
    )

    # --- Stage 6: tune the winner ---------------------------------------
    best_name = test_metrics.iloc[0]["model"]
    tuned_metrics = None
    best_estimator = fitted[best_name]
    if not skip_tuning:
        search = tune_best_model(best_name, X_train, y_train)
        tuned_metrics, y_pred, y_proba = final_test_metrics(
            search, X_test, y_test, model_name=f"{best_name}_tuned"
        )
        print(
            f"[test] {tuned_metrics['model']}: "
            f"roc_auc={tuned_metrics['roc_auc']:.4f}  f1={tuned_metrics['f1']:.4f}"
        )
        best_estimator = search.best_estimator_
    else:
        y_pred = best_estimator.predict(X_test)
        y_proba = best_estimator.predict_proba(X_test)[:, 1]

    # --- Stage 7: operating threshold (F1 on out-of-fold predictions) ----
    best_threshold, threshold_curve = optimise_decision_threshold(best_estimator, X_train, y_train)
    threshold_curve.to_csv(METRICS_DIR / f"{key}_threshold_curve.csv", index=False)
    fig = plot_threshold_curves(
        threshold_curve,
        best_threshold,
        f"{spec.title}: выбор порога решения (F1 на out-of-fold)",
    )
    threshold_figure = save_figure(fig, f"{key}_threshold_curve.png")

    final_model_name = tuned_metrics["model"] if tuned_metrics else best_name
    y_pred_at_threshold = (y_proba >= best_threshold).astype(int)
    threshold_metrics = {
        "model": f"{final_model_name}@{best_threshold:.2f}",
        **compute_metrics(y_test, y_pred_at_threshold, y_proba),
    }
    f1_at_half = (tuned_metrics or compute_metrics(y_test, y_pred, y_proba))["f1"]
    print(
        f"[thr ] порог {best_threshold:.2f}: на тесте f1={threshold_metrics['f1']:.4f} "
        f"(при пороге 0.5: f1={f1_at_half:.4f})"
    )

    # --- Stage 8: artefacts ---------------------------------------------
    test_metrics.to_csv(METRICS_DIR / f"{key}_test_metrics.csv", index=False)
    cv_metrics.to_csv(METRICS_DIR / f"{key}_cv_metrics.csv", index=False)
    confusion_matrix_df(y_test, y_pred_at_threshold).to_csv(METRICS_DIR / f"{key}_confusion_matrix.csv")
    pd.DataFrame([threshold_metrics]).to_csv(METRICS_DIR / f"{key}_final_metrics.csv", index=False)

    combined = test_metrics.copy()
    if tuned_metrics is not None:
        combined = pd.concat([combined, pd.DataFrame([tuned_metrics])], ignore_index=True)
    combined = pd.concat([combined, pd.DataFrame([threshold_metrics])], ignore_index=True)
    combined.to_csv(METRICS_DIR / f"{key}_test_metrics_with_tuned.csv", index=False)

    model_path = MODELS_DIR / f"{key}_best_model.joblib"
    joblib.dump(best_estimator, model_path)

    # Permutation importance — единый способ интерпретации для всех семейств
    # моделей (включая логистическую регрессию без feature_importances_).
    try:
        importance = compute_permutation_importance(best_estimator, X_test, y_test)
        importance.insert(0, "scoring", PRIMARY_METRIC)
        importance.to_csv(METRICS_DIR / f"{key}_permutation_importance.csv", index=False)
        feature_importance = {
            row.feature: float(row.importance_mean) for row in importance.head(20).itertuples()
        }
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[warn] permutation importance не рассчитана: {exc}", file=sys.stderr)
        importance, feature_importance = None, {}

    figure_paths = make_dataset_figures(
        key, X, y, test_metrics, cv_metrics, fitted, X_test, y_test, importance=importance
    )
    figure_paths.append(threshold_figure)

    result = {
        "dataset": key,
        "title": spec.title,
        "n_rows": int(len(X)),
        "n_features_model_matrix": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "best_model": best_name,
        "final_model": final_model_name,
        "decision_threshold": best_threshold,
        "test_metrics": combined.to_dict(orient="records"),
        "top_features": feature_importance,
        "figures": [str(p.name) for p in figure_paths],
        "model_path": str(model_path.name),
    }
    with open(METRICS_DIR / f"{key}_summary.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)

    print(f"[done] {key}: артефакты сохранены в reports/ и models/")
    return result


# ---------------------------------------------------------------------------
# Cross-dataset summary
# ---------------------------------------------------------------------------
def write_summary_md(results: list[dict]) -> Path:
    """Write a compact bilingual-friendly Markdown summary of all runs."""
    lines = [
        "# Сводка результатов пайплайна",
        "",
        "Сгенерировано автоматически: `python -m src.pipeline`.",
        "",
    ]
    for res in results:
        lines += [
            f"## {res['title']} (`{res['dataset']}`)",
            "",
            f"- Строк: **{res['n_rows']:,}** (признаков в матрице: {res['n_features_model_matrix']})".replace(
                ",", " "
            ),
            f"- Положительный класс: **{res['positive_rate']:.1%}**",
            f"- Лучшая модель-кандидат: **{res['best_model']}**",
            f"- Финальная модель: **{res['final_model']}**, "
            f"порог решения {res['decision_threshold']:.2f} (подобран по F1 на out-of-fold)",
            "",
            "| model | accuracy | precision | recall | f1 | roc_auc | avg_precision |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in res["test_metrics"]:
            lines.append(
                "| {model} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | "
                "{f1:.4f} | {roc_auc:.4f} | {avg_precision:.4f} |".format(**row)
            )
        lines.append("")
        if res.get("top_features"):
            lines += [
                f"**Топ-признаки финальной модели (permutation importance, "
                f"падение {PRIMARY_METRIC.upper()}):**",
                "",
            ]
            lines += [f"- `{name}` — {value:.4f}" for name, value in list(res["top_features"].items())[:10]]
            lines.append("")
    path = METRICS_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сквозной ML-пайплайн проекта.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS) + ["all"],
        default="all",
        help="какой датасет обработать (по умолчанию — оба)",
    )
    parser.add_argument("--skip-tuning", action="store_true", help="пропустить подбор гиперпараметров")
    parser.add_argument(
        "--skip-leakage-experiment",
        action="store_true",
        help="не запускать эксперимент с утечкой duration (bank)",
    )
    args = parser.parse_args(argv)

    keys = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    results = []
    for key in keys:
        results.append(run_dataset(key, skip_tuning=args.skip_tuning))

    if "bank" in keys and not args.skip_leakage_experiment:
        run_duration_leakage_experiment()

    summary_path = write_summary_md(results)
    print(f"\n[ok] Сводка: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
