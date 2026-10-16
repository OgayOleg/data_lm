#!/usr/bin/env python
"""Build a self-contained HTML presentation from the pipeline artefacts.

Reads metrics from ``reports/metrics/*.csv`` and embeds figures from
``reports/figures/`` as base64, so the resulting ``reports/presentation.html``
is a single portable file (works offline, easy to attach to a submission).

Usage::

    python scripts/build_presentation.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT = PROJECT_ROOT / "reports" / "presentation.html"

METRIC_LABELS = {
    "model": "Модель",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "avg_precision": "PR-AUC",
}


def img_b64(name: str) -> str:
    """Return a data-URI for a figure (empty string if the file is missing)."""
    path = FIGURES_DIR / name
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def metrics_table(csv_name: str, highlight: str | None = None) -> str:
    """Render a metrics CSV as an HTML table with the best row highlighted."""
    path = METRICS_DIR / csv_name
    if not path.exists():
        return "<p class='muted'>Таблица недоступна</p>"

    df = pd.read_csv(path)
    columns = [c for c in df.columns if c in METRIC_LABELS]
    rows = []
    for _, row in df.iterrows():
        is_best = highlight is not None and highlight in str(row["model"])
        cls = " class='best'" if is_best else ""
        cells = []
        for col in columns:
            value = row[col]
            if isinstance(value, str):
                cells.append(f"<td>{value}</td>")
            else:
                cells.append(f"<td class='num'>{value:.3f}</td>")
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")

    header = "".join(f"<th>{METRIC_LABELS[c]}</th>" for c in columns)
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def load_summary() -> dict[str, dict]:
    """Load per-dataset summary JSON written by the pipeline."""
    out = {}
    for key in ("telco", "bank"):
        path = METRICS_DIR / f"{key}_summary.json"
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))
    return out


def build_html() -> str:
    summary = load_summary()
    telco = summary.get("telco", {})
    bank = summary.get("bank", {})

    telco_best = telco.get("final_model", "logreg_tuned")
    bank_best = bank.get("final_model", "xgboost_tuned")

    slides = []

    # --- 1. title -----------------------------------------------------------
    slides.append("""
        <section class="slide title-slide">
          <div class="badge">Проект-практикум · Введение в науку о данных</div>
          <h1>Пайплайн машинного обучения<br>для бинарной классификации</h1>
          <p class="subtitle">Два датасета Kaggle · 4 модели · кросс-валидация · тюнинг · MLOps-автоматизация</p>
          <div class="roles">
            <div class="role">
              <span class="role-name">Data Scientist / Researcher</span>
              <span class="role-task">EDA · feature engineering · обучение и оценка моделей · интерпретация</span>
            </div>
            <div class="role">
              <span class="role-name">ML Engineer / DevOps Engineer</span>
              <span class="role-task">архитектура кода · тесты · линтинг · DVC · CI · документация</span>
            </div>
          </div>
          <p class="footer-note">Профиль: DevOps-инженерия · 2026</p>
        </section>
        """)

    # --- 2. task & data -----------------------------------------------------
    slides.append("""
        <section class="slide">
          <h2>1. Задача и данные</h2>
          <div class="cards">
            <div class="card">
              <h3>Telco Customer Churn</h3>
              <p class="muted">Уйдёт ли клиент телеком-компании?</p>
              <ul>
                <li><b>7 043</b> записи · 20 признаков</li>
                <li>Дисбаланс <b>73.5% / 26.5%</b></li>
                <li>11 пустых <code>TotalCharges</code></li>
              </ul>
            </div>
            <div class="card">
              <h3>Bank Marketing</h3>
              <p class="muted">Подпишется ли клиент на депозит?</p>
              <ul>
                <li><b>41 188</b> записей · 20 признаков</li>
                <li>Дисбаланс <b>88.7% / 11.3%</b></li>
                <li><b>Утечка</b> в признаке <code>duration</code></li>
              </ul>
            </div>
          </div>
          <p class="takeaway">48 231 наблюдение · разные уровни дисбаланса и разные «ловушки» данных —
          проверяем устойчивость методологии, а не подгонку под один датасет.</p>
        </section>
        """)

    # --- 3. EDA -------------------------------------------------------------
    figure = img_b64("telco_target_balance.png")
    slides.append(f"""
        <section class="slide">
          <h2>2. EDA: главные находки (Data Scientist)</h2>
          <div class="two-col">
            <div>
              <img src="{figure}" alt="Баланс классов Telco">
            </div>
            <div>
              <table>
                <tr><th>Находка</th><th>Цифры</th></tr>
                <tr><td>Контракт Month-to-month</td><td class="num">42.7% ухода</td></tr>
                <tr><td>Контракт Two year</td><td class="num">2.8% ухода</td></tr>
                <tr><td>Стаж 0–10 мес.</td><td class="num">~50% ухода</td></tr>
                <tr><td>Electronic check</td><td class="num">45.3% ухода</td></tr>
                <tr><td>Fiber optic</td><td class="num">41.9% ухода</td></tr>
              </table>
              <p class="takeaway">Риск сконцентрирован в сегменте «новый клиент +
              месячный контракт + ручная оплата» — готовый адресный список для удержания.</p>
            </div>
          </div>
        </section>
        """)

    # --- 4. data problems ---------------------------------------------------
    slides.append("""
        <section class="slide">
          <h2>3. Проблемы данных и их решение</h2>
          <div class="cards three">
            <div class="card">
              <h3>Пропуски</h3>
              <p>11 пустых <code>TotalCharges</code> — все у клиентов с <code>tenure = 0</code>.</p>
              <p class="muted">Не «ноль оплат», а «счёт ещё не выставлялся» → медианная импутация
              внутри train-фолда + признаки сегмента новичков.</p>
            </div>
            <div class="card">
              <h3>Выбросы</h3>
              <p>Правило 1.5·IQR: <b>0%</b> выбросов в числовых признаках Telco.</p>
              <p class="muted">Экстремальные значения — реальные клиентские сегменты,
              а не ошибки ввода → данные не фильтруем.</p>
            </div>
            <div class="card danger">
              <h3>Утечка данных</h3>
              <p><code>duration</code> известен только после звонка.</p>
              <table class="mini">
                <tr><td>без duration</td><td class="num">0.802</td></tr>
                <tr><td>с duration</td><td class="num">0.950</td></tr>
              </table>
              <p class="muted">+0.148 ROC-AUC — «фантомное» качество → признак исключён.</p>
            </div>
          </div>
          <p class="takeaway">Все обучаемые преобразования выполняются внутри sklearn-pipeline:
          импутация, масштабирование и кодирование считаются на train и применяются к тесту без переобучения.</p>
        </section>
        """)

    # --- 5. feature engineering ---------------------------------------------
    slides.append("""
        <section class="slide">
          <h2>4. Feature engineering</h2>
          <div class="cards">
            <div class="card">
              <h3>Telco — 7 новых признаков</h3>
              <ul>
                <li><code>num_addons</code> — число доп. услуг</li>
                <li><code>tenure_group</code> — сегменты стажа</li>
                <li><code>avg_monthly_charge</code>, <code>charge_growth</code></li>
                <li><code>has_family</code>, <code>is_electronic_check</code>, <code>is_new_customer</code></li>
              </ul>
            </div>
            <div class="card">
              <h3>Bank — 5 новых признаков</h3>
              <ul>
                <li><code>was_contacted_before</code></li>
                <li><code>pdays_recency</code> (сентинел 999 → «never»)</li>
                <li><code>campaign_log</code>, <code>season</code>, <code>macro_stress</code></li>
              </ul>
            </div>
          </div>
          <p class="takeaway">Три из пяти важнейших признаков Telco и два из пяти у Bank —
          созданные на этом этапе (по permutation importance).</p>
        </section>
        """)

    # --- 6. architecture ----------------------------------------------------
    slides.append("""
        <section class="slide">
          <h2>5. Архитектура и автоматизация (ML Engineer)</h2>
          <div class="two-col wide-left">
            <div>
              <table class="arch">
                <tr><td><code>src/config.py</code></td><td>пути, seed, реестр датасетов, SHA-256</td></tr>
                <tr><td><code>src/download.py</code>, <code>datasets.py</code></td><td>загрузка с проверкой, очистка</td></tr>
                <tr><td><code>src/features.py</code></td><td>feature engineering</td></tr>
                <tr><td><code>src/preprocessing.py</code></td><td>ColumnTransformer (fit на train)</td></tr>
                <tr><td><code>src/models.py</code>, <code>training.py</code></td><td>4 модели, CV, тюнинг, порог</td></tr>
                <tr><td><code>src/evaluation.py</code>, <code>plotting.py</code></td><td>метрики, 17 графиков</td></tr>
                <tr><td><code>src/pipeline.py</code></td><td><b>весь пайплайн одной командой</b></td></tr>
              </table>
            </div>
            <div class="side-box">
              <h3>Воспроизводимость</h3>
              <ul>
                <li><code>bash scripts/run_all.sh</code></li>
                <li><code>make pipeline</code></li>
                <li><code>dvc repro</code></li>
              </ul>
              <h3>Качество</h3>
              <ul>
                <li><b>23</b> теста (pytest)</li>
                <li>flake8 + black — 0 замечаний</li>
                <li>CI: линт → тесты → smoke-прогон</li>
              </ul>
            </div>
          </div>
        </section>
        """)

    # --- 7. methodology -----------------------------------------------------
    slides.append("""
        <section class="slide">
          <h2>6. Методология: честное сравнение моделей</h2>
          <div class="pipeline-flow">
            <span>Stratified 80/20</span><span class="arrow">→</span>
            <span>5-fold CV<br><small>4 модели</small></span><span class="arrow">→</span>
            <span>RandomizedSearchCV<br><small>лучшая модель</small></span><span class="arrow">→</span>
            <span>порог по out-of-fold F1</span><span class="arrow">→</span>
            <span>оценка на тесте</span>
          </div>
          <div class="cards three">
            <div class="card">
              <h3>Дисбаланс</h3>
              <p><code>class_weight='balanced'</code> / <code>scale_pos_weight</code>
              <b>для всех</b> моделей — иначе сравнение алгоритмов подменялось бы сравнением решающих правил.</p>
            </div>
            <div class="card">
              <h3>Защита от утечек</h3>
              <p>Препроцессинг обучается только на train-фолдах;<br>
              <code>duration</code> удалён до обучения;<br>
              порог выбирается на out-of-fold, а не на тесте.</p>
            </div>
            <div class="card">
              <h3>Метрики</h3>
              <p>Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
              Основная метрика сравнения — <b>ROC-AUC</b>, финальный порог — по <b>F1</b>.</p>
            </div>
          </div>
        </section>
        """)

    # --- 8. results ---------------------------------------------------------
    roc = img_b64("bank_roc_curves.png")
    slides.append(f"""
        <section class="slide">
          <h2>7. Результаты на тестовой выборке</h2>
          <div class="two-col wide-left">
            <div>
              <h3 class="small">Bank Marketing — все модели</h3>
              {metrics_table("bank_test_metrics_with_tuned.csv", highlight=bank_best)}
              <h3 class="small">Telco Churn — все модели</h3>
              {metrics_table("telco_test_metrics_with_tuned.csv", highlight=telco_best)}
            </div>
            <div>
              <img src="{roc}" alt="ROC-кривые Bank Marketing">
            </div>
          </div>
          <p class="takeaway">У лучшей по ROC-AUC модели Bank F1 при пороге 0.5 был наихудшим (0.342):
          подбор порога поднял F1 до 0.528 и recall в 2.7 раза — порог важен не меньше выбора модели.</p>
        </section>
        """)

    # --- 9. threshold & interpretation -------------------------------------
    thr = img_b64("telco_threshold_curve.png")
    imp = img_b64("bank_feature_importance.png")
    slides.append(f"""
        <section class="slide">
          <h2>8. Порог решения и интерпретация</h2>
          <div class="two-col">
            <div>
              <img src="{thr}" alt="Подбор порога решения">
            </div>
            <div>
              <img src="{imp}" alt="Важность признаков">
            </div>
          </div>
          <div class="two-col">
            <p class="takeaway">Порог — такой же гиперпараметр бизнес-задачи, как и модель:
            Telco F1 0.611 → 0.637 (порог 0.58); Bank F1 0.342 → 0.528 (порог 0.19).</p>
            <p class="takeaway">Telco: Contract (−0.038), tenure (−0.025), tenure_group (−0.011).
            Bank: nr.employed (−0.047), emp.var.rate (−0.033), euribor3m (−0.012).</p>
          </div>
          <p class="takeaway">Контраст датасетов: в Telco важность сосредоточена в
          <b>управляемых</b> признаках клиента (на них можно влиять кампанией), в Bank —
          в <b>макроэкономическом фоне</b>, который определяет, <i>когда</i> звонить,
          а индивидуальные признаки — <i>кому</i>.</p>
        </section>
        """)

    # --- 10. conclusions ----------------------------------------------------
    slides.append("""
        <section class="slide">
          <h2>9. Выводы и развитие</h2>
          <div class="cards three">
            <div class="card">
              <h3>Победители</h3>
              <p><b>Telco:</b> Logistic Regression — линейная модель победила бустинги
              (ROC-AUC 0.848 против 0.828–0.835) и дешевле в сопровождении.</p>
              <p><b>Bank:</b> XGBoost — нелинейные взаимодействия макро-фона
              и истории кампаний (ROC-AUC 0.817).</p>
            </div>
            <div class="card">
              <h3>Опровергнутые гипотезы</h3>
              <p>«Бустинг всегда лучше линейной модели» — неверно.</p>
              <p>«Пропуски TotalCharges = 0» — искажает признак.</p>
              <p>«accuracy подходит для permutation importance» — при балансировке
              весов важности получаются отрицательными.</p>
            </div>
            <div class="card">
              <h3>Развитие</h3>
              <ul>
                <li>MLflow — трекинг экспериментов</li>
                <li>FastAPI + Docker — деплой</li>
                <li>мониторинг дрейфа (PSI/KS)</li>
                <li>порог по стоимости ошибок, SHAP</li>
              </ul>
            </div>
          </div>
          <p class="takeaway final">Модели дают качество, пригодное для пилота: ранжируют клиентов
          по вероятности события и направляют ограниченный бюджет в верхний дециль риска.
          Воспроизведение — одна команда: <code>bash scripts/run_all.sh</code></p>
        </section>
        """)

    slides_html = "\n".join(slides)
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>ML-пайплайн для бинарной классификации — презентация</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --navy: #10233f;
    --blue: #2b6cb0;
    --light: #f7f9fc;
    --accent: #c05621;
    --green: #2f855a;
    --red: #c53030;
    --muted: #64748b;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #0b1526; color: #0f172a;
  }}
  .deck {{ max-width: 1180px; margin: 0 auto; padding: 24px 16px 72px; }}
  .slide {{
    background: #fff; border-radius: 18px; padding: 38px 44px;
    margin-bottom: 26px; box-shadow: 0 18px 40px rgba(4, 12, 28, .35);
    min-height: 600px; display: flex; flex-direction: column; gap: 18px;
    page-break-after: always;
  }}
  .slide h1 {{ font-size: 40px; line-height: 1.15; margin: 6px 0; color: var(--navy); }}
  .slide h2 {{
    font-size: 26px; margin: 0; color: var(--navy);
    border-left: 5px solid var(--blue); padding-left: 14px;
  }}
  .slide h3 {{ font-size: 17px; margin: 8px 0 6px; color: var(--navy); }}
  .slide h3.small {{ font-size: 15px; color: var(--muted); text-transform: uppercase;
    letter-spacing: .04em; margin-top: 10px; }}
  .subtitle {{ font-size: 19px; color: var(--blue); margin: 0; font-weight: 600; }}
  .badge {{
    display: inline-block; background: var(--light); color: var(--blue);
    border: 1px solid #dbe6f3; border-radius: 999px; padding: 6px 16px;
    font-size: 13px; font-weight: 700; letter-spacing: .03em; width: fit-content;
  }}
  .title-slide {{ justify-content: center; text-align: left; }}
  .roles {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 14px; }}
  .role {{
    background: var(--light); border-radius: 14px; padding: 16px 18px;
    border-left: 4px solid var(--blue);
  }}
  .role-name {{ display: block; font-weight: 700; color: var(--navy); margin-bottom: 6px; }}
  .role-task {{ color: var(--muted); font-size: 14px; }}
  .footer-note {{ color: var(--muted); font-size: 14px; }}
  .cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .cards.three {{ grid-template-columns: repeat(3, 1fr); }}
  .card {{
    background: var(--light); border-radius: 14px; padding: 18px 20px;
    border: 1px solid #e5edf7;
  }}
  .card.danger {{ background: #fff5f5; border-color: #fbd5d5; }}
  .card ul {{ margin: 8px 0 0 18px; padding: 0; }}
  .card li {{ margin: 5px 0; font-size: 14.5px; }}
  .card p {{ margin: 8px 0 0; font-size: 14.5px; }}
  .muted {{ color: var(--muted); font-size: 14px; }}
  .takeaway {{
    background: #eef7f1; border-left: 4px solid var(--green);
    padding: 12px 16px; border-radius: 10px; font-size: 15px; margin: auto 0 0;
  }}
  .takeaway.final {{ background: #eef4fc; border-left-color: var(--blue); }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; align-items: center; }}
  .two-col.wide-left {{ grid-template-columns: 1.25fr 1fr; align-items: start; }}
  .slide img {{ width: 100%; border-radius: 12px; border: 1px solid #e5edf7; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ padding: 7px 9px; text-align: left; border-bottom: 1px solid #e8eef7; }}
  th {{ background: var(--light); color: var(--navy); font-size: 12.5px; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr.best {{ background: #eef7f1; font-weight: 700; }}
  table.mini {{ font-size: 13px; margin-top: 6px; }}
  table.arch td:first-child {{ white-space: nowrap; color: var(--blue); font-family: ui-monospace, monospace; }}
  .side-box {{ background: var(--navy); color: #e8eef7; border-radius: 14px; padding: 18px 20px; }}
  .side-box h3 {{ color: #fff; }}
  .side-box code {{ color: #ffd7a8; }}
  .side-box ul {{ margin: 6px 0 12px 18px; padding: 0; font-size: 14px; }}
  code {{ background: #eef2f8; padding: 1px 6px; border-radius: 6px;
    font-size: .92em; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .side-box code {{ background: rgba(255,255,255,.12); }}
  .pipeline-flow {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    background: var(--light); padding: 16px; border-radius: 12px; font-size: 14px;
  }}
  .pipeline-flow span:not(.arrow) {{
    background: #fff; border: 1px solid #dbe6f3; border-radius: 10px;
    padding: 8px 12px; text-align: center; font-weight: 600; color: var(--navy);
  }}
  .pipeline-flow small {{ color: var(--muted); font-weight: 400; }}
  .arrow {{ color: var(--blue); font-weight: 700; }}
  .toolbar {{
    position: fixed; right: 18px; bottom: 18px; display: flex; gap: 8px;
    background: rgba(16, 35, 63, .92); padding: 8px; border-radius: 12px;
  }}
  .toolbar a, .toolbar span {{
    color: #fff; text-decoration: none; font-size: 13px; padding: 6px 10px;
    border-radius: 8px; background: rgba(255,255,255,.1);
  }}
  @media print {{
    body {{ background: #fff; }}
    .slide {{ box-shadow: none; border-radius: 0; page-break-after: always; min-height: auto; }}
    .toolbar {{ display: none; }}
  }}
  @media (max-width: 900px) {{
    .cards, .cards.three, .two-col, .two-col.wide-left, .roles {{ grid-template-columns: 1fr; }}
    .slide {{ padding: 24px 20px; }}
  }}
</style>
</head>
<body>
<div class="deck">
{slides_html}
</div>
<div class="toolbar">
  <span id="counter">1 / {len(slides)}</span>
  <a href="#" onclick="document.documentElement.scrollTop=0;return false;">↑ начало</a>
  <a href="#" onclick="window.print();return false;">печать / PDF</a>
</div>
<script>
  const slides = [...document.querySelectorAll('.slide')];
  const counter = document.getElementById('counter');
  const update = () => {{
    const mid = window.innerHeight / 2;
    let idx = slides.findIndex(s => {{
      const r = s.getBoundingClientRect();
      return r.top <= mid && r.bottom >= mid;
    }});
    if (idx === -1) idx = 0;
    counter.textContent = (idx + 1) + ' / ' + slides.length;
  }};
  window.addEventListener('scroll', update);
  window.addEventListener('load', update);
  document.addEventListener('keydown', (e) => {{
    const mid = window.innerHeight / 2;
    let idx = slides.findIndex(s => {{
      const r = s.getBoundingClientRect();
      return r.top <= mid && r.bottom >= mid;
    }});
    if (idx === -1) idx = 0;
    if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {{
      (slides[idx + 1] || slides[idx]).scrollIntoView({{behavior: 'smooth'}});
      e.preventDefault();
    }}
    if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {{
      (slides[idx - 1] || slides[idx]).scrollIntoView({{behavior: 'smooth'}});
      e.preventDefault();
    }}
  }});
</script>
</body>
</html>
"""


def main() -> int:
    html = build_html()
    OUTPUT.write_text(html, encoding="utf-8")
    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"[ok] Презентация сохранена: {OUTPUT} ({size_mb:.2f} MB, самодостаточный HTML)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
