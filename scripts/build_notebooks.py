"""One-time authoring script: builds the course notebooks as .ipynb files."""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook
cell_md = nbf.v4.new_markdown_cell
cell_code = nbf.v4.new_code_cell

# ---------------------------------------------------------------------------
# Notebook 1: Telco EDA
# ---------------------------------------------------------------------------
cells = []
cells.append(cell_md("""# 01 — Разведочный анализ: Telco Customer Churn

**Проект:** Разработка и автоматизация пайплайна машинного обучения для бинарной классификации
**Датасет:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7 043 клиента телеком-компании, 20 признаков, целевая переменная `Churn` (уйдёт ли клиент).

**Задачи ноутбука (роль Data Scientist):**
1. знакомство с данными, типы признаков, статистики;
2. анализ пропусков и «битых» значений;
3. анализ выбросов;
4. распределения признаков и связь с целевой переменной;
5. корреляционный анализ;
6. первые гипотезы и наблюдения для feature engineering."""))

cells.append(cell_code("""# --- Setup ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATASETS
from src.datasets import load_raw_by_key
from src.plotting import setup_style, save_figure

setup_style()
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

spec = DATASETS["telco"]
print(f"Источник (Kaggle): {spec.kaggle_url}")
print(f"Задача: {spec.task_summary}")"""))

cells.append(cell_md("""## 1. Знакомство с данными"""))

cells.append(cell_code("""raw = load_raw_by_key("telco")
print(f"Размер таблицы: {raw.shape[0]} строк × {raw.shape[1]} признаков")
raw.head()"""))

cells.append(cell_code("""raw.info()"""))

cells.append(cell_code("""raw.describe(include="all").T"""))

cells.append(cell_md("""**Наблюдения:**
- `customerID` — уникальный идентификатор, для модели бесполезен (удалим);
- `SeniorCitizen` уже закодирован 0/1, остальные категориальные — строковые `Yes/No/...`;
- `tenure` — срок жизни клиента (месяцы), `MonthlyCharges` / `TotalCharges` — деньги;
- **`TotalCharges` имеет строковый тип** — признак того, что в колонке есть пустые значения."""))

cells.append(cell_md("""## 2. Пропуски и «битые» значения"""))

cells.append(cell_code("""# Формальных NaN в датасете нет...
missing = raw.isna().sum()
print("NaN по колонкам:")
print(missing[missing > 0] if (missing > 0).any() else "  — нет —")

# ...но TotalCharges хранится как строка: ищем пустые/пробельные значения
tc = raw["TotalCharges"].astype(str).str.strip()
blank_mask = tc.eq("")
print(f"\\nПустых значений TotalCharges: {blank_mask.sum()}")

blank_rows = raw.loc[blank_mask, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]]
blank_rows"""))

cells.append(cell_code("""# Интересно: кто эти клиенты?
print("tenure у клиентов с пустым TotalCharges:", raw.loc[blank_mask, "tenure"].unique())
print("Churn у этих клиентов:", raw.loc[blank_mask, "Churn"].value_counts().to_dict())"""))

cells.append(
    cell_md("""**Вывод по пропускам:** все 11 «пустых» `TotalCharges` — это новые клиенты с `tenure = 0`
(они ещё не успели накопить счетов). Это **не случайные** пропуски.

Стратегия обработки (реализована в `src/`):
1. `TotalCharges` → числовой тип, пустые строки → `NaN`;
2. имплементация в sklearn-пайплайне: `SimpleImputer(strategy="median")` —
   медиана считается **только на train-выборке**, что исключает утечку;
3. дополнительно заводим признаки `avg_monthly_charge` и флаг новых клиентов,
   чтобы сохранить сигнал «новичок = риск». """)
)

cells.append(cell_md("""## 3. Баланс целевой переменной"""))

cells.append(cell_code("""from src.plotting import plot_target_balance

y = (raw["Churn"] == "Yes").astype(int)
fig = plot_target_balance(y, "Telco Churn: баланс классов", positive_label="Yes")
plt.show()"""))

cells.append(cell_md("""**Наблюдение:** классы несбалансированы — 73.5% клиентов остаются, 26.5% уходят.
Поэтому Accuracy один вводит в заблуждение: «предсказывай всегда No» — и получи 73.5% accuracy.
Основные метрики для сравнения моделей — **F1 и ROC-AUC** (Precision/Recall тоже смотрим)."""))

cells.append(cell_md("""## 4. Числовые признаки vs Churn"""))

cells.append(cell_code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

plot_df = raw.copy()
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    plot_df[col] = pd.to_numeric(plot_df[col].astype(str).str.strip(), errors="coerce")

for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    for cls, color, label in [(0, "#718096", "остались"), (1, "#c53030", "ушли")]:
        subset = plot_df.loc[y.eq(cls), col].dropna()
        sns.histplot(subset, kde=True, stat="density", element="step",
                     color=color, label=label, ax=ax, alpha=0.35)
    ax.set_title(f"{col} по классам")
    ax.legend()
fig.tight_layout()
plt.show()"""))

cells.append(cell_md("""**Наблюдения:**
- **`tenure`** — главный визуальный сигнал: у клиентов, которые уходят, распределение
  скошено к 0–10 месяцам. Долгожители почти не уходят;
- **`MonthlyCharges`** — ушедшие платили в среднем больше (дорогие тарифы + Fiber optic);
- `TotalCharges` — смесь двух эффектов (tenure × charges), учитываем это в FE."""))

cells.append(cell_code("""# Выбросы: boxplots по классам
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
plot_df["Churn"] = plot_df["Churn"].map({"No": "остались", "Yes": "ушли"})
for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
    sns.boxplot(data=plot_df, x="Churn", y=col, hue="Churn", palette="Set2",
                ax=ax, legend=False, width=0.5)
    ax.set_title(f"Выбросы: {col}")
fig.tight_layout()
plt.show()

# Формальная проверка IQR
for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    s = plot_df[col].dropna()
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"{col}: выбросов по правилу 1.5*IQR — {out} ({out/len(s):.1%})")"""))

cells.append(
    cell_md("""**Вывод по выбросам:** явных «ошибочных» экстремумов нет (это реальные клиентские данные):
`tenure` равномерно заполняет 0–72 месяца, платежи находятся в разумном диапазоне.
Удалять выбросы не будем — они несут сигнал (например, максимальные `MonthlyCharges`).""")
)

cells.append(cell_md("""## 5. Корреляционный анализ"""))

cells.append(cell_code("""from src.plotting import plot_correlation_heatmap

numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
corr_df = plot_df.copy()
corr_df["Churn_bin"] = (corr_df["Churn"] == "ушли").astype(int)

fig = plot_correlation_heatmap(corr_df[numeric_cols + ["Churn_bin"]].rename(columns={"Churn_bin": "Churn"}),
                               "Корреляция числовых признаков и таргета (точечная бисериальная)", annot=True)
plt.show()"""))

cells.append(cell_md("""**Наблюдения:**
- `tenure` ↔ `TotalCharges` — ожидаемая мультиколлинеарность (TotalCharges ≈ tenure × средний платёж);
- `MonthlyCharges` ↔ `TotalCharges` — умеренная связь;
- с таргетом сильнее всех отрицательно коррелирует `tenure` (чем дольше клиент — тем стабильнее)."""))

cells.append(cell_md("""## 6. Категориальные признаки vs уровень оттока"""))

cells.append(cell_code("""cat_cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport",
            "OnlineSecurity", "PaperlessBilling", "SeniorCitizen"]

rates = []
for col in cat_cols:
    tab = raw.assign(churn=y).groupby(col, observed=True)["churn"].agg(["mean", "count"])
    tab.columns = ["churn_rate", "n"]
    tab["feature"] = col
    tab["level"] = tab.index.astype(str)
    rates.append(tab.reset_index(drop=True))
rates = pd.concat(rates, ignore_index=True)

fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, col in zip(axes.ravel(), cat_cols):
    part = rates[rates["feature"] == col].sort_values("churn_rate", ascending=False)
    sns.barplot(data=part, x="level", y="churn_rate", hue="level",
                palette="RdYlGn_r", ax=ax, legend=False)
    ax.set_title(f"Churn rate по {col}")
    ax.set_xlabel("")
    ax.set_ylabel("доля ухода")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 0.7)
axes.ravel()[-1].axis("off")
fig.tight_layout()
plt.show()"""))

cells.append(cell_md("""**Ключевые находки:**
- **`Contract`** — сильнейший категориальный предиктор: уход Month-to-month ≈ 43%,
  у Two year — ≈ 3%;
- **`InternetService = Fiber optic`** — аномально высокий отток (≈ 42%);
- **`PaymentMethod = Electronic check`** — отток ≈ 45% (в 2 раза выше остальных);
- `TechSupport`/`OnlineSecurity = No` — заметно повышают риск ухода;
- `SeniorCitizen` уходят чаще (≈ 42% против 24%)."""))

cells.append(cell_md("""## 7. Разведка признаков для feature engineering"""))

cells.append(cell_code("""# Гипотеза: количество подключённых доп. услуг удерживает клиента
addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
tmp = raw.copy()
tmp["num_addons"] = sum((tmp[c] == "Yes").astype(int) for c in addon_cols)
tmp["churn"] = y

fig, ax = plt.subplots(figsize=(8.5, 4.6))
addon_rate = tmp.groupby("num_addons")["churn"].agg(["mean", "count"])
sns.barplot(x=addon_rate.index, y=addon_rate["mean"], hue=addon_rate.index,
            palette="Blues_d", legend=False, ax=ax)
for i, (rate, n) in enumerate(zip(addon_rate["mean"], addon_rate["count"])):
    ax.text(i, rate + 0.01, f"{rate:.0%}\\n(n={n})", ha="center", fontsize=9)
ax.set_xlabel("количество доп. услуг")
ax.set_ylabel("доля ухода")
ax.set_title("Гипотеза: доп. услуги снижают отток")
plt.show()"""))

cells.append(cell_code("""# Гипотеза: сегменты по tenure
tmp["tenure_group"] = pd.cut(tmp["tenure"], bins=[-0.01, 1, 12, 24, 48, np.inf],
                             labels=["0-1", "1-12", "12-24", "24-48", "48+"])
seg = tmp.groupby("tenure_group", observed=True)["churn"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(8.5, 4.4))
sns.barplot(x=seg.index.astype(str), y=seg["mean"], hue=seg.index.astype(str),
            palette="OrRd", legend=False, ax=ax)
for i, (rate, n) in enumerate(zip(seg["mean"], seg["count"])):
    ax.text(i, rate + 0.01, f"{rate:.0%}\\n(n={n})", ha="center", fontsize=9)
ax.set_xlabel("срок жизни клиента, мес.")
ax.set_ylabel("доля ухода")
ax.set_title("Отток по сегментам tenure")
plt.show()"""))

cells.append(cell_md("""## 8. Выводы по EDA (Telco)

| # | Находка | Решение в пайплайне |
|---|---------|---------------------|
| 1 | `TotalCharges` строковый, 11 пустых у клиентов с `tenure=0` | числовой тип + `SimpleImputer(median)` в sklearn-пайплайне |
| 2 | Дисбаланс классов 73/27 | метрики F1/ROC-AUC, `class_weight='balanced'`, stratified split |
| 3 | `tenure` — сильнейший числовой предиктор | + `tenure_group`, `is_new_customer` |
| 4 | Доп. услуги удерживают клиента (U-образная связь) | + `num_addons` |
| 5 | `Contract`, `PaymentMethod`, `InternetService` — сильные категории | one-hot кодирование в ColumnTransformer |
| 6 | Мультиколлинеарность tenure/TotalCharges | масштабирование + регуляризация; деревья нечувствительны |
| 7 | Выбросов-ошибок нет | данные не фильтруем, работаем как есть |

Переходим к ноутбуку **02** (второй датасет) и **03** (моделирование)."""))

nb1 = nb(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)

# ---------------------------------------------------------------------------
# Notebook 2: Bank EDA
# ---------------------------------------------------------------------------
cells = []
cells.append(cell_md("""# 02 — Разведочный анализ: Bank Marketing (bank-additional-full)

**Датасет:** [Bank Marketing (Kaggle)](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) —
41 188 клиентов банка Португалии, 20 признаков, целевая переменная `y`
(подписался ли клиент на срочный депозит по итогам телефонной кампании).

**Задачи ноутбука (роль Data Scientist):**
1. структура данных, баланс классов;
2. анализ «unknown»-категорий и качества данных;
3. **обнаружение утечки целевой переменной** (признак `duration`);
4. анализ признаков кампании (`pdays`, `previous`, `poutcome`);
5. сезонность и макроэкономические признаки;
6. выводы и стратегия предобработки."""))

cells.append(cell_code("""# --- Setup ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import DATASETS
from src.datasets import load_raw_by_key
from src.plotting import setup_style

setup_style()
pd.set_option("display.max_columns", 40)

spec = DATASETS["bank"]
print(f"Источник (Kaggle): {spec.kaggle_url}")
print(f"Задача: {spec.task_summary}")
print("Примечания:", *spec.notes, sep="\\n  - ")"""))

cells.append(cell_md("""## 1. Знакомство с данными"""))

cells.append(cell_code("""raw = load_raw_by_key("bank")
print(f"Размер таблицы: {raw.shape[0]} строк × {raw.shape[1]} признаков")
raw.head()"""))

cells.append(cell_code("""raw.info()"""))

cells.append(cell_code("""raw.describe().T"""))

cells.append(cell_md("""**Структура:** 10 категориальных признаков (job, marital, education, default, housing,
loan, contact, month, day_of_week, poutcome) и 10 числовых (age, duration, campaign,
pdays, previous + 5 макроэкономических показателей)."""))

cells.append(cell_md("""## 2. Баланс классов"""))

cells.append(cell_code("""from src.plotting import plot_target_balance

y = (raw["y"] == "yes").astype(int)
fig = plot_target_balance(y, "Bank Marketing: баланс классов (подписка на депозит)", positive_label="yes")
plt.show()"""))

cells.append(cell_md("""**Дисбаланс сильнее, чем в Telco:** только 11.3% клиентов подписываются.
Accuracy бесполезен как одиночная метрика; ключевые — **ROC-AUC, F1, Recall**
(банку важно не пропустить потенциального вкладчика)."""))

cells.append(cell_md("""## 3. Качество данных: категории 'unknown'"""))

cells.append(cell_code("""cat_cols = ["job", "marital", "education", "default", "housing", "loan",
            "contact", "month", "day_of_week", "poutcome"]
unknown_share = pd.Series(
    {c: (raw[c] == "unknown").mean() for c in cat_cols}
).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8.5, 4.4))
sns.barplot(x=unknown_share.values, y=unknown_share.index, hue=unknown_share.index,
            palette="YlOrRd", legend=False, ax=ax)
for i, v in enumerate(unknown_share.values):
    ax.text(v + 0.005, i, f"{v:.1%}", va="center", fontsize=9)
ax.set_xlabel("доля значений 'unknown'")
ax.set_title("Пропуски-«unknown» по категориальным признакам")
plt.show()"""))

cells.append(cell_md("""**Наблюдения:**
- `default` (был ли дефолт) почти целиком 'unknown'/'no' — признак почти неинформативен;
- `job`, `education`, `housing`, `loan` содержат 0.8–25% 'unknown';

**Стратегия:** 'unknown' оставляем как **отдельный уровень** категории
(он сам по себе может быть сигналом — клиент не захотел отвечать),
а не выбрасываем строки (потеряли бы до 25% данных)."""))

cells.append(cell_md("""## 4. Утечка целевой переменной: признак `duration`

`duration` — длительность последнего звонка. Она известна **только после
завершения контакта**, поэтому для предсказания «подпишется ли клиент»
до/во время звонка этот признак использовать нельзя — это классическая
**утечка данных (data leakage)**. Посмотрим, насколько он силен:"""))

cells.append(cell_code("""fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))

plot_df = raw.assign(subscribed=raw["y"].map({"no": "нет", "yes": "да"}))
sns.histplot(data=plot_df, x="duration", hue="subscribed", stat="density",
             element="step", common_norm=False, palette=["#718096", "#c53030"],
             log_scale=(False, True), ax=axes[0], alpha=0.4)
axes[0].set_title("duration по классам (log-y)")

sns.boxplot(data=plot_df, x="subscribed", y="duration", hue="subscribed",
            palette=["#718096", "#c53030"], legend=False, ax=axes[1], width=0.5)
axes[1].set_title("duration: выбросы и медианы")
fig.tight_layout()
plt.show()

print("Медиана duration при y=no :", raw.loc[raw.y == "no", "duration"].median(), "сек")
print("Медиана duration при y=yes:", raw.loc[raw.y == "yes", "duration"].median(), "сек")"""))

cells.append(cell_code("""# Корреляция duration с таргетом
corr = np.corrcoef(raw["duration"], y)[0, 1]
print(f"Корреляция duration ↔ подписка: {corr:.3f}")"""))

cells.append(cell_md("""**Вывод:** распределения почти разделимы — средний звонок подписавшегося клиента
в 3–4 раза дольше. Модель с `duration` «обучится» простому правилу
«длинный звонок → да», что на практике бесполезно (звонок ещё идёт!).

**Решение:** `duration` **исключается** из матрицы признаков (`src/features.py`,
`src/preprocessing.prepare_matrix`), а эффект утечки измеряется отдельным
экспериментом (ноутбук 03 / `reports/metrics/bank_duration_leakage.csv`)."""))

cells.append(cell_md("""## 5. Признаки кампании: `pdays`, `previous`, `poutcome`"""))

cells.append(cell_code("""print("pdays == 999 (не контактировали раньше):", (raw["pdays"] == 999).mean())
print("previous == 0:", (raw["previous"] == 0).mean())

fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))

# 1) Уже были контакты?
tmp = raw.assign(was_contacted=raw["previous"] > 0, subscribed=y)
rate = tmp.groupby("was_contacted")["subscribed"].mean()
axes[0].bar(["не контактировали", "контактировали"], rate.values,
            color=["#718096", "#2f855a"])
for i, v in enumerate(rate.values):
    axes[0].text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold")
axes[0].set_title("Доля подписки: был ли контакт раньше")
axes[0].set_ylim(0, rate.max() * 1.25)

# 2) Результат прошлой кампании
part = raw.assign(subscribed=y).groupby("poutcome")["subscribed"].agg(["mean", "count"])
axes[1].bar(part.index, part["mean"], color=["#c05621", "#c53030", "#2f855a"])
for i, (v, n) in enumerate(zip(part["mean"], part["count"])):
    axes[1].text(i, v + 0.01, f"{v:.1%}\\n(n={n})", ha="center", fontsize=9)
axes[1].set_title("Доля подписки по poutcome")
axes[1].set_ylim(0, 0.8)

# 3) Количество контактов в текущей кампании
sns.lineplot(data=raw.assign(subscribed=y, campaign=raw["campaign"].clip(upper=15)),
             x="campaign", y="subscribed", errorbar=("ci", 95), ax=axes[2],
             color="#2b6cb0", lw=2)
axes[2].set_title("Доля подписки vs число контактов (campaign)")
axes[2].set_ylabel("доля подписки")
fig.tight_layout()
plt.show()"""))

cells.append(cell_md("""**Находки:**
- клиенты, с которыми **уже контактировали** (`previous > 0`), подписываются вдвое чаще;
- `poutcome = success` — мощнейший сигнал (≈ 65% подписки против 9% у остальных);
- рост числа контактов `campaign` после 4–6 звонков **снижает** конверсию —
  настойчивость утомляет клиентов → инженерия: `campaign_log`, `was_contacted_before`."""))

cells.append(cell_md("""## 6. Сезонность и макроэкономика"""))

cells.append(cell_code("""month_order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
month_rate = raw.assign(subscribed=y).groupby("month")["subscribed"].agg(["mean", "count"])
month_rate = month_rate.reindex(month_order).dropna()

fig, axes = plt.subplots(1, 2, figsize=(15, 4.4))
sns.barplot(x=month_rate.index, y=month_rate["mean"], hue=month_rate.index,
            palette="coolwarm", legend=False, ax=axes[0])
for i, (v, n) in enumerate(zip(month_rate["mean"], month_rate["count"])):
    axes[0].text(i, v + 0.005, f"{v:.0%}", ha="center", fontsize=9)
axes[0].set_title("Конверсия по месяцам")
axes[0].set_ylabel("доля подписки")

econ = raw.assign(subscribed=y)
sns.scatterplot(data=econ, x="euribor3m", y="cons.conf.idx", hue="subscribed",
                palette={0: "#718096", 1: "#c53030"}, alpha=0.25, s=18,
                ax=axes[1], legend=False)
axes[1].set_title("Макросреда: euribor3m vs доверие потребителей")
fig.tight_layout()
plt.show()"""))

cells.append(cell_md("""**Наблюдения:**
- пики конверсии — **март, сентябрь, октябрь** (начало учебного года, «майский» провал —
  массовые кампании приносят усталость);
- `euribor3m` (ставка межбанковского рынка) хорошо разделяет периоды кампании —
  макропризнаки почти однозначно кодируют дату контакта;
- для модели оставляем и сезон (`season` из месяца), и макропризнаки."""))

cells.append(cell_md("""## 7. Числовые признаки и выбросы"""))

cells.append(cell_code("""fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, col in zip(axes, ["age", "campaign", "pdays"]):
    data = raw[col].replace(999, np.nan) if col == "pdays" else raw[col]
    sns.histplot(data.dropna(), bins=40, ax=ax, color="#2b6cb0")
    ax.set_title(f"Распределение {col}" + (" (без 999)" if col == "pdays" else ""))
fig.tight_layout()
plt.show()

for col in ["age", "campaign"]:
    s = raw[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
    print(f"{col}: выбросов 1.5*IQR — {out} ({out/len(s):.2%})")"""))

cells.append(cell_md("""**Вывод:** `campaign` имеет длинный хвост (до 56 контактов) — логарифмируем
(`campaign_log`); возрастные «выбросы» (70–90 лет) — реальные клиенты, оставляем."""))

cells.append(cell_md("""## 8. Корреляционный анализ"""))

cells.append(cell_code("""from src.plotting import plot_correlation_heatmap

num_cols = ["age", "campaign", "pdays", "previous", "emp.var.rate",
            "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
corr_df = raw[num_cols].replace(999, np.nan).copy()
corr_df["subscribed"] = y
fig = plot_correlation_heatmap(corr_df.rename(columns={"subscribed": "y"}),
                               "Bank Marketing: корреляция числовых признаков", annot=True)
plt.show()"""))

cells.append(cell_md("""**Мультиколлинеарность:** `emp.var.rate`, `euribor3m`, `nr.employed` сильно
коррелируют между собой (все описывают макросреду). Для линейной модели это
лечится регуляризацией, для бустинга — не проблема (деревья выбирают один из
коррелированных признаков)."""))

cells.append(cell_md("""## 9. Выводы по EDA (Bank Marketing)

| # | Находка | Решение в пайплайне |
|---|---------|---------------------|
| 1 | **Утечка:** `duration` сильно коррелирует с таргетом, но неизвестен к моменту предсказания | исключён из матрицы признаков; эффект измерен экспериментом |
| 2 | Дисбаланс 89/11 | stratified split, `class_weight='balanced'`, метрики F1/ROC-AUC/PR-AUC |
| 3 | `pdays = 999` — сентинел «не контактировали» | замена на `NaN` + признак `was_contacted_before`, `pdays_recency` |
| 4 | `poutcome=success` и прошлые контакты — сильный сигнал | признаки `was_contacted_before`, `pdays_recency` |
| 5 | 'unknown'-категории до 25% | сохраняются как отдельный уровень (не теряем данные) |
| 6 | Длинный хвост `campaign` | `campaign_log = log1p(campaign)` |
| 7 | Сезонность и макропризнаки | `season` из месяца + исходные макропризнаки |

Датасет готов к моделированию → ноутбук **03**."""))

nb2 = nb(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)

# ---------------------------------------------------------------------------
# Notebook 3: Modeling
# ---------------------------------------------------------------------------
cells = []
cells.append(cell_md("""# 03 — Моделирование, сравнение и интерпретация моделей

**Цель ноутбука (роли Data Scientist + ML Engineer):**
1. продемонстрировать **рефакторинг** исследовательского кода в пакет `src/`;
2. обучить 4 модели (Logistic Regression, Random Forest, XGBoost, LightGBM);
3. оценить на стратифицированной кросс-валидации (CV=5) и тестовой выборке;
4. подобрать гиперпараметры лучшей модели (`RandomizedSearchCV`);
5. интерпретировать результат (важность признаков, матрица ошибок);
6. проверить гипотезу об утечке `duration` на Bank Marketing;
7. свести результаты по двум датасетам.

> Весь код моделирования живёт в `src/` и импортируется ноутбуком —
> ноутбук = исследование и витрина, пакет = воспроизводимый пайплайн
> (`python -m src.pipeline` / `make pipeline`)."""))

cells.append(cell_code("""# --- Setup ---------------------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.plotting import setup_style
from src.pipeline import prepare_dataset
from src.training import evaluate_models_with_cv, train_test_split_stratified, tune_best_model, final_test_metrics
from src.models import get_models
from src.evaluation import confusion_matrix_df

setup_style()
pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")"""))

cells.append(cell_md("""## 1. Подготовка данных (единый контракт для обоих датасетов)

Функция `prepare_dataset(key)` выполняет: загрузка → чистка → feature engineering →
матрица признаков. Масштабирование/кодирование/имплементация — внутри sklearn
`ColumnTransformer`, который **обучается только на train-части** (см. `tests/test_preprocessing.py`)."""))

cells.append(cell_code("""X, y = prepare_dataset("telco")
X_train, X_test, y_train, y_test = train_test_split_stratified(X, y)
print(f"Telco: X = {X.shape}, train = {X_train.shape}, test = {X_test.shape}")
print(f"Доля положительного класса: train={y_train.mean():.3f}, test={y_test.mean():.3f}")
X_train.head(3)"""))

cells.append(cell_md("""## 2. Обучение и кросс-валидация четырёх моделей (Telco)

Каждая модель обёрнута в `Pipeline(preprocessor → model)`, поэтому препроцессинг
не «протекает» из train в CV/тест."""))

cells.append(cell_code("""test_metrics, cv_metrics, fitted = evaluate_models_with_cv(
    X_train, y_train, X_test, y_test, models=get_models()
)
test_metrics"""))

cells.append(cell_code("""from src.plotting import plot_cv_boxplot

fig = plot_cv_boxplot(cv_metrics, "roc_auc", "ROC-AUC по фолдам кросс-валидации (Telco)")
plt.show()"""))

cells.append(cell_md("""## 3. ROC / PR кривые и матрицы ошибок"""))

cells.append(
    cell_code("""from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
from src.plotting import plot_roc_curves, plot_pr_curves, plot_confusion_matrices

roc_data, pr_data, matrices = [], [], []
for name, model in fitted.items():
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    roc_data.append({"name": name, "fpr": fpr, "tpr": tpr})
    pr_data.append({"name": name, "precision": precision, "recall": recall,
                    "ap": average_precision_score(y_test, proba)})
    matrices.append({"name": name, "cm": confusion_matrix_df(y_test, model.predict(X_test)).values})

fig = plot_roc_curves(roc_data, "Telco: ROC-кривые (test)")
plt.show()
fig = plot_pr_curves(pr_data, "Telco: Precision-Recall кривые (test)")
plt.show()
fig = plot_confusion_matrices(matrices, "Telco: матрицы ошибок (test)")
plt.show()""")
)

cells.append(cell_md("""## 4. Подбор гиперпараметров лучшей модели

Используем `RandomizedSearchCV` с пространством поиска из `src/models.py`,
стратифицированным CV=3 и метрикой `roc_auc` для рефита."""))

cells.append(cell_code("""best_name = test_metrics.iloc[0]["model"]
print(f"Лучшая модель по ROC-AUC на тесте: {best_name}")
search = tune_best_model(best_name, X_train, y_train)
tuned_metrics, y_pred, y_proba = final_test_metrics(search, X_test, y_test, model_name=f"{best_name}_tuned")
tuned_df = pd.concat([test_metrics, pd.DataFrame([tuned_metrics])], ignore_index=True)
tuned_df"""))

cells.append(cell_md("""## 5. Выбор порога решения (важно при дисбалансе)

Порог 0.5 — не догма. При дисбалансе 73/27 (Telco) и 89/11 (Bank) оптимальный
компромисс Precision/Recall часто лежит в другом месте. Подбираем порог по
**out-of-fold** предсказаниям на train-данных (тест не трогаем!) по метрике F1."""))

cells.append(
    cell_code(
        """from src.training import optimise_decision_threshold
from src.plotting import plot_threshold_curves
from src.evaluation import compute_metrics

best_threshold, curve = optimise_decision_threshold(search.best_estimator_, X_train, y_train)
print(f"Оптимальный порог по F1 (out-of-fold): {best_threshold:.2f}")

fig = plot_threshold_curves(curve, best_threshold, "Telco: метрики vs порог решения (out-of-fold)")
plt.show()

y_pred_thr = (y_proba >= best_threshold).astype(int)
print("На тестовой выборке:")
print(f"  при пороге 0.50:  F1 = {tuned_metrics['f1']:.4f}, recall = {tuned_metrics['recall']:.4f}, precision = {tuned_metrics['precision']:.4f}")
m = compute_metrics(y_test, y_pred_thr, y_proba)
print(f"  при пороге {best_threshold:.2f}:  F1 = {m['f1']:.4f}, recall = {m['recall']:.4f}, precision = {m['precision']:.4f}")"""
    )
)

cells.append(cell_md("""**Вывод:** порог — такой же гиперпараметр бизнес-задачи, как и модель.
Оптимум по F1 смещает баланс в сторону Recall (для Telco критично не пропустить
уходящего клиента), при этом ROC-AUC не меняется — он не зависит от порога."""))

cells.append(cell_md("""## 6. Интерпретация: важность признаков"""))

cells.append(cell_code("""from sklearn.inspection import permutation_importance
from src.plotting import plot_feature_importance

best_estimator = search.best_estimator_
# ВАЖНО: scoring='roc_auc', а не accuracy. Модель с class_weight='balanced'
# переоценивает долю позитивов, и при перемешивании признака accuracy может
# даже вырасти -> бессмысленные отрицательные важности. ROC-AUC порого-независим.
result = permutation_importance(best_estimator, X_test, y_test, n_repeats=5,
                                random_state=42, n_jobs=-1, scoring="roc_auc")
fig = plot_feature_importance(
    X_test.columns, result.importances_mean,
    f"Telco: permutation importance ({best_name} tuned)", top_n=18,
    xlabel="среднее падение ROC-AUC при перемешивании",
)
plt.show()"""))

cells.append(cell_md("""**Интерпретация:** permutation importance измеряет, насколько падает качество
модели, если значения признака случайно перемешать — работает для любой модели,
в отличие от встроенных важностей деревьев. Наиболее значимые признаки Telco:
`Contract`, `tenure`/`tenure_group`, `InternetService` — что совпадает с выводами
EDA (ноутбук 01).

*Методическая деталь:* метрику для permutation importance нужно выбирать
осознанно. При `scoring='accuracy'` модель с `class_weight='balanced'`
(она переоценивает долю позитивов) даёт отрицательные «важности» — перемешивание
признака даже повышает accuracy. ROC-AUC порого-независим и даёт корректный
порядок."""))

cells.append(cell_md("""## 7. Второй датасет: Bank Marketing — результаты полного прогона

Bank Marketing в 6 раз больше Telco, поэтому его полный прогон (CV=5 × 4 модели +
тюнинг) выполняется конвейером, а не ноутбуком:

```bash
make pipeline        # или: python -m src.pipeline --dataset bank
```

Ниже читаются **сохранённые артефакты** (`reports/metrics/`) — тот же код,
те же функции, полная воспроизводимость."""))

cells.append(cell_code("""metrics_dir = PROJECT_ROOT / "reports" / "metrics"

bank_test = pd.read_csv(metrics_dir / "bank_test_metrics.csv")
bank_cv = pd.read_csv(metrics_dir / "bank_cv_metrics.csv")
bank_final = pd.read_csv(metrics_dir / "bank_final_metrics.csv")

print("Bank Marketing — метрики моделей на тестовой выборке (порог 0.5):")
display(bank_test)
print("Финальная модель на выбранном пороге решения:")
display(bank_final)"""))

cells.append(
    cell_code("""fig = plot_cv_boxplot(bank_cv, "roc_auc", "Bank: ROC-AUC по фолдам CV (полный прогон)")
plt.show()""")
)

cells.append(cell_md("""## 8. Гипотеза об утечке `duration` (проверена)

Гипотеза: *«признак duration неизвестен в момент предсказания и искусственно
завышает метрики»*. Проверка — один и тот же LightGBM, CV=5, с признаком и без:"""))

cells.append(cell_code("""leak = pd.read_csv(metrics_dir / "bank_duration_leakage.csv")
display(leak)
fig = plt.figure(figsize=(8, 4.2))
plt.barh(leak["experiment"], leak["roc_auc_mean"], color=["#2f855a", "#c53030"])
plt.axvline(leak["roc_auc_mean"].min(), ls="--", lw=1, color="gray")
for i, (m, s) in enumerate(zip(leak["roc_auc_mean"], leak["roc_auc_std"])):
    plt.text(m + 0.005, i, f"{m:.3f} ± {s:.3f}", va="center", fontweight="bold")
plt.xlabel("ROC-AUC (CV=5)")
plt.title("Гипотеза подтверждена: duration — утечка целевой переменной")
plt.xlim(0, 1.05)
plt.tight_layout()
plt.show()"""))

cells.append(cell_md("""**Вывод:** гипотеза **подтверждена** — метрика с утечкой существенно выше,
но такая модель непригодна для практики. Финальная модель Bank Marketing
обучена **без** `duration`."""))

cells.append(cell_md("""## 9. Сводная таблица: два датасета, лучшие модели"""))

cells.append(cell_code("""rows = []
for key in ["telco", "bank"]:
    df = pd.read_csv(metrics_dir / f"{key}_test_metrics.csv")
    best = df.sort_values("roc_auc", ascending=False).iloc[0]
    final = pd.read_csv(metrics_dir / f"{key}_final_metrics.csv").iloc[0]
    rows.append({
        "dataset": key,
        "best_model (0.5)": best["model"],
        "roc_auc": best["roc_auc"],
        "final_model": final["model"],
        "final_f1": final["f1"],
        "final_precision": final["precision"],
        "final_recall": final["recall"],
    })
summary = pd.DataFrame(rows)
display(summary)

fig = plt.figure(figsize=(8.5, 4.2))
x = np.arange(len(summary))
width = 0.34
plt.bar(x - width/2, summary["roc_auc"], width, label="ROC-AUC", color="#2b6cb0")
plt.bar(x + width/2, summary["final_f1"], width, label="F1 (финальный порог)", color="#c05621")
plt.xticks(x, [f"{r.dataset}\\n({r.final_model})" for r in summary.itertuples()])
plt.ylim(0, 1.05)
plt.title("Финальные модели: Telco vs Bank (test)")
for xi, (auc, f1) in enumerate(zip(summary["roc_auc"], summary["final_f1"])):
    plt.text(xi - width/2, auc + 0.015, f"{auc:.3f}", ha="center", fontsize=9)
    plt.text(xi + width/2, f1 + 0.015, f"{f1:.3f}", ha="center", fontsize=9)
plt.legend()
plt.tight_layout()
plt.show()"""))

cells.append(cell_md("""## 10. Итоговые выводы (моделирование)

1. **Воспроизводимость:** весь код моделирования — в `src/`, ноутбук импортирует
   те же функции; тесты (`tests/`) фиксируют контракт препроцессинга;
2. **Сравнение моделей:** 4 модели из 3 семейств (линейная, бэггинг, бустинг) —
   логистическая регрессия дает сильный и интерпретируемый baseline,
   бустинги выигрывают на сложных зависимостях (Bank Marketing);
3. **Метрики:** из-за дисбаланса обоих датасетов главные — F1 и ROC-AUC;
   Accuracy приводится только для полноты;
4. **Порог решения** подобран по F1 на out-of-fold предсказаниях —
   это отдельный, не менее важный рычаг, чем выбор модели;
5. **Все модели учитывают дисбаланс** (class_weight / scale_pos_weight),
   иначе сравнение было бы некорректным;
6. **Гипотеза об утечке `duration` подтверждена** и утечка устранена;
7. **Интерпретация:** permutation importance подтверждает находки EDA
   (`tenure`, `Contract`, `poutcome`, признаки кампании).

Полный отчет — `reports/REPORT.md`, слайды — `reports/presentation.md`."""))

nb3 = nb(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
)

out_dir = Path("/home/user/data_lm/notebooks")
out_dir.mkdir(exist_ok=True)
nbf.write(nb1, out_dir / "01_eda_telco_churn.ipynb")
nbf.write(nb2, out_dir / "02_eda_bank_marketing.ipynb")
nbf.write(nb3, out_dir / "03_modeling_and_comparison.ipynb")
print("notebooks written")
