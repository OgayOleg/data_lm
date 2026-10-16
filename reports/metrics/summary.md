# Сводка результатов пайплайна

Сгенерировано автоматически: `python -m src.pipeline`.

## Bank Marketing (bank-additional-full) (`bank`)

- Строк: **41 188** (признаков в матрице: 24)
- Положительный класс: **11.3%**
- Лучшая модель-кандидат: **xgboost**
- Финальная модель: **xgboost_tuned**, порог решения 0.19 (подобран по F1 на out-of-fold)

| model | accuracy | precision | recall | f1 | roc_auc | avg_precision |
|---|---|---|---|---|---|---|
| xgboost | 0.8492 | 0.3966 | 0.6487 | 0.4922 | 0.8096 | 0.4870 |
| lightgbm | 0.8534 | 0.4054 | 0.6466 | 0.4983 | 0.8089 | 0.4903 |
| random_forest | 0.8654 | 0.4310 | 0.6088 | 0.5047 | 0.8029 | 0.4843 |
| logreg | 0.8352 | 0.3681 | 0.6466 | 0.4691 | 0.8009 | 0.4561 |
| xgboost_tuned | 0.9014 | 0.6895 | 0.2274 | 0.3420 | 0.8170 | 0.4916 |
| xgboost_tuned@0.19 | 0.8784 | 0.4690 | 0.6034 | 0.5278 | 0.8170 | 0.4916 |

**Топ-признаки лучшей модели (gain):**

- `nr.employed` — 0.0475
- `emp.var.rate` — 0.0328
- `euribor3m` — 0.0118
- `contact` — 0.0107
- `cons.price.idx` — 0.0087
- `cons.conf.idx` — 0.0058
- `campaign` — 0.0054
- `macro_stress` — 0.0038
- `pdays_recency` — 0.0031
- `month` — 0.0027

## Telco Customer Churn (`telco`)

- Строк: **7 043** (признаков в матрице: 26)
- Положительный класс: **26.5%**
- Лучшая модель-кандидат: **logreg**
- Финальная модель: **logreg_tuned**, порог решения 0.58 (подобран по F1 на out-of-fold)

| model | accuracy | precision | recall | f1 | roc_auc | avg_precision |
|---|---|---|---|---|---|---|
| logreg | 0.7339 | 0.4992 | 0.7995 | 0.6146 | 0.8469 | 0.6675 |
| xgboost | 0.7594 | 0.5342 | 0.7299 | 0.6169 | 0.8351 | 0.6475 |
| random_forest | 0.7580 | 0.5340 | 0.6925 | 0.6030 | 0.8339 | 0.6442 |
| lightgbm | 0.7757 | 0.5628 | 0.6952 | 0.6220 | 0.8277 | 0.6276 |
| logreg_tuned | 0.7324 | 0.4975 | 0.7914 | 0.6109 | 0.8476 | 0.6697 |
| logreg_tuned@0.58 | 0.7743 | 0.5558 | 0.7460 | 0.6370 | 0.8476 | 0.6697 |

**Топ-признаки лучшей модели (gain):**

- `Contract` — 0.0381
- `tenure` — 0.0251
- `tenure_group` — 0.0105
- `InternetService` — 0.0087
- `is_new_customer` — 0.0063
- `TotalCharges` — 0.0031
- `MultipleLines` — 0.0028
- `TechSupport` — 0.0024
- `OnlineSecurity` — 0.0023
- `is_electronic_check` — 0.0016
