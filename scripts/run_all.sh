#!/usr/bin/env bash
# ============================================================================
# Полное воспроизведение проекта одной командой:
#   bash scripts/run_all.sh
#
# Шаги: окружение -> зависимости -> данные -> пайплайн -> тесты -> линтинг.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
echo "=== [1/6] Виртуальное окружение ==="
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
PYTHON=".venv/bin/python"

echo "=== [2/6] Зависимости ==="
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

echo "=== [3/6] Загрузка данных (checksum-проверка) ==="
"$PYTHON" -m src.download

echo "=== [4/6] ML-пайплайн (EDA-артефакты, CV, тюнинг, оценка) ==="
"$PYTHON" -m src.pipeline --dataset all

echo "=== [5/6] Тесты ==="
"$PYTHON" -m pytest tests/ -q

echo "=== [6/6] Линтинг ==="
"$PYTHON" -m flake8 src tests scripts
"$PYTHON" -m black --check src tests scripts

echo
echo "Готово! Результаты:"
echo "  - reports/metrics/   — таблицы метрик"
echo "  - reports/figures/   — графики"
echo "  - reports/REPORT.md  — итоговый отчёт"
echo "  - models/            — обученные модели"
