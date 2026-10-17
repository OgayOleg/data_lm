# ============================================================================
# ML pipeline — воспроизведение одной командой
#   make all       — полный цикл: окружение -> данные -> пайплайн -> тесты
#   make pipeline  — обучение + оценка + отчёты
# Документация: README.md
# ============================================================================
PYTHON      ?= .venv/bin/python
PIP         ?= .venv/bin/pip
DATASET     ?= all

.DEFAULT_GOAL := help
.PHONY: help setup data pipeline pipeline-fast notebooks presentation presentation-defense test lint format check clean all

help:  ## Показать список команд
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:  ## Создать venv и установить зависимости
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:  ## Скачать датасеты (с проверкой SHA-256)
	$(PYTHON) -m src.download

pipeline:  ## Полный ML-пайплайн: обучение, CV, тюнинг, оценка, графики, отчёты
	$(PYTHON) -m src.pipeline --dataset $(DATASET)

pipeline-fast:  ## Быстрый прогон без подбора гиперпараметров
	$(PYTHON) -m src.pipeline --dataset $(DATASET) --skip-tuning --skip-leakage-experiment

notebooks:  ## Выполнить все Jupyter-ноутбуки (обновить выходы)
	$(PYTHON) scripts/execute_notebooks.py

presentation:  ## Собрать HTML-презентацию из метрик и графиков
	$(PYTHON) scripts/build_presentation.py

presentation-defense:  ## Собрать расширенную PowerPoint-презентацию для защиты (20 слайдов)
	$(PYTHON) scripts/build_extended_defense_presentation.py

test:  ## Модульные тесты
	$(PYTHON) -m pytest tests/ -v

lint:  ## Проверка стиля (flake8 + black --check)
	$(PYTHON) -m flake8 src tests scripts
	$(PYTHON) -m black --check src tests scripts

format:  ## Автоформатирование кода (black)
	$(PYTHON) -m black src tests scripts

check: lint test  ## Линтинг + тесты (используется в CI)

all: setup data pipeline check  ## Полный цикл с нуля
	@echo "Готово: результаты в reports/, модели в models/"

clean:  ## Удалить артефакты (данные и модели сохраняются)
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \;
	rm -rf reports/figures/*.png reports/metrics/*.csv reports/metrics/*.json
