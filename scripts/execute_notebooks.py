#!/usr/bin/env python
"""Execute all project notebooks in place (refreshes their outputs).

Usage::

    python scripts/execute_notebooks.py [--timeout 1800]

Notebooks are executed from the project root, so ``import src`` works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def execute_notebook(path: Path, timeout: int = 1800) -> None:
    """Run one notebook in place, keeping its outputs."""
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
        allow_errors=False,
    )
    client.execute()
    nbformat.write(notebook, path)
    print(f"[ok] выполнен: {path.relative_to(PROJECT_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Выполнить все ноутбуки проекта.")
    parser.add_argument("--timeout", type=int, default=1800, help="таймаут на ячейку, сек")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="выполнить только указанные ноутбуки (по имени файла)",
    )
    args = parser.parse_args(argv)

    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if args.only:
        notebooks = [p for p in notebooks if p.name in set(args.only)]

    if not notebooks:
        print("Ноутбуки не найдены", file=sys.stderr)
        return 1

    for path in notebooks:
        execute_notebook(path, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
