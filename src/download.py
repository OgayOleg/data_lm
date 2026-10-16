"""Reproducible dataset download.

Datasets are public Kaggle datasets (see ``src/config.py`` for links).  In this
environment Kaggle itself is not reachable, so the download script fetches
byte-identical public mirrors from GitHub **pinned to a specific commit** and
verifies every file with a SHA-256 checksum.  Anyone can also place the
original Kaggle CSVs into ``data/raw/`` manually — the checksums will then
fail loudly instead of silently training on wrong data.

Usage::

    python -m src.download            # all datasets
    python -m src.download --dataset telco
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request

from src.config import DATASETS, RAW_DIR, DatasetSpec, ensure_directories

CHUNK_SIZE = 1 << 16  # 64 KiB


def sha256_of_file(path, chunk_size: int = CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of a file (streamed, memory-safe)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_github_content(url: str, destination) -> None:
    """Download a file through the GitHub contents API (raw media type)."""
    request = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw"})
    with urllib.request.urlopen(request, timeout=120) as response, open(destination, "wb") as fh:
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            fh.write(chunk)


def download_dataset(spec: DatasetSpec, force: bool = False, verbose: bool = True) -> bool:
    """Download one dataset and verify its checksum.

    Returns ``True`` when the file is present and valid (freshly downloaded or
    already cached), ``False`` otherwise.
    """
    ensure_directories()
    destination = RAW_DIR / spec.filename

    if destination.exists() and not force:
        digest = sha256_of_file(destination)
        if digest == spec.source_sha256:
            if verbose:
                print(f"[ok] {spec.key}: уже загружен ({destination.name}), checksum совпадает")
            return True
        print(f"[warn] {spec.key}: checksum не совпадает — повторная загрузка", file=sys.stderr)

    if verbose:
        print(f"[..] {spec.key}: скачиваю {spec.title} ...")
    _download_github_content(spec.source_url, destination)

    digest = sha256_of_file(destination)
    if digest != spec.source_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Контрольная сумма не совпала для {spec.key}: "
            f"ожидалось {spec.source_sha256}, получено {digest}"
        )
    if verbose:
        size_mb = destination.stat().st_size / 1e6
        print(f"[ok] {spec.key}: сохранён в {destination} ({size_mb:.2f} MB), checksum подтверждён")
    return True


def download_all(force: bool = False, keys: list[str] | None = None) -> dict[str, bool]:
    """Download all (or selected) datasets. Returns {dataset_key: success}."""
    selected = keys or list(DATASETS)
    unknown = set(selected) - set(DATASETS)
    if unknown:
        raise ValueError(f"Неизвестные датасеты: {sorted(unknown)}. Доступные: {sorted(DATASETS)}")
    return {key: download_dataset(DATASETS[key], force=force) for key in selected}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка датасетов проекта (с проверкой checksum).")
    parser.add_argument("--dataset", choices=sorted(DATASETS), help="загрузить только указанный датасет")
    parser.add_argument("--force", action="store_true", help="скачать заново, даже если файл существует")
    args = parser.parse_args(argv)

    keys = [args.dataset] if args.dataset else None
    results = download_all(force=args.force, keys=keys)
    failed = [key for key, ok in results.items() if not ok]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
