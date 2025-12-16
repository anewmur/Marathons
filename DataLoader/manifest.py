from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import logging
import os

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ManifestFile:
    """
    Элемент манифеста для одного Excel-файла.

    Fields:
        relpath: относительный путь к файлу от корня директории
        size_bytes: размер файла в байтах
        mtime_ns: время модификации в наносекундах
    """

    relpath: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class Manifest:
    """
    Манифест директории для валидации parquet-кэша.

    Fields:
        files: список файлов с метаданными (в фиксированном порядке)
    """

    files: list[ManifestFile]


class ManifestBuilder:
    """
    Строит Manifest из списка Excel-файлов и сравнивает манифесты.
    """

    def build_manifest(self, root_dir: Path, excel_files: list[Path]) -> Manifest:
        """
        Input: корневая директория и список файлов.
        Returns: Manifest.
        Does: формирует манифест по (relpath, size_bytes, mtime_ns) для каждого файла.
        """
        items: list[ManifestFile] = []
        for file_path in excel_files:
            stat_result = file_path.stat()
            items.append(
                ManifestFile(
                    relpath=str(file_path.relative_to(root_dir)),
                    size_bytes=int(stat_result.st_size),
                    mtime_ns=int(stat_result.st_mtime_ns),
                )
            )
        return Manifest(files=items)

    def manifests_equal(self, left: Manifest, right: Manifest) -> bool:
        """
        Input: два манифеста.
        Returns: True если равны.
        Does: сравнивает списки файлов (порядок важен).
        """
        return left.files == right.files


class ManifestStore:
    """
    Читает и пишет Manifest в YAML/JSON.
    """

    def read_manifest(self, manifest_path: Path) -> Manifest | None:
        """
        Input: путь к manifest-файлу.
        Returns: Manifest или None.
        Does: читает YAML/JSON и валидирует структуру.
        """
        if not manifest_path.exists():
            return None

        suffix = manifest_path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                with open(manifest_path, "r", encoding="utf-8") as file_handle:
                    payload = yaml.safe_load(file_handle)
            elif suffix == ".json":
                with open(manifest_path, "r", encoding="utf-8") as file_handle:
                    payload = json.load(file_handle)
            else:
                logger.warning("ManifestStore: неизвестный формат manifest: %s", manifest_path.name)
                return None
        except Exception as error:
            logger.warning("ManifestStore: не удалось прочитать manifest %s: %s", manifest_path.name, error)
            return None

        files_payload = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files_payload, list):
            logger.warning("ManifestStore: manifest %s имеет неверную структуру (ожидался ключ files)", manifest_path.name)
            return None

        items: list[ManifestFile] = []
        for item in files_payload:
            if not isinstance(item, dict):
                logger.warning("ManifestStore: manifest %s содержит неверный элемент files", manifest_path.name)
                return None

            relpath = item.get("relpath")
            size_bytes = item.get("size_bytes")
            mtime_ns = item.get("mtime_ns")

            if not isinstance(relpath, str):
                logger.warning("ManifestStore: manifest %s содержит неверный relpath", manifest_path.name)
                return None
            if not isinstance(size_bytes, int) or isinstance(size_bytes, bool):
                logger.warning("ManifestStore: manifest %s содержит неверный size_bytes", manifest_path.name)
                return None
            if not isinstance(mtime_ns, int) or isinstance(mtime_ns, bool):
                logger.warning("ManifestStore: manifest %s содержит неверный mtime_ns", manifest_path.name)
                return None

            items.append(ManifestFile(relpath=relpath, size_bytes=size_bytes, mtime_ns=mtime_ns))

        return Manifest(files=items)

    def write_manifest_atomic(self, manifest_path: Path, manifest: Manifest) -> bool:
        """
        Input: путь и manifest.
        Returns: True если успешно.
        Does: атомарно записывает YAML/JSON.
        """
        suffix = manifest_path.suffix.lower()
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

        payload: dict[str, Any] = {
            "files": [
                {"relpath": item.relpath, "size_bytes": item.size_bytes, "mtime_ns": item.mtime_ns}
                for item in manifest.files
            ]
        }

        try:
            if suffix in (".yaml", ".yml"):
                with open(tmp_path, "w", encoding="utf-8") as file_handle:
                    yaml.safe_dump(payload, file_handle, allow_unicode=True, sort_keys=False)
            elif suffix == ".json":
                with open(tmp_path, "w", encoding="utf-8") as file_handle:
                    json.dump(payload, file_handle, ensure_ascii=False, indent=2)
            else:
                logger.warning("ManifestStore: manifest должен быть .yaml/.yml/.json: %s", manifest_path.name)
                return False

            os.replace(tmp_path, manifest_path)
            return True
        except Exception as error:
            logger.warning("ManifestStore: не удалось записать manifest %s: %s", manifest_path.name, error)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False
