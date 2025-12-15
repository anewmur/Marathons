"""
Загрузка данных из Excel файлов с результатами марафонов.

Класс DataLoader извлекает сырые данные из Excel файлов специального формата.
Метаданные (название трассы, дата) в первых строках, таблица результатов со строки 6.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
import json
import os
import yaml

import pandas as pd
from utils import parse_time_to_seconds, parse_gender_from_category, parse_status
from dictionaries import CLUSTER_CITY_CANONICAL_MAP, REGION_NORMALIZATION_MAP, KNOWN_RACES
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Загрузчик данных марафонов из Excel файлов.

    Извлекает сырые данные: runner_id, race_id, year, gender, age,
    time_seconds, status, city, bib_number.
    Не выполняет предобработку, только парсинг файлов.

    Attributes:
        loaded_files: Список путей к загруженным файлам
        errors: Словарь ошибок для файлов с проблемами
    """

    MIN_AGE = 10
    MAX_AGE = 100
    HEADER_ROW = 5  # Строка с заголовками (0-based)


    def __init__(self) -> None:
        self.loaded_files: list[Path] = []
        self.errors: dict[str, str] = {}

    def load_directory(
        self,
        directory: str | Path,
        pattern: str = "*.xls*",
    ) -> pd.DataFrame:
        """
        Загрузить все Excel файлы из директории.

        Args:
            directory: Путь к директории
            pattern: Паттерн для поиска файлов

        Returns:
            Объединённый DataFrame
        """


        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory_path}")

        if not directory_path.is_dir():
            raise ValueError(f"Не является директорией: {directory_path}")

        files = list(directory_path.glob(pattern))

        if not files:
            logger.warning(f"Не найдено файлов по паттерну {pattern}")
            return pd.DataFrame()

        all_results: list[pd.DataFrame] = []

        for filepath in sorted(files):
            try:
                dataframe = self.load_file(filepath)
                if not dataframe.empty:
                    all_results.append(dataframe)
                    logger.debug(
                        f"Загружен: {filepath.name} ({len(dataframe)} записей)"
                    )
                else:
                    logger.warning(f"Пропущен (пустой): {filepath.name}")
            except Exception as error:
                self.errors[str(filepath)] = str(error)
                logger.warning(f"Ошибка в {filepath.name}: {error}")

        if not all_results:
            return pd.DataFrame()

        return pd.concat(all_results, ignore_index=True)

    def load_directory_cached(
            self,
            directory: str | Path,
            pattern: str = "*.xls*",
            manifest_name: str = "list_excel.yaml",
            cache_name: str = "data_cache.parquet",
    ) -> pd.DataFrame:
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Не является директорией: {directory_path}")

        excel_files = sorted(directory_path.glob(pattern))
        if not excel_files:
            logger.warning(f"Не найдено файлов по паттерну {pattern}")
            return pd.DataFrame()

        manifest_path = directory_path / manifest_name
        cache_path = directory_path / cache_name

        current_manifest = self._build_excel_manifest(directory_path, excel_files)
        saved_manifest = self._read_manifest_if_exists(manifest_path)

        if self._manifests_equal(saved_manifest, current_manifest) and cache_path.exists():
            cached = self._read_cache(cache_path)
            if cached is not None:
                logger.info("DataLoader: загружен кэш %s", cache_path.name)
                return cached
            logger.warning("DataLoader: кэш не прочитан, перезагружаю Excel")

        dataframe = self.load_directory(directory_path, pattern=pattern)

        if not dataframe.empty:
            if self._write_cache(cache_path, dataframe):
                self._write_manifest(manifest_path, current_manifest)
                logger.info("DataLoader: кэш и manifest обновлены")

        return dataframe

    def _build_excel_manifest(self, root_dir: Path, excel_files: list[Path]) -> dict[str, object]:
        files_payload: list[dict[str, object]] = []
        for filepath in excel_files:
            stat_result = filepath.stat()
            relpath = str(filepath.relative_to(root_dir))
            files_payload.append(
                {
                    "relpath": relpath,
                    "size_bytes": int(stat_result.st_size),
                    "mtime_ns": int(stat_result.st_mtime_ns),
                }
            )
        return {"files": files_payload}

    def _read_manifest_if_exists(self, manifest_path: Path) -> dict[str, object] | None:
        if not manifest_path.exists():
            return None

        suffix = manifest_path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                with open(manifest_path, "r", encoding="utf-8") as file_obj:
                    loaded = yaml.safe_load(file_obj)
                    return loaded if isinstance(loaded, dict) else None
            if suffix == ".json":
                with open(manifest_path, "r", encoding="utf-8") as file_obj:
                    loaded = json.load(file_obj)
                    return loaded if isinstance(loaded, dict) else None
            logger.warning("DataLoader: неизвестный формат manifest: %s", manifest_path.name)
            return None
        except Exception as error:
            logger.warning("DataLoader: не удалось прочитать manifest %s: %s", manifest_path.name, error)
            return None

    def _manifests_equal(
            self,
            saved_manifest: dict[str, object] | None,
            current_manifest: dict[str, object],
    ) -> bool:
        if saved_manifest is None:
            return False
        saved_files = saved_manifest.get("files")
        current_files = current_manifest.get("files")
        if not isinstance(saved_files, list) or not isinstance(current_files, list):
            return False
        return saved_files == current_files

    def _read_cache(self, cache_path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_parquet(cache_path)
        except Exception as error:
            logger.warning("DataLoader: не удалось прочитать кэш %s: %s", cache_path.name, error)
            return None

    def _write_cache(self, cache_path: Path, dataframe: pd.DataFrame) -> bool:
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            dataframe.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, cache_path)
            return True
        except Exception as error:
            logger.warning("DataLoader: не удалось записать кэш %s: %s", cache_path.name, error)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            return False

    def _write_manifest(self, manifest_path: Path, manifest: dict[str, object]) -> None:
        tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        suffix = manifest_path.suffix.lower()

        if suffix not in (".yaml", ".yml", ".json"):
            logger.warning("DataLoader: manifest должен быть .yaml/.yml/.json: %s", manifest_path.name)
            return

        try:
            if suffix in (".yaml", ".yml"):
                with open(tmp_path, "w", encoding="utf-8") as file_obj:
                    yaml.safe_dump(manifest, file_obj, allow_unicode=True, sort_keys=False)
            else:
                with open(tmp_path, "w", encoding="utf-8") as file_obj:
                    json.dump(manifest, file_obj, ensure_ascii=False, indent=2)

            os.replace(tmp_path, manifest_path)
        except Exception as error:
            logger.warning("DataLoader: не удалось записать manifest %s: %s", manifest_path.name, error)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

    def load_file(self, filepath: str | Path) -> pd.DataFrame:
        """
        Загрузить один Excel файл с результатами забега.

        Args:
            filepath: Путь к Excel файлу

        Returns:
            DataFrame с колонками: runner_id, race_id, year, gender,
            age, time_seconds, status, city, bib_number
        """
        path_object = Path(filepath)

        if not path_object.exists():
            raise FileNotFoundError(f"Файл не найден: {path_object}")

        race_id, year = self._extract_metadata(path_object)

        dataframe = pd.read_excel(path_object, header=self.HEADER_ROW)

        self._validate_columns(dataframe, path_object)

        results = self._parse_results(dataframe, race_id, year)

        result_dataframe = pd.DataFrame(results)

        if not result_dataframe.empty:
            result_dataframe["year"] = result_dataframe["year"].astype(int)
            result_dataframe["age"] = result_dataframe["age"].astype(int)

        self.loaded_files.append(path_object)

        return result_dataframe

    def _extract_metadata(self, filepath: Path) -> tuple[str, int]:
        """
        Извлечь название трассы и год из первых строк файла.
        """
        meta_dataframe = pd.read_excel(filepath, header=None, nrows=3)

        raw_race_name = meta_dataframe.iloc[0, 0] if len(meta_dataframe) > 0 else None
        race_id = self._normalize_race_name(raw_race_name, filepath)

        date_value = meta_dataframe.iloc[1, 0] if len(meta_dataframe) > 1 else None
        date_string = str(date_value) if pd.notna(date_value) else ""
        year = self._extract_year(date_string)

        if year is None:
            match = re.search(r"(20\d{2})", filepath.stem)
            if match:
                year = int(match.group(1))

        if year is None:
            raise ValueError(f"Не удалось определить год забега: {filepath}")

        return race_id, year

    def _normalize_race_name(self, raw_name: str | None, filepath: Path) -> str:
        """
        Нормализовать название забега через KNOWN_RACES.
        """
        if pd.isna(raw_name) or not isinstance(raw_name, str):
            raise ValueError(
                f"Пустое название трассы в файле {filepath}. "
                f"Проверьте строку 1 в Excel."
            )

        name_lower = raw_name.strip().lower()

        for pattern, normalized in KNOWN_RACES.items():
            if pattern in name_lower:
                return normalized

        raise ValueError(
            f"Неизвестная трасса: '{raw_name}' в файле {filepath}. "
            f"Добавьте её в KNOWN_RACES в data_loader.py"
        )

    def _validate_columns(self, dataframe: pd.DataFrame, filepath: Path) -> None:
        """
        Проверить наличие обязательных колонок.
        """
        required_columns = ["Фамилия", "Имя", "Возраст", "Категория", "Результат"]
        missing_columns = [
            column for column in required_columns if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Отсутствуют колонки {missing_columns} в файле {filepath}"
            )

    def _extract_year(self, date_string: str) -> int | None:
        """
        Извлечь год из строки с датой.
        """
        match = re.search(r"\b(20\d{2})\b", date_string)
        if match:
            return int(match.group(1))
        return None

    def _parse_results(
        self,
        dataframe: pd.DataFrame,
        race_id: str,
        year: int,
    ) -> list[dict[str, object]]:
        """
        Обработать строки таблицы результатов.
        """
        results: list[dict[str, object]] = []

        has_city_column = "Город" in dataframe.columns

        bib_column: str | None = None
        for candidate in ("Номер", "номер", "№", "№ п/п", "№п/п"):
            if candidate in dataframe.columns:
                bib_column = candidate
                break

        for _, row in dataframe.iterrows():
            if pd.isna(row["Фамилия"]):
                continue

            surname = str(row["Фамилия"]).strip()
            name = str(row["Имя"]).strip() if pd.notna(row["Имя"]) else ""
            runner_id = f"{surname}_{name}"

            gender = parse_gender_from_category(row["Категория"])
            if gender is None:
                continue

            try:
                age = int(row["Возраст"]) if pd.notna(row["Возраст"]) else None
            except (ValueError, TypeError):
                age = None

            if age is None or age < self.MIN_AGE or age > self.MAX_AGE:
                continue

            time_seconds = parse_time_to_seconds(str(row["Результат"]))
            raw_status_value = row.get("Статус", None)
            status = parse_status(raw_status_value, time_seconds)

            city_value = None
            if has_city_column:
                raw_city = row["Город"]
                city_value = self._normalize_city_value(raw_city)

            bib_number: int | None = None
            if bib_column is not None:
                raw_bib = row[bib_column]
                if pd.notna(raw_bib):
                    try:
                        bib_number = int(raw_bib)
                    except (ValueError, TypeError):
                        bib_number = None

            results.append(
                {
                    "runner_id": runner_id,
                    "race_id": race_id,
                    "year": year,
                    "gender": gender,
                    "age": age,
                    "time_seconds": time_seconds if status == "OK" else None,
                    "status": status,
                    "city": city_value,
                    "bib_number": bib_number,
                }
            )

        return results

    def get_summary(self) -> dict[str, object]:
        """
        Получить сводку о загруженных файлах.
        """
        return {
            "loaded_count": len(self.loaded_files),
            "error_count": len(self.errors),
            "errors": self.errors.copy(),
        }

    @staticmethod
    def _normalize_city_value(raw_city: object) -> str | None:
        if raw_city is None or pd.isna(raw_city):
            return None

        text = str(raw_city).strip()
        if not text:
            return None

        lower = text.lower()

        # Убираем 'г', 'г.', 'г,' в начале
        lower = re.sub(r'^\s*г[\.\s,]+', '', lower)
        # Убираем 'г.' в конце
        lower = re.sub(r'[\s,]+г\.?\s*$', '', lower)
        # Убираем скобки
        lower = re.sub(r'\s*\([^)]*\)', '', lower)
        # Схлопываем пробелы
        lower = re.sub(r'\s+', ' ', lower).strip()

        if not lower:
            return None

        # Глобальная нормализация (опечатки, варианты областей и т.д.)
        if lower in CLUSTER_CITY_CANONICAL_MAP:
            return CLUSTER_CITY_CANONICAL_MAP[lower]

        if lower in REGION_NORMALIZATION_MAP:
            return REGION_NORMALIZATION_MAP[lower]

        # По умолчанию — просто нормальный кейс
        return lower.title()

