# data_loader.py
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from DataLoader.manifest import ManifestBuilder, ManifestStore
from DataLoader.parquet_store import ParquetStore
from DataLoader.excel_parsing import (
    ExcelMetadataReader,
    RaceTitleParser,
    ResultsTableParser,
    ResultsTableSchema,
    ExcelMeta,
)

logger = logging.getLogger(__name__)

REQUIRED_TABLE_COLUMNS: list[str] = ["Фамилия", "Имя", "Возраст", "Категория", "Результат"]

OUTPUT_COLUMNS: list[str] = [
    "surname",
    "name",
    "race_id",
    "year",
    "gender",
    "age",
    "time_seconds",
    "status",
    "city",
    "bib_number",
    "distance_km",
]


class DirectoryCache:
    """
    Управляет manifest и parquet-кэшем для директории.

    Смысл:
        - кэш валиден только если манифест текущих Excel-файлов совпал с сохранённым
        - при валидном кэше читается parquet
        - при невалидном кэше читаются Excel и кэш обновляется
    """

    def __init__(
        self,
        manifest_name: str,
        cache_name: str,
        output_columns: list[str],
        manifest_builder: ManifestBuilder,
        manifest_store: ManifestStore,
        parquet_store: ParquetStore,
    ) -> None:
        """
        Input: имена файлов manifest и кэша, выходные колонки, зависимости.
        Returns: ---
        Does: сохраняет зависимости и контракт кэша.
        """
        self._manifest_name = manifest_name
        self._cache_name = cache_name
        self._output_columns = output_columns
        self._manifest_builder = manifest_builder
        self._manifest_store = manifest_store
        self._parquet_store = parquet_store

    def try_read_cache(self, directory_path: Path, excel_files: list[Path]) -> pd.DataFrame | None:
        """
        Input: путь к директории и список Excel-файлов.
        Returns: кэшированный DataFrame или None.
        Does: проверяет манифест, файл и схему, читает parquet при совпадении.
        """
        manifest_path = directory_path / self._manifest_name
        cache_path = directory_path / self._cache_name

        current_manifest = self._manifest_builder.build_manifest(directory_path, excel_files)
        saved_manifest = self._manifest_store.read_manifest(manifest_path)

        if saved_manifest is None or not self._manifest_builder.manifests_equal(saved_manifest, current_manifest):
            logger.debug("DataLoader: manifest не совпал или отсутствует — кэш игнорируется")
            return None

        if not cache_path.exists():
            logger.debug("DataLoader: parquet отсутствует — кэш игнорируется")
            return None

        cached = self._parquet_store.read_parquet(cache_path)
        if cached is None:
            logger.debug("DataLoader: parquet не прочитан — кэш игнорируется")
            return None

        missing_cols = [col for col in self._output_columns if col not in cached.columns]
        if missing_cols:
            logger.debug("DataLoader: в кэше отсутствуют колонки %s — кэш игнорируется", missing_cols)
            return None

        logger.info("DataLoader: загружен кэш %s", cache_path.name)
        return cached[self._output_columns]  # гарантируем порядок

    def write_cache(self, directory_path: Path, excel_files: list[Path], dataframe: pd.DataFrame) -> None:
        """
        Input: путь к директории, список Excel-файлов, DataFrame.
        Returns: ---
        Does: записывает parquet + manifest (атомарно).
        """
        if dataframe.empty:
            return

        manifest_path = directory_path / self._manifest_name
        cache_path = directory_path / self._cache_name

        current_manifest = self._manifest_builder.build_manifest(directory_path, excel_files)

        ok_parquet = self._parquet_store.write_parquet_atomic(cache_path, dataframe[self._output_columns])
        if not ok_parquet:
            return

        ok_manifest = self._manifest_store.write_manifest_atomic(manifest_path, current_manifest)
        if not ok_manifest:
            return

        logger.info("DataLoader: кэш и manifest обновлены")


class DataLoader:
    """
    Загрузчик данных марафонов из Excel файлов.

    Контракт выхода (колонки в итоговом DataFrame):
        surname, name, race_id, year, gender, age, time_seconds, status,
        city, bib_number, distance_km

    Не выполняет предобработку и не присваивает идентификатор человека.
    Идентификатор человека (person_id/runner_id) появляется только после disambiguation.
    Дистанция парсится из названия забега (первая строка метаданных).
    """

    MIN_AGE = 10
    MAX_AGE = 100
    HEADER_ROW = 5
    META_ROWS = 3

    def __init__(self) -> None:
        """
        Input: ---
        Returns: ---
        Does: инициализирует загрузчик, готовит зависимости, кэш и контейнер ошибок.
        """
        self.loaded_files: list[Path] = []
        self.errors: dict[str, str] = {}

        self._title_parser = RaceTitleParser()
        self._metadata_reader = ExcelMetadataReader(meta_rows=self.META_ROWS, title_parser=self._title_parser)
        self._table_schema = ResultsTableSchema(required_columns=REQUIRED_TABLE_COLUMNS)
        self._table_parser = ResultsTableParser(
            min_age=self.MIN_AGE,
            max_age=self.MAX_AGE,
            output_columns=OUTPUT_COLUMNS
        )
        self._cache = DirectoryCache(
            manifest_name="list_excel.yaml",
            cache_name="data_cache.parquet",
            output_columns=OUTPUT_COLUMNS,
            manifest_builder=ManifestBuilder(),
            manifest_store=ManifestStore(),
            parquet_store=ParquetStore(),
        )

    def load(self, path: str | Path, pattern: str = "*.xls*") -> pd.DataFrame:
        """
        Загружает список Excel-файлов без кэша.
        Input: список файлов (если порядок не гарантирован, будет отсортирован по имени).
        Returns: объединённый DataFrame.
        Does: последовательно парсит файлы, собирает ошибки, объединяет результаты.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Путь не существует: {path_obj}")

        if path_obj.is_file():
            return self.load_file(path_obj)

        return self.load_directory_cached(path_obj, pattern=pattern)

    def load_directory_cached(self, directory: str | Path, pattern: str = "*.xls*") -> pd.DataFrame:
        """
        Загружает директорию с использованием кэша.
        Input: путь к директории, паттерн файлов.
        Returns: DataFrame (из кэша или свежепарсенный).
        Does: проверяет/обновляет кэш, делегирует парсинг при необходимости.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Директория не найдена: {directory_path}")
        if not directory_path.is_dir():
            raise ValueError(f"Не является директорией: {directory_path}")

        excel_files = sorted(directory_path.glob(pattern))
        if not excel_files:
            logger.warning("DataLoader: не найдено файлов по паттерну %s", pattern)
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        cached = self._cache.try_read_cache(directory_path, excel_files)
        if cached is not None:
            return cached

        parsed = self.load_directory(excel_files)
        parsed = parsed.reindex(columns=OUTPUT_COLUMNS)

        self._cache.write_cache(directory_path, excel_files, parsed)
        return parsed

    def load_directory(self, excel_files: list[Path]) -> pd.DataFrame:
        """
        Загружает список Excel-файлов без кэша.
        Input: список файлов.
        Returns: объединённый DataFrame.
        Does: последовательно парсит файлы, собирает ошибки, объединяет результаты.
        """
        all_frames: list[pd.DataFrame] = []
        for file_path in excel_files:
            try:
                df_one = self.load_file(file_path)
                df_one = df_one.reindex(columns=OUTPUT_COLUMNS)

                if df_one.empty:
                    logger.warning("DataLoader: пропущен (пустой): %s", file_path.name)
                    continue

                all_frames.append(df_one)
                logger.debug("DataLoader: загружен: %s (%d записей)", file_path.name, len(df_one))
            except Exception as error:
                self.errors[str(file_path)] = str(error)
                logger.warning("DataLoader: ошибка в %s: %s", file_path.name, error)

        if not all_frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        return pd.concat(all_frames, ignore_index=True)

    def load_file(self, filepath: str | Path) -> pd.DataFrame:
        """
        Загружает один Excel-файл.
        Input: путь к файлу.
        Returns: DataFrame с контрактом DataLoader (включая distance_km).
        Does:
            1) извлекает метаданные (race_id, year, distance_km),
            2) читает таблицу результатов,
            3) валидирует колонки,
            4) парсит строки в контракт DataLoader.
        """
        path_obj = Path(filepath)
        if not path_obj.exists():
            raise FileNotFoundError(f"Файл не найден: {path_obj}")

        meta: ExcelMeta = self._metadata_reader.extract_metadata(path_obj)

        table_df = pd.read_excel(path_obj, header=self.HEADER_ROW)
        self._table_schema.validate_table_columns(table_df, path_obj)

        result_df = self._table_parser.parse_results_table(
            table_df=table_df,
            race_id=meta.race_id,
            year=meta.year,
            distance_km=meta.distance_km,
        )

        self.loaded_files.append(path_obj)
        return result_df

    def get_summary(self) -> dict[str, object]:
        """
        Input: ---
        Returns: сводка по загрузке (количество файлов и ошибки).
        Does: формирует отчёт по загруженным файлам.
        """
        return {
            "loaded_count": len(self.loaded_files),
            "error_count": len(self.errors),
            "errors": self.errors.copy(),
        }