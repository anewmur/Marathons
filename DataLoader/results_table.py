
from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from utils.utils import parse_gender_from_category, parse_status, parse_time_to_seconds

logger = logging.getLogger(__name__)


class ResultsTableSchema:
    """
    Проверка схемы таблицы результатов (наличие колонок).
    """

    def __init__(self, required_columns: list[str]) -> None:
        """
        Input: список обязательных колонок.
        Returns: ---
        Does: сохраняет контракт таблицы.
        """
        self.required_columns = tuple(required_columns)

    def validate_table_columns(self, table_df: pd.DataFrame, filepath: Path) -> None:
        """
        Input: таблица результатов и путь к файлу.
        Returns: ---
        Does: проверяет наличие обязательных колонок.
        """
        missing_columns = [col for col in self.required_columns if col not in table_df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют колонки {missing_columns} в файле {filepath}")


class ResultsTableParser:
    """
    Парсит таблицу результатов в DataLoader.
    """

    def __init__(
        self,
        min_age: int,
        max_age: int,
        output_columns: list[str],
    ) -> None:
        """
        Input: границы возраста, выходные колонки, нормализатор города.
        Returns: ---
        Does: сохраняет настройки парсинга.
        """
        self.min_age = min_age
        self.max_age = max_age
        self.output_columns = list(output_columns)

    def find_bib_column(self, table_df: pd.DataFrame) -> str | None:
        """
        Input: таблица результатов.
        Returns: имя колонки с номером или None.
        Does: ищет среди известных вариантов.
        """
        candidates = ("Номер", "номер", "№", "№ п/п", "№п/п")
        for candidate in candidates:
            if candidate in table_df.columns:
                return candidate
        return None

    def parse_results_table(
        self,
        table_df: pd.DataFrame,
        race_id: str,
        year: int,
        distance_km: float | None,
    ) -> pd.DataFrame:
        """
        Input: таблица результатов, race_id, year, distance_km.
        Returns: DataFrame с контрактом DataLoader.
        Does: парсит строки, фильтрует по полу/возрасту, добавляет distance_km.
        """
        bib_column = self.find_bib_column(table_df)

        records: list[dict[str, object]] = []

        skipped_empty_surname = 0
        skipped_no_gender = 0
        skipped_bad_age = 0
        kept = 0

        for _, row in table_df.iterrows():
            raw_surname = row["Фамилия"]
            if pd.isna(raw_surname):
                skipped_empty_surname += 1
                continue

            surname = str(raw_surname).strip()
            if not surname:
                skipped_empty_surname += 1
                continue

            name = str(row["Имя"]).strip() if pd.notna(row["Имя"]) else ""

            gender = parse_gender_from_category(row["Категория"])
            if gender is None:
                skipped_no_gender += 1
                continue

            try:
                age_value = int(row["Возраст"]) if pd.notna(row["Возраст"]) else None
            except (ValueError, TypeError):
                age_value = None

            if age_value is None or age_value < self.min_age or age_value > self.max_age:
                skipped_bad_age += 1
                continue

            time_seconds = parse_time_to_seconds(str(row["Результат"]))
            status = parse_status(row.get("Статус", None), time_seconds)

            if status == "OK" and time_seconds is None:
                status = "BAD_TIME"

            bib_number: int | None = None
            if bib_column is not None and pd.notna(row[bib_column]):
                try:
                    bib_number = int(row[bib_column])
                except (ValueError, TypeError):
                    bib_number = None

            records.append(
                {
                    "surname": surname,
                    "name": name,
                    "race_id": race_id,
                    "year": year,
                    "gender": gender,
                    "age": int(age_value),
                    "time_seconds": time_seconds if status == "OK" else None,
                    "status": status,
                    "city": city_value,
                    "bib_number": bib_number,
                    "distance_km": distance_km,
                }
            )
            kept += 1

        logger.debug(
            "ResultsTableParser: input %d → kept %d (skipped: empty_surname=%d, no_gender=%d, bad_age=%d)",
            len(table_df),
            kept,
            skipped_empty_surname,
            skipped_no_gender,
            skipped_bad_age,
        )

        result_df = pd.DataFrame.from_records(records)
        return result_df.reindex(columns=self.output_columns)
