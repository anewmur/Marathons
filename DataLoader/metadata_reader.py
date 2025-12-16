
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re

import pandas as pd

from DataLoader.race_title import RaceTitleParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExcelMeta:
    """
    Метаданные, извлеченные из Excel.

    Fields:
        race_id: нормализованный идентификатор трассы
        year: год забега
        distance_km: дистанция в км (или None, если не распознано)
    """

    race_id: str
    year: int
    distance_km: float | None


class ExcelMetadataReader:
    """
    Читает метаданные из первых строк Excel: (race_id, year, distance_km).
    """

    def __init__(self, meta_rows: int, title_parser: RaceTitleParser) -> None:
        """
        Input: число строк метаданных и парсер заголовка.
        Returns: ---
        Does: сохраняет зависимости.
        """
        self.meta_rows = meta_rows
        self.title_parser = title_parser

    def extract_year(self, date_text: str) -> int | None:
        """
        Input: строка с датой.
        Returns: год или None.
        Does: извлекает четырёхзначный год (20xx).
        """
        match = re.search(r"\b(20\d{2})\b", date_text)
        if match:
            return int(match.group(1))
        return None

    def extract_metadata(self, filepath: Path) -> ExcelMeta:
        """
        Input: путь к файлу.
        Returns: ExcelMeta(race_id, year, distance_km).
        Does:
            - читает первые строки
            - парсит distance_km из названия (если удаётся)
            - нормализует race_id через KNOWN_RACES
            - извлекает year из второй строки или из имени файла
        """
        meta_df = pd.read_excel(filepath, header=None, nrows=self.meta_rows)

        raw_race_title = meta_df.iloc[0, 0] if len(meta_df) > 0 else None
        distance_km = self.title_parser.parse_distance_km_from_title(raw_race_title)
        race_id = self.title_parser.normalize_race_name(raw_race_title, filepath)

        date_value = meta_df.iloc[1, 0] if len(meta_df) > 1 else None
        date_text = str(date_value) if pd.notna(date_value) else ""
        year = self.extract_year(date_text)

        if year is None:
            match = re.search(r"(20\d{2})", filepath.stem)
            if match:
                year = int(match.group(1))

        if year is None:
            raise ValueError(f"Не удалось определить год забега: {filepath}")

        if distance_km is None:
            logger.warning(
                "ExcelMetadataReader: дистанция не распознана в названии '%s' (файл %s), distance_km будет None",
                raw_race_title,
                filepath.name,
            )

        return ExcelMeta(race_id=race_id, year=year, distance_km=distance_km)
