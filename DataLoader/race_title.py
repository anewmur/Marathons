
from __future__ import annotations

from pathlib import Path
import logging
import re

import pandas as pd

from dictionaries import KNOWN_RACES

logger = logging.getLogger(__name__)


class RaceTitleParser:
    """
    Парсинг и нормализация названия трассы (первая строка метаданных).
    """

    def parse_distance_km_from_title(self, title_text: str | None) -> float | None:
        """
        Input: сырая строка названия забега.
        Returns: дистанция в км (float) или None.
        Does: ищет число перед 'км'/'km', поддерживает запятую как разделитель.
        """
        if not title_text:
            return None

        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:км|km)\b", title_text, re.IGNORECASE)
        if match:
            raw_value = match.group(1).replace(",", ".")
            try:
                return float(raw_value)
            except ValueError:
                return None
        return None

    def clean_title_for_race_normalization(self, raw_title: str) -> str:
        """
        Input: сырая строка названия.
        Returns: очищенная строка.
        Does: удаляет год, дистанцию и схлопывает пробелы.
        """
        text = raw_title.strip()
        text = re.sub(r"\b20\d{2}\b", "", text)
        text = re.sub(r"\d+(?:[.,]\d+)?\s*(?:км|km)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def normalize_race_name(self, raw_name: object, filepath: Path) -> str:
        """
        Input: сырое название трассы и путь к файлу.
        Returns: нормализованный race_id.
        Does: очищает строку и нормализует через KNOWN_RACES.
        """
        if raw_name is None or pd.isna(raw_name) or not isinstance(raw_name, str):
            raise ValueError(
                f"Пустое название трассы в файле {filepath}. "
                f"Проверьте первую строку Excel."
            )

        cleaned_title = self.clean_title_for_race_normalization(raw_name)
        lowered = cleaned_title.lower()

        for pattern, normalized in KNOWN_RACES.items():
            if pattern in lowered:
                return normalized

        raise ValueError(
            f"Неизвестная трасса: '{raw_name}' (очищено: '{cleaned_title}') в файле {filepath}. "
            f"Добавьте её в KNOWN_RACES."
        )
