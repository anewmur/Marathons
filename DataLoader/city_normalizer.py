
from __future__ import annotations

import logging
import re

import pandas as pd

from dictionaries import CLUSTER_CITY_CANONICAL_MAP, REGION_NORMALIZATION_MAP

logger = logging.getLogger(__name__)


class CityNormalizer:
    """
    Нормализация значения города.
    """

    def normalize_city_value(self, raw_city: object) -> str | None:
        """
        Input: сырое значение города.
        Returns: нормализованное название или None.
        Does: чистит строку и применяет словари нормализации.
        """
        if raw_city is None or pd.isna(raw_city):
            return None

        text = str(raw_city).strip()
        if not text:
            return None

        lowered = text.lower()
        lowered = re.sub(r"^\s*г[\.\s,]+", "", lowered)
        lowered = re.sub(r"[\s,]+г\.?\s*$", "", lowered)
        lowered = re.sub(r"\s*\([^)]*\)", "", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()

        if not lowered:
            return None

        if lowered in CLUSTER_CITY_CANONICAL_MAP:
            return CLUSTER_CITY_CANONICAL_MAP[lowered]

        if lowered in REGION_NORMALIZATION_MAP:
            return REGION_NORMALIZATION_MAP[lowered]

        return lowered.title()
