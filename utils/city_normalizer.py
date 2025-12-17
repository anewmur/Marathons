from __future__ import annotations

import logging
import re
import pandas as pd
from dictionaries import CITY_CANONICAL_MAP
logger = logging.getLogger(__name__)
from pathlib import Path
import time


def dump_unrecognized_cities(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    limit: int = 200,
) -> None:
    """
    Выгружает самые частые нераспознанные города
    (те, которые не схлопнулись через CITY_CANONICAL_MAP).

    Пишет файл вида:
        city | count
    """
    start_time = time.time()

    if "city" not in df.columns:
        logger.info("dump_unrecognized_cities: skipped (no city column)")
        return

    city_series = df["city"].dropna().astype("string")

    # все якорные города
    canonical_values = set(CITY_CANONICAL_MAP.values())

    # считаем только те, которые не якорные
    unrecognized = city_series[~city_series.isin(canonical_values)]

    counts = (
        unrecognized
        .value_counts()
        .head(limit)
    )

    lines = ["city | count"]
    for city, count in counts.items():
        lines.append(f"{city} | {count}")

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info(
        "dump_unrecognized_cities: dumped %s cities to %s (elapsed %.3fs)",
        len(counts),
        output_path,
        time.time() - start_time,
    )


class CityNormalizer:
    """
    Нормализация значения города.

    Стратегия:
    - очистка строки (пробелы, скобки, 'г.', пунктуация),
    - приведение к нижнему регистру,
    - сведение к каноническому городу через CITY_CANONICAL_MAP,
    - fallback: Title Case исходного значения.
    """

    _RE_LEADING_CITY = re.compile(r"^\s*г[\.\s,]+", re.IGNORECASE)
    _RE_TRAILING_CITY = re.compile(r"[\s,]+г\.?\s*$", re.IGNORECASE)
    _RE_PARENS = re.compile(r"\s*\([^)]*\)")
    _RE_MULTI_SPACE = re.compile(r"\s+")

    def normalize_city_value(self, raw_city: object) -> str | None:
        """
        Input:
            raw_city — произвольное значение из столбца city.

        Returns:
            Каноническое название города или None.

        Правила:
        - None / NA / пустая строка -> None
        - если после очистки ключ найден в CITY_CANONICAL_MAP -> вернуть якорный город
        - иначе вернуть Title Case очищенного значения
        """
        if raw_city is None or pd.isna(raw_city):
            return None

        text = str(raw_city).strip()
        if not text:
            return None

        lowered = text.lower()

        # удаляем "г.", "(...)", лишние пробелы
        lowered = self._RE_LEADING_CITY.sub("", lowered)
        lowered = self._RE_TRAILING_CITY.sub("", lowered)
        lowered = self._RE_PARENS.sub("", lowered)
        lowered = self._RE_MULTI_SPACE.sub(" ", lowered).strip()

        # область
        lowered = re.sub(r"\bобласть\b", "обл", lowered)
        lowered = re.sub(r"\bобл\.", "обл", lowered)
        lowered = re.sub(r"\bобл\b", "обл", lowered)

        # республика
        lowered = re.sub(r"\bреспублика\b", "респ", lowered)
        lowered = re.sub(r"\bресп\.", "респ", lowered)
        lowered = re.sub(r"\bресп\b", "респ", lowered)

        # поселок
        lowered = re.sub(r"\bпоселок\b", "пос", lowered)
        lowered = re.sub(r"\bп\.", "пос", lowered)
        lowered = re.sub(r"\bпос\.", "пос", lowered)
        lowered = re.sub(r"\bпос\b", "пос", lowered)

        # район
        lowered = re.sub(r"\bрайон\b", "р-н", lowered)
        lowered = re.sub(r"\bр-н\.", "р-н", lowered)
        lowered = re.sub(r"\bр-н\b", "р-н", lowered)
        # республика
        lowered = re.sub(r"\bреспублика\b", "респ", lowered)
        lowered = re.sub(r"\bресп\.", "респ", lowered)
        lowered = re.sub(r"\bресп\b", "респ", lowered)

        # поселок
        lowered = re.sub(r"\bпоселок\b", "пос", lowered)
        lowered = re.sub(r"\bп\.", "пос", lowered)
        lowered = re.sub(r"\bпос\.", "пос", lowered)
        lowered = re.sub(r"\bпос\b", "пос", lowered)

        # район
        lowered = re.sub(r"\bрайон\b", "р-н", lowered)
        lowered = re.sub(r"\bр-н\.", "р-н", lowered)
        lowered = re.sub(r"\bр-н\b", "р-н", lowered)
        # республика
        lowered = re.sub(r"\bреспублика\b", "респ", lowered)
        lowered = re.sub(r"\bресп\.", "респ", lowered)
        lowered = re.sub(r"\bресп\b", "респ", lowered)

        # поселок
        lowered = re.sub(r"\bпоселок\b", "пос", lowered)
        lowered = re.sub(r"\bп\.", "пос", lowered)
        lowered = re.sub(r"\bпос\.", "пос", lowered)
        lowered = re.sub(r"\bпос\b", "пос", lowered)

        # район
        lowered = re.sub(r"\bрайон\b", "р-н", lowered)
        lowered = re.sub(r"\bр-н\.", "р-н", lowered)
        lowered = re.sub(r"\bр-н\b", "р-н", lowered)

        if not lowered:
            return None

        # if lowered == 'минская обл':
        #     print('!')

        if lowered in CITY_CANONICAL_MAP:
            return CITY_CANONICAL_MAP[lowered]

        # fallback: аккуратный Title Case
        return lowered.title()


