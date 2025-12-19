"""
Построение базовых медиан по возрасту для прогнозирования.

AgeReferenceBuilder строит медианы времён по группам (gender, age).
Используется для построения возрастных кривых и прогнозов.

Важно: этот класс не решает, какие годы/трассы брать.
Фильтрация делается снаружи перед вызовом build().

Дисперсия медианы оценивается методом bootstrap.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)

from pathlib import Path


def flatten_age_references(references_by_race: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Превращает dict[race_id -> table(gender, age, ...)] в единый DataFrame
    с колонкой race_id. Это и будет трасса–пол–возраст.
    """
    parts: list[pd.DataFrame] = []
    for race_id_value, table in references_by_race.items():
        if table is None or table.empty:
            continue
        chunk = table.copy()
        chunk.insert(0, "race_id", str(race_id_value))
        parts.append(chunk)

    if not parts:
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    sort_cols = [col for col in ["race_id", "gender", "age"] if col in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    return result

def format_seconds_to_hhmmss(seconds_series: pd.Series) -> pd.Series:
    seconds_int = pd.to_numeric(seconds_series, errors="coerce").round().astype("Int64")

    hours = (seconds_int // 3600).astype("Int64")
    minutes = ((seconds_int % 3600) // 60).astype("Int64")
    seconds = (seconds_int % 60).astype("Int64")

    hours_str = hours.astype("string").str.zfill(2)
    minutes_str = minutes.astype("string").str.zfill(2)
    seconds_str = seconds.astype("string").str.zfill(2)

    return hours_str + ":" + minutes_str + ":" + seconds_str


def to_hhmmss(seconds_value: float) -> str:
    series = pd.Series([seconds_value])
    return format_seconds_to_hhmmss(series).iloc[0]

def save_age_references_xlsx(
    references_by_race: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """
    Сохраняет возрастные референсы в один Excel-файл (один лист).
    Формат строк: race_id - gender - age.
    """
    output_file = Path(output_path)
    if output_file.suffix.lower() != ".xlsx":
        raise ValueError("output_path must end with .xlsx")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    flat = flatten_age_references(references_by_race)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        flat.to_excel(writer, sheet_name="age_references", index=False)

    logger.info("Age references saved: %s, rows=%s", output_file, len(flat))
    return output_file


class AgeReferenceBuilder:
    """
    Построение медиан времён по группам (gender, age).

    Attributes:
        min_group_size: Минимальный размер возрастной группы
        bootstrap_samples: Число bootstrap-выборок для оценки дисперсии
        random_seed: Seed для воспроизводимости bootstrap
    """

    DEFAULT_MIN_GROUP_SIZE: int = 30
    DEFAULT_BOOTSTRAP_SAMPLES: int = 200
    DEFAULT_RANDOM_SEED: int = 456

    REQUIRED_COLUMNS: list[str] = ["gender", "age", "time_seconds"]

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Инициализация из конфигурации.

        Ожидается секция config['age_references'].
        """
        age_config = config.get("age_references", {})

        self.min_group_size = int(
            age_config.get("min_group_size", self.DEFAULT_MIN_GROUP_SIZE)
        )
        self.bootstrap_samples = int(
            age_config.get("bootstrap_samples", self.DEFAULT_BOOTSTRAP_SAMPLES)
        )
        self.random_seed = int(
            age_config.get("random_seed", self.DEFAULT_RANDOM_SEED)
        )

        self._validate_params()

        logger.info(
            "AgeReferenceBuilder config: min_group_size=%d, bootstrap_samples=%d, random_seed=%d",
            self.min_group_size,
            self.bootstrap_samples,
            self.random_seed,
        )

    def _validate_params(self) -> None:
        """Проверить корректность параметров."""
        if self.min_group_size < 1:
            raise ValueError(
                f"age_references.min_group_size должен быть >= 1, получено {self.min_group_size}"
            )
        if self.bootstrap_samples < 1:
            raise ValueError(
                f"age_references.bootstrap_samples должен быть >= 1, получено {self.bootstrap_samples}"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"age_references.random_seed должен быть >= 0, получено {self.random_seed}"
            )

    def build(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Построить медианы для всех групп (gender, age).

        Args:
            dataframe: DataFrame с колонками gender, age, time_seconds

        Returns:
            DataFrame с колонками:
                - gender
                - age
                - median_time (медиана времени в секундах)
                - median_log (ln медианы)
                - median_std (bootstrap std медианы)
                - n_total (размер группы)
        """
        self._validate_dataframe(dataframe)

        working = dataframe.copy()
        working = working[working["time_seconds"].notna()]
        working = working[working["time_seconds"] > 0]

        if working.empty:
            logger.warning("AgeReferenceBuilder: нет валидных данных после фильтрации")
            return self._empty_result()

        groups = working.groupby(["gender", "age"], sort=False)

        rows: list[dict[str, Any]] = []
        for (gender_value, age_value), group in groups:
            row = self._build_one_group(str(gender_value), int(age_value), group)
            if row is not None:
                rows.append(row)

        if not rows:
            logger.warning("AgeReferenceBuilder: не удалось построить ни одной медианы")
            return self._empty_result()

        result = pd.DataFrame(rows)

        # Сортируем по gender, age для удобства
        result = result.sort_values(["gender", "age"]).reset_index(drop=True)

        logger.info(f"AgeReferenceBuilder: построено {len(result)} возрастных медиан")

        return result

    def _build_one_group(
        self,
        gender_value: str,
        age_value: int,
        group: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Построить медиану для одной группы (gender, age)."""
        group_size = len(group)

        if group_size < self.min_group_size:
            # Не логируем — слишком много мелких групп
            return None

        times = group["time_seconds"].astype(float).to_numpy()
        median_time = float(np.median(times))

        # Bootstrap для оценки std медианы
        median_var = self._bootstrap_median_variance(times)
        median_std = float(np.sqrt(median_var))

        return {
            "gender": gender_value,
            "age": age_value,
            "median_time": median_time,
            "median_log": float(np.log(median_time)),
            "median_std": median_std,
            "n_total": int(group_size),
        }

    def _bootstrap_median_variance(self, times: np.ndarray) -> float:
        """
        Оценить дисперсию медианы методом bootstrap (векторизованно).

        Генерирует все bootstrap-выборки разом для максимальной скорости.
        """
        if self.bootstrap_samples <= 1:
            return 0.0

        rng = np.random.default_rng(self.random_seed)
        n = times.size

        # Генерируем ВСЕ индексы разом: матрица (n_bootstrap, n)
        indices = rng.integers(0, n, size=(self.bootstrap_samples, n))

        # Получаем все bootstrap-выборки разом
        all_resamples = times[indices]  # shape: (n_bootstrap, n)

        # Медианы всех выборок — векторно по axis=1
        bootstrap_medians = np.median(all_resamples, axis=1)

        return float(np.var(bootstrap_medians, ddof=1))

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Проверить входной DataFrame."""
        if dataframe is None or dataframe.empty:
            raise ValueError("AgeReferenceBuilder.build: пустой DataFrame")

        missing = [col for col in self.REQUIRED_COLUMNS if col not in dataframe.columns]
        if missing:
            raise ValueError(f"AgeReferenceBuilder.build: отсутствуют колонки: {missing}")

    def _empty_result(self) -> pd.DataFrame:
        """Вернуть пустой DataFrame с правильными колонками."""
        return pd.DataFrame(columns=[
            "gender",
            "age",
            "median_time",
            "median_log",
            "median_std",
            "n_total",
        ])
