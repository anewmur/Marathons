"""
Построение эталонных времён R_{c,g} для пар (race_id, gender).

ReferenceBuilder строит эталон трассы на шкале времени T (секунды).
Эталон строится по всем годам, которые поданы на вход build().
LOY vs production делается снаружи фильтрацией по year.

Протокол:
  1) берём top_fraction самых быстрых;
  2) триммим trim_fraction с каждого края внутри top;
  3) берём median оставшихся (в секундах).

Дисперсия эталона оценивается как bootstrap-дисперсия медианы.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)
from pathlib import Path


def save_trace_references_xlsx(
    references_df: pd.DataFrame,
    output_path: str | Path) -> Path:

    output_file = Path(output_path)
    if output_file.suffix.lower() != ".xlsx":
        raise ValueError("output_path must end with .xlsx")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if references_df is None or references_df.empty:
        raise ValueError("Trace references DataFrame is empty")

    df_out = references_df.copy()

    sort_cols = [col for col in ["race_id", "gender"] if col in df_out.columns]
    if sort_cols:
        df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="trace_references", index=False)

    logger.info("Trace references saved: %s, rows=%s", output_file, len(df_out))
    return output_file


class TraceReferenceBuilder:
    """
    Построение эталонных времён R_{c,g} для пар (race_id, gender).

    Attributes:
        top_fraction: Доля лучших (по времени) для отбора
        trim_fraction: Доля симметричной обрезки внутри top
        min_group_size: Минимальный размер группы
        min_used_runners: Минимум наблюдений после top/trim
        bootstrap_samples: Число bootstrap-выборок для оценки дисперсии
        random_seed: Seed для воспроизводимости bootstrap
    """

    DEFAULT_TOP_FRACTION: float = 0.05
    DEFAULT_TRIM_FRACTION: float = 0.10
    DEFAULT_MIN_GROUP_SIZE: int = 40
    DEFAULT_MIN_USED_RUNNERS: int = 3
    DEFAULT_BOOTSTRAP_SAMPLES: int = 200
    DEFAULT_RANDOM_SEED: int = 123

    REQUIRED_COLUMNS: list[str] = ["race_id", "gender", "time_seconds"]

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Инициализация из конфигурации.

        Ожидается секция config['references'].
        """
        references_config = config.get("references", {})

        self.top_fraction = float(
            references_config.get("top_fraction", self.DEFAULT_TOP_FRACTION)
        )
        self.trim_fraction = float(
            references_config.get("trim_fraction", self.DEFAULT_TRIM_FRACTION)
        )
        self.min_group_size = int(
            references_config.get("min_group_size", self.DEFAULT_MIN_GROUP_SIZE)
        )
        self.min_used_runners = int(
            references_config.get("min_used_runners", self.DEFAULT_MIN_USED_RUNNERS)
        )
        self.bootstrap_samples = int(
            references_config.get("bootstrap_samples", self.DEFAULT_BOOTSTRAP_SAMPLES)
        )
        self.random_seed = int(
            references_config.get("random_seed", self.DEFAULT_RANDOM_SEED)
        )

        self._validate_params()

    def _validate_params(self) -> None:
        """Проверить корректность параметров."""
        if not (0.0 < self.top_fraction <= 1.0):
            raise ValueError(
                f"references.top_fraction должен быть в (0, 1], получено {self.top_fraction}"
            )
        if not (0.0 <= self.trim_fraction < 0.5):
            raise ValueError(
                f"references.trim_fraction должен быть в [0, 0.5), получено {self.trim_fraction}"
            )
        if self.min_group_size < 1:
            raise ValueError(
                f"references.min_group_size должен быть >= 1, получено {self.min_group_size}"
            )
        if self.min_used_runners < 1:
            raise ValueError(
                f"references.min_used_runners должен быть >= 1, получено {self.min_used_runners}"
            )
        if self.bootstrap_samples < 1:
            raise ValueError(
                f"references.bootstrap_samples должен быть >= 1, получено {self.bootstrap_samples}"
            )

    def build(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Построить эталоны для всех пар (race_id, gender).

        Args:
            dataframe: DataFrame с колонками race_id, gender, time_seconds

        Returns:
            DataFrame с колонками:
                - race_id
                - gender
                - reference_time (R_{c,g} в секундах)
                - reference_log (ln R_{c,g})
                - reference_std (bootstrap std эталона)
                - n_total (размер группы)
                - n_used (использовано после top/trim)
        """
        self._validate_dataframe(dataframe)

        working = dataframe.copy()
        working = working[working["time_seconds"].notna()]
        working = working[working["time_seconds"] > 0]

        if working.empty:
            logger.warning("ReferenceBuilder: нет валидных данных после фильтрации")
            return self._empty_result()

        groups = working.groupby(["race_id", "gender"], sort=False)

        rows: list[dict[str, Any]] = []
        for (race_id_value, gender_value), group in groups:
            row = self._build_one_group(str(race_id_value), str(gender_value), group)
            if row is not None:
                rows.append(row)

        if not rows:
            logger.warning("ReferenceBuilder: не удалось построить ни одного эталона")
            return self._empty_result()

        result = pd.DataFrame(rows)
        logger.info(f"ReferenceBuilder: построено {len(result)} эталонов")

        return result

    def _build_one_group(
        self,
        race_id_value: str,
        gender_value: str,
        group: pd.DataFrame,
    ) -> dict[str, Any] | None:
        """Построить эталон для одной группы (race_id, gender)."""
        group_size = len(group)

        if group_size < self.min_group_size:
            logger.warning(
                f"Группа ({race_id_value}, {gender_value}): "
                f"{group_size} < {self.min_group_size}, пропущена"
            )
            return None

        times = group["time_seconds"].astype(float).to_numpy()
        reference_time, used_count = self._reference_from_times(times)

        if reference_time is None:
            logger.warning(
                f"Группа ({race_id_value}, {gender_value}): "
                f"не прошла пороги после top/trim, пропущена"
            )
            return None

        # Bootstrap для оценки std эталона
        reference_var = self._bootstrap_reference_variance(times)
        reference_std = float(np.sqrt(reference_var))

        return {
            "race_id": race_id_value,
            "gender": gender_value,
            "reference_time": float(reference_time),
            "reference_log": float(np.log(reference_time)),
            "reference_std": reference_std,
            "n_total": int(group_size),
            "n_used": int(used_count),
        }

    def _reference_from_times(self, times: np.ndarray) -> tuple[float | None, int]:
        """
        Вычислить эталон по массиву времён.

        Returns:
            (reference_time, used_count) или (None, 0) если не удалось
        """
        if times.size == 0:
            return None, 0

        sorted_times = np.sort(times)

        n_total = int(sorted_times.size)

        # top_size должен быть не меньше min_used_runners, если это возможно
        top_size_fraction = int(np.ceil(n_total * self.top_fraction))
        top_size = min(n_total, max(top_size_fraction, self.min_used_runners))
        if top_size < 1:
            return None, 0

        top_times = sorted_times[:top_size]

        used_times = self._apply_trim(top_times)

        # ВАЖНО: после trim всё ещё может стать меньше min_used_runners, тогда уже ничего не сделать
        if used_times.size < min(self.min_used_runners, n_total):
            return None, int(used_times.size)

        reference_time = float(np.median(used_times))
        return reference_time, int(used_times.size)

    def _apply_trim(self, times: np.ndarray) -> np.ndarray:
        """Применить симметричную обрезку."""
        if times.size == 0:
            return times

        trim_count = int(np.floor(times.size * self.trim_fraction))
        left_idx = trim_count
        right_idx = times.size - trim_count

        if right_idx <= left_idx:
            return np.array([], dtype=float)

        return times[left_idx:right_idx]

    def _bootstrap_reference_variance(self, times: np.ndarray) -> float:
        """
        Оценить дисперсию эталона методом bootstrap (векторизованно).

        Генерирует все bootstrap-выборки разом для максимальной скорости.
        Для каждой выборки применяется тот же протокол top/trim/median.
        """
        if self.bootstrap_samples <= 1:
            return 0.0

        rng = np.random.default_rng(self.random_seed)

        n_total = int(times.size)
        if n_total == 0:
            return 0.0

        # Генерируем ВСЕ индексы разом: матрица (n_bootstrap, n)
        indices = rng.integers(0, n_total, size=(self.bootstrap_samples, n_total))

        # Получаем все bootstrap-выборки разом
        all_resamples = times[indices]  # shape: (n_bootstrap, n)

        # Сортируем каждую выборку по axis=1
        all_sorted = np.sort(all_resamples, axis=1)

        # Top fraction — берём первые top_size элементов каждой строки
        top_size_fraction = int(np.ceil(n_total * self.top_fraction))
        top_size = min(n_total, max(top_size_fraction, self.min_used_runners))
        if top_size < 1:
            return 0.0

        all_top = all_sorted[:, :top_size]

        # Trim — отрезаем trim_count с каждого края
        trim_count = int(np.floor(top_size * self.trim_fraction))
        if trim_count > 0 and top_size > 2 * trim_count:
            all_trimmed = all_top[:, trim_count:-trim_count]
        else:
            all_trimmed = all_top

        # Проверяем что осталось достаточно данных
        min_required = min(self.min_used_runners, n_total)
        if all_trimmed.shape[1] < min_required:
            return 0.0

        # Медианы всех выборок — векторно по axis=1
        bootstrap_medians = np.median(all_trimmed, axis=1)

        return float(np.var(bootstrap_medians, ddof=1))

    def _validate_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Проверить входной DataFrame."""
        if dataframe is None or dataframe.empty:
            raise ValueError("ReferenceBuilder.build: пустой DataFrame")

        missing = [col for col in self.REQUIRED_COLUMNS if col not in dataframe.columns]
        if missing:
            raise ValueError(f"ReferenceBuilder.build: отсутствуют колонки: {missing}")

    def _empty_result(self) -> pd.DataFrame:
        """Вернуть пустой DataFrame с правильными колонками."""
        return pd.DataFrame(columns=[
            "race_id",
            "gender",
            "reference_time",
            "reference_log",
            "reference_std",
            "n_total",
            "n_used",
        ])
