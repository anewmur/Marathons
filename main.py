"""
Прогнозирование времени финиша марафонцев.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import yaml
import time

from utils.raw_filter import RawFilter
import DataLoader.data_loader as data_loader_module
from DataLoader.data_loader import DataLoader

from preprocessor import Preprocessor
from trace_reference_builder import TraceReferenceBuilder
from age_reference_builder import AgeReferenceBuilder

logger = logging.getLogger(__name__)


class MarathonModel:
    """
    Оркестратор пайплайна прогнозирования времени финиша.
    Последовательность шагов:
        load_data → validate_raw → add_row_id → preprocess → build_trace_references →
        build_age_references → fit_age_model → predict

    Гарантирует:
        - После add_row_id все датафреймы используют row_id как уникальный индекс.
        - Все шаги логируют rows_in/rows_out + elapsed + ключевые метрики.
        - Шаги не выполняют I/O кроме как через DataLoader (кэширование).
    """

    data_path: Path
    validation_year: int
    verbose: bool
    config: dict[str, Any]

    df: pd.DataFrame | None                  # сырые данные
    df_clean: pd.DataFrame | None             # предобработанные данные с row_id
    trace_references: pd.DataFrame | None
    age_references: dict[str, pd.DataFrame]

    _data_loader: DataLoader
    _steps_completed: dict[str, bool]

    def __init__(
        self,
        data_path: str | Path,
        validation_year: int | None = None,
        verbose: bool = True,
    ) -> None:
        """
        Инициализирует модель.
        Input: путь к данным, год валидации, режим verbose.
        Returns: экземпляр MarathonModel.
        Does: загружает конфиг, инициализирует внутренние структуры.
        """
        self.data_path = Path(data_path)
        self.verbose = verbose
        self.config = self._load_config()

        if validation_year is None:
            self.validation_year = int(self.config.get("validation", {}).get("year", 0))
        else:
            self.validation_year = int(validation_year)

        self._data_loader = DataLoader()
        logger.info("DataLoader class: %s.%s", self._data_loader.__class__.__module__,
                    self._data_loader.__class__.__name__)
        logger.info("DataLoader file: %s", data_loader_module.__file__)
        logger.info("DataLoader OUTPUT_COLUMNS: %s", getattr(data_loader_module, "OUTPUT_COLUMNS", None))

        self.df = None
        self.df_clean = None
        self.trace_references = None
        self.age_references = {}

        self._steps_completed = {
            "load_data": False,
            "filter_raw": False,
            "validate_raw": False,
            "add_row_id": False,
            "preprocess": False,
            "build_trace_references": False,
            "build_age_references": False,
            "fit_age_model": False,
        }

    # ------------------------------------------------------------------
    # config
    # ------------------------------------------------------------------
    def _load_config(self) -> dict[str, Any]:
        """
        Загружает конфигурацию из config.yaml.
        Input: ---
        Returns: словарь с конфигурацией (по умолчанию, если файл отсутствует).
        Does: читает файл или возвращает дефолтные значения.
        """
        config_path = Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            logger.warning("config.yaml не найден, используются значения по умолчанию")
            return {
                "preprocessing": {"age_center": 35, "age_scale": 10},
                "validation": {"year": 0},
            }
        with open(config_path, "r", encoding="utf-8") as file_handle:
            return yaml.safe_load(file_handle)

    # ------------------------------------------------------------------
    # utils
    # ------------------------------------------------------------------
    def _check_step(self, current: str, required: list[str]) -> None:
        """
        Проверяет выполнение требуемых шагов.
        Input: имя текущего шага и список требуемых.
        Returns: ничего (или RuntimeError).
        Does: поднимает исключение, если предыдущие шаги не выполнены.
        """
        for step_name in required:
            if not self._steps_completed.get(step_name, False):
                raise RuntimeError(
                    f"Шаг '{current}' требует выполнения '{step_name}'. "
                    f"Сначала вызовите model.{step_name}()"
                )

    def _get_loy_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает LOY-выборку (исключает validation_year).
        Input: датафрейм.
        Returns: копия датафрейма без строк validation_year (если задан).
        Does: фильтрует по году валидации.
        """
        if self.validation_year > 0:
            return dataframe[dataframe["year"] != self.validation_year].copy()
        return dataframe.copy()

    def _log_step(
        self,
        step_name: str,
        rows_in: int,
        rows_out: int,
        elapsed_sec: float,
        extra: str | None = None,
    ) -> None:
        """
        Единое логирование метрик шага.
        Input: имя шага, строки на входе/выходе, время, опциональная доп. метрика.
        Returns: ничего.
        Does: выводит лог в едином формате (если verbose=True).
        """
        if not self.verbose:
            return
        message = f"{step_name}: rows {rows_in} → {rows_out}, elapsed {elapsed_sec:.2f}s"
        if extra:
            message += f", {extra}"
        logger.info(message)

    def debug_slice(
        self,
        df: pd.DataFrame | None = None,
        filters: dict[str, Any] | None = None,
    ) -> None:
        """
        Универсальный debug-интерфейс для среза данных.
        Input: датафрейм (по умолчанию df_clean) и словарь фильтров.
        Returns: ничего (печатает срез).
        Does: применяет фильтры и выводит отфильтрованные строки.
        Supports: точное значение, список/множество, callable, np.ndarray.
        """
        if df is None:
            if self.df_clean is None:
                logger.warning("df_clean недоступен для debug_slice")
                return
            df = self.df_clean

        if not filters:
            print(df.to_string(index=True))
            return

        mask = pd.Series(True, index=df.index)
        for col, val in filters.items():
            if col not in df.columns:
                logger.warning(f"Колонка {col} отсутствует в датафрейме")
                continue
            if callable(val):
                mask &= val(df[col])
            elif isinstance(val, (list, tuple, set)):
                mask &= df[col].isin(val)
            elif isinstance(val, np.ndarray):
                mask &= df[col].isin(val)
            else:
                mask &= (df[col] == val)

        sliced = df[mask]
        logger.debug(f"Debug slice ({len(sliced)} строк из {len(df)}):")
        logger.debug(sliced.to_string(index=True))

    # ------------------------------------------------------------------
    # step 1: load and filter
    # ------------------------------------------------------------------
    def load_data(self) -> "MarathonModel":
        """
        Загружает сырые данные.
        Input: ---
        Returns: self.
        Does: читает директорию через DataLoader.
        Гарантирует: df не пустой, путь существует.
        """
        start_time = time.time()
        if not self.data_path.exists():
            raise FileNotFoundError(f"Путь не существует: {self.data_path}")

        self.df = self._data_loader.load_directory_cached(self.data_path)

        logger.info("df columns: %s", list(self.df.columns))

        expected_columns = {
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
        }
        missing_columns = [col for col in expected_columns if col not in self.df.columns]
        if missing_columns:
            raise RuntimeError(
                f"load_data вернул другой контракт. Нет колонок: {missing_columns}. "
                f"Скорее всего подхвачен старый DataLoader или старый parquet-кэш."
            )

        if self.df is None or self.df.empty:
            raise ValueError("Не удалось загрузить данные")

        elapsed = time.time() - start_time
        self._steps_completed["load_data"] = True
        self._log_step("load_data", rows_in=0, rows_out=len(self.df), elapsed_sec=elapsed)

        return self

    def filter_raw(self) -> "MarathonModel":
        """
        Фильтрация сырых данных по config.yaml.
        Input: ---
        Returns: self.
        Does: применяет raw_filter из self.config и обновляет self.df.
        Гарантирует: self.df остаётся DataFrame с теми же колонками, но с подмножеством строк.
        """
        self._check_step("filter_raw", ["load_data"])
        assert self.df is not None

        start_time = time.time()
        rows_in = len(self.df)

        raw_filter_config = self.config.get("raw_filter", {})
        if raw_filter_config is None:
            raw_filter_config = {}
        if not isinstance(raw_filter_config, dict):
            raise ValueError("config.raw_filter должен быть словарём (YAML mapping)")

        raw_filter = RawFilter()
        df_filtered, stats = raw_filter.apply(self.df, raw_filter_config)

        self.df = df_filtered

        elapsed = time.time() - start_time
        self._steps_completed["filter_raw"] = True
        self._log_step("filter_raw", rows_in=rows_in, rows_out=stats.rows_out, elapsed_sec=elapsed)
        return self

    # ------------------------------------------------------------------
    # step 2: validate raw
    # ------------------------------------------------------------------
    def validate_raw(self) -> "MarathonModel":
        """
        Валидация сырых данных в одном месте.
        Input: ---
        Returns: self.
        Does: проверяет наличие обязательных колонок и базовые инварианты.
        Гарантирует: обязательные колонки присутствуют, нет null в ключевых полях,
        для status='OK' time_seconds заполнен и > 0, df не пустой.
        """
        self._check_step("validate_raw", ["load_data"])
        assert self.df is not None

        start_time = time.time()
        rows_in = len(self.df)

        required_columns = {
            "year",
            "race_id",
            "surname",
            "name",
            "gender",
            "age",
            "status",
            "time_seconds",
            "city",
            "bib_number",
            "distance_km",
        }
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_columns}")

        if rows_in == 0:
            raise ValueError("Пустой датафрейм после загрузки")

        always_non_null = ["year", "race_id", "surname", "gender", "age", "status"]
        for column_name in always_non_null:
            if self.df[column_name].isna().any():
                raise ValueError(f"Null-значения в обязательной колонке: {column_name}")

        # surname не должен быть пустой строкой (после strip)
        surname_stripped = self.df["surname"].astype(str).str.strip()
        if (surname_stripped == "").any():
            raise ValueError("Пустые значения в колонке surname (после strip)")

        # name допускаем пустым (как в DataLoader), но не допускаем NaN
        if self.df["name"].isna().any():
            raise ValueError("Null-значения в колонке name")

        # time_seconds может быть NaN/None только если статус не OK
        mask_ok = self.df["status"] == "OK"
        if self.df.loc[mask_ok, "time_seconds"].isna().any():
            raise ValueError("Для status='OK' time_seconds не должен быть пустым")

        # Для OK время должно быть положительным
        if (self.df.loc[mask_ok, "time_seconds"] <= 0).any():
            raise ValueError("Для status='OK' time_seconds должен быть > 0")

        elapsed = time.time() - start_time
        self._steps_completed["validate_raw"] = True
        self._log_step("validate_raw", rows_in=rows_in, rows_out=rows_in, elapsed_sec=elapsed)
        return self

    # ------------------------------------------------------------------
    # step 3: add row_id
    # ------------------------------------------------------------------
    def add_row_id(self) -> "MarathonModel":
        """
        Добавляет технический row_id как уникальный индекс.
        Input: ---
        Returns: self.
        Does: создаёт индекс row_id от 0 до len-1.
        Гарантирует: индекс уникальный.
        """
        self._check_step("add_row_id", ["validate_raw"])
        assert self.df is not None

        start_time = time.time()
        rows_in = len(self.df)

        self.df = self.df.reset_index(drop=True).copy()
        self.df.insert(0, "row_id", range(rows_in))
        self.df = self.df.set_index("row_id")

        if not self.df.index.is_unique:
            raise ValueError("Нарушена уникальность row_id")

        elapsed = time.time() - start_time
        self._steps_completed["add_row_id"] = True
        self._log_step("add_row_id", rows_in=rows_in, rows_out=rows_in, elapsed_sec=elapsed)
        return self

    # ------------------------------------------------------------------
    # step 4: preprocess
    # ------------------------------------------------------------------
    def preprocess(self) -> MarathonModel:
        """
        Предобработка данных.

        Заполняет: self.df_clean

        Гарантирует:
        - Нет NaN и неположительных time_seconds для status='OK' (или для всех строк, если status отсутствует).
        - Возраст в разумных пределах (10-100).
        - Уникальность row_id сохранена.

        Returns:self.
        """

        self._check_step("preprocess", ["add_row_id"])
        assert self.df is not None

        start_time = time.time()
        rows_in = len(self.df)

        preprocessor = Preprocessor(self.config)
        self.df_clean = preprocessor.run(self.df)

        if self.df_clean is None or self.df_clean.empty:
            raise ValueError("После предобработки нет данных")

        if "time_seconds" not in self.df_clean.columns:
            raise ValueError("После предобработки отсутствует колонка time_seconds")

        mask_ok: pd.Series | slice
        if "status" in self.df_clean.columns:
            mask_ok = self.df_clean["status"] == "OK"
        else:
            mask_ok = slice(None)

        if self.df_clean.loc[mask_ok, "time_seconds"].isna().any():
            raise ValueError("NaN в time_seconds для status='OK' после предобработки")
        if (self.df_clean.loc[mask_ok, "time_seconds"] <= 0).any():
            raise ValueError("Отрицательное или нулевое time_seconds для status='OK' после предобработки")

        if "age" in self.df_clean.columns:
            if ((self.df_clean["age"] < 10) | (self.df_clean["age"] > 100)).any():
                raise ValueError("Некорректный возраст после предобработки")

        if not self.df_clean.index.is_unique:
            raise ValueError("Нарушена уникальность row_id после предобработки")

        elapsed = time.time() - start_time

        self._steps_completed["preprocess"] = True
        self._log_step(
            "preprocess",
            rows_in=rows_in,
            rows_out=len(self.df_clean),
            elapsed_sec=elapsed,
        )

        return self
    # ------------------------------------------------------------------
    # step 5: trace references
    # ------------------------------------------------------------------
    def build_trace_references(self) -> "MarathonModel":
        """
        Строит эталоны трасс.
        Input: ---
        Returns: self.
        Does: использует TraceReferenceBuilder на LOY-выборке.
        Гарантирует: уникальность (race_id, gender), положительные медианы.
        """
        self._check_step("build_trace_references", ["preprocess"])
        assert self.df_clean is not None

        start_time = time.time()
        rows_in = len(self.df_clean)

        loy_frame = self._get_loy_frame(self.df_clean)
        builder = TraceReferenceBuilder(self.config)
        self.trace_references = builder.build(loy_frame)

        if self.trace_references is None or self.trace_references.empty:
            raise ValueError("Не удалось построить эталоны трасс")

        key_cols = ["race_id", "gender"]
        if self.trace_references[key_cols].duplicated().any():
            raise ValueError("Дубли в ключе trace_references")

        elapsed = time.time() - start_time
        groups = self.trace_references[key_cols].drop_duplicates().shape[0]
        self._steps_completed["build_trace_references"] = True
        self._log_step(
            "build_trace_references",
            rows_in=rows_in,
            rows_out=len(self.trace_references),
            elapsed_sec=elapsed,
            extra=f"groups={groups}",
        )
        return self

    # ------------------------------------------------------------------
    # step 6: age references
    # ------------------------------------------------------------------
    def build_age_references(self) -> "MarathonModel":
        """
        Строит возрастные таблицы по трассам.
        Input: ---
        Returns: self.
        Does: группирует по race_id и строит таблицы через AgeReferenceBuilder.
        """
        self._check_step("build_age_references", ["preprocess"])
        assert self.df_clean is not None

        start_time = time.time()
        rows_in = len(self.df_clean)

        loy_frame = self._get_loy_frame(self.df_clean)
        builder = AgeReferenceBuilder(self.config)
        references_by_race: dict[str, pd.DataFrame] = {}

        for race_id_value, race_group in loy_frame.groupby("race_id", sort=False):
            table = builder.build(race_group)
            if table is None or table.empty:
                continue
            table = table.rename(
                columns={
                    "median_time": "age_median_time",
                    "median_log": "age_median_log",
                    "median_std": "age_median_std",
                }
            ).reset_index(drop=True)
            table["age_median_var"] = table["age_median_std"] ** 2
            references_by_race[str(race_id_value)] = table

        total_rows = sum(len(df) for df in references_by_race.values())
        elapsed = time.time() - start_time

        self.age_references = references_by_race
        self._steps_completed["build_age_references"] = True
        self._log_step(
            "build_age_references",
            rows_in=rows_in,
            rows_out=total_rows,
            elapsed_sec=elapsed,
            extra=f"races={len(references_by_race)}",
        )
        return self

    # ------------------------------------------------------------------
    # step 7: fit age model
    # ------------------------------------------------------------------
    def fit_age_model(self) -> "MarathonModel":
        """
        Обучает возрастную модель (заглушка).
        Input: ---
        Returns: self.
        Does: помечает шаг выполненным (реальная модель пока не реализована).
        """
        self._check_step(
            "fit_age_model",
            ["preprocess", "build_trace_references"],
        )
        start_time = time.time()
        elapsed = time.time() - start_time
        self._steps_completed["fit_age_model"] = True
        self._log_step("fit_age_model", rows_in=0, rows_out=0, elapsed_sec=elapsed)
        return self

    # ------------------------------------------------------------------
    # pipeline
    # ------------------------------------------------------------------
    def run(self) -> "MarathonModel":
        """
        Полный пайплайн.
        Input: ---
        Returns: self.
        Does: последовательно выполняет все шаги.
        """
        return (
            self.load_data()
            .filter_raw()
            .validate_raw()
            .add_row_id()
            .preprocess()
            .build_trace_references()
            .build_age_references()
            .fit_age_model()
        )

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(
        self,
        runner_id: str,
        race_id: str,
        year: int,
        age: int | None = None,
    ) -> dict[str, float] | None:
        """
        Прогноз (пока заглушка).
        Input: идентификаторы участника, трассы, год и возраст.
        Returns: словарь с прогнозом или None.
        Does: проверяет шаги и возвращает заглушку.
        """
        self._check_step("predict", ["preprocess", "build_trace_references"])
        return None

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------
    def summary(self) -> str:
        """
        Краткая сводка состояния.
        Input: ---
        Returns: строку с информацией о модели и выполненных шагах.
        Does: формирует текстовый отчёт.
        """
        lines = [
            "MarathonModel Summary",
            "=" * 40,
            f"Data path: {self.data_path}",
            f"Validation year: {self.validation_year}",
            "",
            "Steps completed:",
        ]
        for step_name, done in self._steps_completed.items():
            mark = "✓" if done else "✗"
            lines.append(f" [{mark}] {step_name}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        completed = sum(self._steps_completed.values())
        total = len(self._steps_completed)
        return f"MarathonModel(data_path='{self.data_path}', steps={completed}/{total})"


if __name__ == "__main__":
    from logging_setup import easy_logging
    easy_logging(True)

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        validation_year=2025,
        verbose=True
    )
    model.run()
    print(model.summary())

