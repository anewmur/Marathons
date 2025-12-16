
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RawFilterStats:
    """
    Статистика применения фильтра.

    Fields:
        rows_in: число строк до фильтра
        rows_out: число строк после фильтра
        removed: сколько строк удалено
    """
    rows_in: int
    rows_out: int
    removed: int


class RawFilter:
    """
    Фильтрация сырых данных по конфигу.

    Конфиг:
        only_finishers: bool
        allowed_statuses: list[str] | None
        exclude_statuses: list[str] | None
        gender: str | None
        distance_km: float | int | None
        distance_tol_km: float | int
        race_ids: list[str] | None
        years: list[int] | None
    """

    def apply(self, dataframe: pd.DataFrame, filter_config: dict[str, Any]) -> tuple[pd.DataFrame, RawFilterStats]:
        """
        Input:
            dataframe: исходный DataFrame
            filter_config: словарь параметров фильтра
        Returns:
            (отфильтрованный DataFrame, статистика)
        Does:
            применяет фильтры без изменения исходного объекта.
        """
        if not isinstance(filter_config, dict):
            raise ValueError("filter_config должен быть dict")

        rows_in = len(dataframe)
        df_filtered = dataframe

        df_filtered = self._filter_status(df_filtered, filter_config)
        df_filtered = self._filter_gender(df_filtered, filter_config)
        df_filtered = self._filter_race_ids(df_filtered, filter_config)
        df_filtered = self._filter_years(df_filtered, filter_config)
        df_filtered = self._filter_distance(df_filtered, filter_config)

        df_filtered = df_filtered.reset_index(drop=True)
        rows_out = len(df_filtered)
        stats = RawFilterStats(rows_in=rows_in, rows_out=rows_out, removed=rows_in - rows_out)
        return df_filtered, stats

    def _filter_status(self, dataframe: pd.DataFrame, filter_config: dict[str, Any]) -> pd.DataFrame:
        """
        Input: DataFrame и конфиг.
        Returns: DataFrame.
        Does: фильтрует по status с приоритетом only_finishers > allowed_statuses > exclude_statuses.
        """
        only_finishers = bool(filter_config.get("only_finishers", False))
        if only_finishers:
            return dataframe[dataframe["status"] == "OK"]

        allowed_statuses = filter_config.get("allowed_statuses", None)
        if allowed_statuses is not None:
            if not isinstance(allowed_statuses, list):
                raise ValueError("allowed_statuses должен быть list или None")
            return dataframe[dataframe["status"].isin(allowed_statuses)]

        exclude_statuses = filter_config.get("exclude_statuses", None)
        if exclude_statuses is not None:
            if not isinstance(exclude_statuses, list):
                raise ValueError("exclude_statuses должен быть list или None")
            return dataframe[~dataframe["status"].isin(exclude_statuses)]

        return dataframe

    def _filter_gender(self, dataframe: pd.DataFrame, filter_config: dict[str, Any]) -> pd.DataFrame:
        """
        Input: DataFrame и конфиг.
        Returns: DataFrame.
        Does: фильтрует по полу, если задан gender.
        """
        gender_value = filter_config.get("gender", None)
        if gender_value is None:
            return dataframe
        return dataframe[dataframe["gender"] == gender_value]

    def _filter_race_ids(self, dataframe: pd.DataFrame, filter_config: dict[str, Any]) -> pd.DataFrame:
        """
        Input: DataFrame и конфиг.
        Returns: DataFrame.
        Does: фильтрует по race_id, если задан race_ids.
        """
        race_ids = filter_config.get("race_ids", None)
        if race_ids is None:
            return dataframe
        if not isinstance(race_ids, list):
            raise ValueError("race_ids должен быть list или None")
        return dataframe[dataframe["race_id"].isin(race_ids)]

    def _filter_years(self, dataframe: pd.DataFrame, filter_config: dict[str, Any]) -> pd.DataFrame:
        """
        Input: DataFrame и конфиг.
        Returns: DataFrame.
        Does: фильтрует по year, если задан years.
        """
        years = filter_config.get("years", None)
        if years is None:
            return dataframe
        if not isinstance(years, list):
            raise ValueError("years должен быть list или None")
        return dataframe[dataframe["year"].isin(years)]

    def _filter_distance(self, dataframe: pd.DataFrame, filter_config: dict[str, Any]) -> pd.DataFrame:
        """
        Input: DataFrame и конфиг.
        Returns: DataFrame.
        Does: фильтрует по distance_km с допуском, если distance_km задан.
        """
        if "distance_km" not in dataframe.columns:
            raise ValueError(
                "raw_filter.distance_km задан, но в данных нет колонки distance_km (проверь DataLoader/кэш)")

        distance_km = filter_config.get("distance_km", None)

        if distance_km is None:
            return dataframe

        distance_tol_km = filter_config.get("distance_tol_km", 0.5)
        try:
            distance_value = float(distance_km)
        except (TypeError, ValueError):
            raise ValueError("distance_km должен быть числом или None")
        try:
            tol_value = float(distance_tol_km)
        except (TypeError, ValueError):
            raise ValueError("distance_tol_km должен быть числом")

        df_filtered = dataframe[dataframe["distance_km"].notna()]
        df_filtered = df_filtered[(df_filtered["distance_km"] - distance_value).abs() <= tol_value]
        return df_filtered
