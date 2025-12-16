from __future__ import annotations
from utils.city_normalizer import CityNormalizer, dump_unrecognized_cities
import logging
import time
from typing import Any

import pandas as pd
from utils.compress_sorted import dump_homonyms_by_birth_year

logger = logging.getLogger(__name__)


# =============================================================================
# Вспомогательные функции
# =============================================================================

def _log_step(
    step_name: str,
    rows_in: int,
    rows_out: int,
    start_time: float,
    extra_metrics: dict[str, Any] | None = None,
) -> None:
    """
    Единый формат логирования шага.
    """
    dropped = rows_in - rows_out
    elapsed_sec = time.time() - start_time

    parts = [
        f"{step_name}:",
        f"rows {rows_in} → {rows_out}",
        f"dropped {dropped}",
        f"elapsed {elapsed_sec:.3f}s",
    ]
    if extra_metrics:
        for metric_name, metric_value in extra_metrics.items():
            parts.append(f"{metric_name}={metric_value}")

    logger.info(", ".join(parts))


def debug_slice(
    df: pd.DataFrame,
    *,
    surname: str | None = None,
    name: str | None = None,
    runner_id: str | None = None,
    race_id: str | None = None,
    year: int | None = None,
    limit: int = 30,
) -> pd.DataFrame:
    """
    Возвращает срез для отладки (без print).
    Фильтры: частичный поиск case-insensitive для surname/name.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    mask = pd.Series(True, index=df.index)

    if surname is not None and "surname" in df.columns:
        surname_series = df["surname"].astype("string")
        mask &= surname_series.str.contains(surname, case=False, na=False)

    if name is not None and "name" in df.columns:
        name_series = df["name"].astype("string")
        mask &= name_series.str.contains(name, case=False, na=False)

    if runner_id is not None and "runner_id" in df.columns:
        mask &= df["runner_id"] == runner_id

    if race_id is not None and "race_id" in df.columns:
        mask &= df["race_id"] == race_id

    if year is not None and "year" in df.columns:
        mask &= df["year"] == year

    sliced = df.loc[mask]

    display_columns = []
    for column_name in [
        "surname",
        "name",
        "runner_id",
        "race_id",
        "year",
        "age",
        "gender",
        "city",
        "bib_number",
        "time_seconds",
    ]:
        if column_name in sliced.columns:
            display_columns.append(column_name)

    if not display_columns:
        return sliced.head(limit)

    return sliced[display_columns].head(limit)


# =============================================================================
# Стадии пайплайна
# =============================================================================

def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Проверка контракта входа.
    """
    start_time = time.time()
    rows_in = len(df)

    if df is None:
        raise ValueError("Input df is None")
    if rows_in == 0:
        raise ValueError("Input df is empty")
    if df.index.name != "row_id":
        raise ValueError("Index must be named 'row_id'")
    if not df.index.is_unique:
        raise ValueError("Index must be unique 'row_id'")

    required_columns = [
        "surname",
        "name",
        "race_id",
        "year",
        "gender",
        "age",
        "time_seconds",
        "status",
    ]
    missing_columns = []
    for column_name in required_columns:
        if column_name not in df.columns:
            missing_columns.append(column_name)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    _log_step("validate_input_contract", rows_in, rows_in, start_time, {"status": "ok"})
    return df


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет опциональные колонки, если их нет.
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()
    for column_name in ["city", "bib_number", "distance_km"]:
        if column_name not in df_out.columns:
            df_out[column_name] = pd.NA

    extra_metrics = {
        "added_city": "city" not in df.columns,
        "added_bib_number": "bib_number" not in df.columns,
        "added_distance_km": "distance_km" not in df.columns,
    }
    _log_step("ensure_optional_columns", rows_in, len(df_out), start_time, extra_metrics)
    return df_out


def normalize_core_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализация строк.
    Делает: схлопывает пробелы, trim, пустые строки -> NA; surname/name -> upper.
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()

    for column_name in ["surname", "name", "city"]:
        if column_name not in df_out.columns:
            continue

        series = df_out[column_name].astype("string")
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()
        series = series.replace("", pd.NA)

        if column_name in {"surname", "name"}:
            series = series.str.upper()

        df_out[column_name] = series

    if df_out["surname"].isna().any() or df_out["name"].isna().any():
        raise ValueError("Nulls remain in surname or name after normalization")

    null_city_count = int(df_out["city"].isna().sum()) if "city" in df_out.columns else 0
    _log_step(
        "normalize_core_strings",
        rows_in,
        len(df_out),
        start_time,
        {"null_city": null_city_count},
    )
    return df_out


def build_person_base_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Принимает: df с surname/name (UPPER, не NA).
    Возвращает: df с person_ordered_key и person_set_key.
    Делает: два ключа, где person_set_key инвариантен к перестановке.
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()

    surname_series = df_out["surname"].astype("string")
    name_series = df_out["name"].astype("string")
    if surname_series.isna().any() or name_series.isna().any():
        raise ValueError("Nulls in surname or name before build_person_base_key")

    ordered_key = surname_series + "|" + name_series
    swapped_key = name_series + "|" + surname_series
    set_key = ordered_key.where(ordered_key <= swapped_key, swapped_key)

    df_out["person_ordered_key"] = ordered_key
    df_out["person_set_key"] = set_key  # инфвариант к перестановке

    _log_step(
        "build_person_base_key",
        rows_in,
        rows_in,
        start_time,
        {
            "nunique_ordered_key": int(ordered_key.nunique(dropna=False)),
            "nunique_set_key": int(set_key.nunique(dropna=False)),
            "swap_cases": int((ordered_key != swapped_key).sum()),
        },
    )
    return df_out



def compute_approx_birth_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет прокси года рождения approx_birth_year = year - age.

    Возвращает df с колонкой approx_birth_year (nullable Int64).
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()

    year_series = pd.to_numeric(df_out["year"], errors="coerce").astype("Int64")
    age_series = pd.to_numeric(df_out["age"], errors="coerce").astype("Int64")
    df_out["approx_birth_year"] = (year_series - age_series).astype("Int64")

    na_count = int(df_out["approx_birth_year"].isna().sum())
    unique_years = int(df_out["approx_birth_year"].nunique(dropna=False))

    _log_step(
        "compute_approx_birth_year",
        rows_in,
        rows_in,
        start_time,
        {
            "na_birth_year": na_count,
            "unique_birth_years": unique_years,
        },
    )
    return df_out

def assign_runner_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Присваивает runner_id с разукрупнением по году рождения.

    Правило MVP:
    - группируем по person_set_key
    - если в группе >1 уникальный approx_birth_year (dropna), добавляем суффикс _{year}
    - если approx_birth_year NA, суффикс _UNKNOWN
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = (
        df
        .pipe(build_person_base_key)
        .pipe(compute_approx_birth_year)
    )

    df_out = df_out.sort_values(["person_set_key", "approx_birth_year"], kind="mergesort")

    birth_nunique_per_set = df_out.groupby("person_set_key")["approx_birth_year"].transform("nunique")
    multi_birth_mask = birth_nunique_per_set > 1
    na_birth_mask = df_out["approx_birth_year"].isna()

    df_out["birth_suffix"] = ""
    df_out.loc[multi_birth_mask, "birth_suffix"] = "_" + df_out.loc[multi_birth_mask, "approx_birth_year"].astype("Int64").astype("string")
    df_out.loc[na_birth_mask, "birth_suffix"] = "_UNKNOWN"

    df_out["runner_id"] = (df_out["person_set_key"] + df_out["birth_suffix"]).astype("string")

    if df_out["runner_id"].isna().any():
        raise ValueError("Nulls in runner_id after assignment")

    event_dups_mask = df_out.duplicated(subset=["runner_id", "race_id", "year"], keep=False)
    suspicious_multi_start_rows = int(event_dups_mask.sum())

    multi_birth_group_count = int((birth_nunique_per_set > 1).groupby(df_out["person_set_key"]).any().sum())

    runner_id_counts = df_out["runner_id"].value_counts(dropna=False)

    _log_step(
        "assign_runner_id",
        rows_in,
        len(df_out),
        start_time,
        {
            "nunique_runner_id": int(df_out["runner_id"].nunique(dropna=False)),
            "max_group_size": int(runner_id_counts.max()),
            "multi_birth_group_count": multi_birth_group_count,
            "suspicious_multi_start_rows": suspicious_multi_start_rows,
            "mode": "set_key_with_birth_year",
        },
    )

    # dump_homonyms_by_birth_year(df_out.sort_index(), "outputs/homonyms_dump.txt", min_group_size=2)

    return df_out.sort_index()

def compute_city_mode_per_runner(df: pd.DataFrame) -> pd.Series:
    """
    Возвращает модальный город для runner_id,
    либо pd.NA если городов 0 или >1.
    """
    start_time = time.time()
    rows_in = len(df)

    grouped = df.groupby("runner_id")["city"]

    nunique_city = grouped.transform("nunique")
    first_city = grouped.transform("first")

    mode_city = first_city.where(nunique_city == 1, pd.NA)

    runner_with_single_city = int(
        (nunique_city == 1).groupby(df["runner_id"]).any().sum()
    )

    potential_fill_rows = int(mode_city.notna().sum())

    _log_step(
        "compute_city_mode_per_runner",
        rows_in,
        rows_in,
        start_time,
        {
            "runners_with_single_city": runner_with_single_city,
            "potential_fill_rows": potential_fill_rows,
        },
    )

    return mode_city

def fill_city_within_runner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Заполняет city внутри runner_id, если город однозначен.
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()

    city_before = df_out["city"].astype("string")

    mode_city = compute_city_mode_per_runner(df_out)

    na_before_mask = df_out["city"].isna()
    na_before = int(na_before_mask.sum())

    df_out.loc[na_before_mask, "city"] = mode_city

    na_after = int(df_out["city"].isna().sum())
    actual_filled = na_before - na_after

    # if logger.isEnabledFor(logging.DEBUG):
    #     filled_mask = na_before_mask & df_out["city"].notna()
    #     example_row_ids = df_out.index[filled_mask].to_list()[:5]
    #
    #     header = "fill_city_within_runner debug\n"
    #     header += f"  filled_rows: {actual_filled}\n"
    #     header += "  examples:\n"
    #
    #     lines: list[str] = []
    #     for row_id in example_row_ids:
    #         runner_id_value = df_out.at[row_id, "runner_id"] if "runner_id" in df_out.columns else pd.NA
    #         race_id_value = df_out.at[row_id, "race_id"] if "race_id" in df_out.columns else pd.NA
    #         year_value = df_out.at[row_id, "year"] if "year" in df_out.columns else pd.NA
    #         birth_year_value = df_out.at[
    #             row_id, "approx_birth_year"] if "approx_birth_year" in df_out.columns else pd.NA
    #
    #         before_value = city_before.at[row_id]
    #         after_value = df_out.at[row_id, "city"]
    #
    #         lines.append(
    #             "    "
    #             f"row_id={row_id}, "
    #             f"runner_id={runner_id_value}, "
    #             f"birth_year={birth_year_value}, "
    #             f"race_id={race_id_value}, "
    #             f"year={year_value}, "
    #             f"city: {before_value} -> {after_value}, "
    #             f"source=mode_within_runner"
    #         )
    #
    #     logger.debug(header + "\n".join(lines))

    _log_step(
        "fill_city_within_runner",
        rows_in,
        len(df_out),
        start_time,
        {
            "actual_filled_rows": actual_filled,
            "remaining_na_city": na_after,
        },
    )

    return df_out

def deduplicate_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет строгие дубли (точные повторы по ключу).
    Ключ динамический: базовый + bib_number если заполнен > 50%.
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()

    key_cols = ["runner_id", "race_id", "year", "time_seconds"]

    missing_required = [column_name for column_name in key_cols if column_name not in df_out.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for deduplicate_strict: {missing_required}")

    bib_used = False
    bib_fill_rate = 0.0
    if "bib_number" in df_out.columns:
        bib_fill_rate = float(df_out["bib_number"].notna().mean())
        if bib_fill_rate > 0.5:
            key_cols.append("bib_number")
            bib_used = True

    dup_mask = df_out.duplicated(subset=key_cols, keep="first")
    removed_rows = int(dup_mask.sum())

    duplicate_groups_count = 0
    if removed_rows > 0:
        group_sizes = df_out.groupby(key_cols, dropna=False).size()
        duplicate_groups_count = int((group_sizes > 1).sum())

    df_out = df_out.loc[~dup_mask]

    _log_step(
        "deduplicate_strict",
        rows_in,
        len(df_out),
        start_time,
        {
            "removed_rows": removed_rows,
            "duplicate_groups": duplicate_groups_count,
            "bib_fill_rate": f"{bib_fill_rate:.3f}",
            "bib_used": bib_used,
            "key_cols": "|".join(key_cols),
        },
    )

    return df_out.sort_index()

def flag_soft_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Логирует soft-подозрительные случаи (не удаляет строки).
    Добавляет флаг soft_dupe_flag для дебага.
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = df.copy()

    # --------------------------------------------------
    # 1. Несколько записей runner_id в одном забеге
    # --------------------------------------------------
    event_group_size = (
        df_out
        .groupby(["runner_id", "race_id", "year"], observed=True)
        .transform("size")
    )
    event_multi_mask = event_group_size > 1

    event_multi_rows = int(event_multi_mask.sum())
    event_multi_runners = int(df_out.loc[event_multi_mask, "runner_id"].nunique())

    # --------------------------------------------------
    # 2. Несколько городов у одного runner_id
    # --------------------------------------------------
    city_nunique = (
        df_out
        .groupby("runner_id", observed=True)["city"]
        .transform("nunique")
    )
    multi_city_mask = city_nunique > 1

    multi_city_rows = int(multi_city_mask.sum())
    multi_city_runners = int(df_out.loc[multi_city_mask, "runner_id"].nunique())

    # --------------------------------------------------
    # Общий флаг
    # --------------------------------------------------
    df_out["soft_dupe_flag"] = event_multi_mask | multi_city_mask
    flagged_rows = int(df_out["soft_dupe_flag"].sum())

    # --------------------------------------------------
    # DEBUG: примеры подозрительных случаев
    # --------------------------------------------------
    # if logger.isEnabledFor(logging.DEBUG) and flagged_rows > 0:
    #     runner_ids_to_show = (
    #         df_out.loc[df_out["soft_dupe_flag"], "runner_id"]
    #         .dropna()
    #         .astype("string")
    #         .unique()
    #         .tolist()
    #     )[:5]
    #
    #     lines: list[str] = []
    #     lines.append("flag_soft_duplicates groups debug")
    #     lines.append(f"  shown_runner_ids: {len(runner_ids_to_show)}")
    #
    #     for runner_id_value in runner_ids_to_show:
    #         group_df = df_out[df_out["runner_id"] == runner_id_value]
    #
    #         total_rows = int(len(group_df))
    #         city_unique = int(group_df["city"].nunique(dropna=False)) if "city" in group_df.columns else 0
    #         bib_unique = int(group_df["bib_number"].nunique(dropna=False)) if "bib_number" in group_df.columns else 0
    #         birth_year_unique = int(group_df["approx_birth_year"].nunique(dropna=False)) if "approx_birth_year" in group_df.columns else 0
    #
    #         event_sizes = (
    #             group_df
    #             .groupby(["race_id", "year"], observed=True)
    #             .size()
    #         )
    #         multi_start_event_count = int((event_sizes > 1).sum())
    #
    #         race_year_list = (
    #             group_df[["race_id", "year"]]
    #             .drop_duplicates()
    #             .sort_values(["race_id", "year"])
    #             .apply(lambda row: f"{row['race_id']}:{row['year']}", axis=1)
    #             .tolist()
    #         )
    #
    #         race_year_pairs = (
    #             group_df[["race_id", "year"]]
    #             .drop_duplicates()
    #             .sort_values(["race_id", "year"])
    #         )
    #         race_year_list = [f"{race_id}:{year}" for race_id, year in zip(race_year_pairs["race_id"], race_year_pairs["year"])]
    #
    #         city_list = []
    #         if "city" in group_df.columns:
    #             city_list = (
    #                 group_df["city"]
    #                 .astype("string")
    #                 .fillna("NA")
    #                 .drop_duplicates()
    #                 .sort_values()
    #                 .tolist()
    #             )
    #
    #         bib_list = []
    #         if "bib_number" in group_df.columns:
    #             bib_list = (
    #                 group_df["bib_number"]
    #                 .astype("string")
    #                 .fillna("NA")
    #                 .drop_duplicates()
    #                 .sort_values()
    #                 .tolist()
    #             )
    #
    #         lines.append(
    #             "  "
    #             f"runner_id={runner_id_value} "
    #             f"rows={total_rows} "  # сколько строк всего у этого runner_id
    #             f"unique_birth_years={birth_year_unique} "  # сколько РАЗНЫХ approx_birth_year внутри runner_id
    #             # >1 означает возможное слияние разных людей
    #             f"unique_cities={city_unique} "  # сколько РАЗНЫХ городов у runner_id
    #             # >1 означает конфликт города
    #             f"unique_bib_numbers={bib_unique} "  # сколько разных стартовых номеров
    #             # >1 сигнал разных стартов
    #             f"multi_start_events={multi_start_event_count}"
    #             # сколько (race_id, year), где у runner_id >1 строки
    #         )
    #
    #         lines.append(f"    events: {race_year_list}")
    #         lines.append(f"    cities: {city_list}")
    #         lines.append(f"    bib_numbers: {bib_list}")
    #
    #     logger.debug("\n".join(lines))

    _log_step(
        "flag_soft_duplicates",
        rows_in,
        len(df_out),
        start_time,
        {
            "event_multi_rows": event_multi_rows,  # число строк, где один runner_id имеет >1 записи
            # в одном и том же (race_id, year)
            "event_multi_runners": event_multi_runners,  # число runner_id, у которых есть такие мульти-старты
            "multi_city_rows": multi_city_rows,  # число строк, принадлежащих runner_id
            # с >1 различными значениями city
            "multi_city_runners": multi_city_runners,  # число runner_id с конфликтом по city
            "flagged_rows": flagged_rows,  # число строк, которые попали
            # хотя бы в одну подозрительную группу
            # (event_multi OR multi_city)
        })

    return df_out


def normalize_city_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Нормализует значения city с помощью CityNormalizer.
    Не заполняет пропуски, не использует runner_id.
    """
    start_time = time.time()
    rows_in = len(df)

    if "city" not in df.columns:
        _log_step(
            "normalize_city_names",
            rows_in,
            rows_in,
            start_time,
            {"skipped": True},
        )
        return df

    df_out = df.copy()

    normalizer = CityNormalizer()

    city_before = df_out["city"].astype("string")

    # применяем нормализацию ко всем значениям
    df_out["city"] = (
        city_before
        .map(normalizer.normalize_city_value)
        .astype("string")
    )

    changed_rows = int((city_before != df_out["city"]).sum())
    unique_before = int(city_before.nunique(dropna=False))
    unique_after = int(df_out["city"].nunique(dropna=False))

    _log_step(
        "normalize_city_names",
        rows_in,
        len(df_out),
        start_time,
        {
            "changed_rows": changed_rows,
            "unique_before": unique_before,
            "unique_after": unique_after,
        },
    )

    dump_unrecognized_cities(
        df_out,
        "outputs/unrecognized_cities_top100.txt",
    )

    return df_out


def drop_soft_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет строки, попавшие в soft_dupe_flag.

    TODO: Вернуться к этим строкам позже, когда будет отдельная кластеризация однофамильцев.
    Сейчас осознанно выкидываем их, чтобы не тащить структурный шум в модель.
    """
    start_time = time.time()
    rows_in = len(df)

    if "soft_dupe_flag" not in df.columns:
        raise ValueError("soft_dupe_flag is missing. Run flag_soft_duplicates before drop_soft_duplicates.")

    flagged_rows = int(df["soft_dupe_flag"].sum())
    df_out = df.loc[~df["soft_dupe_flag"]].copy()

    _log_step(
        "drop_soft_duplicates",
        rows_in,
        len(df_out),
        start_time,
        {
            "removed_rows": flagged_rows,
            "removed_share": round(flagged_rows / max(rows_in, 1), 6),
        },
    )

    return df_out


import math


def add_features(df: pd.DataFrame, *, age_center: float, age_scale: float) -> pd.DataFrame:
    """
    Добавляет модельные признаки.

    Вход:
    - time_seconds: время финиша в секундах (строго > 0)
    - age: возраст (не NA)
    - параметры нормировки возраста: age_center, age_scale

    Выход:
    - Y: log-время, Y = ln(time_seconds)
    - x: нормированный возраст, x = (age - age_center) / age_scale

    Инварианты:
    - Y конечен
    - x конечен
    """
    start_time = time.time()
    rows_in = len(df)

    if "time_seconds" not in df.columns:
        raise ValueError("Missing column: time_seconds")
    if "age" not in df.columns:
        raise ValueError("Missing column: age")

    if age_scale == 0:
        raise ValueError("age_scale must be non-zero")

    df_out = df.copy()

    time_seconds = pd.to_numeric(df_out["time_seconds"], errors="coerce")
    age_years = pd.to_numeric(df_out["age"], errors="coerce")

    invalid_time_mask = time_seconds.isna() | (time_seconds <= 0)
    invalid_age_mask = age_years.isna()

    invalid_time_count = int(invalid_time_mask.sum())
    invalid_age_count = int(invalid_age_mask.sum())

    if invalid_time_count > 0:
        raise ValueError(f"Invalid time_seconds (NA or <=0): {invalid_time_count} rows")
    if invalid_age_count > 0:
        raise ValueError(f"Invalid age (NA): {invalid_age_count} rows")

    df_out["Y"] = time_seconds.map(math.log).astype(float)
    df_out["x"] = ((age_years - float(age_center)) / float(age_scale)).astype(float)

    nonfinite_y = int((~pd.Series(df_out["Y"]).map(math.isfinite)).sum())
    nonfinite_x = int((~pd.Series(df_out["x"]).map(math.isfinite)).sum())

    if nonfinite_y > 0:
        raise ValueError(f"Non-finite Y: {nonfinite_y} rows")
    if nonfinite_x > 0:
        raise ValueError(f"Non-finite x: {nonfinite_x} rows")

    _log_step(
        "add_features",
        rows_in,
        len(df_out),
        start_time,
        {
            "age_center": age_center,
            "age_scale": age_scale,
        },
    )

    return df_out


# =============================================================================
# Оркестратор
# =============================================================================

class Preprocessor:
    def __init__(self, config: dict[str, Any]) -> None:
        preprocessing_config = config.get("preprocessing", {})
        self.age_center = float(preprocessing_config.get("age_center", 35.0))
        self.age_scale = float(preprocessing_config.get("age_scale", 10.0))

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        rows_in = len(df)

        result = (
            df
            .pipe(validate_input)
            .pipe(ensure_optional_columns)
            .pipe(normalize_core_strings)
            .pipe(assign_runner_id)
            .pipe(normalize_city_names)
            .pipe(fill_city_within_runner)
            .pipe(deduplicate_strict)
            .pipe(flag_soft_duplicates)
            .pipe(drop_soft_duplicates)  # TODO: вернуть позже через кластеризацию однофамильцев
            .pipe(flag_soft_duplicates)
            .pipe(add_features, age_center=self.age_center, age_scale=self.age_scale)
        )

        _log_step("preprocess_total", rows_in, len(result), start_time)
        return result
