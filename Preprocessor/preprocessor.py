from __future__ import annotations

from pathlib import Path

from utils.city_normalizer import CityNormalizer, dump_unrecognized_cities
import logging
import time
from typing import Any
import math
from dictionaries import PERSON_TOKEN_CYR_TO_LAT
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
            series = series.map(translit_person_token_to_lat)
            series = series.astype("string").str.upper()

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

def translit_person_token_to_lat(value: str | None) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None

    lowered = str(value).strip().lower()
    if not lowered:
        return None

    parts: list[str] = []
    for char in lowered:
        mapped = PERSON_TOKEN_CYR_TO_LAT.get(char)
        if mapped is None:
            parts.append(char)
        else:
            parts.append(mapped)

    return "".join(parts).upper()

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
    start_time = time.time()
    rows_in = len(df)

    required_columns = ["person_set_key", "birth_year_stable", "birth_cluster_min_year", "race_id", "year"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for assign_runner_id: {missing_columns}")

    df_out = df.copy()

    birth_year_series = df_out["birth_year_stable"].astype("Int64")

    min_year_series = df_out["birth_cluster_min_year"].astype("Int64")
    has_birth = birth_year_series.notna() & min_year_series.notna()

    df_out["birth_suffix"] = "_UNKNOWN"
    df_out.loc[has_birth, "birth_suffix"] = (
            "_" + birth_year_series.loc[has_birth].astype("string")
            + "_M" + min_year_series.loc[has_birth].astype("string")
    )

    df_out["runner_id"] = (df_out["person_set_key"] + df_out["birth_suffix"]).astype("string")

    event_dups_mask = df_out.duplicated(subset=["runner_id", "race_id", "year"], keep=False)
    suspicious_multi_start_rows = int(event_dups_mask.sum())
    runner_id_counts = df_out["runner_id"].value_counts(dropna=False)

    _log_step(
        "assign_runner_id",
        rows_in,
        len(df_out),
        start_time,
        {
            "nunique_runner_id": int(df_out["runner_id"].nunique(dropna=False)),
            "max_group_size": int(runner_id_counts.max()),
            "suspicious_multi_start_rows": suspicious_multi_start_rows,
            "mode": "set_key_with_birth_year_stable_and_cluster",
        },
    )
    return df_out

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

def drop_ambiguous_event_multi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Удаляет неразрешимые случаи многократных записей в одном событии.

    Правило:
    1) Если в (runner_id, race_id, year) >1 строк и >1 bib_number и city один, удаляем.
    2) После разукрупнения по городу любые оставшиеся event_multi тоже удаляем целиком
       (редкие случаи, не стоящие отдельной логики).
    """
    start_time = time.time()
    rows_in = len(df)

    required_columns = ["runner_id", "race_id", "year", "bib_number", "city"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for drop_ambiguous_event_multi: {missing_columns}")

    group_key = ["runner_id", "race_id", "year"]

    group_size = df.groupby(group_key, observed=True)["runner_id"].transform("size")
    bib_nunique = df.groupby(group_key, observed=True)["bib_number"].transform("nunique")
    city_nunique = df.groupby(group_key, observed=True)["city"].transform("nunique")

    ambiguous_mask = (group_size > 1) & (bib_nunique > 1) & (city_nunique == 1)
    leftover_event_multi_mask = (group_size > 1) & (~ambiguous_mask)

    removed_ambiguous = int(ambiguous_mask.sum())
    removed_leftover = int(leftover_event_multi_mask.sum())
    removed_total = removed_ambiguous + removed_leftover

    if removed_leftover > 0:
        debug_df = df.loc[leftover_event_multi_mask, [
            "runner_id", "race_id", "year", "bib_number", "city", "age", "time_seconds",
        ]]
        logger.info(
            "drop_ambiguous_event_multi: leftover event_multi rows=%d\n%s",
            removed_leftover,
            debug_df.sort_values(["runner_id", "race_id", "year"]).head(50).to_string(index=True),
        )

    df_out = df.loc[~(ambiguous_mask | leftover_event_multi_mask)].copy()

    _log_step(
        "drop_ambiguous_event_multi",
        rows_in,
        len(df_out),
        start_time,
        {
            "removed_rows": removed_total,
            "removed_ambiguous": removed_ambiguous,
            "removed_leftover": removed_leftover,
            "removed_share": round(removed_total / max(rows_in, 1), 6),
        },
    )
    return df_out

def normalize_city_names(df: pd.DataFrame, dumping_info, output_path='outputs') -> pd.DataFrame:
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

    if dumping_info:
        dump_unrecognized_cities(
            df_out,
            limit=200,
            output_path=Path(output_path, "unrecognized_cities_top.txt"),
        )

    return df_out

def add_features(
    df: pd.DataFrame,
    *,
    age_center: float,
    age_scale: float,
    age_min_global: float,
    age_max_global: float,
) -> pd.DataFrame:
    """
    Добавляет модельные признаки и фиксирует домен возраста для сплайна.

    Делает:
    - age_clamped = clip(age, [age_min_global, age_max_global])
    - Y = ln(time_seconds)
    - x = (age_clamped - age_center) / age_scale

    Вход:
    - time_seconds: строго > 0
    - age: не NA
    - age_center, age_scale: параметры нормировки
    - age_min_global, age_max_global: глобальные границы возраста из конфига

    Выход:
    - age: перезаписан на age_clamped (float)
    - Y: float
    - x: float
    """
    start_time = time.time()
    rows_in = len(df)

    if "time_seconds" not in df.columns:
        raise ValueError("Missing column: time_seconds")
    if "age" not in df.columns:
        raise ValueError("Missing column: age")

    if not (age_scale != 0.0):
        raise ValueError("age_scale must be non-zero")
    if not (age_max_global > age_min_global):
        raise ValueError(
            f"age_max_global must be > age_min_global, got {age_min_global}, {age_max_global}"
        )

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

    age_clamped = age_years.clip(lower=float(age_min_global), upper=float(age_max_global)).astype(float)

    df_out["age"] = age_clamped
    df_out["Y"] = time_seconds.map(math.log).astype(float)
    df_out["x"] = ((age_clamped - float(age_center)) / float(age_scale)).astype(float)

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
            "age_min_global": age_min_global,
            "age_max_global": age_max_global,
        },
    )

    return df_out

def build_birth_year_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит таблицу частот (person_set_key, approx_birth_year) -> count.

    Вход:
    - df с колонками person_set_key, approx_birth_year

    Выход:
    - DataFrame: person_set_key, approx_birth_year (Int64), count (int)
    """
    required_columns = ["person_set_key", "approx_birth_year"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working = df[["person_set_key", "approx_birth_year"]].copy()
    working = working.dropna(subset=["approx_birth_year"])
    working["approx_birth_year"] = working["approx_birth_year"].astype("Int64")

    counts = (
        working
        .groupby(["person_set_key", "approx_birth_year"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    return counts.sort_values(["person_set_key", "approx_birth_year"]).reset_index(drop=True)


def add_birth_cluster_id_to_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Присваивает cluster_id внутри каждого person_set_key для подряд идущих годов.

    Правило:
    - новый кластер начинается, если разница с предыдущим годом > 1

    Вход:
    - counts: person_set_key, approx_birth_year, count (отсортировано по году)

    Выход:
    - counts + cluster_id (int)
    """
    counts_out = counts.copy()

    previous_year = counts_out.groupby("person_set_key")["approx_birth_year"].shift(1)
    is_new_cluster = previous_year.isna() | ((counts_out["approx_birth_year"] - previous_year).abs() > 1)

    counts_out["cluster_id"] = is_new_cluster.groupby(counts_out["person_set_key"]).cumsum().astype(int)
    return counts_out


def pick_stable_birth_year_per_cluster(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Выбирает канонический год рождения для каждого (person_set_key, cluster_id).

    Правило:
    - берём год с максимальной count внутри кластера
    - при равенстве count берём меньший год

    Вход:
    - counts: person_set_key, approx_birth_year, count, cluster_id

    Выход:
    - DataFrame: person_set_key, cluster_id, birth_year_stable
    """
    return (
        counts
        .sort_values(
            ["person_set_key", "cluster_id", "count", "approx_birth_year"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(subset=["person_set_key", "cluster_id"], keep="first")
        .rename(columns={"approx_birth_year": "birth_year_stable"})
        [["person_set_key", "cluster_id", "birth_year_stable"]]
    )


def attach_birth_cluster_id_to_rows(df: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """
    Примерживает cluster_id к исходным строкам по (person_set_key, approx_birth_year).

    Вход:
    - df: исходные строки
    - counts: таблица частот с cluster_id

    Выход:
    - df + cluster_id
    """
    cluster_map = counts[["person_set_key", "approx_birth_year", "cluster_id"]].copy()
    df_out = df.copy()
    return df_out.merge(cluster_map, on=["person_set_key", "approx_birth_year"], how="left")


def stabilize_birth_year_within_person_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Стабилизирует год рождения внутри person_set_key без склейки далёких годов.

    Делает:
    1) считает частоты approx_birth_year внутри person_set_key
    2) разбивает годы на кластеры с шагом не более 1 (цепочкой)
    3) выбирает birth_year_stable отдельно для каждого кластера
    4) примерживает к строкам cluster_id и birth_year_stable

    Выходные колонки:
    - cluster_id (int, может быть NA если approx_birth_year NA)
    - birth_year_stable (Int64, может быть NA)
    """
    counts = build_birth_year_counts(df)
    counts = add_birth_cluster_id_to_counts(counts)

    cluster_min = (
        counts
        .groupby(["person_set_key", "cluster_id"], sort=False)["approx_birth_year"]
        .min()
        .rename("birth_cluster_min_year")
        .reset_index()
    )

    stable = pick_stable_birth_year_per_cluster(counts)

    df_out = attach_birth_cluster_id_to_rows(df, counts)
    df_out = df_out.merge(cluster_min, on=["person_set_key", "cluster_id"], how="left")

    return df_out.merge(stable, on=["person_set_key", "cluster_id"], how="left")


def stabilize_birth_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Шаг пайплайна: добавляет cluster_id и birth_year_stable, логирует метрики.

    Предусловия:
    - уже построены person_set_key и approx_birth_year

    Постусловия:
    - в df есть cluster_id и birth_year_stable
    """
    start_time = time.time()
    rows_in = len(df)

    df_out = stabilize_birth_year_within_person_key(df)

    na_stable = int(df_out["birth_year_stable"].isna().sum())
    na_cluster = int(df_out["cluster_id"].isna().sum())

    _log_step(
        "stabilize_birth_year",
        rows_in,
        len(df_out),
        start_time,
        {
            "na_birth_year_stable": na_stable,
            "na_birth_cluster_id": na_cluster,
        },
    )
    return df_out


def split_runner_id_by_city_when_conflicted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Разукрупняет runner_id по городу, если внутри runner_id встречается >1 город.

    Правило:
    - считаем nunique(city) внутри runner_id (NA не считается городом)
    - если nunique(city) > 1 и city не NA, добавляем суффикс "__CITY_{city}"
    - если city NA, оставляем runner_id как есть (пока не умеем разделять)

    Предусловия:
    - city уже нормализован и заполнен по моде внутри runner_id (насколько возможно)
    """
    start_time = time.time()
    rows_in = len(df)

    required_columns = ["runner_id", "city"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for split_runner_id_by_city_when_conflicted: {missing_columns}")

    df_out = df.copy()

    city_series = df_out["city"].astype("string")
    city_nunique = df_out.groupby("runner_id", observed=True)["city"].transform("nunique")

    conflict_mask = (city_nunique > 1) & city_series.notna()
    changed_rows = int(conflict_mask.sum())
    affected_runner_ids = int(df_out.loc[conflict_mask, "runner_id"].nunique())

    df_out.loc[conflict_mask, "runner_id"] = (
        df_out.loc[conflict_mask, "runner_id"].astype("string")
        + "__CITY_" + city_series.loc[conflict_mask]
    )

    _log_step(
        "split_runner_id_by_city_when_conflicted",
        rows_in,
        len(df_out),
        start_time,
        {
            "changed_rows": changed_rows,
            "affected_runner_ids": affected_runner_ids,
        },
    )
    event_group_size = df_out.groupby(["runner_id", "race_id", "year"], observed=True)["runner_id"].transform("size")
    left_event_multi_rows = int((event_group_size > 1).sum())
    logger.info("after_city_split: event_multi_rows=%d", left_event_multi_rows)

    return df_out


def build_gender_counts_by_runner(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит частоты gender внутри runner_id.

    Вход:
    - df с колонками runner_id, gender

    Выход:
    - DataFrame: runner_id, gender, count
    """
    required_columns = ["runner_id", "gender"]
    missing_columns = [column_name for column_name in required_columns if column_name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working = df[["runner_id", "gender"]].copy()
    working["runner_id"] = working["runner_id"].astype("string")
    working["gender"] = working["gender"].astype("string")

    working = working.dropna(subset=["runner_id", "gender"])

    counts = (
        working
        .groupby(["runner_id", "gender"], sort=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return counts


def pick_majority_gender_map(counts: pd.DataFrame) -> pd.DataFrame:
    """
    Для каждого runner_id выбирает gender по большинству.
    Если большинство нестрогое (ничья), помечает runner_id как ambiguous.

    Вход:
    - counts: runner_id, gender, count

    Выход:
    - DataFrame: runner_id, gender_majority, is_ambiguous
    """
    required_columns = ["runner_id", "gender", "count"]
    missing_columns = [column_name for column_name in required_columns if column_name not in counts.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    counts_sorted = counts.sort_values(["runner_id", "count", "gender"], ascending=[True, False, True]).copy()
    top_rows = counts_sorted.drop_duplicates(subset=["runner_id"], keep="first").copy()

    runner_ids = top_rows["runner_id"].astype("string")
    top_count = top_rows["count"].astype(int)

    second_rows = counts_sorted.copy()
    second_rows["rank"] = second_rows.groupby("runner_id")["count"].rank(method="first", ascending=False)
    second_rows = second_rows[second_rows["rank"] == 2].copy()

    second_map = second_rows.set_index("runner_id")["count"].to_dict()

    second_count = runner_ids.map(lambda runner_id_value: second_map.get(runner_id_value, 0)).astype(int)
    is_ambiguous = top_count <= second_count

    out = pd.DataFrame({
        "runner_id": runner_ids,
        "gender_majority": top_rows["gender"].astype("string"),
        "is_ambiguous": is_ambiguous,
    })
    return out


def apply_majority_gender(df: pd.DataFrame, majority_map: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
    """
    Применяет majority gender и выкидывает ambiguous runner_id.

    Возвращает:
    - df_out
    - removed_rows
    - removed_runner_ids
    - changed_rows
    """
    df_out = df.copy()
    df_out["runner_id"] = df_out["runner_id"].astype("string")
    df_out["gender"] = df_out["gender"].astype("string")

    merged = df_out.merge(majority_map, on="runner_id", how="left")

    ambiguous_mask = merged["is_ambiguous"].fillna(False)
    removed_rows = int(ambiguous_mask.sum())
    removed_runner_ids = int(merged.loc[ambiguous_mask, "runner_id"].nunique())

    kept = merged.loc[~ambiguous_mask].copy()

    before_gender = kept["gender"].astype("string")
    kept["gender"] = kept["gender_majority"].astype("string")
    changed_rows = int((before_gender != kept["gender"]).sum())

    kept = kept.drop(columns=["gender_majority", "is_ambiguous"])
    return kept, removed_rows, removed_runner_ids, changed_rows


def fix_gender_by_majority(df: pd.DataFrame, dumping_info=False, output_path='outputs') -> pd.DataFrame:
    """
    Делает пол внутри runner_id однозначным по большинству.

    Правило:
    - учитываем только значения gender из {"M","F"}, остальные считаем NA
    - для каждого runner_id считаем числа M и F
    - если есть строгое большинство: переписываем gender во всех строках этого runner_id на winner
    - если строгого большинства нет (M==F и total>0): удаляем весь runner_id

    Логи:
    - общий шаг: removed_rows, removed_runner_ids, changed_rows
    - два отчёта в outputs:
      * gender_fixes_changed.txt: группы, где были и M и F, и был победитель
      * gender_fixes_removed.txt: группы без строгого большинства
    """
    start_time = time.time()
    rows_in = len(df)

    required_columns = ["runner_id", "gender"]
    missing_columns = [column_name for column_name in required_columns if column_name not in df.columns]
    if missing_columns:
        raise ValueError(f"fix_gender_by_majority: missing columns: {missing_columns}")

    df_out = df.copy()

    gender_series = df_out["gender"].astype("string")
    gender_series = gender_series.where(gender_series.isin(["M", "F"]), pd.NA)
    df_out["gender"] = gender_series

    counts = (
        df_out
        .dropna(subset=["runner_id", "gender"])
        .groupby(["runner_id", "gender"], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )

    pivot = (
        counts
        .pivot(index="runner_id", columns="gender", values="count")
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    if "M" not in pivot.columns:
        pivot["M"] = 0
    if "F" not in pivot.columns:
        pivot["F"] = 0

    pivot["total"] = pivot["M"] + pivot["F"]
    pivot["winner"] = pd.NA
    pivot.loc[pivot["M"] > pivot["F"], "winner"] = "M"
    pivot.loc[pivot["F"] > pivot["M"], "winner"] = "F"

    removed_ids = pivot.loc[pivot["winner"].isna() & (pivot["total"] > 0), "runner_id"].astype("string")
    winner_map = (
        pivot.loc[pivot["winner"].notna(), ["runner_id", "winner"]]
        .rename(columns={"winner": "gender_major"})
        .copy()
    )

    df_out = df_out.merge(winner_map, on="runner_id", how="left")

    removed_mask = df_out["runner_id"].astype("string").isin(removed_ids)
    removed_rows = int(removed_mask.sum())
    removed_runner_ids = int(removed_ids.nunique())

    before_gender = df_out["gender"].astype("string")
    apply_mask = df_out["gender_major"].notna()
    df_out.loc[apply_mask, "gender"] = df_out.loc[apply_mask, "gender_major"]
    after_gender = df_out["gender"].astype("string")

    changed_rows = int((before_gender != after_gender).sum())

    if removed_rows > 0:
        df_out = df_out.loc[~removed_mask].copy()

    report_df = pivot.sort_values(["total", "runner_id"], ascending=[False, True]).copy()
    changed_report = report_df[(report_df["winner"].notna()) & (report_df["M"] > 0) & (report_df["F"] > 0)].copy()
    removed_report = report_df[report_df["winner"].isna() & (report_df["total"] > 0)].copy()

    if dumping_info:
        report_path = Path(output_path, 'gender_fixes.txt')
        with open(report_path, "w", encoding="utf-8") as file_handle:
            file_handle.write("ИЗМЕНЕННО (БЫЛИ ЗНАЧЕНИЯ M и Ж. ЕСТЬ БОЛЬШИНСТВО)\n")
            file_handle.write(changed_report.to_string(index=False))
            file_handle.write("\n\n")
            file_handle.write("УДАЛЕНЫ, НЕТ БОЛЬШИНСТВА\n")
            file_handle.write(removed_report.to_string(index=False))
            file_handle.write("\n")

        logger.info("fix_gender_by_majority report saved: %s", report_path)

    if len(changed_report) > 0:
        logger.info(
            "fix_gender_by_majority changed winners (top 30):\n%s",
            changed_report.head(30).to_string(index=False),
        )
    if len(removed_report) > 0:
        logger.info(
            "fix_gender_by_majority removed ambiguous (top 30):\n%s",
            removed_report.head(30).to_string(index=False),
        )

    df_out = df_out.drop(columns=["gender_major"], errors="ignore")

    _log_step(
        "fix_gender_by_majority",
        rows_in,
        len(df_out),
        start_time,
        {
            "removed_rows": removed_rows,
            "removed_runner_ids": removed_runner_ids,
            "changed_rows": changed_rows,
        },
    )

    return df_out


# =============================================================================
# Оркестратор
# =============================================================================

class Preprocessor:
    def __init__(self, config: dict[str, Any]) -> None:
        self.output_path = config.get("path_for_outputs")
        self.dumping_info = config.get("dumping_info", {})
        self.age_center =  float(config["preprocessing"]["age_center"])
        self.age_scale = float(config["preprocessing"]["age_scale"])

        self.age_min_global = float(config["age_spline_model"]["age_min_global"])
        self.age_max_global = float(config["age_spline_model"]["age_max_global"])

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        start_time = time.time()
        rows_in = len(df)

        result = (
            df
            .pipe(validate_input)
            .pipe(ensure_optional_columns)
            .pipe(normalize_core_strings)
            .pipe(build_person_base_key)
            .pipe(compute_approx_birth_year)
            .pipe(stabilize_birth_year)
            .pipe(assign_runner_id)
            .pipe(normalize_city_names, dumping_info=self.dumping_info, output_path=self.output_path)
            .pipe(fill_city_within_runner)
            .pipe(split_runner_id_by_city_when_conflicted)
            .pipe(fix_gender_by_majority, dumping_info=self.dumping_info, output_path=self.output_path)
            .pipe(deduplicate_strict)
            .pipe(drop_ambiguous_event_multi)
            .pipe(add_features,
                    age_center=self.age_center,
                    age_scale=self.age_scale,
                    age_min_global=self.age_min_global,
                    age_max_global=self.age_max_global,)
        )

        _log_step("preprocess_total", rows_in, len(result), start_time)
        return result
