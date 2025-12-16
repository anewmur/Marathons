from __future__ import annotations

from pathlib import Path
import pandas as pd


def compress_years_to_ranges(sorted_unique_years: list[int]) -> list[tuple[int, int]]:
    if not sorted_unique_years:
        return []

    ranges: list[tuple[int, int]] = []
    range_start = sorted_unique_years[0]
    range_end = sorted_unique_years[0]

    for year_value in sorted_unique_years[1:]:
        if year_value == range_end + 1:
            range_end = year_value
            continue
        ranges.append((range_start, range_end))
        range_start = year_value
        range_end = year_value

    ranges.append((range_start, range_end))
    return ranges


def dump_homonyms_by_birth_year(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    min_group_size: int = 2,
) -> None:
    required_columns = ["person_ordered_key", "approx_birth_year"]
    missing_columns = [column_name for column_name in required_columns if column_name not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for dump: {missing_columns}")

    key_counts = df["person_ordered_key"].value_counts(dropna=False)
    keys_to_dump = key_counts[key_counts >= min_group_size].index.tolist()

    rows_for_sort: list[tuple[int, str]] = []

    for person_key in keys_to_dump:
        group_df = df[df["person_ordered_key"] == person_key]
        years_series = group_df["approx_birth_year"].dropna().astype("int64")

        if years_series.empty:
            rows_for_sort.append((0, f"{person_key} [UNKNOWN]"))
            continue

        year_counts = years_series.value_counts().sort_index()
        sorted_unique_years = year_counts.index.to_list()
        year_ranges = compress_years_to_ranges(sorted_unique_years)

        parts: list[str] = []
        total_items = 0

        for start_year, end_year in year_ranges:
            if start_year == end_year:
                count_in_range = int(year_counts.loc[start_year])
                token = f"{start_year}"
            else:
                count_in_range = int(year_counts.loc[start_year:end_year].sum())
                token = f"{start_year}-{end_year}"

            total_items += count_in_range
            for _ in range(count_in_range):
                parts.append(token)

        rows_for_sort.append((total_items, f"{person_key} [{', '.join(parts)}]"))

    rows_for_sort.sort(key=lambda pair: pair[0], reverse=True)
    lines = [line for _, line in rows_for_sort]

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
