from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class RaceYearStats:
    race_id: str
    year: int
    count: int
    mean_residual_z: float
    se_residual_z: float
    mean_age: float
    std_age: float


def load_residuals_z(csv_path: str) -> pd.DataFrame:
    data_frame = pd.read_csv(csv_path)
    return data_frame


def build_race_year_stats(data_frame: pd.DataFrame, min_count: int) -> list[RaceYearStats]:
    grouped = data_frame.groupby(["race_id", "year"], as_index=False)

    rows: list[RaceYearStats] = []
    for _, group_frame in grouped:
        count_value = int(len(group_frame))
        if count_value < min_count:
            continue

        residual_values = group_frame["residual_z"].to_numpy(dtype=float)
        age_values = group_frame["age"].to_numpy(dtype=float)

        mean_residual = float(np.mean(residual_values))
        std_residual = float(np.std(residual_values, ddof=1))
        se_residual = float(std_residual / math.sqrt(count_value))

        mean_age = float(np.mean(age_values))
        std_age = float(np.std(age_values, ddof=1))

        race_id_value = str(group_frame["race_id"].iloc[0])
        year_value = int(group_frame["year"].iloc[0])

        rows.append(
            RaceYearStats(
                race_id=race_id_value,
                year=year_value,
                count=count_value,
                mean_residual_z=mean_residual,
                se_residual_z=se_residual,
                mean_age=mean_age,
                std_age=std_age,
            )
        )

    return rows


def print_top_year_effects(rows: list[RaceYearStats], top_k: int) -> None:
    sorted_rows = sorted(rows, key=lambda item: abs(item.mean_residual_z), reverse=True)[:top_k]

    print("\nTOP year effects by |mean_residual_z|")
    for item in sorted_rows:
        print(
            f"{item.race_id:20s} {item.year}  "
            f"mean={item.mean_residual_z:+.3f}  se={item.se_residual_z:.3f}  "
            f"n={item.count:4d}  mean_age={item.mean_age:.1f}  std_age={item.std_age:.1f}"
        )


def anova_by_race(data_frame: pd.DataFrame, min_count: int) -> None:
    print("\nANOVA: within each race, do years differ (residual_z)?")
    for race_id_value in sorted(data_frame["race_id"].unique()):
        race_frame = data_frame.loc[data_frame["race_id"] == race_id_value].copy()
        year_values = sorted(race_frame["year"].unique())

        samples: list[np.ndarray] = []
        for year_value in year_values:
            year_frame = race_frame.loc[race_frame["year"] == year_value]
            if len(year_frame) < min_count:
                continue
            samples.append(year_frame["residual_z"].to_numpy(dtype=float))

        if len(samples) < 3:
            continue

        f_stat, p_value = stats.f_oneway(*samples)
        print(f"{race_id_value:20s}  years={len(samples):2d}  p={p_value:.3e}  F={f_stat:.2f}")


def regression_delta_on_composition(rows: list[RaceYearStats]) -> None:
    # Простейшая проверка идеи "bias зависит от mean_age и std_age"
    bias_values = np.array([item.mean_residual_z for item in rows], dtype=float)
    mean_age_values = np.array([item.mean_age for item in rows], dtype=float)
    std_age_values = np.array([item.std_age for item in rows], dtype=float)

    design_matrix = np.column_stack(
        [
            np.ones(len(rows), dtype=float),
            mean_age_values,
            std_age_values,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, bias_values, rcond=None)
    fitted = design_matrix @ coefficients
    residuals = bias_values - fitted

    ss_total = float(np.sum((bias_values - np.mean(bias_values)) ** 2))
    ss_res = float(np.sum(residuals**2))
    r2 = 1.0 - ss_res / ss_total if ss_total > 0 else float("nan")

    intercept_value = float(coefficients[0])
    coef_mean_age = float(coefficients[1])
    coef_std_age = float(coefficients[2])

    print("\nRegression check: mean_residual_z ~ 1 + mean_age + std_age")
    print(f"bias ≈ {intercept_value:+.3f} + ({coef_mean_age:+.4f})*mean_age + ({coef_std_age:+.4f})*std_age")
    print(f"R^2 = {r2:.3f}")


def main() -> None:
    csv_path = "C:\\Users\\andre\\github\\Marathons\\Global tests\\year residuals analysis\\outputs\\validation_analysis_2025\\residuals_validation_2025.csv"  # положи файл рядом или поменяй путь
    min_count = 20

    data_frame = load_residuals_z(csv_path=csv_path)

    rows = build_race_year_stats(data_frame=data_frame, min_count=min_count)
    print_top_year_effects(rows=rows, top_k=12)

    anova_by_race(data_frame=data_frame, min_count=min_count)
    regression_delta_on_composition(rows=rows)


if __name__ == "__main__":
    main()
