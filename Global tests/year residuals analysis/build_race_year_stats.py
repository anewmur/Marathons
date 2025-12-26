from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except Exception:  # scipy может отсутствовать
    scipy_stats = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SignalCheckConfig:
    min_observations_per_race_year: int = 20
    min_years_per_race: int = 2
    top_k: int = 10


def build_race_year_stats(residuals: pd.DataFrame, min_observations: int) -> pd.DataFrame:
    required_columns = ["race_id", "year", "residual_z", "age"]
    missing_columns = [column_name for column_name in required_columns if column_name not in residuals.columns]
    if missing_columns:
        raise RuntimeError(f"build_race_year_stats: missing columns: {missing_columns}")

    stats_frame = (
        residuals
        .groupby(["race_id", "year"], as_index=False)
        .agg(
            count=("residual_z", "count"),
            mean_residual_z=("residual_z", "mean"),
            std_residual_z=("residual_z", "std"),
            mean_age=("age", "mean"),
            std_age=("age", "std"),
        )
    )

    stats_frame = stats_frame[stats_frame["count"] >= int(min_observations)].copy()
    stats_frame["se_residual_z"] = stats_frame["std_residual_z"] / np.sqrt(stats_frame["count"].clip(lower=1))
    stats_frame["abs_mean_residual_z"] = stats_frame["mean_residual_z"].abs()
    stats_frame = stats_frame.sort_values("abs_mean_residual_z", ascending=False).reset_index(drop=True)

    race_mean = stats_frame.groupby("race_id")["mean_residual_z"].transform("mean")
    stats_frame["delta_centered"] = stats_frame["mean_residual_z"] - race_mean
    stats_frame["abs_delta_centered"] = stats_frame["delta_centered"].abs()

    top = stats_frame.sort_values("abs_delta_centered", ascending=False).head(10)
    print("")
    print("TOP centered year shifts within race: |delta_centered|")
    for _, row in top.iterrows():
        print(
            f"{str(row['race_id']):<18} {int(row['year'])}  "
            f"delta={float(row['delta_centered']):+.3f}  "
            f"mean={float(row['mean_residual_z']):+.3f}  "
            f"n={int(row['count']):>4}"
        )

    return stats_frame


def print_top_year_effects(stats_frame: pd.DataFrame, top_k: int) -> None:
    print("")
    print("TOP year effects by |mean_residual_z|")
    top_frame = stats_frame.head(int(top_k))
    for _, row in top_frame.iterrows():
        print(
            f"{str(row['race_id']):<18} {int(row['year'])}  "
            f"mean={float(row['mean_residual_z']):+.3f}  "
            f"se={float(row['se_residual_z']):.3f}  "
            f"n={int(row['count']):>4}  "
            f"mean_age={float(row['mean_age']):.1f}  "
            f"std_age={float(row['std_age']):.1f}"
        )


def run_anova_within_race(residuals: pd.DataFrame, stats_frame: pd.DataFrame, config: SignalCheckConfig) -> None:
    print("")
    print("ANOVA: within each race, do years differ (residual_z)?")

    if scipy_stats is None:
        print("  scipy not available, skipping ANOVA")
        return

    years_per_race = stats_frame.groupby("race_id")["year"].nunique()
    eligible_races = years_per_race[years_per_race >= int(config.min_years_per_race)].index.tolist()

    if not eligible_races:
        print("  No races with >=2 years after filtering")
        return

    for race_id in eligible_races:
        race_slice = residuals[residuals["race_id"] == race_id].copy()
        race_slice = race_slice.groupby("year").filter(lambda group: len(group) >= int(config.min_observations_per_race_year))

        years_sorted = sorted(race_slice["year"].unique().tolist())
        if len(years_sorted) < int(config.min_years_per_race):
            continue

        year_arrays: list[np.ndarray] = []
        for year_value in years_sorted:
            values = race_slice.loc[race_slice["year"] == year_value, "residual_z"].to_numpy(dtype=float)
            year_arrays.append(values)

        if len(year_arrays) < int(config.min_years_per_race):
            continue

        test_result = scipy_stats.f_oneway(*year_arrays)
        print(f"  {race_id}: p={float(test_result.pvalue):.3g}  (years={len(years_sorted)})")


def run_group_regression(stats_frame: pd.DataFrame) -> None:
    print("")
    print("Regression check: mean_residual_z ~ 1 + mean_age + std_age")

    if len(stats_frame) < 3:
        print(f"  Too few race-year groups: {len(stats_frame)}. R^2 will be unstable.")
        return

    response_vector = stats_frame["mean_residual_z"].to_numpy(dtype=float)

    design_matrix = np.column_stack(
        [
            np.ones(len(stats_frame), dtype=float),
            stats_frame["mean_age"].to_numpy(dtype=float),
            stats_frame["std_age"].to_numpy(dtype=float),
        ]
    )

    coef, _, _, _ = np.linalg.lstsq(design_matrix, response_vector, rcond=None)
    fitted = design_matrix @ coef

    ss_res = float(np.sum((response_vector - fitted) ** 2))
    ss_tot = float(np.sum((response_vector - float(np.mean(response_vector))) ** 2))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot

    intercept, coef_mean_age, coef_std_age = float(coef[0]), float(coef[1]), float(coef[2])
    print(f"bias ≈ {intercept:+.4f} + ({coef_mean_age:+.4f})*mean_age + ({coef_std_age:+.4f})*std_age")
    print(f"R^2 = {r2:.3f}")


def run_year_effects_signal_checks(residuals: pd.DataFrame, config: SignalCheckConfig) -> None:
    stats_frame = build_race_year_stats(
        residuals=residuals,
        min_observations=int(config.min_observations_per_race_year),
    )

    race_year_groups = int(len(stats_frame))
    races_with_years = int((stats_frame.groupby("race_id")["year"].nunique() >= int(config.min_years_per_race)).sum())

    print(f"Race-year groups available: {race_year_groups}")
    print(f"Races with >={config.min_years_per_race} years: {races_with_years}")

    print_top_year_effects(stats_frame=stats_frame, top_k=int(config.top_k))
    run_anova_within_race(residuals=residuals, stats_frame=stats_frame, config=config)
    run_group_regression(stats_frame=stats_frame)


if __name__ == "__main__":
    # Здесь ты сам решаешь, какие residuals подать:
    # 1) train (много лет, годовые сравнения имеют смысл)
    # 2) validation (один год, ANOVA по годам не сработает по определению)

    from MarathonAgeModel import MarathonModel, easy_logging
    from test_year_effects_on_z import compute_z_residuals  # поправь импорт под свой файл

    easy_logging(True)

    validation_year = 2025
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
        validation_year=validation_year,
    )
    model.run()

    residuals_train = compute_z_residuals(model=model, use_train=True)
    config = SignalCheckConfig(min_observations_per_race_year=20, min_years_per_race=2, top_k=10)
    run_year_effects_signal_checks(residuals=residuals_train, config=config)
