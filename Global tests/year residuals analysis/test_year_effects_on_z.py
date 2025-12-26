
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

from MarathonAgeModel import MarathonModel, easy_logging
from trace_reference_builder import get_reference_log

logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.pyplot").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

def run_year_effects_signal_checks(stats: pd.DataFrame) -> None:
    required_columns = ["race_id", "year", "mean_residual_z", "mean_age", "std_age", "count"]
    missing_columns = [name for name in required_columns if name not in stats.columns]
    if missing_columns:
        raise RuntimeError(f"stats missing columns: {missing_columns}")

    stats_valid = stats.copy()
    stats_valid = stats_valid[np.isfinite(stats_valid["mean_residual_z"].to_numpy())].copy()
    stats_valid = stats_valid[np.isfinite(stats_valid["mean_age"].to_numpy())].copy()
    stats_valid = stats_valid[np.isfinite(stats_valid["std_age"].to_numpy())].copy()

    groups_total = int(len(stats_valid))
    print(f"Race-year groups available: {groups_total}")

    years_per_race = stats_valid.groupby("race_id")["year"].nunique()
    races_with_2plus_years = years_per_race[years_per_race >= 2]
    print(f"Races with >=2 years: {int(len(races_with_2plus_years))}")

    if int(len(races_with_2plus_years)) == 0:
        print("ANOVA within race-years: skipped (no race has >=2 years in this slice).")

    min_groups_for_regression = 20
    if groups_total < min_groups_for_regression:
        print(
            "Regression check: skipped "
            f"(need >= {min_groups_for_regression} race-year groups, got {groups_total})."
        )
        return

    feature_matrix = np.column_stack(
        [
            np.ones(groups_total, dtype=float),
            stats_valid["mean_age"].astype(float).to_numpy(),
            stats_valid["std_age"].astype(float).to_numpy(),
        ]
    )
    target_vector = stats_valid["mean_residual_z"].astype(float).to_numpy()

    coef, _, _, _ = np.linalg.lstsq(feature_matrix, target_vector, rcond=None)
    fitted = feature_matrix @ coef
    residuals = target_vector - fitted

    ss_total = float(np.sum((target_vector - float(np.mean(target_vector))) ** 2))
    ss_res = float(np.sum(residuals**2))
    r2 = 1.0 - ss_res / ss_total if ss_total > 0 else float("nan")

    print("Regression check: mean_residual_z ~ 1 + mean_age + std_age")
    print(f"bias ≈ {coef[0]:+.4f} + ({coef[1]:+.4f})*mean_age + ({coef[2]:+.4f})*std_age")
    print(f"R^2 = {r2:.3f}")


def compute_z_residuals(model: MarathonModel, use_train: bool) -> pd.DataFrame:
    """
    Считаем остатки на нормированной шкале:
      r = Z - h(x)

    Требует, чтобы в фрейме были колонки:
      race_id, year, gender, x, Y (или Z)
    """
    if use_train:
        if getattr(model, "train_frame_loy", None) is None:
            raise RuntimeError("train_frame_loy отсутствует. Запусти model.run().")
        data = model.train_frame_loy.copy()
        logger.info("Using train_frame_loy: %d rows", int(len(data)))
    else:
        if getattr(model, "df_clean", None) is None:
            raise RuntimeError("df_clean отсутствует. Запусти model.run().")
        if getattr(model, "validation_year", None) is None:
            raise RuntimeError("validation_year отсутствует.")
        data = model.df_clean.copy()
        data = data[data["year"] == int(model.validation_year)].copy()
        data = data[data["status"] == "OK"].copy()
        logger.info("Using validation slice: %d rows", int(len(data)))

    # Проверяем базовые колонки
    required_columns = ["race_id", "year", "gender", "x", "Y"]
    missing_columns = [column_name for column_name in required_columns if column_name not in data.columns]
    if missing_columns:
        raise RuntimeError(f"compute_z_residuals: missing columns: {missing_columns}")

    # Если Z нет, вычисляем: Z = Y - ln(R^use)
    if "Z" not in data.columns:
        logger.info("Column Z not found, computing Z = Y - ln(R^use)")

        trace_references = getattr(model, "trace_references", None)
        if trace_references is None or len(trace_references) == 0:
            raise RuntimeError("trace_references is missing")

        # Вычисляем Z для каждой строки
        z_values = []
        for idx, row in data.iterrows():
            race_id = str(row["race_id"])
            gender = str(row["gender"])
            y_value = float(row["Y"])
            year = int(row["year"])

            try:
                reference_log = get_reference_log(
                    trace_references=trace_references,
                    race_id=race_id,
                    gender=gender,
                    _year=year,
                )
                z_value = y_value - reference_log
                z_values.append(z_value)
            except KeyError:
                logger.warning(f"Reference not found for ({race_id}, {gender}), skipping row")
                z_values.append(np.nan)

        data["Z"] = z_values

        # Фильтруем строки где Z не вычислился
        data = data[data["Z"].notna()].copy()
        logger.info("After computing Z: %d rows", int(len(data)))

    age_models = getattr(model, "age_spline_models", None)
    if not age_models:
        raise RuntimeError("age_spline_models пусто. Запусти model.run().")

    parts: list[pd.DataFrame] = []

    for gender in ["M", "F"]:
        gender_data = data[data["gender"] == gender].copy()
        if len(gender_data) == 0:
            continue
        if gender not in age_models:
            continue

        x_values = gender_data["x"].astype(float).to_numpy()
        z_values = gender_data["Z"].astype(float).to_numpy()

        h_values = age_models[gender].predict_h(x_values)
        residual_z = z_values - h_values

        part = pd.DataFrame(
            {
                "race_id": gender_data["race_id"].to_numpy(),
                "year": gender_data["year"].astype(int).to_numpy(),
                "gender": gender,
                "x": x_values,
                "Z": z_values,
                "h": h_values,
                "residual_z": residual_z,
                "age": gender_data["age"].astype(float).to_numpy() if "age" in gender_data.columns else np.nan,
            }
        )
        parts.append(part)

    if not parts:
        raise RuntimeError("compute_z_residuals: no rows produced")

    result = pd.concat(parts, ignore_index=True)
    logger.info("Computed Z residuals: %d rows", int(len(result)))
    logger.info("Mean residual_z: %.6f", float(result["residual_z"].mean()))
    logger.info("Std  residual_z: %.6f", float(result["residual_z"].std()))
    return result


def analyze_race_year(residuals: pd.DataFrame) -> pd.DataFrame:
    stats = (
        residuals
        .groupby(["race_id", "year"], as_index=False)
        .agg(
            count=("residual_z", "count"),
            mean_residual_z=("residual_z", "mean"),
            median_residual_z=("residual_z", "median"),
            std_residual_z=("residual_z", "std"),
            mean_age=("age", "mean"),
            std_age=("age", "std"),
        )
    )
    stats["se_residual_z"] = stats["std_residual_z"] / np.sqrt(stats["count"].clip(lower=1))
    stats["abs_mean_residual_z"] = stats["mean_residual_z"].abs()
    stats = stats.sort_values("abs_mean_residual_z", ascending=False).reset_index(drop=True)
    return stats


def plot_boxplot(residuals: pd.DataFrame, output_path: Path, min_observations: int, max_races_to_plot: int) -> None:
    work = residuals.copy()
    work["race_year"] = work["race_id"].astype(str) + "_" + work["year"].astype(str)

    counts = work.groupby("race_year").size()
    valid_race_years = counts[counts >= int(min_observations)].index
    filtered = work[work["race_year"].isin(valid_race_years)].copy()
    if len(filtered) == 0:
        logger.warning("No data after filtering.")
        return

    race_counts = filtered.groupby("race_id").size().sort_values(ascending=False)
    top_races = race_counts.head(int(max_races_to_plot)).index
    plot_data = filtered[filtered["race_id"].isin(top_races)].copy()
    plot_data = plot_data.sort_values(["race_id", "year"])

    plt.figure(figsize=(18, 8))
    sns.boxplot(data=plot_data, x="race_year", y="residual_z")
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1, alpha=0.7)
    plt.xlabel("Race-Year")
    plt.ylabel("Residual on Z scale:  Z - h(x)")
    plt.title("Residuals by Race-Year on Z scale")
    plt.xticks(rotation=90, fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Plot saved to: %s", str(output_path))


def main(validation_year, use_train= True) -> None:
    easy_logging(True)

    if use_train:
        split_tag = "train_loy"
        year_tag = "all_years"
    else:
        split_tag = "holdout"
        year_tag = str(int(validation_year))

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
        validation_year=validation_year
    )
    model.run()

    output_dir = Path(f"outputs/year_analysis_{split_tag}_{year_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # use_train=True берёт model.train_frame_loy (обучающая выборка после LOY, то есть без validation_year).
    # Это нужно, когда хочется увидеть картину по многим годам и сравнивать годы внутри одной трассы.
    #
    # use_train=False берёт model.df_clean, режет ровно year == model.validation_year и оставляет только status == "OK".
    # Это нужно, когда хочется честно оценить out-of-sample на holdout-году.

    residuals = compute_z_residuals(model=model, use_train=use_train)

    if len(residuals) == 0:
        logger.error("No residuals computed for validation year %d", validation_year)
        return

    stats = analyze_race_year(residuals=residuals)

    run_year_effects_signal_checks(stats=stats)


    residuals_path = output_dir / f"residuals_{split_tag}_{year_tag}.csv"
    stats_path = output_dir / f"stats_{split_tag}_{year_tag}.csv"
    plot_path = output_dir / f"residuals_{split_tag}_{year_tag}_boxplot.png"

    residuals.to_csv(residuals_path, index=False)
    stats.to_csv(stats_path, index=False)
    plot_boxplot(residuals=residuals, output_path=plot_path, min_observations=10, max_races_to_plot=15)

    logger.info("=" * 80)
    if use_train:
        logger.info("TRAIN LOY analysis (holdout year %d excluded)", int(validation_year))
    else:
        logger.info("HOLDOUT analysis for year %d (out-of-sample slice)", int(validation_year))

    logger.info("=" * 80)
    logger.info("Total observations: %d", len(residuals))
    logger.info("Mean residual_z: %.6f", float(residuals["residual_z"].mean()))
    logger.info("Median residual_z: %.6f", float(residuals["residual_z"].median()))
    logger.info("Std residual_z: %.6f", float(residuals["residual_z"].std()))

    # Bias в секундах для марафона 3.5 часа
    mean_res = float(residuals["residual_z"].mean())
    time_seconds = 12600
    bias_seconds = time_seconds * (np.exp(mean_res) - 1)
    bias_minutes = bias_seconds / 60

    logger.info("")
    logger.info("For marathon ~3.5 hours (%d seconds):", time_seconds)
    logger.info("  Bias: %.1f seconds (%.1f minutes)", bias_seconds, bias_minutes)
    logger.info("  Relative bias: %.1f%%", 100 * (np.exp(mean_res) - 1))

    logger.info("")
    logger.info("=" * 80)
    logger.info("TOP 10 BY RACE")
    logger.info("=" * 80)
    top_rows = stats.head(10)
    for _, row in top_rows.iterrows():
        logger.info(
            "%s %d: mean=%.6f (±%.6f), n=%d, mean_age=%.2f",
            str(row["race_id"]),
            int(row["year"]),
            float(row["mean_residual_z"]),
            float(row["se_residual_z"]),
            int(row["count"]),
            float(row["mean_age"]) if np.isfinite(row["mean_age"]) else float("nan"),
        )
    logger.info("=" * 80)

    # Итоговый вывод
    if mean_res > 0.15:
        logger.info("✗ STRONG POSITIVE BIAS: Model systematically UNDERESTIMATES time")
        logger.info("  Recommendation: Add year effects to the model")
    elif mean_res > 0.05:
        logger.info("⚠ MODERATE POSITIVE BIAS: Model underestimates time")
        logger.info("  Recommendation: Consider adding year effects")
    elif mean_res < -0.05:
        logger.info("⚠ NEGATIVE BIAS: Model overestimates time")
    else:
        logger.info("✓ SMALL BIAS: Model performs acceptably")


if __name__ == "__main__":
    main(validation_year=2025, use_train=True)