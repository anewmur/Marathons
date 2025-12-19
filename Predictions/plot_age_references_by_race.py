# file: Predictions/plot_age_references_by_race.py
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def seconds_to_hhmmss(seconds_value: float) -> str:
    if seconds_value is None:
        return ""
    if isinstance(seconds_value, float) and math.isnan(seconds_value):
        return ""
    if seconds_value <= 0:
        return "00:00:00"

    total_seconds = int(round(float(seconds_value)))
    hours_value = total_seconds // 3600
    minutes_value = (total_seconds % 3600) // 60
    seconds_rest = total_seconds % 60
    return f"{hours_value:02d}:{minutes_value:02d}:{seconds_rest:02d}"


def hhmmss_tick(x_value: float, pos: int) -> str:
    return seconds_to_hhmmss(x_value)


def main() -> None:

    excel_path = Path(r"C:\Users\andre\github\Marathons\test_outputs\age_references.xlsx")
    if not excel_path.exists():
        raise FileNotFoundError(f"Not found: {excel_path}")

    data = pd.read_excel(excel_path)

    required_columns = ["race_id", "gender", "age", "age_median_time"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {excel_path.name}: {missing_columns}")

    data = data.dropna(subset=["race_id", "gender", "age", "age_median_time"]).copy()
    data["age"] = data["age"].astype(int)
    data["age_median_time"] = pd.to_numeric(data["age_median_time"], errors="coerce")
    data = data.dropna(subset=["age_median_time"]).copy()

    race_ids = sorted(data["race_id"].unique().tolist())
    genders = ["M", "F"]

    for race_id in race_ids:
        race_frame = data.loc[data["race_id"] == race_id].copy()

        plt.figure(figsize=(10, 6))
        for gender_value in genders:
            gender_frame = race_frame.loc[race_frame["gender"] == gender_value].copy()
            if len(gender_frame) == 0:
                continue

            gender_frame = gender_frame.sort_values("age")
            ages = gender_frame["age"].to_numpy(dtype=int)
            times = gender_frame["age_median_time"].to_numpy(dtype=float)

            valid_mask = np.isfinite(times)
            plt.plot(ages[valid_mask], times[valid_mask], marker="o", linewidth=1.5, label=f"{gender_value} median")

        axis = plt.gca()
        axis.yaxis.set_major_formatter(FuncFormatter(hhmmss_tick))

        plt.xlabel("Age (years)")
        plt.ylabel("Time (HH:MM:SS)")
        plt.title(f"Эмпирическое медианное время в зависимости от возраста, трасса={race_id}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Один график на окно. Если окон много, можно сохранить в PNG и не показывать.
        plt.show()


if __name__ == "__main__":
    main()
