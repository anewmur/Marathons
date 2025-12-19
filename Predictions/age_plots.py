import math
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter
from MarathonAgeModel import MarathonModel

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

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()

    race_id = "Белые ночи"
    year_value = 2025
    confidence_value = 0.95

    ages = np.arange(20, 71, 1, dtype=float)
    genders = ["M", "F"]

    plt.figure(figsize=(10, 6))

    for gender_value in genders:
        result = model.predict_with_uncertainty(
            race_id=race_id,
            gender=gender_value,
            age=ages,
            confidence=confidence_value,
            year=year_value,
        )

        time_pred = np.asarray(result["time_pred"], dtype=float)
        time_lower = np.asarray(result["time_lower"], dtype=float)
        time_upper = np.asarray(result["time_upper"], dtype=float)

        valid_mask = np.isfinite(time_pred) & np.isfinite(time_lower) & np.isfinite(time_upper)
        ages_valid = ages[valid_mask]

        plt.plot(ages_valid, time_pred[valid_mask], label=f"{gender_value} prediction")
        plt.fill_between(ages_valid, time_lower[valid_mask], time_upper[valid_mask], alpha=0.20)

    axis = plt.gca()
    axis.yaxis.set_major_formatter(FuncFormatter(hhmmss_tick))

    plt.xlabel("Age (years)")
    plt.ylabel("Time")
    plt.title(f"Model prediction by age, race={race_id}, year={year_value}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
