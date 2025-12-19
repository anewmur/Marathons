
from __future__ import annotations

import math

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
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


def hhmmss_tick(seconds_value: float, pos: int) -> str:
    return seconds_to_hhmmss(seconds_value)


def main() -> None:
    # Если PyCharm ломает plt.show(), раскомментируй:
    # matplotlib.use("TkAgg")

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()

    confidence = 0.95
    tail = (1.0 - confidence) / 2.0
    result = model.predict_with_uncertainty(
        race_id="Белые ночи",
        gender="M",
        age=40.0,
        year=2025,
        confidence=confidence,
        method="monte_carlo",
        n_samples=10000,
    )

    time_samples = np.asarray(result["samples"]["time_samples"], dtype=float)
    time_pred = float(result["time_pred"])
    time_lower = float(result["time_lower"])
    time_upper = float(result["time_upper"])

    valid_mask = np.isfinite(time_samples)
    time_samples = time_samples[valid_mask]

    plt.figure(figsize=(10, 6))
    plt.hist(time_samples, bins=100, density=True, alpha=0.7, edgecolor="black")

    plt.axvline(
        time_pred,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {seconds_to_hhmmss(time_pred)}",
    )
    plt.axvline(
        time_lower,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"{tail*100:.0f}%: {seconds_to_hhmmss(time_lower)}",
    )
    plt.axvline(
        time_upper,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"{(1.0-tail)*100:.0f}%: {seconds_to_hhmmss(time_upper)}",
    )

    axis = plt.gca()
    axis.xaxis.set_major_formatter(FuncFormatter(hhmmss_tick))

    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.title("Predicted Time Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

