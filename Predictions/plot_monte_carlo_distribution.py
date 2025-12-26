
from __future__ import annotations
import numpy as np
import matplotlib
import pandas as pd

from Predictions.plot_age_references_by_race import hhmmss_tick
from age_reference_builder import seconds_to_hhmmss

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from MarathonAgeModel import MarathonModel


def main(race_id = "Казанский марафон",
         gender = "M",
         age_target=40.0,
         year_target = 2025) -> None:

    if year_target != 2025:
        raise ValueError('Нельзя предсказывать распределение на год меньше 2025. Этот год есть в обучении')

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()

    confidence = 0.95
    tail = (1.0 - confidence) / 2.0

    age_band = 0.5
    age_min = age_target - age_band
    age_max = age_target + age_band

    result = model.predict_with_uncertainty(
        race_id=race_id,
        gender=gender,
        age=age_target,
        year=year_target,
        confidence=confidence,
        method="monte_carlo",
        n_samples=10000,
    )

    time_samples = np.asarray(result["samples"]["time_samples"], dtype=float)
    time_pred = float(result["time_pred"])
    time_lower = float(result["time_lower"])
    time_upper = float(result["time_upper"])

    time_samples = time_samples[np.isfinite(time_samples)]

    # -----------------------------
    # Реальные данные для сравнения
    # -----------------------------
    if model.df_clean is None or model.df_clean.empty:
        raise RuntimeError("model.df_clean is missing or empty after run()")

    df_real = model.df_clean

    mask = (
            (df_real["race_id"] == race_id) &
            (df_real["gender"] == gender) &
            (df_real["year"] == year_target) &
            (df_real["status"] == "OK") &
            (df_real["age"] >= age_min) &
            (df_real["age"] <= age_max)
    )

    real_times = df_real.loc[mask, "time_seconds"].astype(float).to_numpy()
    real_times = real_times[np.isfinite(real_times)]
    real_times = real_times[real_times > 0.0]

    # Если в конкретном году мало наблюдений, сравнение будет шумным
    print(f"Real sample size: n={int(real_times.size)} (age in [{age_min},{age_max}])")

    df_real["race_id_str"] = df_real["race_id"].astype("string").str.strip()
    df_real["gender_str"] = df_real["gender"].astype("string").str.strip()
    df_real["status_str"] = df_real["status"].astype("string").str.strip()
    df_real["year_int"] = pd.to_numeric(df_real["year"], errors="coerce").astype("Int64")
    df_real["age_float"] = pd.to_numeric(df_real["age"], errors="coerce").astype(float)



    base = df_real[
        df_real["race_id_str"].notna()
        & df_real["gender_str"].notna()
        & df_real["status_str"].notna()
        & df_real["year_int"].notna()
        & np.isfinite(df_real["age_float"])
        ].copy()

    print("rows total:", int(len(df_real)))
    print("rows base:", int(len(base)))

    step1 = base[base["race_id_str"] == race_id]
    print("race_id match:", int(len(step1)))
    if len(step1) == 0:
        print("top race_id values:\n", base["race_id_str"].value_counts().head(20).to_string())

    step2 = step1[step1["gender_str"] == gender]
    print("gender match:", int(len(step2)))
    if len(step2) == 0 and len(step1) > 0:
        print("gender values in race slice:\n", step1["gender_str"].value_counts().to_string())

    step3 = step2[step2["year_int"] == year_target]
    print("year match:", int(len(step3)))
    if len(step3) == 0 and len(step2) > 0:
        print("years in slice:\n", step2["year_int"].value_counts().head(20).to_string())

    step4 = step3[step3["status_str"] == "OK"]
    print("status==OK:", int(len(step4)))
    if len(step4) == 0 and len(step3) > 0:
        print("status values in slice:\n", step3["status_str"].value_counts().head(20).to_string())

    step5 = step4[(step4["age_float"] >= age_min) & (step4["age_float"] <= age_max)]
    print("age band match:", int(len(step5)))
    if len(step4) > 0:
        print("age range in slice:", float(step4["age_float"].min()), float(step4["age_float"].max()))

    # -----------------------------
    # Рисуем совместно
    # -----------------------------
    plt.figure(figsize=(10, 6))

    # Общие bins, чтобы сравнение было честным
    combined = np.concatenate([time_samples, real_times]) if real_times.size > 0 else time_samples
    left = float(np.quantile(combined, 0.001))
    right = float(np.quantile(combined, 0.999))
    bins = np.linspace(left, right, 80)

    plt.hist(time_samples, bins=bins, density=True, alpha=0.6, edgecolor="black", label="Model (MC samples)")

    if real_times.size > 0:
        plt.hist(real_times, bins=bins, density=True, alpha=0.4, edgecolor="black", label="Real data (year, age)")

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
       label=f"{tail * 100:.2f}%: {seconds_to_hhmmss(time_lower)}",
    )
    plt.axvline(
        time_upper,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"{(1.0-tail)*100:.2f}%: {seconds_to_hhmmss(time_upper)}"
    )

    axis = plt.gca()
    axis.xaxis.set_major_formatter(FuncFormatter(hhmmss_tick))

    plt.xlabel("Time")
    plt.ylabel("Density")
    plt.title(f"Predicted vs Real Distribution: {race_id}, {gender}, year={year_target}, age={age_target}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    race_id = "Казанский марафон"
    gender = "M"
    age_target = 40.0
    year_target = 2025
    main(race_id=race_id,
         gender=gender,
         age_target=age_target,
         year_target=year_target)