"""
ТЕСТ 1: Оценка дрейфа формы бегуна (σ_форма)

Цель: Доказать, что способность бегуна меняется даже на одной трассе.

Метод:
1. Находим бегунов, пробежавших одну и ту же трассу 2+ раза
2. Вычисляем Δ = ln(T_2) - ln(T_1) - [h(age_2) - h(age_1)]
3. Строим распределение Δ
4. Оцениваем σ_форма = std(Δ)

Интерпретация:
- Если σ_форма ≈ 0 → способность стабильна, h(age) достаточно
- Если σ_форма > 0 → форма меняется, калибровка будет biased
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from MarathonAgeModel import MarathonModel, easy_logging

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)


def find_repeat_runners(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит бегунов, пробежавших одну и ту же трассу 2+ раза.
    
    Returns:
        DataFrame с колонками: runner_id, race_id, gender, runs (список забегов)
    """
    # Группируем по (runner_id, race_id, gender)
    repeat_groups = (
        df.groupby(["runner_id", "race_id", "gender"])
        .filter(lambda x: len(x) >= 2)  # Только 2+ забега
    )
    
    if len(repeat_groups) == 0:
        return pd.DataFrame()
    
    # Создаем список забегов для каждого бегуна-трассы
    runs_list = []
    
    for (runner_id, race_id, gender), group in repeat_groups.groupby(["runner_id", "race_id", "gender"]):
        # Сортируем по году
        group_sorted = group.sort_values("year")
        
        runs = []
        for _, row in group_sorted.iterrows():
            runs.append({
                "year": int(row["year"]),
                "age": float(row["age"]),
                "time_seconds": float(row["time_seconds"]),
                "ln_time": float(np.log(row["time_seconds"])),
            })
        
        runs_list.append({
            "runner_id": runner_id,
            "race_id": race_id,
            "gender": gender,
            "n_runs": len(runs),
            "runs": runs,
        })
    
    return pd.DataFrame(runs_list)


def compute_form_drift(
    repeat_df: pd.DataFrame,
    age_models: Dict,
) -> pd.DataFrame:
    """
    Вычисляет дрейф формы Δ для каждой пары забегов.
    
    Δ = ln(T_2) - ln(T_1) - [h(age_2) - h(age_1)]
    
    Returns:
        DataFrame с колонками: runner_id, race_id, year_1, year_2, delta, delta_years
    """
    results = []
    
    for _, row in repeat_df.iterrows():
        runner_id = row["runner_id"]
        race_id = row["race_id"]
        gender = row["gender"]
        runs = row["runs"]
        
        if gender not in age_models:
            continue
        
        age_model = age_models[gender]
        
        # Берем все пары забегов
        for i in range(len(runs) - 1):
            for j in range(i + 1, len(runs)):
                run_1 = runs[i]
                run_2 = runs[j]
                
                # Вычисляем h(age) для обоих забегов
                age_1 = run_1["age"]
                age_2 = run_2["age"]
                
                x_1 = age_1 - 18.0  # Преобразование в x
                x_2 = age_2 - 18.0
                
                h_1 = float(age_model.predict_h(np.array([x_1]))[0])
                h_2 = float(age_model.predict_h(np.array([x_2]))[0])
                
                # Вычисляем Δ
                delta = run_2["ln_time"] - run_1["ln_time"] - (h_2 - h_1)
                
                results.append({
                    "runner_id": runner_id,
                    "race_id": race_id,
                    "gender": gender,
                    "year_1": run_1["year"],
                    "year_2": run_2["year"],
                    "delta_years": run_2["year"] - run_1["year"],
                    "age_1": age_1,
                    "age_2": age_2,
                    "delta": delta,
                    "delta_pct": 100.0 * (np.exp(delta) - 1.0),  # В процентах
                })
    
    return pd.DataFrame(results)


def analyze_form_drift(drift_df: pd.DataFrame) -> Dict:
    """
    Анализирует распределение дрейфа формы.
    """
    if len(drift_df) == 0:
        return {
            "n_pairs": 0,
            "sigma_forma": np.nan,
            "mean_delta": np.nan,
            "median_delta": np.nan,
            "ci_95": (np.nan, np.nan),
        }
    
    deltas = drift_df["delta"].to_numpy()
    
    # Основная статистика
    sigma_forma = float(np.std(deltas, ddof=1))
    mean_delta = float(np.mean(deltas))
    median_delta = float(np.median(deltas))
    
    # 95% CI
    ci_95 = (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )
    
    # В процентах для интерпретации
    sigma_forma_pct = 100.0 * (np.exp(sigma_forma) - 1.0)
    mean_delta_pct = 100.0 * (np.exp(mean_delta) - 1.0)
    
    return {
        "n_pairs": len(drift_df),
        "n_runners": drift_df["runner_id"].nunique(),
        "n_races": drift_df["race_id"].nunique(),
        "sigma_forma": sigma_forma,
        "sigma_forma_pct": sigma_forma_pct,
        "mean_delta": mean_delta,
        "mean_delta_pct": mean_delta_pct,
        "median_delta": median_delta,
        "ci_95": ci_95,
        "ci_95_pct": (100.0 * (np.exp(ci_95[0]) - 1.0), 100.0 * (np.exp(ci_95[1]) - 1.0)),
    }


def plot_form_drift(drift_df: pd.DataFrame, stats: Dict, output_dir: Path) -> None:
    """
    Визуализация дрейфа формы.
    """
    if len(drift_df) == 0:
        logger.warning("Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Гистограмма Δ (log-scale)
    ax = axes[0, 0]
    ax.hist(drift_df["delta"], bins=50, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Нулевой дрейф")
    ax.axvline(stats["mean_delta"], color="blue", linestyle="-", linewidth=2, label=f"Mean = {stats['mean_delta']:.3f}")
    ax.set_xlabel("Δ = ln(T₂) - ln(T₁) - [h(age₂) - h(age₁)]")
    ax.set_ylabel("Частота")
    ax.set_title(f"Распределение дрейфа формы\nσ_форма = {stats['sigma_forma']:.3f} ({stats['sigma_forma_pct']:.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Гистограмма Δ в процентах
    ax = axes[0, 1]
    ax.hist(drift_df["delta_pct"], bins=50, alpha=0.7, edgecolor="black", color="orange")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Δ (%)")
    ax.set_ylabel("Частота")
    ax.set_title("Дрейф формы в процентах")
    ax.grid(alpha=0.3)
    
    # 3. Δ vs временной разрыв
    ax = axes[1, 0]
    ax.scatter(drift_df["delta_years"], drift_df["delta"], alpha=0.5, s=20)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Разрыв между забегами (лет)")
    ax.set_ylabel("Δ")
    ax.set_title("Дрейф формы vs временной разрыв")
    ax.grid(alpha=0.3)
    
    # 4. Boxplot по временным интервалам
    ax = axes[1, 1]
    
    # Группируем по временному разрыву
    drift_df["delta_years_bin"] = pd.cut(
        drift_df["delta_years"],
        bins=[0, 1, 2, 3, 100],
        labels=["1 год", "2 года", "3 года", "4+ года"],
    )
    
    sns.boxplot(data=drift_df, x="delta_years_bin", y="delta", ax=ax)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Временной разрыв")
    ax.set_ylabel("Δ")
    ax.set_title("Дрейф по временным интервалам")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "test_1_form_drift.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"График сохранен: {output_path}")


def print_summary(stats: Dict, drift_df: pd.DataFrame) -> None:
    """
    Печатает итоговую сводку.
    """
    print("\n" + "=" * 80)
    print("ТЕСТ 1: ДРЕЙФ ФОРМЫ БЕГУНА (σ_форма)")
    print("=" * 80)
    
    if stats["n_pairs"] == 0:
        print("\n❌ НЕТ ДАННЫХ")
        print("   Не найдено бегунов с повторными забегами на одной трассе")
        return
    
    print(f"\nДанные:")
    print(f"  Пар забегов: {stats['n_pairs']}")
    print(f"  Уникальных бегунов: {stats['n_runners']}")
    print(f"  Уникальных трасс: {stats['n_races']}")
    
    print(f"\nОсновная статистика:")
    print(f"  σ_форма = {stats['sigma_forma']:.3f} (log-scale)")
    print(f"  σ_форма = {stats['sigma_forma_pct']:.1f}% (натуральная шкала)")
    print(f"  Mean(Δ) = {stats['mean_delta']:+.3f} ({stats['mean_delta_pct']:+.1f}%)")
    print(f"  Median(Δ) = {stats['median_delta']:+.3f}")
    print(f"  95% CI = [{stats['ci_95'][0]:.3f}, {stats['ci_95'][1]:.3f}]")
    print(f"  95% CI = [{stats['ci_95_pct'][0]:.1f}%, {stats['ci_95_pct'][1]:.1f}%]")
    
    # Интерпретация
    print("\n" + "=" * 80)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 80)
    
    sigma = stats["sigma_forma_pct"]
    
    if sigma < 5:
        print("\n✓ НИЗКИЙ ДРЕЙФ (<5%)")
        print("  Способность бегуна стабильна после учета h(age)")
        print("  Калибровка через связующих будет надежной")
    elif sigma < 10:
        print("\n⚠️  УМЕРЕННЫЙ ДРЕЙФ (5-10%)")
        print("  Способность меняется, но умеренно")
        print("  Калибровка возможна при N_linking > 30")
        print(f"  Неопределенность калибровки: σ_calib ≈ {sigma:.1f}% / √N")
    elif sigma < 15:
        print("\n⚠️⚠️  ВЫСОКИЙ ДРЕЙФ (10-15%)")
        print("  Способность сильно меняется")
        print("  Калибровка требует N_linking > 50")
        print(f"  При N=50: σ_calib ≈ {sigma / np.sqrt(50):.1f}%")
    else:
        print("\n❌ КРИТИЧЕСКИЙ ДРЕЙФ (>15%)")
        print("  Способность крайне нестабильна")
        print("  Калибровка через связующих ненадежна")
        print("  Требуется явное моделирование u_i (индивидуальная форма)")
    
    # Топ трасс по количеству повторов
    print("\n" + "=" * 80)
    print("ТОП ТРАСС С ПОВТОРНЫМИ ЗАБЕГАМИ")
    print("=" * 80)
    
    top_races = (
        drift_df.groupby("race_id")
        .agg(
            n_pairs=("runner_id", "count"),
            n_runners=("runner_id", "nunique"),
            mean_delta=("delta", "mean"),
            std_delta=("delta", "std"),
        )
        .sort_values("n_pairs", ascending=False)
        .head(10)
    )
    
    for race_id, row in top_races.iterrows():
        print(f"\n  {race_id}:")
        print(f"    Пар: {int(row['n_pairs'])}, Бегунов: {int(row['n_runners'])}")
        print(f"    Mean(Δ): {row['mean_delta']:+.3f}, Std(Δ): {row['std_delta']:.3f}")
    
    print("=" * 80)


def main():
    """
    Главная функция.
    """
    easy_logging(True)
    
    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТА 1: Дрейф формы бегуна")
    print("=" * 80)
    
    # Загружаем модель
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()
    
    # Находим повторные забеги
    df = model.df_clean.copy()
    df = df[df["status"] == "OK"].copy()
    
    logger.info("Поиск бегунов с повторными забегами...")
    repeat_df = find_repeat_runners(df)
    
    if len(repeat_df) == 0:
        print("\n❌ Не найдено бегунов с повторными забегами")
        return
    
    logger.info(f"Найдено {len(repeat_df)} бегунов с повторными забегами")
    
    # Вычисляем дрейф формы
    logger.info("Вычисление дрейфа формы...")
    drift_df = compute_form_drift(
        repeat_df=repeat_df,
        age_models=model.age_spline_models,
    )
    
    # Анализ
    stats = analyze_form_drift(drift_df)
    
    # Сохраняем результаты
    output_dir = Path("outputs/test_1_form_drift")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    drift_df.to_csv(output_dir / "drift_data.csv", index=False)
    repeat_df.to_csv(output_dir / "repeat_runners.csv", index=False)
    
    # Визуализация
    plot_form_drift(drift_df, stats, output_dir)
    
    # Итоговая сводка
    print_summary(stats, drift_df)
    
    # Сохраняем статистику
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "summary_stats.csv", index=False)
    
    logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
