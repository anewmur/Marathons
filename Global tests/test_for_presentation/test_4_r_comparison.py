"""
ТЕСТ 4: Сравнение R_top5 vs R_calib

Цель: Проверить, сходятся ли два метода оценки эталона или систематически расходятся.

Метод:
1. Для каждой трассы вычисляем:
   - R_top5: median(ln T) в топ-5%
   - R_calib: оценка через связующих бегунов с базовой трассой
2. Строим scatter plot: R_top5 vs R_calib
3. Вычисляем разности и их распределение

Интерпретация:
- Если точки на диагонали → методы согласны
- Если систематический сдвиг → один метод смещен относительно другого
- Если высокий разброс → методы измеряют разные вещи
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as scipy_stats

from MarathonAgeModel import MarathonModel, easy_logging

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

def compute_r_top5(df: pd.DataFrame, race_id: str, gender: str) -> float:
    """
    Вычисляет R_top5 для трассы: median(ln T) в топ-5%.
    """
    race_data = df[
        (df["race_id"] == race_id) & 
        (df["gender"] == gender) & 
        (df["status"] == "OK")
    ].copy()
    
    if len(race_data) == 0:
        return np.nan
    
    # Топ-5%
    threshold_95 = race_data["time_seconds"].quantile(0.05)
    top5_data = race_data[race_data["time_seconds"] <= threshold_95]
    
    if len(top5_data) == 0:
        return np.nan
    
    # Медиана на log-шкале
    r_top5 = float(np.median(np.log(top5_data["time_seconds"])))
    
    return r_top5


def compute_r_calib(
    df: pd.DataFrame,
    race_id: str,
    gender: str,
    base_race: str,
    base_r: float,
    age_models: Dict,
) -> Tuple[float, int, float]:
    """
    Вычисляет R_calib для трассы через связующих бегунов с base_race.
    
    Returns:
        (r_calib, n_linking, sigma_calib)
    """
    # Находим связующих бегунов
    base_data = df[
        (df["race_id"] == base_race) & 
        (df["gender"] == gender) & 
        (df["status"] == "OK")
    ].copy()
    
    target_data = df[
        (df["race_id"] == race_id) & 
        (df["gender"] == gender) & 
        (df["status"] == "OK")
    ].copy()
    
    if len(base_data) == 0 or len(target_data) == 0:
        return np.nan, 0, np.nan
    
    # Находим пересечение бегунов
    base_runners = set(base_data["runner_id"].unique())
    target_runners = set(target_data["runner_id"].unique())
    
    linking_runners = base_runners & target_runners
    
    if len(linking_runners) == 0:
        return np.nan, 0, np.nan
    
    # Для каждого связующего бегуна вычисляем разность
    deltas = []
    
    age_model = age_models.get(gender)
    if age_model is None:
        return np.nan, 0, np.nan
    
    for runner_id in linking_runners:
        # Забеги в базовой трассе
        base_runs = base_data[base_data["runner_id"] == runner_id]
        # Забеги в целевой трассе
        target_runs = target_data[target_data["runner_id"] == runner_id]
        
        # Берем средние (если несколько забегов)
        base_time = float(base_runs["time_seconds"].mean())
        base_age = float(base_runs["age"].mean())
        
        target_time = float(target_runs["time_seconds"].mean())
        target_age = float(target_runs["age"].mean())
        
        # Вычисляем h(age)
        x_base = base_age - 18.0
        x_target = target_age - 18.0
        
        h_base = float(age_model.predict_h(np.array([x_base]))[0])
        h_target = float(age_model.predict_h(np.array([x_target]))[0])
        
        # Разность после учета возраста
        delta = np.log(target_time) - np.log(base_time) - (h_target - h_base)
        deltas.append(delta)
    
    if len(deltas) == 0:
        return np.nan, 0, np.nan
    
    # Оцениваем разницу трасс
    mean_delta = float(np.mean(deltas))
    sigma_delta = float(np.std(deltas, ddof=1))
    n_linking = len(deltas)
    
    # R_calib = R_base + mean_delta
    r_calib = base_r + mean_delta
    
    # Неопределенность
    sigma_calib = sigma_delta / np.sqrt(n_linking) if n_linking > 0 else np.nan
    
    return r_calib, n_linking, sigma_calib


def compare_methods(
    df: pd.DataFrame,
    age_models: Dict,
    base_race: str = "Чемпионат России",
) -> pd.DataFrame:
    """
    Сравнивает R_top5 и R_calib для всех трасс.
    """
    results = []
    
    races = df["race_id"].unique()
    
    # Вычисляем R_top5 для базовой трассы
    base_r_m = compute_r_top5(df, base_race, "M")
    base_r_f = compute_r_top5(df, base_race, "F")
    
    logger.info(f"Базовая трасса: {base_race}")
    logger.info(f"  R_top5(M) = {np.exp(base_r_m) / 60:.1f} мин" if not np.isnan(base_r_m) else "  R_top5(M) = N/A")
    logger.info(f"  R_top5(F) = {np.exp(base_r_f) / 60:.1f} мин" if not np.isnan(base_r_f) else "  R_top5(F) = N/A")
    
    for race_id in races:
        if race_id == base_race:
            continue
        
        for gender in ["M", "F"]:
            base_r = base_r_m if gender == "M" else base_r_f
            
            if np.isnan(base_r):
                continue
            
            # Вычисляем R_top5
            r_top5 = compute_r_top5(df, race_id, gender)
            
            # Вычисляем R_calib
            r_calib, n_linking, sigma_calib = compute_r_calib(
                df, race_id, gender, base_race, base_r, age_models
            )
            
            if np.isnan(r_top5) or np.isnan(r_calib):
                continue
            
            # Разность
            delta = r_calib - r_top5
            delta_pct = 100.0 * (np.exp(delta) - 1.0)
            
            results.append({
                "race_id": race_id,
                "gender": gender,
                "r_top5": r_top5,
                "r_calib": r_calib,
                "delta": delta,
                "delta_pct": delta_pct,
                "n_linking": n_linking,
                "sigma_calib": sigma_calib,
                "r_top5_minutes": np.exp(r_top5) / 60,
                "r_calib_minutes": np.exp(r_calib) / 60,
            })
    
    return pd.DataFrame(results)


def analyze_comparison(comparison_df: pd.DataFrame) -> Dict:
    """
    Анализирует сравнение методов.
    """
    if len(comparison_df) == 0:
        return {}
    
    deltas = comparison_df["delta"].to_numpy()
    deltas_pct = comparison_df["delta_pct"].to_numpy()
    
    # Основная статистика
    stats = {
        "n_races": len(comparison_df),
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "std_delta": float(np.std(deltas, ddof=1)),
        "mean_delta_pct": float(np.mean(deltas_pct)),
        "median_delta_pct": float(np.median(deltas_pct)),
        "std_delta_pct": float(np.std(deltas_pct, ddof=1)),
        "ci_95": (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))),
        "ci_95_pct": (float(np.percentile(deltas_pct, 2.5)), float(np.percentile(deltas_pct, 97.5))),
    }
    
    # Корреляция
    r_top5 = comparison_df["r_top5"].to_numpy()
    r_calib = comparison_df["r_calib"].to_numpy()
    
    corr, p_value = scipy_stats.pearsonr(r_top5, r_calib)
    stats["correlation"] = float(corr)
    stats["correlation_p"] = float(p_value)
    
    # t-test: отличается ли среднее от нуля?
    t_stat, t_p = scipy_stats.ttest_1samp(deltas, 0)
    stats["ttest_vs_zero"] = {
        "t_stat": float(t_stat),
        "p_value": float(t_p),
        "significant": t_p < 0.05,
    }
    
    return stats


def plot_comparison(comparison_df: pd.DataFrame, stats: Dict, output_dir: Path) -> None:
    """
    Визуализация сравнения методов.
    """
    if len(comparison_df) == 0:
        logger.warning("Нет данных для визуализации")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter plot: R_top5 vs R_calib (в минутах)
    ax = axes[0, 0]
    
    for gender, color, marker in [("M", "blue", "o"), ("F", "red", "s")]:
        gender_data = comparison_df[comparison_df["gender"] == gender]
        if len(gender_data) > 0:
            ax.scatter(
                gender_data["r_top5_minutes"],
                gender_data["r_calib_minutes"],
                alpha=0.6,
                s=80,
                color=color,
                marker=marker,
                label=gender,
            )
    
    # Диагональ
    lim_min = min(comparison_df["r_top5_minutes"].min(), comparison_df["r_calib_minutes"].min())
    lim_max = max(comparison_df["r_top5_minutes"].max(), comparison_df["r_calib_minutes"].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=2, label="x=y")
    
    ax.set_xlabel("R_top5 (минуты)")
    ax.set_ylabel("R_calib (минуты)")
    ax.set_title(f"Сравнение методов\n(correlation = {stats['correlation']:.3f})")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Гистограмма разностей (log-scale)
    ax = axes[0, 1]
    ax.hist(comparison_df["delta"], bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Нулевая разность")
    ax.axvline(stats["mean_delta"], color="blue", linestyle="-", linewidth=2, 
               label=f"Mean = {stats['mean_delta']:.3f}")
    ax.set_xlabel("Δ = R_calib - R_top5 (log-scale)")
    ax.set_ylabel("Частота")
    ax.set_title(f"Распределение разностей\nσ = {stats['std_delta']:.3f}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Гистограмма разностей (в процентах)
    ax = axes[1, 0]
    ax.hist(comparison_df["delta_pct"], bins=20, alpha=0.7, edgecolor="black", color="orange")
    ax.axvline(0, color="red", linestyle="--", linewidth=2)
    ax.axvline(stats["mean_delta_pct"], color="blue", linestyle="-", linewidth=2, 
               label=f"Mean = {stats['mean_delta_pct']:+.1f}%")
    ax.set_xlabel("Δ (%)")
    ax.set_ylabel("Частота")
    ax.set_title("Разности в процентах")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Разности vs N_linking
    ax = axes[1, 1]
    ax.scatter(comparison_df["n_linking"], comparison_df["delta_pct"], alpha=0.6, s=60)
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Количество связующих бегунов")
    ax.set_ylabel("Δ (%)")
    ax.set_title("Разность vs Количество связующих")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "test_4_r_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"График сохранен: {output_path}")


def print_summary(comparison_df: pd.DataFrame, stats: Dict) -> None:
    """
    Печатает итоговую сводку.
    """
    print("\n" + "=" * 80)
    print("ТЕСТ 4: СРАВНЕНИЕ R_TOP5 VS R_CALIB")
    print("=" * 80)
    
    if len(comparison_df) == 0:
        print("\n❌ НЕТ ДАННЫХ")
        return
    
    print(f"\nДанные:")
    print(f"  Трасс проанализировано: {stats['n_races']}")
    
    print(f"\nРазность (R_calib - R_top5):")
    print(f"  Mean:   {stats['mean_delta']:+.4f} ({stats['mean_delta_pct']:+.1f}%)")
    print(f"  Median: {stats['median_delta']:+.4f} ({stats['median_delta_pct']:+.1f}%)")
    print(f"  Std:    {stats['std_delta']:.4f} ({stats['std_delta_pct']:.1f}%)")
    print(f"  95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
    print(f"  95% CI: [{stats['ci_95_pct'][0]:.1f}%, {stats['ci_95_pct'][1]:.1f}%]")
    
    print(f"\nКорреляция:")
    print(f"  r = {stats['correlation']:.3f}")
    print(f"  p-value = {stats['correlation_p']:.4f}")
    
    ttest = stats["ttest_vs_zero"]
    print(f"\nt-test (H0: mean_delta = 0):")
    print(f"  t = {ttest['t_stat']:+.3f}")
    print(f"  p-value = {ttest['p_value']:.4f}")
    
    if ttest["significant"]:
        print(f"  ✓ СИСТЕМАТИЧЕСКИЙ СДВИГ ОБНАРУЖЕН (p < 0.05)")
    else:
        print(f"  ✗ Систематический сдвиг не обнаружен")
    
    # Интерпретация
    print("\n" + "=" * 80)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 80)
    
    mean_pct = abs(stats["mean_delta_pct"])
    std_pct = stats["std_delta_pct"]
    
    if mean_pct < 2 and std_pct < 5:
        print("\n✓ МЕТОДЫ СОГЛАСУЮТСЯ")
        print("  Систематический сдвиг <2%, разброс <5%")
        print("  R_top5 и R_calib измеряют одно и то же")
    elif mean_pct < 5 and std_pct < 10:
        print("\n⚠️  УМЕРЕННОЕ РАСХОЖДЕНИЕ")
        print(f"  Систематический сдвиг ~{mean_pct:.1f}%, разброс ~{std_pct:.1f}%")
        print("  Методы близки, но не идентичны")
    else:
        print("\n❌ МЕТОДЫ РАСХОДЯТСЯ")
        print(f"  Систематический сдвиг {mean_pct:.1f}%, разброс {std_pct:.1f}%")
        print("  R_top5 и R_calib измеряют РАЗНЫЕ вещи")
        print("  Рекомендация: выбрать один метод как базовый")
    
    # Топ расхождений
    print("\n" + "=" * 80)
    print("ТОП-10 ТРАСС ПО ВЕЛИЧИНЕ РАСХОЖДЕНИЯ")
    print("=" * 80)
    
    top_discrepancies = comparison_df.nlargest(10, "delta_pct")
    
    for _, row in top_discrepancies.iterrows():
        print(f"\n  {row['race_id']} ({row['gender']}):")
        print(f"    R_top5:  {row['r_top5_minutes']:.1f} мин")
        print(f"    R_calib: {row['r_calib_minutes']:.1f} мин")
        print(f"    Δ: {row['delta_pct']:+.1f}% (N_linking={int(row['n_linking'])})")
    
    print("=" * 80)


def main():
    """
    Главная функция.
    """
    easy_logging(True)
    
    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТА 4: Сравнение R_top5 vs R_calib")
    print("=" * 80)
    
    # Загружаем модель
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()
    
    df = model.df_clean.copy()
    
    # Выбираем базовую трассу (обычно самая элитная)
    base_race = "Чемпионат России"
    
    logger.info(f"Базовая трасса: {base_race}")
    
    # Сравниваем методы
    comparison_df = compare_methods(
        df=df,
        age_models=model.age_spline_models,
        base_race=base_race,
    )
    
    if len(comparison_df) == 0:
        print("\n❌ Не удалось вычислить эталоны")
        return
    
    # Анализ
    stats = analyze_comparison(comparison_df)
    
    # Сохраняем результаты
    output_dir = Path("outputs/test_4_r_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(output_dir / "comparison_data.csv", index=False)
    
    # Визуализация
    plot_comparison(comparison_df, stats, output_dir)
    
    # Итоговая сводка
    print_summary(comparison_df, stats)
    
    # Сохраняем статистику
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "summary_stats.csv", index=False)
    
    logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
