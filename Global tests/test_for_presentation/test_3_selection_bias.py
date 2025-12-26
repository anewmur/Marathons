"""
ТЕСТ 3: Selection bias — связующие vs Top-5%

Цель: Доказать, что связующие бегуны НЕ репрезентативны для Top-5%.

Метод:
1. Для выбранной трассы (например, Казань) находим:
   - Top-5% всех участников
   - Связующих бегунов (участвовали также в других трассах)
2. Сравниваем распределения:
   - Возраст
   - Время финиша
   - Доля в Top-5%
3. Строим boxplot и гистограммы

Интерпретация:
- Если распределения совпадают → selection bias нет
- Если различаются → связующие смещены относительно Top-5%
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Set

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
def classify_runners_on_race(
    df: pd.DataFrame,
    race_id: str,
    gender: str,
) -> pd.DataFrame:
    """
    Классифицирует бегунов на трассе:
    - top5: входят в топ-5% на этой трассе
    - linking: участвовали в других трассах
    - linking_top5: связующие И в топ-5%
    """
    # Выбираем данные для конкретной трассы
    race_data = df[
        (df["race_id"] == race_id) & 
        (df["gender"] == gender) & 
        (df["status"] == "OK")
    ].copy()
    
    if len(race_data) == 0:
        return pd.DataFrame()
    
    # Находим Top-5%
    threshold_95 = race_data["time_seconds"].quantile(0.05)
    race_data["is_top5"] = race_data["time_seconds"] <= threshold_95
    
    # Находим связующих (участвовали в других трассах)
    all_runner_races = df.groupby("runner_id")["race_id"].apply(set)
    
    race_data["is_linking"] = race_data["runner_id"].apply(
        lambda runner_id: len(all_runner_races.get(runner_id, set())) > 1
    )
    
    # Комбинированные категории
    race_data["category"] = "regular"
    race_data.loc[race_data["is_top5"], "category"] = "top5"
    race_data.loc[race_data["is_linking"], "category"] = "linking"
    race_data.loc[race_data["is_top5"] & race_data["is_linking"], "category"] = "linking_top5"
    
    return race_data


def analyze_selection_bias(race_data: pd.DataFrame) -> Dict:
    """
    Анализирует selection bias между группами.
    """
    if len(race_data) == 0:
        return {}
    
    stats_dict = {}
    
    # Группируем по категориям
    for category in ["top5", "linking", "linking_top5"]:
        cat_data = race_data[race_data["category"] == category]
        
        if len(cat_data) == 0:
            continue
        
        stats_dict[category] = {
            "n": len(cat_data),
            "mean_age": float(cat_data["age"].mean()),
            "std_age": float(cat_data["age"].std()),
            "mean_time": float(cat_data["time_seconds"].mean()),
            "median_time": float(cat_data["time_seconds"].median()),
            "pct_of_total": 100.0 * len(cat_data) / len(race_data),
        }
    
    # Статистические тесты
    if "top5" in stats_dict and "linking" in stats_dict:
        top5_ages = race_data[race_data["category"] == "top5"]["age"].values
        linking_ages = race_data[race_data["category"] == "linking"]["age"].values
        
        if len(top5_ages) > 5 and len(linking_ages) > 5:
            # t-test для возраста
            t_stat, p_value = scipy_stats.ttest_ind(top5_ages, linking_ages)
            stats_dict["age_ttest"] = {
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
            }
            
            # KS test для времени
            top5_times = race_data[race_data["category"] == "top5"]["time_seconds"].values
            linking_times = race_data[race_data["category"] == "linking"]["time_seconds"].values
            
            ks_stat, ks_p = scipy_stats.ks_2samp(top5_times, linking_times)
            stats_dict["time_kstest"] = {
                "ks_stat": float(ks_stat),
                "p_value": float(ks_p),
                "significant": ks_p < 0.05,
            }
    
    return stats_dict


def plot_selection_bias(
    race_data: pd.DataFrame,
    stats: Dict,
    race_id: str,
    gender: str,
    output_dir: Path,
) -> None:
    """
    Визуализация selection bias.
    """
    if len(race_data) == 0:
        logger.warning("Нет данных для визуализации")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Гистограммы возраста (3 subplot)
    categories = ["top5", "linking", "linking_top5"]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    titles = ["Top-5% (все)", "Связующие (все)", "Связующие ∩ Top-5%"]
    
    for idx, (cat, color, title) in enumerate(zip(categories, colors, titles)):
        ax = fig.add_subplot(gs[0, idx])
        cat_data = race_data[race_data["category"] == cat]
        
        if len(cat_data) > 0:
            ax.hist(cat_data["age"], bins=20, alpha=0.7, color=color, edgecolor="black")
            ax.set_xlabel("Возраст")
            ax.set_ylabel("Частота")
            ax.set_title(f"{title}\n(n={len(cat_data)}, mean={cat_data['age'].mean():.1f})")
            ax.grid(alpha=0.3)
    
    # 2. Сравнение распределений возраста (overlaid)
    ax = fig.add_subplot(gs[1, :2])
    
    for cat, color, title in zip(categories, colors, titles):
        cat_data = race_data[race_data["category"] == cat]
        if len(cat_data) > 0:
            ax.hist(cat_data["age"], bins=20, alpha=0.4, color=color, label=title, density=True)
    
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Плотность")
    ax.set_title("Наложенные распределения возраста")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Boxplot времени финиша
    ax = fig.add_subplot(gs[1, 2])
    
    plot_data = []
    plot_labels = []
    
    for cat, title in zip(categories, titles):
        cat_data = race_data[race_data["category"] == cat]
        if len(cat_data) > 0:
            plot_data.append(cat_data["time_seconds"] / 60)  # В минутах
            plot_labels.append(title)
    
    if plot_data:
        ax.boxplot(plot_data, tick_labels=plot_labels)
        ax.set_ylabel("Время финиша (минуты)")
        ax.set_title("Сравнение времени финиша")
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    
    # 4. Scatter: возраст vs время
    ax = fig.add_subplot(gs[2, :2])
    
    for cat, color, title in zip(categories, colors, titles):
        cat_data = race_data[race_data["category"] == cat]
        if len(cat_data) > 0:
            ax.scatter(
                cat_data["age"],
                cat_data["time_seconds"] / 60,
                alpha=0.5,
                s=20,
                color=color,
                label=title,
            )
    
    ax.set_xlabel("Возраст")
    ax.set_ylabel("Время финиша (минуты)")
    ax.set_title("Возраст vs Время финиша")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 5. Статистика
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    
    text_lines = [
        f"Трасса: {race_id} ({gender})",
        "",
        "Статистические тесты:",
    ]
    
    if "age_ttest" in stats:
        ttest = stats["age_ttest"]
        sig = "✓ Да" if ttest["significant"] else "✗ Нет"
        text_lines.append(f"Возраст (t-test):")
        text_lines.append(f"  p = {ttest['p_value']:.4f}")
        text_lines.append(f"  Различие: {sig}")
    
    if "time_kstest" in stats:
        kstest = stats["time_kstest"]
        sig = "✓ Да" if kstest["significant"] else "✗ Нет"
        text_lines.append(f"\nВремя (KS test):")
        text_lines.append(f"  p = {kstest['p_value']:.4f}")
        text_lines.append(f"  Различие: {sig}")
    
    ax.text(0.1, 0.9, "\n".join(text_lines), fontsize=10, verticalalignment="top", family="monospace")
    
    plt.suptitle(f"Selection Bias: {race_id} ({gender})", fontsize=14, fontweight="bold")
    
    output_path = output_dir / f"test_3_selection_bias_{race_id}_{gender}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"График сохранен: {output_path}")


def print_summary(stats: Dict, race_id: str, gender: str) -> None:
    """
    Печатает итоговую сводку.
    """
    print("\n" + "=" * 80)
    print(f"ТЕСТ 3: SELECTION BIAS — {race_id} ({gender})")
    print("=" * 80)
    
    if not stats:
        print("\n❌ НЕТ ДАННЫХ")
        return
    
    print(f"\nРазмеры групп:")
    for cat in ["top5", "linking", "linking_top5"]:
        if cat in stats:
            s = stats[cat]
            print(f"  {cat:15s}: n={s['n']:4d} ({s['pct_of_total']:5.1f}%)")
    
    print(f"\nСредний возраст:")
    for cat in ["top5", "linking", "linking_top5"]:
        if cat in stats:
            s = stats[cat]
            print(f"  {cat:15s}: {s['mean_age']:5.1f} ± {s['std_age']:.1f} лет")
    
    print(f"\nСреднее время:")
    for cat in ["top5", "linking", "linking_top5"]:
        if cat in stats:
            s = stats[cat]
            hours = s['mean_time'] / 3600
            print(f"  {cat:15s}: {hours:.2f} часов ({s['mean_time']:.0f} сек)")
    
    # Статистические тесты
    print("\n" + "=" * 80)
    print("СТАТИСТИЧЕСКИЕ ТЕСТЫ")
    print("=" * 80)
    
    if "age_ttest" in stats:
        ttest = stats["age_ttest"]
        print(f"\nСравнение возраста (Top-5% vs Связующие):")
        print(f"  t-статистика: {ttest['t_stat']:+.3f}")
        print(f"  p-value: {ttest['p_value']:.4f}")
        
        if ttest["significant"]:
            print(f"  ✓ РАЗЛИЧИЕ ЗНАЧИМО (p < 0.05)")
        else:
            print(f"  ✗ Различие незначимо")
    
    if "time_kstest" in stats:
        kstest = stats["time_kstest"]
        print(f"\nСравнение времени (Top-5% vs Связующие):")
        print(f"  KS-статистика: {kstest['ks_stat']:.3f}")
        print(f"  p-value: {kstest['p_value']:.4f}")
        
        if kstest["significant"]:
            print(f"  ✓ РАСПРЕДЕЛЕНИЯ РАЗЛИЧАЮТСЯ (p < 0.05)")
        else:
            print(f"  ✗ Распределения не различаются")
    
    # Интерпретация
    print("\n" + "=" * 80)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 80)
    
    has_age_bias = stats.get("age_ttest", {}).get("significant", False)
    has_time_bias = stats.get("time_kstest", {}).get("significant", False)
    
    if not has_age_bias and not has_time_bias:
        print("\n✓ SELECTION BIAS НЕ ОБНАРУЖЕН")
        print("  Связующие бегуны репрезентативны для Top-5%")
        print("  Калибровка через связующих корректна")
    elif has_age_bias and not has_time_bias:
        print("\n⚠️  УМЕРЕННЫЙ BIAS (только возраст)")
        print("  Связующие имеют другой возрастной состав")
        print("  Но время финиша схожее → после учета h(age) bias уменьшится")
    elif has_time_bias:
        print("\n❌ SELECTION BIAS ОБНАРУЖЕН")
        print("  Связующие бегуны НЕ репрезентативны для Top-5%")
        print("  Калибровка через связующих даст смещенные эталоны")
        print("  Рекомендация: использовать Top-5% как базис")
    
    print("=" * 80)


def main():
    """
    Главная функция.
    """
    easy_logging(True)
    
    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТА 3: Selection bias")
    print("=" * 80)
    
    # Загружаем модель
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()
    
    df = model.df_clean.copy()
    
    # Выбираем трассы для анализа (самые крупные)
    race_counts = df[df["status"] == "OK"].groupby("race_id").size().sort_values(ascending=False)
    top_races = race_counts.head(5).index.tolist()
    
    logger.info(f"Анализируем топ-5 трасс: {top_races}")
    
    output_dir = Path("outputs/test_3_selection_bias")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_stats = []
    
    # Анализируем каждую трассу
    for race_id in top_races:
        for gender in ["M", "F"]:
            logger.info(f"Анализ: {race_id} ({gender})")
            
            race_data = classify_runners_on_race(df, race_id, gender)
            
            if len(race_data) == 0:
                logger.warning(f"Нет данных для {race_id} ({gender})")
                continue
            
            stats = analyze_selection_bias(race_data)
            
            if not stats:
                continue
            
            # Добавляем метаданные
            stats["race_id"] = race_id
            stats["gender"] = gender
            all_stats.append(stats)
            
            # Визуализация
            plot_selection_bias(race_data, stats, race_id, gender, output_dir)
            
            # Печатаем сводку
            print_summary(stats, race_id, gender)
            
            # Сохраняем данные
            race_data.to_csv(output_dir / f"data_{race_id}_{gender}.csv", index=False)
    
    # Сводная таблица
    if all_stats:
        summary_rows = []
        for s in all_stats:
            row = {
                "race_id": s["race_id"],
                "gender": s["gender"],
                "n_top5": s.get("top5", {}).get("n", 0),
                "n_linking": s.get("linking", {}).get("n", 0),
                "n_linking_top5": s.get("linking_top5", {}).get("n", 0),
                "age_bias_p": s.get("age_ttest", {}).get("p_value", np.nan),
                "time_bias_p": s.get("time_kstest", {}).get("p_value", np.nan),
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(output_dir / "summary_all_races.csv", index=False)
        
        logger.info(f"Сводная таблица: {output_dir / 'summary_all_races.csv'}")
    
    logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
