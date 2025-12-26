"""
ТЕСТ 2: Временной разрыв для пар связующих бегунов

Цель: Доказать, что большинство связующих бегают трассы в разные годы.

Метод:
1. Находим связующих бегунов (бегали 2+ трассы)
2. Для каждой пары трасс вычисляем Δt = |год_B - год_A|
3. Строим распределение Δt
4. Вычисляем долю same-year vs cross-year

Интерпретация:
- Если большинство same-year → калибровка чистая
- Если большинство cross-year → нужен явный учет δ_year
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


def find_linking_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Находит все пары забегов для связующих бегунов.
    
    Returns:
        DataFrame с колонками:
        runner_id, race_A, race_B, year_A, year_B, delta_years, age_A, age_B
    """
    # Группируем по бегунам
    runner_groups = df.groupby("runner_id")
    
    pairs = []
    
    for runner_id, group in runner_groups:
        # Берем только бегунов с 2+ забегами
        if len(group) < 2:
            continue
        
        # Сортируем по году
        group_sorted = group.sort_values("year")
        races = group_sorted.to_dict("records")
        
        # Генерируем все пары
        for i in range(len(races)):
            for j in range(i + 1, len(races)):
                race_A = races[i]
                race_B = races[j]
                
                pairs.append({
                    "runner_id": runner_id,
                    "gender": race_A["gender"],
                    "race_A": race_A["race_id"],
                    "race_B": race_B["race_id"],
                    "year_A": int(race_A["year"]),
                    "year_B": int(race_B["year"]),
                    "delta_years": int(race_B["year"] - race_A["year"]),
                    "age_A": float(race_A["age"]),
                    "age_B": float(race_B["age"]),
                    "time_A": float(race_A["time_seconds"]),
                    "time_B": float(race_B["time_seconds"]),
                    "same_year": (race_A["year"] == race_B["year"]),
                    "same_race": (race_A["race_id"] == race_B["race_id"]),
                })
    
    return pd.DataFrame(pairs)


def analyze_time_gaps(pairs_df: pd.DataFrame) -> Dict:
    """
    Анализирует распределение временных разрывов.
    """
    if len(pairs_df) == 0:
        return {
            "n_pairs_total": 0,
            "n_runners": 0,
            "pct_same_year": np.nan,
            "pct_1_year": np.nan,
            "pct_2_years": np.nan,
            "pct_3plus_years": np.nan,
        }
    
    # Фильтруем: только разные трассы
    pairs_diff_races = pairs_df[~pairs_df["same_race"]].copy()
    
    n_pairs_total = len(pairs_diff_races)
    n_runners = pairs_diff_races["runner_id"].nunique()
    
    # Распределение по интервалам
    n_same_year = (pairs_diff_races["delta_years"] == 0).sum()
    n_1_year = (pairs_diff_races["delta_years"] == 1).sum()
    n_2_years = (pairs_diff_races["delta_years"] == 2).sum()
    n_3plus = (pairs_diff_races["delta_years"] >= 3).sum()
    
    return {
        "n_pairs_total": n_pairs_total,
        "n_runners": n_runners,
        "n_same_year": n_same_year,
        "n_1_year": n_1_year,
        "n_2_years": n_2_years,
        "n_3plus_years": n_3plus,
        "pct_same_year": 100.0 * n_same_year / n_pairs_total if n_pairs_total > 0 else 0,
        "pct_1_year": 100.0 * n_1_year / n_pairs_total if n_pairs_total > 0 else 0,
        "pct_2_years": 100.0 * n_2_years / n_pairs_total if n_pairs_total > 0 else 0,
        "pct_3plus_years": 100.0 * n_3plus / n_pairs_total if n_pairs_total > 0 else 0,
        "mean_delta_years": float(pairs_diff_races["delta_years"].mean()) if n_pairs_total > 0 else np.nan,
        "median_delta_years": float(pairs_diff_races["delta_years"].median()) if n_pairs_total > 0 else np.nan,
    }


def plot_time_gaps(pairs_df: pd.DataFrame, stats: Dict, output_dir: Path) -> None:
    """
    Визуализация временных разрывов.
    """
    if len(pairs_df) == 0:
        logger.warning("Нет данных для визуализации")
        return
    
    # Фильтруем: только разные трассы
    pairs_diff = pairs_df[~pairs_df["same_race"]].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Гистограмма Δt
    ax = axes[0, 0]
    counts = pairs_diff["delta_years"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Временной разрыв (лет)")
    ax.set_ylabel("Количество пар")
    ax.set_title(f"Распределение временных разрывов\n(всего {len(pairs_diff)} пар)")
    ax.grid(axis="y", alpha=0.3)
    
    # 2. Pie chart: same-year vs cross-year
    ax = axes[0, 1]
    labels = [
        f"Один год\n({stats['pct_same_year']:.1f}%)",
        f"1 год разрыв\n({stats['pct_1_year']:.1f}%)",
        f"2 года разрыв\n({stats['pct_2_years']:.1f}%)",
        f"3+ года\n({stats['pct_3plus_years']:.1f}%)",
    ]
    sizes = [
        stats["n_same_year"],
        stats["n_1_year"],
        stats["n_2_years"],
        stats["n_3plus_years"],
    ]
    colors = ["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"]
    
    ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90)
    ax.set_title("Доля пар по временному разрыву")
    
    # 3. Cumulative distribution
    ax = axes[1, 0]
    sorted_deltas = np.sort(pairs_diff["delta_years"].values)
    cumulative = np.arange(1, len(sorted_deltas) + 1) / len(sorted_deltas) * 100
    ax.plot(sorted_deltas, cumulative, linewidth=2)
    ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50%")
    ax.set_xlabel("Временной разрыв (лет)")
    ax.set_ylabel("Кумулятивная доля (%)")
    ax.set_title("Кумулятивное распределение")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Heatmap: топовые пары трасс
    ax = axes[1, 1]
    
    # Считаем частые пары трасс
    race_pairs = (
        pairs_diff.groupby(["race_A", "race_B"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )
    
    # Для каждой пары считаем среднее Δt
    race_pairs["mean_delta"] = race_pairs.apply(
        lambda row: pairs_diff[
            (pairs_diff["race_A"] == row["race_A"]) & 
            (pairs_diff["race_B"] == row["race_B"])
        ]["delta_years"].mean(),
        axis=1,
    )
    
    # Строим bar chart
    labels = [f"{row['race_A'][:10]} → {row['race_B'][:10]}" for _, row in race_pairs.iterrows()]
    ax.barh(range(len(race_pairs)), race_pairs["mean_delta"], alpha=0.7)
    ax.set_yticks(range(len(race_pairs)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Средний временной разрыв (лет)")
    ax.set_title("Топ-15 пар трасс")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    output_path = output_dir / "test_2_time_gaps.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"График сохранен: {output_path}")


def print_summary(stats: Dict, pairs_df: pd.DataFrame) -> None:
    """
    Печатает итоговую сводку.
    """
    print("\n" + "=" * 80)
    print("ТЕСТ 2: ВРЕМЕННОЙ РАЗРЫВ ДЛЯ СВЯЗУЮЩИХ БЕГУНОВ")
    print("=" * 80)
    
    if stats["n_pairs_total"] == 0:
        print("\n❌ НЕТ ДАННЫХ")
        return
    
    print(f"\nДанные:")
    print(f"  Всего пар (разные трассы): {stats['n_pairs_total']}")
    print(f"  Уникальных бегунов: {stats['n_runners']}")
    
    print(f"\nРаспределение по временным разрывам:")
    print(f"  Один год (same-year):  {stats['n_same_year']:5d} ({stats['pct_same_year']:5.1f}%)")
    print(f"  1 год разрыв:          {stats['n_1_year']:5d} ({stats['pct_1_year']:5.1f}%)")
    print(f"  2 года разрыв:         {stats['n_2_years']:5d} ({stats['pct_2_years']:5.1f}%)")
    print(f"  3+ года разрыв:        {stats['n_3plus_years']:5d} ({stats['pct_3plus_years']:5.1f}%)")
    
    print(f"\nСредний разрыв: {stats['mean_delta_years']:.2f} года")
    print(f"Медианный разрыв: {stats['median_delta_years']:.0f} года")
    
    # Интерпретация
    print("\n" + "=" * 80)
    print("ИНТЕРПРЕТАЦИЯ")
    print("=" * 80)
    
    pct_same = stats["pct_same_year"]
    pct_cross = 100 - pct_same
    
    if pct_same > 70:
        print("\n✓ БОЛЬШИНСТВО SAME-YEAR")
        print("  Калибровка через связующих надежна")
        print("  Год не подмешивается в разницу трасс")
    elif pct_same > 40:
        print("\n⚠️  СМЕШАННАЯ КАРТИНА")
        print(f"  Same-year: {pct_same:.1f}%, Cross-year: {pct_cross:.1f}%")
        print("  Рекомендуется учитывать δ_year в калибровке")
    else:
        print("\n❌ БОЛЬШИНСТВО CROSS-YEAR")
        print(f"  Same-year: {pct_same:.1f}%, Cross-year: {pct_cross:.1f}%")
        print("  Калибровка БЕЗ учета δ_year будет смешивать трассу и год")
        print("  КРИТИЧНО: Требуется явное моделирование годовых эффектов")
    
    # Топ пар по количеству связующих
    print("\n" + "=" * 80)
    print("ТОП-10 ПАР ТРАСС ПО КОЛИЧЕСТВУ СВЯЗУЮЩИХ")
    print("=" * 80)
    
    pairs_diff = pairs_df[~pairs_df["same_race"]].copy()
    
    top_pairs = (
        pairs_diff.groupby(["race_A", "race_B"])
        .agg(
            n_linking=("runner_id", "nunique"),
            n_pairs=("runner_id", "count"),
            mean_delta_years=("delta_years", "mean"),
            pct_same_year=("same_year", lambda x: 100.0 * x.sum() / len(x)),
        )
        .sort_values("n_linking", ascending=False)
        .head(10)
    )
    
    for (race_A, race_B), row in top_pairs.iterrows():
        print(f"\n  {race_A} ↔ {race_B}:")
        print(f"    Связующих: {int(row['n_linking'])}, Пар: {int(row['n_pairs'])}")
        print(f"    Средний разрыв: {row['mean_delta_years']:.1f} года")
        print(f"    Same-year: {row['pct_same_year']:.1f}%")
    
    print("=" * 80)


def main():
    """
    Главная функция.
    """
    easy_logging(True)
    
    print("\n" + "=" * 80)
    print("ЗАПУСК ТЕСТА 2: Временной разрыв для связующих")
    print("=" * 80)
    
    # Загружаем модель
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()
    
    # Находим пары забегов
    df = model.df_clean.copy()
    df = df[df["status"] == "OK"].copy()
    
    logger.info("Поиск пар забегов для связующих бегунов...")
    pairs_df = find_linking_pairs(df)
    
    if len(pairs_df) == 0:
        print("\n❌ Не найдено пар забегов")
        return
    
    logger.info(f"Найдено {len(pairs_df)} пар забегов")
    
    # Анализ
    stats = analyze_time_gaps(pairs_df)
    
    # Сохраняем результаты
    output_dir = Path("outputs/test_2_time_gaps")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs_df.to_csv(output_dir / "linking_pairs.csv", index=False)
    
    # Визуализация
    plot_time_gaps(pairs_df, stats, output_dir)
    
    # Итоговая сводка
    print_summary(stats, pairs_df)
    
    # Сохраняем статистику
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "summary_stats.csv", index=False)
    
    logger.info(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
