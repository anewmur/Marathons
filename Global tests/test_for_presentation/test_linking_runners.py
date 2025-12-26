"""
Проверка гипотезы: достаточно ли "связующих бегунов" для калибровки эталонов элиты vs массовки?

Цель: Ответить на вопрос руководителя:
"Можем ли мы построить отдельные эталоны для элитных и массовых стартов,
используя бегунов, которые участвуют в обоих типах?"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from MarathonAgeModel import MarathonModel, easy_logging

logger = logging.getLogger(__name__)


def classify_race_type(race_id: str) -> str:
    """
    Классификация трасс на элитные и массовые.
    
    Элитные: Чемпионаты, отборочные
    Массовые: Крупные городские марафоны
    """
    race_lower = race_id.lower()
    
    # Элитные старты
    if "чемпионат" in race_lower or "championship" in race_lower:
        return "elite"
    if "отбор" in race_lower or "qualifying" in race_lower:
        return "elite"
    
    # Массовые старты (крупные городские)
    mass_keywords = ["казан", "белые ночи", "white nights", "дорога жизни", "road of life", "пермь"]
    if any(keyword in race_lower for keyword in mass_keywords):
        return "mass"
    
    # По умолчанию считаем массовым (консервативная оценка)
    return "mass"


def analyze_linking_runners(model: MarathonModel) -> Dict:
    """
    Анализирует "связующих" бегунов между элитными и массовыми стартами.
    
    Returns:
        Dict с результатами анализа
    """
    if model.df_clean is None:
        raise RuntimeError("model.df_clean отсутствует. Запустите model.run()")
    
    df = model.df_clean.copy()
    df = df[df["status"] == "OK"].copy()
    
    # Классифицируем трассы
    df["race_type"] = df["race_id"].apply(classify_race_type)
    
    logger.info("Классификация трасс:")
    for race_id in sorted(df["race_id"].unique()):
        race_type = classify_race_type(race_id)
        n_participants = len(df[df["race_id"] == race_id])
        logger.info(f"  {race_id}: {race_type} (n={n_participants})")
    
    # Считаем участие бегунов
    runner_participation = (
        df.groupby("runner_id")
        .agg(
            n_races=("race_id", "nunique"),
            n_elite=("race_type", lambda x: (x == "elite").sum()),
            n_mass=("race_type", lambda x: (x == "mass").sum()),
            races_list=("race_id", lambda x: list(x.unique())),
        )
    )
    
    # Связующие бегуны: участвуют и в элитных, и в массовых
    linking_runners = runner_participation[
        (runner_participation["n_elite"] >= 1) & 
        (runner_participation["n_mass"] >= 1)
    ]
    
    total_runners = len(runner_participation)
    n_linking = len(linking_runners)
    pct_linking = 100.0 * n_linking / total_runners if total_runners > 0 else 0.0
    
    # Анализ пар трасс
    race_pairs: Dict[Tuple[str, str], Set[str]] = {}
    
    for _, row in linking_runners.iterrows():
        races = row["races_list"]
        elite_races = [r for r in races if classify_race_type(r) == "elite"]
        mass_races = [r for r in races if classify_race_type(r) == "mass"]
        
        for elite in elite_races:
            for mass in mass_races:
                pair = (elite, mass)
                if pair not in race_pairs:
                    race_pairs[pair] = set()
                race_pairs[pair].add(row.name)  # runner_id
    
    # Топ пар по количеству связующих бегунов
    pairs_sorted = sorted(
        [(pair, len(runners)) for pair, runners in race_pairs.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    results = {
        "total_runners": total_runners,
        "n_linking_runners": n_linking,
        "pct_linking_runners": pct_linking,
        "linking_runners_df": linking_runners,
        "race_pairs": pairs_sorted,
        "n_elite_races": len(df[df["race_type"] == "elite"]["race_id"].unique()),
        "n_mass_races": len(df[df["race_type"] == "mass"]["race_id"].unique()),
    }
    
    return results


def print_summary(results: Dict) -> None:
    """
    Печатает краткую сводку результатов.
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ СВЯЗУЮЩИХ БЕГУНОВ")
    print("=" * 80)
    
    print(f"\nОбщая статистика:")
    print(f"  Всего уникальных бегунов: {results['total_runners']}")
    print(f"  Элитных трасс: {results['n_elite_races']}")
    print(f"  Массовых трасс: {results['n_mass_races']}")
    
    print(f"\nСвязующие бегуны (участвуют в обоих типах):")
    print(f"  Количество: {results['n_linking_runners']}")
    print(f"  Процент: {results['pct_linking_runners']:.1f}%")
    
    print(f"\nТОП-10 пар трасс по количеству связующих бегунов:")
    for i, ((elite, mass), n) in enumerate(results["race_pairs"][:10], 1):
        print(f"  {i}. {elite} ↔ {mass}: {n} бегунов")
    
    # Проверка достаточности данных
    print("\n" + "=" * 80)
    print("ОЦЕНКА ДОСТАТОЧНОСТИ ДАННЫХ")
    print("=" * 80)
    
    threshold_weak = 10
    threshold_ok = 30
    threshold_good = 50
    
    pairs_with_data = [n for _, n in results["race_pairs"]]
    
    if not pairs_with_data:
        print("❌ НЕТ СВЯЗУЮЩИХ БЕГУНОВ")
        print("   Калибровка невозможна")
    else:
        max_linking = max(pairs_with_data)
        n_pairs_ok = sum(1 for n in pairs_with_data if n >= threshold_ok)
        
        print(f"Максимальное количество связующих для одной пары: {max_linking}")
        print(f"Пар с ≥{threshold_ok} связующими: {n_pairs_ok}")
        
        if max_linking < threshold_weak:
            print("\n❌ КРИТИЧЕСКИ МАЛО ДАННЫХ")
            print(f"   Даже лучшая пара имеет <{threshold_weak} связующих бегунов")
            print("   Калибровка будет крайне ненадежной")
        elif max_linking < threshold_ok:
            print("\n⚠️  ДАННЫХ НЕДОСТАТОЧНО")
            print(f"   Лучшая пара имеет <{threshold_ok} связующих")
            print("   Калибровка возможна, но с высокой неопределенностью")
        elif max_linking < threshold_good:
            print("\n✓ ДАННЫХ ДОСТАТОЧНО (С ОГРАНИЧЕНИЯМИ)")
            print(f"   Лучшая пара имеет {max_linking} связующих")
            print("   Калибровка возможна для нескольких пар трасс")
            print("   Но не все пары будут хорошо калиброваны")
        else:
            print("\n✓✓ ДАННЫХ ДОСТАТОЧНО")
            print(f"   Лучшая пара имеет {max_linking} связующих")
            print("   Калибровка надежна для топовых пар")
    
    print("=" * 80)


def plot_linking_network(results: Dict, output_dir: Path) -> None:
    """
    Визуализация сети связующих бегунов.
    """
    pairs = results["race_pairs"][:15]  # Топ-15 пар
    
    if not pairs:
        logger.warning("Нет пар для визуализации")
        return
    
    # Готовим данные для матрицы
    elite_races = sorted(set(elite for (elite, _), _ in pairs))
    mass_races = sorted(set(mass for (_, mass), _ in pairs))
    
    matrix = np.zeros((len(elite_races), len(mass_races)))
    
    for (elite, mass), n in pairs:
        i = elite_races.index(elite)
        j = mass_races.index(mass)
        matrix[i, j] = n
    
    # Создаем heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        xticklabels=mass_races,
        yticklabels=elite_races,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Количество связующих бегунов"},
    )
    plt.title("Связующие бегуны между элитными и массовыми стартами")
    plt.xlabel("Массовые старты")
    plt.ylabel("Элитные старты")
    plt.tight_layout()
    
    output_path = output_dir / "linking_runners_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Heatmap сохранен: {output_path}")


def main():
    """
    Главная функция.
    """
    easy_logging(True)
    
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ГИПОТЕЗЫ: Достаточно ли связующих бегунов?")
    print("=" * 80)
    
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()
    
    results = analyze_linking_runners(model)
    print_summary(results)
    
    # Сохраняем результаты
    output_dir = Path("outputs/linking_runners_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем CSV с детальной информацией
    linking_df = results["linking_runners_df"]
    linking_df.to_csv(output_dir / "linking_runners.csv")
    logger.info(f"Детальные данные: {output_dir / 'linking_runners.csv'}")
    
    # Визуализация
    plot_linking_network(results, output_dir)
    
    # ИТОГОВЫЙ ВЫВОД ДЛЯ ПРЕЗЕНТАЦИИ
    print("\n" + "=" * 80)
    print("ВЫВОД ДЛЯ ПРЕЗЕНТАЦИИ")
    print("=" * 80)
    
    n_linking = results["n_linking_runners"]
    pct = results["pct_linking_runners"]
    
    if not results["race_pairs"]:
        print("\n❌ ГИПОТЕЗА РУКОВОДИТЕЛЯ НЕ ПОДТВЕРЖДАЕТСЯ")
        print("   Связующих бегунов между элитой и массовкой недостаточно")
        print("   Рекомендация: защищать текущий подход (Top-5% + κ_c + δ_year)")
    else:
        max_linking = max(n for _, n in results["race_pairs"])
        
        if max_linking < 30:
            print("\n⚠️  ГИПОТЕЗА РУКОВОДИТЕЛЯ СЛАБО ПОДТВЕРЖДАЕТСЯ")
            print(f"   Связующих бегунов есть ({n_linking}, {pct:.1f}%)")
            print(f"   Но их недостаточно для надежной калибровки (<30 на пару)")
            print("\n   Рекомендация для презентации:")
            print("   1. Показать эти данные")
            print("   2. Объяснить, что калибровка возможна, но с высокой неопределенностью")
            print("   3. Предложить MVP (Top-5%) + R&D (калибровка на связующих)")
        else:
            print("\n✓ ГИПОТЕЗА РУКОВОДИТЕЛЯ ПОДТВЕРЖДАЕТСЯ")
            print(f"   Связующих бегунов достаточно для калибровки топовых пар")
            print(f"   Лучшая пара: {max_linking} бегунов")
            print("\n   Рекомендация для презентации:")
            print("   1. Признать, что данные ЕСТЬ")
            print("   2. Сравнить два подхода:")
            print("      - Top-5% + поправки (простой, работает везде)")
            print("      - Калибровка через связующих (точнее, но требует данных)")
            print("   3. Предложить HYBRID: калибровка где можно, Top-5% где нет данных")


if __name__ == "__main__":
    main()
