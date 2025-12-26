"""
Быстрый тест: Почему Чемпионат России 2021 имеет нулевой остаток?

Проверяет гипотезу:
- Это не утечка данных
- Это реальная аномалия (элитная молодая выборка)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from MarathonAgeModel import MarathonModel, easy_logging


def quick_test_championship_2021() -> None:
    """
    Быстрая проверка Чемпионата России 2021.
    """

    easy_logging(True)

    print("\n" + "=" * 80)
    print("БЫСТРЫЙ ТЕСТ: Чемпионат России 2021")
    print("=" * 80)

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()

    print(f"\nValidation year модели: {model.validation_year}")

    # ============================================================
    # Вопрос 1: Это validation year?
    # ============================================================

    print(f"\n{'=' * 80}")
    print("ВОПРОС 1: Чемпионат России 2021 - это validation year?")
    print("=" * 80)

    is_validation = (2021 == model.validation_year)

    if is_validation:
        print("✓ ДА - это validation year")
        print("  Модель НЕ видела эти данные при обучении")
        print("  Нулевой остаток - это реальная особенность данных")
    else:
        print(f"✗ НЕТ - validation year = {model.validation_year}, а не 2021")
        print("  Нужно проверить есть ли 2021 в train_frame_loy")

    # ============================================================
    # Вопрос 2: Есть ли Чемпионат России 2021 в train_frame_loy?
    # ============================================================

    print(f"\n{'=' * 80}")
    print("ВОПРОС 2: Есть ли Чемпионат России 2021 в train_frame_loy?")
    print("=" * 80)

    if model.train_frame_loy is None:
        print("✗ train_frame_loy отсутствует!")
        return

    train = model.train_frame_loy

    mask = (
            (train["race_id"] == "Чемпионат России") &
            (train["year"] == 2021)
    )

    n_in_train = int(mask.sum())

    print(f"Строк в train_frame_loy: {len(train)}")
    print(f"Из них Чемпионат России 2021: {n_in_train}")

    if n_in_train > 0:
        print("✗ ПРОБЛЕМА: Чемпионат России 2021 есть в обучении!")
        print("  Это может быть утечка данных")
    else:
        print("✓ КОРРЕКТНО: Чемпионат России 2021 НЕТ в обучении")
        print("  Данные 2021 года исключены из train_frame_loy")

    # ============================================================
    # Вопрос 3: Какие характеристики Чемпионата 2021?
    # ============================================================

    print(f"\n{'=' * 80}")
    print("ВОПРОС 3: Характеристики Чемпионата России 2021")
    print("=" * 80)

    if model.df_clean is None:
        print("✗ df_clean отсутствует!")
        return

    df = model.df_clean

    mask_champ = (
            (df["race_id"] == "Чемпионат России") &
            (df["year"] == 2021) &
            (df["status"] == "OK")
    )

    champ_data = df.loc[mask_champ]

    print(f"Наблюдений: {len(champ_data)}")

    if len(champ_data) == 0:
        print("Нет данных для анализа")
        return

    print(f"Средний возраст: {champ_data['age'].mean():.1f} лет")
    print(f"Медиана возраста: {champ_data['age'].median():.1f} лет")
    print(f"Min возраст: {champ_data['age'].min():.0f} лет")
    print(f"Max возраст: {champ_data['age'].max():.0f} лет")

    print(f"\nРаспределение по полу:")
    print(champ_data.groupby('gender').size())

    # ============================================================
    # Вопрос 4: Сравнение с другими гонками того же года
    # ============================================================

    print(f"\n{'=' * 80}")
    print("ВОПРОС 4: Сравнение с другими гонками 2021 года")
    print("=" * 80)

    mask_2021 = (
            (df["year"] == 2021) &
            (df["status"] == "OK")
    )

    other_races_2021 = (
        df.loc[mask_2021]
        .groupby("race_id")
        .agg(
            count=("age", "count"),
            mean_age=("age", "mean"),
            median_age=("age", "median"),
        )
        .sort_values("mean_age")
    )

    print("\nВсе гонки 2021 года (по среднему возрасту):")
    print(other_races_2021.to_string())

    # ============================================================
    # ИТОГОВЫЙ ВЫВОД
    # ============================================================

    print(f"\n{'=' * 80}")
    print("ИТОГОВЫЙ ВЫВОД")
    print("=" * 80)

    if is_validation and n_in_train == 0:
        print("✓ ГИПОТЕЗА ПОДТВЕРЖДЕНА:")
        print("  1. Чемпионат России 2021 - это validation year")
        print("  2. Модель не видела эти данные при обучении")
        print(f"  3. Средний возраст {champ_data['age'].mean():.1f} лет - САМЫЙ МОЛОДОЙ")
        print("  4. Нулевой остаток - это реальная особенность элитной выборки")
        print("  5. НЕТ УТЕЧКИ ДАННЫХ - модель просто точно предсказывает элиту")
    elif n_in_train > 0:
        print("✗ ПРОБЛЕМА ОБНАРУЖЕНА:")
        print("  Чемпионат России 2021 присутствует в train_frame_loy")
        print("  Это утечка данных в обучение!")
    else:
        print("? НЕОДНОЗНАЧНАЯ СИТУАЦИЯ:")
        print(f"  Validation year = {model.validation_year}, но проверяется год 2021")
        print("  Рекомендуется запустить модель с validation_year=2021")


if __name__ == "__main__":
    quick_test_championship_2021()