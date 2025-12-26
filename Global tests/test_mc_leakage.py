"""
Тест для проверки утечки данных в визуализации Монте-Карло.

Проверяет:
1. Откуда берутся реальные данные для сравнения с MC?
2. Есть ли пересечение с train_frame_loy?
3. Правильно ли используется validation year?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from MarathonAgeModel import MarathonModel, easy_logging


def test_mc_data_sources(
        model: MarathonModel,
        race_id: str = "Казанский марафон",
        gender: str = "M",
        year_target: int = 2025,
        age_target: float = 40.0,
        age_band: float = 0.5,
) -> dict:
    """
    Анализирует источники данных для MC визуализации.

    Returns:
        dict с результатами анализа
    """

    print("=" * 80)
    print("ТЕСТ: Проверка источников данных для MC визуализации")
    print("=" * 80)

    # ============================================================
    # 1. Проверяем что используется в plot_monte_carlo_distribution
    # ============================================================

    print(f"\n1. ТЕКУЩИЙ СПОСОБ (как в plot_monte_carlo_distribution.py)")
    print(f"   Источник: model.df_clean")

    if model.df_clean is None:
        raise RuntimeError("model.df_clean is None")

    df_real_current = model.df_clean

    age_min = age_target - age_band
    age_max = age_target + age_band

    mask_current = (
            (df_real_current["race_id"] == race_id) &
            (df_real_current["gender"] == gender) &
            (df_real_current["year"] == year_target) &
            (df_real_current["status"] == "OK") &
            (df_real_current["age"] >= age_min) &
            (df_real_current["age"] <= age_max)
    )

    real_times_current = df_real_current.loc[mask_current, "time_seconds"].astype(float).to_numpy()
    real_times_current = real_times_current[np.isfinite(real_times_current)]
    real_times_current = real_times_current[real_times_current > 0.0]

    print(f"   Найдено наблюдений: {len(real_times_current)}")
    print(f"   Год: {year_target}")
    print(f"   Validation year модели: {model.validation_year}")

    # ============================================================
    # 2. Проверяем пересечение с train_frame_loy
    # ============================================================

    print(f"\n2. ПРОВЕРКА: Есть ли эти данные в train_frame_loy?")

    if model.train_frame_loy is None:
        raise RuntimeError("model.train_frame_loy is None")

    # Берём row_id из текущих реальных данных
    real_row_ids_current = df_real_current.loc[mask_current].index.tolist()

    # Проверяем сколько из них в train_frame_loy
    train_row_ids = model.train_frame_loy.index.tolist()

    overlap = set(real_row_ids_current) & set(train_row_ids)

    print(f"   Real data row_ids: {len(real_row_ids_current)}")
    print(f"   Train frame row_ids: {len(train_row_ids)}")
    print(f"   ПЕРЕСЕЧЕНИЕ: {len(overlap)} rows")

    if len(real_row_ids_current) > 0:
        overlap_percent = 100.0 * len(overlap) / len(real_row_ids_current)
        print(f"   Процент пересечения: {overlap_percent:.1f}%")

    # ============================================================
    # 3. Проверяем правильный LOY способ
    # ============================================================

    print(f"\n3. ПРАВИЛЬНЫЙ LOY СПОСОБ")
    print(f"   Источник: model.df_clean с фильтром year != validation_year")

    # Если year_target == validation_year, тогда это честная проверка
    # Если year_target != validation_year, тогда это утечка

    is_validation_year = (year_target == model.validation_year)

    if is_validation_year:
        print(f"   ✓ КОРРЕКТНО: year_target ({year_target}) == validation_year ({model.validation_year})")
        print(f"   Модель НЕ видела эти данные при обучении")
    else:
        print(f"   ✗ ПРОБЛЕМА: year_target ({year_target}) != validation_year ({model.validation_year})")
        print(f"   Модель МОГЛА видеть эти данные при обучении!")

        # Проверяем есть ли year_target в train_frame_loy
        train_years = model.train_frame_loy["year"].unique()
        if year_target in train_years:
            print(f"   ✗✗ УТЕЧКА ПОДТВЕРЖДЕНА: год {year_target} есть в train_frame_loy!")
        else:
            print(f"   ? Год {year_target} отсутствует в train_frame_loy")

    # ============================================================
    # 4. Сравниваем медианы
    # ============================================================

    print(f"\n4. СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ")

    if len(real_times_current) > 0:
        print(f"   Real data (текущий способ):")
        print(f"     Median: {np.median(real_times_current):.1f} сек")
        print(f"     Mean: {np.mean(real_times_current):.1f} сек")
        print(f"     Std: {np.std(real_times_current):.1f} сек")

    # Получаем MC предсказание
    result_mc = model.predict_with_uncertainty(
        race_id=race_id,
        gender=gender,
        age=age_target,
        year=year_target,
        confidence=0.95,
        method="monte_carlo",
        n_samples=10000,
    )

    time_pred = float(result_mc["time_pred"])

    print(f"   MC prediction:")
    print(f"     Median: {time_pred:.1f} сек")

    if len(real_times_current) > 0:
        diff = np.median(real_times_current) - time_pred
        diff_percent = 100.0 * diff / time_pred
        print(f"   Разница (Real - MC): {diff:.1f} сек ({diff_percent:+.1f}%)")

    # ============================================================
    # 5. Итоговый вердикт
    # ============================================================

    print(f"\n{'=' * 80}")
    print("ИТОГОВЫЙ ВЕРДИКТ:")
    print("=" * 80)

    if is_validation_year:
        if len(overlap) == 0:
            print("✓ НЕТ УТЕЧКИ: Реальные данные из validation year, которого модель не видела")
        else:
            print(f"⚠ СТРАННО: {len(overlap)} строк из validation year почему-то в train_frame_loy!")
            print("  Это может указывать на ошибку в _build_train_frame_for_loy()")
    else:
        print(f"✗ ПСЕВДОПРОВЕРКА: Вы сравниваете MC с данными из года {year_target},")
        print(f"  но validation_year = {model.validation_year}")
        print("  Для честной проверки используйте year_target = validation_year")

        if len(overlap) > 0:
            print(f"  ✗✗ УТЕЧКА: {len(overlap)} из {len(real_row_ids_current)} строк были в обучении!")

    return {
        'year_target': year_target,
        'validation_year': model.validation_year,
        'is_validation_year': is_validation_year,
        'n_real_data': len(real_times_current),
        'n_overlap_with_train': len(overlap),
        'overlap_percent': 100.0 * len(overlap) / len(real_row_ids_current) if len(real_row_ids_current) > 0 else 0.0,
        'real_median': np.median(real_times_current) if len(real_times_current) > 0 else np.nan,
        'mc_median': time_pred,
    }


def test_multiple_scenarios() -> None:
    """
    Тестирует несколько сценариев.
    """
    easy_logging(True)

    print("\n" + "=" * 80)
    print("ЗАПУСК МОДЕЛИ")
    print("=" * 80)

    model = MarathonModel(
        data_path=r"/Data",
        verbose=True,
    )
    model.run()

    print(f"\nМодель обучена с validation_year = {model.validation_year}")

    # ============================================================
    # Сценарий 1: Проверка на validation year (ПРАВИЛЬНО)
    # ============================================================

    print("\n" + "=" * 80)
    print("СЦЕНАРИЙ 1: Проверка на validation_year (должно быть корректно)")
    print("=" * 80)

    result1 = test_mc_data_sources(
        model=model,
        race_id="Казанский марафон",
        gender="M",
        year_target=model.validation_year,  # Используем validation year
        age_target=40.0,
    )

    # ============================================================
    # Сценарий 2: Проверка на другом году (НЕПРАВИЛЬНО)
    # ============================================================

    print("\n" + "=" * 80)
    print("СЦЕНАРИЙ 2: Проверка на другом году (может быть утечка)")
    print("=" * 80)

    # Берём год, который точно есть в данных и не является validation
    other_year = 2024 if model.validation_year != 2024 else 2023

    result2 = test_mc_data_sources(
        model=model,
        race_id="Казанский марафон",
        gender="M",
        year_target=other_year,  # Используем другой год
        age_target=40.0,
    )

    # ============================================================
    # Сценарий 3: Чемпионат России 2021 (из вашего вопроса)
    # ============================================================

    if model.validation_year == 2021:
        print("\n" + "=" * 80)
        print("СЦЕНАРИЙ 3: Чемпионат России 2021 (из вашего вопроса)")
        print("=" * 80)

        result3 = test_mc_data_sources(
            model=model,
            race_id="Чемпионат России",
            gender="M",
            year_target=2021,
            age_target=32.0,  # Средний возраст из анализа
        )

    # ============================================================
    # Итоговая сводка
    # ============================================================

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СВОДКА")
    print("=" * 80)

    print(f"\nВаш validation_year: {model.validation_year}")
    print(f"\nДля ЧЕСТНОЙ проверки MC всегда используйте:")
    print(f"  year_target = {model.validation_year}")
    print(f"\nДля ВИЗУАЛИЗАЦИИ на других годах:")
    print(f"  Укажите явно что это 'ретроспективная визуализация'")
    print(f"  и что данные года были в обучении")


if __name__ == "__main__":
    test_multiple_scenarios()