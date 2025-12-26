"""
Тест для проверки гипотезы "Парадокс Элиты".

Проверяет:
1. Все ли элитные забеги (Чемпионаты) показывают bias ≈ 0?
2. Есть ли корреляция между составом участников и bias?
3. Отличаются ли характеристики разных годов Чемпионата?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from MarathonAgeModel import MarathonModel, easy_logging


def analyze_championship_bias(validation_year: int = 2024) -> None:
    """
    Детальный анализ Чемпионатов России по годам.
    """

    easy_logging(True)

    print("\n" + "=" * 80)
    print("АНАЛИЗ ГИПОТЕЗЫ 'ПАРАДОКС ЭЛИТЫ'")
    print("=" * 80)

    # Строим модель
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
        validation_year=validation_year,
    )
    model.run()

    # Берём train_frame_loy
    if model.train_frame_loy is None:
        raise RuntimeError("train_frame_loy is None")

    train = model.train_frame_loy.copy()

    print(f"\nValidation year: {validation_year}")
    print(f"Train data: {len(train)} rows")

    # ============================================================
    # 1. Находим все Чемпионаты России
    # ============================================================

    print("\n" + "=" * 80)
    print("1. ВСЕ ЧЕМПИОНАТЫ РОССИИ В TRAIN ДАННЫХ")
    print("=" * 80)

    championships = train[train['race_id'] == 'Чемпионат России'].copy()

    if len(championships) == 0:
        print("✗ Нет данных по Чемпионатам России в train")
        return

    print(f"\nНайдено: {len(championships)} финишей в Чемпионатах России")

    # Группируем по годам
    champ_by_year = (
        championships
        .groupby('year')
        .agg(
            count=('Z', 'count'),
            mean_age=('age', 'mean'),
            median_age=('age', 'median'),
            std_age=('age', 'std'),
            min_age=('age', 'min'),
            max_age=('age', 'max'),
            mean_time=('time_seconds', 'mean'),
            median_time=('time_seconds', 'median'),
        )
        .sort_index()
    )

    print("\nЧемпионаты России по годам:")
    print(champ_by_year.to_string())

    # ============================================================
    # 2. Вычисляем остатки для каждого года Чемпионата
    # ============================================================

    print("\n" + "=" * 80)
    print("2. ОСТАТКИ ПО ГОДАМ ЧЕМПИОНАТА")
    print("=" * 80)

    # Вычисляем остатки для каждого наблюдения
    champ_residuals = []

    for gender in ['M', 'F']:
        gender_data = championships[championships['gender'] == gender].copy()
        if len(gender_data) == 0:
            continue

        if gender not in model.age_spline_models:
            continue

        x_values = gender_data['x'].astype(float).to_numpy()
        z_values = gender_data['Z'].astype(float).to_numpy()
        h_values = model.age_spline_models[gender].predict_h(x_values)
        residual_z = z_values - h_values

        for idx, (year, res, age, time, x, z, h) in enumerate(zip(
                gender_data['year'],
                residual_z,
                gender_data['age'],
                gender_data['time_seconds'],
                x_values,
                z_values,
                h_values,
        )):
            champ_residuals.append({
                'year': int(year),
                'gender': gender,
                'residual_z': float(res),
                'age': float(age),
                'time_seconds': float(time),
                'x': float(x),
                'Z': float(z),
                'h': float(h),
            })

    champ_res_df = pd.DataFrame(champ_residuals)

    # Статистика по годам
    champ_stats = (
        champ_res_df
        .groupby('year')
        .agg(
            count=('residual_z', 'count'),
            mean_residual=('residual_z', 'mean'),
            median_residual=('residual_z', 'median'),
            std_residual=('residual_z', 'std'),
            mean_age=('age', 'mean'),
        )
    )
    champ_stats['se_residual'] = champ_stats['std_residual'] / np.sqrt(champ_stats['count'])

    print("\nОстатки Чемпионатов по годам:")
    print(champ_stats.to_string())

    # ============================================================
    # 3. Сравнение с массовыми марафонами
    # ============================================================

    print("\n" + "=" * 80)
    print("3. СРАВНЕНИЕ: ЧЕМПИОНАТЫ vs МАССОВЫЕ ЗАБЕГИ")
    print("=" * 80)

    # Берём массовые марафоны
    mass_races = ['Казанский марафон', 'Дорога жизни', 'Пермский марафон']

    comparison = []

    # Чемпионаты
    for year, row in champ_stats.iterrows():
        comparison.append({
            'race_type': 'Чемпионат',
            'race_id': 'Чемпионат России',
            'year': int(year),
            'count': int(row['count']),
            'mean_residual': float(row['mean_residual']),
            'se_residual': float(row['se_residual']),
            'mean_age': float(row['mean_age']),
        })

    # Массовые марафоны
    for race_id in mass_races:
        race_data = train[train['race_id'] == race_id].copy()

        if len(race_data) == 0:
            continue

        for year in race_data['year'].unique():
            year_data = race_data[race_data['year'] == year].copy()

            # Вычисляем остатки
            residuals_year = []
            for gender in ['M', 'F']:
                gender_data = year_data[year_data['gender'] == gender].copy()
                if len(gender_data) == 0 or gender not in model.age_spline_models:
                    continue

                x = gender_data['x'].astype(float).to_numpy()
                z = gender_data['Z'].astype(float).to_numpy()
                h = model.age_spline_models[gender].predict_h(x)
                residuals_year.extend(z - h)

            if len(residuals_year) > 0:
                comparison.append({
                    'race_type': 'Массовый',
                    'race_id': race_id,
                    'year': int(year),
                    'count': len(residuals_year),
                    'mean_residual': float(np.mean(residuals_year)),
                    'se_residual': float(np.std(residuals_year) / np.sqrt(len(residuals_year))),
                    'mean_age': float(year_data['age'].mean()),
                })

    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values(['race_type', 'race_id', 'year'])

    print("\nСравнительная таблица:")
    print(comparison_df.to_string(index=False))

    # ============================================================
    # 4. Статистический анализ
    # ============================================================

    print("\n" + "=" * 80)
    print("4. СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 80)

    champ_residuals_all = comparison_df[comparison_df['race_type'] == 'Чемпионат']['mean_residual'].values
    mass_residuals_all = comparison_df[comparison_df['race_type'] == 'Массовый']['mean_residual'].values

    print(f"\nЧемпионаты России:")
    print(f"  N годов: {len(champ_residuals_all)}")
    print(f"  Mean residual: {np.mean(champ_residuals_all):+.6f}")
    print(f"  Median residual: {np.median(champ_residuals_all):+.6f}")
    print(f"  Std residual: {np.std(champ_residuals_all):.6f}")
    print(f"  Range: [{np.min(champ_residuals_all):.6f}, {np.max(champ_residuals_all):.6f}]")

    print(f"\nМассовые марафоны:")
    print(f"  N комбинаций: {len(mass_residuals_all)}")
    print(f"  Mean residual: {np.mean(mass_residuals_all):+.6f}")
    print(f"  Median residual: {np.median(mass_residuals_all):+.6f}")
    print(f"  Std residual: {np.std(mass_residuals_all):.6f}")
    print(f"  Range: [{np.min(mass_residuals_all):.6f}, {np.max(mass_residuals_all):.6f}]")

    # T-test
    from scipy import stats as scipy_stats
    t_stat, p_value = scipy_stats.ttest_ind(champ_residuals_all, mass_residuals_all)

    print(f"\nT-test (Чемпионаты vs Массовые):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")

    if p_value < 0.05:
        print(f"  ✓ Различие статистически значимо (p < 0.05)")
    else:
        print(f"  ✗ Различие НЕ значимо (p >= 0.05)")

    # ============================================================
    # 5. Анализ характеристик участников
    # ============================================================

    print("\n" + "=" * 80)
    print("5. ХАРАКТЕРИСТИКИ УЧАСТНИКОВ")
    print("=" * 80)

    # Добавляем процентиль времени для каждого забега
    comparison_detailed = []

    for _, row in comparison_df.iterrows():
        race_id = row['race_id']
        year = row['year']

        # Берём данные для этого забега
        race_year_data = train[
            (train['race_id'] == race_id) &
            (train['year'] == year)
            ].copy()

        if len(race_year_data) > 0:
            # Вычисляем процентили
            times = race_year_data['time_seconds'].values
            p5 = np.percentile(times, 5)
            p25 = np.percentile(times, 25)
            p50 = np.percentile(times, 50)
            p75 = np.percentile(times, 75)
            p95 = np.percentile(times, 95)

            comparison_detailed.append({
                **row.to_dict(),
                'time_p5': p5,
                'time_p25': p25,
                'time_p50': p50,
                'time_p75': p75,
                'time_p95': p95,
                'time_range': p95 - p5,
            })

    detailed_df = pd.DataFrame(comparison_detailed)

    print("\nПроцентили времени (секунды):")
    print(detailed_df[['race_type', 'race_id', 'year', 'time_p5', 'time_p50', 'time_p95', 'time_range']].to_string(
        index=False))

    # ============================================================
    # 6. Корреляционный анализ
    # ============================================================

    print("\n" + "=" * 80)
    print("6. КОРРЕЛЯЦИИ")
    print("=" * 80)

    # Корреляция: возраст vs остаток
    corr_age = comparison_df['mean_age'].corr(comparison_df['mean_residual'])
    print(f"\nКорреляция (mean_age, mean_residual): {corr_age:.4f}")

    # Корреляция: размер выборки vs остаток
    corr_size = comparison_df['count'].corr(comparison_df['mean_residual'])
    print(f"Корреляция (count, mean_residual): {corr_size:.4f}")

    # ============================================================
    # 7. Визуализация
    # ============================================================

    print("\n" + "=" * 80)
    print("7. СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 80)

    output_dir = Path("outputs/elite_hypothesis_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # График 1: Boxplot Чемпионаты vs Массовые
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Остатки
    ax = axes[0]
    comparison_df.boxplot(column='mean_residual', by='race_type', ax=ax)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_title('Residuals: Championships vs Mass Races')
    ax.set_xlabel('Race Type')
    ax.set_ylabel('Mean Residual (Z scale)')
    plt.sca(ax)
    plt.xticks(rotation=0)

    # Subplot 2: Возраст
    ax = axes[1]
    comparison_df.boxplot(column='mean_age', by='race_type', ax=ax)
    ax.set_title('Age: Championships vs Mass Races')
    ax.set_xlabel('Race Type')
    ax.set_ylabel('Mean Age')
    plt.sca(ax)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plot1_path = output_dir / 'championship_vs_mass_boxplot.png'
    plt.savefig(plot1_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot1_path}")

    # График 2: Временной ряд остатков для Чемпионата
    fig, ax = plt.subplots(figsize=(10, 6))

    champ_only = comparison_df[comparison_df['race_type'] == 'Чемпионат'].copy()

    ax.errorbar(
        champ_only['year'],
        champ_only['mean_residual'],
        yerr=champ_only['se_residual'],
        fmt='o-',
        capsize=5,
        label='Чемпионат России',
        markersize=8,
    )

    ax.axhline(0, color='red', linestyle='--', alpha=0.7, label='Zero bias')
    ax.axhline(np.mean(mass_residuals_all), color='orange', linestyle='--', alpha=0.7,
               label=f'Mean mass races ({np.mean(mass_residuals_all):.3f})')

    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Residual (Z scale)')
    ax.set_title('Championship Russia: Year Effects Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot2_path = output_dir / 'championship_timeline.png'
    plt.savefig(plot2_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot2_path}")

    # ============================================================
    # 8. Сохранение результатов
    # ============================================================

    comparison_path = output_dir / 'championship_vs_mass_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved: {comparison_path}")

    champ_stats_path = output_dir / 'championship_by_year.csv'
    champ_stats.to_csv(champ_stats_path)
    print(f"Saved: {champ_stats_path}")

    # ============================================================
    # 9. ИТОГОВЫЕ ВЫВОДЫ
    # ============================================================

    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("=" * 80)

    # Проверка гипотезы "элита"
    champ_mean = np.mean(champ_residuals_all)
    mass_mean = np.mean(mass_residuals_all)

    print(f"\n1. ГИПОТЕЗА 'ПАРАДОКС ЭЛИТЫ':")

    if abs(champ_mean) < 0.05 and mass_mean > 0.15:
        print(f"   ✓ ПОДТВЕРЖДЕНА!")
        print(f"   Чемпионаты: {champ_mean:+.3f} (близко к 0)")
        print(f"   Массовые: {mass_mean:+.3f} (сильный положительный bias)")
    elif abs(champ_mean - mass_mean) < 0.05:
        print(f"   ✗ ОПРОВЕРГНУТА!")
        print(f"   Чемпионаты и массовые забеги показывают одинаковый bias")
        print(f"   Чемпионаты: {champ_mean:+.3f}")
        print(f"   Массовые: {mass_mean:+.3f}")
    else:
        print(f"   ? ЧАСТИЧНО ПОДТВЕРЖДЕНА")
        print(f"   Чемпионаты: {champ_mean:+.3f}")
        print(f"   Массовые: {mass_mean:+.3f}")
        print(f"   Разница: {mass_mean - champ_mean:.3f}")

    print(f"\n2. КОНСИСТЕНТНОСТЬ ЧЕМПИОНАТОВ:")
    champ_std = np.std(champ_residuals_all)
    print(f"   Std остатков Чемпионатов: {champ_std:.3f}")

    if champ_std < 0.05:
        print(f"   ✓ Все года Чемпионата показывают схожий bias")
    else:
        print(f"   ✗ Разные года показывают разный bias")
        print(f"   Нужно искать другое объяснение")

    print(f"\n3. РЕКОМЕНДАЦИИ:")

    if abs(champ_mean) < 0.1 and mass_mean > 0.15:
        print(f"   → Модель действительно описывает элиту")
        print(f"   → Joint Estimation с δ(race, year) необходим")
        print(f"   → δ будет разделять элиту (≈0) и любителей (≈+0.2)")
    else:
        print(f"   → Гипотеза 'Парадокс Элиты' не объясняет всё")
        print(f"   → Нужен более детальный анализ факторов")
        print(f"   → Возможно, year effects связаны с другими причинами")


if __name__ == "__main__":
    analyze_championship_bias(validation_year=2025)