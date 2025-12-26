"""
График: Временная стабильность трассовой поправки Δ_c(age)

Проверяет: переносима ли трассовая поправка между годами?

Идея:
- Для одной массовой трассы (например, Белые ночи)
- По каждому году j вычисляем: Δ̂_{c,j}(age) = median(Z) в возрастных бинах
- Где Z = ln(T) - f(age), f(age) = глобальная граница (фиксирована!)
- Накладываем кривые разных лет

Если кривые совпадают → поправка стабильна → идея работает
Если кривые расходятся → поправка впитала год/состав → идея НЕ работает

Две версии:
1. Все участники
2. Только "связующие" (с историей на других стартах)
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

sys.path.insert(0, str(Path(__file__).parent.parent))

from MarathonAgeModel import MarathonModel


def estimate_delta_by_year(
    df: pd.DataFrame,
    spline: UnivariateSpline,
    age_min: float,
    age_max: float,
    bin_width: float = 1.0,
) -> pd.DataFrame:
    """
    Оценивает Δ̂(age) = median(Z) по возрастным бинам.

    Args:
        df: DataFrame с колонками age, ln_time
        spline: Функция f(age)
        age_min, age_max: Диапазон возрастов
        bin_width: Ширина бина в годах

    Returns:
        DataFrame с колонками: age, delta, n
    """

    # Вычисляем остатки
    df = df.copy()
    df['f_age'] = spline(df['age'])
    df['residual_z'] = df['ln_time'] - df['f_age']

    # Возрастные бины
    ages = np.arange(age_min, age_max + 1, bin_width)

    results = []

    for age_center in ages:
        df_bin = df[
            (df['age'] >= age_center - bin_width/2) &
            (df['age'] < age_center + bin_width/2)
        ]

        if len(df_bin) >= 10:  # Минимум 10 наблюдений
            results.append({
                'age': age_center,
                'delta': df_bin['residual_z'].median(),
                'n': len(df_bin),
            })

    return pd.DataFrame(results)


def smooth_delta(df_delta: pd.DataFrame, s: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Сглаживает Δ(age) сплайном.

    Returns:
        ages_smooth, delta_smooth
    """

    if len(df_delta) < 4:
        return df_delta['age'].values, df_delta['delta'].values

    spline = UnivariateSpline(df_delta['age'], df_delta['delta'], s=s)
    ages_smooth = np.linspace(df_delta['age'].min(), df_delta['age'].max(), 100)
    delta_smooth = spline(ages_smooth)

    return ages_smooth, delta_smooth


def create_temporal_stability_plot() -> None:
    """
    Создает график временной стабильности трассовой поправки.
    """

    MIN_DELTA_LINKING = 3
    MIN_NUM_RUNNER = 100
    MIN_YEAR_LINKING = 10
    print("=" * 80)
    print("ГРАФИК: Временная стабильность трассовой поправки Δ_c(age)")
    print("=" * 80)

    # ============================================================
    # 1. Загружаем данные
    # ============================================================

    data_path = Path("C:/Users/andre/github/Marathons/Data")

    if not data_path.exists():
        data_path = Path("/Data")

    if not data_path.exists():
        raise RuntimeError(f"Не найден путь к данным: {data_path}")

    print(f"\nСоздаем модель из: {data_path}")

    model = MarathonModel(data_path=data_path, verbose=True)
    model.load_data().filter_raw().validate_raw().add_row_id().preprocess()

    if model.df_clean is None:
        raise RuntimeError("model.df_clean is None")

    df = model.df_clean.copy()

    print(f"\n✓ Загружено финишей: {len(df)}")

    # ============================================================
    # 2. Загружаем глобальную границу f(age)
    # ============================================================

    boundary_path = Path("outputs/boundary_function.csv")

    if not boundary_path.exists():
        raise RuntimeError(
            f"Файл границы не найден: {boundary_path}\n"
            "Сначала запустите create_1_global_boundary.py"
        )

    print("\n✓ Загружаем границу f(age)...")
    boundary_data = pd.read_csv(boundary_path)
    spline = UnivariateSpline(
        boundary_data['age'],
        boundary_data['ln_time_boundary'],
        s=0
    )

    age_min, age_max = boundary_data['age'].min(), boundary_data['age'].max()
    print(f"✓ Граница определена для возрастов: {age_min:.0f}-{age_max:.0f} лет")

    # ============================================================
    # 3. Выбираем массовую трассу с несколькими годами
    # ============================================================

    print("\nАнализируем трассы...")

    # Ищем массовую трассу с несколькими годами
    race_year_counts = df.groupby(['race_id', 'year']).size().reset_index(name='n')
    race_year_counts = race_year_counts[race_year_counts['n'] >= MIN_NUM_RUNNER]

    # Считаем сколько лет у каждой трассы
    races_with_years = race_year_counts.groupby('race_id')['year'].nunique().reset_index(name='n_years')
    races_with_years = races_with_years[races_with_years['n_years'] >= 3]  # Минимум 3 года

    if len(races_with_years) == 0:
        raise RuntimeError("Нет трасс с ≥3 годами и ≥150 участников/год")

    # Выбираем самую массовую
    race_total_counts = df.groupby('race_id').size().reset_index(name='total_n')
    race_total_counts = race_total_counts[race_total_counts['race_id'].isin(races_with_years['race_id'])]
    race_total_counts = race_total_counts.sort_values('total_n', ascending=False)

    # Показываем ТОП-5 кандидатов
    print("\nКандидаты (≥3 года, ≥150 участников/год):")
    for i, row in race_total_counts.head(5).iterrows():
        race_id = row['race_id']
        total_n = row['total_n']
        n_years = races_with_years[races_with_years['race_id'] == race_id]['n_years'].values[0]
        print(f"  {race_id:30s}: {total_n:5.0f} участников, {n_years} лет")

    # Выбираем самую массовую
    target_race = race_total_counts.iloc[0]['race_id']

    print(f"\n✓ ВЫБРАНА: {target_race}")


    # Фильтруем данные
    df_race = df[
        (df['race_id'] == target_race) &
        (df['gender'] == 'M') &  # Только мужчины
        (df['age'] >= age_min) &
        (df['age'] <= age_max)
    ].copy()

    years = sorted(df_race['year'].unique())
    print(f"✓ Годы: {years}")
    print(f"✓ Всего наблюдений: {len(df_race)}")

    # Добавляем ln(time)
    df_race['ln_time'] = np.log(df_race['time_seconds'])

    # ============================================================
    # 4. Определяем "связующих" участников (ДВА ВАРИАНТА)
    # ============================================================

    print("\nОпределяем типы участников...")

    # ВАРИАНТ 1: Опытные (многостартовые) — бегали ≥2 трассы ЗА ВСЁ ВРЕМЯ
    runner_race_counts = df[df['gender'] == 'M'].groupby('runner_id')['race_id'].nunique()
    experienced_runners = runner_race_counts[runner_race_counts >= 2].index

    df_race['is_experienced'] = df_race['runner_id'].isin(experienced_runners)

    n_experienced = df_race['is_experienced'].sum()
    print(f"✓ Опытных участников (≥2 трассы за всё время): {n_experienced} ({100*n_experienced/len(df_race):.1f}%)")

    # ВАРИАНТ 2: Сезонные linking — бегали ≥2 трассы В ОДИН ГОД
    df_race['is_season_linking'] = False

    for year in years:
        # Для этого года: кто бегал ≥2 трассы?
        df_year_all_races = df[(df['year'] == year) & (df['gender'] == 'M')]

        runner_races_this_year = df_year_all_races.groupby('runner_id')['race_id'].nunique()
        linking_this_year = runner_races_this_year[runner_races_this_year >= 2].index

        # Помечаем в нашей трассе
        mask = (df_race['year'] == year) & (df_race['runner_id'].isin(linking_this_year))
        df_race.loc[mask, 'is_season_linking'] = True

    n_season_linking = df_race['is_season_linking'].sum()
    print(f"✓ Сезонных linking (≥2 трассы в один год): {n_season_linking} ({100*n_season_linking/len(df_race):.1f}%)")


    # ============================================================
    # 5. Вычисляем Δ̂_{c,j}(age) для каждого года
    # ============================================================

    print("\nВычисляем трассовые поправки по годам...")

    deltas_all = {}
    deltas_experienced = {}
    deltas_season_linking = {}

    for year in years:
        # Все участники
        df_year_all = df_race[df_race['year'] == year]
        delta_all = estimate_delta_by_year(df_year_all, spline, age_min, age_max)

        if len(delta_all) >= 4:
            deltas_all[year] = delta_all
            print(f"  {year} (все): {len(df_year_all)} наблюдений, {len(delta_all)} возрастных бинов")

        # Опытные
        df_year_experienced = df_race[(df_race['year'] == year) & (df_race['is_experienced'])]

        if len(df_year_experienced) >= 50:
            delta_experienced = estimate_delta_by_year(df_year_experienced, spline, age_min, age_max)

            if len(delta_experienced) >= 4:
                deltas_experienced[year] = delta_experienced
                print(f"  {year} (опыт): {len(df_year_experienced)} наблюдений, {len(delta_experienced)} возрастных бинов")

        # Сезонные linking (калибровочные бегуны)
        df_year_linking = df_race[(df_race['year'] == year) & (df_race['is_season_linking'])]

        print(f"  {year} (калибр): {len(df_year_linking)} наблюдений", end='')

        if len(df_year_linking) >= MIN_YEAR_LINKING:
            delta_linking = estimate_delta_by_year(df_year_linking, spline, age_min, age_max)

            print(f", {len(delta_linking)} возрастных бинов", end='')

            if len(delta_linking) >= MIN_DELTA_LINKING:  #
                deltas_season_linking[year] = delta_linking
                print(f" ✓")
            else:
                print(f" ✗ (мало бинов)")
        else:
            print(f" ✗ (мало участников)")



    # ============================================================
    # 6. Визуализация
    # ============================================================

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

    # Цвета для годов
    colors = plt.cm.rainbow(np.linspace(0, 1, len(years)))

    # --- График 1: Все участники ---

    for year, color in zip(years, colors):
        if year not in deltas_all:
            continue

        delta_df = deltas_all[year]
        ages_smooth, delta_smooth = smooth_delta(delta_df, s=0.05)

        # Сглаженная кривая
        ax1.plot(ages_smooth, delta_smooth, '-',
                linewidth=2.5, color=color, label=f'{year}', alpha=0.8)

        # Сырые точки (полупрозрачные)
        ax1.scatter(delta_df['age'], delta_df['delta'],
                   s=30, color=color, alpha=0.3, zorder=1)

    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Глобальная граница f(age)')
    ax1.set_xlabel('Возраст (лет)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Δ̂(age) = median(Z)', fontsize=11, fontweight='bold')
    ax1.set_title(
        f'{target_race}\nВСЕ УЧАСТНИКИ',
        fontsize=12, fontweight='bold'
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=8, ncol=2)

    # --- График 2: Опытные участники ---

    for year, color in zip(years, colors):
        if year not in deltas_experienced:
            continue

        delta_df = deltas_experienced[year]
        ages_smooth, delta_smooth = smooth_delta(delta_df, s=0.05)

        # Сглаженная кривая
        ax2.plot(ages_smooth, delta_smooth, '-',
                linewidth=2.5, color=color, label=f'{year}', alpha=0.8)

        # Сырые точки
        ax2.scatter(delta_df['age'], delta_df['delta'],
                   s=30, color=color, alpha=0.3, zorder=1)

    ax2.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Глобальная граница f(age)')
    ax2.set_xlabel('Возраст (лет)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Δ̂(age) = median(Z)', fontsize=11, fontweight='bold')
    ax2.set_title(
        f'{target_race}\nОПЫТНЫЕ БЕГУНЫ\n(≥2 трассы за всё время)',
        fontsize=12, fontweight='bold'
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=8, ncol=2)

    # --- График 3: Сезонные linking ---

    for year, color in zip(years, colors):
        if year not in deltas_season_linking:
            continue

        delta_df = deltas_season_linking[year]
        ages_smooth, delta_smooth = smooth_delta(delta_df, s=0.05)

        # Сглаженная кривая
        ax3.plot(ages_smooth, delta_smooth, '-',
                linewidth=2.5, color=color, label=f'{year}', alpha=0.8)

        # Сырые точки
        ax3.scatter(delta_df['age'], delta_df['delta'],
                   s=30, color=color, alpha=0.3, zorder=1)

    ax3.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
               label='Глобальная граница f(age)')
    ax3.set_xlabel('Возраст (лет)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Δ̂(age) = median(Z)', fontsize=11, fontweight='bold')
    ax3.set_title(
        f'{target_race}\nКАЛИБРОВОЧНЫЕ БЕГУНЫ\n(≥2 трассы в один сезон)',
        fontsize=12, fontweight='bold'
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best', fontsize=8, ncol=2)

    # Общий заголовок
    fig.suptitle(
        'Временная стабильность трассовой поправки Δ_c(age)\n'
        'Если кривые расходятся → поправка впитала год/состав → идея НЕ переносима',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ============================================================
    # 7. Сохранение
    # ============================================================

    output_path = Path("outputs/temporal_stability.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ График сохранен: {output_path}")

    # ============================================================
    # 8. Статистика: разброс кривых
    # ============================================================

    print("\n" + "=" * 80)
    print("СТАТИСТИКА: Разброс трассовых поправок между годами")
    print("=" * 80)

    # Общая сетка возрастов для интерполяции
    age_grid = np.linspace(age_min + 5, age_max - 5, 50)

    # --- ВСЕ УЧАСТНИКИ ---
    if len(deltas_all) >= 2:
        delta_matrix = []
        for year in years:
            if year not in deltas_all:
                continue
            delta_df = deltas_all[year]
            ages_smooth, delta_smooth = smooth_delta(delta_df, s=0.05)
            delta_interp = np.interp(age_grid, ages_smooth, delta_smooth)
            delta_matrix.append(delta_interp)

        delta_matrix = np.array(delta_matrix)
        spread = delta_matrix.max(axis=0) - delta_matrix.min(axis=0)
        mean_spread = spread.mean()
        max_spread = spread.max()

        print(f"\n1. ВСЕ УЧАСТНИКИ:")
        print(f"   Средний размах Δ(age): {mean_spread:.3f}")
        print(f"   Максимальный размах: {max_spread:.3f}")
        print(f"   В натуральной шкале: {100*(np.exp(mean_spread)-1):.1f}% - {100*(np.exp(max_spread)-1):.1f}%")

        if mean_spread > 0.05:
            print(f"   ✗ Нестабильно (порог 0.05)")
        else:
            print(f"   ✓ Относительно стабильно")

    # --- ОПЫТНЫЕ УЧАСТНИКИ ---
    if len(deltas_experienced) >= 2:
        delta_matrix_exp = []
        for year in years:
            if year not in deltas_experienced:
                continue
            delta_df = deltas_experienced[year]
            ages_smooth, delta_smooth = smooth_delta(delta_df, s=0.05)
            delta_interp = np.interp(age_grid, ages_smooth, delta_smooth)
            delta_matrix_exp.append(delta_interp)

        delta_matrix_exp = np.array(delta_matrix_exp)
        spread_exp = delta_matrix_exp.max(axis=0) - delta_matrix_exp.min(axis=0)
        mean_spread_exp = spread_exp.mean()
        max_spread_exp = spread_exp.max()

        print(f"\n2. ОПЫТНЫЕ БЕГУНЫ (≥2 трассы за всё время):")
        print(f"   Средний размах Δ(age): {mean_spread_exp:.3f}")
        print(f"   Максимальный размах: {max_spread_exp:.3f}")
        print(f"   В натуральной шкале: {100*(np.exp(mean_spread_exp)-1):.1f}% - {100*(np.exp(max_spread_exp)-1):.1f}%")

        if mean_spread_exp > 0.05:
            print(f"   ✗ Нестабильно даже на опытных")
        else:
            print(f"   ✓ Стабильнее чем на всех")

    # --- СЕЗОННЫЕ LINKING (КАЛИБРОВОЧНЫЕ) ---
    if len(deltas_season_linking) >= 2:
        delta_matrix_link = []
        for year in years:
            if year not in deltas_season_linking:
                continue
            delta_df = deltas_season_linking[year]
            ages_smooth, delta_smooth = smooth_delta(delta_df, s=0.05)
            delta_interp = np.interp(age_grid, ages_smooth, delta_smooth)
            delta_matrix_link.append(delta_interp)

        delta_matrix_link = np.array(delta_matrix_link)
        spread_link = delta_matrix_link.max(axis=0) - delta_matrix_link.min(axis=0)
        mean_spread_link = spread_link.mean()
        max_spread_link = spread_link.max()

        print(f"\n3. КАЛИБРОВОЧНЫЕ БЕГУНЫ (≥2 трассы в один сезон):")
        print(f"   Средний размах Δ(age): {mean_spread_link:.3f}")
        print(f"   Максимальный размах: {max_spread_link:.3f}")
        print(f"   В натуральной шкале: {100*(np.exp(mean_spread_link)-1):.1f}% - {100*(np.exp(max_spread_link)-1):.1f}%")

        if mean_spread_link > 0.05:
            print(f"   ✗ КРИТИЧНО: Даже калибровочные бегуны нестабильны!")
            print(f"      Это опровергает идею Максима напрямую")
        else:
            print(f"   ✓ Калибровочные бегуны дают стабильную поправку")
            print(f"      Идея Максима работает на этом тесте")
    else:
        print(f"\n3. КАЛИБРОВОЧНЫЕ БЕГУНЫ:")
        print(f"   ⚠ Недостаточно лет с данными (< 2)")


    print("=" * 80)


if __name__ == "__main__":
    create_temporal_stability_plot()