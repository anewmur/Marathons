"""
График 2: Распределения остатков по трассам

Показывает:
- Для выбранных показательных трасс: Z = ln(T) - f(age)
- Boxplot/violin для визуализации различий
- Проверка: различаются ли трассы устойчиво после вычитания глобальной границы?
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


def create_residuals_by_race_plot() -> None:
    """
    Создает boxplot/violin остатков для показательных трасс.
    """

    print("=" * 80)
    print("ГРАФИК 2: Распределения остатков по трассам")
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
    # 2. Загружаем границу f(age)
    # ============================================================

    boundary_path = Path("outputs/boundary_function.csv")

    if not boundary_path.exists():
        raise RuntimeError(
            f"Файл границы не найден: {boundary_path}\n"
            "Сначала запустите create_global_boundary.py"
        )

    print("\n✓ Загружаем границу f(age) из файла...")
    boundary_data = pd.read_csv(boundary_path)
    spline = UnivariateSpline(
        boundary_data['age'],
        boundary_data['ln_time_boundary'],
        s=0
    )

    age_min, age_max = boundary_data['age'].min(), boundary_data['age'].max()
    print(f"✓ Граница определена для возрастов: {age_min:.0f}-{age_max:.0f} лет")

    # ============================================================
    # 3. Фильтруем данные: только мужчины, возраст в диапазоне
    # ============================================================

    df_filtered = df[
        (df['gender'] == 'M') &
        (df['age'] >= age_min) &
        (df['age'] <= age_max)
        ].copy()

    print(f"✓ Отфильтровано наблюдений (M, возраст {age_min:.0f}-{age_max:.0f}): {len(df_filtered)}")

    # ============================================================
    # 4. Вычисляем остатки Z = ln(T) - f(age)
    # ============================================================

    print("\nВычисляем остатки Z = ln(T) - f(age)...")

    df_filtered['ln_time'] = np.log(df_filtered['time_seconds'])
    df_filtered['f_age'] = spline(df_filtered['age'])
    df_filtered['residual_z'] = df_filtered['ln_time'] - df_filtered['f_age']

    print(f"✓ Остатки вычислены для {len(df_filtered)} наблюдений")

    # ============================================================
    # 5. Выбираем 2-4 показательные трассы
    # ============================================================

    print("\nВыбираем показательные трассы...")

    # Группируем по трассам
    race_stats = []

    for race_id in df_filtered['race_id'].unique():
        df_race = df_filtered[df_filtered['race_id'] == race_id]

        if len(df_race) < 100:  # Минимальный порог
            continue

        race_stats.append({
            'race_id': race_id,
            'n': len(df_race),
            'median_z': df_race['residual_z'].median(),
            'std_z': df_race['residual_z'].std(),
            'q25': df_race['residual_z'].quantile(0.25),
            'q75': df_race['residual_z'].quantile(0.75),
        })

    df_stats = pd.DataFrame(race_stats)
    df_stats = df_stats.sort_values('median_z')

    print(f"✓ Трасс с достаточными данными (n≥100): {len(df_stats)}")

    # Выбираем показательные трассы
    selected_races = []

    # 1. Самая быстрая (элитная)
    fastest = df_stats.iloc[0]['race_id']
    selected_races.append(('Самая быстрая (элитная)', fastest))

    # 2. Самая медленная (массовая)
    slowest = df_stats.iloc[-1]['race_id']
    selected_races.append(('Самая медленная (массовая)', slowest))

    # 3-4. Средние трассы (если есть)
    if len(df_stats) >= 4:
        mid_indices = [len(df_stats) // 3, 2 * len(df_stats) // 3]
        for idx in mid_indices[:2]:  # Максимум 2 средние
            mid_race = df_stats.iloc[idx]['race_id']
            selected_races.append(('Средняя', mid_race))

    print(f"\nВыбрано трасс для анализа: {len(selected_races)}")
    for label, race_id in selected_races:
        n = df_stats[df_stats['race_id'] == race_id]['n'].values[0]
        median_z = df_stats[df_stats['race_id'] == race_id]['median_z'].values[0]
        print(f"  {label}: {race_id} (n={n}, median_z={median_z:+.3f})")

    # ============================================================
    # 6. Подготовка данных для визуализации
    # ============================================================

    data_for_plot = []
    labels_for_plot = []
    stats_for_table = []

    for label, race_id in selected_races:
        residuals = df_filtered[df_filtered['race_id'] == race_id]['residual_z'].values
        data_for_plot.append(residuals)

        # Короткая подпись
        short_label = race_id.replace('марафон', 'м.').replace('Марафон', 'М.')
        labels_for_plot.append(short_label)

        # Статистика
        stats_for_table.append({
            'race': race_id,
            'n': len(residuals),
            'median': np.median(residuals),
            'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
            'std': np.std(residuals),
        })

    # ============================================================
    # 7. Визуализация
    # ============================================================

    fig, (ax_main, ax_table) = plt.subplots(
        2, 1,
        figsize=(14, 10),
        gridspec_kw={'height_ratios': [4, 1]}
    )

    # --- Верхний график: boxplot + violin ---

    # Violin plot (распределение)
    parts = ax_main.violinplot(
        data_for_plot,
        positions=range(len(data_for_plot)),
        widths=0.7,
        showmeans=False,
        showmedians=False,
    )

    # Раскраска violin
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    # Boxplot поверх (медиана, квартили)
    bp = ax_main.boxplot(
        data_for_plot,
        positions=range(len(data_for_plot)),
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='red', linewidth=2),
        boxprops=dict(facecolor='white', alpha=0.8),
    )

    # Отмечаем медианы явно
    for i, (data, label) in enumerate(zip(data_for_plot, labels_for_plot)):
        median = np.median(data)
        ax_main.text(
            i, median, f'{median:+.3f}',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7)
        )

    # Горизонтальная линия на нуле
    ax_main.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5,
                    label='Граница f(age)')

    # Оформление
    ax_main.set_xticks(range(len(labels_for_plot)))
    ax_main.set_xticklabels(labels_for_plot, fontsize=11, fontweight='bold')
    ax_main.set_ylabel('Остаток Z = ln(T) - f(age)', fontsize=13, fontweight='bold')
    ax_main.set_title(
        'Распределения остатков по трассам\n'
        'ПОСЛЕ вычитания глобальной границы f(age)',
        fontsize=15, fontweight='bold', pad=15
    )
    ax_main.grid(True, alpha=0.3, axis='y')
    ax_main.legend(loc='upper right', fontsize=11)

    # --- Нижняя таблица: статистика ---

    ax_table.axis('off')

    # Заголовки таблицы
    table_data = [['Трасса', 'N', 'Median(Z)', 'IQR(Z)', 'Std(Z)']]

    # Данные
    for stat in stats_for_table:
        table_data.append([
            stat['race'],
            f"{stat['n']}",
            f"{stat['median']:+.3f}",
            f"{stat['iqr']:.3f}",
            f"{stat['std']:.3f}",
        ])

    # Создаём таблицу
    table = ax_table.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.15, 0.15, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Раскраска заголовков
    for i in range(5):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Раскраска строк
    for i in range(1, len(table_data)):
        for j in range(5):
            table[(i, j)].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

    plt.tight_layout()

    # ============================================================
    # 8. Сохранение
    # ============================================================

    output_path = Path("outputs/residuals_by_race.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ График сохранен: {output_path}")

    print("\n" + "=" * 80)
    print("ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 80)
    for stat in stats_for_table:
        print(f"{stat['race']:25s}: n={stat['n']:5.0f}, "
              f"median={stat['median']:+.3f}, "
              f"IQR={stat['iqr']:.3f}, "
              f"std={stat['std']:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    create_residuals_by_race_plot()