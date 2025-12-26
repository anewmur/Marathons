"""
График 1: Глобальная граница возможностей (идея Максима)

Показывает:
- Все наблюдения на плоскости (возраст, ln(время))
- Нижняя огибающая = "граница человеческих возможностей"
- Метод: квантильная регрессия по 5-му перцентилю
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


def create_global_boundary_plot() -> None:
    """
    Создает график глобальной границы возможностей.
    """

    print("=" * 80)
    print("ГРАФИК 1: Глобальная граница возможностей")
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

    # Фильтруем только мужчин для начала (для наглядности)
    df_m = df[df['gender'] == 'M'].copy()

    print(f"✓ Мужчин: {len(df_m)}")

    # Добавляем ln(время)
    df_m['ln_time'] = np.log(df_m['time_seconds'])

    # ============================================================
    # 2. Строим нижнюю огибающую (квантильная регрессия)
    # ============================================================

    print("\nСтроим нижнюю огибающую...")

    # Группируем по возрасту и находим 1-й перцентиль (ближе к нижней границе)
    age_min, age_max = 18, 70
    ages = np.arange(age_min, age_max + 1)

    boundary_times = []

    for age in ages:
        df_age = df_m[(df_m['age'] >= age - 0.5) & (df_m['age'] < age + 0.5)]  # Узкое окно

        if len(df_age) >= 10:  # Меньше порог
            # Берём МИНИМУМ вместо перцентиля
            min_time = df_age['ln_time'].min()
            boundary_times.append(min_time)
        else:
            boundary_times.append(np.nan)

    # Сглаживаем сплайном (убираем NaN)
    ages_valid = ages[~np.isnan(boundary_times)]
    boundary_valid = np.array([b for b in boundary_times if not np.isnan(b)])

    # Сглаживаем сплайном с небольшим параметром
    spline = UnivariateSpline(ages_valid, boundary_valid, s=0.1)  # s=0.1 = лёгкое сглаживание
    ages_smooth = np.linspace(ages_valid.min(), ages_valid.max(), 200)
    boundary_smooth = spline(ages_smooth)

    print(f"✓ Огибающая построена для возрастов {ages_valid.min()}-{ages_valid.max()}")

    # ============================================================
    # 3. Визуализация
    # ============================================================

    fig, ax = plt.subplots(figsize=(14, 9))

    # Сэмплируем данные для отображения (иначе слишком много точек)
    df_sample = df_m.sample(n=min(10000, len(df_m)), random_state=42)

    # Создаём цветовую карту для трасс
    unique_races = df_m['race_id'].unique()
    n_races = len(unique_races)

    # Генерируем цвета
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_races, 10)))  # До 10 цветов
    race_color_map = dict(zip(unique_races, colors))

    # Присваиваем цвет каждой точке
    point_colors = df_sample['race_id'].map(race_color_map)

    # Scatter plot всех наблюдений с цветами по трассам
    scatter = ax.scatter(
        df_sample['age'],
        df_sample['ln_time'],
        s=3,              # Размер точек (можешь изменить: 1, 2, 5, 10...)
        alpha=0.6,        # Прозрачность
        c=point_colors,   # Цвета по трассам
        rasterized=True,  # Для быстрого рендеринга
    )

    # Нижняя огибающая (жирная красная линия)
    ax.plot(
        ages_smooth,
        boundary_smooth,
        'r-',
        linewidth=3,
        label='Нижняя граница (минимум в возрасте)',
        zorder=10
    )

    # Подсветка огибающей
    ax.plot(
        ages_smooth,
        boundary_smooth,
        'yellow',
        linewidth=6,
        alpha=0.3,
        zorder=9
    )

    # Оформление
    ax.set_xlabel('Возраст (лет)', fontsize=13, fontweight='bold')
    ax.set_ylabel('ln(Время финиша)', fontsize=13, fontweight='bold')
    ax.set_title(
        'Идея: Глобальная граница возможностей\n'
        'Все наблюдения + нижняя огибающая как эталон',
        fontsize=15, fontweight='bold', pad=15
    )

    ax.grid(True, alpha=0.3)

    # Легенда для линии границы
    ax.legend(loc='upper left', fontsize=11)

    # Добавляем легенду для трасс (справа от графика)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=race_color_map[race], label=race, alpha=0.8)
        for race in sorted(unique_races)
    ]

    # Вторая легенда для трасс
    legend2 = ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.15, 0.5),  # Справа от графика
        fontsize=9,
        title='Трассы',
        title_fontsize=10
    )

    # Добавляем второй Y-axis справа с минутами
    ax2 = ax.twinx()

    # Преобразуем ln шкалу в минуты
    def ln_to_min(ln_val):
        return np.exp(ln_val) / 60

    def min_to_ln(min_val):
        return np.log(min_val * 60)

    ax2.set_ylabel('Время финиша (минуты)', fontsize=13, fontweight='bold')
    ax2.set_ylim(ax.get_ylim())

    # Устанавливаем ticks на правой оси
    ln_ticks = ax.get_yticks()
    min_ticks = [ln_to_min(t) for t in ln_ticks]
    ax2.set_yticks(ln_ticks)
    ax2.set_yticklabels([f'{m:.0f}' for m in min_ticks])

    # Текстовый блок с интерпретацией
    textstr = (
         'ИДЕЯ\n'
    'f(age) — глобальная граница\n'
    'элитных результатов.\n'
    'Трасса проявляется как сдвиг\n'
    'относительно этой границы.'
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom',
           horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # ============================================================
    # 4. Сохранение
    # ============================================================

    output_path = Path("outputs/1_global_boundary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ График сохранен: {output_path}")

    # ============================================================
    # 5. Сохраняем границу для следующих графиков
    # ============================================================

    boundary_data = pd.DataFrame({
        'age': ages_smooth,
        'ln_time_boundary': boundary_smooth,
    })

    boundary_path = Path("outputs/boundary_function.csv")
    boundary_data.to_csv(boundary_path, index=False)

    print(f"✓ Граница сохранена: {boundary_path}")

    print("\n" + "=" * 80)
    print("СТАТИСТИКА:")
    print("=" * 80)
    print(f"Возрастной диапазон: {ages_valid.min()}-{ages_valid.max()} лет")
    print(f"Граница при возрасте 30: {np.exp(spline(30))/60:.1f} мин")
    print(f"Граница при возрасте 40: {np.exp(spline(40))/60:.1f} мин")
    print(f"Граница при возрасте 50: {np.exp(spline(50))/60:.1f} мин")
    print("=" * 80)


if __name__ == "__main__":
    create_global_boundary_plot()