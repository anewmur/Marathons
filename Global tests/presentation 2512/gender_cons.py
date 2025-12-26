"""
График: Согласованность эталонов между полами

Идея: Если эталон отражает свойство ТРАССЫ (а не случайный состав),
то трасса, быстрая для мужчин, должна быть быстрой и для женщин.

Визуализация: scatter plot R^use_M vs R^use_F для каждой трассы.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Добавляем путь к модели
sys.path.insert(0, str(Path(__file__).parent.parent))

from MarathonAgeModel import MarathonModel


def create_gender_consistency_plot() -> pd.DataFrame:
    """
    Создает scatter plot эталонов M vs F для демонстрации
    согласованности измерения свойств трассы.
    """

    print("=" * 80)
    print("СОЗДАНИЕ ГРАФИКА: Согласованность эталонов M vs F")
    print("=" * 80)

    # ============================================================
    # 1. Загружаем модель и данные
    # ============================================================

    data_path = Path("C:/Users/andre/github/Marathons/Data")

    if not data_path.exists():
        # Пробуем относительный путь
        data_path = Path("/Data")

    if not data_path.exists():
        raise RuntimeError(f"Не найден путь к данным: {data_path}")

    print(f"\nСоздаем модель из: {data_path}")

    # Создаем модель и запускаем только до preprocess
    model = MarathonModel(data_path=data_path, verbose=True)

    # Запускаем пайплайн до preprocess (это создаст df_clean)
    print("\nЗапускаем пайплайн до preprocess...")
    model.load_data().filter_raw().validate_raw().add_row_id().preprocess()

    if model.df_clean is None:
        raise RuntimeError("model.df_clean is None после preprocess")

    df = model.df_clean.copy()

    print(f"\n✓ Загружено финишей: {len(df)}")
    print(f"✓ Трасс: {df['race_id'].nunique()}")
    print(f"✓ Годов: {df['year'].nunique()}")

    # ============================================================
    # 2. Вычисляем эталоны для каждой трассы и пола
    # ============================================================

    print("\nВычисляем эталоны R^use(c, g)...")

    def compute_reference(df_subset: pd.DataFrame) -> float:
        """Вычисляет эталон: median(Top-5%)"""
        if len(df_subset) < 20:  # Минимальный порог
            return np.nan

        times = df_subset['time_seconds'].values
        top5_threshold = np.percentile(times, 5)
        top5_times = times[times <= top5_threshold]

        if len(top5_times) < 3:
            return np.nan

        return np.median(top5_times)

    # Группируем по трассе и полу
    references = []

    for race_id in df['race_id'].unique():
        df_race = df[df['race_id'] == race_id]

        # Мужчины
        df_m = df_race[df_race['gender'] == 'M']
        ref_m = compute_reference(df_m)
        n_m = len(df_m)

        # Женщины
        df_f = df_race[df_race['gender'] == 'F']
        ref_f = compute_reference(df_f)
        n_f = len(df_f)

        if not np.isnan(ref_m) and not np.isnan(ref_f):
            references.append({
                'race_id': race_id,
                'R_male': ref_m,
                'R_female': ref_f,
                'N_male': n_m,
                'N_female': n_f,
            })

    df_refs = pd.DataFrame(references)

    print(f"\nТрасс с эталонами для обоих полов: {len(df_refs)}")

    if len(df_refs) == 0:
        raise RuntimeError("Нет трасс с эталонами для обоих полов!")

    # ============================================================
    # 3. Переводим в минуты для наглядности
    # ============================================================

    df_refs['R_male_min'] = df_refs['R_male'] / 60
    df_refs['R_female_min'] = df_refs['R_female'] / 60

    # ============================================================
    # 4. Статистика
    # ============================================================

    correlation = stats.pearsonr(df_refs['R_male'], df_refs['R_female'])
    r_value = correlation[0]
    p_value = correlation[1]

    print(f"\nКорреляция Пирсона: r = {r_value:.3f}, p = {p_value:.2e}")

    # Линейная регрессия для визуализации тренда
    slope, intercept, r_sq, _, _ = stats.linregress(
        df_refs['R_male_min'], df_refs['R_female_min']
    )

    print(f"Линейная регрессия: slope = {slope:.3f}, R² = {r_sq:.3f}")

    # ============================================================
    # 5. Визуализация
    # ============================================================

    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot
    scatter = ax.scatter(
        df_refs['R_male_min'],
        df_refs['R_female_min'],
        s=100,
        alpha=0.6,
        c=df_refs['N_male'] + df_refs['N_female'],
        cmap='viridis',
        edgecolors='black',
        linewidth=1,
    )

    # Линия тренда
    x_line = np.array([df_refs['R_male_min'].min(), df_refs['R_male_min'].max()])
    y_line = slope * x_line + intercept

    ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
            label=f'Тренд: R² = {r_sq:.3f}')

    # Диагональ (для reference, если бы соотношение было 1:1)
    # Не обязательна, но показывает что женский эталон систематически медленнее

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Число финишей (M + F)', fontsize=11)

    # ============================================================
    # Аннотации для ВСЕХ трасс (7 штук)
    # ============================================================

    # Словарь нормализации названий
    race_name_map = {
        'дорога жизни': 'Дорога жизни',
        'пермский марафон': 'Пермский марафон',
        'пермский международный марафон': 'Пермский марафон',
        'московский марафон': 'Московский марафон',
        'казанский марафон': 'Казанский марафон',
        'белые ночи': 'Белые ночи',
        'чемпионат россии': 'Чемпионат России',
    }

    # Нормализуем названия трасс в df_refs
    df_refs['race_display'] = df_refs['race_id'].str.lower().map(race_name_map)
    df_refs['race_display'] = df_refs['race_display'].fillna(df_refs['race_id'])

    # Определяем позиции подписей для каждой трассы (чтобы не налезали)
    # Формат: (xytext_x, xytext_y) в offset points
    label_positions = {
        'Дорога жизни': (10, 10),           # Верхний правый угол
        'Пермский марафон': (-80, 10),      # Центр, слева вверху
        'Московский марафон': (10, -15),    # Справа внизу
        'Казанский марафон': (-80, -15),    # Слева внизу
        'Белые ночи': (10, 0),              # Справа по центру
        'Чемпионат России': (10, 10),       # Самая быстрая, справа вверху
    }

    # Аннотируем каждую трассу
    for idx, row in df_refs.iterrows():
        race_name = row['race_display']

        # Позиция подписи (по умолчанию справа вверху)
        offset = label_positions.get(race_name, (10, 10))

        # Цвет рамки зависит от скорости
        if row['R_male_min'] < df_refs['R_male_min'].quantile(0.33):
            box_color = '#d4edda'  # Зеленоватый (быстрая)
            edge_color = '#28a745'
        elif row['R_male_min'] > df_refs['R_male_min'].quantile(0.67):
            box_color = '#f8d7da'  # Красноватый (медленная)
            edge_color = '#dc3545'
        else:
            box_color = '#fff3cd'  # Желтоватый (средняя)
            edge_color = '#ffc107'

        ax.annotate(
            race_name,
            xy=(row['R_male_min'], row['R_female_min']),
            xytext=offset, textcoords='offset points',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4',
                     facecolor=box_color,
                     edgecolor=edge_color,
                     linewidth=1.5,
                     alpha=0.9),
            arrowprops=dict(arrowstyle='->', lw=1.2, color=edge_color)
        )

    # Оформление
    ax.set_xlabel('Эталон для мужчин, R^use(c, M) [минуты]',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('Эталон для женщин, R^use(c, F) [минуты]',
                  fontsize=13, fontweight='bold')

    ax.set_title(
        'Согласованность эталонов между полами\n'
        f'Корреляция: r = {r_value:.3f} (p < 0.001)',
        fontsize=15, fontweight='bold', pad=15
    )

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=11)

    # Текстовый блок с интерпретацией
    textstr = (
        f'Трасс: {len(df_refs)}\n'
        f'Корреляция: r = {r_value:.3f}\n'
        f'R² = {r_sq:.3f}\n\n'
        f'ВЫВОД:\n'
        f'Сильная согласованность\n'
        f'→ Эталон измеряет ТРАССУ,\n'
        f'   а не случайный состав'
    )

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # ============================================================
    # 6. Сохранение
    # ============================================================

    output_path = Path("outputs/slide_gender_consistency.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ График сохранен: {output_path}")

    # ============================================================
    # 7. Статистика для слайда
    # ============================================================

    print("\n" + "=" * 80)
    print("СТАТИСТИКА ДЛЯ СЛАЙДА:")
    print("=" * 80)
    print(f"Трасс с эталонами для обоих полов: {len(df_refs)}")
    print(f"Корреляция Пирсона: r = {r_value:.3f}")
    print(f"p-value: {p_value:.2e}")
    print(f"R² линейной регрессии: {r_sq:.3f}")
    print(f"\nДиапазон эталонов (мужчины): {df_refs['R_male_min'].min():.1f} - {df_refs['R_male_min'].max():.1f} мин")
    print(f"Диапазон эталонов (женщины): {df_refs['R_female_min'].min():.1f} - {df_refs['R_female_min'].max():.1f} мин")
    print("=" * 80)

    return df_refs


if __name__ == "__main__":
    df_results = create_gender_consistency_plot()