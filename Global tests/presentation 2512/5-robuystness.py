"""
График: Устойчивость эталонов к способу расчёта

Идея: Если эталон — это свойство трассы (а не артефакт метода),
то разумные вариации параметров (Top-5% vs Top-10%, median vs trim-mean)
должны давать близкие результаты.

Визуализация: scatter plot эталона при базовых настройках vs альтернативных.
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


def compute_reference_variants(df: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    """
    Вычисляет эталоны для разных вариантов параметров.

    Варианты:
    1. Baseline: Top-5%, median
    2. Alt1: Top-10%, median
    3. Alt2: Top-5%, trimmed mean (обрезаем 10% с каждой стороны)
    4. Alt3: Top-3%, median (более элитный)
    """

    def compute_single_reference(times: np.ndarray,
                                 percentile: float,
                                 method: str = 'median') -> float:
        """Вычисляет эталон для одного варианта"""
        if len(times) < min_n:
            return np.nan

        threshold = np.percentile(times, percentile)
        top_times = times[times <= threshold]

        if len(top_times) < 3:
            return np.nan

        if method == 'median':
            return np.median(top_times)
        elif method == 'trim_mean':
            # Trimmed mean: обрезаем 10% с каждой стороны
            sorted_times = np.sort(top_times)
            n_trim = max(1, int(len(sorted_times) * 0.1))
            trimmed = sorted_times[n_trim:-n_trim] if len(sorted_times) > 2 * n_trim else sorted_times
            return np.mean(trimmed)
        else:
            return np.median(top_times)

    results = []

    for race_id in df['race_id'].unique():
        for gender in ['M', 'F']:
            df_subset = df[(df['race_id'] == race_id) & (df['gender'] == gender)]

            if len(df_subset) < min_n:
                continue

            times = df_subset['time_seconds'].values

            # Базовый вариант: Top-5%, median
            ref_baseline = compute_single_reference(times, 5.0, 'median')

            # Альтернатива 1: Top-10%, median
            ref_top10 = compute_single_reference(times, 10.0, 'median')

            # Альтернатива 2: Top-5%, trimmed mean
            ref_trim = compute_single_reference(times, 5.0, 'trim_mean')

            # Альтернатива 3: Top-3%, median (более элитный)
            ref_top3 = compute_single_reference(times, 3.0, 'median')

            if not np.isnan(ref_baseline):
                results.append({
                    'race_id': race_id,
                    'gender': gender,
                    'n': len(df_subset),
                    'baseline': ref_baseline,  # Top-5% median
                    'top10': ref_top10,  # Top-10% median
                    'trim': ref_trim,  # Top-5% trim-mean
                    'top3': ref_top3,  # Top-3% median
                })

    return pd.DataFrame(results)


def create_robustness_plot() -> pd.DataFrame:
    """
    Создает график устойчивости эталонов к способу расчёта.
    """

    print("=" * 80)
    print("СОЗДАНИЕ ГРАФИКА: Устойчивость эталонов к параметрам")
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
    # 2. Вычисляем эталоны для разных вариантов
    # ============================================================

    print("\nВычисляем эталоны для разных параметров...")
    df_refs = compute_reference_variants(df, min_n=20)

    print(f"✓ Всего комбинаций (трасса × пол): {len(df_refs)}")

    # Переводим в минуты
    for col in ['baseline', 'top10', 'trim', 'top3']:
        df_refs[f'{col}_min'] = df_refs[col] / 60

    # ============================================================
    # 3. Создаем 3 subplot для сравнений
    # ============================================================

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    variants = [
        ('top10_min', 'Top-10% vs Top-5%', 'baseline_min'),
        ('trim_min', 'Trimmed Mean vs Median', 'baseline_min'),
        ('top3_min', 'Top-3% vs Top-5%', 'baseline_min'),
    ]

    for idx, (alt_col, title, base_col) in enumerate(variants):
        ax = axes[idx]

        # Фильтруем данные где оба варианта доступны
        mask = df_refs[alt_col].notna() & df_refs[base_col].notna()
        df_plot = df_refs[mask].copy()

        if len(df_plot) == 0:
            ax.text(0.5, 0.5, 'Нет данных', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            continue

        # Scatter plot
        colors = ['#3498db' if g == 'M' else '#e74c3c' for g in df_plot['gender']]

        scatter = ax.scatter(
            df_plot[base_col],
            df_plot[alt_col],
            c=colors,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=1,
        )

        # Диагональ (идеальное совпадение)
        min_val = min(df_plot[base_col].min(), df_plot[alt_col].min())
        max_val = max(df_plot[base_col].max(), df_plot[alt_col].max())
        ax.plot([min_val, max_val], [min_val, max_val],
                'k--', linewidth=2, alpha=0.5, label='y = x (идеал)')

        # Статистика
        corr = stats.pearsonr(df_plot[base_col], df_plot[alt_col])
        r_value = corr[0]

        # MAE (Mean Absolute Error) в процентах
        mae_abs = np.mean(np.abs(df_plot[alt_col] - df_plot[base_col]))
        mae_pct = 100 * mae_abs / df_plot[base_col].mean()

        # Оформление
        ax.set_xlabel('Baseline: Top-5% median [мин]', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{title.split(" vs ")[0]} [мин]', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

        # Текстовый блок со статистикой
        textstr = (
            f'Корреляция: r = {r_value:.3f}\n'
            f'Разница (MAE): {mae_abs:.1f} мин\n'
            f'Разница (%): {mae_pct:.1f}%'
        )

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                horizontalalignment='right', bbox=props)

        # Легенда для полов
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Мужчины'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Женщины')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.suptitle(
        'Устойчивость эталонов к способу расчёта\n'
        'Точки близко к диагонали → метод робастен к параметрам',
        fontsize=15, fontweight='bold', y=1.02
    )

    plt.tight_layout()

    # ============================================================
    # 4. Сохранение
    # ============================================================

    output_path = Path("outputs/slide_robustness.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✓ График сохранен: {output_path}")

    # ============================================================
    # 5. Статистика для слайда
    # ============================================================

    print("\n" + "=" * 80)
    print("СТАТИСТИКА ДЛЯ СЛАЙДА:")
    print("=" * 80)

    for alt_col, title, base_col in variants:
        mask = df_refs[alt_col].notna() & df_refs[base_col].notna()
        df_plot = df_refs[mask]

        if len(df_plot) == 0:
            continue

        corr = stats.pearsonr(df_plot[base_col], df_plot[alt_col])
        mae_abs = np.mean(np.abs(df_plot[alt_col] - df_plot[base_col]))
        mae_pct = 100 * mae_abs / df_plot[base_col].mean()

        print(f"\n{title}:")
        print(f"  Корреляция: r = {corr[0]:.3f}")
        print(f"  MAE: {mae_abs:.2f} мин ({mae_pct:.1f}%)")

    print("=" * 80)

    return df_refs


if __name__ == "__main__":
    df_results = create_robustness_plot()