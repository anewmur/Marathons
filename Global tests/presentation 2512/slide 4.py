"""
Слайд 4: Как сейчас считается эталон трассы
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_current_reference_approach() -> None:
    """
    Визуализация текущего подхода к эталону:
    - Что берем: Top-5% быстрых результатов
    - Как агрегируем: median/trim для устойчивости
    - Почему это работает
    """
    fig = plt.figure(figsize=(14, 9))

    # Создаем сетку для компоновки
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.5, 1],
                          width_ratios=[1, 1], hspace=0.35, wspace=0.3)

    # ============================================================
    # ВЕРХ: Схема "что берем"
    # ============================================================
    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 2)
    ax_top.axis('off')

    ax_top.text(5, 1.8, 'КАК СЕЙЧАС СЧИТАЕТСЯ ЭТАЛОН ТРАССЫ',
                ha='center', va='top', fontsize=16, fontweight='bold')

    # Визуализация Top-5%
    # Гистограмма всех результатов
    np.random.seed(42)
    all_times = np.random.lognormal(5.3, 0.25, 1000)
    top5_threshold = np.percentile(all_times, 5)

    bins = np.linspace(all_times.min(), all_times.max(), 40)
    hist, edges = np.histogram(all_times, bins=bins)

    # Рисуем распределение
    x_offset = 1.5
    x_scale = 0.008
    y_scale = 0.15

    for i in range(len(hist)):
        if edges[i] <= top5_threshold:
            color = '#3498db'
            alpha = 0.9
            linewidth = 1.5
        else:
            color = '#bdc3c7'
            alpha = 0.4
            linewidth = 0.5

        width = (edges[i + 1] - edges[i]) * x_scale
        height = hist[i] / hist.max() * y_scale

        rect = mpatches.Rectangle(
            (x_offset + edges[i] * x_scale, 0.15),
            width, height,
            facecolor=color, edgecolor='black',
            alpha=alpha, linewidth=linewidth
        )
        ax_top.add_patch(rect)

    # Стрелка на Top-5%
    arrow_x = x_offset + top5_threshold * x_scale
    ax_top.annotate('', xy=(arrow_x, 0.12), xytext=(arrow_x, 0.02),
                    arrowprops=dict(arrowstyle='->', lw=3, color='#3498db'))
    ax_top.text(arrow_x - 0.3, 0.0, 'Top-5%\nбыстрых', ha='center',
                fontsize=11, fontweight='bold', color='#3498db')

    # Формула справа
    formula_box = FancyBboxPatch(
        (5.5, 0.05), 4.0, 1.7,
        boxstyle="round,pad=0.12",
        edgecolor='#e67e22', facecolor='#fff3cd', linewidth=3
    )
    ax_top.add_patch(formula_box)

    ax_top.text(7.5, 1.5, 'ФОРМУЛА ЭТАЛОНА', ha='center', va='center',
                fontsize=12, fontweight='bold', color='#d35400')

    ax_top.text(7.5, 1.1, r'$R^{use}(c, g) = \mathrm{median}(\mathrm{Top5\%}_{all\_years})$',
                ha='center', va='center', fontsize=11, family='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax_top.text(7.5, 0.65, 'c = трасса, g = пол', ha='center', va='center',
                fontsize=9, style='italic', color='#7f8c8d')

    ax_top.text(7.5, 0.25, 'median = робастность к выбросам', ha='center', va='center',
                fontsize=9, color='#2c3e50')

    # ============================================================
    # СЕРЕДИНА СЛЕВА: Почему разумно
    # ============================================================
    ax_why = fig.add_subplot(gs[1, 0])
    ax_why.set_xlim(0, 5)
    ax_why.set_ylim(0, 5)
    ax_why.axis('off')

    ax_why.text(2.5, 4.8, 'ПОЧЕМУ РАЗУМНО', ha='center', va='top',
                fontsize=13, fontweight='bold', color='#2ecc71')

    reasons = [
        ('Элитный смысл', 'Эталон = потенциал трассы\nна уровне быстрых бегунов'),
        ('Устойчивость', 'median вместо mean:\nне зависит от экстремальных\nрекордов или ошибок'),
        ('Покрытие', 'Работает на любой трассе\nс достаточным N'),
        ('Интерпретируемость', 'Понятный смысл:\n"какое время показывают\nбыстрые участники"'),
    ]

    y_pos = 4.2
    for i, (title, desc) in enumerate(reasons):
        # Иконка
        circle = mpatches.Circle((0.4, y_pos), 0.25,
                                 facecolor='#2ecc71', edgecolor='black', linewidth=2)
        ax_why.add_patch(circle)
        ax_why.text(0.4, y_pos, '✓', ha='center', va='center',
                    fontsize=16, color='white', fontweight='bold')

        # Текст
        ax_why.text(0.8, y_pos + 0.1, title, ha='left', va='center',
                    fontsize=10, fontweight='bold', color='#27ae60')
        ax_why.text(0.8, y_pos - 0.25, desc, ha='left', va='top',
                    fontsize=8.5, linespacing=1.3, color='#2c3e50')

        y_pos -= 1.0

    # ============================================================
    # СЕРЕДИНА СПРАВА: Сильные стороны
    # ============================================================
    ax_strengths = fig.add_subplot(gs[1, 1])
    ax_strengths.set_xlim(0, 5)
    ax_strengths.set_ylim(0, 5)
    ax_strengths.axis('off')

    ax_strengths.text(2.5, 4.8, 'СИЛЬНЫЕ СТОРОНЫ', ha='center', va='top',
                      fontsize=13, fontweight='bold', color='#3498db')

    # Три блока
    strengths = [
        {
            'title': 'Покрытие данных',
            'metric': '100%',
            'desc': 'трасс с N>50',
            'color': '#3498db',
            'y': 3.6
        },
        {
            'title': 'Нет переобучения',
            'metric': '0',
            'desc': 'параметров на трассу',
            'color': '#9b59b6',
            'y': 2.2
        },
        {
            'title': 'Явная σ',
            'metric': '±15%',
            'desc': 'неопределенность',
            'color': '#e67e22',
            'y': 0.8
        },
    ]

    for item in strengths:
        # Блок
        box = FancyBboxPatch(
            (0.5, item['y'] - 0.5), 4.0, 1.0,
            boxstyle="round,pad=0.1",
            edgecolor=item['color'], facecolor=item['color'],
            linewidth=2, alpha=0.15
        )
        ax_strengths.add_patch(box)

        # Заголовок
        ax_strengths.text(2.5, item['y'] + 0.35, item['title'],
                          ha='center', va='center', fontsize=11,
                          fontweight='bold', color=item['color'])

        # Метрика (крупно)
        ax_strengths.text(2.5, item['y'] - 0.05, item['metric'],
                          ha='center', va='center', fontsize=18,
                          fontweight='bold', color=item['color'])

        # Описание
        ax_strengths.text(2.5, item['y'] - 0.3, item['desc'],
                          ha='center', va='center', fontsize=9,
                          color='#2c3e50')

    # ============================================================
    # НИЗ: Вывод
    # ============================================================
    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.set_xlim(0, 10)
    ax_bottom.set_ylim(0, 2)
    ax_bottom.axis('off')

    conclusion_box = FancyBboxPatch(
        (1.5, 0.3), 7.0, 1.4,
        boxstyle="round,pad=0.15",
        edgecolor='#34495e', facecolor='#ecf0f1', linewidth=3
    )
    ax_bottom.add_patch(conclusion_box)

    ax_bottom.text(5, 1.4, 'ВЫВОД', ha='center', va='center',
                   fontsize=12, fontweight='bold', color='#2c3e50')

    ax_bottom.text(5, 0.95,
                   'Эталон через Top-5% — это базовый подход с понятным смыслом',
                   ha='center', va='center', fontsize=11)

    ax_bottom.text(5, 0.55,
                   'Он работает везде, не требует калибровки, дает явную неопределенность',
                   ha='center', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()

    output_path = Path("outputs/slide_4_current_reference.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ График сохранен: {output_path}")


if __name__ == "__main__":
    create_current_reference_approach()