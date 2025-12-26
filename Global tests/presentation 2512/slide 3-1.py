"""
Слайд 3: Как рейтингуют в спорте — мировые примеры
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def create_world_ratings_examples() -> None:
    """
    Простая визуализация: 3 примера мировых рейтинговых систем.
    Цель: показать знакомство с практикой и выделить ключевую идею.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Заголовок
    ax.text(5, 9.3, 'Как рейтингуют в спорте: примеры мировых систем',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # ============================================================
    # Три примера систем
    # ============================================================

    systems = [
        {
            'name': 'World Athletics Rankings',
            'sport': '(легкая атлетика)',
            'formula': 'Score = Result points + Placing points + Competition level',
            'key_idea': 'Разделение: результат ≠ престиж старта',
            'color': '#3498db',
            'y': 7.5
        },
        {
            'name': 'ITRA / UTMB Index',
            'sport': '(трейл-раннинг)',
            'formula': 'Performance = Time score - Age factor - Elevation factor',
            'key_idea': 'Разделение: результат ≠ возраст ≠ трасса',
            'color': '#2ecc71',
            'y': 5.0
        },
        {
            'name': 'UCI Cycling Points',
            'sport': '(велоспорт)',
            'formula': 'Points = Position × Race category coefficient',
            'key_idea': 'Разделение: место ≠ уровень гонки',
            'color': '#e74c3c',
            'y': 2.5
        }
    ]

    for i, sys in enumerate(systems):
        # Номер
        circle = mpatches.Circle((0.5, sys['y']), 0.3,
                                facecolor=sys['color'],
                                edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.5, sys['y'], str(i+1), ha='center', va='center',
                fontsize=14, fontweight='bold', color='white')

        # Название системы
        ax.text(1.2, sys['y'] + 0.3, sys['name'], ha='left', va='center',
                fontsize=13, fontweight='bold', color=sys['color'])
        ax.text(1.2, sys['y'], sys['sport'], ha='left', va='center',
                fontsize=10, style='italic', color='#7f8c8d')

        # Формула (в рамке)
        formula_box = FancyBboxPatch(
            (1.2, sys['y'] - 0.8), 6.5, 0.6,
            boxstyle="round,pad=0.08",
            edgecolor='#95a5a6', facecolor='#ecf0f1', linewidth=1.5
        )
        ax.add_patch(formula_box)

        ax.text(4.5, sys['y'] - 0.5, sys['formula'], ha='center', va='center',
                fontsize=10, family='monospace')

        # Ключевая идея (выделено)
        key_box = FancyBboxPatch(
            (8.0, sys['y'] - 0.65), 1.8, 0.5,
            boxstyle="round,pad=0.05",
            edgecolor=sys['color'], facecolor=sys['color'],
            linewidth=2, alpha=0.2
        )
        ax.add_patch(key_box)

        ax.text(8.9, sys['y'] - 0.4, sys['key_idea'], ha='center', va='center',
                fontsize=8.5, fontweight='bold', color=sys['color'])

    # ============================================================
    # Общий паттерн (вывод)
    # ============================================================

    conclusion_box = FancyBboxPatch(
        (1.0, 0.3), 8.0, 1.2,
        boxstyle="round,pad=0.15",
        edgecolor='#34495e', facecolor='#fff3cd', linewidth=3
    )
    ax.add_patch(conclusion_box)

    ax.text(5, 1.2, 'ОБЩИЙ ПАТТЕРН', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#2c3e50')

    ax.text(5, 0.8,
            'Везде разделяют компоненты: результат ≠ контекст старта (возраст, уровень, условия)',
            ha='center', va='center', fontsize=11)

    ax.text(5, 0.45,
            'Наша задача: выделить потенциал трассы, отделив его от возраста и года',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='#c0392b')

    plt.tight_layout()

    output_path = Path("outputs/slide_3_1_world_ratings.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"✓ График сохранен: {output_path}")


if __name__ == "__main__":
    create_world_ratings_examples()