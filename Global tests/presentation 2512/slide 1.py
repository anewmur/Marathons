"""
Визуализация для слайда 1: Декомпозиция времени финиша
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Создаем фигуру
fig, ax_bar = plt.subplots(figsize=(12, 6))

# Компоненты (в минутах для примера)
components = {
    'Потенциал трассы\n(эталон)': 150,
    'Возраст участника': 45,
    'Конкретный год\nи условия': 18,
    'Индивидуальный уровень\nбегуна': 10,
    'Случайный шум': 2,
}

labels = list(components.keys())
values = list(components.values())

# Цвета
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#95a5a6']

# Рисуем горизонтальный стек
left = 0
y_pos = 1

k = len(components.items())-2
for i, (label, value) in enumerate(components.items()):
    # Основной бар
    ax_bar.barh(y_pos, value, left=left, height=0.5,
                color=colors[i], edgecolor='black', linewidth=1.5,
                alpha=0.8)

    # Текст внутри (если достаточно места)
    if value >= 10:
        ax_bar.text(left + value/2, y_pos, f'{value}\nмин',
                    ha='center', va='center', fontsize=10,
                    fontweight='bold', color='white')

    # Метка компоненты сверху
    if i < len(components.items())-2:
        ax_bar.text(left + value/2, y_pos + 0.35, label,
                    ha='center', va='bottom', fontsize=10,
                        fontweight='bold', color=colors[i])
    else:
        ax_bar.text(left + value / 2, y_pos + 0.35 + (i-k+1)*0.25, label,
                    ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color=colors[i])

    left += value

# Итоговое время
total_time = sum(values)
ax_bar.text(total_time/2, y_pos - 0.5,
            f'Итого: {total_time} мин (3:45:00)',
            ha='center', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.7))

# Стрелка "сложение"
arrow_y = y_pos + 0.8
for i in range(len(values) - 1):
    x_arrow = sum(values[:i+1])
    ax_bar.annotate('', xy=(x_arrow + 5, arrow_y), xytext=(x_arrow - 5, arrow_y),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
    ax_bar.text(x_arrow, arrow_y + 0.1, '+', ha='center', va='bottom',
                fontsize=18, fontweight='bold')

# Настройки осей
ax_bar.set_xlim(0, total_time + 10)
ax_bar.set_ylim(0, 2.5)
ax_bar.set_xlabel('Время (минуты)', fontsize=18, fontweight='bold')
ax_bar.set_title('Иллюстрация структуры времени финиша',
                 fontsize=24, fontweight='bold',  pad=40)
ax_bar.set_yticks([])
ax_bar.spines['left'].set_visible(False)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig('outputs/slide_1_decomposition.png', dpi=400, bbox_inches='tight')
print("✓ График сохранен: /mnt/user-data/outputs/slide_1_decomposition.png")
plt.close()