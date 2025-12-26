"""
Слайд 3: Как рейтингуют в спорте: примеры мировых систем
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def create_world_ratings_examples() -> None:
    figure, axes = plt.subplots(figsize=(14, 8))
    axes.set_xlim(0, 10)
    axes.set_ylim(0, 10)
    axes.axis("off")

    axes.text(
        5,
        9.35,
        "Как рейтингуют в спорте: примеры мировых систем",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    systems: list[dict[str, str | float]] = [
        {
            "name": "World Athletics Rankings (бег, лёгкая атлетика)",
            "period": "Период: rolling window, обычно 12 месяцев (часто 12–18)",
            "aggregation": "Агрегация: среднее нескольких лучших результатов (типично top-5), есть минимум стартов",
            "formula": "Итог = Result Score + Placing Score + уровень соревнования",
            "key_idea": "Время и престиж старта\nразведены",
            "details": "Рейтинг устойчив:\nне зависит от одного рекорда и\nпоощряет сильные старты",
            "color": "#3498db",
            "y": 7.85,
        },
        {
            "name": "ITRA / UTMB Index (трейл и ультра)",
            "period": "Период: 36 месяцев (скользящее окно)",
            "aggregation": "Агрегация: взвешенное среднее лучших результатов (часто top-5), свежие важнее старых",
            "formula": "Итог = оценка выступления, нормированная по трассе и условиям (рельеф, профиль)",
            "key_idea": "Результат приводят\nк общему масштабу трасс",
            "details": "Можно сравнивать старты\nс разным рельефом и дистанцией",
            "color": "#2ecc71",
            "y": 5.25,
        },
        {
            "name": "UCI Points (велоспорт) (шоссе, WorldTour и др.)",
            "period": "Период: сезонные и rolling-окна (зависит от зачёта), суммируют лучшие результаты",
            "aggregation": "Агрегация: очки за места, затем суммирование (спортсмен, команда, нация)",
            "formula": "Очки = место × коэффициент категории гонки (уровень и статус события)",
            "key_idea": "Место и уровень гонки\nразведены",
            "details": "Победа на старте высокого уровня\nвесит больше, чем на локальном",
            "color": "#e74c3c",
            "y": 2.65,
        },
    ]

    dashed_edge_color = "#7f8c8d"
    meta_face_color = "#ffffff"
    formula_face_color = "#ecf0f1"

    # ============================================================
    # Управляемые зазоры (все в координатах оси Y)
    # ============================================================

    # Положение названия относительно центра кружка (y_pos)
    name_y_offset = 0.80

    # Зазор между названием и верхней гранью пунктирного блока (meta_box)
    gap_name_to_meta = 0.28

    # Высота пунктирного блока
    meta_height = 0.82

    # Зазор между пунктирным блоком и блоком "формула"
    gap_meta_to_formula = 0.14

    # Высота блока "формула"
    formula_height = 0.62

    # Высота правого блока (ключевая идея + пояснение)
    key_height = 1.85

    # Внутренние отступы текста внутри блоков
    meta_text_top_pad = 0.27
    meta_text_second_line_gap = 0.30

    for system_index, system_item in enumerate(systems):
        y_pos = float(system_item["y"])
        accent_color = str(system_item["color"])

        # Номер
        circle_patch = mpatches.Circle(
            (0.6, y_pos),
            0.3,
            facecolor=accent_color,
            edgecolor="black",
            linewidth=2,
        )
        axes.add_patch(circle_patch)
        axes.text(
            0.6,
            y_pos,
            str(system_index + 1),
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

        # Название системы
        name_y = y_pos + name_y_offset
        axes.text(
            1.15,
            name_y,
            str(system_item["name"]),
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=accent_color,
        )

        # ------------------------------------------------------------
        # Геометрия блоков, привязанная к названию + управляемые зазоры
        # ------------------------------------------------------------

        meta_top_y = name_y - gap_name_to_meta
        meta_bottom_y = meta_top_y - meta_height

        formula_top_y = meta_bottom_y - gap_meta_to_formula
        formula_bottom_y = formula_top_y - formula_height

        # Пунктирный блок: период + агрегация
        meta_box = FancyBboxPatch(
            (1.15, meta_bottom_y),
            6.95,
            meta_height,
            boxstyle="round,pad=0.08",
            edgecolor=dashed_edge_color,
            facecolor=meta_face_color,
            linewidth=1.4,
            linestyle="--",
        )
        axes.add_patch(meta_box)

        axes.text(
            1.35,
            meta_top_y - meta_text_top_pad,
            str(system_item["period"]),
            ha="left",
            va="center",
            fontsize=9.2,
            color="#2c3e50",
        )
        axes.text(
            1.35,
            meta_top_y - (meta_text_top_pad + meta_text_second_line_gap),
            str(system_item["aggregation"]),
            ha="left",
            va="center",
            fontsize=9.2,
            color="#2c3e50",
        )

        # Блок "формула"
        formula_box = FancyBboxPatch(
            (1.15, formula_bottom_y),
            6.95,
            formula_height,
            boxstyle="round,pad=0.08",
            edgecolor=dashed_edge_color,
            facecolor=formula_face_color,
            linewidth=1.5,
        )
        axes.add_patch(formula_box)
        axes.text(
            4.62,
            formula_bottom_y + 0.5 * formula_height,
            str(system_item["formula"]),
            ha="center",
            va="center",
            fontsize=9.6,
            family="monospace",
            color="#2c3e50",
        )

        # Правый блок (ключевая идея + пояснение) — привязываем по низу к формуле
        key_box = FancyBboxPatch(
            (8.25, formula_bottom_y),
            1.65,
            key_height,
            boxstyle="round,pad=0.06",
            edgecolor=accent_color,
            facecolor=accent_color,
            linewidth=2,
            alpha=0.16,
        )
        axes.add_patch(key_box)

        axes.text(
            9.075,
            formula_bottom_y + key_height - 0.62,
            str(system_item["key_idea"]),
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=accent_color,
            wrap=True,
        )
        axes.text(
            9.075,
            formula_bottom_y + 0.64,
            str(system_item["details"]),
            ha="center",
            va="center",
            fontsize=8.6,
            color="#2c3e50",
            wrap=True,
        )

    # Вывод
    conclusion_box = FancyBboxPatch(
        (1.0, 0.20),
        8,
        1.00,
        boxstyle="round,pad=0.15",
        edgecolor="#34495e",
        facecolor="#fff3cd",
        linewidth=3,
    )
    axes.add_patch(conclusion_box)

    axes.text(
        5.25,
        1.08,
        "ОБЩИЙ ПАТТЕРН",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#2c3e50",
    )
    axes.text(
        5.45,
        0.72,
        "Два шага: (1) разложение на компоненты, (2) агрегация лучших результатов на скользящем окне.",
        ha="center",
        va="center",
        fontsize=10.6,
        color="#2c3e50",
    )
    axes.text(
        5.45,
        0.36,
        "Наша задача: выделить потенциал трассы как нормировку, отделив его от возраста и от сдвигов по годам.",
        ha="center",
        va="center",
        fontsize=10.9,
        fontweight="bold",
        color="#c0392b",
    )

    plt.tight_layout()

    output_path = Path("outputs/slide_3_world_ratings.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"✓ График сохранен: {output_path}")


if __name__ == "__main__":
    create_world_ratings_examples()
