"""
Модуль для хранения и применения возрастных сплайновых моделей.

Содержит:
- AgeSplineModel: dataclass с параметрами обученной модели для одного пола
- normalize_age: нормировка возраста по проектной формуле x = (age - age_center) / age_scale
- save_age_spline_models: сохранение моделей в .pkl
- load_age_spline_models: загрузка моделей из .pkl
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
import logging

from scipy.interpolate import BSpline

logger = logging.getLogger(__name__)

import math
from typing import Any


@dataclass
class AgeSplineModel:
    """
    Модель возрастного сплайна для одного пола.
    
    Содержит все параметры для предсказания m_g(a) = μ + γ*x + h(x)
    без доступа к обучающим данным.
    
    Attributes:
        gender: Пол ("M" или "F")
        
        # Параметры нормировки (из preprocessing в config)
        age_center: Центр возраста (единственный источник: config.preprocessing.age_center)
        age_scale: Масштаб возраста (из config.preprocessing.age_scale)
        
        # Опорная точка для центрирования (всегда 0.0)
        x0: Опорная точка x0 = (age_center - age_center) / age_scale = 0.0
        
        # Глобальные границы возраста (для справки, использовались при обучении)
        age_min_global: Минимальный возраст для clamp при обучении
        age_max_global: Максимальный возраст для clamp при обучении
        
        # Фактический диапазон данных пола (для справки и логов)
        age_range_actual: Кортеж (a_min_g, a_max_g) фактических границ для этого пола
        
        # B-сплайн параметры
        degree: Степень сплайна (обычно 3 - кубический)
        knots_x: Полный список узлов в шкале x (НЕ в шкале age!)
        
        # Центрирование базиса
        basis_centering: Словарь с ключами:
            - "centering_method": "constraints_value_slope_at_x0"
            - "x0": 0.0
            - "A": np.ndarray формы (2, K_raw)
            - "C": np.ndarray формы (K_raw, K_cent)
            - "K_raw": int
            - "K_cent": int
        
        # Оцененные коэффициенты
        coef_mu: Интерсепт μ_g
        coef_gamma: Коэффициент линейного тренда γ_g
        coef_beta: pd.Series с коэффициентами β_g центрированного сплайна
                   Индекс = имена базисных функций (например "spline_0", "spline_1", ...)
        
        # Параметр гладкости и дисперсии
        lambda_value: Выбранный параметр гладкости λ
        sigma2_reml: Остаточная дисперсия по REML (или GCV)
        tau2_bar: Средняя дисперсия эталона для этого пола
        sigma2_use: Используемая дисперсия = max(floor, sigma2_reml - tau2_bar)
        nu: Степени свободы для t-распределения
        
        # Winsorization параметры
        winsor_params: Словарь с ключами:
            - "enabled": bool
            - "k": float
            - "median": float
            - "mad": float
            - "lower": float
            - "upper": float
            - "fraction_clamped": float
        
        # Отчет о подгонке
        fit_report: Словарь с диагностической информацией:
            - "n": int (число наблюдений)
            - "age_range_actual": tuple
            - "age_range_years": float
            - "knots_count_inner": int
            - "degree": int
            - "K_raw": int
            - "K_cent": int
            - "lambda_value": float
            - "lambda_method": str ("REML" или "GCV")
            - "edf": float (effective degrees of freedom)
            - "nu": float
            - "sigma2_reml": float
            - "tau2_bar": float
            - "sigma2_use": float
            - "rmse_z": float
            - "mae_z": float
            - "degrade": str ("none", "linear_only", "constant_only")
            - "winsor": dict
            - "timings": dict
            - "warnings": list[str]
    """
    
    gender: str
    
    # Нормировка
    age_center: float
    age_scale: float
    x0: float = 0.0
    
    # Границы
    age_min_global: float = 18.0
    age_max_global: float = 80.0
    age_range_actual: tuple[float, float] = (0.0, 0.0)
    
    # Сплайн
    degree: int = 3
    knots_x: list[float] = field(default_factory=list)
    basis_centering: dict[str, Any] = field(default_factory=dict)
    
    # Коэффициенты
    coef_mu: float = 0.0
    coef_gamma: float = 0.0
    coef_beta: pd.Series = field(default_factory=lambda: pd.Series([], dtype=float))
    
    # Дисперсии
    lambda_value: float = 0.0
    sigma2_reml: float = 0.0
    tau2_bar: float = 0.0
    sigma2_use: float = 0.0
    nu: float = 3.0
    
    # Winsor
    winsor_params: dict[str, Any] = field(default_factory=dict)
    
    # Отчет
    fit_report: dict[str, Any] = field(default_factory=dict)

    def predict_h(self, age: float | np.ndarray) -> float | np.ndarray:
        """
        Вычисляет сплайновую часть h(x) = B_raw(x) @ beta.

        Args:
            age: Возраст в годах (скаляр или массив).
                 Clamp к [age_min_global, age_max_global] применяется автоматически.

        Returns:
            h: Значение сплайновой поправки на шкале Z.
               Скаляр если вход скаляр, ndarray если вход массив.

        Контракт:
            - h(age_center) ≈ 0 (по центрированию h(0)=0)
            - h'(age_center) ≈ 0 (по центрированию h'(0)=0)
        """
        # Проверка что модель обучена
        if len(self.knots_x) == 0:
            raise RuntimeError("predict_h: model not fitted (knots_x is empty)")
        if self.coef_beta.empty:
            raise RuntimeError("predict_h: model not fitted (coef_beta is empty)")

        # Определяем был ли вход скаляром
        is_scalar = np.isscalar(age)

        # Конвертируем в массив для единообразной обработки
        age_array = np.atleast_1d(np.asarray(age, dtype=float))

        if not np.isfinite(age_array).all():
            raise ValueError("predict_h: age contains non-finite values")

        # Clamp возраста к глобальным границам
        age_clamped = np.clip(
            age_array,
            a_min=float(self.age_min_global),
            a_max=float(self.age_max_global),
        )

        # Нормировка: x = (age - age_center) / age_scale
        x_std = (age_clamped - float(self.age_center)) / float(self.age_scale)

        # Строим базис B_raw
        knots_array = np.asarray(self.knots_x, dtype=float)
        degree = int(self.degree)

        # Clamp x_std к диапазону узлов (на случай экстраполяции)
        x_left = float(knots_array[0])
        x_right = float(knots_array[-1])
        x_std_clamped = np.clip(x_std, a_min=x_left, a_max=x_right)

        # Используем BSpline.design_matrix для эффективности
        dm_sparse = BSpline.design_matrix(
            x_std_clamped, knots_array, degree, extrapolate=False
        )
        b_raw = dm_sparse.toarray()

        # Получаем коэффициенты как numpy array
        beta = self.coef_beta.to_numpy(dtype=float)

        # h = B_raw @ beta
        h_values = b_raw @ beta

        # Возвращаем скаляр если вход был скаляром
        if is_scalar:
            return float(h_values[0])
        return h_values

    def predict_mean(self, age: float | np.ndarray) -> float | np.ndarray:
        """
        Вычисляет m_g(a) = μ + γ*x + h(x).

        Args:
            age: Возраст в годах (скаляр или массив).
                 Clamp к [age_min_global, age_max_global] применяется автоматически.

        Returns:
            m: Предсказанное значение на нормированной шкале Z.
               Скаляр если вход скаляр, ndarray если вход массив.

        Примечание:
            Сейчас coef_mu=0 и coef_gamma=0 (не реализованы),
            поэтому predict_mean эквивалентен predict_h.
            Когда добавим линейную часть, формула станет полной.
        """
        # Определяем был ли вход скаляром
        is_scalar = np.isscalar(age)

        # Конвертируем в массив
        age_array = np.atleast_1d(np.asarray(age, dtype=float))

        if not np.isfinite(age_array).all():
            raise ValueError("predict_mean: age contains non-finite values")

        # Clamp возраста
        age_clamped = np.clip(
            age_array,
            a_min=float(self.age_min_global),
            a_max=float(self.age_max_global),
        )

        # x = (age - age_center) / age_scale
        x_std = (age_clamped - float(self.age_center)) / float(self.age_scale)

        # Сплайновая часть
        h_values = self.predict_h(age_array)
        if is_scalar:
            h_values = np.atleast_1d(h_values)

        # Полная модель: m = μ + γ*x + h(x)
        mu = float(self.coef_mu)
        gamma = float(self.coef_gamma)

        m_values = mu + gamma * x_std + h_values

        # Возвращаем скаляр если вход был скаляром
        if is_scalar:
            return float(m_values[0])
        return m_values

    def design_row(self, age: float) -> dict[str, float]:
        """
        Возвращает столбцы центрированной марицы  для заданного возраста.

        Args:
            age: Возраст в годах

        Returns:
            Словарь с ключами:
            - "intercept": 1.0
            - "x": нормированный возраст
            - "spline_0", "spline_1", ...: значения базисных функций B_raw

        Примечание:
            Используется для диагностики и тестирования.
            Ключи spline_* соответствуют индексу coef_beta.
        """
        if len(self.knots_x) == 0:
            raise RuntimeError("design_row: model not fitted (knots_x is empty)")

        # Clamp и нормировка
        age_clamped = float(np.clip(
            age,
            a_min=float(self.age_min_global),
            a_max=float(self.age_max_global),
        ))
        x_std = (age_clamped - float(self.age_center)) / float(self.age_scale)

        # Базис
        knots_array = np.asarray(self.knots_x, dtype=float)
        x_left = float(knots_array[0])
        x_right = float(knots_array[-1])
        x_std_clamped = float(np.clip(x_std, a_min=x_left, a_max=x_right))

        dm_sparse = BSpline.design_matrix(
            np.array([x_std_clamped]), knots_array, int(self.degree), extrapolate=False
        )
        b_row = dm_sparse.toarray()[0]

        result: dict[str, float] = {
            "intercept": 1.0,
            "x": x_std,
        }

        for j, val in enumerate(b_row):
            result[f"spline_{j}"] = float(val)

        return result


def save_age_spline_models(
        models: dict[str, AgeSplineModel],
        output_path: str | Path,
        metadata: dict | None = None
) -> Path:
    """
    Сохранение словаря моделей в .pkl файл.

    Args:
        models: Словарь {"M": model_m, "F": model_f}
        output_path: Путь к .pkl файлу
        metadata: Дополнительные данные (дата обучения, версия кода, и т.д.)

    Returns:
        Path к созданному файлу

    Raises:
        NotImplementedError: Пока не реализовано (заглушка для Этапа 0)

    Примечание:
        Сохраняет:
        - models (dict с AgeSplineModel)
        - metadata (дата, версия, validation_year, etc.)
    """
    raise NotImplementedError("save_age_spline_models будет реализован в Этапе 12")


def load_age_spline_models(input_path: str | Path) -> dict[str, AgeSplineModel]:
    """
    Загрузка моделей из .pkl файла.

    Args:
        input_path: Путь к .pkl файлу

    Returns:
        Словарь {"M": model_m, "F": model_f}

    Raises:
        NotImplementedError: Пока не реализовано (заглушка для Этапа 0)

    Примечание:
        Загружает models и выводит информацию из metadata в лог.
    """
    raise NotImplementedError("load_age_spline_models будет реализован в Этапе 12")






