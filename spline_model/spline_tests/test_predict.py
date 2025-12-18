
"""
Тесты для AgeSplineModel.predict_h и predict_mean.
"""

import numpy as np
import pandas as pd

from spline_model.age_spline_fit import AgeSplineFitter


def test_predict_h_at_age_center_is_near_zero() -> None:
    """
    Проверяет контракт центрирования: h(age_center) ≈ 0.

    По построению h(0)=0, а x=0 соответствует age=age_center.
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    # Синтетические данные с линейным трендом
    ages = np.linspace(20.0, 70.0, 500)
    z_values = 0.02 * (ages - 35.0) + np.random.default_rng(42).normal(0, 0.01, len(ages))

    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages.astype(float),
        "Z": z_values.astype(float),
    })

    model = fitter.fit_gender(gender_df=df, gender="M")

    # h(35) должен быть близок к 0
    h_at_center = model.predict_h(35.0)

    if not np.isfinite(h_at_center):
        raise RuntimeError("predict_h(35.0) is not finite")

    if abs(h_at_center) > 1e-8:
        raise RuntimeError(f"predict_h(age_center) should be ~0, got {h_at_center}")


def test_predict_h_returns_finite_across_age_range() -> None:
    """
    Проверяет что predict_h возвращает конечные значения по всему диапазону возрастов.
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    ages = np.linspace(20.0, 70.0, 400)
    z_values = 0.01 * (ages - 35.0)

    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages.astype(float),
        "Z": z_values.astype(float),
    })

    model = fitter.fit_gender(gender_df=df, gender="M")

    # Проверяем на сетке возрастов включая границы
    test_ages = np.array([18.0, 25.0, 35.0, 50.0, 65.0, 80.0])
    h_values = model.predict_h(test_ages)

    if not isinstance(h_values, np.ndarray):
        raise RuntimeError("predict_h(array) should return ndarray")

    if h_values.shape != test_ages.shape:
        raise RuntimeError(f"predict_h output shape mismatch: {h_values.shape} vs {test_ages.shape}")

    if not np.isfinite(h_values).all():
        raise RuntimeError("predict_h returned non-finite values")


def test_predict_h_scalar_vs_array_consistent() -> None:
    """
    Проверяет что predict_h дает одинаковые результаты для скаляра и массива.
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    ages = np.linspace(20.0, 70.0, 400)
    z_values = 0.01 * (ages - 35.0)

    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages.astype(float),
        "Z": z_values.astype(float),
    })

    model = fitter.fit_gender(gender_df=df, gender="M")

    test_ages = [25.0, 35.0, 50.0]

    # Скалярные вызовы
    h_scalars = [model.predict_h(a) for a in test_ages]

    # Массивный вызов
    h_array = model.predict_h(np.array(test_ages))

    for i, age in enumerate(test_ages):
        if not isinstance(h_scalars[i], float):
            raise RuntimeError(f"predict_h(scalar) should return float, got {type(h_scalars[i])}")

        diff = abs(h_scalars[i] - h_array[i])
        if diff > 1e-12:
            raise RuntimeError(f"predict_h scalar vs array mismatch at age={age}: {diff}")


def test_predict_h_clamps_age_outside_bounds() -> None:
    """
    Проверяет что predict_h корректно обрабатывает возраста вне глобальных границ.
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    ages = np.linspace(20.0, 70.0, 400)
    z_values = 0.01 * (ages - 35.0)

    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages.astype(float),
        "Z": z_values.astype(float),
    })

    model = fitter.fit_gender(gender_df=df, gender="M")

    # Возраст ниже минимума должен давать то же что минимум
    h_below = model.predict_h(10.0)
    h_at_min = model.predict_h(18.0)

    if abs(h_below - h_at_min) > 1e-12:
        raise RuntimeError(f"predict_h(10) should equal predict_h(18), diff={abs(h_below - h_at_min)}")

    # Возраст выше максимума должен давать то же что максимум
    h_above = model.predict_h(100.0)
    h_at_max = model.predict_h(80.0)

    if abs(h_above - h_at_max) > 1e-12:
        raise RuntimeError(f"predict_h(100) should equal predict_h(80), diff={abs(h_above - h_at_max)}")


def test_predict_mean_equals_predict_h_when_mu_gamma_zero() -> None:
    """
    Проверяет что predict_mean = predict_h когда coef_mu=0 и coef_gamma=0.
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    ages = np.linspace(20.0, 70.0, 400)
    z_values = 0.01 * (ages - 35.0)

    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages.astype(float),
        "Z": z_values.astype(float),
    })

    model = fitter.fit_gender(gender_df=df, gender="M")

    # Пока coef_mu и coef_gamma равны 0
    if model.coef_mu != 0.0 or model.coef_gamma != 0.0:
        raise RuntimeError("This test assumes coef_mu=0 and coef_gamma=0")

    test_ages = np.array([25.0, 35.0, 50.0])

    h_values = model.predict_h(test_ages)
    m_values = model.predict_mean(test_ages)

    max_diff = float(np.max(np.abs(h_values - m_values)))
    if max_diff > 1e-12:
        raise RuntimeError(f"predict_mean should equal predict_h when mu=gamma=0, max_diff={max_diff}")


def test_design_row_returns_correct_structure() -> None:
    """
    Проверяет что design_row возвращает словарь с правильными ключами.
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    ages = np.linspace(20.0, 70.0, 400)
    z_values = 0.01 * (ages - 35.0)

    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages.astype(float),
        "Z": z_values.astype(float),
    })

    model = fitter.fit_gender(gender_df=df, gender="M")

    row = model.design_row(40.0)

    if not isinstance(row, dict):
        raise RuntimeError("design_row should return dict")

    if "intercept" not in row:
        raise RuntimeError("design_row must have 'intercept' key")

    if "x" not in row:
        raise RuntimeError("design_row must have 'x' key")

    if row["intercept"] != 1.0:
        raise RuntimeError(f"intercept should be 1.0, got {row['intercept']}")

    # x = (40 - 35) / 10 = 0.5
    expected_x = 0.5
    if abs(row["x"] - expected_x) > 1e-12:
        raise RuntimeError(f"x should be {expected_x}, got {row['x']}")

    # Должны быть ключи spline_0, spline_1, ...
    k_raw = model.coef_beta.size
    for j in range(k_raw):
        key = f"spline_{j}"
        if key not in row:
            raise RuntimeError(f"design_row must have key '{key}'")
        if not np.isfinite(row[key]):
            raise RuntimeError(f"design_row['{key}'] is not finite")




def test_predict() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """


    tests: list[tuple[str, callable]] = [
        ("test_predict_h_at_age_center_is_near_zero",
         test_predict_h_at_age_center_is_near_zero),
        ("test_predict_h_returns_finite_across_age_range",
         test_predict_h_returns_finite_across_age_range),
        ("test_predict_h_scalar_vs_array_consistent",
         test_predict_h_scalar_vs_array_consistent),

        ("test_predict_h_clamps_age_outside_bounds",
         test_predict_h_clamps_age_outside_bounds),
        ("test_predict_mean_equals_predict_h_when_mu_gamma_zero",
         test_predict_mean_equals_predict_h_when_mu_gamma_zero),
        ("test_design_row_returns_correct_structure",
         test_design_row_returns_correct_structure),
    ]

    for test_name, test_fn in tests:
        print(f"\n== Running: {test_name}")
        try:
            test_fn()
            print(f"PASSED: {test_name}")
        except Exception as e:
            print(f"FAILED: {test_name}")
            print(f"Error: {e}")
            raise

    print("\n" + "=" * 60)
    print("=" * 60)