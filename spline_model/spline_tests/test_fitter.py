

import numpy as np
import pandas as pd

from spline_model.age_spline_fit import AgeSplineFitter


def test_fit_gender_produces_model_and_beta_is_finite() -> None:
    """
    Быстрый unit-тест: обучение на синтетике создаёт модель,
    и коэффициенты coef_beta конечны.

    Проверяет связку:
    knots -> B_raw -> (A,C) -> P_raw -> solve -> coef_beta
    """
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.1,
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)

    ages = np.linspace(20.0, 70.0, 400)
    z_values = 0.01 * (ages - 35.0)

    df = pd.DataFrame(
        {
            "gender": ["M"] * int(len(ages)),
            "age": ages.astype(float),
            "Z": z_values.astype(float),
        }
    )

    model = fitter.fit_gender(gender_df=df, gender="M")

    beta_series = model.coef_beta
    if not isinstance(beta_series, pd.Series):
        raise RuntimeError("model.coef_beta must be pd.Series")

    beta = beta_series.to_numpy(dtype=float)

    if beta.shape[0] <= 2:
        raise RuntimeError("coef_beta has too few coefficients for cubic spline")

    if not np.isfinite(beta).all():
        raise RuntimeError("coef_beta contains non-finite values")

    expected_names = [f"spline_{index}" for index in range(int(beta.shape[0]))]
    if beta_series.index.tolist() != expected_names:
        raise RuntimeError("coef_beta index is not spline_0..spline_{K-1}")


def test_fitter() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """


    tests: list[tuple[str, callable]] = [
        ("test_fit_gender_produces_model_and_beta_is_finite",
         test_fit_gender_produces_model_and_beta_is_finite),
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