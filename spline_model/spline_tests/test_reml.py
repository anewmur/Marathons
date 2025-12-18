# spline_model/spline_tests/test_reml.py
"""
Тесты для REML выбора lambda.

Стиль: как у тебя, без pytest.
"""

import numpy as np
import pandas as pd

from spline_model.age_spline_fit import build_second_difference_matrix, AgeSplineFitter
from spline_model.select_lambda_reml import select_lambda_reml


def test_select_lambda_reml_returns_positive_finite_lambda_and_valid_terms() -> None:
    rng = np.random.default_rng(123)

    n_obs = 400
    param_count = 12

    W = rng.normal(0.0, 1.0, size=(n_obs, param_count))
    beta_true = rng.normal(0.0, 1.0, size=param_count)
    y = (W @ beta_true) + rng.normal(0.0, 0.4, size=n_obs)

    D2 = build_second_difference_matrix(coefficient_count=param_count)
    K = D2.T @ D2

    best_lambda, info = select_lambda_reml(W=W, y=y, penalty_matrix=K)

    if not np.isfinite(best_lambda):
        raise RuntimeError("best_lambda is not finite")
    if best_lambda <= 0.0:
        raise RuntimeError(f"best_lambda must be > 0, got {best_lambda}")

    best_terms = info.get("best_terms")
    if not isinstance(best_terms, dict):
        raise RuntimeError("best_terms missing or invalid type")

    if not bool(best_terms.get("ok")):
        raise RuntimeError(f"best_terms not ok: {best_terms}")

    nu = float(best_terms["nu"])
    edf = float(best_terms["edf"])
    rss = float(best_terms["rss"])
    reml = float(best_terms["reml"])

    if not np.isfinite(nu) or not np.isfinite(edf) or not np.isfinite(rss) or not np.isfinite(reml):
        raise RuntimeError("best_terms contains non-finite values")

    if nu <= 3.0:
        raise RuntimeError(f"nu must be > 3, got {nu}")

    if edf > float(n_obs) - 1.0:
        raise RuntimeError(f"edf must be <= n-1, got edf={edf} n={n_obs}")


def test_fit_gender_with_lambda_method_reml_runs_and_returns_positive_lambda() -> None:
    rng = np.random.default_rng(7)

    ages = np.linspace(18.0, 80.0, 1200)
    z_values = 0.03 * (ages - 35.0) + 0.002 * (ages - 55.0) ** 2
    z_values = z_values + rng.normal(0.0, 0.25, size=len(ages))

    df = pd.DataFrame(
        {"gender": ["M"] * len(ages), "age": ages.astype(float), "Z": z_values.astype(float)}
    )

    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 6,
            "min_knot_gap": 0.2,
            "lambda_method": "REML",
            "lambda_value": 1.0,
            "centering_tol": 1e-10,
        },
    }

    fitter = AgeSplineFitter(config=config)
    model = fitter.fit_gender(gender_df=df, gender="M")

    if not np.isfinite(float(model.lambda_value)):
        raise RuntimeError("model.lambda_value is not finite")
    if float(model.lambda_value) <= 0.0:
        raise RuntimeError(f"model.lambda_value must be > 0, got {model.lambda_value}")


def test_reml() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """


    tests: list[tuple[str, callable]] = [
        ("test_select_lambda_reml_returns_positive_finite_lambda_and_valid_terms",
         test_select_lambda_reml_returns_positive_finite_lambda_and_valid_terms),
        ("test_fit_gender_with_lambda_method_reml_runs_and_returns_positive_lambda",
         test_fit_gender_with_lambda_method_reml_runs_and_returns_positive_lambda),
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