import numpy as np
import pandas as pd

from main import MarathonModel


class _DummyAgeModel:
    """Минимальный дублёр AgeSplineModel для unit-теста."""

    def __init__(self, constant_h: float) -> None:
        self._constant_h = float(constant_h)

    def predict_h(self, age: float | np.ndarray) -> float | np.ndarray:
        age_array = np.asarray(age, dtype=float)
        if age_array.ndim == 0:
            return float(self._constant_h)
        return np.full(shape=age_array.shape, fill_value=float(self._constant_h), dtype=float)


def test_predict_log_time_equals_reference_plus_h_unit() -> None:
    model = MarathonModel(data_path=".", verbose=False)

    model.trace_references = pd.DataFrame(
        {
            "race_id": ["R1"],
            "gender": ["M"],
            "reference_log": [np.log(100.0)],
        }
    )
    model.age_spline_models = {"M": _DummyAgeModel(constant_h=0.25)}
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True

    predicted = model.predict_log_time(race_id="R1", gender="M", age=30.0, year=2020)
    expected = float(np.log(100.0) + 0.25)
    if not np.isfinite(float(predicted)):
        raise RuntimeError("predict_log_time returned non-finite")
    if abs(float(predicted) - expected) > 1e-12:
        raise RuntimeError(f"predict_log_time mismatch: {predicted} vs {expected}")

    ages = np.array([20.0, 30.0, 40.0], dtype=float)
    predicted_vec = model.predict_log_time(race_id="R1", gender="M", age=ages, year=2020)
    if not isinstance(predicted_vec, np.ndarray):
        raise RuntimeError("predict_log_time(array) must return ndarray")
    if predicted_vec.shape != ages.shape:
        raise RuntimeError(f"predict_log_time shape mismatch: {predicted_vec.shape} vs {ages.shape}")

    expected_vec = np.full(shape=ages.shape, fill_value=float(np.log(100.0) + 0.25), dtype=float)
    max_abs_err = float(np.max(np.abs(predicted_vec - expected_vec)))
    if max_abs_err > 1e-12:
        raise RuntimeError(f"predict_log_time(array) mismatch: max_abs_err={max_abs_err}")


def test_predict_log_time() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """


    tests: list[tuple[str, callable]] = [
        ("test_predict_log_time_equals_reference_plus_h_unit",
         test_predict_log_time_equals_reference_plus_h_unit),
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