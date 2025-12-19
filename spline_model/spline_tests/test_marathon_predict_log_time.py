import numpy as np
import pandas as pd

from main import MarathonModel


class _DummyAgeModel:
    """Минимальный дублёр AgeSplineModel для unit-теста."""

    def __init__(self, constant_mean: float) -> None:
        self._constant_mean = float(constant_mean)

    def predict_mean(self, age: float | np.ndarray) -> float | np.ndarray:
        """Возвращает константное значение mean для любого возраста."""
        age_array = np.asarray(age, dtype=float)
        if age_array.ndim == 0:
            return float(self._constant_mean)
        return np.full(shape=age_array.shape,
                       fill_value=float(self._constant_mean),
                       dtype=float)


def test_predict_log_time_equals_reference_plus_mean_unit() -> None:
    """
        Тест проверяет контракт: predict_log_time = reference_log + predict_mean.

        Использует dummy модель с константным predict_mean=0.25.
        Проверяет и скалярный, и векторный вход.
        """
    model = MarathonModel(data_path=".", verbose=False)

    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_log": [float(np.log(100.0))],
    })

    # Dummy модель с константным mean
    model.age_spline_models = {"M": _DummyAgeModel(constant_mean=0.25)}

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


def test_predict_log_time_uses_predict_mean() -> None:
    """
    Тест проверяет, что predict_log_time вызывает predict_mean, а не predict_h.

    Использует dummy модель, которая различает два метода разными значениями.
    """

    # ═══════════════════════════════════════════════════════════
    # ШАГ 1: Создаём специальную tracked dummy модель
    # ═══════════════════════════════════════════════════════════
    class _TrackedDummyModel:
        """Dummy модель, которая отслеживает какой метод вызывается."""

        def __init__(self) -> None:
            # Флаги для отслеживания вызовов
            self.predict_mean_called = False
            self.predict_h_called = False

        def predict_mean(self, age: float | np.ndarray) -> float | np.ndarray:
            # Устанавливаем флаг что этот метод был вызван
            self.predict_mean_called = True
            age_array = np.asarray(age, dtype=float)
            if age_array.ndim == 0:
                return 0.5  # Возвращаем 0.5 (отличное от predict_h значение!)
            return np.full(age_array.shape, 0.5, dtype=float)

        def predict_h(self, age: float | np.ndarray) -> float | np.ndarray:
            # Устанавливаем флаг что этот метод был вызван
            self.predict_h_called = True
            age_array = np.asarray(age, dtype=float)
            if age_array.ndim == 0:
                return 0.999  # Возвращаем 0.999 (ДРУГОЕ значение!)
            return np.full(age_array.shape, 0.999, dtype=float)

    # ═══════════════════════════════════════════════════════════
    # ШАГ 2: Настраиваем тестовую модель
    # ═══════════════════════════════════════════════════════════
    model = MarathonModel(data_path=".", verbose=False)

    # trace_references с reference_log = 0 (для простоты)
    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_log": [0.0],
    })

    # Создаём tracked модель и подставляем её
    tracked_model = _TrackedDummyModel()
    model.age_spline_models = {"M": tracked_model}

    # Отмечаем что нужные шаги выполнены
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True

    # ═══════════════════════════════════════════════════════════
    # ШАГ 3: Вызываем predict_log_time
    # ═══════════════════════════════════════════════════════════
    result = model.predict_log_time(race_id="R1", gender="M", age=30.0, year=2020)

    # ═══════════════════════════════════════════════════════════
    # ШАГ 4: Проверяем ЧТО было вызвано
    # ═══════════════════════════════════════════════════════════

    # Проверка 1: predict_mean ДОЛЖЕН быть вызван
    if not tracked_model.predict_mean_called:
        raise RuntimeError("predict_log_time должен вызывать predict_mean!")

    # Проверка 2: predict_h НЕ должен быть вызван
    if tracked_model.predict_h_called:
        raise RuntimeError("predict_log_time НЕ должен вызывать predict_h!")

    # Проверка 3: результат должен быть 0.5 (из predict_mean), а не 0.999 (из predict_h)
    # reference_log = 0, поэтому:
    # - если вызвал predict_mean: result = 0.0 + 0.5 = 0.5 ✓
    # - если вызвал predict_h:    result = 0.0 + 0.999 = 0.999 ✗
    if abs(float(result) - 0.5) > 1e-12:
        raise RuntimeError(f"predict_log_time должен использовать predict_mean: {result} != 0.5")


def test_predict_log_time() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """


    tests: list[tuple[str, callable]] = [
        ("test_predict_log_time_equals_reference_plus_mean_unit",
         test_predict_log_time_equals_reference_plus_mean_unit),
        ('test_predict_log_time_uses_predict_mean',
         test_predict_log_time_uses_predict_mean),
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