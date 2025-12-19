"""
Тесты для метода predict_with_uncertainty.

Включает:
1. Unit-тесты на синтетических данных
2. Интеграционные тесты на реальных данных
"""

import numpy as np
import pandas as pd
from scipy import stats

from MarathonAgeModel import MarathonModel


def test_predict_with_uncertainty_scalar_age_analytical() -> None:
    """
    Тест: скалярный возраст + analytical метод.
    
    Проверяет:
    - Все обязательные ключи присутствуют
    - Значения конечны
    - Интервал корректный: time_lower < time_pred < time_upper
    - confidence правильно применяется
    """
    # Создаём минимальную модель
    model = MarathonModel(data_path=".", verbose=False)
    
    # Mock данные
    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_log": [np.log(180.0)],  # ~5.19
        "reference_variance": [0.01],
    })
    
    # Создаём простую age модель с известными параметрами
    from spline_model.age_spline_fit import AgeSplineFitter
    
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 5,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "sigma2_floor": 1e-8,
        },
    }
    
    fitter = AgeSplineFitter(config=config)
    
    # Синтетические данные: линейный тренд
    ages = np.linspace(20.0, 70.0, 300)
    z_values = 0.01 * (ages - 35.0) + np.random.default_rng(42).normal(0, 0.05, len(ages))
    
    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages,
        "Z": z_values,
        "race_id": ["R1"] * len(ages),
    })
    
    model.age_spline_models = {"M": fitter.fit_gender(df, "M", trace_references=None)}
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True
    
    # ═══════════════════════════════════════════════════════════
    # Делаем прогноз
    # ═══════════════════════════════════════════════════════════
    
    result = model.predict_with_uncertainty(
        race_id="R1",
        gender="M",
        age=40.0,
        year=2025,
        confidence=0.95,
        method="analytical"
    )
    
    # ═══════════════════════════════════════════════════════════
    # Проверки
    # ═══════════════════════════════════════════════════════════
    
    # 1. Проверка наличия всех ключей
    required_keys = [
        'time_pred', 'time_lower', 'time_upper',
        'log_pred', 'log_lower', 'log_upper',
        'sigma', 'sigma2', 'confidence',
        'race_id', 'gender', 'age', 'year', 'method'
    ]
    for key in required_keys:
        if key not in result:
            raise RuntimeError(f"Missing key in result: {key}")
    
    # 2. Проверка типов для скаляра
    if not isinstance(result['time_pred'], (int, float)):
        raise RuntimeError(f"time_pred должен быть скаляром, got {type(result['time_pred'])}")
    
    # 3. Проверка конечности
    for key in ['time_pred', 'time_lower', 'time_upper', 'sigma', 'sigma2']:
        if not np.isfinite(result[key]):
            raise RuntimeError(f"{key} is not finite: {result[key]}")
    
    # 4. Проверка порядка интервала
    if not (result['time_lower'] < result['time_pred'] < result['time_upper']):
        raise RuntimeError(
            f"Interval order violated: {result['time_lower']} < {result['time_pred']} < {result['time_upper']}"
        )

    # 5. Проверка метаданных
    if result['confidence'] != 0.95:
        raise RuntimeError(f"confidence mismatch: {result['confidence']} != 0.95")
    
    if result['method'] != "analytical":
        raise RuntimeError(f"method mismatch: {result['method']} != 'analytical'")

    print(f"✓ time_pred: {result['time_pred']:.2f} min")
    print(f"✓ 95% CI: [{result['time_lower']:.2f}, {result['time_upper']:.2f}] min")
    print(f"✓ sigma: {result['sigma']:.4f}")


def test_predict_with_uncertainty_array_age_analytical() -> None:
    """
    Тест: массив возрастов + analytical метод.
    
    Проверяет:
    - Результаты являются массивами правильной формы
    - Все значения конечны
    - Интервалы корректны для каждого возраста
    """
    model = MarathonModel(data_path=".", verbose=False)
    
    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["F"],
        "reference_log": [np.log(200.0)],
        "reference_variance": [0.008],
    })
    
    from spline_model.age_spline_fit import AgeSplineFitter
    
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 5,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "sigma2_floor": 1e-8,
        },
    }
    
    fitter = AgeSplineFitter(config=config)
    
    ages = np.linspace(20.0, 70.0, 300)
    z_values = 0.015 * (ages - 35.0) + np.random.default_rng(123).normal(0, 0.06, len(ages))
    
    df = pd.DataFrame({
        "gender": ["F"] * len(ages),
        "age": ages,
        "Z": z_values,
        "race_id": ["R1"] * len(ages),
    })
    
    model.age_spline_models = {"F": fitter.fit_gender(df, "F", trace_references=None)}
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True
    
    # ═══════════════════════════════════════════════════════════
    # Делаем прогноз для массива возрастов
    # ═══════════════════════════════════════════════════════════
    
    test_ages = np.array([25.0, 35.0, 45.0, 55.0])
    
    result = model.predict_with_uncertainty(
        race_id="R1",
        gender="F",
        age=test_ages,
        year=2025,
        confidence=0.90,
        method="analytical"
    )
    
    # ═══════════════════════════════════════════════════════════
    # Проверки
    # ═══════════════════════════════════════════════════════════
    
    # 1. Проверка типов для массива
    if not isinstance(result['time_pred'], np.ndarray):
        raise RuntimeError(f"time_pred должен быть массивом, got {type(result['time_pred'])}")
    
    # 2. Проверка размерности
    if result['time_pred'].shape != test_ages.shape:
        raise RuntimeError(
            f"Shape mismatch: {result['time_pred'].shape} != {test_ages.shape}"
        )
    
    # 3. Проверка конечности всех элементов
    for key in ['time_pred', 'time_lower', 'time_upper']:
        if not np.isfinite(result[key]).all():
            raise RuntimeError(f"{key} contains non-finite values")
    
    # 4. Проверка порядка интервалов для каждого возраста
    for i in range(len(test_ages)):
        if not (result['time_lower'][i] < result['time_pred'][i] < result['time_upper'][i]):
            raise RuntimeError(
                f"Interval order violated at age={test_ages[i]}: "
                f"{result['time_lower'][i]} < {result['time_pred'][i]} < {result['time_upper'][i]}"
            )
    
    # 5. sigma и sigma2 должны быть скалярами (одинаковы для всех возрастов)
    if not isinstance(result['sigma'], (int, float)):
        raise RuntimeError(f"sigma должен быть скаляром, got {type(result['sigma'])}")
    
    print(f"✓ Array predictions for ages: {test_ages}")
    print(f"✓ time_pred: {result['time_pred']}")
    print(f"✓ Intervals valid for all ages")


def test_predict_with_uncertainty_monte_carlo_scalar() -> None:
    """
    Тест: Monte Carlo метод для скаляра.
    
    Проверяет:
    - samples присутствует в результате
    - Размерность samples корректна
    - Результаты близки к analytical (с учётом стохастичности)
    """
    model = MarathonModel(data_path=".", verbose=False)
    
    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_log": [np.log(180.0)],
        "reference_variance": [0.01],
    })
    
    from spline_model.age_spline_fit import AgeSplineFitter
    
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 5,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "sigma2_floor": 1e-8,
        },
    }
    
    fitter = AgeSplineFitter(config=config)
    
    ages = np.linspace(20.0, 70.0, 300)
    z_values = 0.01 * (ages - 35.0) + np.random.default_rng(456).normal(0, 0.05, len(ages))
    
    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages,
        "Z": z_values,
        "race_id": ["R1"] * len(ages),
    })
    
    model.age_spline_models = {"M": fitter.fit_gender(df, "M", trace_references=None)}
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True
    
    # ═══════════════════════════════════════════════════════════
    # Делаем прогноз Monte Carlo
    # ═══════════════════════════════════════════════════════════
    
    n_samples = 5000
    result = model.predict_with_uncertainty(
        race_id="R1",
        gender="M",
        age=40.0,
        year=2025,
        confidence=0.95,
        method="monte_carlo",
        n_samples=n_samples
    )
    
    # ═══════════════════════════════════════════════════════════
    # Проверки
    # ═══════════════════════════════════════════════════════════
    
    # 1. Проверка наличия samples
    if 'samples' not in result:
        raise RuntimeError("samples missing in monte_carlo result")
    
    samples_dict = result['samples']
    
    if 'time_samples' not in samples_dict:
        raise RuntimeError("time_samples missing in samples dict")
    
    if 'log_samples' not in samples_dict:
        raise RuntimeError("log_samples missing in samples dict")
    
    # 2. Проверка размерности
    if samples_dict['time_samples'].shape != (n_samples,):
        raise RuntimeError(
            f"time_samples shape mismatch: {samples_dict['time_samples'].shape} != ({n_samples},)"
        )
    
    # 3. Проверка что samples конечны
    if not np.isfinite(samples_dict['time_samples']).all():
        raise RuntimeError("time_samples contain non-finite values")
    
    # 4. Сравнение с analytical (должны быть близки)
    result_analytical = model.predict_with_uncertainty(
        race_id="R1",
        gender="M",
        age=40.0,
        year=2025,
        confidence=0.95,
        method="analytical"
    )
    
    # time_pred должны быть близки (в пределах 1%)
    diff_pred = abs(result['time_pred'] - result_analytical['time_pred'])
    if diff_pred / result_analytical['time_pred'] > 0.01:
        raise RuntimeError(
            f"Monte Carlo time_pred differs from analytical by {diff_pred:.2f} min "
            f"({diff_pred/result_analytical['time_pred']*100:.1f}%)"
        )
    
    print(f"✓ Monte Carlo samples: {n_samples}")
    print(f"✓ time_pred (MC): {result['time_pred']:.2f}")
    print(f"✓ time_pred (analytical): {result_analytical['time_pred']:.2f}")
    print(f"✓ Difference: {diff_pred:.2f} min ({diff_pred/result_analytical['time_pred']*100:.2f}%)")


def test_predict_with_uncertainty_confidence_levels() -> None:
    """
    Тест: разные уровни доверия дают разные интервалы.
    
    Проверяет:
    - Интервал расширяется с ростом confidence
    - Порядок корректен для всех уровней
    """
    model = MarathonModel(data_path=".", verbose=False)
    
    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_log": [np.log(180.0)],
        "reference_variance": [0.01],
    })
    
    from spline_model.age_spline_fit import AgeSplineFitter
    
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 5,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "sigma2_floor": 1e-8,
        },
    }
    
    fitter = AgeSplineFitter(config=config)
    
    ages = np.linspace(20.0, 70.0, 300)
    z_values = 0.01 * (ages - 35.0) + np.random.default_rng(789).normal(0, 0.05, len(ages))
    
    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages,
        "Z": z_values,
        "race_id": ["R1"] * len(ages),
    })
    
    model.age_spline_models = {"M": fitter.fit_gender(df, "M", trace_references=None)}
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True
    
    # ═══════════════════════════════════════════════════════════
    # Проверяем разные уровни доверия
    # ═══════════════════════════════════════════════════════════
    
    confidence_levels = [0.80, 0.90, 0.95, 0.99]
    widths = []
    
    for conf in confidence_levels:
        result = model.predict_with_uncertainty(
            race_id="R1",
            gender="M",
            age=40.0,
            year=2025,
            confidence=conf,
            method="analytical"
        )
        
        width = result['time_upper'] - result['time_lower']
        widths.append(width)
        
        print(f"✓ {conf*100:.0f}% CI: [{result['time_lower']:.2f}, {result['time_upper']:.2f}] "
              f"(width={width:.2f})")
    
    # Проверка что интервалы расширяются
    for i in range(len(widths) - 1):
        if widths[i] >= widths[i + 1]:
            raise RuntimeError(
                f"Interval width should increase with confidence: "
                f"{widths[i]:.2f} >= {widths[i+1]:.2f} at {confidence_levels[i]}->{confidence_levels[i+1]}"
            )


def test_predict_with_uncertainty_validates_inputs() -> None:
    """
    Тест: валидация входных параметров.
    
    Проверяет что метод выбрасывает исключения для некорректных входов.
    """
    model = MarathonModel(data_path=".", verbose=False)
    
    model.trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_log": [np.log(180.0)],
        "reference_variance": [0.01],
    })
    
    from spline_model.age_spline_fit import AgeSplineFitter
    
    config = {
        "preprocessing": {"age_center": 35.0, "age_scale": 10.0},
        "age_spline_model": {
            "age_min_global": 18.0,
            "age_max_global": 80.0,
            "degree": 3,
            "max_inner_knots": 5,
            "min_knot_gap": 0.2,
            "lambda_value": 1.0,
            "sigma2_floor": 1e-8,
        },
    }
    
    fitter = AgeSplineFitter(config=config)
    
    ages = np.linspace(20.0, 70.0, 300)
    z_values = np.zeros(len(ages))
    
    df = pd.DataFrame({
        "gender": ["M"] * len(ages),
        "age": ages,
        "Z": z_values,
        "race_id": ["R1"] * len(ages),
    })
    
    model.age_spline_models = {"M": fitter.fit_gender(df, "M", trace_references=None)}
    model._steps_completed["build_trace_references"] = True
    model._steps_completed["fit_age_model"] = True
    
    # ═══════════════════════════════════════════════════════════
    # Тест 1: некорректный confidence
    # ═══════════════════════════════════════════════════════════
    
    try:
        model.predict_with_uncertainty(
            race_id="R1", gender="M", age=40.0, year=2025,
            confidence=1.5  # > 1
        )
        raise RuntimeError("Should have raised ValueError for confidence > 1")
    except ValueError as e:
        if "confidence" not in str(e):
            raise RuntimeError(f"Wrong error message: {e}")
    
    try:
        model.predict_with_uncertainty(
            race_id="R1", gender="M", age=40.0, year=2025,
            confidence=0.0  # <= 0
        )
        raise RuntimeError("Should have raised ValueError for confidence <= 0")
    except ValueError as e:
        if "confidence" not in str(e):
            raise RuntimeError(f"Wrong error message: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # Тест 2: некорректный method
    # ═══════════════════════════════════════════════════════════
    
    try:
        model.predict_with_uncertainty(
            race_id="R1", gender="M", age=40.0, year=2025,
            method="unknown"
        )
        raise RuntimeError("Should have raised ValueError for unknown method")
    except ValueError as e:
        if "method" not in str(e):
            raise RuntimeError(f"Wrong error message: {e}")
    
    # ═══════════════════════════════════════════════════════════
    # Тест 3: некорректный n_samples для monte_carlo
    # ═══════════════════════════════════════════════════════════
    
    try:
        model.predict_with_uncertainty(
            race_id="R1", gender="M", age=40.0, year=2025,
            method="monte_carlo", n_samples=10  # < 100
        )
        raise RuntimeError("Should have raised ValueError for n_samples < 100")
    except ValueError as e:
        if "n_samples" not in str(e):
            raise RuntimeError(f"Wrong error message: {e}")
    
    print("✓ All validation checks passed")




# ═══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════


def test_predict_with_uncertainty() -> None:
    """
    Точка входа для запуска всех тестов predict_with_uncertainty.
    """
    tests = [
        # Unit тесты на синтетике
        ("test_predict_with_uncertainty_scalar_age_analytical",
         test_predict_with_uncertainty_scalar_age_analytical),
        ("test_predict_with_uncertainty_array_age_analytical",
         test_predict_with_uncertainty_array_age_analytical),
        ("test_predict_with_uncertainty_monte_carlo_scalar",
         test_predict_with_uncertainty_monte_carlo_scalar),
        ("test_predict_with_uncertainty_confidence_levels",
         test_predict_with_uncertainty_confidence_levels),
        ("test_predict_with_uncertainty_validates_inputs",
         test_predict_with_uncertainty_validates_inputs),

    ]
    
    print("\n" + "=" * 70)
    print("TESTING predict_with_uncertainty")
    print("=" * 70)
    
    for test_name, test_fn in tests:
        print(f"\n== Running: {test_name}")
        try:
            test_fn()
            print(f"✓ PASSED: {test_name}")
        except Exception as e:
            print(f"✗ FAILED: {test_name}")
            print(f"Error: {e}")
            raise
    
    print("\n" + "=" * 70)
    print("ALL predict_with_uncertainty TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    test_predict_with_uncertainty()
