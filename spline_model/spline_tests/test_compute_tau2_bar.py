"""
Тесты для функции compute_tau2_bar.

Проверяют правильность вычисления средней дисперсии эталона
в различных сценариях.
"""

import numpy as np
import pandas as pd

from spline_model.compute_tau2_bar import compute_tau2_bar


def test_compute_tau2_bar_basic() -> None:
    """
    Базовый тест: простое вычисление среднего reference_variance.
    
    Сценарий:
    - 4 записи для пола M, 2 трассы (R1, R2)
    - R1 имеет reference_variance = 0.001
    - R2 имеет reference_variance = 0.002
    - Ожидаемое tau2_bar = (0.001 + 0.001 + 0.002 + 0.002) / 4 = 0.0015
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R1", "R2", "R2"],
        "gender": ["M", "M", "M", "M"],
        "age": [30.0, 35.0, 30.0, 35.0],
        "Z": [0.1, 0.2, 0.15, 0.25]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "reference_variance": [0.001, 0.002]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "M")
    
    expected = 0.0015
    if abs(tau2_bar - expected) > 1e-10:
        raise RuntimeError(
            f"tau2_bar mismatch: got {tau2_bar}, expected {expected}"
        )


def test_compute_tau2_bar_single_race() -> None:
    """
    Тест с единственной трассой.
    
    Все записи относятся к одной трассе, tau2_bar = её reference_variance.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R1", "R1"],
        "gender": ["F", "F", "F"],
        "age": [25.0, 30.0, 35.0],
        "Z": [0.05, 0.10, 0.15]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["F"],
        "reference_variance": [0.0025]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "F")
    
    expected = 0.0025
    if abs(tau2_bar - expected) > 1e-10:
        raise RuntimeError(
            f"tau2_bar mismatch: got {tau2_bar}, expected {expected}"
        )


def test_compute_tau2_bar_multiple_races_unbalanced() -> None:
    """
    Тест с несбалансированным количеством записей по трассам.
    
    Сценарий:
    - R1: 1 запись, variance = 0.001
    - R2: 3 записи, variance = 0.003
    - Среднее взвешивается по числу записей
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R2", "R2", "R2"],
        "gender": ["M", "M", "M", "M"],
        "age": [30.0, 25.0, 30.0, 35.0],
        "Z": [0.1, 0.05, 0.10, 0.15]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "reference_variance": [0.001, 0.003]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "M")
    
    # Среднее: (1*0.001 + 3*0.003) / 4 = 0.0025
    expected = 0.0025
    if abs(tau2_bar - expected) > 1e-10:
        raise RuntimeError(
            f"tau2_bar mismatch: got {tau2_bar}, expected {expected}"
        )


def test_compute_tau2_bar_no_reference_variance_column() -> None:
    """
    Тест случая когда reference_variance отсутствует в trace_references.
    
    Должен вернуть 0.0 и залогировать предупреждение.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "age": [30.0, 35.0],
        "Z": [0.1, 0.2]
    })
    
    # trace_references без reference_variance
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "reference_log": [4.5, 4.6]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "M")
    
    if tau2_bar != 0.0:
        raise RuntimeError(
            f"Expected tau2_bar=0.0 when reference_variance missing, got {tau2_bar}"
        )


def test_compute_tau2_bar_no_matching_races() -> None:
    """
    Тест случая когда gender_df содержит трассы отсутствующие в trace_references.
    
    После merge reference_variance будет NA, должен вернуть 0.0.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R3", "R4"],  # Трассы которых нет в trace_references
        "gender": ["M", "M"],
        "age": [30.0, 35.0],
        "Z": [0.1, 0.2]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "reference_variance": [0.001, 0.002]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "M")
    
    if tau2_bar != 0.0:
        raise RuntimeError(
            f"Expected tau2_bar=0.0 when no matching races, got {tau2_bar}"
        )


def test_compute_tau2_bar_partial_match() -> None:
    """
    Тест случая когда только часть трасс матчится.
    
    Среднее берётся только по записям с непустым reference_variance.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R1", "R3", "R3"],  # R1 есть, R3 нет в references
        "gender": ["M", "M", "M", "M"],
        "age": [30.0, 35.0, 30.0, 35.0],
        "Z": [0.1, 0.2, 0.15, 0.25]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "reference_variance": [0.004, 0.002]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "M")
    
    # R1 матчится: 2 записи * 0.004
    # R3 не матчится: 2 записи с NA
    # mean([0.004, 0.004, NA, NA]) = 0.004
    expected = 0.004
    if abs(tau2_bar - expected) > 1e-10:
        raise RuntimeError(
            f"tau2_bar mismatch: got {tau2_bar}, expected {expected}"
        )


def test_compute_tau2_bar_zero_variance() -> None:
    """
    Тест случая когда reference_variance = 0.0.
    
    Должен корректно вернуть 0.0.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R1"],
        "gender": ["F", "F"],
        "age": [30.0, 35.0],
        "Z": [0.1, 0.2]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["F"],
        "reference_variance": [0.0]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "F")
    
    if tau2_bar != 0.0:
        raise RuntimeError(
            f"Expected tau2_bar=0.0, got {tau2_bar}"
        )


def test_compute_tau2_bar_negative_variance_warning() -> None:
    """
    Тест случая когда reference_variance отрицательна (ошибка в данных).
    
    Должен вернуть 0.0 и залогировать предупреждение.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "age": [30.0],
        "Z": [0.1]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_variance": [-0.001]  # Некорректное значение
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "M")
    
    if tau2_bar != 0.0:
        raise RuntimeError(
            f"Expected tau2_bar=0.0 for negative variance, got {tau2_bar}"
        )


def test_compute_tau2_bar_missing_race_id_in_gender_df() -> None:
    """
    Тест ошибки: gender_df не содержит race_id.
    
    Должен поднять ValueError.
    """
    gender_df = pd.DataFrame({
        "gender": ["M", "M"],
        "age": [30.0, 35.0],
        "Z": [0.1, 0.2]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1"],
        "gender": ["M"],
        "reference_variance": [0.001]
    })
    
    try:
        compute_tau2_bar(gender_df, trace_references, "M")
        raise RuntimeError("Should have raised ValueError for missing race_id")
    except ValueError as e:
        if "race_id" not in str(e):
            raise RuntimeError(f"Wrong error message: {e}")


def test_compute_tau2_bar_missing_gender_in_trace_references() -> None:
    """
    Тест ошибки: trace_references не содержит gender.
    
    Должен поднять ValueError.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "gender": ["M", "M"],
        "age": [30.0, 35.0],
        "Z": [0.1, 0.2]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2"],
        "reference_variance": [0.001, 0.002]
    })
    
    try:
        compute_tau2_bar(gender_df, trace_references, "M")
        raise RuntimeError("Should have raised ValueError for missing gender")
    except ValueError as e:
        if "gender" not in str(e):
            raise RuntimeError(f"Wrong error message: {e}")


def test_compute_tau2_bar_returns_finite_positive() -> None:
    """
    Тест что результат всегда конечный и неотрицательный.
    """
    gender_df = pd.DataFrame({
        "race_id": ["R1", "R2", "R3"],
        "gender": ["F", "F", "F"],
        "age": [25.0, 30.0, 35.0],
        "Z": [0.05, 0.10, 0.15]
    })
    
    trace_references = pd.DataFrame({
        "race_id": ["R1", "R2", "R3"],
        "gender": ["F", "F", "F"],
        "reference_variance": [0.001, 0.002, 0.003]
    })
    
    tau2_bar = compute_tau2_bar(gender_df, trace_references, "F")
    
    if not np.isfinite(tau2_bar):
        raise RuntimeError(f"tau2_bar is not finite: {tau2_bar}")
    
    if tau2_bar < 0:
        raise RuntimeError(f"tau2_bar is negative: {tau2_bar}")


def test_compute_tau2_bar() -> None:
    """
    Точка входа для запуска всех тестов.
    """
    tests = [
        ("test_compute_tau2_bar_basic", test_compute_tau2_bar_basic),
        ("test_compute_tau2_bar_single_race", test_compute_tau2_bar_single_race),
        ("test_compute_tau2_bar_multiple_races_unbalanced", test_compute_tau2_bar_multiple_races_unbalanced),
        ("test_compute_tau2_bar_no_reference_variance_column", test_compute_tau2_bar_no_reference_variance_column),
        ("test_compute_tau2_bar_no_matching_races", test_compute_tau2_bar_no_matching_races),
        ("test_compute_tau2_bar_partial_match", test_compute_tau2_bar_partial_match),
        ("test_compute_tau2_bar_zero_variance", test_compute_tau2_bar_zero_variance),
        ("test_compute_tau2_bar_negative_variance_warning", test_compute_tau2_bar_negative_variance_warning),
        ("test_compute_tau2_bar_missing_race_id_in_gender_df", test_compute_tau2_bar_missing_race_id_in_gender_df),
        ("test_compute_tau2_bar_missing_gender_in_trace_references", test_compute_tau2_bar_missing_gender_in_trace_references),
        ("test_compute_tau2_bar_returns_finite_positive", test_compute_tau2_bar_returns_finite_positive),
    ]
    
    for test_name, test_fn in tests:
        print(f"\n== Running: {test_name}")
        try:
            test_fn()
            print(f"✓ PASSED: {test_name}")
        except Exception as e:
            print(f"✗ FAILED: {test_name}")
            print(f"Error: {e}")
            raise
    
    print("\n" + "=" * 60)
    print("All compute_tau2_bar tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_compute_tau2_bar()
