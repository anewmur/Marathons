"""
метод predict_with_uncertainty с учётом reference_variance.

sigma2_total = sigma2_use + reference_variance

где:
- sigma2_use: остаточная дисперсия возрастной модели (после вычитания tau2_bar)
- reference_variance: дисперсия эталона конкретной трассы

"""

import numpy as np
from scipy import stats


def predict_with_uncertainty(
        model,
        race_id: str,
        gender: str,
        age: float | np.ndarray,
        year: int,
        confidence: float = 0.95,
        method: str = "analytical",
        n_samples: int = 10000,
) -> dict:
    """
    Прогноз времени забега с интервалом неопределённости.

    Модель: ln(T) ~ N(log_pred, sigma2_total), где
    sigma2_total = sigma2_use + reference_variance.

    sigma2_use: остаточная дисперсия возрастной модели (один для пола)
    reference_variance: дисперсия эталона трассы (разный для каждой трассы)

    Args:
        race_id: Идентификатор трассы
        gender: "M" или "F"
        age: Возраст (скаляр или массив)
        year: Год забега
        confidence: Уровень доверия (0 < confidence < 1)
        method: "analytical" или "monte_carlo"
        n_samples: Число симуляций для monte_carlo

    Returns:
        dict с ключами:
            # Прогноз
            'time_pred', 'time_lower', 'time_upper': в минутах
            'log_pred', 'log_lower', 'log_upper': в log-шкале

            # Компоненты дисперсии
            'sigma2_use': дисперсия возрастной модели
            'reference_variance': дисперсия эталона трассы
            'sigma2_total': итоговая дисперсия (сумма)
            'sigma': sqrt(sigma2_total)

            # Метаданные
            'confidence', 'method', 'race_id', 'gender', 'age', 'year'

            # Для monte_carlo:
            'samples': {'log_samples', 'time_samples', 'n_samples'}


    """
    # ═══════════════════════════════════════════════════════════
    # Валидация параметров
    # ═══════════════════════════════════════════════════════════


    if not (0.0 < float(confidence) < 1.0):
        raise ValueError(
            f"predict_with_uncertainty: confidence must be in (0, 1), got {confidence}"
        )

    method = method.lower()
    if method not in ["analytical", "monte_carlo"]:
        raise ValueError(
            f"predict_with_uncertainty: method must be 'analytical' or 'monte_carlo', "
            f"got '{method}'"
        )

    if method == "monte_carlo" and n_samples < 100:
        raise ValueError(
            f"predict_with_uncertainty: n_samples must be >= 100 for monte_carlo, "
            f"got {n_samples}"
        )

    # ═══════════════════════════════════════════════════════════
    # Точечный прогноз в log-шкале
    # ═══════════════════════════════════════════════════════════

    log_pred = model.predict_log_time(
        race_id=race_id,
        gender=gender,
        age=age,
        year=year,
    )

    # ═══════════════════════════════════════════════════════════
    # Получаем sigma2_use из возрастной модели
    # ═══════════════════════════════════════════════════════════

    gender_key = str(gender)
    if gender_key not in model.age_spline_models:
        raise KeyError(
            f"predict_with_uncertainty: no age model for gender={gender_key}"
        )

    age_model = model.age_spline_models[gender_key]
    sigma2_use = float(age_model.sigma2_use)

    if not np.isfinite(sigma2_use):
        raise RuntimeError(
            f"predict_with_uncertainty: sigma2_use is not finite (sigma2_use={sigma2_use})"
        )

    if sigma2_use < 0.0:
        raise RuntimeError(
            f"predict_with_uncertainty: sigma2_use must be >= 0, got {sigma2_use}"
        )

    if not hasattr(model, "trace_references") or model.trace_references is None:
        raise RuntimeError(
            "predict_with_uncertainty: trace_references is not available"
        )

    trace_frame = model.trace_references

    # Проверяем наличие reference_variance (если нет, используем reference_std²)
    if "reference_variance" in trace_frame.columns:
        variance_col = "reference_variance"
    elif "reference_std" in trace_frame.columns:
        # Вычисляем variance из std если нужно
        variance_col = "reference_std"
        use_std_squared = True
    else:
        raise RuntimeError(
            "predict_with_uncertainty: trace_references missing both "
            "'reference_variance' and 'reference_std' columns"
        )

    # Находим нужную трассу и пол
    mask = (trace_frame["race_id"] == str(race_id)) & (trace_frame["gender"] == gender_key)
    matched = trace_frame.loc[mask, [variance_col]]

    if len(matched) != 1:
        raise RuntimeError(
            f"predict_with_uncertainty: cannot find unique trace reference for "
            f"race_id={race_id}, gender={gender_key} (found {len(matched)} rows)"
        )

    # Получаем reference_variance
    if variance_col == "reference_variance":
        reference_variance = float(matched["reference_variance"].iloc[0])
    else:
        # Используем std²
        reference_std = float(matched["reference_std"].iloc[0])
        reference_variance = reference_std ** 2

    # Проверяем корректность
    if not np.isfinite(reference_variance):
        raise RuntimeError(
            f"predict_with_uncertainty: reference_variance is not finite "
            f"(race_id={race_id}, gender={gender_key}, value={reference_variance})"
        )

    if reference_variance < 0.0:
        raise RuntimeError(
            f"predict_with_uncertainty: reference_variance must be >= 0 "
            f"(race_id={race_id}, gender={gender_key}, value={reference_variance})"
        )

    sigma2_total = float(sigma2_use + reference_variance)
    sigma = float(np.sqrt(sigma2_total))

    if not np.isfinite(sigma):
        raise RuntimeError(
            f"predict_with_uncertainty: sigma is not finite "
            f"(sigma2_total={sigma2_total}, sigma2_use={sigma2_use}, "
            f"reference_variance={reference_variance})"
        )

    # ═══════════════════════════════════════════════════════════
    # Формируем базовый результат
    # ═══════════════════════════════════════════════════════════

    result = {
        'race_id': str(race_id),
        'gender': gender_key,
        'age': age if isinstance(age, (int, float)) else np.asarray(age, dtype=float).copy(),
        'year': int(year),
        'confidence': float(confidence),
        'method': method,

        # Компоненты дисперсии
        'sigma2_use': sigma2_use,
        'reference_variance': reference_variance,

        # Итоговая дисперсия. Держим оба ключа ради обратной совместимости тестов.
        'sigma2_total': sigma2_total,
        'sigma2': sigma2_total,

        'sigma': sigma,
    }

    alpha = 1.0 - confidence

    # ═══════════════════════════════════════════════════════════
    # ANALYTICAL METHOD
    # ═══════════════════════════════════════════════════════════

    if method == "analytical":
        # Квантиль нормального распределения
        z_score = stats.norm.ppf(1.0 - alpha / 2.0)

        # Интервал в log-шкале
        log_lower = log_pred - z_score * sigma
        log_upper = log_pred + z_score * sigma

        # В натуральной шкале
        time_pred = np.exp(log_pred)
        time_lower = np.exp(log_lower)
        time_upper = np.exp(log_upper)

        result.update({
            'time_pred': time_pred,
            'time_lower': time_lower,
            'time_upper': time_upper,
            'log_pred': log_pred,
            'log_lower': log_lower,
            'log_upper': log_upper,
        })

        return result

    # ═══════════════════════════════════════════════════════════
    # MONTE CARLO METHOD
    # ═══════════════════════════════════════════════════════════

    is_scalar = isinstance(age, (int, float)) or (
            isinstance(age, np.ndarray) and age.ndim == 0
    )

    if is_scalar:
        # ───────────────────────────────────────────────────────
        # Скалярный случай
        # ───────────────────────────────────────────────────────

        log_samples = np.random.normal(
            loc=float(log_pred),
            scale=sigma,
            size=n_samples,
        )

        # Перцентили в log-шкале
        log_lower = float(np.percentile(log_samples, (alpha / 2.0) * 100.0))
        log_upper = float(np.percentile(log_samples, (1.0 - alpha / 2.0) * 100.0))

        # В натуральной шкале
        time_samples = np.exp(log_samples)
        time_lower = float(np.percentile(time_samples, (alpha / 2.0) * 100.0))
        time_upper = float(np.percentile(time_samples, (1.0 - alpha / 2.0) * 100.0))
        time_pred = float(np.median(time_samples))

        result.update({
            'time_pred': time_pred,
            'time_lower': time_lower,
            'time_upper': time_upper,
            'log_pred': float(log_pred),
            'log_lower': log_lower,
            'log_upper': log_upper,
            'samples': {
                'log_samples': log_samples,
                'time_samples': time_samples,
                'n_samples': n_samples,
            },
        })

        return result

    # ───────────────────────────────────────────────────────
    # Массив возрастов
    # ───────────────────────────────────────────────────────

    age_array = np.asarray(age, dtype=float)
    log_pred_array = np.asarray(log_pred, dtype=float)
    n_ages = int(age_array.size)

    # Матрица симуляций
    log_samples_matrix = np.random.normal(
        loc=log_pred_array.reshape(n_ages, 1),
        scale=sigma,
        size=(n_ages, n_samples),
    )

    # Перцентили для каждого возраста
    log_lower = np.percentile(log_samples_matrix, (alpha / 2.0) * 100.0, axis=1)
    log_upper = np.percentile(log_samples_matrix, (1.0 - alpha / 2.0) * 100.0, axis=1)

    # В натуральной шкале
    time_samples_matrix = np.exp(log_samples_matrix)
    time_lower = np.percentile(time_samples_matrix, (alpha / 2.0) * 100.0, axis=1)
    time_upper = np.percentile(time_samples_matrix, (1.0 - alpha / 2.0) * 100.0, axis=1)
    time_pred = np.median(time_samples_matrix, axis=1)

    result.update({
        'time_pred': time_pred,
        'time_lower': time_lower,
        'time_upper': time_upper,
        'log_pred': log_pred_array,
        'log_lower': log_lower,
        'log_upper': log_upper,
        'samples': {
            'log_samples': log_samples_matrix,
            'time_samples': time_samples_matrix,
            'n_samples': n_samples,
        },
    })

    return result