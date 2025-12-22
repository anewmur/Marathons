"""
Модуль для обучения возрастных сплайновых моделей.

Содержит:
- AgeSplineFitter: класс для построения моделей для всех полов
- Вспомогательные функции для построения узлов, базиса, центрирования, оценки параметров
"""
import math
import time

from typing import Any
from scipy.interpolate import BSpline
import numpy as np
import pandas as pd
import logging

from spline_model.build_centering_matrix import build_centering_matrix
from spline_model.select_lambda_reml import select_lambda_reml
from spline_model.compute_tau2_bar import compute_tau2_bar

logger = logging.getLogger(__name__)

from spline_model.age_spline_model import AgeSplineModel


# ============================================================================
# Вспомогательные функции
# ============================================================================


def build_knots_x(
    x_values: pd.Series,
    degree: int,
    max_inner_knots: int,
    min_knot_gap: float,
) -> list[float]:
    """
    Строит полный список узлов knots_x для B-сплайна степени degree по обучающим x_std.

    Вход:
    - x_values: значения x_std (вещественные, без NA), уже посчитанные по формуле
      x_std = (age_clamped - age_center) / age_scale.
      Здесь age_clamped получен клипом по глобальным границам возраста из конфига.
    - degree >= 1
    - max_inner_knots >= 0
    - min_knot_gap > 0 (в шкале x_std; например 0.2 соответствует 2 годам при age_scale=10)

    Выход:
    - knots_x: список float вида:
      [x_left] * (degree + 1) + inner_knots + [x_right] * (degree + 1),
      где inner_knots построены по квантилям и затем детерминированно "схлопнуты".

    Правила:
    1) Границы x_left/x_right берутся из диапазона x_values (после клипа по возрасту).
    2) Внутренние узлы: квантили на равномерной сетке размером max_inner_knots.
    3) Узлы у границ не допускаются (строго внутри (x_left, x_right) с допуском eps).
    4) Схлопывание: оставляем первый, следующий добавляем только если разрыв >= min_knot_gap.
    """
    if degree < 1:
        raise ValueError(f"build_knots_x: degree must be >= 1, got {degree}")
    if max_inner_knots < 0:
        raise ValueError(f"build_knots_x: max_inner_knots must be >= 0, got {max_inner_knots}")
    if not (min_knot_gap > 0.0) or math.isnan(min_knot_gap):
        raise ValueError(f"build_knots_x: min_knot_gap must be > 0, got {min_knot_gap}")

    if x_values is None or len(x_values) == 0:
        raise ValueError("build_knots_x: x_values is empty")
    if x_values.isna().any():
        raise ValueError("build_knots_x: x_values contains NA")

    x_left = float(x_values.min())
    x_right = float(x_values.max())
    if not (x_right > x_left):
        raise ValueError(
            "build_knots_x: x_values has zero range; cannot build knots "
            f"(x_left={x_left}, x_right={x_right})"
        )

    boundary_left = [x_left] * (degree + 1)
    boundary_right = [x_right] * (degree + 1)
    if max_inner_knots == 0:
        return boundary_left + boundary_right

    # Квантили на фиксированной сетке, затем уникализация и сортировка.
    quantile_grid = [float(step) / float(max_inner_knots + 1) for step in range(1, max_inner_knots + 1)]
    raw_inner = [float(x_values.quantile(q)) for q in quantile_grid]
    raw_inner_sorted = sorted(set(raw_inner))

    # Запрещаем внутренние узлы, совпадающие с границами (или слишком близкие).
    eps = 1e-9
    inner_candidates: list[float] = []
    for candidate in raw_inner_sorted:
        if (candidate <= x_left + eps) or (candidate >= x_right - eps):
            continue
        inner_candidates.append(candidate)

    # Детерминированное схлопывание по min_knot_gap.
    merged_inner: list[float] = []
    for candidate in inner_candidates:
        if not merged_inner:
            merged_inner.append(candidate)
            continue
        if (candidate - merged_inner[-1]) >= min_knot_gap:
            merged_inner.append(candidate)

    if len(merged_inner) > max_inner_knots:
        merged_inner = merged_inner[:max_inner_knots]

    return boundary_left + merged_inner + boundary_right


def build_raw_basis(
    x_values: np.ndarray,
    knots_x: list[float],
    degree: int,
) -> np.ndarray:
    """
    Строит матрицу "сырого" B-сплайнового базиса B_raw в шкале x_std.

     Определения:
    - x_values: точки x_std = (age_clamped - age_center) / age_scale, shape (n,)
    - knots_x: полный список узлов в той же шкале x_std, включая граничные кратности degree+1
    - degree: степень сплайна (обычно 3)

    - x_values лежат в [knots_x[0], knots_x[-1]] с небольшим допуском; если выходит за границы, функция падает
    - build_raw_basis не делает clamp/clip; ответственность за clamp возраста и вычисление x_std выше по пайплайну

    Возвращает:
    - B_raw shape (n, K_raw), где K_raw = len(knots_x) - degree - 1

    Контракт:
    - x_values одномерный, без NA/inf
    - knots_x не пустой, отсортирован неубывающе, корректен по размерам
    - x_values лежат в [knots_x[0], knots_x[-1]] с небольшим допуском; иначе ValueError
    - функция не делает clamp/clip x_values; clamp возраста выполняется раньше (при построении train_frame и x_std)
    """
    x_array = np.asarray(x_values, dtype=float)
    if x_array.ndim != 1:
        raise ValueError(f"build_raw_basis: x_values must be 1d, got ndim={x_array.ndim}")
    if x_array.size == 0:
        raise ValueError("build_raw_basis: x_values is empty")
    if not np.isfinite(x_array).all():
        raise ValueError("build_raw_basis: x_values contains non-finite values")

    if degree < 0:
        raise ValueError(f"build_raw_basis: degree must be >= 0, got {degree}")
    if len(knots_x) == 0:
        raise ValueError("build_raw_basis: knots_x is empty")

    knots_array = np.asarray(knots_x, dtype=float)
    if knots_array.ndim != 1:
        raise ValueError("build_raw_basis: knots_x must be 1d")
    if not np.isfinite(knots_array).all():
        raise ValueError("build_raw_basis: knots_x contains non-finite values")
    if np.any(np.diff(knots_array) < 0.0):
        raise ValueError("build_raw_basis: knots_x must be non-decreasing")

    k_raw = int(len(knots_array) - int(degree) - 1)
    if k_raw <= 0:
        raise ValueError(
            f"build_raw_basis: invalid sizes: len(knots_x)={len(knots_x)}, degree={degree}, K_raw={k_raw}"
        )

    eps = 1e-12
    x_min = float(np.min(x_array))
    x_max = float(np.max(x_array))
    left = float(knots_array[0])
    right = float(knots_array[-1])
    if (x_min < left - eps) or (x_max > right + eps):
        raise ValueError(
            "build_raw_basis: x_values must lie within [knots_x[0], knots_x[-1]]; "
            f"got x_range=[{x_min}, {x_max}], knots_range=[{left}, {right}]"
        )

    # Используем BSpline.design_matrix — O(n) вместо O(n * K_raw),
    # возвращает sparse CSR, конвертируем в dense для совместимости с остальным кодом
    dm_sparse = BSpline.design_matrix(x_array, knots_array, degree, extrapolate=False)
    b_raw = dm_sparse.toarray()

    return b_raw


def build_second_difference_matrix(coefficient_count: int) -> np.ndarray:
    """
    Построить оператор вторых разностей D для вектора beta длины coefficient_count.

    D имеет shape (K-2, K), где K=coefficient_count.
    (D beta)[i] = beta[i] - 2*beta[i+1] + beta[i+2]
    """
    if coefficient_count < 3:
        raise ValueError(f"coefficient_count must be >= 3, got {coefficient_count}")

    row_count = coefficient_count - 2
    difference_matrix = np.zeros((row_count, coefficient_count), dtype=float)

    for row_index in range(row_count):
        difference_matrix[row_index, row_index] = 1.0
        difference_matrix[row_index, row_index + 1] = -2.0
        difference_matrix[row_index, row_index + 2] = 1.0

    return difference_matrix


def solve_penalized_lsq(
    b_raw: np.ndarray,
    y: np.ndarray,
    null_basis: np.ndarray,
    penalty_matrix_raw: np.ndarray,
    lambda_value: float,
) -> dict[str, np.ndarray | float]:
    """
    Решить штрафуемую МНК-задачу в редуцированной параметризации.

    Модель:
        y ~ B_raw @ beta,  beta = C @ gamma
    Ограничения уже "зашиты" в C (null_basis).

    Задача:
        min_gamma || y - (B_raw C) gamma ||^2 + lambda * gamma^T (C^T P C) gamma
    где P = penalty_matrix_raw (на beta), а P_cent = C^T P C (на gamma).

    Args:
        b_raw: (n, K_raw) дизайн-матрица базисов ( матрица объясняющих переменных)
        y: (n,) вектор отклика
        null_basis: C, shape (K_raw, K_cent)
        penalty_matrix_raw: P, shape (K_raw, K_raw)
        lambda_value: lambda >= 0

    Returns:
        dict с ключами:
            gamma: (K_cent,)
            beta: (K_raw,)
            rss: float
            fitted: (n,)
    """
    b_raw_array = np.asarray(b_raw, dtype=float)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    null_basis_array = np.asarray(null_basis, dtype=float)
    penalty_raw_array = np.asarray(penalty_matrix_raw, dtype=float)

    if b_raw_array.ndim != 2:
        raise ValueError("b_raw must be 2D")
    if y_array.ndim != 1:
        raise ValueError("y must be 1D")
    if b_raw_array.shape[0] != y_array.shape[0]:
        raise ValueError("b_raw row count must match y length")

    if null_basis_array.ndim != 2:
        raise ValueError("null_basis must be 2D")
    if null_basis_array.shape[0] != b_raw_array.shape[1]:
        raise ValueError("null_basis row count must match b_raw column count")

    if penalty_raw_array.shape != (b_raw_array.shape[1], b_raw_array.shape[1]):
        raise ValueError("penalty_matrix_raw has invalid shape")

    if not np.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError(f"lambda_value must be finite and >= 0, got {lambda_value}")

    b_cent = b_raw_array @ null_basis_array
    penalty_cent = null_basis_array.T @ penalty_raw_array @ null_basis_array

    left_matrix = b_cent.T @ b_cent + lambda_value * penalty_cent
    right_vector = b_cent.T @ y_array

    gamma = np.linalg.solve(left_matrix, right_vector)

    beta = null_basis_array @ gamma
    fitted = b_raw_array @ beta
    residual = y_array - fitted
    rss = float(residual.T @ residual)

    return {
        "gamma": gamma,
        "beta": beta,
        "fitted": fitted,
        "rss": rss,
    }

def solve_penalized_lsq_with_linear(
    b_raw: np.ndarray,
    x_std: np.ndarray,
    y: np.ndarray,
    null_basis: np.ndarray,
    penalty_matrix_raw: np.ndarray,
    lambda_value: float,
) -> dict[str, Any]:
    """
    Решает penalized LSQ для модели:
        y = mu + gamma * x_std + B_raw * beta + eps,
    где beta = C @ gamma_spline и штраф применяется только к spline-части.

    Возвращает:
      mu, gamma_linear, gamma_spline, beta, fitted, rss
    """
    b_raw_array = np.asarray(b_raw, dtype=float)
    x_std_array = np.asarray(x_std, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    null_basis_array = np.asarray(null_basis, dtype=float)
    penalty_raw_array = np.asarray(penalty_matrix_raw, dtype=float)

    row_count = int(b_raw_array.shape[0])
    if x_std_array.shape[0] != row_count:
        raise RuntimeError("solve_penalized_lsq_with_linear: x_std length mismatch")
    if y_array.shape[0] != row_count:
        raise RuntimeError("solve_penalized_lsq_with_linear: y length mismatch")

    b_cent = b_raw_array @ null_basis_array
    penalty_cent = null_basis_array.T @ penalty_raw_array @ null_basis_array

    ones = np.ones(shape=(row_count, 1), dtype=float)
    x_col = x_std_array.reshape(-1, 1)
    design = np.concatenate([ones, x_col, b_cent], axis=1)

    p_spline = np.asarray(penalty_cent, dtype=float)
    p_size = int(p_spline.shape[0])
    penalty_block = np.zeros(shape=(2 + p_size, 2 + p_size), dtype=float)
    penalty_block[2:, 2:] = float(lambda_value) * p_spline

    left_matrix = design.T @ design + penalty_block
    right_vector = design.T @ y_array

    theta = np.linalg.solve(left_matrix, right_vector)

    mu = float(theta[0])
    gamma_linear = float(theta[1])
    gamma_spline = np.asarray(theta[2:], dtype=float)

    beta = null_basis_array @ gamma_spline
    fitted = mu + gamma_linear * x_std_array + (b_raw_array @ beta)
    residual = y_array - fitted
    rss = float(residual.T @ residual)

    return {
        "mu": mu,
        "gamma_linear": gamma_linear,
        "gamma_spline": gamma_spline,
        "beta": beta,
        "fitted": fitted,
        "rss": rss,
    }

def apply_winsor(
    z: np.ndarray,
    z_hat: np.ndarray,
    k: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Применение winsorization к остаткам.
    
    Args:
        z: Целевая переменная (n,)
        z_hat: Предсказанные значения (n,)
        k: Порог = k * MAD
    
    Returns:
        Кортеж (z_winsor, params):
        - z_winsor: Зажатая целевая переменная (n,)
        - params: словарь с "median", "mad", "lower", "upper", "fraction_clamped"
    
    Raises:
        NotImplementedError: Будет реализовано в Этапе 6
    """
    raise NotImplementedError("apply_winsor будет реализован в Этапе 6")


def determine_degradation(
    n: int,
    age_range_years: float,
    n_min_spline: int,
    range_min_years: float,
    n_min_linear: int,
    range_min_linear: float
) -> str:
    """
    Определение режима деградации модели при малых данных.
    
    Args:
        n: Число наблюдений
        age_range_years: Диапазон возрастов в годах
        n_min_spline: Минимум для сплайна
        range_min_years: Минимальный диапазон для сплайна
        n_min_linear: Минимум для линейной модели
        range_min_linear: Минимальный диапазон для линейной
    
    Returns:
        Режим: "none", "linear_only", "constant_only"
    
    Raises:
        NotImplementedError: Будет реализовано в Этапе 7
    """
    raise NotImplementedError("determine_degradation будет реализован в Этапе 7")






# ============================================================================
# Класс AgeSplineFitter
# ============================================================================

class AgeSplineFitter:
    """
    Обучение возрастной поправки h_g для каждого пола.

    Вход в fit/fit_gender:
    DataFrame с колонками:
        - gender: "M"/"F"
        - age: возраст в годах
        - z: целевая переменная на шкале Z, обычно z = Y - ln R^{use}

    Выход:
    - fit(): dict[gender, AgeSplineModel]
    - fit_gender(): AgeSplineModel для одного пола

    Ключевой контракт центрирования:
    Ограничения h(0)=0 и h'(0)=0 накладываются через матрицу ограничений A
    и базис нулевого пространства C, так что beta = C @ gamma всегда допустим.
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Инициализация параметров из config.
        
        ВСЕ параметры считываются ЗДЕСЬ и фиксируются.
        В методе fit обращения к config НЕТ.
        
        Args:
            config: Полный словарь конфигурации с секциями:
                - config["age_spline_model"] — параметры сплайна
                - config["preprocessing"] — age_center, age_scale
        
        Raises:
            KeyError: Если отсутствуют обязательные секции в config
        """
        self.config = config

        preprocessing = config["preprocessing"]
        age_model = config["age_spline_model"]


        # Нормировка возраста (из preprocessing)
        self.age_center = float(preprocessing["age_center"])
        self.age_scale = float(preprocessing["age_scale"])
        
        # Глобальные границы для clamp при обучении
        self.age_min_global = float(age_model["age_min_global"])
        self.age_max_global = float(age_model["age_max_global"])
        
        # B-сплайн параметры
        self.degree = int(age_model["degree"])
        self.max_inner_knots = int(age_model["max_inner_knots"])
        self.min_knot_gap = float(age_model["min_knot_gap"])
        
        # λ выбор
        self.lambda_value = float(age_model.get("lambda_value", 0.0))
        self.lambda_method = str(age_model.get("lambda_method", "fixed")).upper()

        self.centering_tol = float(age_model.get("centering_tol", 1e-10))

        self.sigma2_floor = float(age_model.get("sigma2_floor", 1e-8))

        logger.info(
            "AgeSplineFitter initialized: age_center=%s, age_scale=%s, degree=%s, lambda_method=%s",
            self.age_center,
            self.age_scale,
            self.degree,
            self.lambda_method,
        )

    def fit(self, df: pd.DataFrame, trace_references: pd.DataFrame | None = None) -> dict[str, AgeSplineModel]:
        """
       Строит модели возрастного сплайна по всем полам из df.

        Принимает: df с колонками gender, age, Z (Z вычислен заранее).
        Возвращает: dict[gender, AgeSplineModel].
        Делает: валидирует вход, режет по полу, зовёт fit_gender для каждого пола.
        """
        start_time = time.time()

        work_df = self._validate_fit_input(df=df)
        genders = self._get_sorted_genders(df=work_df)

        models: dict[str, AgeSplineModel] = {}
        for gender in genders:
            gender_df = work_df.loc[work_df["gender"] == gender, ["gender", "age", "Z", "race_id"]].copy()

            rows_in_gender = int(len(gender_df))
            if rows_in_gender == 0:
                raise RuntimeError(f"AgeSplineFitter.fit: empty gender slice for gender={gender}")

            logger.info("fit_gender: gender=%s, n=%d", gender, rows_in_gender)
            models[gender] = self.fit_gender(gender_df=gender_df, gender=str(gender), trace_references=trace_references)

        elapsed_sec = float(time.time() - start_time)
        logger.info(
            "AgeSplineFitter.fit: completed genders=%s in %.3fs",
            list(models.keys()),
            elapsed_sec,
        )
        return models

    def _get_sorted_genders(self, df: pd.DataFrame) -> list[str]:
        """
        Принимает: df с колонкой gender.
        Возвращает: список полов в детерминированном порядке.
        Делает: нормализует тип к str и сортирует так, чтобы M шёл перед F если оба есть.
        """
        genders_raw = df["gender"].astype(str).unique().tolist()
        genders = [str(value) for value in genders_raw]

        order = {"M": 0, "F": 1}
        return sorted(genders, key=lambda item: (order.get(item, 99), item))

    def fit_gender(self, gender_df: pd.DataFrame,
                   gender: str,
                   trace_references: pd.DataFrame | None = None) -> "AgeSplineModel":
        """
        Обучить модель для одного пола через конвейер шагов.

        TODO:
        Сейчас на этом этапе мы подгоняем только сплайн-часть (coef_beta).
        Линейная часть (coef_mu, coef_gamma) и дисперсии будут добавлены следующим шагом.

        Шаги:
        1) извлечение age и z
        2) построение x_std
        3) узлы и сырой базис
        4) центрирование (A, C)
        5) штраф P-spline (в коэффициентах)
        6) решение penalized LSQ в редуцированной параметризации
        7) проверка ограничений
        8) сборка AgeSplineModel
        """
        self._validate_fit_gender_input(gender_df=gender_df, gender=gender)

        ages_raw, z_values = self._extract_age_and_z(gender_df)
        x_std, ages_clamped = self._compute_x_std_and_clamped_age(ages_raw)

        knots_x_np, b_raw = self._build_knots_and_basis(x_std)
        constraints_matrix_A, null_basis_C = self._build_centering(knots_x_np)

        penalty_matrix_raw = self._build_penalty_matrix_raw(b_raw)

        lambda_value, reml_terms = self._select_lambda_value(
            b_raw=b_raw,
            x_std=x_std,
            y=z_values,
            null_basis=null_basis_C,
            penalty_matrix_raw=penalty_matrix_raw,
        )

        tau2_bar, sigma2_use = self._compute_dispersions(
            gender_df=gender_df,
            gender=gender,
            trace_references=trace_references,
            reml_terms=reml_terms,
        )

        solution = solve_penalized_lsq_with_linear(
            b_raw=b_raw,
            x_std=x_std,
            y=z_values,
            null_basis=null_basis_C,
            penalty_matrix_raw=penalty_matrix_raw,
            lambda_value=lambda_value,
        )

        beta = np.asarray(solution["beta"], dtype=float)
        coef_mu = float(solution["mu"])
        coef_gamma = float(solution["gamma_linear"])

        self._assert_constraints(constraints_matrix=constraints_matrix_A, beta=beta)

        age_range_actual = self._compute_age_range_actual(ages_clamped)

        model = self._assemble_model(
            gender=gender,
            knots_x=knots_x_np,
            degree=int(self.degree),
            constraints_matrix_A=constraints_matrix_A,
            null_basis_C=null_basis_C,
            coef_mu=coef_mu,
            coef_gamma=coef_gamma,
            beta=beta,
            tau2_bar=tau2_bar,
            sigma2_use=sigma2_use,
            lambda_value=lambda_value,
            age_range_actual=age_range_actual,
            sample_size=int(len(z_values)),
            reml_terms=reml_terms,
        )

        # DEBUG: sigma2 consistency check for final assembled model
        # Проверяем, что sigma2_use согласован с RSS/nu по фактическим остаткам этой модели на train.
        nu_debug = None
        sigma2_reml_debug = None
        if isinstance(reml_terms, dict):
            best_terms = reml_terms.get("best_terms")
            if isinstance(best_terms, dict):
                nu_debug = best_terms.get("nu")
                sigma2_reml_debug = best_terms.get("sigma2_hat")
            if nu_debug is None:
                nu_debug = reml_terms.get("nu")
            if sigma2_reml_debug is None:
                sigma2_reml_debug = reml_terms.get("sigma2_hat")

        if nu_debug is None:
            raise RuntimeError("DEBUG_SIGMA2: cannot find nu in reml_terms")
        nu_debug = float(nu_debug)

        mean_train_debug = model.predict_mean(ages_raw)
        if not isinstance(mean_train_debug, np.ndarray):
            mean_train_debug = np.asarray([float(mean_train_debug)], dtype=float)

        residual_debug = np.asarray(z_values, dtype=float).reshape(-1) - mean_train_debug.reshape(-1)
        rss_debug = float(np.sum(residual_debug * residual_debug))
        sigma2_from_rss_debug = rss_debug / nu_debug

        nu_plus2 = nu_debug - 2.0
        if nu_plus2 > 0.0:
            sigma2_from_rss_plus2 = rss_debug / nu_plus2
            print(f"  sigma2_from_rss_plus2=RSS/(nu-2): {sigma2_from_rss_plus2:.12f}")


        print("DEBUG_SIGMA2_FINAL_MODEL:")
        print(f"  n={int(len(z_values))}")
        print(f"  nu(reml_terms)={nu_debug:.6f}")
        print(f"  rss(final_model)={rss_debug:.6f}")
        print(f"  sigma2_from_rss(final_model)=RSS/nu={sigma2_from_rss_debug:.12f}")
        print(f"  sigma2_use(passed_to_model)={float(sigma2_use):.12f}")
        if sigma2_reml_debug is not None:
            print(f"  sigma2_hat(reml_terms)={float(sigma2_reml_debug):.12f}")

        ratio_debug = float(sigma2_use) / float(sigma2_from_rss_debug)
        print(f"  ratio sigma2_use / (RSS/nu) = {ratio_debug:.6f}")

        # Жёсткий стоп, если расходится сильно. Порог можешь временно сделать шире.
        if not (0.98 <= ratio_debug <= 1.02):
            print(
                "DEBUG_SIGMA2: mismatch between sigma2_use and RSS/nu for final model. "
                f"sigma2_use={float(sigma2_use):.6f}, "
                f"sigma2_from_rss={float(sigma2_from_rss_debug):.6f}, "
                f"ratio={ratio_debug:.3f}"
            )


        if sigma2_reml_debug is not None:
            print(f"  rss(reml_terms)={float(reml_terms.get('rss')):.6f}")
            print(f"  nu(reml_terms)={float(reml_terms.get('nu')):.6f}")
            print(f"  sigma2_hat(reml_terms)=rss/nu={float(reml_terms.get('sigma2_hat')):.12f}")


        return model

    def _compute_dispersions(
            self,
            gender_df: pd.DataFrame,
            gender: str,
            trace_references: pd.DataFrame | None,
            reml_terms: dict[str, Any] | None,
    ) -> tuple[float, float]:
        """
        Вычисляет tau2_bar и sigma2_use.

        Returns:
            (tau2_bar, sigma2_use): оба float >= 0
        """
        # 1. Вычисляем tau2_bar
        tau2_bar = 0.0
        if trace_references is not None and not trace_references.empty:
            tau2_bar = compute_tau2_bar(
                gender_df=gender_df,
                trace_references=trace_references,
                gender=gender,
            )

        # 2. Получаем sigma2_reml
        sigma2_reml = 0.0
        if reml_terms is not None and "sigma2_hat" in reml_terms:
            sigma2_reml = float(reml_terms["sigma2_hat"])

        # 3. Вычисляем sigma2_use
        sigma2_floor = float(getattr(self, "sigma2_floor", 1e-8))
        # sigma2_use = max(sigma2_floor, sigma2_reml - tau2_bar)
        sigma2_use = max(sigma2_floor, sigma2_reml)

        return float(tau2_bar), float(sigma2_use)

    def _validate_fit_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Принимает: df.
        Возвращает: копию df (минимально очищенную).
        Делает: проверяет наличие нужных колонок и отсутствие NA в них.
        """
        required_columns = ["gender", "age", "Z"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"AgeSplineFitter.fit: missing columns: {missing_columns}")

        columns_to_copy = required_columns.copy()
        if "race_id" in df.columns:
            columns_to_copy.append("race_id")

        if df.empty:
            raise ValueError("AgeSplineFitter.fit: df is empty")

        work_df = df[columns_to_copy].copy()
        if work_df[required_columns].isna().any().any():
            na_share = work_df[required_columns].isna().mean().to_dict()
            raise ValueError(f"AgeSplineFitter.fit: NA in required columns: {na_share}")

        return work_df

    def _validate_fit_gender_input(self, gender_df: pd.DataFrame, gender: str) -> None:
        required_columns = ["age", "Z"]
        missing_columns = [col for col in required_columns if col not in gender_df.columns]
        if missing_columns:
            raise ValueError(f"fit_gender: missing required columns: {missing_columns}")

        if gender_df.empty:
            raise ValueError(f"fit_gender: gender_df is empty for gender={gender}")

        if gender_df[required_columns].isna().any().any():
            na_share = gender_df[required_columns].isna().mean().to_dict()
            raise ValueError(f"fit_gender: gender_df has NA: {na_share}")

    @staticmethod
    def _extract_age_and_z(gender_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        ages_raw = gender_df["age"].astype(float).to_numpy(dtype=float)
        z_values = gender_df["Z"].astype(float).to_numpy(dtype=float)

        if not np.isfinite(ages_raw).all():
            raise RuntimeError("_extract_age_and_z: ages contain non-finite values")
        if not np.isfinite(z_values).all():
            raise RuntimeError("_extract_age_and_z: z contains non-finite values")

        return ages_raw, z_values

    def _compute_x_std_and_clamped_age(self, ages_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Строит x_std и одновременно возвращает ages_clamped.

        Зачем:
        - x_std нужен для узлов и базиса
        - ages_clamped нужен для age_range_actual и отчёта
        """
        age_min_global = float(self.age_min_global)
        age_max_global = float(self.age_max_global)

        ages_clamped = np.clip(ages_raw, a_min=age_min_global, a_max=age_max_global)

        age_center = float(self.age_center)
        age_scale = float(self.age_scale)
        if not (age_scale > 0.0):
            raise ValueError(f"_compute_x_std_and_clamped_age: age_scale must be > 0, got {age_scale}")

        x_std = (ages_clamped - age_center) / age_scale
        x_std = np.asarray(x_std, dtype=float)

        if not np.isfinite(x_std).all():
            raise RuntimeError("_compute_x_std_and_clamped_age: x_std contains non-finite values")

        return x_std, np.asarray(ages_clamped, dtype=float)

    @staticmethod
    def _compute_age_range_actual(ages_clamped: np.ndarray) -> tuple[float, float]:
        """
        Фактический диапазон возраста для пола (после clamp).

        Это не глобальные границы, а фактические min/max в данных пола.
        """
        if ages_clamped.size == 0:
            return 0.0, 0.0

        min_age = float(np.min(ages_clamped))
        max_age = float(np.max(ages_clamped))
        return min_age, max_age



    def _build_knots_and_basis(self, x_std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        knots_x = build_knots_x(
            x_values=pd.Series(x_std, dtype=float),
            degree=int(self.degree),
            max_inner_knots=int(self.max_inner_knots),
            min_knot_gap=float(self.min_knot_gap),
        )
        b_raw = build_raw_basis(
            x_values=np.asarray(x_std, dtype=float),
            knots_x=knots_x,
            degree=int(self.degree),
        )
        knots_x_np = np.asarray(knots_x, dtype=float)
        return knots_x_np, b_raw

    def _build_centering(self, knots_x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        constraints_matrix, null_basis = build_centering_matrix(
            knots_x=knots_x_np,
            degree=int(self.degree),
            x0=0.0,
            svd_tol=1e-12,
        )
        return constraints_matrix, null_basis

    @staticmethod
    def _build_penalty_matrix_raw(b_raw: np.ndarray) -> np.ndarray:
        coefficient_count = int(b_raw.shape[1])
        difference_matrix = build_second_difference_matrix(coefficient_count=coefficient_count)
        penalty_matrix_raw = difference_matrix.T @ difference_matrix
        return penalty_matrix_raw

    def _select_lambda_value(
            self,
            b_raw: np.ndarray,
            x_std: np.ndarray,
            y: np.ndarray,
            null_basis: np.ndarray,
            penalty_matrix_raw: np.ndarray,
    ) -> tuple[float, dict[str, Any] | None]:
        method = str(getattr(self, "lambda_method", "FIXED")).upper()

        if method == "FIXED":
            lambda_value = float(getattr(self, "lambda_value", 1.0))
            if not np.isfinite(lambda_value) or lambda_value < 0.0:
                raise ValueError(f"_select_lambda_value: lambda_value must be finite and >= 0, got {lambda_value}")
            return float(lambda_value), None

        if method == "REML":
            b_raw_array = np.asarray(b_raw, dtype=float)
            y_array = np.asarray(y, dtype=float).reshape(-1)
            x_std_array = np.asarray(x_std, dtype=float).reshape(-1)

            C = np.asarray(null_basis, dtype=float)
            P_raw = np.asarray(penalty_matrix_raw, dtype=float)

            row_count = int(b_raw_array.shape[0])
            if x_std_array.shape[0] != row_count:
                raise RuntimeError("_select_lambda_value: x_std length mismatch")
            if y_array.shape[0] != row_count:
                raise RuntimeError("_select_lambda_value: y length mismatch")

            b_cent = b_raw_array @ C
            P_cent = C.T @ P_raw @ C

            ones = np.ones((row_count, 1), dtype=float)
            x_col = x_std_array.reshape(-1, 1)
            design = np.concatenate([ones, x_col, b_cent], axis=1)

            p_size = int(P_cent.shape[0])
            P_full = np.zeros((2 + p_size, 2 + p_size), dtype=float)
            P_full[2:, 2:] = P_cent

            best_lambda, info = select_lambda_reml(
                W=design,
                y=y_array,
                penalty_matrix=P_full,
            )

            best_terms = info.get("best_terms")
            if not isinstance(best_terms, dict):
                raise RuntimeError(f"_select_lambda_value: REML best_terms missing or invalid: {best_terms}")
            if not bool(best_terms.get("ok")):
                raise RuntimeError(f"_select_lambda_value: REML best_terms not ok: {best_terms}")

            return float(best_lambda), best_terms

        raise ValueError(f"_select_lambda_value: unknown lambda_method={method}")

    @staticmethod
    def _solve(
            b_raw: np.ndarray,
            x_std: np.ndarray,
            y: np.ndarray,
            null_basis: np.ndarray,
            penalty_matrix_raw: np.ndarray,
            lambda_value: float,
    ) -> dict[str, Any]:
        solution = solve_penalized_lsq_with_linear(
            b_raw=b_raw,
            x_std=x_std,
            y=y,
            null_basis=null_basis,
            penalty_matrix_raw=penalty_matrix_raw,
            lambda_value=lambda_value,
        )
        return solution

    def _assert_constraints(self, constraints_matrix: np.ndarray, beta: np.ndarray) -> None:
        max_abs_constraint = float(np.max(np.abs(constraints_matrix @ beta)))
        if not np.isfinite(max_abs_constraint):
            raise RuntimeError("_assert_constraints: non-finite constraint error")

        if max_abs_constraint > float(self.centering_tol):
            raise RuntimeError(
                "_assert_constraints: centering constraints violated "
                f"(max_abs_constraint={max_abs_constraint}, tol={self.centering_tol})"
            )

    def _assemble_model(
            self,
            gender: str,
            knots_x: np.ndarray,
            degree: int,
            constraints_matrix_A: np.ndarray,
            null_basis_C: np.ndarray,
            beta: np.ndarray,
            coef_mu: float,
            tau2_bar: float,
            sigma2_use: float,
            coef_gamma: float,
            lambda_value: float,
            age_range_actual: tuple[float, float],
            sample_size: int,
            reml_terms: dict[str, Any] | None,
    ) -> "AgeSplineModel":
        """
        Собрать AgeSplineModel под твой текущий класс.

        Что заполняем сейчас:
        - базовую нормировку (age_center, age_scale, границы)
        - сплайн: knots_x, degree
        - центрирование: basis_centering с A и C
        - coef_beta: коэффициенты сплайна (пока без mu/gamma)
        - lambda_value
        - fit_report: минимальные метрики для контроля контракта

        Что оставляем нулями до следующего шага:
        - coef_mu, coef_gamma
        - sigma2_reml, tau2_bar, sigma2_use, nu
        - winsor_params
        """
        knots_list = [float(value) for value in np.asarray(knots_x, dtype=float).tolist()]

        k_raw = int(constraints_matrix_A.shape[1])
        k_cent = int(null_basis_C.shape[1])

        basis_centering = {
            "centering_method": "constraints_value_slope_at_x0",
            "x0": 0.0,
            "A": np.asarray(constraints_matrix_A, dtype=float),
            "C": np.asarray(null_basis_C, dtype=float),
            "K_raw": k_raw,
            "K_cent": k_cent,
        }

        # coef_beta по контракту как pd.Series с индексом spline_0, spline_1, ...
        beta_np = np.asarray(beta, dtype=float)
        beta_names = [f"spline_{index}" for index in range(int(beta_np.size))]
        coef_beta = pd.Series(beta_np, index=beta_names, dtype=float)

        lambda_method = str(getattr(self, "lambda_method", "FIXED")).upper()

        report_edf: float | None = None
        report_nu: float | None = None
        report_sigma2: float | None = None

        if lambda_method == "REML":
            if reml_terms is None:
                raise RuntimeError("_assemble_model: lambda_method=REML but reml_terms is None")

            report_edf = float(reml_terms["edf"])
            report_nu = float(reml_terms["nu"])
            report_sigma2 = float(reml_terms["sigma2_hat"])

        fit_report = {
            "n": int(sample_size),
            "coef_mu": float(coef_mu),
            "coef_gamma": float(coef_gamma),
            "age_range_actual": (float(age_range_actual[0]), float(age_range_actual[1])),
            "age_range_years": float(age_range_actual[1] - age_range_actual[0]),
            "knots_count_inner": int(max(0, len(knots_list) - 2 * (int(degree) + 1))),
            "degree": int(degree),
            "K_raw": int(k_raw),
            "K_cent": int(k_cent),

            "lambda_value": float(lambda_value),
            "lambda_method": str(lambda_method),

            "edf": report_edf,
            "nu": report_nu,
            "sigma2_reml": report_sigma2,

            "warnings": [],
        }
        return AgeSplineModel(
            gender=str(gender),

            # Нормировка
            age_center=float(self.age_center),
            age_scale=float(self.age_scale),
            x0=0.0,

            # Границы
            age_min_global=float(self.age_min_global),
            age_max_global=float(self.age_max_global),
            age_range_actual=(float(age_range_actual[0]), float(age_range_actual[1])),

            # Сплайн
            degree=int(degree),
            knots_x=knots_list,
            basis_centering=basis_centering,

            # Коэффициенты
            coef_mu=float(coef_mu),
            coef_gamma=float(coef_gamma),
            coef_beta=coef_beta,

            # Дисперсии
            lambda_value=float(lambda_value),
            sigma2_reml=0.0 if report_sigma2 is None else float(report_sigma2),
            nu=3.0 if report_nu is None else float(report_nu),
            tau2_bar=float(tau2_bar),
            sigma2_use=float(sigma2_use),


            # Winsor
            winsor_params={},

            # Отчет
            fit_report=fit_report,
        )