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


def select_lambda_gcv(
    W: np.ndarray,
    z: np.ndarray,
    lambda_grid: np.ndarray,
    penalty_idx: list[int]
) -> tuple[float, dict[str, Any]]:
    """
    Выбор λ по GCV (Generalized Cross-Validation).
    
    Args:
        W: Дизайн-матрица (n, p)
        z: Целевая переменная (n,)
        lambda_grid: Сетка значений λ для перебора
        penalty_idx: Индексы столбцов для штрафа
    
    Returns:
        Кортеж (best_lambda, info):
        - best_lambda: оптимальное значение λ
        - info: словарь с "gcv_values", "edf_values" и др.
    
    Raises:
        NotImplementedError: Будет реализовано в Этапе 4
    """
    raise NotImplementedError("select_lambda_gcv будет реализован в Этапе 4")


def compute_tau2_bar(
    gender_df: pd.DataFrame,
    trace_references: pd.DataFrame,
    gender: str
) -> float:
    """
    Вычисление средней дисперсии эталона для пола.
    
    Args:
        gender_df: Данные для пола с колонкой race_id
        trace_references: Таблица с reference_variance по (race_id, gender)
        gender: "M" или "F"
    
    Returns:
        tau2_bar: Среднее по строкам gender_df
    
    Raises:
        NotImplementedError: Будет реализовано в Этапе 5
    
    Примечание:
        Если reference_variance недоступна, возвращает 0.0 с предупреждением.
    """
    raise NotImplementedError("compute_tau2_bar будет реализован в Этапе 5")


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

        logger.info(
            "AgeSplineFitter initialized: age_center=%s, age_scale=%s, degree=%s, lambda_method=%s",
            self.age_center,
            self.age_scale,
            self.degree,
            self.lambda_method,
        )

    def fit(self, df: pd.DataFrame) -> dict[str, AgeSplineModel]:
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
            gender_df = work_df.loc[work_df["gender"] == gender, ["gender", "age", "Z"]].copy()

            rows_in_gender = int(len(gender_df))
            if rows_in_gender == 0:
                raise RuntimeError(f"AgeSplineFitter.fit: empty gender slice for gender={gender}")

            logger.info("fit_gender: gender=%s, n=%d", gender, rows_in_gender)
            models[gender] = self.fit_gender(gender_df=gender_df, gender=str(gender))

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

    def fit_gender(self, gender_df: pd.DataFrame, gender: str) -> "AgeSplineModel":
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

        lambda_value = self._get_lambda_value()
        solution = self._solve(
            b_raw=b_raw,
            y=z_values,
            null_basis=null_basis_C,
            penalty_matrix_raw=penalty_matrix_raw,
            lambda_value=lambda_value,
        )

        beta = np.asarray(solution["beta"], dtype=float)

        self._assert_constraints(constraints_matrix=constraints_matrix_A, beta=beta)

        age_range_actual = self._compute_age_range_actual(ages_clamped)

        model = self._assemble_model(
            gender=gender,
            knots_x=knots_x_np,
            degree=int(self.degree),
            constraints_matrix_A=constraints_matrix_A,
            null_basis_C=null_basis_C,
            beta=beta,
            lambda_value=lambda_value,
            age_range_actual=age_range_actual,
            sample_size=int(len(z_values)),
        )
        return model

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

        if df.empty:
            raise ValueError("AgeSplineFitter.fit: df is empty")

        work_df = df[required_columns].copy()
        if work_df.isna().any().any():
            na_share = work_df.isna().mean().to_dict()
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

    def _get_lambda_value(self) -> float:
        lambda_value = float(getattr(self, "lambda_value", 1.0))
        if not (lambda_value >= 0.0):
            raise ValueError(f"_get_lambda_value: lambda_value must be >= 0, got {lambda_value}")
        return lambda_value

    @staticmethod
    def _solve(
            b_raw: np.ndarray,
            y: np.ndarray,
            null_basis: np.ndarray,
            penalty_matrix_raw: np.ndarray,
            lambda_value: float,
    ) -> dict[str, np.ndarray]:
        solution = solve_penalized_lsq(
            b_raw=b_raw,
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
            lambda_value: float,
            age_range_actual: tuple[float, float],
            sample_size: int,
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

        fit_report = {
            "n": int(sample_size),
            "age_range_actual": (float(age_range_actual[0]), float(age_range_actual[1])),
            "age_range_years": float(age_range_actual[1] - age_range_actual[0]),
            "knots_count_inner": int(max(0, len(knots_list) - 2 * (int(degree) + 1))),
            "degree": int(degree),
            "K_raw": int(k_raw),
            "K_cent": int(k_cent),
            "lambda_value": float(lambda_value),
            "lambda_method": "fixed",
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
            coef_mu=0.0,
            coef_gamma=0.0,
            coef_beta=coef_beta,

            # Дисперсии
            lambda_value=float(lambda_value),
            sigma2_reml=0.0,
            tau2_bar=0.0,
            sigma2_use=0.0,
            nu=3.0,

            # Winsor
            winsor_params={},

            # Отчет
            fit_report=fit_report,
        )