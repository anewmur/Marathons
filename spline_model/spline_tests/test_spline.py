import numpy as np
import pandas as pd

from spline_model.age_spline_fit import build_knots_x, build_raw_basis
from spline_model.age_spline_fit import solve_penalized_lsq
import logging
logger = logging.getLogger(__name__)

def test_build_raw_basis_partition_of_unity() -> None:
    """
    Проверяет базовое свойство B-сплайнового базиса:
    сумма базисных функций в каждой точке должна быть близка к 1.0.
    """
    degree = 3
    x_values = np.linspace(0.0, 1.0, 101)

    # простой набор внутренних узлов через build_knots_x
    knots_x = build_knots_x(
        x_values=pd.Series(x_values),
        degree=degree,
        max_inner_knots=6,
        min_knot_gap=0.05,
    )

    b_raw = build_raw_basis(x_values=x_values, knots_x=knots_x, degree=degree)
    row_sums = b_raw.sum(axis=1)

    max_abs_err = float(np.max(np.abs(row_sums - 1.0)))
    print("build_raw_basis partition of unity max_abs_err:", max_abs_err)

    if max_abs_err > 1e-10:
        raise RuntimeError(f"partition of unity violated, max_abs_err={max_abs_err}")



def test_solve_penalized_lsq_recovers_gamma_when_lambda_zero() -> None:
    rng = np.random.default_rng(123)

    row_count = 80
    k_raw = 14
    k_cent = k_raw - 2

    # Делаем C с ортонормальными столбцами (как null(A)), но тесту не важно откуда оно.
    random_matrix = rng.normal(size=(k_raw, k_cent))
    orthonormal_basis, _ = np.linalg.qr(random_matrix)
    null_basis = orthonormal_basis[:, :k_cent]

    # Строим B_raw так, чтобы B_cent был полного ранга:
    # берём B_cent = случайная матрица (n x k_cent), и определяем B_raw = B_cent @ C.T
    b_cent_target = rng.normal(size=(row_count, k_cent))
    b_raw = b_cent_target @ null_basis.T

    # Нулевой штраф (но P должен быть корректной формы)
    penalty_raw = np.eye(k_raw, dtype=float)
    lambda_value = 0.0

    gamma_true = rng.normal(size=k_cent)
    beta_true = null_basis @ gamma_true
    y = b_raw @ beta_true

    result = solve_penalized_lsq(
        b_raw=b_raw,
        y=y,
        null_basis=null_basis,
        penalty_matrix_raw=penalty_raw,
        lambda_value=lambda_value,
    )

    gamma_hat = result["gamma"]
    assert isinstance(gamma_hat, np.ndarray)

    max_abs_err = float(np.max(np.abs(gamma_hat - gamma_true)))
    assert max_abs_err < 1e-8
