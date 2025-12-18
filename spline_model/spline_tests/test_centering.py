# spline_model/spline_tests/test_centering.py

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline

from spline_model.build_centering_matrix import build_centering_matrix

from spline_model.age_spline_fit import solve_penalized_lsq


def test_solve_penalized_lsq_preserves_constraints_when_beta_is_C_gamma() -> None:
    rng = np.random.default_rng(456)

    row_count = 60
    k_raw = 12
    k_cent = k_raw - 2

    # Искусственная матрица ограничений A (2, K_raw) с рангом 2
    a_random = rng.normal(size=(2, k_raw))
    while np.linalg.matrix_rank(a_random) < 2:
        a_random = rng.normal(size=(2, k_raw))
    constraints_matrix = a_random

    # Строим C как null(A) через SVD (в тесте не пользуемся твоей build_centering_matrix)
    _, singular_values, vt = np.linalg.svd(constraints_matrix, full_matrices=True)
    max_singular = float(np.max(singular_values))
    threshold = 1e-12 * max_singular
    rank = int(np.sum(singular_values > threshold))
    null_basis = vt.T[:, rank:]

    # Делаем B_cent full rank и B_raw = B_cent @ C.T
    b_cent_target = rng.normal(size=(row_count, k_cent))
    b_raw = b_cent_target @ null_basis.T

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
    beta_hat = result["beta"]
    assert isinstance(beta_hat, np.ndarray)

    constrained = constraints_matrix @ beta_hat
    assert float(np.max(np.abs(constrained))) < 1e-8


def test_centering_matrix_shapes_and_null_properties() -> None:
    degree = 3
    knots_x = np.array(
        [-2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        dtype=float,
    )

    constraints_matrix, null_basis = build_centering_matrix(
        knots_x=knots_x,
        degree=degree,
        x0=0.0,
        svd_tol=1e-12,
    )

    expected_k_raw = knots_x.size - degree - 1
    assert constraints_matrix.shape == (2, expected_k_raw)
    assert null_basis.shape == (expected_k_raw, expected_k_raw - 2)

    residual = constraints_matrix @ null_basis
    assert float(np.max(np.abs(residual))) < 1e-9

    gram = null_basis.T @ null_basis
    identity = np.eye(gram.shape[0], dtype=float)
    assert float(np.max(np.abs(gram - identity))) < 1e-9


def test_centering_constraints_hold_for_random_gamma() -> None:
    degree = 3
    x0 = 0.0
    knots_x = np.array(
        [-2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
        dtype=float,
    )

    constraints_matrix, null_basis = build_centering_matrix(
        knots_x=knots_x,
        degree=degree,
        x0=x0,
        svd_tol=1e-12,
    )

    gamma_dim = null_basis.shape[1]
    rng = np.random.default_rng(123)
    gamma = rng.normal(size=gamma_dim)

    beta = null_basis @ gamma

    # Проверяем ограничения через матрицу A напрямую
    constrained = constraints_matrix @ beta
    assert float(np.max(np.abs(constrained))) < 1e-9

    # И проверяем “физический смысл”: h(x0)=0 и h'(x0)=0 через BSpline
    coefficient_count = knots_x.size - degree - 1
    assert beta.shape == (coefficient_count,)

    spline = BSpline(knots_x, beta, degree, extrapolate=False)
    value_at_x0 = float(spline(x0))
    deriv_at_x0 = float(spline.derivative(1)(x0))

    assert abs(value_at_x0) < 1e-9
    assert abs(deriv_at_x0) < 1e-9
