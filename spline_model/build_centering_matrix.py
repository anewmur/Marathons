

import numpy as np
from scipy.interpolate import BSpline


def build_centering_matrix(
    knots_x: np.ndarray,
    degree: int,
    x0: float = 0.0,
    *,
    svd_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Построить матрицу ограничений A и базис нулевого пространства C для центрирования.

    Ограничения:
        A[0, :] @ beta = 0  <=>  h(x0) = 0
        A[1, :] @ beta = 0  <=>  h'(x0) = 0
    где h(x) = sum_j beta_j * B_j(x), B_j - B-сплайны степени degree на knots_x.

    Возвращает:
        A: shape (2, K_raw)
        C: shape (K_raw, K_raw - 2), ортонормированный базис null(A)
           (то есть A @ C = 0 и C.T @ C = I)

    Требования:
        - knots_x строго неубывающий, длина >= degree + 2
        - x0 лежит внутри [knots_x[degree], knots_x[-degree-1]] (внутренний носитель)
    """
    knots_array = _prepare_knots_array(knots_x)
    _validate_degree(degree)
    _validate_x0_inside_support(knots_array, degree, x0)

    basis_value_row = _evaluate_basis_row_at_x(knots_array, degree, x0)
    basis_derivative_row = _evaluate_basis_derivative_row_at_x(knots_array, degree, x0)

    constraints_matrix = np.vstack([basis_value_row, basis_derivative_row])
    null_basis = _compute_null_space_basis(constraints_matrix, svd_tol=svd_tol)

    _validate_centering_outputs(constraints_matrix, null_basis, svd_tol=svd_tol)

    return constraints_matrix, null_basis


def _prepare_knots_array(knots_x: np.ndarray) -> np.ndarray:
    knots_array = np.asarray(knots_x, dtype=float)
    if knots_array.ndim != 1:
        raise ValueError(f"knots_x must be 1D, got shape={knots_array.shape}")
    if knots_array.size < 4:
        raise ValueError(f"knots_x is too short, size={knots_array.size}")
    if np.any(np.isnan(knots_array)):
        raise ValueError("knots_x contains NaN")
    if np.any(np.diff(knots_array) < 0.0):
        raise ValueError("knots_x must be nondecreasing")
    return knots_array


def _validate_degree(degree: int) -> None:
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")


def _validate_x0_inside_support(knots_array: np.ndarray, degree: int, x0: float) -> None:
    if not np.isfinite(x0):
        raise ValueError(f"x0 must be finite, got {x0}")

    left_support = float(knots_array[degree])
    right_support = float(knots_array[-degree - 1])

    if x0 < left_support or x0 > right_support:
        raise ValueError(
            "x0 must be inside spline support, "
            f"got x0={x0}, support=[{left_support}, {right_support}]"
        )


def _evaluate_basis_row_at_x(knots_array: np.ndarray, degree: int, x_value: float) -> np.ndarray:
    coefficient_count = knots_array.size - degree - 1
    if coefficient_count <= 0:
        raise ValueError(
            f"Invalid knots/degree: knots={knots_array.size}, degree={degree}"
        )

    basis_row = np.zeros(coefficient_count, dtype=float)
    for basis_index in range(coefficient_count):
        coefficient_vector = np.zeros(coefficient_count, dtype=float)
        coefficient_vector[basis_index] = 1.0
        spline = BSpline(knots_array, coefficient_vector, degree, extrapolate=False)

        evaluated = spline(x_value)
        if np.isnan(evaluated):
            raise ValueError("BSpline evaluation produced NaN")
        basis_row[basis_index] = float(evaluated)

    return basis_row


def _evaluate_basis_derivative_row_at_x(knots_array: np.ndarray, degree: int, x_value: float) -> np.ndarray:
    coefficient_count = knots_array.size - degree - 1
    derivative_order = 1

    derivative_row = np.zeros(coefficient_count, dtype=float)
    for basis_index in range(coefficient_count):
        coefficient_vector = np.zeros(coefficient_count, dtype=float)
        coefficient_vector[basis_index] = 1.0
        spline = BSpline(knots_array, coefficient_vector, degree, extrapolate=False)

        spline_derivative = spline.derivative(derivative_order)
        evaluated = spline_derivative(x_value)
        if np.isnan(evaluated):
            raise ValueError("BSpline derivative evaluation produced NaN")
        derivative_row[basis_index] = float(evaluated)

    return derivative_row


def _compute_null_space_basis(constraints_matrix: np.ndarray, *, svd_tol: float) -> np.ndarray:
    if constraints_matrix.ndim != 2:
        raise ValueError("constraints_matrix must be 2D")
    if constraints_matrix.shape[0] != 2:
        raise ValueError(f"constraints_matrix must have 2 rows, got {constraints_matrix.shape}")

    left_singular_vectors, singular_values, right_singular_vectors_transposed = np.linalg.svd(
        constraints_matrix, full_matrices=True
    )
    del left_singular_vectors

    if singular_values.size != 2:
        raise ValueError("Unexpected SVD output for 2-row constraint matrix")

    max_singular = float(np.max(singular_values))
    if max_singular == 0.0:
        raise ValueError("Constraint matrix is all zeros, cannot define null space")

    threshold = svd_tol * max_singular
    rank = int(np.sum(singular_values > threshold))

    column_count = constraints_matrix.shape[1]
    if rank > 2:
        raise ValueError(f"Rank cannot exceed 2 for two constraints, got rank={rank}")
    if column_count - rank <= 0:
        raise ValueError("Null space dimension is zero, constraints overconstrain basis")

    right_singular_vectors = right_singular_vectors_transposed.T
    null_space_columns = right_singular_vectors[:, rank:]

    null_space_columns = _canonicalize_column_signs(null_space_columns)
    return null_space_columns


def _canonicalize_column_signs(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")

    fixed = matrix.copy()
    for column_index in range(fixed.shape[1]):
        column = fixed[:, column_index]
        max_abs_index = int(np.argmax(np.abs(column)))
        sign = float(np.sign(column[max_abs_index]))
        if sign == 0.0:
            continue
        if sign < 0.0:
            fixed[:, column_index] = -fixed[:, column_index]
    return fixed


def _validate_centering_outputs(
    constraints_matrix: np.ndarray,
    null_basis: np.ndarray,
    *,
    svd_tol: float,
) -> None:
    if constraints_matrix.shape[0] != 2:
        raise ValueError("constraints_matrix must have shape (2, K)")
    if null_basis.ndim != 2:
        raise ValueError("null_basis must be 2D")
    if null_basis.shape[0] != constraints_matrix.shape[1]:
        raise ValueError("null_basis row count must match constraint columns")

    residual = constraints_matrix @ null_basis
    residual_max = float(np.max(np.abs(residual)))
    if residual_max > 1e3 * svd_tol:
        raise ValueError(f"A @ C is not close to zero, max_abs={residual_max}")

    gram = null_basis.T @ null_basis
    identity = np.eye(gram.shape[0], dtype=float)
    gram_err = float(np.max(np.abs(gram - identity)))
    if gram_err > 1e3 * svd_tol:
        raise ValueError(f"C columns are not orthonormal enough, max_abs={gram_err}")
