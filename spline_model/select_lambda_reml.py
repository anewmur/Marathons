import math

import numpy as np
from scipy.optimize import minimize_scalar


def _build_penalty_root(
    penalty_matrix: np.ndarray,
    eig_tol: float,
) -> tuple[np.ndarray, dict[str, any]]:
    """
    Строит L так, что L^T L = penalty_matrix (в численном смысле).
    Делает: eigh, отбрасывает собственные значения <= eig_tol.
    """
    penalty_array = np.asarray(penalty_matrix, dtype=float)
    if penalty_array.ndim != 2 or penalty_array.shape[0] != penalty_array.shape[1]:
        raise ValueError("_build_penalty_root: penalty_matrix must be square 2D")

    eigvals, eigvecs = np.linalg.eigh(penalty_array)

    kept_mask = eigvals > float(eig_tol)
    kept_count = int(np.sum(kept_mask))

    if kept_count == 0:
        L = np.zeros((0, int(penalty_array.shape[0])), dtype=float)
        info = {
            "rank": 0,
            "eigvals_kept": np.asarray([], dtype=float),
            "eigvals_dropped": np.asarray(eigvals, dtype=float),
        }
        return L, info

    eigvals_kept = eigvals[kept_mask]
    eigvecs_kept = eigvecs[:, kept_mask]

    root_diag = np.sqrt(eigvals_kept)
    L = (root_diag[:, None] * eigvecs_kept.T)

    info = {
        "rank": kept_count,
        "eigvals_kept": np.asarray(eigvals_kept, dtype=float),
        "eigvals_dropped": np.asarray(eigvals[~kept_mask], dtype=float),
    }
    return L, info


def _compute_reml_terms_for_lambda(
    W: np.ndarray,
    y: np.ndarray,
    penalty_root_L: np.ndarray,
    lambda_value: float,
    r_diag_tol: float,
) -> dict[str, any]:
    """
    Схема как в latex.

    Расширенная система:
      [W; sqrt(lambda) L] theta ≈ [y; 0]

    QR: W_tilde = Q R

    EDF:
      (R^T) M = W^T
      edf = ||M||_F^2

    REML:
      nu = n - edf
      sigma2_hat = RSS / nu
      reml = nu * log(sigma2_hat) + 2 * sum(log(|diag(R)|))

    Фильтр вырождения:
      min(|diag(R)|) < r_diag_tol -> invalid
    """
    W_array = np.asarray(W, dtype=float)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    L_array = np.asarray(penalty_root_L, dtype=float)

    if W_array.ndim != 2:
        raise ValueError("_compute_reml_terms_for_lambda: W must be 2D")
    if y_array.ndim != 1:
        raise ValueError("_compute_reml_terms_for_lambda: y must be 1D")
    if W_array.shape[0] != y_array.shape[0]:
        raise ValueError("_compute_reml_terms_for_lambda: W row count must match y length")
    if L_array.ndim != 2:
        raise ValueError("_compute_reml_terms_for_lambda: L must be 2D")
    if L_array.shape[1] != W_array.shape[1]:
        raise ValueError("_compute_reml_terms_for_lambda: L column count must match W columns")
    if not np.isfinite(W_array).all():
        raise ValueError("_compute_reml_terms_for_lambda: W contains non-finite values")
    if not np.isfinite(y_array).all():
        raise ValueError("_compute_reml_terms_for_lambda: y contains non-finite values")
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise ValueError("_compute_reml_terms_for_lambda: lambda_value must be finite and > 0")

    n_obs = int(W_array.shape[0])
    sqrt_lambda = math.sqrt(float(lambda_value))

    W_tilde = np.vstack([W_array, sqrt_lambda * L_array])
    y_tilde = np.concatenate([y_array, np.zeros((int(L_array.shape[0]),), dtype=float)])

    Q_mat, R_mat = np.linalg.qr(W_tilde, mode="reduced")
    diag_R = np.diag(R_mat)
    min_abs_diag_R = float(np.min(np.abs(diag_R))) if diag_R.size > 0 else 0.0

    if min_abs_diag_R < float(r_diag_tol):
        return {"ok": False, "reason": "R_diag_too_small", "min_abs_diag_R": min_abs_diag_R}

    theta_hat = np.linalg.solve(R_mat, Q_mat.T @ y_tilde)

    fitted = W_array @ theta_hat
    residual = y_array - fitted
    rss = float(residual.T @ residual)

    M_mat = np.linalg.solve(R_mat.T, W_array.T)
    edf = float(np.sum(M_mat * M_mat))

    nu = float(n_obs) - float(edf)

    if not np.isfinite(rss) or not np.isfinite(edf) or not np.isfinite(nu):
        return {"ok": False, "reason": "non_finite_terms", "min_abs_diag_R": min_abs_diag_R}

    if nu <= 0.0:
        return {
            "ok": False,
            "reason": "nu_non_positive",
            "min_abs_diag_R": min_abs_diag_R,
            "edf": edf,
            "nu": nu,
            "rss": rss,
        }

    sigma2_hat = float(rss / nu)
    if sigma2_hat <= 0.0 or not np.isfinite(sigma2_hat):
        return {
            "ok": False,
            "reason": "sigma2_invalid",
            "min_abs_diag_R": min_abs_diag_R,
            "edf": edf,
            "nu": nu,
            "rss": rss,
        }

    log_det_term = float(2.0 * np.sum(np.log(np.abs(diag_R))))
    reml_value = float(nu * math.log(sigma2_hat) + log_det_term)

    return {
        "ok": True,
        "reml": reml_value,
        "rss": rss,
        "edf": edf,
        "nu": nu,
        "sigma2_hat": sigma2_hat,
        "min_abs_diag_R": min_abs_diag_R,
        "theta_hat": np.asarray(theta_hat, dtype=float),
    }


class _RemlObjective:
    def __init__(
        self,
        W: np.ndarray,
        y: np.ndarray,
        penalty_root_L: np.ndarray,
        r_diag_tol: float,
        nu_min: float,
        max_edf_gap: float,
    ) -> None:
        self.W = np.asarray(W, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        self.L = np.asarray(penalty_root_L, dtype=float)

        self.r_diag_tol = float(r_diag_tol)
        self.nu_min = float(nu_min)
        self.max_edf_gap = float(max_edf_gap)

        self.eval_count = 0

    def __call__(self, eta_log10: float) -> float:
        self.eval_count += 1

        lambda_value = float(10.0 ** float(eta_log10))
        terms = _compute_reml_terms_for_lambda(
            W=self.W,
            y=self.y,
            penalty_root_L=self.L,
            lambda_value=lambda_value,
            r_diag_tol=self.r_diag_tol,
        )
        if not bool(terms.get("ok")):
            return float("inf")

        edf = float(terms["edf"])
        nu = float(terms["nu"])
        n_obs = float(self.W.shape[0])

        if nu <= self.nu_min:
            return float("inf")
        if edf > n_obs - self.max_edf_gap:
            return float("inf")

        return float(terms["reml"])


def select_lambda_reml(
    W: np.ndarray,
    y: np.ndarray,
    penalty_matrix: np.ndarray,
    eta_min: float = -8.0,
    eta_max: float = 8.0,
    eig_tol: float = 1e-12,
    r_diag_tol: float = 1e-12,
    nu_min: float = 3.0,
    max_edf_gap: float = 1.0,
    grid_points: int = 41,
) -> tuple[float, dict[str, any]]:
    """
    Подбор lambda по REML как в latex.

    Оптимизируем eta = log10(lambda) на [eta_min, eta_max] методом bounded Brent.
    Недопустимые значения дают +inf (вырождение R, nu <= nu_min, edf > n - max_edf_gap).
    """
    W_array = np.asarray(W, dtype=float)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    penalty_array = np.asarray(penalty_matrix, dtype=float)

    if W_array.ndim != 2:
        raise ValueError("select_lambda_reml: W must be 2D")
    if y_array.ndim != 1:
        raise ValueError("select_lambda_reml: y must be 1D")
    if W_array.shape[0] != y_array.shape[0]:
        raise ValueError("select_lambda_reml: W row count must match y length")
    if penalty_array.ndim != 2 or penalty_array.shape[0] != penalty_array.shape[1]:
        raise ValueError("select_lambda_reml: penalty_matrix must be square 2D")
    if penalty_array.shape[0] != W_array.shape[1]:
        raise ValueError("select_lambda_reml: penalty_matrix shape must match W column count")
    if not np.isfinite(W_array).all():
        raise ValueError("select_lambda_reml: W contains non-finite values")
    if not np.isfinite(y_array).all():
        raise ValueError("select_lambda_reml: y contains non-finite values")

    penalty_root_L, root_info = _build_penalty_root(penalty_matrix=penalty_array, eig_tol=float(eig_tol))

    objective = _RemlObjective(
        W=W_array,
        y=y_array,
        penalty_root_L=penalty_root_L,
        r_diag_tol=float(r_diag_tol),
        nu_min=float(nu_min),
        max_edf_gap=float(max_edf_gap),
    )

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(float(eta_min), float(eta_max)),
        options={"xatol": 1e-3},
    )

    best_eta = float(result.x)
    best_lambda = float(10.0 ** best_eta)

    best_terms = _compute_reml_terms_for_lambda(
        W=W_array,
        y=y_array,
        penalty_root_L=penalty_root_L,
        lambda_value=best_lambda,
        r_diag_tol=float(r_diag_tol),
    )

    eta_grid = np.linspace(float(eta_min), float(eta_max), int(grid_points))
    reml_grid: list[float] = []
    edf_grid: list[float] = []
    nu_grid: list[float] = []
    rss_grid: list[float] = []

    n_obs = float(W_array.shape[0])

    for eta_value in eta_grid.tolist():
        lambda_value = float(10.0 ** float(eta_value))
        terms = _compute_reml_terms_for_lambda(
            W=W_array,
            y=y_array,
            penalty_root_L=penalty_root_L,
            lambda_value=lambda_value,
            r_diag_tol=float(r_diag_tol),
        )
        if not bool(terms.get("ok")):
            reml_grid.append(float("inf"))
            edf_grid.append(float("nan"))
            nu_grid.append(float("nan"))
            rss_grid.append(float("nan"))
            continue

        edf = float(terms["edf"])
        nu = float(terms["nu"])

        if nu <= float(nu_min) or edf > n_obs - float(max_edf_gap):
            reml_grid.append(float("inf"))
            edf_grid.append(edf)
            nu_grid.append(nu)
            rss_grid.append(float(terms["rss"]))
            continue

        reml_grid.append(float(terms["reml"]))
        edf_grid.append(edf)
        nu_grid.append(nu)
        rss_grid.append(float(terms["rss"]))

    info: dict[str, any] = {
        "best_eta": best_eta,
        "best_lambda": best_lambda,
        "best_terms": best_terms,
        "eta_grid": np.asarray(eta_grid, dtype=float),
        "reml_grid": np.asarray(reml_grid, dtype=float),
        "edf_grid": np.asarray(edf_grid, dtype=float),
        "nu_grid": np.asarray(nu_grid, dtype=float),
        "rss_grid": np.asarray(rss_grid, dtype=float),
        "penalty_root": root_info,
        "minimize_success": bool(result.success),
        "minimize_message": str(result.message),
        "eval_count": int(objective.eval_count),
    }
    return best_lambda, info
