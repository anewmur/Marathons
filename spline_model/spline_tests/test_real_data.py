import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from main import MarathonModel, easy_logging
from spline_model.age_spline_fit import build_knots_x, AgeSplineFitter
from spline_model.age_spline_fit import build_raw_basis
from spline_model.build_centering_matrix import build_centering_matrix
from spline_model.age_spline_fit import build_second_difference_matrix
from spline_model.age_spline_fit import solve_penalized_lsq

logger = logging.getLogger(__name__)

def test_real_data_prepare_z_frame(model=None) -> None:
    """
    Интеграционный smoke-тест на реальных данных для подготовки шкалы x_std и B-сплайна.

    Проверяет:
    1) train_frame LOY + OK собран и содержит нужные колонки.
    2) Возраст в train_frame уже лежит в [age_min_global, age_max_global].
       Если нет, тест падает: значит clamp не реализован там, где должен быть.
    3) train_frame['x'] совпадает с пересчётом x_std = (age_clamped - age_center) / age_scale.
    4) build_knots_x и build_raw_basis работают в шкале x_std, и выполняется partition of unity.
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)
    params = _real_data_get_age_params(model)

    _real_data_check_train_frame_contract(train_frame)
    _real_data_check_loy_and_ok(train_frame, validation_year=int(model.validation_year))

    age_min_global = float(params["age_min_global"])
    age_max_global = float(params["age_max_global"])
    age_center = float(params["age_center"])
    age_scale = float(params["age_scale"])
    degree = int(params["degree"])
    max_inner_knots = int(params["max_inner_knots"])
    min_knot_gap = float(params["min_knot_gap"])

    _real_data_assert_ages_are_clamped(
        train_frame=train_frame,
        age_min_global=age_min_global,
        age_max_global=age_max_global,
    )

    x_std_recomputed = _real_data_recompute_x_std(
        train_frame=train_frame,
        age_center=age_center,
        age_scale=age_scale,
        age_min_global=age_min_global,
        age_max_global=age_max_global,
    )

    _real_data_compare_x_with_frame(
        train_frame=train_frame,
        x_std_recomputed=x_std_recomputed)

    knots_x = build_knots_x(
        x_values=x_std_recomputed,
        degree=degree,
        max_inner_knots=max_inner_knots,
        min_knot_gap=min_knot_gap,
    )

    _real_data_build_raw_basis_and_check_partition_of_unity(
        x_values=x_std_recomputed,
        knots_x=knots_x,
        degree=degree,
    )

def _real_data_build_model() -> MarathonModel:
    """
    Строит и прогоняет MarathonModel на реальных данных.

    Возвращает модель после run(), чтобы были заполнены train_frame LOY и нужные артефакты.
    """
    easy_logging(True)
    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()
    return model

def _real_data_get_train_frame(model: MarathonModel) -> pd.DataFrame:
    """
    Возвращает LOY train_frame, который используется для обучения возрастной модели.
    """
    train_frame = getattr(model, "train_frame_loy", None)
    if train_frame is None:
        train_frame = getattr(model, "age_spline_train_frame", None)

    if train_frame is None:
        raise RuntimeError("train_frame_loy/age_spline_train_frame is None after run()")

    return train_frame

def _real_data_get_age_params(model: MarathonModel) -> dict[str, Any]:
    """
    Достаёт параметры возраста и сплайна из конфига модели.
    """
    config = model.config
    preprocessing = config["preprocessing"]
    age_model = config["age_spline_model"]

    return {
        "age_center": float(preprocessing["age_center"]),
        "age_scale": float(preprocessing["age_scale"]),
        "age_min_global": float(age_model["age_min_global"]),
        "age_max_global": float(age_model["age_max_global"]),
        "degree": int(age_model["degree"]),
        "max_inner_knots": int(age_model["max_inner_knots"]),
        "min_knot_gap": float(age_model["min_knot_gap"]),
    }

def _real_data_assert_ages_are_clamped(
    train_frame: pd.DataFrame,
    age_min_global: float,
    age_max_global: float,
) -> None:
    """
    Проверяет, что возраст уже находится в [age_min_global, age_max_global].

    Если нет, тест падает: это значит, что clamp должен быть реализован раньше (в preprocess/add_features),
    иначе шкала x_std будет расходиться со спецификацией.
    """
    ages = train_frame["age"].astype(float)

    below_mask = ages < age_min_global
    above_mask = ages > age_max_global
    if below_mask.any() or above_mask.any():
        below_count = int(below_mask.sum())
        above_count = int(above_mask.sum())
        min_age = float(ages.min())
        max_age = float(ages.max())
        raise RuntimeError(
            "age is outside global bounds; implement clamp before x_std is computed. "
            f"age_min_global={age_min_global}, age_max_global={age_max_global}, "
            f"below_count={below_count}, above_count={above_count}, "
            f"age_range=[{min_age}, {max_age}]"
        )


def _real_data_recompute_x_std(
    train_frame: pd.DataFrame,
    age_center: float,
    age_scale: float,
    age_min_global: float,
    age_max_global: float,
) -> pd.Series:
    """
    Пересчитывает x_std по формуле x = (age_clamped - age_center) / age_scale,
    где age_clamped = clip(age, age_min_global, age_max_global).
    """
    if not (age_scale > 0.0):
        raise ValueError(f"age_scale must be > 0, got {age_scale}")
    if not (age_max_global > age_min_global):
        raise ValueError(
            "age_max_global must be > age_min_global, "
            f"got {age_min_global}, {age_max_global}"
        )

    ages_raw = train_frame["age"].astype(float)
    ages_clamped = ages_raw.clip(lower=float(age_min_global), upper=float(age_max_global))
    x_std = (ages_clamped - float(age_center)) / float(age_scale)
    return x_std.astype(float)


def _real_data_compare_x_with_frame(train_frame: pd.DataFrame, x_std_recomputed: pd.Series) -> None:
    """
    Сверяет train_frame['x'] и пересчитанный x_std.

    Это должно совпадать, если x в train_frame строится по той же формуле.
    """
    x_frame = train_frame["x"].astype(float)
    x_recomputed = x_std_recomputed.astype(float)

    diff = (x_frame - x_recomputed).abs()
    max_abs_diff = float(diff.max())

    if not np.isfinite(max_abs_diff):
        raise RuntimeError("x comparison produced non-finite max_abs_diff")

    if max_abs_diff > 1e-12:
        raise RuntimeError(
            "train_frame['x'] differs from recomputed x_std. "
            f"max_abs_diff={max_abs_diff}"
        )


def _real_data_build_raw_basis_and_check_partition_of_unity(
    x_values: pd.Series,
    knots_x: list[float],
    degree: int,
) -> None:
    """
    Строит сырой B-сплайновый базис и проверяет partition of unity.
    """
    x_np = x_values.to_numpy(dtype=float)

    t0 = time.time()
    b_raw = build_raw_basis(
        x_values=x_np,
        knots_x=knots_x,
        degree=int(degree),
    )
    elapsed = time.time() - t0

    row_sums = b_raw.sum(axis=1)
    max_abs_err = float(np.max(np.abs(row_sums - 1.0)))

    logger.debug("spline degree: %s", int(degree))
    logger.debug(
        "x_used_for_spline range: %s %s",
        float(np.min(x_np)),
        float(np.max(x_np)),
    )
    logger.debug(
        "knots_x range: %s %s",
        float(knots_x[0]),
        float(knots_x[-1]),
    )
    logger.info(
        "real_data build_raw_basis: shape=%s elapsed_sec=%s",
        (int(b_raw.shape[0]), int(b_raw.shape[1])),
        float(elapsed),
    )
    logger.info("real_data partition_of_unity max_abs_err: %s", float(max_abs_err))

    if max_abs_err > 1e-10:
        raise RuntimeError(f"partition of unity failed: max_abs_err={max_abs_err}")

def _real_data_check_train_frame_contract(train_frame: pd.DataFrame) -> None:
    """
    Проверяет минимальный контракт train_frame для возрастной части.
    """
    required_columns = ["race_id", "gender", "age", "time_seconds", "Y", "x", "year", "status"]
    missing_columns = [col for col in required_columns if col not in train_frame.columns]
    if missing_columns:
        raise RuntimeError(f"train_frame missing columns: {missing_columns}")

    if train_frame.empty:
        raise RuntimeError("train_frame is empty")

    if train_frame[required_columns].isna().any().any():
        na_share = train_frame[required_columns].isna().mean().to_dict()
        raise RuntimeError(f"train_frame has NA in required columns: {na_share}")


def _real_data_check_loy_and_ok(train_frame: pd.DataFrame, validation_year: int) -> None:
    """
    Проверяет, что train_frame содержит только status='OK' и не содержит validation_year.
    """
    if (train_frame["status"] != "OK").any():
        bad_count = int((train_frame["status"] != "OK").sum())
        raise RuntimeError(f"train_frame contains non-OK rows: {bad_count}")

    if validation_year > 0 and (train_frame["year"] == validation_year).any():
        bad_count = int((train_frame["year"] == validation_year).sum())
        raise RuntimeError(f"train_frame contains validation_year={validation_year} rows: {bad_count}")


def test_real_data_solve_penalized_lsq_runs_and_preserves_centering(model=None) -> None:
    """
    Интеграционный smoke-тест на реальных данных: solve_penalized_lsq в редуцированной параметризации.

    Проверяет:
    - сбор B_raw и (A, C)
    - сбор штрафа P = D.T @ D
    - решение существует (конечные числа)
    - A @ beta ~= 0
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)
    params = _real_data_get_age_params(model)

    _real_data_check_train_frame_contract(train_frame)
    _real_data_check_loy_and_ok(train_frame, validation_year=int(model.validation_year))

    age_center = float(params["age_center"])
    age_scale = float(params["age_scale"])
    age_min_global = float(params["age_min_global"])
    age_max_global = float(params["age_max_global"])
    degree = int(params["degree"])
    max_inner_knots = int(params["max_inner_knots"])
    min_knot_gap = float(params["min_knot_gap"])

    x_std_recomputed = _real_data_recompute_x_std(
        train_frame=train_frame,
        age_center=age_center,
        age_scale=age_scale,
        age_min_global=age_min_global,
        age_max_global=age_max_global,
    )

    knots_x = build_knots_x(
        x_values=x_std_recomputed,
        degree=degree,
        max_inner_knots=max_inner_knots,
        min_knot_gap=min_knot_gap,
    )

    x_np = x_std_recomputed.to_numpy(dtype=float)

    b_raw = build_raw_basis(
        x_values=x_np,
        knots_x=knots_x,
        degree=degree,
    )

    constraints_matrix, null_basis = build_centering_matrix(
        knots_x=np.asarray(knots_x, dtype=float),
        degree=degree,
        x0=0.0,
        svd_tol=1e-12,
    )

    k_raw = int(b_raw.shape[1])

    difference_matrix = build_second_difference_matrix(coefficient_count=k_raw)
    penalty_raw = difference_matrix.T @ difference_matrix

    y = train_frame["Y"].to_numpy(dtype=float)

    lambda_value = 1.0

    result = solve_penalized_lsq(
        b_raw=b_raw,
        y=y,
        null_basis=null_basis,
        penalty_matrix_raw=penalty_raw,
        lambda_value=lambda_value,
    )

    gamma = result["gamma"]
    beta = result["beta"]
    fitted = result["fitted"]
    rss = float(result["rss"])

    if not isinstance(gamma, np.ndarray):
        raise RuntimeError("gamma is not ndarray")
    if not isinstance(beta, np.ndarray):
        raise RuntimeError("beta is not ndarray")
    if not isinstance(fitted, np.ndarray):
        raise RuntimeError("fitted is not ndarray")

    if gamma.ndim != 1:
        raise RuntimeError(f"gamma must be 1D, got {gamma.ndim}")
    if beta.ndim != 1:
        raise RuntimeError(f"beta must be 1D, got {beta.ndim}")
    if fitted.ndim != 1:
        raise RuntimeError(f"fitted must be 1D, got {fitted.ndim}")

    if beta.shape[0] != k_raw:
        raise RuntimeError(f"beta length mismatch: expected {k_raw}, got {beta.shape[0]}")
    if fitted.shape[0] != y.shape[0]:
        raise RuntimeError(f"fitted length mismatch: expected {y.shape[0]}, got {fitted.shape[0]}")

    if not np.isfinite(rss):
        raise RuntimeError("rss is not finite")

    max_abs_constraint = float(np.max(np.abs(constraints_matrix @ beta)))
    logger.info("real_data solve_penalized_lsq: rss=%s max_abs_constraint=%s", rss, max_abs_constraint)

    if max_abs_constraint > 1e-8:
        raise RuntimeError(f"centering constraints violated: max_abs_constraint={max_abs_constraint}")


def test_real_data_fit_gender_runs_and_preserves_centering(model=None) -> None:
    """
    Интеграционный smoke-тест на реальных данных:
    строим train_frame LOY, формируем df (gender, age, z),
    обучаем AgeSplineFitter.fit_gender и проверяем центрирование.

    Проверяет:
    - модель создаётся
    - coef_beta конечна
    - A @ beta ≈ 0 (центрирование h(0)=0, h'(0)=0 через ограничения)
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    fitter = AgeSplineFitter(config=model.config)
    fitted = fitter.fit_gender(gender_df=df_gender, gender="M")

    beta_series = fitted.coef_beta
    beta = beta_series.to_numpy(dtype=float)

    if beta.size <= 2:
        raise RuntimeError("real_data: coef_beta has too few coefficients")

    if not np.isfinite(beta).all():
        raise RuntimeError("real_data: coef_beta contains non-finite values")

    basis_centering = fitted.basis_centering
    if "A" not in basis_centering or "C" not in basis_centering:
        raise RuntimeError("real_data: basis_centering must contain keys A and C")

    constraints_matrix_A = np.asarray(basis_centering["A"], dtype=float)
    max_abs_constraint = float(np.max(np.abs(constraints_matrix_A @ beta)))

    logger.info(
        "real_data fit_gender: beta_len=%s max_abs_constraint=%s",
        int(beta.size),
        max_abs_constraint,
    )

    if not np.isfinite(max_abs_constraint):
        raise RuntimeError("real_data: constraint check produced non-finite value")

    tol = float(getattr(fitter, "centering_tol", 1e-10))
    if max_abs_constraint > tol * 10.0:
        raise RuntimeError(
            "real_data: centering constraints violated "
            f"(max_abs_constraint={max_abs_constraint}, tol={tol})"
        )

def test_real_data_train_frame_has_Z(model=None) -> None:
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    if "Z" not in train_frame.columns:
        raise RuntimeError("real_data: train_frame_loy must contain column 'Z'")

    z_values = train_frame["Z"].astype(float)
    if not np.isfinite(z_values.to_numpy(dtype=float)).all():
        raise RuntimeError("real_data: train_frame_loy['Z'] contains non-finite values")

def _real_data_make_gender_fit_frame(train_frame: pd.DataFrame, gender: str) -> pd.DataFrame:
    """
    Собирает минимальный фрейм для fit_gender: gender, age, z.

    Контракт: колонка Z ОБЯЗАНА существовать в train_frame.
    Если её нет, это ошибка пайплайна (нужно чинить код, а не тест).
    """
    required_columns = ["gender", "age", "Z"]
    missing_columns = [col for col in required_columns if col not in train_frame.columns]
    if missing_columns:
        raise RuntimeError(f"real_data: train_frame missing columns: {missing_columns}")

    df_gender = train_frame.loc[
        train_frame["gender"].astype(str) == str(gender),
        ["gender", "age", "Z"],
    ].copy()

    if df_gender.empty:
        raise RuntimeError(f"real_data: no rows for gender={gender}")

    if df_gender.isna().any().any():
        na_share = df_gender.isna().mean().to_dict()
        raise RuntimeError(f"real_data: NA in gender fit frame: {na_share}")

    df_gender["age"] = df_gender["age"].astype(float)
    df_gender["z"] = df_gender["Z"].astype(float)
    df_gender = df_gender.drop(columns=["Z"])

    if not np.isfinite(df_gender["z"].to_numpy(dtype=float)).all():
        raise RuntimeError("real_data: z contains non-finite values")

    return df_gender


def test_real_data() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """

    model = _real_data_build_model()

    tests: list[tuple[str, callable]] = [
        ("test_real_data_prepare_z_frame", test_real_data_prepare_z_frame),
        ("test_real_data_solve_penalized_lsq_runs_and_preserves_centering",
         test_real_data_solve_penalized_lsq_runs_and_preserves_centering),
        ("test_real_data_train_frame_has_Z", test_real_data_train_frame_has_Z),
        ("test_real_data_fit_gender_runs_and_preserves_centering",
         test_real_data_fit_gender_runs_and_preserves_centering),
    ]

    for test_name, test_fn in tests:
        print(f"\n== Running: {test_name}")
        try:
            test_fn(model=model)
            print(f"PASSED: {test_name}")
        except Exception as e:
            print(f"FAILED: {test_name}")
            print(f"Error: {e}")
            raise

    print("\n" + "=" * 60)
    print("ALL REAL TESTS PASSED")
    print("=" * 60)

