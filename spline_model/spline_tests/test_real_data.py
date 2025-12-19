import copy
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
from trace_reference_builder import get_reference_log
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


def _real_data_config_with_lambda_method(model: MarathonModel, lambda_method: str) -> dict[str, Any]:
    config_copy: dict[str, Any] = copy.deepcopy(model.config)
    config_copy["age_spline_model"]["lambda_method"] = str(lambda_method)
    return config_copy


def _real_data_assert_reml_report_fields(fitted: Any) -> None:
    report = getattr(fitted, "fit_report", None)
    if not isinstance(report, dict):
        raise RuntimeError("real_data: fit_report must be dict")

    required_keys = ["lambda_method", "lambda_value", "edf", "nu", "sigma2_reml"]
    missing_keys = [key for key in required_keys if key not in report]
    if missing_keys:
        raise RuntimeError(f"real_data: fit_report missing keys: {missing_keys}")

    lambda_method = str(report["lambda_method"])
    if lambda_method.upper() != "REML":
        raise RuntimeError(f"real_data: fit_report.lambda_method must be REML, got {lambda_method}")

    lambda_value = float(report["lambda_value"])
    edf = float(report["edf"])
    nu = float(report["nu"])
    sigma2_reml = float(report["sigma2_reml"])

    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise RuntimeError(f"real_data: lambda_value must be finite and > 0, got {lambda_value}")
    if not np.isfinite(edf) or edf <= 0.0:
        raise RuntimeError(f"real_data: edf must be finite and > 0, got {edf}")
    if not np.isfinite(nu) or nu <= 3.0:
        raise RuntimeError(f"real_data: nu must be finite and > 3, got {nu}")
    if not np.isfinite(sigma2_reml) or sigma2_reml <= 0.0:
        raise RuntimeError(f"real_data: sigma2_reml must be finite and > 0, got {sigma2_reml}")



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
    -  beta (центрирование h(0)=0, h'(0)=0 через ограничения)
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    trace_references = getattr(model, "trace_references", None)
    fitter = AgeSplineFitter(config=model.config)

    fitted = fitter.fit_gender(
        gender_df=df_gender,
        gender="M",
        trace_references=trace_references
    )

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
        max_abs_constraint)

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
    Собирает минимальный фрейм для fit_gender: gender, age, Z, race_id.

    Контракт: колонки Z и race_id ОБЯЗАНЫ существовать в train_frame.
    race_id нужен для вычисления tau2_bar.
    """
    required_columns = ["gender", "age", "Z", "race_id"]  # ✅ ДОБАВЛЕН race_id
    missing_columns = [col for col in required_columns if col not in train_frame.columns]
    if missing_columns:
        raise RuntimeError(f"real_data: train_frame missing columns: {missing_columns}")

    df_gender = train_frame.loc[
        train_frame["gender"].astype(str) == str(gender),
        ["gender", "age", "Z", "race_id"],  # ✅ ДОБАВЛЕН race_id
    ].copy()

    if df_gender.empty:
        raise RuntimeError(f"real_data: no rows for gender={gender}")

    df_gender["age"] = df_gender["age"].astype(float)
    df_gender["Z"] = df_gender["Z"].astype(float)

    if not np.isfinite(df_gender["Z"].to_numpy(dtype=float)).all():
        raise RuntimeError("real_data: Z contains non-finite values")

    return df_gender
def test_real_data_predict_h_at_age_center_is_near_zero(model=None) -> None:
    """
    Проверяет контракт центрирования на реальных данных: h(age_center) ≈ 0.

    По построению h(0)=0 на шкале x, а x=0 соответствует age=age_center.
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)
    params = _real_data_get_age_params(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    fitter = AgeSplineFitter(config=model.config)
    fitted_model = fitter.fit_gender(gender_df=df_gender, gender="M")

    age_center = float(params["age_center"])
    h_at_center = fitted_model.predict_h(age_center)

    if not np.isfinite(h_at_center):
        raise RuntimeError(f"predict_h({age_center}) is not finite")

    # Допуск: должен быть очень близок к 0
    tol = 1e-8
    if abs(h_at_center) > tol:
        raise RuntimeError(
            f"predict_h(age_center={age_center}) should be ~0, got {h_at_center}"
        )

    logger.info("real_data predict_h(age_center=%s) = %s", age_center, h_at_center)


def test_real_data_predict_h_is_finite_across_age_range(model=None) -> None:
    """
    Проверяет что predict_h возвращает конечные значения по всему диапазону.
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)
    params = _real_data_get_age_params(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    fitter = AgeSplineFitter(config=model.config)
    fitted_model = fitter.fit_gender(gender_df=df_gender, gender="M")

    age_min = float(params["age_min_global"])
    age_max = float(params["age_max_global"])
    age_center = float(params["age_center"])

    # Тестируем на сетке возрастов
    test_ages = np.array([age_min, 25.0, age_center, 50.0, 65.0, age_max])
    h_values = fitted_model.predict_h(test_ages)

    if not isinstance(h_values, np.ndarray):
        raise RuntimeError("predict_h(array) should return ndarray")

    if h_values.shape != test_ages.shape:
        raise RuntimeError(f"predict_h output shape mismatch: {h_values.shape} vs {test_ages.shape}")

    if not np.isfinite(h_values).all():
        raise RuntimeError("predict_h returned non-finite values")

    logger.info(
        "real_data predict_h: ages=%s h_values=%s",
        test_ages.tolist(),
        h_values.tolist(),
    )


def test_real_data_predict_h_clamps_correctly(model=None) -> None:
    """
    Проверяет что predict_h корректно кламп-ит возраста вне границ.
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)
    params = _real_data_get_age_params(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    fitter = AgeSplineFitter(config=model.config)
    fitted_model = fitter.fit_gender(gender_df=df_gender, gender="M")

    age_min = float(params["age_min_global"])
    age_max = float(params["age_max_global"])

    # Возраст ниже минимума
    h_below = fitted_model.predict_h(10.0)
    h_at_min = fitted_model.predict_h(age_min)

    if abs(h_below - h_at_min) > 1e-12:
        raise RuntimeError(
            f"predict_h(10) should equal predict_h({age_min}), "
            f"got {h_below} vs {h_at_min}"
        )

    # Возраст выше максимума
    h_above = fitted_model.predict_h(100.0)
    h_at_max = fitted_model.predict_h(age_max)

    if abs(h_above - h_at_max) > 1e-12:
        raise RuntimeError(
            f"predict_h(100) should equal predict_h({age_max}), "
            f"got {h_above} vs {h_at_max}"
        )

    logger.info("real_data predict_h clamp: OK")


def test_real_data_predict_h_monotonic_after_peak(model=None) -> None:
    """
    Проверяет разумность формы h(age): после пика (обычно 25-35) должна расти.

    Это не строгий контракт, а sanity check. Для марафонцев ожидаем:
    - h(25) < h(50) < h(70) (время растёт с возрастом после пика)
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    fitter = AgeSplineFitter(config=model.config)
    fitted_model = fitter.fit_gender(gender_df=df_gender, gender="M")

    h_30 = fitted_model.predict_h(30.0)
    h_50 = fitted_model.predict_h(50.0)
    h_70 = fitted_model.predict_h(70.0)

    logger.info("real_data predict_h shape: h(30)=%s h(50)=%s h(70)=%s", h_30, h_50, h_70)

    # Sanity check: h должна расти после пика
    # (это не строгий тест, просто проверяем что форма разумная)
    if not (h_50 > h_30):
        logger.warning("predict_h shape warning: h(50) <= h(30), форма может быть необычной")

    if not (h_70 > h_50):
        logger.warning("predict_h shape warning: h(70) <= h(50), форма может быть необычной")


def test_real_data_fit_report_contains_required_fields(model=None) -> None:
    """
    Проверяет что fit_report содержит обязательные поля.
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    fitter = AgeSplineFitter(config=model.config)
    fitted_model = fitter.fit_gender(gender_df=df_gender, gender="M")

    fit_report = fitted_model.fit_report

    required_fields = ["n", "age_range_actual", "K_raw", "K_cent", "lambda_value", "degree", "lambda_method"]
    missing_fields = [f for f in required_fields if f not in fit_report]
    if missing_fields:
        raise RuntimeError(f"fit_report missing fields: {missing_fields}")

    n = int(fit_report["n"])
    if n <= 0:
        raise RuntimeError(f"fit_report['n'] should be > 0, got {n}")

    k_raw = int(fit_report["K_raw"])
    k_cent = int(fit_report["K_cent"])
    if k_cent != k_raw - 2:
        raise RuntimeError(f"K_cent should be K_raw - 2: {k_cent} vs {k_raw - 2}")

    lambda_value = float(fit_report["lambda_value"])
    if not np.isfinite(lambda_value):
        raise RuntimeError("fit_report['lambda_value'] must be finite")

    lambda_method = str(fit_report["lambda_method"]).upper()

    if lambda_method == "REML":
        required_reml = ["edf", "nu", "sigma2_reml"]
        missing_reml = [f for f in required_reml if f not in fit_report]
        if missing_reml:
            raise RuntimeError(f"fit_report missing REML fields: {missing_reml}")

        edf = float(fit_report["edf"])
        nu = float(fit_report["nu"])
        sigma2_reml = float(fit_report["sigma2_reml"])

        if lambda_value <= 0.0:
            raise RuntimeError(f"REML: lambda_value must be > 0, got {lambda_value}")
        if not np.isfinite(edf) or edf <= 0.0:
            raise RuntimeError(f"REML: edf invalid: {edf}")
        if not np.isfinite(nu) or nu <= 3.0:
            raise RuntimeError(f"REML: nu invalid: {nu}")
        if not np.isfinite(sigma2_reml) or sigma2_reml <= 0.0:
            raise RuntimeError(f"REML: sigma2_reml invalid: {sigma2_reml}")

    logger.info("real_data fit_report: n=%s K_raw=%s K_cent=%s", n, k_raw, k_cent)


def test_real_data_fit_gender_with_reml_selects_positive_lambda_and_preserves_centering(model=None) -> None:
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")
    config_reml = _real_data_config_with_lambda_method(model=model, lambda_method="REML")

    fitter = AgeSplineFitter(config=config_reml)
    fitted = fitter.fit_gender(gender_df=df_gender, gender="M")

    lambda_value = float(getattr(fitted, "lambda_value", float("nan")))
    if not np.isfinite(lambda_value) or lambda_value <= 0.0:
        raise RuntimeError(f"real_data: fitted.lambda_value must be finite and > 0, got {lambda_value}")

    basis_centering = fitted.basis_centering
    if "A" not in basis_centering:
        raise RuntimeError("real_data: basis_centering must contain key A")

    beta = fitted.coef_beta.to_numpy(dtype=float)
    constraints_matrix_A = np.asarray(basis_centering["A"], dtype=float)
    max_abs_constraint = float(np.max(np.abs(constraints_matrix_A @ beta)))

    tol = float(getattr(fitter, "centering_tol", 1e-10))
    if not np.isfinite(max_abs_constraint):
        raise RuntimeError("real_data: constraint check produced non-finite value")
    if max_abs_constraint > tol * 10.0:
        raise RuntimeError(
            "real_data: centering constraints violated "
            f"(max_abs_constraint={max_abs_constraint}, tol={tol})"
        )

    _real_data_assert_reml_report_fields(fitted=fitted)


def test_real_data_fit_both_genders_with_reml(model=None) -> None:
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    config_reml = _real_data_config_with_lambda_method(model=model, lambda_method="REML")
    fitter = AgeSplineFitter(config=config_reml)

    df_m = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")
    df_f = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="F")

    fitted_m = fitter.fit_gender(gender_df=df_m, gender="M")
    fitted_f = fitter.fit_gender(gender_df=df_f, gender="F")

    if float(fitted_m.lambda_value) <= 0.0 or not np.isfinite(float(fitted_m.lambda_value)):
        raise RuntimeError(f"real_data: M lambda_value invalid: {fitted_m.lambda_value}")
    if float(fitted_f.lambda_value) <= 0.0 or not np.isfinite(float(fitted_f.lambda_value)):
        raise RuntimeError(f"real_data: F lambda_value invalid: {fitted_f.lambda_value}")

    _real_data_assert_reml_report_fields(fitted=fitted_m)
    _real_data_assert_reml_report_fields(fitted=fitted_f)

def test_real_data_fit_both_genders(model=None) -> None:
    """
    Проверяет что AgeSplineFitter.fit() обучает модели для обоих полов.
    """
    if model is None:
        model = _real_data_build_model()
    train_frame = _real_data_get_train_frame(model)

    # Готовим z_frame с обоими полами
    required_columns = ["gender", "age", "Z"]
    missing_columns = [col for col in required_columns if col not in train_frame.columns]
    if missing_columns:
        raise RuntimeError(f"real_data: train_frame missing columns: {missing_columns}")

    z_frame = train_frame[["gender", "age", "Z"]].copy()
    z_frame["age"] = z_frame["age"].astype(float)
    z_frame["Z"] = z_frame["Z"].astype(float)

    trace_references = getattr(model, "trace_references", None)
    fitter = AgeSplineFitter(config=model.config)

    models = fitter.fit(z_frame, trace_references=trace_references)

    if not isinstance(models, dict):
        raise RuntimeError("fit() should return dict")

    # Проверяем что оба пола присутствуют
    for gender in ["M", "F"]:
        if gender not in models:
            raise RuntimeError(f"fit() did not return model for gender={gender}")

        gender_model = models[gender]

        if gender_model.coef_beta.empty:
            raise RuntimeError(f"fit() returned empty coef_beta for gender={gender}")

        if not np.isfinite(gender_model.coef_beta.to_numpy()).all():
            raise RuntimeError(f"fit() returned non-finite coef_beta for gender={gender}")

        # Проверяем центрирование
        h_at_center = gender_model.predict_h(35.0)
        if abs(h_at_center) > 1e-8:
            raise RuntimeError(
                f"fit() gender={gender}: h(35) should be ~0, got {h_at_center}"
            )

    logger.info(
        "real_data fit both genders: M_beta_len=%s F_beta_len=%s",
        models["M"].coef_beta.size,
        models["F"].coef_beta.size,
    )


def _real_data_pick_predict_case(model: MarathonModel) -> dict[str, Any]:
    train_frame = _real_data_get_train_frame(model)
    trace_references = getattr(model, "trace_references", None)
    age_models = getattr(model, "age_spline_models", None)

    if trace_references is None or len(trace_references) == 0:
        raise RuntimeError("real_data: trace_references is missing or empty")
    if age_models is None or len(age_models) == 0:
        raise RuntimeError("real_data: age_spline_models is missing or empty")

    required_columns = ["race_id", "gender", "age", "year"]
    missing_columns = [col for col in required_columns if col not in train_frame.columns]
    if missing_columns:
        raise RuntimeError(f"real_data: train_frame missing columns: {missing_columns}")

    max_rows = min(int(len(train_frame)), 5000)
    for row_index in range(max_rows):
        row = train_frame.iloc[row_index]
        race_id = str(row["race_id"])
        gender = str(row["gender"])
        age = float(row["age"])
        year = int(row["year"])

        if gender not in age_models:
            continue

        try:
            _ = get_reference_log(
                trace_references=trace_references,
                race_id=race_id,
                gender=gender,
                _year=year,
            )
        except KeyError:
            continue

        return {"race_id": race_id, "gender": gender, "age": age, "year": year}

    raise RuntimeError("real_data: could not pick a valid (race_id, gender) case")


def test_real_data_predict_log_time_equals_reference_plus_mean(model=None) -> None:
    """
       Интеграционный тест: проверяет что predict_log_time = reference_log + predict_mean.

       predict_mean включает μ + γ*x + h(x), то есть полную возрастную модель.
       """
    if model is None:
        model = _real_data_build_model()

    case = _real_data_pick_predict_case(model)
    race_id = str(case["race_id"])
    gender = str(case["gender"])
    age = float(case["age"])
    year = int(case["year"])

    trace_references = getattr(model, "trace_references", None)
    if trace_references is None:
        raise RuntimeError("real_data: trace_references is None")

    age_models = getattr(model, "age_spline_models", None)
    if age_models is None:
        raise RuntimeError("real_data: age_spline_models is None")

    reference_log = get_reference_log(
        trace_references=trace_references,
        race_id=race_id,
        gender=gender,
        _year=year,
    )
    mean_value = age_models[gender].predict_mean(age)
    expected = float(reference_log) + float(mean_value)

    predicted = model.predict_log_time(race_id=race_id, gender=gender, age=age, year=year)
    if not np.isfinite(float(predicted)):
        raise RuntimeError("real_data: predict_log_time returned non-finite")

    abs_err = abs(float(predicted) - expected)
    if abs_err > 1e-10:
        raise RuntimeError(f"real_data: predict_log_time mismatch: abs_err={abs_err}")

    ages = np.array([age - 1.0, age, age + 1.0], dtype=float)
    predicted_vec = model.predict_log_time(race_id=race_id, gender=gender, age=ages, year=year)
    if not isinstance(predicted_vec, np.ndarray):
        raise RuntimeError("real_data: predict_log_time(array) must return ndarray")
    if predicted_vec.shape != ages.shape:
        raise RuntimeError(f"real_data: predict_log_time(array) shape mismatch: {predicted_vec.shape}")
    if not np.isfinite(predicted_vec).all():
        raise RuntimeError("real_data: predict_log_time(array) returned non-finite")


def test_real_data_tau2_bar_is_computed(model=None) -> None:
    """
    Проверяет что tau2_bar вычисляется и имеет разумное значение.
    """
    if model is None:
        model = _real_data_build_model()

    age_models = getattr(model, "age_spline_models", None)
    if age_models is None or len(age_models) == 0:
        raise RuntimeError("real_data: age_spline_models is missing or empty")

    for gender in ["M", "F"]:
        if gender not in age_models:
            continue

        fitted_model = age_models[gender]
        tau2_bar = float(getattr(fitted_model, "tau2_bar", float("nan")))

        # Проверяем что tau2_bar конечен
        if not np.isfinite(tau2_bar):
            raise RuntimeError(f"real_data: tau2_bar for gender={gender} is not finite: {tau2_bar}")

        # Проверяем что tau2_bar неотрицателен
        if tau2_bar < 0.0:
            raise RuntimeError(f"real_data: tau2_bar for gender={gender} is negative: {tau2_bar}")

        logger.info(f"real_data tau2_bar: gender={gender}, tau2_bar={tau2_bar:.6f}")


def test_real_data_sigma2_use_is_computed(model=None) -> None:
    """
    Проверяет что sigma2_use вычисляется правильно.

    Контракт: sigma2_use = max(sigma2_floor, sigma2_reml - tau2_bar)
    """
    if model is None:
        model = _real_data_build_model()

    age_models = getattr(model, "age_spline_models", None)
    if age_models is None or len(age_models) == 0:
        raise RuntimeError("real_data: age_spline_models is missing or empty")

    config = model.config
    sigma2_floor = float(config["age_spline_model"].get("sigma2_floor", 1e-8))

    for gender in ["M", "F"]:
        if gender not in age_models:
            continue

        fitted_model = age_models[gender]

        sigma2_use = float(getattr(fitted_model, "sigma2_use", float("nan")))
        sigma2_reml = float(getattr(fitted_model, "sigma2_reml", float("nan")))
        tau2_bar = float(getattr(fitted_model, "tau2_bar", float("nan")))

        # Проверяем что sigma2_use конечен
        if not np.isfinite(sigma2_use):
            raise RuntimeError(f"real_data: sigma2_use for gender={gender} is not finite: {sigma2_use}")

        # Проверяем что sigma2_use >= sigma2_floor
        if sigma2_use < sigma2_floor:
            raise RuntimeError(
                f"real_data: sigma2_use for gender={gender} is less than floor: "
                f"sigma2_use={sigma2_use}, floor={sigma2_floor}"
            )

        # Проверяем контракт: sigma2_use = max(floor, sigma2_reml - tau2_bar)
        expected_sigma2_use = max(sigma2_floor, sigma2_reml - tau2_bar)

        if abs(sigma2_use - expected_sigma2_use) > 1e-12:
            raise RuntimeError(
                f"real_data: sigma2_use contract violated for gender={gender}: "
                f"sigma2_use={sigma2_use}, expected={expected_sigma2_use}, "
                f"sigma2_reml={sigma2_reml}, tau2_bar={tau2_bar}, floor={sigma2_floor}"
            )

        logger.info(
            f"real_data sigma2_use: gender={gender}, "
            f"sigma2_reml={sigma2_reml:.6f}, tau2_bar={tau2_bar:.6f}, "
            f"sigma2_use={sigma2_use:.6f}"
        )


def test_real_data_tau2_bar_with_missing_references(model=None) -> None:
    """
    Проверяет что модель работает когда trace_references=None.

    В этом случае tau2_bar должен быть 0.0.
    """
    if model is None:
        model = _real_data_build_model()

    train_frame = _real_data_get_train_frame(model)
    df_gender = _real_data_make_gender_fit_frame(train_frame=train_frame, gender="M")

    # Обучаем без trace_references
    fitter = AgeSplineFitter(config=model.config)
    fitted_model = fitter.fit_gender(
        gender_df=df_gender,
        gender="M",
        trace_references=None  # Явно передаём None
    )

    tau2_bar = float(getattr(fitted_model, "tau2_bar", float("nan")))

    # Проверяем что tau2_bar = 0.0 когда нет reference_variance
    if tau2_bar != 0.0:
        raise RuntimeError(
            f"real_data: tau2_bar should be 0.0 when trace_references=None, "
            f"got {tau2_bar}"
        )

    logger.info("real_data tau2_bar with missing references: OK (tau2_bar=0.0)")
# ============================================================================
# Точка входа
# ============================================================================

def test_real_data() -> None:
    """
    Точка входа для ручного прогона smoke-тестов.
    """

    model = _real_data_build_model()

    tests: list[tuple[str, callable]] = [
        # Базовые проверки пайплайна
        ("test_real_data_prepare_z_frame", test_real_data_prepare_z_frame),
        ("test_real_data_train_frame_has_Z", test_real_data_train_frame_has_Z),
        ("test_real_data_solve_penalized_lsq_runs_and_preserves_centering",
         test_real_data_solve_penalized_lsq_runs_and_preserves_centering),

        # fit_gender (FIXED по умолчанию)
        ("test_real_data_fit_gender_runs_and_preserves_centering",
         test_real_data_fit_gender_runs_and_preserves_centering),

        # fit_gender (REML)
        ("test_real_data_fit_gender_with_reml_selects_positive_lambda_and_preserves_centering",
         test_real_data_fit_gender_with_reml_selects_positive_lambda_and_preserves_centering),
        ("test_real_data_fit_both_genders_with_reml",
         test_real_data_fit_both_genders_with_reml),

        # predict_h (на FIXED)
        ("test_real_data_predict_h_at_age_center_is_near_zero",
         test_real_data_predict_h_at_age_center_is_near_zero),
        ("test_real_data_predict_h_is_finite_across_age_range",
         test_real_data_predict_h_is_finite_across_age_range),
        ("test_real_data_predict_h_clamps_correctly",
         test_real_data_predict_h_clamps_correctly),
        ("test_real_data_predict_h_monotonic_after_peak",
         test_real_data_predict_h_monotonic_after_peak),

        #  fit() и отчёт (самые интеграционные, ставим в конце)
        ("test_real_data_fit_both_genders", test_real_data_fit_both_genders),
        ("test_real_data_fit_report_contains_required_fields",
         test_real_data_fit_report_contains_required_fields),

        ('test_real_data_predict_log_time_equals_reference_plus_mean',
         test_real_data_predict_log_time_equals_reference_plus_mean),

        ("test_real_data_tau2_bar_is_computed",
         test_real_data_tau2_bar_is_computed),
        ("test_real_data_sigma2_use_is_computed",
         test_real_data_sigma2_use_is_computed),
        ("test_real_data_tau2_bar_with_missing_references",
         test_real_data_tau2_bar_with_missing_references),
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
    print("ALL REAL DATA TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_real_data()







