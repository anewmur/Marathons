"""
Тест predict_with_uncertainty на реальных данных.

Зачем:
1) Проверить, что интервалы в log-шкале имеют разумную ширину.
2) Проверить покрытие: доля наблюдений, попавших в свой (age-specific) 95% интервал.
3) Сравнить эмпирическую дисперсию остатков в log-шкале с sigma2_total модели.
4) Диагностика: что именно оценивается в REML (размер LOY-выборки, число трасс).
5) Диагностика: оценка дисперсии остатков по трассам (race-specific), shrinkage к глобальной.

Важно:
- Проверка делается по наблюдениям: для каждого старта берём его age и строим свой интервал.
- "observations" это подтаблица df_clean после фильтров (race_id, gender, year опционально).
- Для LOY-диагностики используем model.train_frame_loy и model.trace_references.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from MarathonAgeModel import MarathonModel
from age_reference_builder import seconds_to_hhmmss
from spline_model.z_frame import build_z_frame


def _as_1d_float_array(values: object) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=float).reshape(-1)
    if isinstance(values, (list, tuple)):
        return np.asarray(values, dtype=float).reshape(-1)
    return np.asarray([float(values)], dtype=float)


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise RuntimeError(f"Required columns missing: {missing}")


def _format_debug_block(lines: list[str]) -> str:
    return "\n".join(lines)


def _raise_failed(check_name: str, details: str, debug_block: str) -> None:
    message = f"{check_name} FAILED: {details}"
    if debug_block.strip():
        message = message + "\n\nDEBUG:\n" + debug_block
    raise RuntimeError(message)


def _select_observations(
    df: pd.DataFrame,
    race_id: str,
    gender: str,
    year: int | None,
    age_center: float | None,
    age_tolerance: float,
) -> pd.DataFrame:
    _require_columns(df=df, required=["race_id", "gender", "age", "time_seconds"])

    mask = (df["race_id"] == race_id) & (df["gender"] == gender)

    if year is not None and "year" in df.columns:
        mask &= (df["year"] == year)

    base = df.loc[mask].copy()
    if len(base) == 0:
        return base

    base["age"] = pd.to_numeric(base["age"], errors="coerce")
    base["time_seconds"] = pd.to_numeric(base["time_seconds"], errors="coerce")
    base = base.loc[base["age"].notna() & base["time_seconds"].notna()]
    base = base.loc[base["time_seconds"] > 0]

    if len(base) == 0:
        return base

    if age_center is None:
        return base

    age_low = float(age_center - age_tolerance)
    age_high = float(age_center + age_tolerance)

    narrowed = base.loc[(base["age"] >= age_low) & (base["age"] <= age_high)].copy()
    if len(narrowed) > 0:
        return narrowed

    return base


def _sample_variance_ddof1(values: pd.Series) -> float:
    array_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    count = int(array_values.shape[0])
    if count <= 1:
        return float("nan")
    return float(np.var(array_values, ddof=1))


def _compute_coverage(times_real: np.ndarray, time_lower: np.ndarray, time_upper: np.ndarray) -> float:
    valid_mask = np.isfinite(times_real) & np.isfinite(time_lower) & np.isfinite(time_upper)
    if int(np.sum(valid_mask)) == 0:
        return float("nan")
    inside = (times_real[valid_mask] >= time_lower[valid_mask]) & (times_real[valid_mask] <= time_upper[valid_mask])
    return float(np.mean(inside))


def run_test_predict_uncertainty_real_data(model=None) -> None:
    race_id = "Белые ночи"
    gender = "M"

    year_test: int | None = None
    age_center: float | None = 40.0
    age_tolerance = 0.0

    confidence = 0.95
    method = "analytical"
    if model is None:
        model = MarathonModel(
            data_path=r"C:\Users\andre\github\Marathons\Data",
            verbose=True,
        )
        model.run()

    if not hasattr(model, "df_clean"):
        raise RuntimeError("model.df_clean is missing")
    if not hasattr(model, "trace_references") or model.trace_references is None:
        raise RuntimeError("model.trace_references is missing")
    if not hasattr(model, "train_frame_loy") or model.train_frame_loy is None:
        raise RuntimeError("model.train_frame_loy is missing")

    df = model.df_clean.copy()
    observations = _select_observations(
        df=df,
        race_id=race_id,
        gender=gender,
        year=year_test,
        age_center=age_center,
        age_tolerance=age_tolerance,
    )
    if len(observations) == 0:
        raise RuntimeError("No observations after filters")

    ages_obs = observations["age"].astype(float).to_numpy(dtype=float)
    times_real = observations["time_seconds"].astype(float).to_numpy(dtype=float)
    log_times_real = np.log(times_real)

    # Прогноз по каждому наблюдению (age-specific)
    year_value = int(year_test) if year_test is not None else 0
    pred = model.predict_with_uncertainty(
        race_id=race_id,
        gender=gender,
        age=ages_obs,
        year=year_value,
        confidence=confidence,
        method=method,
    )

    time_lower = _as_1d_float_array(pred["time_lower"])
    time_upper = _as_1d_float_array(pred["time_upper"])
    log_pred = _as_1d_float_array(pred["log_pred"])

    sigma2_use = float(pred["sigma2_use"])
    reference_variance = float(pred["reference_variance"])
    sigma2_total = float(pred["sigma2_total"])

    # Собираем “ключевой лог” (но НЕ печатаем)
    debug_lines: list[str] = []
    debug_lines.append(f"params: race_id={race_id}, gender={gender}, year_test={year_test}, age_center={age_center}, tol={age_tolerance}")
    debug_lines.append(f"obs: n={int(times_real.shape[0])}, time_median={float(np.median(times_real)):.1f}s ({seconds_to_hhmmss(float(np.median(times_real)))})")
    debug_lines.append(f"obs: var(log_time)={float(np.var(log_times_real)):.6f}")
    debug_lines.append(f"pred: sigma2_use={sigma2_use:.6f}, reference_variance={reference_variance:.6f}, sigma2_total={sigma2_total:.6f}")

    # CHECK 1: сигмы и дисперсии конечные и неотрицательные
    if (not np.isfinite(sigma2_use)) or sigma2_use < 0.0:
        _raise_failed("sigma2_use", f"bad value: {sigma2_use}", _format_debug_block(debug_lines))
    if (not np.isfinite(reference_variance)) or reference_variance < 0.0:
        _raise_failed("reference_variance", f"bad value: {reference_variance}", _format_debug_block(debug_lines))
    if (not np.isfinite(sigma2_total)) or sigma2_total < 0.0:
        _raise_failed("sigma2_total", f"bad value: {sigma2_total}", _format_debug_block(debug_lines))

    # CHECK 2: покрытие около confidence (мягкая проверка, диапазон можно настроить)
    coverage_time = _compute_coverage(times_real=times_real, time_lower=time_lower, time_upper=time_upper)
    if not np.isfinite(coverage_time):
        _raise_failed("coverage_time", "coverage is nan", _format_debug_block(debug_lines))

    # Например, допустим [0.90, 0.99] для 0.95 на реальных данных (подстрой при желании)
    if coverage_time < 0.90 or coverage_time > 0.99:
        debug_lines.append(f"coverage_time={coverage_time:.4f} (expected around {confidence:.2f})")
        _raise_failed("coverage_time", f"out of bounds: {coverage_time:.4f}", _format_debug_block(debug_lines))

    # CHECK 3: дисперсия остатков примерно равна sigma2_total (очень мягко)
    residuals_log = log_times_real - log_pred
    var_resid = float(np.var(residuals_log))
    if not np.isfinite(var_resid):
        _raise_failed("var_residuals", "var_residuals is nan", _format_debug_block(debug_lines))

    ratio = var_resid / sigma2_total if sigma2_total > 0.0 else float("nan")
    debug_lines.append(f"resid: var={var_resid:.6f}, ratio(var/sigma2_total)={ratio:.3f}")

    if np.isfinite(ratio) and (ratio < 0.7 or ratio > 1.3):
        _raise_failed("residual_variance_ratio", f"ratio out of bounds: {ratio:.3f}", _format_debug_block(debug_lines))

    # LOY diagnostics (только если падаем, иначе молчим):
    train_frame_loy = model.train_frame_loy
    trace_references = model.trace_references

    _require_columns(train_frame_loy, ["race_id", "gender", "age"])
    z_frame_loy = build_z_frame(train_frame=train_frame_loy.copy(), trace_references=trace_references)
    _require_columns(z_frame_loy, ["race_id", "gender", "age", "Z"])

    z_gender = z_frame_loy.loc[z_frame_loy["gender"] == gender, ["race_id", "age", "Z"]].copy()
    if len(z_gender) == 0:
        _raise_failed("loy_z_frame", "empty gender slice", _format_debug_block(debug_lines))

    if gender not in model.age_spline_models:
        _raise_failed("age_spline_models", f"missing gender={gender}", _format_debug_block(debug_lines))

    age_model = model.age_spline_models[gender]
    if not hasattr(age_model, "predict_mean"):
        _raise_failed("age_model", "no predict_mean", _format_debug_block(debug_lines))

    ages_train = pd.to_numeric(z_gender["age"], errors="coerce").to_numpy(dtype=float)
    z_values = pd.to_numeric(z_gender["Z"], errors="coerce").to_numpy(dtype=float)
    mean_z = age_model.predict_mean(ages_train)
    if not isinstance(mean_z, np.ndarray):
        mean_z = np.asarray([float(mean_z)], dtype=float)

    residual_z = z_values - mean_z
    z_gender["residual_z"] = residual_z

    race_stats = (
        z_gender.groupby("race_id", sort=False)["residual_z"]
        .agg(n="size", var=_sample_variance_ddof1, mean="mean")
        .reset_index()
    )
    race_row = race_stats.loc[race_stats["race_id"] == race_id]
    if len(race_row) != 1:
        debug_lines.append(f"loy: race not found in residual table: {race_id}")
        _raise_failed("loy_race_presence", "race not found", _format_debug_block(debug_lines))

    # Здесь никаких asserts: это чисто диагностика “на случай падения выше”.
    # Если хочешь, можно добавить жёсткую проверку на n_race>=X, но это будет флапать при обновлении данных.


def test_predict_uncertainty_real_data() -> None:
    tests: list[tuple[str, callable]] = [
        ("test_predict_uncertainty_real_data", run_test_predict_uncertainty_real_data),
    ]

    for test_name, test_fn in tests:
        print(f"\n== Running: {test_name}")
        try:
            test_fn()
            print(f"PASSED: {test_name}")
        except Exception as exc:
            print(f"FAILED: {test_name}")
            print(f"Error: {exc}")
            raise



if __name__ == "__main__":
    test_predict_uncertainty_real_data()
