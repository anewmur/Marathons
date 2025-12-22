"""
Тест predict_with_uncertainty на реальных данных.

Зачем:
1) Проверить, что интервалы в log-шкале имеют разумную ширину.
2) Проверить покрытие: доля наблюдений, попавших в свой (age-specific) 95% интервал.
3) Сравнить эмпирическую дисперсию остатков в log-шкале с sigma2_total модели.
4) Диагностика: что именно оценивается в REML (размер LOY-выборки, число трасс).
5) Диагностика: оценка дисперсии остатков по трассам (race-specific), shrinkage к глобальной.

Важно:
- Проверка делается ПО НАБЛЮДЕНИЯМ: для каждого старта берём его age и строим свой интервал.
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


def _safe_percent(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.1f}%"


def _sample_variance_ddof1(values: pd.Series) -> float:
    array_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    count = int(array_values.shape[0])
    if count <= 1:
        return float("nan")
    return float(np.var(array_values, ddof=1))


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


def _print_fit_report(model: MarathonModel, gender: str) -> None:
    print("=" * 80)
    print("FIT_REPORT (age_spline_models)")
    print("=" * 80)

    if not hasattr(model, "age_spline_models"):
        raise RuntimeError("model has no attribute age_spline_models (check MarathonModel.run)")

    if gender not in model.age_spline_models:
        raise RuntimeError(f"No age_spline_model for gender={gender}")

    age_model = model.age_spline_models[gender]
    fit_report = getattr(age_model, "fit_report", None)
    if not isinstance(fit_report, dict):
        raise RuntimeError("age_model.fit_report is missing or not a dict")

    print(f"gender={gender}")
    print(f"lambda_method={fit_report.get('lambda_method')}")
    print(f"lambda_value={fit_report.get('lambda_value')}")
    print(f"edf={fit_report.get('edf')}")
    print(f"nu={fit_report.get('nu')}")
    print(f"sigma2_reml={fit_report.get('sigma2_reml')}")
    print(f"tau2_bar={fit_report.get('tau2_bar')}")
    print(f"sigma2_use={fit_report.get('sigma2_use')}")
    print(f"rmse_z={fit_report.get('rmse_z')}")
    print(f"mae_z={fit_report.get('mae_z')}")
    print(f"degrade={fit_report.get('degrade')}")

    print("-" * 80)
    print("fit_report (full, sorted keys):")
    for report_key in sorted(fit_report.keys()):
        report_value = fit_report[report_key]
        print(f"  {report_key}: {report_value}")


def _print_observations_stats(observations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("=" * 80)
    print("РЕАЛЬНЫЕ НАБЛЮДЕНИЯ")
    print("=" * 80)
    print(f"Найдено наблюдений: {len(observations)}")

    ages_obs = observations["age"].astype(float).to_numpy(dtype=float)
    times_real = observations["time_seconds"].astype(float).to_numpy(dtype=float)
    log_times_real = np.log(times_real)

    print("Статистика time_seconds:")
    print(f"  mean:   {float(np.mean(times_real)):.1f} ({seconds_to_hhmmss(float(np.mean(times_real)))})")
    print(f"  median: {float(np.median(times_real)):.1f} ({seconds_to_hhmmss(float(np.median(times_real)))})")
    print(f"  std:    {float(np.std(times_real)):.1f}")
    print(f"  min:    {float(np.min(times_real)):.1f} ({seconds_to_hhmmss(float(np.min(times_real)))})")
    print(f"  max:    {float(np.max(times_real)):.1f} ({seconds_to_hhmmss(float(np.max(times_real)))})")

    print("Статистика ln(time_seconds):")
    empirical_var_log = float(np.var(log_times_real))
    print(f"  mean:   {float(np.mean(log_times_real)):.6f}")
    print(f"  median: {float(np.median(log_times_real)):.6f}")
    print(f"  std:    {float(np.std(log_times_real)):.6f}")
    print(f"  var:    {empirical_var_log:.6f}")

    return ages_obs, times_real, log_times_real


def _predict_all(
    model: MarathonModel,
    race_id: str,
    gender: str,
    year_test: int | None,
    ages_obs: np.ndarray,
    confidence: float,
    method: str,
) -> dict[str, np.ndarray | float]:
    print("=" * 80)
    print("ПРОГНОЗ МОДЕЛИ ДЛЯ КАЖДОГО НАБЛЮДЕНИЯ")
    print("=" * 80)

    year_value = int(year_test) if year_test is not None else 0

    result = model.predict_with_uncertainty(
        race_id=race_id,
        gender=gender,
        age=ages_obs,
        year=year_value,
        confidence=confidence,
        method=method,
    )

    time_pred = _as_1d_float_array(result["time_pred"])
    time_lower = _as_1d_float_array(result["time_lower"])
    time_upper = _as_1d_float_array(result["time_upper"])

    log_pred = _as_1d_float_array(result["log_pred"])
    log_lower = _as_1d_float_array(result["log_lower"])
    log_upper = _as_1d_float_array(result["log_upper"])

    sigma2_use = float(result["sigma2_use"])
    reference_variance = float(result["reference_variance"])
    sigma2_total = float(result.get("sigma2_total", sigma2_use + reference_variance))
    sigma = float(result.get("sigma", np.sqrt(sigma2_total)))

    print("Компоненты дисперсии (log-шкала):")
    print(f"  sigma2_use:         {sigma2_use:.6f}")
    print(f"  reference_variance: {reference_variance:.6f}")
    print(f"  sigma2_total:       {sigma2_total:.6f}")
    print(f"  sigma:              {sigma:.6f}")

    return {
        "time_pred": time_pred,
        "time_lower": time_lower,
        "time_upper": time_upper,
        "log_pred": log_pred,
        "log_lower": log_lower,
        "log_upper": log_upper,
        "sigma2_use": sigma2_use,
        "reference_variance": reference_variance,
        "sigma2_total": sigma2_total,
        "sigma": sigma,
    }


def _print_coverage_and_widths(
    ages_obs: np.ndarray,
    times_real: np.ndarray,
    log_times_real: np.ndarray,
    pred: dict[str, np.ndarray | float],
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    time_lower = pred["time_lower"]
    time_upper = pred["time_upper"]
    log_lower = pred["log_lower"]
    log_upper = pred["log_upper"]

    if not isinstance(time_lower, np.ndarray) or not isinstance(time_upper, np.ndarray):
        raise RuntimeError("time_lower/time_upper must be arrays")
    if not isinstance(log_lower, np.ndarray) or not isinstance(log_upper, np.ndarray):
        raise RuntimeError("log_lower/log_upper must be arrays")

    valid_mask = (
        np.isfinite(times_real)
        & np.isfinite(log_times_real)
        & np.isfinite(time_lower)
        & np.isfinite(time_upper)
        & np.isfinite(log_lower)
        & np.isfinite(log_upper)
    )

    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        raise RuntimeError("Нет валидных строк для сравнения (всё NaN/inf).")

    times_real_v = times_real[valid_mask]
    log_times_real_v = log_times_real[valid_mask]
    time_lower_v = time_lower[valid_mask]
    time_upper_v = time_upper[valid_mask]
    log_lower_v = log_lower[valid_mask]
    log_upper_v = log_upper[valid_mask]

    in_interval_time = (times_real_v >= time_lower_v) & (times_real_v <= time_upper_v)
    in_interval_log = (log_times_real_v >= log_lower_v) & (log_times_real_v <= log_upper_v)

    coverage_time = 100.0 * float(np.mean(in_interval_time))
    coverage_log = 100.0 * float(np.mean(in_interval_log))

    width_time = time_upper_v - time_lower_v
    width_log = log_upper_v - log_lower_v

    print("=" * 80)
    print("ПОКРЫТИЕ И ШИРИНЫ ИНТЕРВАЛОВ")
    print("=" * 80)
    print(f"Валидных строк: {valid_count} / {len(times_real)}")

    print("Покрытие (по времени):")
    print(f"  {_safe_percent(coverage_time)} (ожидается {confidence*100:.1f}%)")

    print("Покрытие (по log-времени):")
    print(f"  {_safe_percent(coverage_log)} (ожидается {confidence*100:.1f}%)")

    print("Ширина интервалов (time_seconds):")
    print(f"  median width: {float(np.median(width_time)):.1f} сек ({seconds_to_hhmmss(float(np.median(width_time)))})")
    print(f"  mean width:   {float(np.mean(width_time)):.1f} сек ({seconds_to_hhmmss(float(np.mean(width_time)))})")

    print("Ширина интервалов (log):")
    print(f"  median width: {float(np.median(width_log)):.6f}")
    print(f"  mean width:   {float(np.mean(width_log)):.6f}")

    return valid_mask, in_interval_time


def _print_residuals_vs_sigma(
    times_real: np.ndarray,
    log_times_real: np.ndarray,
    pred: dict[str, np.ndarray | float],
    valid_mask: np.ndarray,
) -> None:
    log_pred = pred["log_pred"]
    sigma2_total = float(pred["sigma2_total"])

    if not isinstance(log_pred, np.ndarray):
        raise RuntimeError("log_pred must be array")

    log_pred_v = log_pred[valid_mask]
    log_times_real_v = log_times_real[valid_mask]

    residuals_log = log_times_real_v - log_pred_v
    residuals_var_log = float(np.var(residuals_log))

    print("=" * 80)
    print("СРАВНЕНИЕ ДИСПЕРСИИ ОСТАТКОВ С sigma2_total")
    print("=" * 80)
    print("Остатки: r_i = ln(time_i) - log_pred_i")
    print(f"  var(residuals_log): {residuals_var_log:.6f}")
    print(f"  sigma2_total(model): {sigma2_total:.6f}")
    if sigma2_total > 0:
        print(f"  ratio var(residuals)/sigma2_total: {residuals_var_log / sigma2_total:.2f}x")


def _print_examples(
    ages_obs: np.ndarray,
    times_real: np.ndarray,
    pred: dict[str, np.ndarray | float],
    valid_mask: np.ndarray,
    in_interval_time: np.ndarray,
    show_count: int = 5,
) -> None:
    time_lower = pred["time_lower"]
    time_upper = pred["time_upper"]

    if not isinstance(time_lower, np.ndarray) or not isinstance(time_upper, np.ndarray):
        raise RuntimeError("time_lower/time_upper must be arrays")

    ages_v = ages_obs[valid_mask]
    times_v = times_real[valid_mask]
    time_lower_v = time_lower[valid_mask]
    time_upper_v = time_upper[valid_mask]

    print("=" * 80)
    print("КОНТРОЛЬНЫЕ ПРИМЕРЫ (первые 5 строк)")
    print("=" * 80)

    rows_to_show = int(min(show_count, times_v.shape[0]))
    for idx in range(rows_to_show):
        time_value = float(times_v[idx])
        print(
            f"age={float(ages_v[idx]):.1f} "
            f"time={seconds_to_hhmmss(time_value)} "
            f"pred=[{seconds_to_hhmmss(float(time_lower_v[idx]))}, {seconds_to_hhmmss(float(time_upper_v[idx]))}] "
            f"in={bool(in_interval_time[idx])}"
        )


def _print_reml_training_sample(model: MarathonModel, gender: str) -> None:
    print("=" * 80)
    print("REML TRAINING SAMPLE (LOY)")
    print("=" * 80)

    train_frame_loy = getattr(model, "train_frame_loy", None)
    if train_frame_loy is None:
        raise RuntimeError("model.train_frame_loy is None")

    _require_columns(train_frame_loy, ["race_id", "gender", "age"])

    train_gender = train_frame_loy.loc[train_frame_loy["gender"] == gender].copy()

    print(f"train_frame_loy rows (all genders): {len(train_frame_loy)}")
    print(f"train_frame_loy rows (gender={gender}): {len(train_gender)}")

    if "year" in train_gender.columns:
        years_series = pd.to_numeric(train_gender["year"], errors="coerce").dropna()
        years_unique = sorted(years_series.astype(int).unique().tolist())
        if len(years_unique) > 0:
            print(f"years: {years_unique[0]}..{years_unique[-1]} (count={len(years_unique)})")

    race_count = int(train_gender["race_id"].nunique())
    print(f"unique race_id (gender={gender}): {race_count}")

    print("top races by n (gender slice):")
    race_sizes = (
        train_gender.groupby("race_id", sort=False)
        .size()
        .sort_values(ascending=False)
        .head(15)
    )
    print(race_sizes.to_string())


def _print_race_variance_diagnostics(
    model: MarathonModel,
    race_id: str,
    gender: str,
    prior_n: int = 200,
) -> None:
    print("=" * 80)
    print("RACE-SPECIFIC RESIDUAL VARIANCE (LOY, in Z-scale)")
    print("=" * 80)

    train_frame_loy = getattr(model, "train_frame_loy", None)
    if train_frame_loy is None:
        raise RuntimeError("model.train_frame_loy is None")

    trace_references = getattr(model, "trace_references", None)
    if trace_references is None:
        raise RuntimeError("model.trace_references is None")

    z_frame_loy = build_z_frame(train_frame=train_frame_loy.copy(), trace_references=trace_references)

    _require_columns(z_frame_loy, ["race_id", "gender", "age", "Z"])

    z_gender = z_frame_loy.loc[z_frame_loy["gender"] == gender, ["race_id", "age", "Z"]].copy()
    if len(z_gender) == 0:
        raise RuntimeError(f"Empty z_frame slice for gender={gender}")

    if gender not in model.age_spline_models:
        raise RuntimeError(f"No age_spline_model for gender={gender}")

    age_model = model.age_spline_models[gender]
    if not hasattr(age_model, "predict_mean"):
        raise RuntimeError("age_model has no predict_mean(age) method")

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

    race_stats = race_stats.loc[np.isfinite(race_stats["var"])].copy()
    race_stats = race_stats.sort_values(["n"], ascending=False)

    print("Top races by n with residual variance:")
    print(race_stats.head(15).to_string(index=False))

    target_race_row = race_stats.loc[race_stats["race_id"] == race_id]
    if len(target_race_row) != 1:
        print(f"Race not found in LOY residual table: {race_id}")
        return

    sigma2_race_hat = float(target_race_row["var"].iloc[0])
    n_race = int(target_race_row["n"].iloc[0])
    sigma2_global = float(age_model.sigma2_use)

    print("-" * 80)
    print(f"race_id={race_id}")
    print(f"n_race={n_race}")
    print(f"sigma2_race_hat(ddof1)={sigma2_race_hat:.6f}")
    print(f"sigma2_global(age_model.sigma2_use)={sigma2_global:.6f}")

    print("=" * 80)
    print("RACE VARIANCE SHRINKAGE (empirical Bayes style)")
    print("=" * 80)

    race_stats = race_stats.copy()
    race_stats["n_eff"] = (race_stats["n"].astype(float) - 1.0).clip(lower=0.0)
    race_stats["rss"] = race_stats["var"].astype(float) * race_stats["n_eff"]

    sigma2_global = float(age_model.sigma2_use)
    race_stats["sigma2_shrunk"] = (race_stats["rss"] + float(prior_n) * sigma2_global) / (
        race_stats["n_eff"] + float(prior_n)
    )

    print(race_stats.head(15)[["race_id", "n", "var", "sigma2_shrunk"]].to_string(index=False))

    target_race_row = race_stats.loc[race_stats["race_id"] == race_id]
    if len(target_race_row) == 1:
        print("-" * 80)
        print(f"race_id={race_id}")
        print(f"sigma2_race_hat={float(target_race_row['var'].iloc[0]):.6f}")
        print(f"sigma2_race_shrunk={float(target_race_row['sigma2_shrunk'].iloc[0]):.6f}")


def _print_residual_var_check_loy(model: MarathonModel, race_id: str, gender: str) -> None:
    print("=" * 80)
    print("RESIDUAL VAR CHECK (LOY)")
    print("=" * 80)

    train_frame_loy = getattr(model, "train_frame_loy", None)
    if train_frame_loy is None:
        raise RuntimeError("model.train_frame_loy is None")

    _require_columns(train_frame_loy, ["race_id", "gender", "age"])

    if gender not in model.age_spline_models:
        raise RuntimeError(f"No age_spline_model for gender={gender}")

    age_model = model.age_spline_models[gender]
    if not hasattr(age_model, "predict_mean"):
        raise RuntimeError("age_model has no predict_mean(age) method")

    train_gender = train_frame_loy.loc[train_frame_loy["gender"] == gender].copy()
    if "Z" not in train_gender.columns:
        raise RuntimeError("train_frame_loy must have column Z")

    ages_train = train_gender["age"].astype(float).to_numpy(dtype=float)
    z_train = train_gender["Z"].astype(float).to_numpy(dtype=float)

    mean_train = age_model.predict_mean(ages_train)
    if not isinstance(mean_train, np.ndarray):
        mean_train = np.asarray([float(mean_train)], dtype=float)

    residual_all = z_train - mean_train
    var_all = float(np.var(residual_all, ddof=0))

    print(f"var(residual_z) on LOY, gender={gender}: {var_all:.6f}")
    print(f"sigma2_use(global): {float(age_model.sigma2_use):.6f}")

    # --- дополнительная диагностика: sigma2_from_rss = RSS / nu ---
    fit_report = model.age_spline_models[gender].fit_report

    nu_report_raw = fit_report.get("nu")
    if nu_report_raw is None:
        raise RuntimeError("fit_report['nu'] is None (cannot compute sigma2_from_rss)")
    nu_report = float(nu_report_raw)

    edf_report_raw = fit_report.get("edf")
    edf_report = float(edf_report_raw) if edf_report_raw is not None else float("nan")

    rss_all = float(np.sum(residual_all * residual_all))
    n_all = int(residual_all.shape[0])

    sigma2_from_rss = rss_all / nu_report
    sigma2_from_var0 = float(np.var(residual_all, ddof=0))
    sigma2_from_var1 = float(np.var(residual_all, ddof=1)) if n_all > 1 else float("nan")

    sigma2_reml_report = float(fit_report.get("sigma2_reml"))
    sigma2_use_report = fit_report.get("sigma2_use")
    sigma2_use_attr = getattr(age_model, "sigma2_use", None)

    print("-" * 80)
    print("SIGMA2 CHECK: RSS/nu vs var(residuals)")
    print(f"rss(residual_z) on LOY, gender={gender}: {rss_all:.3f}")
    print(f"n={n_all}, nu(report)={nu_report:.3f}, edf(report)={edf_report:.6f}")
    print(f"rss/n: {rss_all / float(n_all):.6f}")
    if n_all > 1:
        print(f"rss/(n-1): {rss_all / float(n_all - 1):.6f}")
    print(f"sigma2_from_var(ddof0): {sigma2_from_var0:.6f}")
    print(f"sigma2_from_var(ddof1): {sigma2_from_var1:.6f}")
    print(f"sigma2_from_rss=RSS/nu: {sigma2_from_rss:.6f}")
    print(f"sigma2_reml(report): {sigma2_reml_report:.6f}")
    print(f"sigma2_use(report): {sigma2_use_report}")
    print(f"sigma2_use(attr): {sigma2_use_attr}")


    train_race = train_gender.loc[train_gender["race_id"] == race_id].copy()
    if len(train_race) <= 5:
        print(f"LOY slice for race={race_id} has too few rows: n={len(train_race)}")
        return

    ages_race = train_race["age"].astype(float).to_numpy(dtype=float)
    z_race = train_race["Z"].astype(float).to_numpy(dtype=float)

    mean_race = age_model.predict_mean(ages_race)
    if not isinstance(mean_race, np.ndarray):
        mean_race = np.asarray([float(mean_race)], dtype=float)

    residual_race = z_race - mean_race
    var_race = float(np.var(residual_race, ddof=0))
    print(f"var(residual_z) on LOY, race={race_id}, gender={gender}: {var_race:.6f} (n={len(train_race)})")


def main() -> None:
    race_id = "Белые ночи"
    gender = "M"

    year_test: int | None = None
    age_center: float | None = 40.0
    age_tolerance = 0.0

    confidence = 0.95
    method = "analytical"

    print("=" * 80)
    print("ТЕСТ predict_with_uncertainty НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 80)

    model = MarathonModel(
        data_path=r"C:\Users\andre\github\Marathons\Data",
        verbose=True,
    )
    model.run()

    _print_fit_report(model=model, gender=gender)

    print("Параметры теста:")
    print(f"  race_id: {race_id}")
    print(f"  gender: {gender}")
    print(f"  year_test: {year_test}")
    print(f"  age_center: {age_center}")
    print(f"  age_tolerance: {age_tolerance}")
    print(f"  confidence: {confidence}")
    print(f"  method: {method}")

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
        raise RuntimeError("Нет наблюдений после фильтров. Ослабь фильтры (year/age).")

    ages_obs, times_real, log_times_real = _print_observations_stats(observations=observations)

    pred = _predict_all(
        model=model,
        race_id=race_id,
        gender=gender,
        year_test=year_test,
        ages_obs=ages_obs,
        confidence=confidence,
        method=method,
    )

    valid_mask, in_interval_time = _print_coverage_and_widths(
        ages_obs=ages_obs,
        times_real=times_real,
        log_times_real=log_times_real,
        pred=pred,
        confidence=confidence,
    )

    _print_residuals_vs_sigma(
        times_real=times_real,
        log_times_real=log_times_real,
        pred=pred,
        valid_mask=valid_mask,
    )

    _print_examples(
        ages_obs=ages_obs,
        times_real=times_real,
        pred=pred,
        valid_mask=valid_mask,
        in_interval_time=in_interval_time,
        show_count=5,
    )

    _print_reml_training_sample(model=model, gender=gender)
    _print_race_variance_diagnostics(model=model, race_id=race_id, gender=gender, prior_n=200)
    _print_residual_var_check_loy(model=model, race_id=race_id, gender=gender)

    print("=" * 80)
    print("ГОТОВО")
    print("=" * 80)


if __name__ == "__main__":
    main()
