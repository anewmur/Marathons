"""
Модуль для вычисления средней дисперсии эталона (tau2_bar).

Функция compute_tau2_bar вычисляет среднюю дисперсию эталона для заданного пола,
используя информацию о дисперсиях эталонов из trace_references.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def compute_tau2_bar(
    gender_df: pd.DataFrame,
    trace_references: pd.DataFrame,
    gender: str,
) -> float:
    """
    Вычисление средней дисперсии эталона для пола.
    
    Дисперсия эталона (reference_variance) характеризует вариабельность
    эталонных времён R^{use} между разными трассами для данного пола.
    
    tau2_bar используется для корректировки остаточной дисперсии:
        sigma2_use = max(floor, sigma2_reml - tau2_bar)
    
    Алгоритм:
    1. Проверяет наличие reference_variance в trace_references
    2. Джойнит gender_df с trace_references по (race_id, gender)
    3. Вычисляет среднее reference_variance по всем строкам gender_df
    4. Возвращает 0.0 если данные недоступны
    
    Args:
        gender_df: DataFrame с данными для пола, должен содержать колонки:
                   - race_id: идентификатор трассы
                   - gender: пол ("M" или "F")
        trace_references: DataFrame с эталонами, ожидаемые колонки:
                         - race_id: идентификатор трассы
                         - gender: пол
                         - reference_variance: дисперсия эталона для (race_id, gender)
        gender: str, пол для которого вычисляется tau2_bar ("M" или "F")
    
    Returns:
        tau2_bar: float >= 0, средняя дисперсия эталона
    
    Raises:
        ValueError: если gender_df не содержит race_id или gender
        ValueError: если trace_references не содержит race_id или gender
    
    Примечания:
        - Если reference_variance отсутствует в trace_references, возвращает 0.0
        - Если после merge все reference_variance = NA, возвращает 0.0
        - Логирует предупреждения в обоих случаях
        - Всегда возвращает конечное неотрицательное число
    """
    # Проверка обязательных колонок в gender_df
    required_gender_cols = ["race_id", "gender"]
    missing_cols = [col for col in required_gender_cols if col not in gender_df.columns]
    if missing_cols:
        raise ValueError(
            f"compute_tau2_bar: gender_df missing required columns: {missing_cols}"
        )
    
    # Проверка обязательных колонок в trace_references
    required_trace_cols = ["race_id", "gender"]
    missing_cols = [col for col in required_trace_cols if col not in trace_references.columns]
    if missing_cols:
        raise ValueError(
            f"compute_tau2_bar: trace_references missing required columns: {missing_cols}"
        )
    
    # Проверка наличия reference_variance
    if "reference_variance" not in trace_references.columns:
        logger.warning(
            "compute_tau2_bar: reference_variance not in trace_references, "
            "using tau2_bar=0.0 for gender=%s",
            gender
        )
        return 0.0
    
    # Merge gender_df с trace_references
    # Используем только нужные колонки для эффективности
    merged = gender_df.merge(
        trace_references[["race_id", "gender", "reference_variance"]],
        on=["race_id", "gender"],
        how="left"
    )
    
    # Вычисляем среднее reference_variance
    tau2_bar = merged["reference_variance"].mean()
    
    # Проверка на NA (может быть если все merge не матчнулись или все reference_variance=NA)
    if pd.isna(tau2_bar):
        logger.warning(
            "compute_tau2_bar: tau2_bar is NA (no matching reference_variance), "
            "using tau2_bar=0.0 for gender=%s",
            gender
        )
        return 0.0
    
    # Проверка на неотрицательность (на случай если в данных ошибка)
    if tau2_bar < 0:
        logger.warning(
            "compute_tau2_bar: tau2_bar=%.6f is negative, "
            "using tau2_bar=0.0 for gender=%s",
            tau2_bar, gender
        )
        return 0.0
    
    return float(tau2_bar)
