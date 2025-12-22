
import pandas as pd


def build_z_frame(
    train_frame: pd.DataFrame,
    trace_references: pd.DataFrame,
) -> pd.DataFrame:
    """
    Делает таблицу для возрастного фиттера: (gender, age, z) + опционально x.

    - возрастная модель обучается на нормированной шкале Z,
      где Z = Y - ln R^{use} для пары (race_id, gender).
    """
    required_train = ["race_id", "gender", "age", "Y"]
    missing_train = [col for col in required_train if col not in train_frame.columns]
    if missing_train:
        raise RuntimeError(f"train_frame missing columns: {missing_train}")

    required_ref = ["race_id", "gender", "reference_log"]
    missing_ref = [col for col in required_ref if col not in trace_references.columns]
    if missing_ref:
        raise RuntimeError(f"trace_references missing columns: {missing_ref}")

    merged = train_frame.merge(
        trace_references[["race_id", "gender", "reference_log"]],
        on=["race_id", "gender"],
        how="left",
        validate="many_to_one",
    )

    if merged["reference_log"].isna().any():
        bad_rows = merged.loc[merged["reference_log"].isna(), ["race_id", "gender"]].drop_duplicates()
        raise RuntimeError(
            "Missing reference_log for some (race_id, gender):\n"
            f"{bad_rows.to_string(index=False)}"
        )

    merged = merged.copy()
    merged["Z"] = merged["Y"].astype(float) - merged["reference_log"].astype(float)

    keep_cols = ["gender", "age", "Z", "race_id"]
    if "x" in merged.columns:
        keep_cols.append("x")

    return merged[keep_cols]
