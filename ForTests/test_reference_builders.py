from __future__ import annotations

import unittest
import numpy as np
import pandas as pd

from reference_builder import TraceReferenceBuilder
from age_reference_builder import AgeReferenceBuilder


class ReferenceBuilderTests(unittest.TestCase):
    def test_reference_builder_builds_rows(self) -> None:
        config = {
            "references": {
                "top_fraction": 0.5,
                "trim_fraction": 0.0,
                "min_group_size": 4,
                "min_used_runners": 2,
                "bootstrap_samples": 50,
                "random_seed": 1,
            }
        }

        dataframe = pd.DataFrame(
            {
                "race_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "gender": ["M", "M", "M", "M", "F", "F", "F", "F"],
                "time_seconds": [1000, 1100, 1200, 2000, 1500, 1600, 1700, 2500],
            }
        )

        builder = TraceReferenceBuilder(config)
        result = builder.build(dataframe)

        self.assertEqual(set(result.columns), {
            "race_id", "gender", "reference_time", "reference_var", "reference_std", "n_total", "n_used"
        })
        self.assertEqual(len(result), 2)

        row_a = result[(result["race_id"] == "A") & (result["race_id"] == "A") & (result["gender"] == "M")].iloc[0]
        self.assertTrue(row_a["reference_time"] > 0)
        self.assertTrue(row_a["reference_var", "reference_std"] >= 0)

    def test_reference_builder_skips_small_group(self) -> None:
        config = {
            "references": {
                "min_group_size": 10,
            }
        }
        dataframe = pd.DataFrame(
            {
                "race_id": ["A", "A", "A"],
                "gender": ["M", "M", "M"],
                "time_seconds": [1000, 1100, 1200],
            }
        )
        builder = TraceReferenceBuilder(config)
        result = builder.build(dataframe)
        self.assertEqual(len(result), 0)


class AgeReferenceBuilderTests(unittest.TestCase):
    def test_age_reference_builder_builds_rows(self) -> None:
        config = {
            "age_references": {
                "min_group_size": 3,
                "bootstrap_samples": 50,
                "random_seed": 2,
            }
        }

        dataframe = pd.DataFrame(
            {
                "race_id": ["A", "A", "A", "A", "A", "A", "A"],
                "gender": ["M", "M", "M", "M", "F", "F", "F"],
                "age": [30, 30, 30, 31, 30, 30, 30],
                "time_seconds": [1000, 1100, 1200, 1300, 1500, 1400, 1600],
            }
        )

        builder = AgeReferenceBuilder(config)
        result = builder.build(dataframe)

        self.assertEqual(set(result.columns), {
            "race_id", "gender", "age", "age_median_time", "age_median_var", "age_median_std", "n_total"
        })

        self.assertTrue(((result["race_id"] == "A") & (result["gender"] == "M") & (result["age"] == 30)).any())
        self.assertTrue(((result["race_id"] == "A") & (result["gender"] == "F") & (result["age"] == 30)).any())

    def test_age_reference_builder_empty(self) -> None:
        config = {"age_references": {"min_group_size": 2}}
        dataframe = pd.DataFrame(columns=["race_id", "gender", "age", "time_seconds"])
        builder = AgeReferenceBuilder(config)
        with self.assertRaises(ValueError):
            builder.build(dataframe)


if __name__ == "__main__":
    unittest.main(verbosity=2)