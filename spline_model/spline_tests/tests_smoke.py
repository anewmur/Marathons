from __future__ import annotations

import logging

from spline_model.spline_tests.test_centering import (
    test_centering_matrix_shapes_and_null_properties,
    test_centering_constraints_hold_for_random_gamma,
    test_solve_penalized_lsq_preserves_constraints_when_beta_is_C_gamma,
)
from spline_model.spline_tests.test_fitter import test_fit_gender_produces_model_and_beta_is_finite

from spline_model.spline_tests.test_spline import test_build_raw_basis_partition_of_unity, \
    test_solve_penalized_lsq_recovers_gamma_when_lambda_zero
from spline_model.spline_tests.test_real_data import test_real_data

class _OnlyLevelFilter(logging.Filter):
    def __init__(self, level: int) -> None:
        super().__init__()
        self._level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self._level


def _disable_logger() -> None:
    for logger_name in [
        "Preprocessor.preprocessor",
        "DataLoader.data_loader",
        "main",
        "trace_reference_builder",
        "age_reference_builder",
        "root",
    ]:
        target_logger = logging.getLogger(logger_name)
        target_logger.handlers.clear()
        target_logger.addHandler(logging.NullHandler())
        target_logger.propagate = False


def _configure_logging_debug_only() -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "%H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(_OnlyLevelFilter(logging.DEBUG))

    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    _disable_logger()


def main() -> None:
    _configure_logging_debug_only()

    tests: list[tuple[str, callable]] = [
        ("test_build_raw_basis_partition_of_unity", test_build_raw_basis_partition_of_unity),
        ("test_centering_matrix_shapes_and_null_properties", test_centering_matrix_shapes_and_null_properties),
        ("test_centering_constraints_hold_for_random_gamma", test_centering_constraints_hold_for_random_gamma),
        ("test_fit_gender_produces_model_and_beta_is_finite", test_fit_gender_produces_model_and_beta_is_finite),

        ("test_solve_penalized_lsq_recovers_gamma_when_lambda_zero",
         test_solve_penalized_lsq_recovers_gamma_when_lambda_zero),
        ("test_solve_penalized_lsq_preserves_constraints_when_beta_is_C_gamma",
         test_solve_penalized_lsq_preserves_constraints_when_beta_is_C_gamma),

        # test real_data

        ("test_real_data", test_real_data),
    ]

    print("RUNNING tests_smoke.py FROM:", __file__)
    print("SMOKE TESTS ORDER:", [name for name, _ in tests])

    for test_name, test_fn in tests:
        print("\n==", test_name)
        test_fn()
        print("OK:", test_name)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
