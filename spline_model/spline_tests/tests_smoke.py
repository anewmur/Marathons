from __future__ import annotations

import logging

from spline_model.spline_tests.test_centering import test_centoring
from spline_model.spline_tests.test_fitter import test_fitter
from spline_model.spline_tests.test_marathon_predict_log_time import test_predict_log_time
from spline_model.spline_tests.test_predict import test_predict
from spline_model.spline_tests.test_reml import test_reml
from spline_model.spline_tests.test_spline import test_spline
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
        ("test_spline", test_spline),
        ("test_centoring", test_centoring),
        ("test_fitter", test_fitter),
        ("test_predict", test_predict),
        ('test_reml', test_reml),
        ('test_predict_Log_time', test_predict_log_time),

        # test real_data

        ("test_real_data", test_real_data),
    ]

    print("RUNNING tests_smoke.py FROM:", __file__)
    print("SMOKE TESTS ORDER:", [name for name, _ in tests])

    for test_name, test_fn in tests:
        print("\n==", test_name)
        test_fn()
        print("OK:", test_name)

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
