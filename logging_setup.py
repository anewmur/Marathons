"""
Настройка логирования для проекта MarathonModel.

Цветной вывод в консоль + запись в файл.
"""

import logging
from pathlib import Path


class CustomFormatter(logging.Formatter):
    """Цветной вывод в консоль."""

    grey = "\x1b[38;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    base_fmt = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    date_fmt = "%H:%M:%S"

    FORMATS = {
        logging.DEBUG: green + base_fmt + reset,
        logging.INFO: grey + base_fmt + reset,
        logging.WARNING: yellow + base_fmt + reset,
        logging.ERROR: red + base_fmt + reset,
        logging.CRITICAL: bold_red + base_fmt + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        fmt = self.FORMATS.get(record.levelno, self.base_fmt)
        formatter = logging.Formatter(fmt, self.date_fmt)
        return formatter.format(record)


def easy_logging(enable: bool = True) -> None:
    """
    Полная настройка логов: файл + цветная консоль.
    
    Args:
        enable: Включить логирование (True) или отключить (False)
    """
    root = logging.getLogger()

    if not enable:
        root.handlers.clear()
        root.disabled = True
        return

    root.disabled = False
    root.setLevel(logging.DEBUG)

    # Удалить ВСЕ обработчики: root + дочерние
    for logger_name in list(logging.root.manager.loggerDict.keys()):
        logging.getLogger(logger_name).handlers.clear()
    root.handlers.clear()

    # Создать директорию logs
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "last_run.log"

    # Файл: принимает всё
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(file_handler)

    # Консоль
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(CustomFormatter())

    # Фильтр: консоль НЕ печатает INFO
    class ConsoleFilter(logging.Filter):
        allowed = {
            logging.DEBUG,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        }

        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno in self.allowed

    console.addFilter(ConsoleFilter())
    root.addHandler(console)

    root.info("=== Логирование включено ===")
