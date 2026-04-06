from __future__ import annotations

import logging
import os
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Final


DEFAULT_LOG_DIR: Final[Path] = Path(os.getenv("LOG_DIR", "out/logs")).resolve()
DEFAULT_LOG_LEVEL: Final[str] = os.getenv("LOG_LEVEL", "INFO").upper()
DEFAULT_LOG_FILE_NAME: Final[str] = os.getenv("LOG_FILE_NAME", "app.log")
DEFAULT_LOG_BACKUP_COUNT: Final[int] = int(os.getenv("LOG_BACKUP_COUNT", "14"))

_CONFIGURED = False


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level

    resolved = logging.getLevelName(level.upper())
    if isinstance(resolved, int):
        return resolved

    raise ValueError(
        f"Invalid log level: {level!r}. "
        f"Use one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
    )


def _build_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(process)d:%(threadName)s | %(filename)s:%(lineno)d | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_logging(
    *,
    app_name: str = "app",
    level: str | int = DEFAULT_LOG_LEVEL,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    enable_console: bool = True,
    enable_file: bool = True,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
    force: bool = False,
) -> Logger:
    """
    Configure application logging once.

    Args:
        app_name: Name of the application logger to return.
        level: Logging level (e.g. INFO / DEBUG).
        log_dir: Directory for log files.
        enable_console: Enable stdout/stderr logging.
        enable_file: Enable file logging.
        backup_count: How many rotated log files to keep.
        force: Reconfigure logging even if handlers already exist.

    Returns:
        Configured application logger.
    """
    global _CONFIGURED

    root_logger = logging.getLogger()

    if _CONFIGURED and not force:
        return logging.getLogger(app_name)

    if force:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    elif root_logger.handlers:
        # Respect existing logging configuration if already set by app/framework.
        _CONFIGURED = True
        return logging.getLogger(app_name)

    resolved_level = _resolve_level(level)
    formatter = _build_formatter()

    root_logger.setLevel(resolved_level)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    if enable_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = TimedRotatingFileHandler(
            filename=str(log_path / DEFAULT_LOG_FILE_NAME),
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(resolved_level)
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(file_handler)

    logging.captureWarnings(True)
    _CONFIGURED = True

    logger = logging.getLogger(app_name)
    logger.info("Logging configured")
    return logger


def get_logger(name: str | None = None) -> Logger:
    """
    Return a named logger.
    """
    return logging.getLogger(name or __name__)