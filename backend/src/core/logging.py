from __future__ import annotations

import logging
import logging.config
import os


def setup_logging(level: str | None = None) -> None:
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                }
            },
            "root": {
                "level": log_level,
                "handlers": ["console"],
            },
            "loggers": {
                "uvicorn": {"level": log_level, "handlers": ["console"], "propagate": False},
                "uvicorn.error": {"level": log_level, "handlers": ["console"], "propagate": False},
                "uvicorn.access": {"level": log_level, "handlers": ["console"], "propagate": False},
            },
        }
    )