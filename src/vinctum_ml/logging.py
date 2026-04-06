"""Structured logging for Vinctum ML."""

import logging
import json
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "data"):
            log["data"] = record.data

        if record.exc_info and record.exc_info[0]:
            log["error"] = self.formatException(record.exc_info)

        return json.dumps(log)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get a structured JSON logger."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
