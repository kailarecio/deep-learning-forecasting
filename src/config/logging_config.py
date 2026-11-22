import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_style: str = "detailed",
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Logging level (logging.DEBUG, logging.INFO, etc.)
        format_style: 'detailed' or 'simple'
        log_file: Optional file path to also log to a file
    """
    formats = {
        "detailed": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        "simple": "%(asctime)s - %(levelname)s - %(message)s"
    }
    date_format = "%Y-%m-%d %H:%M:%S"
    log_format = formats.get(format_style, formats["detailed"])

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True  # Override any existing configuration (e.g. Jupyter's)
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        logging.getLogger().addHandler(file_handler)

    # Suppress noisy third-party loggers
    for name in ("matplotlib", "urllib3", "DUKASCRIPT", "dukascopy_python"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
