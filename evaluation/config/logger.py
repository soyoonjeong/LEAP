import os
import logging
from logging.handlers import RotatingFileHandler

from .path import LOG_DIR


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger."""

    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if name == "tqdm":
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter(
            "[%(asctime)s]-[%(levelname)s|%(filename)s:%(lineno)s]-[%(message)s]",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if name != "tqdm":
        logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
