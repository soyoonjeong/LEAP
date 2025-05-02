import os
import logging
from logging.handlers import RotatingFileHandler

from config.setting import ROOT_DIR

LOG_DIR = os.path.join(os.path.dirname(ROOT_DIR), "logs")


def setup_logger(name, log_level=logging.ERROR):
    """
    * How to Use
    logger = setup_logger(__name__)
    logger.error("Error Occured!", exc_info=True)
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    log_dir = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_level <= logging.INFO:
        log_file = os.path.join(log_dir, "info.log")
    else:
        log_file = os.path.join(log_dir, "error.log")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=1 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


ml_logger = setup_logger("logger", log_level=logging.INFO)
