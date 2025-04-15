import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from utils.directories import LOG_FILES_DIR

def setup_logger(name="face_recognition", to_file=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    if to_file:
        os.makedirs(LOG_FILES_DIR, exist_ok=True)
        log_file = os.path.join(
            LOG_FILES_DIR,
            f"face_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
        ))
        logger.addHandler(file_handler)

    logging.getLogger().handlers.clear()
    return logger


logger = setup_logger(to_file=False)
