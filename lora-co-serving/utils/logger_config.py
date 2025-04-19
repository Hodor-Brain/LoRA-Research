import logging
import sys
import os
from datetime import datetime

LOG_DIR = "logs"

def setup_logger(log_level=logging.INFO, log_to_file=True):
    """Configures the root logger for console and optional file logging.

    Args:
        log_level (int): The minimum logging level (e.g., logging.INFO, logging.DEBUG).
        log_to_file (bool): Whether to log to a timestamped file in the LOG_DIR.
    """
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger = logging.getLogger() # Get the root logger
    root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    if log_to_file:
        if not os.path.exists(LOG_DIR):
            try:
                os.makedirs(LOG_DIR)
            except OSError as e:
                root_logger.error(f"Error creating log directory {LOG_DIR}: {e}", exc_info=True)
                return # Exit if we cannot create the log directory

        log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(LOG_DIR, log_filename)

        try:
            file_handler = logging.FileHandler(log_filepath)
            file_handler.setFormatter(log_formatter)
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging initialized. Log file: {log_filepath}")
        except Exception as e:
             root_logger.error(f"Error setting up file handler {log_filepath}: {e}", exc_info=True)

    else:
        root_logger.info("Logging initialized (Console only).")
