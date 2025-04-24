import logging
import sys
import os
from datetime import datetime

LOG_DIR = "logs"

def setup_logger(log_level=logging.INFO, log_to_file=True):
    """Configures the root logger for console and optional file logging.

    Args:
        log_level (int): The minimum logging level (e.g., logging.INFO, logging.DEBUG).
                         This controls the overall level and the file log level.
        log_to_file (bool): Whether to log to a timestamped file in the LOG_DIR.
    """
    log_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)-5.5s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(min(log_level, logging.INFO))
    if log_level < logging.INFO:
         root_logger.setLevel(log_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    try:
         console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
         pass 
    root_logger.addHandler(console_handler)

    if log_to_file:
        if not os.path.exists(LOG_DIR):
            try:
                os.makedirs(LOG_DIR)
            except OSError as e:
                console_handler.handle(root_logger.makeRecord(
                    name=root_logger.name, level=logging.ERROR, 
                    fn="", lno=0, msg=f"Error creating log directory {LOG_DIR}: {e}", 
                    args=[], exc_info=True, func="setup_logger"
                ))
                return

        log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_filepath = os.path.join(LOG_DIR, log_filename)

        try:
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            file_handler.setLevel(log_level) 
            root_logger.addHandler(file_handler)
            root_logger.info(f"Logging initialized. Log file: {log_filepath}. Console Level: INFO. File Level: {logging.getLevelName(log_level)}.")
        except Exception as e:
             console_handler.handle(root_logger.makeRecord(
                name=root_logger.name, level=logging.ERROR, 
                fn="", lno=0, msg=f"Error setting up file handler {log_filepath}: {e}", 
                args=[], exc_info=True, func="setup_logger"
             ))

    else:
        root_logger.info(f"Logging initialized (Console only). Console Level: INFO.")
