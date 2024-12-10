# src/utils/logger.py

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Union

def setup_logger(
    name: str,
    log_dir: Union[str, Path],
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    rotation: str = "size"
) -> logging.Logger:
    """
    Setup a logger with both console and file output.
    
    Args:
        name (str): Logger name.
        log_dir (str|Path): Directory for log files.
        console_level (int): Console logging level.
        file_level (int): File logging level.
        rotation (str): 'size' for size-based or 'time' for time-based rotation.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # master level

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # File handler
    timestamp = "log"
    log_file = log_dir / f"{timestamp}.log"
    if rotation == 'time':
        fh = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=7)
    else:  # size by default
        fh = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)

    fh.setLevel(file_level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    return logger

if __name__ == "__main__":
    # Example usage
    logger = setup_logger("myapp", "logs", console_level=logging.INFO, file_level=logging.DEBUG, rotation="size")
    logger.info("This is an info message.")
    logger.debug("This is a debug message.")
