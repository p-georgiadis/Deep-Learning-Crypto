# src/utils/logger.py
import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Union


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: grey + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


class Logger:
    """
    Custom logger configuration for the cryptocurrency prediction project.

    Features:
    - Console output with color formatting
    - File output with rotation
    - JSON formatting option
    - Different log levels for console and file
    - Automatic log directory creation
    """

    def __init__(
            self,
            name: str,
            log_dir: Union[str, Path],
            console_level: int,
            file_level: int,
            rotation: str,
            json_format: bool
    ):
        """
        Initialize logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            console_level: Logging level for console output
            file_level: Logging level for file output
            rotation: Type of log rotation ('time' or 'size')
            json_format: Whether to use JSON formatting for file output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers = []

        # Add handlers
        self._add_console_handler(console_level)
        self._add_file_handler(file_level, rotation, json_format)

    def _add_console_handler(self, level: int) -> None:
        """
        Add console handler with color formatting.

        Args:
            level: Logging level for console output
        """
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(CustomFormatter())
        self.logger.addHandler(console_handler)

    def _add_file_handler(
            self,
            level: int,
            rotation: str,
            json_format: bool
    ) -> None:
        """
        Add file handler with rotation.

        Args:
            level: Logging level for file output
            rotation: Type of log rotation
            json_format: Whether to use JSON formatting
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{timestamp}.log"

        if rotation == 'time':
            file_handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=30
            )
        else:  # size-based rotation
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )

        file_handler.setLevel(level)

        if json_format:
            formatter = self.JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt="%Y-%m-%d %H:%M:%S"
            )

        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    class JsonFormatter(logging.Formatter):
        """JSON formatter for structured logging."""

        def format(self, record):
            """Format log record as JSON."""
            log_data = {
                'timestamp': self.formatTime(record),
                'name': record.name,
                'level': record.levelname,
                'message': record.getMessage()
            }

            if hasattr(record, 'props'):
                log_data.update(record.props)

            return json.dumps(log_data)

    def get_logger(self) -> logging.Logger:
        """Get configured logger instance."""
        return self.logger

    def set_level(self, level: int) -> None:
        """
        Set logging level for all handlers.

        Args:
            level: New logging level
        """
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def add_file_handler(
            self,
            filename: str,
            level: int = logging.DEBUG
    ) -> None:
        """
        Add additional file handler.

        Args:
            filename: Log file name
            level: Logging level
        """
        handler = logging.FileHandler(self.log_dir / filename)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)


def setup_logger(
        name: str,
        log_dir: Union[str, Path],
        console_level: int,
        file_level: int,
        rotation: str,
        json_format: bool
) -> logging.Logger:
    """
    Convenience function to setup and get logger.

    Args:
        name: Logger name
        log_dir: Directory for log files
        console_level: Logging level for console output
        file_level: Logging level for file output
        rotation: Type of log rotation
        json_format: Whether to use JSON formatting

    Returns:
        Configured logger instance
    """
    logger_instance = Logger(
        name,
        log_dir,
        console_level,
        file_level,
        rotation,
        json_format
    )
    return logger_instance.get_logger()


if __name__ == "__main__":
    # Example usage
    logger = setup_logger(
        "crypto_predictor",
        log_dir="logs",
        json_format=True
    )

    logger.info("Starting application")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.exception("Caught an error")
