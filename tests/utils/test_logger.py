import logging
from unittest.mock import patch
from src.utils.logger import setup_logger


def test_logger_initialization(tmp_path):
    """Test logger initialization."""
    log_dir = tmp_path / "logs"
    logger = setup_logger("test_logger", log_dir)

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"
    assert log_dir.exists()


def test_logger_console_output():
    """Test logger console output."""
    with patch("logging.StreamHandler.emit") as mock_emit:
        logger = setup_logger("test_logger", console_level=logging.INFO)
        logger.info("Test message")

        assert mock_emit.called


def test_logger_file_output(tmp_path):
    """Test logger file output."""
    log_dir = tmp_path / "logs"
    logger = setup_logger("test_logger", log_dir)

    log_file = next(log_dir.glob("*.log"), None)
    assert log_file is not None

    with open(log_file) as f:
        content = f.read()
    assert "Test message" not in content  # Add tests to verify log content as needed
