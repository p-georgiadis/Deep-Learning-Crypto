"""
Test configuration and shared utilities for the Crypto Prediction project.
This file allows the tests directory to be treated as a Python package.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test constants
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
TEST_CONFIG_PATH = PROJECT_ROOT / "tests" / "test_data" / "test_config.yaml"

# Ensure test directories exist
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Utility functions for tests
def get_test_file_path(filename: str) -> Path:
    """Get the full path for a test file"""
    return TEST_DATA_DIR / filename

def cleanup_test_file(filepath: Path) -> None:
    """Clean up a test file if it exists"""
    if filepath.exists():
        filepath.unlink()