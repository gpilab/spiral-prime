"""
Logging configuration for Spiral-PRIME.
"""

import logging
import sys
from pathlib import Path


def setup_logging(level="INFO", log_file=None, verbose=True):
    """
    Configure logging for Spiral-PRIME.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        verbose: If True, show detailed formatter
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger("spiral_prime")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    if verbose:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default: spiral_prime)
        
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger("spiral_prime")
    else:
        return logging.getLogger(f"spiral_prime.{name}")
