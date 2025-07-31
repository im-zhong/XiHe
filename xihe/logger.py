# 2025/7/31
# zhangzhong

import logging


def create_logger(filename: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to a file and the console.

    Args:
        filename (str): The name of the file to write logs to.
        level (int): The logging level (default is INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter and add it to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
