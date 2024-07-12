import logging
import sys


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Create handler
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter and add it to handler
    formatter = logging.Formatter(
        "%Y-%m-%d %H:%M:%S - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(handler)

    return logger
