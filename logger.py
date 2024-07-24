import logging
import sys

# Setup basic configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)  # Ensure logs are directed to stdout
    ]
)


def setup_logger(name=__name__):
    logger = logging.getLogger(name)
    return logger
