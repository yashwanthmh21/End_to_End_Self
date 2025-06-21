import logging
import os
from datetime import datetime

def get_logger(name):
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{datetime.now().strftime('%Y-%m-%d')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)
