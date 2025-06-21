import pandas as pd
import os 
from src.logger import get_logger

logger = get_logger(__name__)

os.makedirs("outputs", exist_ok=True)


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the FDA drug product data with selected columns and log memory usage.
    """
    logger.info(f"Loading data from {file_path}...")

    try:
        # Load only required columns to save memory
        df = pd.read_excel(
            file_path,
            usecols=[
                'PROPRIETARYNAME', 'NONPROPRIETARYNAME', 'ROUTENAME',
                'DOSAGEFORMNAME', 'SUBSTANCENAME', 'PHARM_CLASSES',
                'LABELERNAME', 'MARKETINGCATEGORYNAME', 'STARTMARKETINGDATE'
            ]
        )
        logger.info(f"Data loaded successfully with shape {df.shape}")
        memory = df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory usage: {memory:.2f} MB")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
