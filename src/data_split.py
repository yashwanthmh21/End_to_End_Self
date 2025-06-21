import gc
from sklearn.model_selection import train_test_split
from src.logger import get_logger

logger = get_logger(__name__)

def prepare_data_for_modeling(df):
    """
    Select features, encode target, split into train/test, and optimize memory.
    """
    logger.info("Preparing data for modeling...")

    # Define column types
    text_cols = ['PROPRIETARYNAME', 'DOSAGEFORMNAME', 'ROUTENAME', 'LABELERNAME']
    length_cols = [col for col in df.columns if '_LENGTH' in col]
    count_cols = [col for col in df.columns if '_COUNT' in col]
    binary_cols = [col for col in df.columns if 'IS_' in col]
    tfidf_cols = [col for col in df.columns if 'TFIDF' in col]
    other_cols = ['MARKETING_YEAR']

    all_feature_cols = text_cols + length_cols + count_cols + binary_cols + tfidf_cols + other_cols
    feature_cols = [col for col in all_feature_cols if col in df.columns]

    # Features and target
    X = df[feature_cols]
    y = df['GROUPED_MARKETING_CATEGORY']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Log class distribution
    logger.info("Class distribution in training set:")
    logger.info(y_train.value_counts(normalize=True).to_string())

    logger.info("Class distribution in test set:")
    logger.info(y_test.value_counts(normalize=True).to_string())

    logger.info(f"Train set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")

    # Memory cleanup
    del X, y
    gc.collect()

    return X_train, X_test, y_train, y_test, feature_cols
