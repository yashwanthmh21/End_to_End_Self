import pandas as pd
import numpy as np
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import get_logger

logger = get_logger(__name__)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering...")

    df['GROUPED_MARKETING_CATEGORY'] = df['GROUPED_MARKETING_CATEGORY'].astype('category')

    for col in ['PROPRIETARYNAME', 'DOSAGEFORMNAME', 'SUBSTANCENAME', 'NONPROPRIETARYNAME']:
        if col in df.columns:
            df[f'{col}_LENGTH'] = df[col].str.len()
            df[f'{col}_WORD_COUNT'] = df[col].str.split().str.len()

    if 'SUBSTANCENAME' in df.columns:
        df['ACTIVE_INGREDIENT_COUNT'] = df['SUBSTANCENAME'].str.count(';') + 1
        df.loc[df['SUBSTANCENAME'] == 'no text', 'ACTIVE_INGREDIENT_COUNT'] = 0

    if 'PHARM_CLASSES' in df.columns:
        df['PHARM_CLASS_COUNT'] = df['PHARM_CLASSES'].str.count(';') + 1
        df.loc[df['PHARM_CLASSES'] == 'no text', 'PHARM_CLASS_COUNT'] = -1

        key_classes = [
            'analgesic', 'antibiotic', 'antiviral', 'cardiovascular',
            'central nervous system', 'anti-inflammatory', 'hormone',
            'vaccine', 'immunologic'
        ]
        for cls in key_classes:
            df[f'IS_{cls.upper().replace(" ", "_")}'] = df['PHARM_CLASSES'].str.contains(cls, case=False, na=False).astype(np.int8)

    if 'SUBSTANCENAME' in df.columns:
        logger.info("Generating TF-IDF features for SUBSTANCENAME...")
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['SUBSTANCENAME'].fillna('no text'))

        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'TFIDF_{i}' for i in range(tfidf_matrix.shape[1])]
        )
        tfidf_df.index = df.index
        df = pd.concat([df, tfidf_df], axis=1)

    return df  
