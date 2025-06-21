import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import get_logger

logger = get_logger(__name__)

def clean_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning and preprocessing...")
    os.makedirs("outputs", exist_ok=True)

    # Original distribution plot
    original_distribution = df['MARKETINGCATEGORYNAME'].value_counts()
    plt.figure(figsize=(14, 8))
    ax = original_distribution.plot(kind='bar', color=sns.color_palette("viridis", len(original_distribution)))
    plt.title('Original Marketing Category Distribution')
    plt.xlabel('Marketing Category')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    for i, v in enumerate(original_distribution.values):
        ax.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig('outputs/original_category_distribution.jpg')
    plt.close()

    # Category mapping
    marketing_category_mapping = {
        'OTC MONOGRAPH DRUG': 'OTC',
        'OTC MONOGRAPH FINAL': 'OTC',
        'OTC MONOGRAPH NOT FINAL': 'OTC',
        'UNAPPROVED DRUG FOR USE IN DRUG SHORTAGE': 'UNAPPROVED',
        'UNAPPROVED MEDICAL GAS': 'UNAPPROVED',
        'UNAPPROVED DRUG OTHER': 'UNAPPROVED',
        'UNAPPROVED HOMEOPATHIC': 'UNAPPROVED',
        'NDA': 'NDA',
        'NDA AUTHORIZED GENERIC': 'NDA',
        'ANDA': 'ANDA',
        'BLA': 'BLA'
    }

    df['GROUPED_MARKETING_CATEGORY'] = df['MARKETINGCATEGORYNAME'].map(marketing_category_mapping, na_action='ignore')
    df = df[df['MARKETINGCATEGORYNAME'] != 'EMERGENCY USE AUTHORIZATION']
    df = df.dropna(subset=['GROUPED_MARKETING_CATEGORY'])

    # Fill missing text
    text_columns = ['PROPRIETARYNAME', 'NONPROPRIETARYNAME', 'ROUTENAME', 'DOSAGEFORMNAME', 'SUBSTANCENAME', 'PHARM_CLASSES', 'LABELERNAME']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].fillna('no text')

    # Convert to datetime
    if 'STARTMARKETINGDATE' in df.columns:
        df['STARTMARKETINGDATE'] = pd.to_datetime(df['STARTMARKETINGDATE'], errors='coerce')
        df['MARKETING_YEAR'] = df['STARTMARKETINGDATE'].dt.year

    # Grouped distribution plot
    grouped_distribution = df['GROUPED_MARKETING_CATEGORY'].value_counts()
    plt.figure(figsize=(10, 6))
    ax = grouped_distribution.plot(kind='bar', color=sns.color_palette("viridis", len(grouped_distribution)))
    plt.title('Grouped Marketing Category Distribution')
    plt.xlabel('Marketing Category')
    plt.ylabel('Count')
    for i, v in enumerate(grouped_distribution.values):
        ax.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig('outputs/grouped_category_distribution.jpg')
    plt.close()

    # Missing heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Value Heatmap')
    plt.tight_layout()
    plt.savefig('outputs/missing_value_heatmap.jpg')
    plt.close()

    # Yearly trend
    if 'MARKETING_YEAR' in df.columns:
        plt.figure(figsize=(14, 6))
        year_counts = df['MARKETING_YEAR'].value_counts().sort_index()
        year_counts.plot(kind='line', marker='o')
        plt.title('Year-wise Drug Listing Distribution')
        plt.xlabel('Year')
        plt.ylabel('Number of Listings')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/year_wise_distribution.jpg')
        plt.close()

    df = df.reset_index(drop=True)
    logger.info(f"Data cleaned. Final shape: {df.shape}")
    return df
