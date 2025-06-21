from src.data_ingestion import load_data

df = load_data("data/raw/fda_data.xlsx")
print(df.head())
