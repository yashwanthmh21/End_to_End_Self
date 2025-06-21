from src.data_ingestion import load_data
from src.data_preprocessing import clean_and_preprocess_data
from src.feature_engineering import engineer_features
from src.data_split import prepare_data_for_modeling
from src.model_trainer import train_and_evaluate_models
from src.deep_learning_trainer import train_deep_learning_model

def main():
    print("Pipeline started")

    # Loading raw data
    df = load_data("data_\\fda_product.xlsx")

    # Cleaning and preprocess data
    df_clean = clean_and_preprocess_data(df)

    # Features engineer
    df_features = engineer_features(df_clean)

    # Spliting data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data_for_modeling(df_features)

    # Train ML models
    train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_cols)

    # Train Deep Learning model
    train_deep_learning_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
