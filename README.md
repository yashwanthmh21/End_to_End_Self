#  End-to-End FDA Product Classification Pipeline

##  Why this project?

The U.S. FDA (Food and Drug Administration) maintains records of thousands of pharmaceutical products. These records include a wide variety of fields like proprietary names, active ingredients, dosage forms, pharmacological classes, and more.

However, classifying these products into appropriate marketing categories (`ANDA`, `NDA`, `OTC`, `BLA`, `UNAPPROVED`) based on such data can be challenging due to text-heavy fields and unstructured patterns. This project aims to solve that problem using an automated machine learning pipeline.

---

##  What does it do?

This project builds an **end-to-end machine learning system** that:

- Loads raw Excel data containing drug product information
- Cleans and preprocesses the dataset
- Engineers numerical and text-based features (TF-IDF, word counts, etc.)
- Trains multiple classification models:
  - Random Forest
  - Logistic Regression
  - XGBoost
- Evaluates performance with metrics like:
  - Confusion matrix
  - Classification report
  - ROC curves
- Uses **SHAP** to interpret model predictions
- Outputs predictions, metrics, and feature importance plots

---

## ⚙️ How does it work?

###  Project Structure

End_to_End_Self/
├── data_/ # Excel input data
├── models/ # Saved trained models
├── outputs/ # Plots and evaluation reports
├── src/
│ ├── data_ingestion.py # Loads Excel data
│ ├── data_preprocessing.py # Cleans and prepares data
│ ├── feature_engineering.py # Extracts features (including TF-IDF)
│ ├── data_split.py # Train-test splitting
│ ├── model_trainer.py # Trains and evaluates models
│ └── logger.py # Custom logging setup
├── main.py # Orchestrates the entire pipeline
├── requirements.txt # Package dependencies
└── README.md


### 🧪 Pipeline Flow

1. **Start the pipeline**
   ```bash
   python main.py


