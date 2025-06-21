import os
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from src.logger import get_logger

logger = get_logger(__name__)

# ----------------- Helper Functions -----------------

def save_confusion_matrix(preds, y_true, class_labels, model_name):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name.upper()} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_confusion_matrix.jpg')
    plt.close()

def save_feature_importance(importances, feature_names, model_name):
    plt.figure(figsize=(12, 10))
    indices = np.argsort(importances)[::-1]
    plt.title(f'{model_name.upper()} Feature Importances')
    plt.bar(range(min(20, len(importances))), importances[indices][:20], align='center')
    plt.xticks(range(min(20, len(importances))),
               np.array(feature_names)[indices][:20], rotation=90)
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_feature_importance.jpg')
    plt.close()

def save_roc_curves(y_test_encoded, pred_proba, label_encoder, model_name):
    plt.figure(figsize=(12, 10))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(label_encoder.classes_)):
        fpr[i], tpr[i], _ = roc_curve((y_test_encoded == i).astype(int), pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC for {label_encoder.classes_[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.upper()} ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_roc_curves.jpg')
    plt.close()

# ----------------- Main Function -----------------

def train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_cols):
    logger.info("Starting model training and evaluation...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Encode target labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    logger.info(f"Class mapping: {dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))}")

    # Label encode text features
    text_cols = ['PROPRIETARYNAME', 'DOSAGEFORMNAME', 'ROUTENAME', 'LABELERNAME']
    text_cols = [col for col in text_cols if col in X_train.columns]

    for col in text_cols:
        le = LabelEncoder()
        all_values = pd.concat([X_train[col], X_test[col]]).astype(str)
        le.fit(all_values)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        joblib.dump(le, f'models/{col}_encoder_rf.pkl')

    # ----------- Handle Missing Values -----------
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # ----------- RANDOM FOREST -----------
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=10,
                                      min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_imputed, y_train)
    rf_pred = rf_model.predict(X_test_imputed)
    rf_pred_proba = rf_model.predict_proba(X_test_imputed)
    logger.info("Random Forest Classification Report:\n" + classification_report(y_test, rf_pred))

    save_confusion_matrix(rf_pred, y_test, label_encoder.classes_, 'rf')
    save_feature_importance(rf_model.feature_importances_, X_train.columns, 'rf')
    save_roc_curves(y_test_encoded, rf_pred_proba, label_encoder, 'random_forest')
    joblib.dump(rf_model, 'models/random_forest_model.pkl')

    # ----------- LOGISTIC REGRESSION -----------
    lr_model = LogisticRegression(C=1.0, max_iter=1000, multi_class='multinomial',
                                  solver='saga', n_jobs=-1, random_state=42)
    lr_model.fit(X_train_imputed, y_train)
    lr_pred = lr_model.predict(X_test_imputed)
    lr_pred_proba = lr_model.predict_proba(X_test_imputed)
    logger.info("Logistic Regression Classification Report:\n" + classification_report(y_test, lr_pred))

    save_confusion_matrix(lr_pred, y_test, label_encoder.classes_, 'lr')
    save_roc_curves(y_test_encoded, lr_pred_proba, label_encoder, 'logistic_regression')
    joblib.dump(lr_model, 'models/logistic_regression_model.pkl')

    # ----------- XGBOOST -----------
    xgb_model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
    xgb_model.fit(X_train_imputed, y_train_encoded)
    xgb_pred = xgb_model.predict(X_test_imputed)
    xgb_pred_class = label_encoder.inverse_transform(xgb_pred)
    xgb_pred_proba = xgb_model.predict_proba(X_test_imputed)
    logger.info("XGBoost Classification Report:\n" + classification_report(y_test, xgb_pred_class))

    save_confusion_matrix(xgb_pred_class, y_test, label_encoder.classes_, 'xgb')
    save_feature_importance(xgb_model.feature_importances_, X_train.columns, 'xgb')
    save_roc_curves(y_test_encoded, xgb_pred_proba, label_encoder, 'xgboost')
    joblib.dump(xgb_model, 'models/xgboost_model.pkl')

    # Export predictions
    predictions_df = pd.DataFrame({
        'true_label': y_test.values,
        'rf_prediction': rf_pred,
        'lr_prediction': lr_pred,
        'xgb_prediction': xgb_pred_class
    })
    predictions_df.to_csv('outputs/test_predictions.csv', index=False)

    # SHAP Explainability for RF
    try:
        logger.info("Calculating SHAP values for Random Forest...")
        explainer = shap.TreeExplainer(rf_model)
        sample_size = min(1000, X_test_imputed.shape[0])
        X_sample = X_test_imputed.sample(sample_size, random_state=42)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.savefig('outputs/shap_feature_importance.jpg')
        plt.close()
        shap.summary_plot(shap_values[0], X_sample, show=False)
        plt.savefig('outputs/shap_values_detail.jpg')
        plt.close()
    except Exception as e:
        logger.error(f"SHAP analysis error: {e}")

    del rf_model, lr_model, xgb_model
    gc.collect()
