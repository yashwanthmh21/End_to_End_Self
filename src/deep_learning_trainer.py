import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.logger import get_logger

logger = get_logger(__name__)

def train_deep_learning_model(X_train, X_test, y_train, y_test):
    logger.info("\nTraining Deep Learning Model...")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if tf.config.list_physical_devices('GPU'):
        logger.info("GPU is available. Using GPU for training.")
    else:
        logger.info("GPU is not available. Using CPU for training.")

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    n_classes = len(label_encoder.classes_)
    y_train_onehot = to_categorical(y_train_encoded, num_classes=n_classes)
    y_test_onehot = to_categorical(y_test_encoded, num_classes=n_classes)

    class_mapping = dict(zip(range(n_classes), label_encoder.classes_))
    logger.info(f"Class mapping: {class_mapping}")

    text_cols = ['PROPRIETARYNAME', 'DOSAGEFORMNAME', 'ROUTENAME', 'LABELERNAME']
    text_cols = [col for col in text_cols if col in X_train.columns]
    numeric_cols = [col for col in X_train.columns if col not in text_cols]

    X_train[numeric_cols] = X_train[numeric_cols].fillna(0)
    X_test[numeric_cols] = X_test[numeric_cols].fillna(0)

    X_train_numeric = X_train[numeric_cols].values
    X_test_numeric = X_test[numeric_cols].values

    numeric_mean = X_train_numeric.mean(axis=0)
    numeric_std = X_train_numeric.std(axis=0)
    numeric_std[numeric_std == 0] = 1
    X_train_numeric = (X_train_numeric - numeric_mean) / numeric_std
    X_test_numeric = (X_test_numeric - numeric_mean) / numeric_std

    np.save('models/numeric_mean.npy', numeric_mean)
    np.save('models/numeric_std.npy', numeric_std)

    numeric_input = Input(shape=(X_train_numeric.shape[1],), name='numeric_input')
    x = Dense(128, activation='relu')(numeric_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=numeric_input, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train_numeric, y_train_onehot,
        epochs=20,
        batch_size=256,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test_numeric, y_test_onehot, verbose=0)
    logger.info(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

    dl_pred_proba = model.predict(X_test_numeric)
    dl_pred_class = np.argmax(dl_pred_proba, axis=1)
    dl_pred = label_encoder.inverse_transform(dl_pred_class)

    # Training history plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('outputs/dl_training_history.jpg')
    plt.close()

    logger.info("\nDeep Learning Classification Report:\n" + classification_report(y_test, dl_pred))

    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, dl_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Deep Learning Model Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/dl_confusion_matrix.jpg')
    plt.close()

    plt.figure(figsize=(12, 10))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_test_encoded == i).astype(int), dl_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve for {class_mapping[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Deep Learning: ROC Curves per Class')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('outputs/deep_learning_roc_curves.jpg')
    plt.close()

    model.save('models/dl_model.keras')
    logger.info("Deep Learning model saved to models/dl_model.keras")

    del model, history
    gc.collect()
