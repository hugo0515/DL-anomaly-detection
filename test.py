import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

from config import *
from model import AutoEncoder
from preprocess import load_and_preprocess
from preprocess_unsw import load_and_preprocess_unsw
from utils import load_threshold, reconstruction_errors


def load_dataset():
    if DATASET == "nsl_kdd":
        return load_and_preprocess(TRAIN_PATH, TEST_PATH)
    if DATASET == "unsw_nb15":
        return load_and_preprocess_unsw(TRAIN_PATH, TEST_PATH)
    raise ValueError(f"Unsupported DATASET: {DATASET}")


def evaluate_autoencoder(model_path=AE_MODEL_PATH, threshold_path=THRESHOLD_PATH):
    X_train, y_train, X_test, y_test, X_val, y_val = load_dataset()

    model = AutoEncoder(input_dim=X_test.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    threshold = load_threshold(threshold_path)
    test_errors = reconstruction_errors(model, X_test, DEVICE)
    y_pred = (test_errors > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    print("\nFinal Test Evaluation")
    print(f"Dataset   : {DATASET}")
    print(f"Threshold : {threshold:.6f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    return {
        "dataset": DATASET,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }
