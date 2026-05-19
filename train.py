import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from config import *
from model import AutoEncoder
from preprocess import load_and_preprocess
from preprocess_unsw import load_and_preprocess_unsw
from utils import compute_threshold, save_threshold, reconstruction_errors, set_seed


def load_dataset():
    if DATASET == "nsl_kdd":
        return load_and_preprocess(TRAIN_PATH, TEST_PATH)
    if DATASET == "unsw_nb15":
        return load_and_preprocess_unsw(TRAIN_PATH, TEST_PATH)
    raise ValueError(f"Unsupported DATASET: {DATASET}")


def train_autoencoder(seed=RANDOM_STATE, model_path=AE_MODEL_PATH, threshold_path=THRESHOLD_PATH):
    set_seed(seed)

    X_train, y_train, X_test, y_test, X_val, y_val = load_dataset()
    X_train_normal = X_train[y_train == 0]

    print(f"Dataset: {DATASET}")
    print(f"Training on {X_train_normal.shape[0]} normal samples only.")
    print(f"Input dimension: {X_train.shape[1]}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    model = AutoEncoder(input_dim=X_train.shape[1]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_threshold = None

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for (inputs,) in train_loader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(X_train_normal)

        # Validation-only threshold optimization. No test-set leakage.
        val_errors = reconstruction_errors(model, X_val, DEVICE)
        threshold, val_f1 = compute_threshold(val_errors, y_val)
        y_val_pred = (val_errors > threshold).astype(int)

        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_accuracy = accuracy_score(y_val, y_val_pred)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            torch.save(model.state_dict(), model_path)
            save_threshold(best_threshold, threshold_path)

        print(
            f"Epoch [{epoch:02d}/{NUM_EPOCHS}] | "
            f"Loss: {avg_loss:.6f} | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Threshold: {threshold:.6f}"
        )

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    print(f"Best threshold saved to: {threshold_path}")
    print(f"Best model saved to: {model_path}")
    return model
