import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import AutoEncoder
from config import *
from preprocess import load_and_preprocess
from preprocess_unsw import load_and_preprocess_unsw
from utils import compute_threshold, save_threshold


def train_autoencoder():
    # 1. Load and preprocess data
    # NSL-KDD dataset
    # X_train, y_train, X_test, y_test, X_val, y_val = load_and_preprocess(
    #     TRAIN_PATH, TEST_PATH
    # )

    # UNSW-NB15 dataset
    X_train, y_train, X_test, y_test, X_val, y_val = load_and_preprocess_unsw(
        TRAIN_PATH, TEST_PATH
    )

    # Only use normal traffic for training
    X_train_normal = X_train[y_train == 0]
    print(f"Training on {X_train_normal.shape[0]} normal samples...")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0

        for batch in train_loader:
            inputs = batch[0].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], "
            f"Loss: {epoch_loss / len(train_loader):.6f}"
        )

    torch.save(model.state_dict(), "autoencoder_nslkdd.pth")

    # Threshold optimization using validation set
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

        reconstructed = model(X_val_tensor)

        val_errors = torch.mean(
            (reconstructed-X_val_tensor) ** 2,
            dim=1
        ).cpu().numpy()

        # threshold = compute_threshold(val_errors, y_val)
        #
        # save_threshold(threshold)

        threshold, best_f1 = compute_threshold(val_errors, y_val)

        save_threshold(threshold)

        print(f"\nBest threshold: {threshold:.6f}")
        print(f"Validation F1-score: {best_f1:.4f}")

        # print(f"\nThreshold optimized on validation set and saved: {threshold:.6f}")
