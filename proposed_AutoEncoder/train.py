import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from model import AutoEncoder
from config import *
from preprocess import load_and_preprocess  # assuming same as baseline
from utils import compute_threshold, save_threshold

def train_autoencoder():
    # 1. Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess(TRAIN_PATH, TEST_PATH)
    
    # Only use normal traffic for training
    X_train_normal = X_train[y_train == 0]  # assuming 0 = normal
    print(f"Training on {X_train_normal.shape[0]} normal samples...")

    # 2. Prepare dataset and loader
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_normal, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # 3. Initialize model
    input_dim = X_train.shape[1]
    model = AutoEncoder(input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training loop
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
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}")

    # 5. Save model
    torch.save(model.state_dict(), "autoencoder_nslkdd.pth")

    # 6. Compute threshold
    model.eval()
    with torch.no_grad():
        reconstructed = model(torch.tensor(X_train_normal, dtype=torch.float32).to(DEVICE))
        errors = torch.mean((reconstructed - torch.tensor(X_train_normal, dtype=torch.float32).to(DEVICE))**2, dim=1).cpu().numpy()
        threshold = compute_threshold(errors)
        save_threshold(threshold)
        print(f"\n✅ Threshold for anomaly detection saved: {threshold:.6f}")
