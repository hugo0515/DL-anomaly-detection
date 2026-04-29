# visualize_results.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model import AutoEncoder
from config import *
from preprocess import load_and_preprocess
from utils import load_threshold

def visualize_errors():
    X_train, X_test, y_train, y_test = load_and_preprocess(TRAIN_PATH, TEST_PATH)
    input_dim = X_test.shape[1]

    model = AutoEncoder(input_dim).to(DEVICE)
    model.load_state_dict(torch.load("autoencoder_nslkdd.pth", map_location=DEVICE))
    model.eval()

    threshold = load_threshold()

    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        reconstructed = model(X_test_tensor)
        mse = torch.mean((reconstructed - X_test_tensor) ** 2, dim=1).cpu().numpy()

    # Separate normal and anomaly errors
    normal_errors = mse[y_test == 0]
    anomaly_errors = mse[y_test == 1]

    # Plot
    plt.figure(figsize=(8, 5))
    sns.kdeplot(normal_errors, label='Normal', fill=True, color='green')
    sns.kdeplot(anomaly_errors, label='Anomaly', fill=True, color='red')
    plt.axvline(threshold, color='blue', linestyle='--', label=f'Threshold = {threshold:.4f}')
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('MSE Reconstruction Error')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    visualize_errors()
