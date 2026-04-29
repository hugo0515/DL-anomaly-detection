import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from model import AutoEncoder
from config import *
from preprocess import load_and_preprocess
from utils import load_threshold

def evaluate_autoencoder():
    # Load data
    X_train, y_train, X_test, y_test = load_and_preprocess(TRAIN_PATH, TEST_PATH)

    input_dim = X_test.shape[1]
    model = AutoEncoder(input_dim).to(DEVICE)
    model.load_state_dict(torch.load("autoencoder_nslkdd.pth", map_location=DEVICE))
    model.eval()

    threshold = load_threshold()

    with torch.no_grad():
        reconstructed = model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE))
        mse = torch.mean((reconstructed - torch.tensor(X_test, dtype=torch.float32).to(DEVICE)) ** 2, dim=1).cpu().numpy()

    y_pred = (mse > threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print("\n📊 Confusion Matrix:")
    print(cm)
    print("\n🧾 Classification Report:")
    print(report)
