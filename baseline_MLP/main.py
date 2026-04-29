# main.py
import os
import torch
from train import *
from test import *
from config import DEVICE

def main():
    print(f"\n🚀 Device in use: {DEVICE}\n")

    model_path = "mlp_nslkdd.pth"

    # Check if model already trained
    if os.path.exists(model_path):
        print("✅ Found existing trained model. Skipping training...\n")
    else:
        print("🧠 Training model from scratch...\n")
        from train import X_train, y_train, X_test, y_test, model
        # Training loop is already inside train.py
        print("✅ Model training completed.\n")

    print("🔍 Starting evaluation on test dataset...\n")
    from test import precision, recall, f1
    print(f"✅ Evaluation Results:\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\n")

if __name__ == "__main__":
    main()
