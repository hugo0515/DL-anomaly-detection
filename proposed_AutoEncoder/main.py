import os
import torch
from config import DEVICE
from train import train_autoencoder
from test import evaluate_autoencoder

def main():
    print(f"\n🚀 Device in use: {DEVICE}\n")

    model_path = "autoencoder_nslkdd.pth"

    # Check if model already trained
    if os.path.exists(model_path):
        print("✅ Found existing trained model. Skipping training...\n")
    else:
        print("🧠 Training AutoEncoder model from scratch...\n")
        train_autoencoder()
        print("✅ Model training completed.\n")

    print("🔍 Starting evaluation on test dataset...\n")
    evaluate_autoencoder()

if __name__ == "__main__":
    main()
