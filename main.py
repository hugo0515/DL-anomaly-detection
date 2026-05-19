from pathlib import Path
from config import DEVICE, AE_MODEL_PATH, THRESHOLD_PATH
from train import train_autoencoder
from test import evaluate_autoencoder


def main():
    print(f"\nDevice in use: {DEVICE}\n")

    model_exists = Path(AE_MODEL_PATH).exists()
    threshold_exists = Path(THRESHOLD_PATH).exists()

    if model_exists and threshold_exists:
        print("Found existing trained model and threshold. Skipping training.\n")
    else:
        print("Training AutoEncoder model from scratch...\n")
        train_autoencoder()
        print("\nModel training completed.\n")

    print("Starting final evaluation on the test dataset...\n")
    evaluate_autoencoder()


if __name__ == "__main__":
    main()
