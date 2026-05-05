# quick_test.py (temporary file)
from train import train_autoencoder
from test import evaluate_autoencoder

print("🚀 Starting quick training for 3 epochs just to verify pipeline...\n")
train_autoencoder()

print("\n🔍 Evaluating trained model...\n")
evaluate_autoencoder()
