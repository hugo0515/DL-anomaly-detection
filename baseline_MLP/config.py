# config.py

# Dataset paths
TRAIN_PATH = "dataset/KDDTrain+.txt"
TEST_PATH = "dataset/KDDTest+.txt"

# Model parameters
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
DROPOUT_RATE = 0.3

# Training parameters                                     ;.
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Threshold for binary classification
THRESHOLD = 0.5

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
