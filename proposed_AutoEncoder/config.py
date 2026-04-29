# Dataset paths
TRAIN_PATH = "dataset/KDDTrain+.txt"
TEST_PATH = "dataset/KDDTest+.txt"

# Model parameters
INPUT_DIM = None  # will be set automatically after preprocessing
LATENT_DIM = 32   # bottleneck dimension
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
DROPOUT_RATE = 0.3

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Device configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threshold path (to store reconstruction error threshold)
THRESHOLD_PATH = "threshold_value.npy"

#trained model path
AE_MODEL_PATH = "autoencoder_nslkdd.pth"
