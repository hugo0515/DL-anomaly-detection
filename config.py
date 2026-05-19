import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Choose dataset: "nsl_kdd" or "unsw_nb15"
DATASET = "nsl_kdd"

DATASET_PATHS = {
    "nsl_kdd": {
        "train": BASE_DIR / "dataset" / "NSL-KDD" / "KDDTrain+.txt",
        "test": BASE_DIR / "dataset" / "NSL-KDD" / "KDDTest+.txt",
        "model": "autoencoder_nslkdd.pth",
        "threshold": "threshold_nslkdd.npy",
    },
    "unsw_nb15": {
        "train": BASE_DIR / "dataset" / "UNSW_NB15" / "UNSW_NB15_training-set.csv",
        "test": BASE_DIR / "dataset" / "UNSW_NB15" / "UNSW_NB15_testing-set.csv",
        "model": "autoencoder_unsw_nb15.pth",
        "threshold": "threshold_unsw_nb15.npy",
    },
}

TRAIN_PATH = DATASET_PATHS[DATASET]["train"]
TEST_PATH = DATASET_PATHS[DATASET]["test"]
AE_MODEL_PATH = DATASET_PATHS[DATASET]["model"]
THRESHOLD_PATH = DATASET_PATHS[DATASET]["threshold"]

# Model parameters
LATENT_DIM = 32
HIDDEN_DIM1 = 128
HIDDEN_DIM2 = 64
DROPOUT_RATE = 0.3

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_STATE = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
