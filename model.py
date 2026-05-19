import torch.nn as nn
from config import HIDDEN_DIM1, HIDDEN_DIM2, LATENT_DIM, DROPOUT_RATE


class AutoEncoder(nn.Module):
    """Fully connected autoencoder for tabular network-traffic anomaly detection."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM1),
            nn.BatchNorm1d(HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
            nn.BatchNorm1d(HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, LATENT_DIM),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM2),
            nn.BatchNorm1d(HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, HIDDEN_DIM1),
            nn.BatchNorm1d(HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM1, input_dim),
            nn.Sigmoid(),  # suitable because features are MinMax scaled to [0, 1]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
