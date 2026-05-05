import torch
import torch.nn as nn
from config import HIDDEN_DIM1, HIDDEN_DIM2, LATENT_DIM, DROPOUT_RATE

class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, LATENT_DIM)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM2, HIDDEN_DIM1),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM1, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
