# model.py
import torch
import torch.nn as nn
from config import HIDDEN_DIM1, HIDDEN_DIM2, DROPOUT_RATE

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM1),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM1, HIDDEN_DIM2),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.model(x)