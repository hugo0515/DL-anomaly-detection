from torchviz import make_dot
import torch
from model import AutoEncoder

# Create dummy input
input_dim = 122  # adjust based on NSL-KDD after preprocessing
model = AutoEncoder(input_dim)
x = torch.randn(1, input_dim)

# Visualize computation graph
dot = make_dot(model(x), params=dict(model.named_parameters()))
dot.render("autoencoder_architecture", format="png")
