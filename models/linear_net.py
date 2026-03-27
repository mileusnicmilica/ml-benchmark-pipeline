# models/linear_net.py
import torch
import torch.nn as nn


class LinearNet(nn.Module):
    """
    Simple fully-connected neural network for MNIST classification.
    Architecture: 784 -> 128 -> 64 -> 10
    Serves as the baseline model in our benchmark.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)