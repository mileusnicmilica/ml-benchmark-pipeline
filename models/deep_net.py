# models/deep_net.py
import torch
import torch.nn as nn


class DeepNet(nn.Module):
    """
    Deeper fully-connected network with BatchNorm for MNIST.
    More layers than LinearNet, but no convolutions like CNN.
    Architecture: 784 -> 512 -> 256 -> 128 -> 64 -> 10
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.layers(x)