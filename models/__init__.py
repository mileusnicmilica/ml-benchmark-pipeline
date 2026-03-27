# models/__init__.py
from .linear_net import LinearNet
from .cnn_net import CNNNet
from .deep_net import DeepNet

__all__ = ["LinearNet", "CNNNet", "DeepNet"]