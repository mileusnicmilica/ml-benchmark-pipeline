# benchmark/runner.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_model(model: nn.Module, train_loader: DataLoader, epochs: int = 5):
    """
    Trains model and returns loss history and training time.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_history = []
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time
    print(f"Finished in: {training_time:.2f}s")

    return loss_history, training_time


def evaluate_model(model: nn.Module, test_loader: DataLoader):
    """
    Evaluates model accuracy on test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy


def get_model_size(model: nn.Module):
    """
    Returns total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters())