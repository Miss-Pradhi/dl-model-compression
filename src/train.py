import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds,  batch_size=batch_size))

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in loader:
            # Always use CPU for evaluation — safe for both normal + quantized models
            X, y = X.cpu(), y.cpu()
            try:
                out = model.cpu()(X)
            except Exception:
                out = model(X.to(device))
            correct += (out.argmax(1) == y).sum().item()
    return correct / len(loader.dataset)