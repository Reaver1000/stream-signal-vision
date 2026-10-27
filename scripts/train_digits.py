"""Train a tiny digit classifier (MNIST) for reading the POSITION number.

The streamer panel uses a clean sans-serif digit; a small CNN reads it in a few
milliseconds, which OCR libraries fail at on an isolated glyph. Trains once and
caches to runs/digits_cnn.pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "runs" / "digits_cnn.pt"


class DigitNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train(epochs: int = 4, device: str | None = None) -> Path:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(str(REPO_ROOT / "data" / "mnist"), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(REPO_ROOT / "data" / "mnist"), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512)

    model = DigitNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            total += 1
        model.eval()
        correct = 0
        seen = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += int((predictions == labels).sum())
                seen += labels.numel()
        print(f"epoch {epoch + 1}/{epochs}: test accuracy {correct / seen:.4f}", flush=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"saved {MODEL_PATH}")
    return MODEL_PATH


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    train(epochs)
