from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
BATCH_SIZE = 128
LAB_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = LAB_DIR.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS = LAB_DIR / "results"
RESULTS.mkdir(exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,)),
])


def load_datasets():
    full_train = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)
    train, val = random_split(full_train, [55_000, 5_000], generator=torch.Generator().manual_seed(SEED))
    return train, val, test


def load_loaders(batch_size=BATCH_SIZE):
    train, val, test = load_datasets()
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train_loader, val_loader, test_loader
