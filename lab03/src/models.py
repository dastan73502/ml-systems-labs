import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.net(x)


class PatchTransformer(nn.Module):
    def __init__(self, patch=7, d_model=64, nhead=4, dim_feedforward=128, num_layers=2):
        super().__init__()
        self.patch = patch
        self.seq = (28 // patch) ** 2
        self.embed = nn.Linear(patch * patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.seq, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def patchify(self, x):
        b = x.shape[0]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)
        return x.reshape(b, -1, p * p)

    def forward(self, x):
        x = self.embed(self.patchify(x)) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


MODELS = {"MLP": MLP, "CNN": CNN, "Attention": PatchTransformer}


def build(name, seed=0):
    torch.manual_seed(seed)
    return MODELS[name]()
