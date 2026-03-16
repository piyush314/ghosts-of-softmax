"""Model architectures for ghost envelope experiment."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(x)


class MLP(nn.Module):
    def __init__(self, w=128):
        super().__init__()
        self.fc1 = nn.Linear(64, w)
        self.ln1 = nn.LayerNorm(w)
        self.fc2 = nn.Linear(w, w)
        self.ln2 = nn.LayerNorm(w)
        self.fc3 = nn.Linear(w, 10)

    def forward(self, x):
        x = F.gelu(self.ln1(self.fc1(x)))
        x = F.gelu(self.ln2(self.fc2(x)))
        return self.fc3(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.gelu(self.c1(x))
        x = F.max_pool2d(x, 2)
        x = F.gelu(self.c2(x))
        x = F.max_pool2d(x, 2)
        return self.fc2(F.gelu(self.fc1(x.flatten(1))))


class TinyTfm(nn.Module):
    def __init__(self, d=32, depth=2, heads=4):
        super().__init__()
        self.emb = nn.Linear(8, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 9, d))
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4*d,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 10)
        nn.init.normal_(self.cls, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x):
        b = x.shape[0]
        t = self.emb(x.view(b, 8, 8))
        h = torch.cat([self.cls.expand(b, -1, -1), t], 1) + self.pos
        h = self.enc(h)
        return self.head(self.norm(h[:, 0]))


ARCHS = {
    "Linear": (Linear, 6e-3, 500),
    "MLP": (MLP, 2.5e-3, 520),
    "CNN": (CNN, 2.5e-3, 560),
    "TinyTfm": (TinyTfm, 2e-3, 600),
}
