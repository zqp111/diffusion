import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, out_encode=True):
        super(AutoEncoder, self).__init__()
        self.out_encode = out_encode
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )


    def forward(self, x):
        b, c, h ,w = x.shape
        x = x.view(b, -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(b, c, h ,w)
        if self.out_encode:
            return encoded, decoded
        return decoded