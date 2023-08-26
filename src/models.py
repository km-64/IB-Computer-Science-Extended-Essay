import torch
from torch import nn


class Generator(nn.Module):
    def __init__(
        self, z_dim=100, img_size=28, n_channels=1, n_classes=10, embed_size=100
    ):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(n_classes, embed_size)
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim + embed_size, 256, 4, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, n_channels, 5, 2, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, y):
        embedding = self.embed(y).unsqueeze(2).unsqueeze(3)
        z = torch.cat([z, embedding], dim=1)
        return self.gen(z)


class Critic(nn.Module):
    def __init__(self, n_channels=1, n_classes=10, img_size=28):
        super(Critic, self).__init__()
        self.img_size = img_size
        self.embed = nn.Embedding(n_classes, img_size * img_size)
        self.critic = nn.Sequential(
            nn.Conv2d(n_channels + 1, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x, y):
        embedding = self.embed(y).view(y.shape[0], 1, self.img_size, self.img_size)
        x = torch.cat([x, embedding], dim=1)
        return self.critic(x)
