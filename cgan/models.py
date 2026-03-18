import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_classes, latent_dim=100, embed_dim=50, img_channels=3, img_size=32):
        super().__init__()
        assert img_size % 8 == 0
        self.latent_dim = latent_dim
        init_size = img_size // 8

        self.label_emb = nn.Embedding(num_classes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 512 * init_size * init_size),
            nn.Unflatten(1, (512, init_size, init_size)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d( 64, img_channels, 3, 1, 1), nn.Tanh(),
        )

    def forward(self, z, labels):
        return self.net(torch.cat([z, self.label_emb(labels)], dim=1))


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_channels=3, img_size=32):
        super().__init__()
        assert img_size % 16 == 0
        self.img_size = img_size
        final_size = img_size // 16

        self.label_emb = nn.Embedding(num_classes, img_size * img_size)
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + 1,  64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d( 64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(512 * final_size * final_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, 1, self.img_size, self.img_size)
        return self.net(torch.cat([img, label_map], dim=1))
