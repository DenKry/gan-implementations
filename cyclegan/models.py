import random
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False), nn.InstanceNorm2d(channels), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False), nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    # n_blocks: 9 for 256px input, 6 for 128px
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf,   ngf*2, 3, stride=2, padding=1, bias=False), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=False), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
            *[ResBlock(ngf*4) for _ in range(n_blocks)],
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7), nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CycleDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  ndf,   4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf,    ndf*2, 4, 2, 1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2,  ndf*4, 4, 2, 1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4,  ndf*8, 4, 1, 1), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8,  1,     4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class ImageBuffer:
    # Stores past generated images and randomly returns old vs new
    # to stabilise discriminator training (prevents overfitting to latest G outputs)
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, batch):
        result = []
        for img in batch:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                result.append(self.data[idx].clone())
                self.data[idx] = img
            else:
                result.append(img)
        return torch.cat(result)
