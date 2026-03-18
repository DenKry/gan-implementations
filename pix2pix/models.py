import torch
import torch.nn as nn


def enc_block(in_ch, out_ch, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def dec_block(in_ch, out_ch, dropout=False):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False), nn.BatchNorm2d(out_ch)]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64):
        super().__init__()
        self.e1 = enc_block(in_ch,  ngf,    norm=False)
        self.e2 = enc_block(ngf,    ngf*2)
        self.e3 = enc_block(ngf*2,  ngf*4)
        self.e4 = enc_block(ngf*4,  ngf*8)
        self.e5 = enc_block(ngf*8,  ngf*8)
        self.e6 = enc_block(ngf*8,  ngf*8)
        self.e7 = enc_block(ngf*8,  ngf*8)
        self.e8 = enc_block(ngf*8,  ngf*8, norm=False)
        self.d1 = dec_block(ngf*8,  ngf*8, dropout=True)
        self.d2 = dec_block(ngf*16, ngf*8, dropout=True)
        self.d3 = dec_block(ngf*16, ngf*8, dropout=True)
        self.d4 = dec_block(ngf*16, ngf*8)
        self.d5 = dec_block(ngf*16, ngf*4)
        self.d6 = dec_block(ngf*8,  ngf*2)
        self.d7 = dec_block(ngf*4,  ngf)
        self.out = nn.Sequential(nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1=self.e1(x);  e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        e5=self.e5(e4); e6=self.e6(e5); e7=self.e7(e6); e8=self.e8(e7)
        d=self.d1(e8)
        d=self.d2(torch.cat([d,e7],1)); d=self.d3(torch.cat([d,e6],1))
        d=self.d4(torch.cat([d,e5],1)); d=self.d5(torch.cat([d,e4],1))
        d=self.d6(torch.cat([d,e3],1)); d=self.d7(torch.cat([d,e2],1))
        return self.out(torch.cat([d,e1],1))


class PatchGAN(nn.Module):
    # in_ch = source(3) + target(3) = 6 by default
    def __init__(self, in_ch=6, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  ndf,   4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf,    ndf*2, 4, 2, 1), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2,  ndf*4, 4, 2, 1), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4,  ndf*8, 4, 1, 1), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8,  1,     4, 1, 1),
        )

    def forward(self, source, target):
        return self.net(torch.cat([source, target], dim=1))
