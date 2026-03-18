"""
Pix2Pix — training script.

Usage:
  python train.py --data_dir ./maps/train --download_maps
  python train.py --data_dir ./facades/train --epochs 200
  python train.py --data_dir ./maps/train --swap
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import UNetGenerator, PatchGAN
from dataset import get_pix2pix_loaders


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def save_samples(G, val_loader, device, path, epoch):
    G.eval()
    with torch.no_grad():
        src, tgt = next(iter(val_loader))
        fake = G(src.to(device)).cpu()
    N = min(4, src.size(0))
    fig, axes = plt.subplots(3, N, figsize=(N * 3.5, 10))
    fig.suptitle(f"Pix2Pix — Epoch {epoch}", fontsize=13)
    for i in range(N):
        axes[0, i].imshow(denorm(src[i]).permute(1, 2, 0));  axes[0, i].axis("off")
        axes[1, i].imshow(denorm(fake[i]).permute(1, 2, 0)); axes[1, i].axis("off")
        axes[2, i].imshow(denorm(tgt[i]).permute(1, 2, 0));  axes[2, i].axis("off")
    for label, ax in zip(["Input", "Generated", "Target"], axes[:, 0]):
        ax.set_ylabel(label, fontsize=11, rotation=0, labelpad=45, va="center")
    plt.tight_layout(); plt.savefig(path); plt.close()
    G.train()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.download_maps and not os.path.exists(args.data_dir):
        subprocess.check_call(["wget", "-q", "--show-progress",
                               "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz"])
        subprocess.check_call(["tar", "-xzf", "maps.tar.gz"])
        os.remove("maps.tar.gz")

    train_loader, val_loader = get_pix2pix_loaders(
        args.data_dir, args.img_size, args.batch_size, args.val_ratio, swap=args.swap)

    G = UNetGenerator(in_ch=3, out_ch=3, ngf=args.ngf).to(device)
    D = PatchGAN(in_ch=6, ndf=args.ndf).to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1  = nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.save_dir, exist_ok=True)
    g_losses, d_losses = [], []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        g_ep = d_ep = 0.0
        G.train(); D.train()

        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            fake = G(src)

            real_pred = D(src, tgt)
            fake_pred = D(src, fake.detach())
            loss_D = (criterion_GAN(real_pred, torch.ones_like(real_pred)) +
                      criterion_GAN(fake_pred, torch.zeros_like(fake_pred))) * 0.5
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            fake_pred = D(src, fake)
            loss_G = criterion_GAN(fake_pred, torch.ones_like(fake_pred)) + args.lambda_l1 * criterion_L1(fake, tgt)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            d_ep += loss_D.item()
            g_ep += loss_G.item()

        d_losses.append(d_ep / len(train_loader))
        g_losses.append(g_ep / len(train_loader))
        print(f"Epoch {epoch:3d}/{args.epochs} | D: {d_losses[-1]:.4f} | G: {g_losses[-1]:.4f} | {time.time()-start:.0f}s")

        if epoch % args.log_interval == 0:
            save_samples(G, val_loader, device,
                         os.path.join(args.save_dir, f"epoch_{epoch:03d}.png"), epoch)

    torch.save(G.state_dict(), os.path.join(args.save_dir, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(args.save_dir, "discriminator.pth"))

    plt.figure(figsize=(9, 4))
    plt.plot(g_losses, label="Generator (GAN + L1)")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Pix2Pix training losses")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "losses.png")); plt.close()
    print(f"Saved to {args.save_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./maps/train")
    p.add_argument("--save_dir", type=str, default="./samples")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_l1", type=float, default=100.0)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--log_interval", type=int, default=25)
    p.add_argument("--swap", action="store_true")
    p.add_argument("--download_maps", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
