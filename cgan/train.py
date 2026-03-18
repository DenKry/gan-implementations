"""
Conditional GAN — training script.

Usage:
  python train.py --selected_classes 2 3 5           # bird, cat, dog
  python train.py --selected_classes 0 1 2 3 4 5 6 7 8 9
  python train.py --selected_classes 0 8 --img_size 64 --epochs 100
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models import Generator, Discriminator
from dataset import get_cifar10_loaders


def denorm(t):
    return t * 0.5 + 0.5


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, label_map, class_names = get_cifar10_loaders(
        data_dir=args.data_dir,
        selected_classes=args.selected_classes,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    num_classes = len(args.selected_classes)
    print(f"Classes ({num_classes}): {class_names}")

    G = Generator(num_classes, args.latent_dim, args.embed_dim, 3, args.img_size).to(device)
    D = Discriminator(num_classes, 3, args.img_size).to(device)

    criterion = nn.BCELoss()
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))

    os.makedirs(args.save_dir, exist_ok=True)

    fixed_z    = torch.randn(num_classes * 8, args.latent_dim, device=device)
    fixed_lbls = torch.arange(num_classes, device=device).repeat_interleave(8)

    g_losses, d_losses = [], []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        g_ep = d_ep = 0.0

        for real_imgs, labels in train_loader:
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            bs = real_imgs.size(0)

            real = torch.ones(bs, 1, device=device)
            fake = torch.zeros(bs, 1, device=device)

            z = torch.randn(bs, args.latent_dim, device=device)
            loss_D = (criterion(D(real_imgs, labels), real) +
                      criterion(D(G(z, labels).detach(), labels), fake)) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            z = torch.randn(bs, args.latent_dim, device=device)
            loss_G = criterion(D(G(z, labels), labels), real)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            d_ep += loss_D.item()
            g_ep += loss_G.item()

        d_losses.append(d_ep / len(train_loader))
        g_losses.append(g_ep / len(train_loader))

        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} | D: {d_losses[-1]:.4f} | G: {g_losses[-1]:.4f} | {time.time()-start:.0f}s")
            G.eval()
            with torch.no_grad():
                samples = G(fixed_z, fixed_lbls).cpu()
            G.train()
            grid = make_grid(denorm(samples), nrow=8)
            plt.figure(figsize=(12, num_classes * 1.6))
            plt.imshow(grid.permute(1, 2, 0)); plt.axis("off")
            plt.title(f"Epoch {epoch} — {' · '.join(class_names.values())}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, f"epoch_{epoch:03d}.png"))
            plt.close()

    torch.save(G.state_dict(), os.path.join(args.save_dir, "generator.pth"))
    torch.save(D.state_dict(), os.path.join(args.save_dir, "discriminator.pth"))

    plt.figure(figsize=(9, 4))
    plt.plot(g_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("cGAN training losses")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "losses.png"))
    plt.close()
    print(f"Saved to {args.save_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--selected_classes", type=int, nargs="+", default=[2, 3, 5],
                   help="CIFAR-10 indices (0=airplane,1=auto,2=bird,3=cat,4=deer,5=dog,6=frog,7=horse,8=ship,9=truck)")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--save_dir", type=str, default="./samples")
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--embed_dim", type=int, default=50)
    p.add_argument("--img_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
