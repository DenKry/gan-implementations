"""
CycleGAN — training script.

Usage:
  python train.py --domain_a_dir ./horse2zebra/trainA --domain_b_dir ./horse2zebra/trainB
  python train.py --from_paired --data_dir ./maps/train
  python train.py --domain_a_dir ./domainA --domain_b_dir ./domainB --img_size 128 --n_blocks 6
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import ResNetGenerator, CycleDiscriminator, ImageBuffer
from dataset import get_cyclegan_loaders, SplitPairedDataset


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def save_samples(G_AB, G_BA, loader_a, loader_b, device, path, epoch):
    G_AB.eval(); G_BA.eval()
    with torch.no_grad():
        real_A = next(iter(loader_a)).to(device)
        real_B = next(iter(loader_b)).to(device)
        fake_B = G_AB(real_A).cpu()
        fake_A = G_BA(real_B).cpu()
    N = min(4, real_A.size(0), real_B.size(0))
    fig, axes = plt.subplots(4, N, figsize=(N * 3.5, 14))
    fig.suptitle(f"CycleGAN — Epoch {epoch}", fontsize=13)
    for r, (label, row) in enumerate(zip(
        ["Domain A", "G_AB(A)", "Domain B", "G_BA(B)"],
        [real_A.cpu(), fake_B, real_B.cpu(), fake_A],
    )):
        for c in range(N):
            axes[r, c].imshow(denorm(row[c]).permute(1, 2, 0)); axes[r, c].axis("off")
        axes[r, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=60, va="center")
    plt.tight_layout(); plt.savefig(path); plt.close()
    G_AB.train(); G_BA.train()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    if args.from_paired:
        ds_a = SplitPairedDataset(args.data_dir, transform, side="left")
        ds_b = SplitPairedDataset(args.data_dir, transform, side="right")
        loader_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        loader_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        print(f"Domain A: {len(ds_a)} | Domain B: {len(ds_b)}")
    else:
        loader_a, loader_b = get_cyclegan_loaders(
            args.domain_a_dir, args.domain_b_dir, args.img_size, args.batch_size)

    G_AB = ResNetGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    G_BA = ResNetGenerator(ngf=args.ngf, n_blocks=args.n_blocks).to(device)
    D_A  = CycleDiscriminator(ndf=args.ndf).to(device)
    D_B  = CycleDiscriminator(ndf=args.ndf).to(device)

    criterion_GAN      = nn.MSELoss()  # LSGAN
    criterion_cycle    = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    opt_G   = optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Linear LR decay starting at the halfway epoch
    def lr_lambda(epoch):
        decay_start = args.epochs // 2
        return 1.0 if epoch < decay_start else 1.0 - (epoch - decay_start) / (args.epochs - decay_start)

    sched_G  = optim.lr_scheduler.LambdaLR(opt_G,   lr_lambda)
    sched_DA = optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda)
    sched_DB = optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda)

    buf_A = ImageBuffer()
    buf_B = ImageBuffer()
    os.makedirs(args.save_dir, exist_ok=True)

    g_losses, da_losses, db_losses = [], [], []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        g_ep = da_ep = db_ep = 0.0

        for real_A, real_B in zip(loader_a, loader_b):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            bs = real_A.size(0)
            ph = real_A.shape[2] // 16
            pw = real_A.shape[3] // 16
            ones  = torch.ones (bs, 1, ph, pw, device=device)
            zeros = torch.zeros(bs, 1, ph, pw, device=device)

            fake_B = G_AB(real_A); fake_A = G_BA(real_B)
            rec_A  = G_BA(fake_B); rec_B  = G_AB(fake_A)
            idt_A  = G_BA(real_A); idt_B  = G_AB(real_B)

            loss_G = (
                criterion_GAN(D_B(fake_B), ones) +
                criterion_GAN(D_A(fake_A), ones) +
                args.lambda_cycle    * (criterion_cycle(rec_A, real_A)    + criterion_cycle(rec_B, real_B)) +
                args.lambda_identity * (criterion_identity(idt_A, real_A) + criterion_identity(idt_B, real_B))
            )
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            fake_A_buf = buf_A.push_and_pop(fake_A.detach())
            loss_D_A = 0.5 * (criterion_GAN(D_A(real_A),     ones) +
                               criterion_GAN(D_A(fake_A_buf), zeros))
            opt_D_A.zero_grad(); loss_D_A.backward(); opt_D_A.step()

            fake_B_buf = buf_B.push_and_pop(fake_B.detach())
            loss_D_B = 0.5 * (criterion_GAN(D_B(real_B),     ones) +
                               criterion_GAN(D_B(fake_B_buf), zeros))
            opt_D_B.zero_grad(); loss_D_B.backward(); opt_D_B.step()

            g_ep  += loss_G.item()
            da_ep += loss_D_A.item()
            db_ep += loss_D_B.item()

        n = len(loader_a)
        g_losses.append(g_ep / n); da_losses.append(da_ep / n); db_losses.append(db_ep / n)
        sched_G.step(); sched_DA.step(); sched_DB.step()

        print(f"Epoch {epoch:3d}/{args.epochs} | G: {g_losses[-1]:.4f} | D_A: {da_losses[-1]:.4f} | D_B: {db_losses[-1]:.4f} | {time.time()-start:.0f}s")

        if epoch % args.log_interval == 0:
            save_samples(G_AB, G_BA, loader_a, loader_b, device,
                         os.path.join(args.save_dir, f"epoch_{epoch:03d}.png"), epoch)

    torch.save(G_AB.state_dict(), os.path.join(args.save_dir, "G_AB.pth"))
    torch.save(G_BA.state_dict(), os.path.join(args.save_dir, "G_BA.pth"))
    torch.save(D_A.state_dict(),  os.path.join(args.save_dir, "D_A.pth"))
    torch.save(D_B.state_dict(),  os.path.join(args.save_dir, "D_B.pth"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(g_losses, label="Generator"); ax1.set_title("Generator"); ax1.legend()
    ax2.plot(da_losses, label="D_A"); ax2.plot(db_losses, label="D_B")
    ax2.set_title("Discriminators"); ax2.legend()
    for ax in (ax1, ax2): ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "losses.png")); plt.close()
    print(f"Saved to {args.save_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain_a_dir", type=str, default="./domainA")
    p.add_argument("--domain_b_dir", type=str, default="./domainB")
    p.add_argument("--from_paired", action="store_true")
    p.add_argument("--data_dir", type=str, default="./maps/train")
    p.add_argument("--save_dir", type=str, default="./samples")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=9)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_identity", type=float, default=5.0)
    p.add_argument("--log_interval", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
