# GAN Implementations: Pix2Pix · CycleGAN · cGAN

Three generative adversarial network architectures implemented **from scratch** in PyTorch, based on the original papers:

- **Pix2Pix** — Isola et al., [*Image-to-Image Translation with Conditional Adversarial Networks*](https://arxiv.org/abs/1611.07004), 2017
- **CycleGAN** — Zhu et al., [*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*](https://arxiv.org/abs/1703.10593), 2017
- **cGAN** — Mirza & Osindero, [*Conditional Generative Adversarial Nets*](https://arxiv.org/abs/1411.1784), 2014

All models are fully parameterized — not tied to any specific dataset, class count, or resolution.

**→ [See training results, generated samples, and metric comparisons in RESULTS.md](RESULTS.md)**

---

## Structure

```
├── pix2pix/
│   ├── models.py     # UNetGenerator, PatchGAN (in_ch, out_ch, ngf, ndf)
│   ├── dataset.py    # Generic paired image dataset
│   └── train.py      # Training script with argparse
├── cyclegan/
│   ├── models.py     # ResNetGenerator, CycleDiscriminator, ImageBuffer
│   ├── dataset.py    # Unpaired single-domain dataset
│   └── train.py      # Training script with argparse
└── cgan/
    ├── models.py     # Generator, Discriminator (num_classes, latent_dim, img_size)
    ├── dataset.py    # CIFAR-10 loader with configurable class selection
    └── train.py      # Training script with argparse
```

---

## Setup

```bash
pip install -r requirements.txt
```

> All commands below should be run from the **repo root** directory.

---

## 1. Pix2Pix

Supervised image-to-image translation using paired images.

**Architecture:**
- Generator: U-Net with 8-level encoder-decoder + skip connections
- Discriminator: PatchGAN (classifies 70×70 patches)
- Loss: Adversarial (BCE) + L1 pixel reconstruction (λ=100)

**Quick start:**

```bash
# Maps dataset (auto-download — requires wget, Linux/macOS only)
python pix2pix/train.py --data_dir ./maps/train --download_maps

# Facades dataset (manual download from pix2pix authors)
python pix2pix/train.py --data_dir ./facades/train --epochs 200

# Reverse translation direction
python pix2pix/train.py --data_dir ./maps/train --swap
```

**Dataset format:** Each image file should be a side-by-side pair `[domain_A | domain_B]`.
Compatible with all datasets from the [pix2pix project](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `./maps/train` | Directory of paired images |
| `--img_size` | `256` | Image resolution |
| `--lambda_l1` | `100.0` | L1 reconstruction weight |
| `--epochs` | `100` | Training epochs |
| `--swap` | `False` | Reverse A→B to B→A |

---

## 2. CycleGAN

Unsupervised image-to-image translation — no paired data required.

**Architecture:**
- Generators: Two ResNet-9 generators (G_AB: A→B, G_BA: B→A)
- Discriminators: Two PatchGAN discriminators (D_A, D_B)
- Loss: Adversarial (LSGAN/MSE) + Cycle consistency (λ=10) + Identity (λ=5)

**Quick start:**

```bash
# Two separate unpaired directories
python cyclegan/train.py --domain_a_dir ./horse2zebra/trainA --domain_b_dir ./horse2zebra/trainB

# From a pix2pix-style dataset, treated as unpaired
python cyclegan/train.py --from_paired --data_dir ./maps/train

# 128×128 with 6 ResNet blocks (faster)
python cyclegan/train.py --domain_a_dir ./domainA --domain_b_dir ./domainB --img_size 128 --n_blocks 6
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--domain_a_dir` | `./domainA` | Domain A image directory |
| `--domain_b_dir` | `./domainB` | Domain B image directory |
| `--from_paired` | `False` | Use pix2pix-style data as unpaired |
| `--n_blocks` | `9` | ResNet blocks (9 for 256px, 6 for 128px) |
| `--lambda_cycle` | `10.0` | Cycle consistency weight |
| `--lambda_identity` | `5.0` | Identity loss weight |
| `--epochs` | `50` | Training epochs |

---

## 3. Conditional GAN (cGAN)

Generates images conditioned on a class label. Trained on CIFAR-10 (configurable classes).

**Architecture:**
- Generator: noise + label embedding → ConvTranspose2d upsampling → 32×32 RGB
- Discriminator: image + label map → Conv2d → real/fake score

**Quick start:**

```bash
# Train on bird, cat, dog (CIFAR-10 classes 2, 3, 5)
python cgan/train.py --selected_classes 2 3 5

# Train on all 10 CIFAR-10 classes
python cgan/train.py --selected_classes 0 1 2 3 4 5 6 7 8 9

# Train on airplane and ship only, 100 epochs
python cgan/train.py --selected_classes 0 8 --epochs 100
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--selected_classes` | `2 3 5` | CIFAR-10 class indices |
| `--latent_dim` | `100` | Noise vector size |
| `--embed_dim` | `50` | Label embedding size |
| `--img_size` | `32` | Image resolution (must be ÷8) |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `64` | Batch size |

CIFAR-10 class index reference: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck

---

## Results

See **[RESULTS.md](RESULTS.md)** for:
- Generated sample grids at different training epochs
- Quantitative metrics: FID, SSIM, PSNR
- Pix2Pix vs CycleGAN visual and numerical comparison

Sample outputs are also saved to `--save_dir` (default `./samples/`) during training.
