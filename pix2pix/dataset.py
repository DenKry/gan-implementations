import os
import random
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PairedImageDataset(Dataset):
    # Each file is a side-by-side pair [domain_A | domain_B]
    # Default: left=A (satellite), right=B (map) — matches the pix2pix maps dataset
    def __init__(self, file_list, data_dir, transform, swap=False):
        self.file_list = file_list
        self.data_dir  = data_dir
        self.transform = transform
        self.swap      = swap

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        img  = Image.open(os.path.join(self.data_dir, self.file_list[i])).convert("RGB")
        w, h = img.size
        left  = self.transform(img.crop((0,      0, w // 2, h)))
        right = self.transform(img.crop((w // 2, 0, w,      h)))
        return (right, left) if not self.swap else (left, right)


def get_pix2pix_loaders(data_dir, img_size=256, batch_size=4, val_ratio=0.15,
                        num_workers=2, swap=False, seed=42):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    all_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith((".jpg", ".png"))])
    if not all_files:
        raise FileNotFoundError(f"No images found in {data_dir}")

    random.seed(seed)
    random.shuffle(all_files)
    n_val = int(len(all_files) * val_ratio)
    val_files, train_files = all_files[:n_val], all_files[n_val:]

    train_loader = DataLoader(PairedImageDataset(train_files, data_dir, transform, swap),
                              batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(PairedImageDataset(val_files,   data_dir, transform, swap),
                              batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_files)} | Val: {len(val_files)} | Size: {img_size}×{img_size}")
    return train_loader, val_loader
