import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class UnpairedImageDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir  = data_dir
        self.transform = transform
        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        if not self.files:
            raise FileNotFoundError(f"No images in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.transform(Image.open(os.path.join(self.data_dir, self.files[i])).convert("RGB"))


class SplitPairedDataset(Dataset):
    # Extracts one domain from side-by-side paired images (pix2pix format)
    def __init__(self, data_dir, transform, side="left"):
        assert side in ("left", "right")
        self.data_dir  = data_dir
        self.transform = transform
        self.side      = side
        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img  = Image.open(os.path.join(self.data_dir, self.files[i])).convert("RGB")
        w, h = img.size
        crop = img.crop((0, 0, w // 2, h)) if self.side == "left" else img.crop((w // 2, 0, w, h))
        return self.transform(crop)


def get_cyclegan_loaders(domain_a_dir, domain_b_dir, img_size=256, batch_size=1, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    ds_a = UnpairedImageDataset(domain_a_dir, transform)
    ds_b = UnpairedImageDataset(domain_b_dir, transform)
    loader_a = DataLoader(ds_a, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    loader_b = DataLoader(ds_b, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    print(f"Domain A: {len(ds_a)} | Domain B: {len(ds_b)} | Size: {img_size}×{img_size}")
    return loader_a, loader_b
