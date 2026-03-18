import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset

CIFAR10_NAMES = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck",
}


class RemappedSubset(Dataset):
    def __init__(self, subset, label_map):
        self.subset = subset
        self.label_map = label_map

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        img, lbl = self.subset[i]
        return img, self.label_map[lbl]


def get_cifar10_loaders(data_dir="./data", selected_classes=None, batch_size=64,
                        img_size=32, num_workers=2):
    if selected_classes is None:
        selected_classes = list(range(10))

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])

    full_train = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    full_test  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    def filter_classes(dataset, classes):
        idx = np.where(np.isin(dataset.targets, classes))[0]
        return Subset(dataset, idx)

    label_map   = {orig: new for new, orig in enumerate(selected_classes)}
    class_names = {new: CIFAR10_NAMES[orig] for new, orig in enumerate(selected_classes)}

    train_ds = RemappedSubset(filter_classes(full_train, selected_classes), label_map)
    test_ds  = RemappedSubset(filter_classes(full_test,  selected_classes), label_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader, label_map, class_names
