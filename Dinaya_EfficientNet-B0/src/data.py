# src/data.py
import os, torch, random, numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_to3 = transforms.Lambda(lambda img: img.convert("RGB"))

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_transforms(img_size: int, augment: bool = True):
    if augment:
        train_tf = transforms.Compose([
            _to3, transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(7),
            transforms.ColorJitter(0.08, 0.08, 0.08, 0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_tf = transforms.Compose([
            _to3, transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    test_tf = transforms.Compose([
        _to3, transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, test_tf

def build_loaders(
    data_root: str,
    img_size: int = 224,
    batch_size: int = 32,
    augment: bool = True,
    workers: int | None = None,
    pin_memory: bool = True,
):
    """Assumes Chest X-Ray dataset folder structure with train/val/test subfolders."""
    workers = 2 if workers is None else workers
    train_tf, test_tf = build_transforms(img_size, augment)

    tr = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
    va = datasets.ImageFolder(os.path.join(data_root, "val"),   transform=test_tf)
    te = datasets.ImageFolder(os.path.join(data_root, "test"),  transform=test_tf)

    train_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=pin_memory)
    val_loader   = DataLoader(va, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    test_loader  = DataLoader(te, batch_size=batch_size, shuffle=False,
                              num_workers=workers, pin_memory=pin_memory)
    return train_loader, val_loader, test_loader, tr.classes
