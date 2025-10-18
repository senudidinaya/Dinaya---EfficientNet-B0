import random
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Import config.py file
from .config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, TRAIN_DIR, TEST_DIR, VAL_DIR, CLASS_NAMES


# -------------------- Dataset Class --------------------
class ChestXrayDataset(Dataset):
    """
    Custom Dataset for Chest X-Ray images.
    """

    def __init__(self, files: List[Path], labels: List[int], transform: transforms.Compose = None):
        self.files = files
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Open and convert to RGB (important for models like ViT)
        img = Image.open(self.files[idx]).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label


# -------------------- Data Transformations --------------------

# Use data augmentation only on the training set.
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Validation transform: Resize, CenterCrop, ToTensor, Normalize
val_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.15)), # Resize to a larger size first
    transforms.CenterCrop(IMG_SIZE),         # Then crop to the target size
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# -------------------- Data Preparation Utility  --------------------

def get_file_paths_and_labels(data_dir: Path) -> Tuple[List[Path], List[int], np.ndarray]:
    """
    Collects image file paths and corresponding labels from a directory.
    Assumes structure: data_dir/CLASS_NAME/image.jpeg   [structure in config file]
    """
    all_files: List[Path] = []
    all_labels: List[int] = []
    class_counts = {}

    class_to_label = CLASS_NAMES  # Use the mapping from config

    for class_name, label in class_to_label.items():
        class_path = data_dir / class_name

        # Check for multiple common image extensions, especially '.jpeg' for this dataset
        files = (
                list(class_path.glob("*.jpeg")) +
                list(class_path.glob("*.png")) +
                list(class_path.glob("*.jpg"))
        )

        # Check if the directory exists and contains files
        if not files:
            print(f"Warning: No image files found in {class_path}")

        all_files.extend(files)
        all_labels.extend([label] * len(files))
        class_counts[class_name] = len(files)

    if not all_files:
        raise FileNotFoundError(f"No files loaded from directory: {data_dir}. Check your path in config.py.")

    # Calculate class weights (Inverse frequency weighting)
    total_samples = len(all_labels)
    # Ensure keys in class_counts match the order of labels [0, 1]
    counts = np.array([class_counts[name] for name, label in sorted(class_to_label.items(), key=lambda item: item[1])])

    # Check for zero counts to avoid division by zero
    if 0 in counts:
        print("Warning: One or more classes have zero samples. Using uniform weights.")
        class_weights = np.ones_like(counts, dtype=np.float32)
    else:
        # Inverse frequency: N_total / (N_classes * N_class)
        class_weights = total_samples / (len(class_to_label) * counts)

    print(f"Loaded {len(all_files)} samples from {data_dir}. Class counts: {class_counts}")
    print(f"Calculated class weights: {class_weights} (Index 0: NORMAL, Index 1: PNEUMONIA)")

    # Zip, shuffle, and unzip
    combined = list(zip(all_files, all_labels))
    random.seed(42)  # Ensure file path loading is consistent
    random.shuffle(combined)

    shuffled_files, shuffled_labels = zip(*combined)

    return list(shuffled_files), list(shuffled_labels), class_weights


if __name__ == '__main__':

    train_files, train_labels, train_weights_np = get_file_paths_and_labels(TRAIN_DIR)

    test_files, test_labels, _ = get_file_paths_and_labels(TEST_DIR)

    val_files, val_labels, _ = get_file_paths_and_labels(VAL_DIR)
