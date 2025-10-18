from pathlib import Path
import torch

# --- Dataset and Directory Configuration ---

# This should point to the parent directory containing 'chest_xray' or the directory
# containing 'train', 'test', and 'val' folders.
KAGGLE_DATASET_BASE_PATH = Path("C:\\Users\\HP\\Desktop\\SLIIT\\Y4 SEM 1\\DL\\Ass\\Assignment\\DL-project\\ViT\\dataset\\chest_xray")


TRAIN_DIR = KAGGLE_DATASET_BASE_PATH / "train"
TEST_DIR = KAGGLE_DATASET_BASE_PATH / "test"
VAL_DIR = KAGGLE_DATASET_BASE_PATH / "val"

# Class names based on the directory structure (NORMAL, PNEUMONIA)
CLASS_NAMES = {"NORMAL": 0, "PNEUMONIA": 1}

# --- Model and Training Configuration ---
MODEL_NAME = 'vit_base_patch16_224'
NUM_CLASSES = len(CLASS_NAMES) # Should be 2 [NORMAL, PNEUMONIA]
IMG_SIZE = 224
BATCH_SIZE = 16  # ViT is memory-intensive
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 30
T_MAX_LR_SCHEDULER_EPOCHS = 20 # Used to calculate steps for CosineAnnealingLR

# Save/Logging
CHECKPOINT_PATH = 'vit_chest_xray_best.pth'

# ImageNet means and stds for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]