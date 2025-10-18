# src/models.py
import torch, timm
from torch import nn

def create_model(model_name: str, num_classes: int = 2, pretrained: bool = True, device: str = "cuda"):
    """
    Wrap timm to create a classifier with desired num_classes.
    Example names: efficientnet_b0, densenet121, resnet18, vit_tiny_patch16_224
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model.to(device)

def count_params(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6
