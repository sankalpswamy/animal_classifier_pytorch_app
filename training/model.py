from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, MobileNet_V2_Weights


def build_model(config: dict, num_classes: int, train_backbone: bool = False) -> nn.Module:
    backbone_name = config["model"].get("backbone", "efficientnet_b0").lower()
    dropout = float(config["model"].get("dropout_rate", 0.3))
    dense_units = int(config["model"].get("dense_units", 256))

    if backbone_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )
        backbone_params = model.features.parameters()
    elif backbone_name == "mobilenetv2":
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes),
        )
        backbone_params = model.features.parameters()
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    for param in backbone_params:
        param.requires_grad = train_backbone
    return model


def unfreeze_top_layers(model: nn.Module, backbone_name: str, unfreeze_last_blocks: int = 2) -> None:
    backbone_name = backbone_name.lower()
    if backbone_name == "efficientnet_b0":
        blocks = list(model.features.children())
    elif backbone_name == "mobilenetv2":
        blocks = list(model.features.children())
    else:
        return

    for block in blocks:
        for p in block.parameters():
            p.requires_grad = False
    for block in blocks[-max(1, unfreeze_last_blocks):]:
        for p in block.parameters():
            p.requires_grad = True
