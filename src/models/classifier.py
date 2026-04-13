"""Factory de classifieurs basée sur timm."""
from __future__ import annotations

import torch.nn as nn

MODEL_ALIASES = {
    "efficientnet_b3": "tf_efficientnet_b3.ns_jft_in1k",
    "efficientnet_b0": "efficientnet_b0",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "vit_b16": "vit_base_patch16_224",
    "resnet50": "resnet50",
}


def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    import timm
    timm_name = MODEL_ALIASES.get(name, name)
    return timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
