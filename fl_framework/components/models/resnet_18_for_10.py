# fl_framework/components/models/resnet_18_for_10.py

import torch.nn as nn
from torchvision import models
import torch
from pytorch_model_summary import summary

def convert_bn_to_gn(module, num_groups=8):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            if num_channels % num_groups != 0:
                raise ValueError(f"无法将通道数 {num_channels} 分为 {num_groups} 组。请为 {name} 选择一个合适的组数（{num_channels} 的因子）。")
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
            setattr(module, name, gn)
        else:
            convert_bn_to_gn(child, num_groups)
    return module

class ResNet18For10(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes: int = 10,
                 pretrained: bool = False,
                 freeze_backbone: bool = False):
        super().__init__()

        self.backbone = models.resnet18(weights='default' if pretrained else None)
        
        self.backbone.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.backbone = convert_bn_to_gn(self.backbone, num_groups=8)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
        # print(summary(self.backbone, torch.randn(1, 3, 64, 64)))
        # input()

    def forward(self, x):
        return self.backbone(x)

