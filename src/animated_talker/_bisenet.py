"""BiSeNet face parsing model for semantic segmentation.

Adapted from: https://github.com/zllrunning/face-parsing.PyTorch
Used for generating compositing masks in the lip-sync pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNReLU(in_ch, out_ch)
        self.conv_atten = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.bn_atten(self.conv_atten(atten))
        atten = torch.sigmoid(atten)
        return feat * atten


class ContextPath(nn.Module):
    def __init__(self, pretrained_resnet: str | None = None):
        super().__init__()
        net = resnet18(weights=None)
        if pretrained_resnet:
            state = torch.load(pretrained_resnet, map_location="cpu", weights_only=True)
            net.load_state_dict(state)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)

        self.conv_head32 = ConvBNReLU(128, 128)
        self.conv_head16 = ConvBNReLU(128, 128)

        self.conv_avg = ConvBNReLU(512, 128, ks=1, padding=0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.relu(self.bn1(self.conv1(x)))
        feat = self.maxpool(feat)
        feat8 = self.layer2(self.layer1(feat))
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)

        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, size=feat32.shape[2:], mode="nearest")

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, size=feat16.shape[2:], mode="nearest")
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, size=feat8.shape[2:], mode="nearest")
        feat16_up = self.conv_head16(feat16_up)

        return feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.convblk = ConvBNReLU(in_ch, out_ch, ks=1, padding=0)
        self.conv1 = nn.Conv2d(out_ch, out_ch // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_ch // 4, out_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fsp: torch.Tensor, fcp: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(feat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.relu(self.conv1(atten))
        atten = torch.sigmoid(self.conv2(atten))
        return feat + feat * atten


class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBNReLU(3, 64, stride=2)
        self.conv2 = ConvBNReLU(64, 128, stride=2)
        self.conv3 = ConvBNReLU(128, 256, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(self.conv1(x)))


class BiSeNet(nn.Module):
    """BiSeNet for face parsing with 19 semantic classes."""

    def __init__(self, n_classes: int = 19, pretrained_resnet: str | None = None):
        super().__init__()
        self.sp = SpatialPath()
        self.cp = ContextPath(pretrained_resnet=pretrained_resnet)
        self.ffm = FeatureFusionModule(256 + 128, 256)
        self.conv_out = nn.Sequential(
            ConvBNReLU(256, 256),
            nn.Conv2d(256, n_classes, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        feat_sp = self.sp(x)
        feat_cp16, _ = self.cp(x)
        feat_fuse = self.ffm(feat_sp, feat_cp16)
        out = self.conv_out(feat_fuse)
        out = F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=True)
        return (out,)
