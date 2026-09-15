"""EfficientNet-B0 backbone adapted for arbitrary input channel counts (2 or 4
multispectral bands instead of 3 RGB channels), plus an optional input-level
Squeeze-and-Excitation block that produces an interpretable per-band importance
weight (Phase 4 / thesis objective 3.1).
"""
import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .config import CHANNEL_GROUPS


def _adapt_first_conv(model, in_channels, pretrained):
    old_conv = model.features[0][0]
    if old_conv.in_channels == in_channels:
        return
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, bias=old_conv.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)  # (out,1,k,k)
            new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
    model.features[0][0] = new_conv


def build_efficientnet(in_channels, num_classes, pretrained=True):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    _adapt_first_conv(model, in_channels, pretrained)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


class InputChannelSE(nn.Module):
    """Squeeze-and-Excitation applied directly to the raw input bands, so the
    learned excitation weight is a direct, per-wavelength importance score."""

    def __init__(self, channels, reduction=2):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        squeezed = self.pool(x).view(b, c)
        weights = self.fc(squeezed).view(b, c, 1, 1)
        return x * weights, weights.view(b, c)


class EfficientNetWithChannelSE(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True, reduction=2):
        super().__init__()
        self.se = InputChannelSE(in_channels, reduction)
        self.backbone = build_efficientnet(in_channels, num_classes, pretrained)
        self.last_channel_weights = None

    def forward(self, x):
        x, weights = self.se(x)
        self.last_channel_weights = weights.detach()
        return self.backbone(x)


def build_model(channel_group, num_classes=3, pretrained=True, use_se=False, se_reduction=2):
    in_channels = len(CHANNEL_GROUPS[channel_group])
    if use_se:
        return EfficientNetWithChannelSE(in_channels, num_classes, pretrained, se_reduction)
    return build_efficientnet(in_channels, num_classes, pretrained)


if __name__ == "__main__":
    for group in CHANNEL_GROUPS:
        m = build_model(group, use_se=False)
        x = torch.randn(2, len(CHANNEL_GROUPS[group]), 224, 224)
        out = m(x)
        print(group, "no-SE ->", out.shape)

    m = build_model("full", use_se=True)
    x = torch.randn(2, 4, 224, 224)
    out = m(x)
    print("full + SE ->", out.shape, "channel weights:", m.last_channel_weights)
