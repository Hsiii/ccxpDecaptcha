from typing import List

import torch
from torch import nn

DIGITS = 6


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SixHeadCaptchaNet(nn.Module):
    def __init__(self, digits: int = DIGITS, channels: int = 64):
        super().__init__()
        self.digits = digits
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            DepthwiseSeparableBlock(24, 32, stride=1),
            DepthwiseSeparableBlock(32, 48, stride=2),
            DepthwiseSeparableBlock(48, channels, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, digits))
        self.heads = nn.ModuleList(nn.Linear(channels, 10) for _ in range(digits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(x)).squeeze(2).permute(0, 2, 1)
        return torch.stack([head(features[:, idx]) for idx, head in enumerate(self.heads)], dim=1)


def decode_predictions(logits: torch.Tensor) -> List[str]:
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    return [''.join(str(digit) for digit in answer) for answer in predictions]
