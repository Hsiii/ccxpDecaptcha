from typing import List

import torch
from torch import nn

DIGITS = 4
FEATURE_HEIGHT = 5
FEATURE_WIDTH = 10


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.depthwise_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.activation(x)
        x = self.pointwise(x)
        x = self.pointwise_bn(x)
        x = self.activation(x)
        return x


class ResidualDSBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableBlock(in_channels, out_channels, stride=stride),
            DepthwiseSeparableBlock(out_channels, out_channels, stride=1),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(x) + self.shortcut(x))


class SpatialAttentionHead(nn.Module):
    def __init__(self, channels: int, digits: int = DIGITS):
        super().__init__()
        self.digits = digits
        self.feature_proj = nn.Linear(channels, channels, bias=False)
        self.query = nn.Parameter(torch.randn(digits, channels) * 0.02)
        self.classifiers = nn.ModuleList(nn.Linear(channels, 10) for _ in range(digits))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        projected_tokens = self.feature_proj(tokens)
        attention_scores = torch.einsum('bnc,dc->bdn', projected_tokens, self.query)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        pooled = torch.einsum('bdn,bnc->bdc', attention_weights, tokens)
        return torch.stack(
            [classifier(pooled[:, idx]) for idx, classifier in enumerate(self.classifiers)],
            dim=1,
        )


class Net(nn.Module):
    def __init__(self, digits: int = DIGITS, channels: int = 128):
        super().__init__()
        self.digits = digits
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            ResidualDSBlock(32, 48, stride=1),
            ResidualDSBlock(48, 64, stride=2),
            ResidualDSBlock(64, 96, stride=2),
            ResidualDSBlock(96, channels, stride=2),
        )
        self.position = nn.Parameter(torch.randn(1, FEATURE_HEIGHT * FEATURE_WIDTH, channels) * 0.02)
        self.attention_head = SpatialAttentionHead(channels, digits=digits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        batch_size, channels, height, width = x.shape
        if height != FEATURE_HEIGHT or width != FEATURE_WIDTH:
            raise ValueError(
                f'Expected backbone feature map {(FEATURE_HEIGHT, FEATURE_WIDTH)}, got {(height, width)}'
            )
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.position
        return self.attention_head(tokens)


def decode_predictions(logits: torch.Tensor) -> List[str]:
    predictions = logits.argmax(dim=-1).detach().cpu().tolist()
    return [''.join(str(digit) for digit in answer) for answer in predictions]
