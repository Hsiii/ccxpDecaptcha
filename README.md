# CCXP Captcha Training

Local-only training pipeline for a NTHU CCXP decaptcha model.

Based on the work by [25349023](https://github.com/25349023), this fork keeps only the local training pipeline and improves the model by replacing fixed-width per-digit slicing with a full-image six-head architecture.

## Current Pipeline

### Data Collection

```bash
pipenv install
python decaptcha/collect_data.py
```

- Download captcha images from CCXP and label them manually.
- Save repeated renders with the same `pwdstr` in the filename so they stay in the same data group.

### Data Preprocessing

```bash
python decaptcha/image_splitting.py
```

- Load each captcha as one full RGB image.
- Build `captcha_images.npy`, `captcha_labels.npy`, and `captcha_groups.npy`.
- Split train and validation by captcha group instead of by individual digit crop.
- Apply training augmentation with affine transform, color jitter, blur, and random erasing.

### Model Structure

```bash
python decaptcha/training.py
```

- Input: full captcha image at native size.
- Backbone:
  `Conv2d(3->24, stride=2) -> BatchNorm2d -> ReLU`
  `DepthwiseConv2d -> BatchNorm2d -> ReLU -> PointwiseConv2d -> BatchNorm2d -> ReLU`
  `DepthwiseConv2d(stride=2) -> BatchNorm2d -> ReLU -> PointwiseConv2d -> BatchNorm2d -> ReLU`
  `DepthwiseConv2d -> BatchNorm2d -> ReLU -> PointwiseConv2d -> BatchNorm2d -> ReLU`
- Head:
  `AdaptiveAvgPool2d((1, 6)) -> six independent Linear(...->10) heads`
- Output: 6 digit logits, one head per position.
- Validation metrics: exact-sequence accuracy and per-digit accuracy.
