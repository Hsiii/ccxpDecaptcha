# CCXP Captcha Training

Local-only training pipeline for the NTHU CCXP six-head captcha model.

## Scope

This repo only keeps:

- captcha download and manual labeling
- dataset generation for full-image grouped samples
- six-head model definition
- local training and export

## Setup

```bash
pipenv install
```

## Workflow

1. Collect labeled captcha images into `raw_data/`:

```bash
python decaptcha/collect_data.py
```

Each saved filename keeps the source `pwdstr` so repeated renders from the same captcha stay grouped together.

2. Build the training arrays:

```bash
python decaptcha/image_splitting.py
```

This writes:

- `captcha_images.npy`
- `captcha_labels.npy`
- `captcha_groups.npy`

3. Train the six-head model:

```bash
python decaptcha/training.py
```

This writes:

- `decaptcha.pt`
- `decaptcha.int8.pt`

## Model Notes

- Input is the full captcha image at native size.
- The backbone is a lightweight CNN.
- Adaptive pooling produces six positions.
- Each position has an independent digit head.
- Validation is tracked with exact-sequence accuracy and per-digit accuracy.
