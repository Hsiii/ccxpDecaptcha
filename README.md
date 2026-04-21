# CCXP Captcha Training

Local-only training pipeline for the NTHU CCXP six-head captcha model.

Based on the original CCXP decaptcha work by the previous author, with this branch narrowing the repo to local training only and improving model performance by moving from fixed-width per-digit slicing to a full-image six-head architecture.

## Scope

This repo only keeps:

- captcha download and manual labeling
- dataset generation for full-image grouped samples
- six-head model definition
- local training and export

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow

1. Collect labeled captcha images into `raw_data/`:

```bash
python decaptcha/collect_labeled_captchas.py
```

Each saved filename keeps the source `pwdstr` so repeated renders from the same captcha stay grouped together.

2. Build the training arrays:

```bash
python decaptcha/build_dataset.py
```

This writes:

- `captcha_images.npy`
- `captcha_labels.npy`
- `captcha_groups.npy`

3. Train the six-head model:

```bash
python decaptcha/train_six_head_model.py
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
