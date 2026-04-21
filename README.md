# ccxpDecaptcha

Local-only training pipeline for the NTHU CCXP captcha model.

Based on the work by [25349023](https://github.com/25349023), this fork keeps only the local training workflow and replaces fixed-width per-digit slicing with a full-image six-head architecture. The model is then re-trained from scratch on a new dataset of 500+ manually labeled captcha renders collected in 2026, achieving a test set exact-sequence accuracy of **97.7%**, and test set digit-level accuracy of **99.6%**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow

1. Collect labeled captchas:

```bash
python decaptcha/collect_data.py
```

- Download captcha images from CCXP and label them manually.
- Captchas will be rendered inline in the terminal.
- Repeated renders from the same `pwdstr` stay grouped in the saved filename.

2. Build the dataset arrays:

```bash
python decaptcha/build_dataset.py
```

This writes:

- `captcha_images.npy`
- `captcha_labels.npy`
- `captcha_groups.npy`

If you mislabel a captcha batch, relabel the whole grouped filename set before rebuilding:

```bash
python decaptcha/relabel_data.py --latest
```

3. Train and evaluate:

```bash
python decaptcha/train_model.py
```

Training behavior:

- grouped `train/val/test` split by captcha `pwdstr`
- train split capped to `20` renders per captcha group
- group-balanced weighted sampling during training
- `ReduceLROnPlateau` scheduler on validation loss
- early stopping after `8` stale validation epochs
- best checkpoint selected by validation exact-sequence accuracy

Outputs:

- `decaptcha_best_val_seq.pt`
- `decaptcha_last.pt`
- `decaptcha.int8.pt`
- `val_failures.csv`
- `test_failures.csv`
- `val_confusion_matrix.npy`
- `test_confusion_matrix.npy`
