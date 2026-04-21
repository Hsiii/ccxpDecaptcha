# ccxpDecaptcha

Local-only training pipeline for the NTHU CCXP captcha model.

Based on the work by [25349023](https://github.com/25349023), this fork keeps only the local training workflow and replaces fixed-width per-digit slicing with a full-image six-head architecture.

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
- Captchas are rendered inline in the terminal.
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
python decaptcha/relabel_data.py 410892
```

- The command previews one render from the group in the terminal first.
- With one positional argument, it relabels the latest submitted batch.
- Use `--edit-latest 410892` if you want that behavior to be explicit.
- Use `--dry-run` to inspect the rename plan without modifying files.
- Use `--old-label` if the group already contains mixed labels.

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

Reported metrics:

- image-level exact-sequence accuracy
- group-level majority-vote exact-sequence accuracy
- per-digit accuracy
- per-position accuracy
