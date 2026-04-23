# ccxpDecaptcha

A training pipeline for NTHU CCXP decaptcha model.

Based on the work by [25349023](https://github.com/25349023), this fork enhances the local training workflow and the model architecture with a new cropped full-image six-head model. The model is retrained from scratch on a new dataset of 600 manually labeled captcha group (30000 images) collected in 2026 April, achieving significantly improved performance of 99.96% six-digit sequence accuracy and 99.99% individual digit accuracy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow

### 1. Collect captcha data
Download captcha images from CCXP and directly render in terminal for labeling:
```bash
python -m decaptcha.collect
```
If you mislabel a captcha, relabel before labelling the next one:
```bash
python -m decaptcha.relabel
```

### 2. Build the dataset arrays
Build from `./data` and writes `data/images.npy`, `data/labels.npy`, and `data/groups.npy`:
```bash
python -m decaptcha.build
```

### 3. Train and evaluate
Train for 30 epoch (pass `--epochs` to customize):
```bash
python -m decaptcha.train
```
Training behavior:
- grouped `train/val/test` split by captcha `pwdstr`, capped to `20` renders per group
- group-balanced weighted sampling during training
- `ReduceLROnPlateau` scheduler on validation loss
- early stopping after `8` stale validation epochs
- best checkpoint selected by validation and test set exact-sequence accuracy
- resumes from `out/last.pt` if exists, otherwise starts fresh
- overwrites canonical artifacts by default, pass `--no-overwrite` to disable
- random split seed by default, pass `--seed` to set a fixed seed for reproducibility

Outputs:

- `out/best.pt`
- `out/last.pt`
- `out/int8.pt`
- `out/val.csv`
- `out/test.csv`
- `out/val_cm.npy`
- `out/test_cm.npy`
- `out/metrics.json`

## License

This project is released under the MIT License.

- Original forked work remains attributed to [25349023](https://github.com/25349023).
- Modifications and rewritten portions are additionally copyright (c) 2026 Hsi.
