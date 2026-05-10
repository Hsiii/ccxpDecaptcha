# ccxpDecaptcha

A training pipeline for NTHU CCXP captcha models.

Based on the work by [25349023](https://github.com/25349023), this fork enhances the local training workflow and the model architecture with a new cropped full-image six-head model. The model is retrained from scratch on a new dataset of 600 manually labeled captcha group (30000 images) collected in 2026 April, achieving significantly improved performance of 99.96% six-digit sequence accuracy and 99.99% individual digit accuracy.

The repo also includes a parallel `oauth` pipeline for the newer OAuth login captcha observed on `oauth.ccxp.nthu.edu.tw`, which currently uses 4 digits and is generated locally with the real Securimage PHP library using the observed site settings.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
composer install
```

## Workflow

### 1. Collect captcha data
Download captcha images from CCXP and directly render in terminal for labeling:
```bash
python -m ccxp.collect
```
If you mislabel a captcha, relabel before labelling the next one:
```bash
python -m ccxp.relabel
```

### 2. Build the dataset arrays
Build from `./data` and writes `data/images.npy`, `data/labels.npy`, and `data/groups.npy`:
```bash
python -m ccxp.build
```

### 3. Train and evaluate
Train for 30 epoch (pass `--epochs` to customize):
```bash
python -m ccxp.train
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

## OAuth 4-digit pipeline

The original `ccxp` package remains the 6-digit CCXP pipeline. The `oauth` package is an alternative collector/build/train flow for the OAuth login page captcha.

### 1. Generate OAuth captcha data
Generate OAuth captcha groups with a local Securimage renderer. Output images match the site size exactly at `150x80`:
```bash
python -m oauth.collect
```
Defaults:
- `1000` groups
- `10` rendered variants per group
- up to `8` persistent PHP workers in parallel
- output to `./data/oauth`

Requirements:
- `php` with the `gd` extension available at `/opt/homebrew/bin/php`
- Composer dependencies installed via `composer install`

Example with explicit sizing:
```bash
python -m oauth.collect --groups 1500 --renders-per-group 12 --workers 4 --seed 42
```

### 2. Build the OAuth dataset arrays
Build raw and cleaned variants from `./data/oauth`:
```bash
python -m oauth.build
```
This writes:
- raw arrays to `./data/oauth`
- cleaned arrays to `./data/oauth_clean`

If you only want one variant:
```bash
python -m oauth.build --variant raw
python -m oauth.build --variant clean
```

### 3. Train the OAuth model
By default, train both raw and cleaned variants:
```bash
python -m oauth.train
```
This writes:
- raw outputs to `./out/oauth`
- cleaned outputs to `./out/oauth_clean`

If you only want one variant:
```bash
python -m oauth.train --variant raw
python -m oauth.train --variant clean
```

## License

This project is released under the MIT License.

- Original forked work remains attributed to [25349023](https://github.com/25349023).
- Modifications and rewritten portions are additionally copyright (c) 2026 Hsi.
