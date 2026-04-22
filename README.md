# ccxpDecaptcha

A training pipeline for NTHU CCXP decaptcha model.

Based on the work by [25349023](https://github.com/25349023), this fork enhances the local training workflow and the model architecture with a new cropped full-image six-head model. The model is retrained from scratch on a new dataset of 600 manually labeled captcha renders collected in 2026 April, achieving significantly improved performance of 99.96% six-digit sequence accuracy and 99.99% individual digit accuracy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Workflow

1. Collect captcha data:

```bash
python -m decaptcha.collect
```

- Download captcha images from CCXP and directly render in terminal for labeling

If you mislabel a captcha, relabel before labelling the next one:

```bash
python -m decaptcha.relabel
```

2. Build the dataset arrays:

```bash
python -m decaptcha.build
```

This writes:

- `data/images.npy`
- `data/labels.npy`
- `data/groups.npy`


3. Train and evaluate:

```bash
python -m decaptcha.train
```

- grouped `train/val/test` split by captcha `pwdstr`
- train split capped to `20` renders per captcha group
- group-balanced weighted sampling during training
- `ReduceLROnPlateau` scheduler on validation loss
- early stopping after `8` stale validation epochs
- best checkpoint selected by validation exact-sequence accuracy
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
- The upstream MIT notice is preserved in [LICENSE](LICENSE).
