# ccxpDecaptcha

Local-only training pipeline for the NTHU CCXP captcha model.

Based on the work by [25349023](https://github.com/25349023), this fork keeps only the local training workflow and replaces fixed-width per-digit slicing with a full-image six-head architecture. The model is then re-trained from scratch on a new dataset of 500+ manually labeled captcha renders collected in 2026, achieving a test set exact-sequence accuracy of **97.7%**, and test set digit-level accuracy of **99.6%**.

## License

This project is released under the MIT License.

- Original forked work remains attributed to [25349023](https://github.com/25349023).
- Modifications and rewritten portions are additionally copyright (c) 2026 Hsi.
- The upstream MIT notice is preserved in [LICENSE](LICENSE).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You can run the tools in either style:

```bash
python decaptcha/train.py
python -m decaptcha.train
```

The same pattern works for `collect`, `relabel`, and `build`.

## Workflow

1. Collect labeled captchas:

```bash
python decaptcha/collect.py
python -m decaptcha.collect
```

- Download captcha images from CCXP and label them manually.
- Captchas will be rendered inline in the terminal.
- Repeated renders from the same `pwdstr` stay grouped in the saved filename.

2. Build the dataset arrays:

```bash
python decaptcha/build.py
python -m decaptcha.build
```

This writes:

- `images.npy`
- `labels.npy`
- `groups.npy`

If you mislabel a captcha batch, relabel the whole grouped filename set before rebuilding:

```bash
python decaptcha/relabel.py
python -m decaptcha.relabel
```

3. Train and evaluate:

```bash
python decaptcha/train.py
python -m decaptcha.train
```

Training behavior:

- grouped `train/val/test` split by captcha `pwdstr`
- train split capped to `20` renders per captcha group
- group-balanced weighted sampling during training
- `ReduceLROnPlateau` scheduler on validation loss
- early stopping after `8` stale validation epochs
- best checkpoint selected by validation exact-sequence accuracy

Outputs:

- `best.pt`
- `last.pt`
- `int8.pt`
- `val.csv`
- `test.csv`
- `val_cm.npy`
- `test_cm.npy`
- `metrics.json`

`train.py` resumes from `/out/last.pt`, overwrites the canonical artifacts by default, and uses a random split seed unless `--seed` is provided. Pass `--no-overwrite` if you want the safety guard back.

Default paths:

- raw captcha PNGs: `/data`
- built dataset arrays: `/data`
- training outputs and checkpoints: `/out`
- resume checkpoint: `/out/last.pt`
