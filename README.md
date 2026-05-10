# ccxpDecaptcha

A training pipeline for NTHU CCXP decaptcha models.

Based on the work by [25349023](https://github.com/25349023), this fork enhances the local training workflow and the model architecture with a new cropped full-image six-head model. The model is retrained from scratch on a new dataset of 600 manually labeled captcha group (30000 images) collected in 2026 April, achieving significantly improved performance of 99.96% six-digit sequence accuracy and 99.99% individual digit accuracy.

The repo also adds a parallel pipeline for the NTHU OAuth login decaptcha with a 4-attention-pooling-heads model, achieving 98.79% four-digit sequence accuracy and 99.69% individual digit accuracy on a dataset of 23000 images collected with Securimage PHP in 2026 May.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
composer install # for Securimage PHP in the oauth pipeline
```

## Workflow
Fill in `<mode>` with either `ccxp` or `oauth` to run the respective pipeline.
### 1. Collect captcha data
- `<mode>` = `ccxp`: Fetch captcha images and render in terminal for manual labeling
- `<mode>` = `oauth`: Use Securimage PHP to generate captcha images + labels
```bash
python -m <mode>.collect
```
If you mislabel a captcha in `ccxp` mode:
```bash
python -m ccxp.relabel
```

### 2. Build the dataset arrays
```bash
python -m <mode>.build
```

### 3. Train and evaluate
Train the model and evaluate on the test set, with model checkpoints saved in `best.pt`:
```bash
python -m <mode>.train
```

## License

This project is released under the MIT License.

- Original forked work remains attributed to [25349023](https://github.com/25349023).
- Modifications and rewritten portions are additionally copyright (c) 2026 Hsi.
