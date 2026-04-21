import os
import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
import tqdm
from torch import nn
from torch.utils import data

from decaptcha.six_head_model import DIGITS, SixHeadCaptchaNet


SEED = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class DatasetSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray


class CaptchaDataset(data.Dataset):
    def __init__(self, path_to_data_root='.', transform=None, train=True, val_ratio: float = 0.1, seed: int = SEED):
        images = np.load(os.path.join(path_to_data_root, 'captcha_images.npy'))
        labels = np.load(os.path.join(path_to_data_root, 'captcha_labels.npy'))
        groups = np.load(os.path.join(path_to_data_root, 'captcha_groups.npy'))

        split = self._build_split(groups, val_ratio=val_ratio, seed=seed)
        indices = split.train_indices if train else split.val_indices

        self.images = images[indices]
        self.labels = labels[indices]
        self.transform = transform

    @staticmethod
    def _build_split(groups: np.ndarray, val_ratio: float, seed: int) -> DatasetSplit:
        unique_groups = np.array(sorted(set(groups.tolist())))
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_groups)

        val_size = max(1, int(round(len(unique_groups) * val_ratio)))
        if val_size >= len(unique_groups):
            val_size = max(1, len(unique_groups) - 1)

        val_groups = set(unique_groups[:val_size].tolist())
        train_indices, val_indices = [], []
        for idx, group in enumerate(groups.tolist()):
            if group in val_groups:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        return DatasetSplit(
            train_indices=np.array(train_indices, dtype=np.int64),
            val_indices=np.array(val_indices, dtype=np.int64),
        )

    def __getitem__(self, item):
        image = self.images[item]
        label = torch.tensor(self.labels[item], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.images.shape[0]


def build_transforms() -> Tuple[T.Compose, T.Compose]:
    train_transform = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees=12, translate=(0.05, 0.08), shear=(-8, 8, -4, 4), fill=(255, 255, 255)),
        T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.2, hue=0.03),
        T.ToTensor(),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.2),
        T.RandomErasing(p=0.2, scale=(0.01, 0.05), ratio=(0.3, 3.0), value=1.0),
    ])
    eval_transform = T.Compose([T.ToTensor()])
    return train_transform, eval_transform


def multi_head_loss(logits: torch.Tensor, labels: torch.Tensor, loss_fn: nn.Module) -> torch.Tensor:
    return sum(loss_fn(logits[:, idx], labels[:, idx]) for idx in range(DIGITS)) / DIGITS


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    predictions = logits.argmax(dim=-1)
    digit_accuracy = (predictions == labels).float().mean().item()
    sequence_accuracy = (predictions == labels).all(dim=1).float().mean().item()
    return {
        'digit_accuracy': digit_accuracy,
        'sequence_accuracy': sequence_accuracy,
    }


def train_one_epoch(model, dataset, loss_fn, optim, device: torch.device):
    losses = []
    metric_history = []
    for x, y in tqdm.tqdm(dataset, desc='Training... ', leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = multi_head_loss(logits, y, loss_fn)
        losses.append(loss.item())
        metric_history.append(compute_metrics(logits, y))

        optim.zero_grad()
        loss.backward()
        optim.step()

    return float(np.mean(losses)), average_metrics(metric_history)


def test_one_epoch(model, dataset, loss_fn, device: torch.device):
    losses = []
    metric_history = []
    with torch.no_grad():
        for x, y in tqdm.tqdm(dataset, desc='Testing... ', leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = multi_head_loss(logits, y, loss_fn)
            losses.append(loss.item())
            metric_history.append(compute_metrics(logits, y))

    return float(np.mean(losses)), average_metrics(metric_history)


def average_metrics(metric_history):
    return {
        name: float(np.mean([metrics[name] for metrics in metric_history]))
        for name in metric_history[0]
    }


def fit(model, train_ld, test_ld, loss_fn, optim, device: torch.device, scheduler=None, epochs=20):
    best_state = None
    best_sequence_accuracy = -1.0

    for epoch in range(epochs):
        model.train()
        train_loss, train_metrics = train_one_epoch(model, train_ld, loss_fn, optim, device)
        model.eval()
        test_loss, test_metrics = test_one_epoch(model, test_ld, loss_fn, device)

        print(
            f'Epoch {epoch:>2}: '
            f'train_loss = {train_loss:.6f}, '
            f'val_loss = {test_loss:.6f}, '
            f'train_seq_acc = {train_metrics["sequence_accuracy"]:.6f}, '
            f'val_seq_acc = {test_metrics["sequence_accuracy"]:.6f}, '
            f'val_digit_acc = {test_metrics["digit_accuracy"]:.6f}'
        )

        if test_metrics['sequence_accuracy'] > best_sequence_accuracy:
            best_sequence_accuracy = test_metrics['sequence_accuracy']
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

        if scheduler is not None:
            scheduler.step()

    return best_state, best_sequence_accuracy


def export_quantized_model(model: nn.Module, output_path: str):
    quantized_model = torch.quantization.quantize_dynamic(model.cpu(), {nn.Linear}, dtype=torch.qint8)
    scripted_model = torch.jit.script(quantized_model)
    scripted_model.save(output_path)


if __name__ == '__main__':
    seed_everything()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform, eval_transform = build_transforms()
    train_dataset = CaptchaDataset(transform=train_transform, train=True)
    val_dataset = CaptchaDataset(transform=eval_transform, train=False)

    train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    model = SixHeadCaptchaNet().to(device)

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    epochs = 60

    best_state, best_sequence_accuracy = fit(
        model,
        train_loader,
        val_loader,
        loss,
        optimizer,
        device=device,
        scheduler=scheduler,
        epochs=epochs,
    )
    if best_state is None:
        raise RuntimeError('Training did not produce a checkpoint.')

    model.load_state_dict(best_state)
    checkpoint = {
        'state_dict': best_state,
        'sequence_accuracy': best_sequence_accuracy,
        'digits': DIGITS,
    }
    torch.save(checkpoint, 'decaptcha.pt')
    export_quantized_model(model, 'decaptcha.int8.pt')
