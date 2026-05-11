import argparse
import pathlib
from typing import Tuple

import numpy as np
from PIL import Image

try:
    from .paths import PIPELINE, resolve_repo_path
except ImportError:
    from paths import PIPELINE, resolve_repo_path

def parse_metadata(path: pathlib.Path) -> Tuple[str, str]:
    stem = path.stem
    if '__' in stem and '_' in stem:
        group, remainder = stem.split('__', maxsplit=1)
        label = remainder.rsplit('_', maxsplit=1)[0]
        if len(label) == 4 and label.isdigit():
            return label, group

    label = stem[:4]
    if len(label) != 4 or not label.isdigit():
        raise ValueError(f'Unable to parse a 4-digit label from {path.name}')
    return label, stem


def load_image(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def build_arrays(src: pathlib.Path, out_dir: pathlib.Path):
    images = []
    labels = []
    groups = []

    for count, path in enumerate(sorted(src.glob('*.png')), start=1):
        label, group = parse_metadata(path)
        image = load_image(path)
        images.append(image)
        labels.append([int(digit) for digit in label])
        groups.append(group)

        if count % 500 == 0:
            print(count, 'processed.')

    if not images:
        raise RuntimeError(f'No PNG files found in {src}')

    image_array = np.stack(images, axis=0)
    label_array = np.array(labels, dtype=np.int64)
    group_array = np.array(groups)

    print(image_array.shape, label_array.shape, group_array.shape)

    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / 'images.npy'
    label_path = out_dir / 'labels.npy'
    group_path = out_dir / 'groups.npy'

    np.save(image_path, image_array)
    np.save(label_path, label_array)
    np.save(group_path, group_array)
    print(f'Saved dataset to {image_path}, {label_path}, {group_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=str(PIPELINE.default_raw_dir()))
    parser.add_argument('--out', default=str(PIPELINE.default_processed_dir()))
    args = parser.parse_args()
    args.src = str(resolve_repo_path(args.src))
    args.out = str(resolve_repo_path(args.out))
    return args


if __name__ == '__main__':
    args = parse_args()
    src_dir = pathlib.Path(args.src)

    if not src_dir.exists():
        raise OSError(f'Source directory {src_dir} not found.')

    build_arrays(src_dir, pathlib.Path(args.out))
