import pathlib
import argparse
from typing import Tuple

import numpy as np
from PIL import Image


def parse_metadata(path: pathlib.Path) -> Tuple[str, str]:
    stem = path.stem
    if '__' in stem and '_' in stem:
        group, remainder = stem.split('__', maxsplit=1)
        label = remainder.rsplit('_', maxsplit=1)[0]
        if len(label) == 6 and label.isdigit():
            return label, group

    label = stem[:6]
    if len(label) != 6 or not label.isdigit():
        raise ValueError(f'Unable to parse a 6-digit label from {path.name}')
    return label, stem


def load_image(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def crop_image(image: np.ndarray, crop_right: int) -> np.ndarray:
    if crop_right <= 0:
        return image
    if crop_right >= image.shape[1]:
        raise ValueError(f'crop_right={crop_right} removes the entire image width {image.shape[1]}')
    return image[:, :-crop_right, :]


def generate_dataset_from(src: pathlib.Path, output_prefix: pathlib.Path, crop_right: int = 0):
    images = []
    labels = []
    groups = []

    for count, path in enumerate(sorted(src.glob('*.png')), start=1):
        label, group = parse_metadata(path)
        images.append(crop_image(load_image(path), crop_right=crop_right))
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

    image_path = output_prefix.with_name(f'{output_prefix.name}_images.npy')
    label_path = output_prefix.with_name(f'{output_prefix.name}_labels.npy')
    group_path = output_prefix.with_name(f'{output_prefix.name}_groups.npy')
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    np.save(image_path, image_array)
    np.save(label_path, label_array)
    np.save(group_path, group_array)
    print(f'Saved dataset to {image_path}, {label_path}, {group_path}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='./raw_data/')
    parser.add_argument('--output-prefix', default='captcha')
    parser.add_argument('--crop-right', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    src_dir = pathlib.Path(args.src_dir)

    if not src_dir.exists():
        raise OSError(f'Source directory {src_dir} not found.')

    generate_dataset_from(src_dir, pathlib.Path(args.output_prefix), crop_right=args.crop_right)
