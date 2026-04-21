import pathlib
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


def generate_dataset_from(src: pathlib.Path):
    images = []
    labels = []
    groups = []

    for count, path in enumerate(sorted(src.glob('*.png')), start=1):
        label, group = parse_metadata(path)
        images.append(load_image(path))
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

    np.save('captcha_images.npy', image_array)
    np.save('captcha_labels.npy', label_array)
    np.save('captcha_groups.npy', group_array)


if __name__ == '__main__':
    src_dir = pathlib.Path('./raw_data/')

    if not src_dir.exists():
        raise OSError(f'Source directory {src_dir} not found.')

    generate_dataset_from(src_dir)
