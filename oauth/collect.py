import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .paths import resolve_repo_path
except ImportError:
    from paths import resolve_repo_path

# Mirrors the observed Securimage settings from the target OAuth deployment.
CHARSET = '0123456789'
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 80
CODE_LENGTH = 4
NUM_LINES = 5
PERTURBATION = 0.80
TEXT_GRAY = 112
BACKGROUND_GRAY = 255
NOISE_GRAY = 112

FONT_CANDIDATES = (
    'DejaVuSans-Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Supplemental/Arial.ttf',
)


def resolve_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def random_code(rng: random.Random) -> str:
    return ''.join(rng.choice(CHARSET) for _ in range(CODE_LENGTH))


def group_id(index: int) -> str:
    return f'synth-{index:08d}'


def render_digit_layer(
    digit: str,
    digit_index: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    rng: random.Random,
) -> Image.Image:
    canvas = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), digit, font=font)
    glyph_width = bbox[2] - bbox[0]
    glyph_height = bbox[3] - bbox[1]

    base_x = 10 + digit_index * 29 + rng.randint(-6, 6)
    base_y = 10 + rng.randint(-5, 5)
    draw.text(
        (base_x - bbox[0], base_y - bbox[1]),
        digit,
        font=font,
        fill=255,
        stroke_width=rng.randint(0, 1),
        stroke_fill=255,
    )

    crop_left = max(0, base_x - 10)
    crop_top = max(0, base_y - 10)
    crop_right = min(IMAGE_WIDTH, base_x + glyph_width + 14)
    crop_bottom = min(IMAGE_HEIGHT, base_y + glyph_height + 14)
    glyph = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

    rotate = rng.uniform(-26, 26)
    glyph = glyph.rotate(rotate, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=0)

    shear = rng.uniform(-0.28, 0.28)
    width, height = glyph.size
    offset = int(abs(shear) * height)
    sheared_width = width + offset
    glyph = glyph.transform(
        (sheared_width, height),
        Image.Transform.AFFINE,
        (1, shear, -offset if shear > 0 else 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC,
        fillcolor=0,
    )

    layer = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    paste_x = min(max(0, base_x + rng.randint(-4, 4)), max(0, IMAGE_WIDTH - glyph.size[0]))
    paste_y = min(max(0, base_y + rng.randint(-3, 3)), max(0, IMAGE_HEIGHT - glyph.size[1]))
    layer.paste(glyph, (paste_x, paste_y), glyph)
    return layer


def apply_wave_distortion(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = mask.shape
    yy, xx = np.indices((height, width), dtype=np.float32)

    amp_x = PERTURBATION * rng.uniform(1.5, 3.2)
    amp_y = PERTURBATION * rng.uniform(1.0, 2.6)
    freq_x = rng.uniform(0.055, 0.095)
    freq_y = rng.uniform(0.04, 0.085)
    phase_x = rng.uniform(0, math.tau)
    phase_y = rng.uniform(0, math.tau)

    src_x = xx + amp_x * np.sin(freq_x * yy + phase_x)
    src_y = yy + amp_y * np.sin(freq_y * xx + phase_y)

    src_x = np.clip(np.rint(src_x).astype(np.int32), 0, width - 1)
    src_y = np.clip(np.rint(src_y).astype(np.int32), 0, height - 1)
    return mask[src_y, src_x]


def draw_noise(image: Image.Image, rng: random.Random) -> None:
    draw = ImageDraw.Draw(image)

    for _ in range(NUM_LINES):
        points = []
        y = rng.randint(0, IMAGE_HEIGHT - 1)
        slope = rng.uniform(-0.35, 0.35)
        for x in range(-20, IMAGE_WIDTH + 21, 12):
            y += rng.randint(-8, 8)
            points.append((x, int(y + slope * x)))
        draw.line(points, fill=NOISE_GRAY, width=rng.randint(2, 4))

    for _ in range(rng.randint(60, 110)):
        x = rng.randint(0, IMAGE_WIDTH - 1)
        y = rng.randint(0, IMAGE_HEIGHT - 1)
        shape = rng.choice(('dot', 'plus', 'corner'))
        if shape == 'dot':
            draw.rectangle((x, y, x + 1, y + 1), fill=NOISE_GRAY)
        elif shape == 'plus':
            draw.line((x - 1, y, x + 1, y), fill=NOISE_GRAY, width=1)
            draw.line((x, y - 1, x, y + 1), fill=NOISE_GRAY, width=1)
        else:
            size = rng.randint(1, 2)
            draw.line((x, y, x + size, y), fill=NOISE_GRAY, width=1)
            draw.line((x, y, x, y + size), fill=NOISE_GRAY, width=1)


def render_captcha(code: str, rng: random.Random) -> np.ndarray:
    font = resolve_font(rng.randint(48, 58))
    layers = [render_digit_layer(digit, idx, font, rng) for idx, digit in enumerate(code)]
    mask = np.maximum.reduce([np.asarray(layer, dtype=np.uint8) for layer in layers])
    mask = apply_wave_distortion(mask, rng)

    image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND_GRAY)
    image_np = np.asarray(image, dtype=np.uint8).copy()
    image_np[mask > 32] = TEXT_GRAY
    image = Image.fromarray(image_np, mode='L')
    draw_noise(image, rng)
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    if rgb.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(
            f'Synthetic OAuth captcha has unexpected shape {rgb.shape}; '
            f'expected {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}'
        )
    return rgb


def collect_many(save_dir: Path, groups: int, renders_per_group: int, seed: int | None) -> None:
    rng = random.Random(seed if seed is not None else int(time.time() * 1000))
    save_dir.mkdir(parents=True, exist_ok=True)

    for group_index in range(groups):
        code = random_code(rng)
        group = group_id(group_index)
        for render_index in range(renders_per_group):
            image = render_captcha(code, rng)
            path = save_dir / f'{group}__{code}_{render_index}.png'
            Image.fromarray(image).save(path)

        if (group_index + 1) % 100 == 0:
            print(f'{group_index + 1} groups generated.')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate synthetic 150x80 OAuth captcha images using observed Securimage settings.'
    )
    parser.add_argument('--out', default='data/oauth', help='Directory to write generated PNG files.')
    parser.add_argument('--groups', type=int, default=1000, help='Number of unique captcha codes to generate.')
    parser.add_argument('--renders-per-group', type=int, default=10, help='Number of rendered variants per code.')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducible generation.')
    args = parser.parse_args()
    args.out = str(resolve_repo_path(args.out))
    return args


if __name__ == '__main__':
    args = parse_args()
    collect_many(Path(args.out), groups=args.groups, renders_per_group=args.renders_per_group, seed=args.seed)
