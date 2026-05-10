import argparse
import struct
import random
import subprocess
import time
from pathlib import Path

try:
    from .paths import resolve_repo_path
except ImportError:
    from paths import resolve_repo_path

CHARSET = '0123456789'
IMAGE_WIDTH = 150
IMAGE_HEIGHT = 80
CODE_LENGTH = 4
PHP_BIN = '/opt/homebrew/bin/php'
RENDER_SCRIPT = resolve_repo_path('oauth/render_securimage.php')


def random_code(rng: random.Random) -> str:
    return ''.join(rng.choice(CHARSET) for _ in range(CODE_LENGTH))


def group_id(index: int) -> str:
    return f'securimage-{index:08d}'


def assert_renderer_ready() -> None:
    if not Path(PHP_BIN).is_file():
        raise RuntimeError(f'PHP runtime not found at {PHP_BIN}.')
    if not RENDER_SCRIPT.is_file():
        raise RuntimeError(f'Securimage renderer not found at {RENDER_SCRIPT}.')


def png_size(image: bytes) -> tuple[int, int]:
    if len(image) < 24 or image[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Rendered captcha is not a valid PNG.')
    width, height = struct.unpack('>II', image[16:24])
    return width, height


def render_captcha(code: str) -> bytes:
    result = subprocess.run(
        [PHP_BIN, str(RENDER_SCRIPT), code],
        check=True,
        capture_output=True,
    )
    width, height = png_size(result.stdout)
    if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
        raise ValueError(
            f'Rendered captcha size was {width}x{height}, expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}.'
        )
    return result.stdout


def render_group(save_dir: Path, code: str, group: str, renders_per_group: int) -> None:
    result = subprocess.run(
        [PHP_BIN, str(RENDER_SCRIPT), code, str(renders_per_group), str(save_dir), f'{group}__{code}'],
        check=True,
        capture_output=True,
        text=True,
    )
    files = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if len(files) != renders_per_group:
        raise RuntimeError(f'Renderer wrote {len(files)} files, expected {renders_per_group}.')
    for path in files:
        width, height = png_size(path.read_bytes())
        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            raise ValueError(
                f'Rendered captcha size was {width}x{height}, expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}.'
            )


def collect_many(save_dir: Path, groups: int, renders_per_group: int, seed: int | None) -> None:
    assert_renderer_ready()
    rng = random.Random(seed if seed is not None else int(time.time() * 1000))
    save_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for group_index in range(groups):
        code = random_code(rng)
        group = group_id(group_index)
        render_group(save_dir, code, group, renders_per_group)

        completed = group_index + 1
        if completed <= 5 or completed % 25 == 0 or completed == groups:
            elapsed = time.time() - started
            rate = completed / elapsed if elapsed else 0.0
            print(f'{completed}/{groups} groups generated ({rate:.2f} groups/s).')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate local 150x80 OAuth captcha images using the real Securimage PHP renderer.'
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
