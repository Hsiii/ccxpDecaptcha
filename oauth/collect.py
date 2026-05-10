import argparse
import json
import os
import random
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
WORKER_SCRIPT = resolve_repo_path('oauth/render_securimage_worker.php')


def random_code(rng: random.Random) -> str:
    return ''.join(rng.choice(CHARSET) for _ in range(CODE_LENGTH))


def group_id(index: int) -> str:
    return f'securimage-{index:08d}'


def png_size(image: bytes) -> tuple[int, int]:
    if len(image) < 24 or image[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Rendered captcha is not a valid PNG.')
    width, height = struct.unpack('>II', image[16:24])
    return width, height


def assert_renderer_ready() -> None:
    if not Path(PHP_BIN).is_file():
        raise RuntimeError(f'PHP runtime not found at {PHP_BIN}.')
    if not WORKER_SCRIPT.is_file():
        raise RuntimeError(f'Securimage worker not found at {WORKER_SCRIPT}.')


class SecurimageWorker:
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.process = subprocess.Popen(
            [PHP_BIN, str(WORKER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def render_group(self, code: str, group: str, renders_per_group: int) -> None:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError('Securimage worker pipes are unavailable.')

        request = {
            'code': code,
            'count': renders_per_group,
            'out_dir': str(self.save_dir),
            'prefix': f'{group}__{code}',
        }
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

        line = self.process.stdout.readline()
        if not line:
            stderr = ''
            if self.process.stderr is not None:
                stderr = self.process.stderr.read().strip()
            raise RuntimeError(f'Securimage worker exited unexpectedly. {stderr}'.strip())

        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(f"Securimage worker error: {response['error']}")

        files = [Path(path_str) for path_str in response['files']]
        if len(files) != renders_per_group:
            raise RuntimeError(f'Renderer wrote {len(files)} files, expected {renders_per_group}.')

        for path in files:
            width, height = png_size(path.read_bytes())
            if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
                raise ValueError(
                    f'Rendered captcha size was {width}x{height}, expected {IMAGE_WIDTH}x{IMAGE_HEIGHT}.'
                )

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        if self.process.stdin is not None:
            self.process.stdin.write(json.dumps({'shutdown': True}) + '\n')
            self.process.stdin.flush()
            self.process.stdin.close()
        self.process.wait(timeout=5)


def progress_message(completed: int, groups: int, started: float) -> str | None:
    if completed > 5 and completed % 25 != 0 and completed != groups:
        return None
    elapsed = time.time() - started
    rate = completed / elapsed if elapsed else 0.0
    return f'{completed}/{groups} groups generated ({rate:.2f} groups/s).'


def collect_many(save_dir: Path, groups: int, renders_per_group: int, seed: int | None, workers: int) -> None:
    assert_renderer_ready()
    rng = random.Random(seed if seed is not None else int(time.time() * 1000))
    save_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(group_id(index), random_code(rng)) for index in range(groups)]
    worker_count = max(1, min(workers, groups))
    started = time.time()
    completed = 0
    progress_lock = threading.Lock()

    def run_shard(worker_index: int) -> None:
        nonlocal completed
        worker = SecurimageWorker(save_dir)
        try:
            for group, code in jobs[worker_index::worker_count]:
                worker.render_group(code, group, renders_per_group)
                with progress_lock:
                    completed += 1
                    message = progress_message(completed, groups, started)
                if message is not None:
                    print(message)
        finally:
            worker.close()

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_shard, worker_index) for worker_index in range(worker_count)]
        for future in futures:
            future.result()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate local 150x80 OAuth captcha images using persistent Securimage PHP workers.'
    )
    parser.add_argument('--out', default='data/oauth', help='Directory to write generated PNG files.')
    parser.add_argument('--groups', type=int, default=1000, help='Number of unique captcha codes to generate.')
    parser.add_argument('--renders-per-group', type=int, default=10, help='Number of rendered variants per code.')
    parser.add_argument('--seed', type=int, default=None, help='Optional random seed for reproducible generation.')
    parser.add_argument(
        '--workers',
        type=int,
        default=min(8, os.cpu_count() or 1),
        help='Number of persistent PHP workers to run in parallel.',
    )
    args = parser.parse_args()
    args.out = str(resolve_repo_path(args.out))
    return args


if __name__ == '__main__':
    args = parse_args()
    collect_many(
        Path(args.out),
        groups=args.groups,
        renders_per_group=args.renders_per_group,
        seed=args.seed,
        workers=args.workers,
    )
