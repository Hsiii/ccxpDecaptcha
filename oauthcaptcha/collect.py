import io
from pathlib import Path
from typing import Tuple

import requests
from PIL import Image
from bs4 import BeautifulSoup

try:
    from .paths import resolve_repo_path
except ImportError:
    from paths import resolve_repo_path

AUTHORIZE_URL = (
    'https://oauth.ccxp.nthu.edu.tw/v1.1/authorize.php'
    '?response_type=code'
    '&client_id=eeclass'
    '&redirect_uri=https%3A%2F%2Feeclass.nthu.edu.tw%2Fservice%2Foauth%2F'
    '&scope=lmsid+userid'
    '&state='
    '&ui_locales=zh-TW'
)
CAPTCHA_BASE_URL = 'https://oauth.ccxp.nthu.edu.tw/v1.1/'


def build_oauth_session() -> requests.Session:
    return requests.Session()


def get_captcha(session: requests.Session) -> Tuple[str, str]:
    res = session.get(AUTHORIZE_URL, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'lxml')
    captcha_input = soup.select_one('input[name="captcha_id"]')
    captcha_image = soup.select_one('#captcha_image')
    if captcha_input is None or captcha_image is None:
        raise ValueError('Unable to find captcha metadata on the OAuth login page.')
    captcha_id = captcha_input.get('value', '').strip()
    captcha_src = captcha_image.get('src', '').strip()
    if not captcha_id or not captcha_src:
        raise ValueError('Captcha metadata is empty on the OAuth login page.')
    return captcha_id, captcha_src


def count_solved_captchas(save_dir: Path) -> int:
    solved = {
        path.stem.split('__', maxsplit=1)[0]
        for path in save_dir.glob('*.png')
        if '__' in path.stem
    }
    return len(solved)


def render_image_in_terminal(content: bytes):
    with Image.open(io.BytesIO(content)) as image:
        rgb_image = image.convert('RGB')
        width, height = rgb_image.size
        scale = max(1, 180 // width)
        resized = rgb_image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        pixels = resized.load()

        print('\x1b[2J\x1b[H', end='')
        print()
        for y in range(0, resized.height - 1, 2):
            row = []
            for x in range(resized.width):
                top = pixels[x, y]
                bottom = pixels[x, y + 1]
                row.append(
                    f'\x1b[38;2;{top[0]};{top[1]};{top[2]}m'
                    f'\x1b[48;2;{bottom[0]};{bottom[1]};{bottom[2]}m▀'
                )
            row.append('\x1b[0m')
            print(''.join(row))
        if resized.height % 2 == 1:
            row = []
            y = resized.height - 1
            for x in range(resized.width):
                top = pixels[x, y]
                row.append(f'\x1b[38;2;{top[0]};{top[1]};{top[2]}m▀')
            row.append('\x1b[0m')
            print(''.join(row))
        print()


def show_image(content: bytes):
    render_image_in_terminal(content)


def manually_label(captcha_src: str, captcha_id: str, session: requests.Session, solved_count: int) -> Tuple[str, str]:
    res = session.get(CAPTCHA_BASE_URL + captcha_src, timeout=20)
    res.raise_for_status()
    show_image(res.content)
    label = input(f'({solved_count} solved) solve the 4-digit captcha: ').strip()
    if len(label) != 4 or not label.isdigit():
        raise ValueError(f'Expected a 4-digit label, got {label!r}')
    return label, captcha_id


def collect_one(save_dir: Path, generate_count: int, session: requests.Session):
    captcha_id, captcha_src = get_captcha(session)
    solved_count = count_solved_captchas(save_dir)
    file_prefix, group = manually_label(captcha_src, captcha_id, session, solved_count)

    for i in range(generate_count):
        res = session.get(CAPTCHA_BASE_URL + captcha_src, timeout=20)
        res.raise_for_status()
        with open(save_dir / f'{group}__{file_prefix}_{i}.png', 'wb') as f:
            f.write(res.content)


def collect_many(save_dir: Path, n_round: int, cnt_per_round: int):
    sess = build_oauth_session()
    for _ in range(n_round):
        collect_one(save_dir, cnt_per_round, sess)


if __name__ == '__main__':
    dire = resolve_repo_path('data/oauthcaptcha')
    if not dire.exists():
        dire.mkdir(parents=True)

    collect_many(dire, 50, 50)
