import io
import re
from pathlib import Path
from typing import Tuple

import requests
import urllib3
from PIL import Image
from bs4 import BeautifulSoup

BASE_URL = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'


def build_ccxp_session() -> requests.Session:
    session = requests.Session()
    # The CCXP site currently presents a certificate chain that Python 3.14
    # rejects, so collection opts into site-scoped insecure TLS.
    session.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    return session


def get_img_src(session: requests.Session) -> str:
    res = session.get(BASE_URL, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, 'lxml')
    img = soup.select_one('.inputtext ~ img')
    return img['src']


def parse_pwdstr(src: str) -> str:
    matched = re.search(r'pwdstr=([0-9-]+)', src)
    if matched is None:
        raise ValueError(f'Unable to parse pwdstr from {src}')
    return matched.group(1)


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


def manually_label(src: str, session: requests.Session, solved_count: int) -> Tuple[str, str]:
    res = session.get(BASE_URL + src, timeout=20)
    res.raise_for_status()
    show_image(res.content)
    label = input(f'({solved_count} solved) solve the captcha: ').strip()
    return label, parse_pwdstr(src)


def collect_one(save_dir: Path, generate_count: int, session: requests.Session):
    src = get_img_src(session)
    solved_count = count_solved_captchas(save_dir)
    file_prefix, pwdstr = manually_label(src, session, solved_count)

    for i in range(generate_count):
        res = session.get(BASE_URL + src, timeout=20)
        res.raise_for_status()
        with open(save_dir / f'{pwdstr}__{file_prefix}_{i}.png', 'wb') as f:
            f.write(res.content)


def collect_many(save_dir: Path, n_round: int, cnt_per_round: int):
    sess = build_ccxp_session()
    for _ in range(n_round):
        collect_one(save_dir, cnt_per_round, sess)


if __name__ == '__main__':
    dire = Path('data')
    if not dire.exists():
        dire.mkdir(parents=True)

    collect_many(dire, 50, 50)
