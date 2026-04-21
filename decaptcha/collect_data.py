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


def show_image(content: bytes):
    with Image.open(io.BytesIO(content)) as image:
        image.convert('RGB').show()


def manually_label(src: str, session: requests.Session) -> Tuple[str, str]:
    res = session.get(BASE_URL + src, timeout=20)
    res.raise_for_status()
    show_image(res.content)
    label = input('input the numbers you just saw: ').strip()
    return label, parse_pwdstr(src)


def collect_one(save_dir: Path, generate_count: int, session: requests.Session):
    src = get_img_src(session)
    file_prefix, pwdstr = manually_label(src, session)

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
    dire = Path('./raw_data/')
    if not dire.exists():
        dire.mkdir(parents=True)

    collect_many(dire, 50, 50)
