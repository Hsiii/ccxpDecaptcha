import io
from pathlib import Path

import numpy as np
import requests
import torch
from bs4 import BeautifulSoup
from PIL import Image

from decaptcha.model import SixHeadCaptchaNet, decode_predictions

BASE_URL = 'https://www.ccxp.nthu.edu.tw/ccxp/INQUIRE/'
MODEL_PATH = Path(__file__).with_name('decaptcha.pt')


def get_img_src() -> str:
    res = requests.get(BASE_URL)
    soup = BeautifulSoup(res.text, 'lxml')
    img = soup.select_one('.input_box + img')
    return img['src']


def download_img(src: str) -> np.ndarray:
    res = requests.get(BASE_URL + src)
    with Image.open(io.BytesIO(res.content)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


def preprocess(raw_img: np.ndarray) -> torch.Tensor:
    tensor = torch.tensor(raw_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0)


def load_model(ckpt_path: Path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint

    model = SixHeadCaptchaNet()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def decaptcha_one(model_path: Path = MODEL_PATH):
    src = get_img_src()
    raw_img = download_img(src)

    model = load_model(model_path)
    logits = model(preprocess(raw_img))
    answer = decode_predictions(logits)[0]
    print(answer)
    return answer


if __name__ == '__main__':
    decaptcha_one()
