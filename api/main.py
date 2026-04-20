import io
import json
import math
import pickle
import struct
import zipfile
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, make_response, request
from PIL import Image

app = Flask(__name__)
EPS = 1e-5
DIGITS = 6
MAX_VALID_WIDTH = 100
ALLOWED_CORS_ORIGINS = {"https://www.ccxp.nthu.edu.tw"}


class FloatStorage:  # marker class for pickle loading
    pass


class LongStorage:  # marker class for pickle loading
    pass


class TensorData:
    def __init__(self, shape, data):
        self.shape = tuple(int(v) for v in shape)
        self.data = data


class PTUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name == "FloatStorage":
            return FloatStorage
        if module == "torch" and name == "LongStorage":
            return LongStorage
        raise ValueError(f"Unsupported pickle class: {module}.{name}")


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    if isinstance(size, int):
        shape = (size,)
    else:
        shape = tuple(size)

    if len(shape) == 0:
        return TensorData((), [storage[int(storage_offset)]])

    flat = []
    total = 1
    for dim in shape:
        total *= dim

    for linear_idx in range(total):
        rem = linear_idx
        storage_idx = int(storage_offset)
        for axis in range(len(shape) - 1, -1, -1):
            dim = shape[axis]
            idx = rem % dim
            rem //= dim
            storage_idx += idx * int(stride[axis])
        flat.append(storage[storage_idx])

    return TensorData(shape, flat)


@lru_cache(maxsize=1)
def load_weights():
    checkpoint_path = Path(__file__).resolve().parent / "decaptcha.pt"

    with zipfile.ZipFile(checkpoint_path, "r") as zf:
        storage_cache = {}

        def persistent_load(pid):
            if not isinstance(pid, tuple) or pid[0] != "storage":
                raise ValueError(f"Unexpected persistent id: {pid}")

            _, storage_type, key, location, size = pid
            size = int(size)
            cache_key = str(key)
            if cache_key in storage_cache:
                return storage_cache[cache_key]

            raw = zf.read(f"archive/data/{cache_key}")
            if storage_type is FloatStorage:
                values = list(struct.unpack("<" + ("f" * size), raw[: 4 * size]))
            elif storage_type is LongStorage:
                values = list(struct.unpack("<" + ("q" * size), raw[: 8 * size]))
            else:
                raise ValueError(f"Unsupported storage type: {storage_type}")

            storage_cache[cache_key] = values
            return values

        unpickler = PTUnpickler(io.BytesIO(zf.read("archive/data.pkl")))
        unpickler.persistent_load = persistent_load
        state = unpickler.load()

    return state


def _tensor_get(tensor, indices):
    shape = tensor.shape
    flat_idx = 0
    stride = 1
    for axis in range(len(shape) - 1, -1, -1):
        flat_idx += indices[axis] * stride
        stride *= shape[axis]
    return tensor.data[flat_idx]


def _conv2d(input_tensor, weight, bias, stride=2, padding=1):
    in_channels, in_h, in_w = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape

    out_h = (in_h + (2 * padding) - kernel_h) // stride + 1
    out_w = (in_w + (2 * padding) - kernel_w) // stride + 1
    out = [0.0] * (out_channels * out_h * out_w)

    out_idx = 0
    for oc in range(out_channels):
        for oy in range(out_h):
            for ox in range(out_w):
                acc = bias.data[oc]
                in_y0 = oy * stride - padding
                in_x0 = ox * stride - padding

                for ic in range(in_channels):
                    for ky in range(kernel_h):
                        in_y = in_y0 + ky
                        if in_y < 0 or in_y >= in_h:
                            continue
                        for kx in range(kernel_w):
                            in_x = in_x0 + kx
                            if in_x < 0 or in_x >= in_w:
                                continue

                            x = _tensor_get(input_tensor, (ic, in_y, in_x))
                            w = _tensor_get(weight, (oc, ic, ky, kx))
                            acc += x * w

                out[out_idx] = acc
                out_idx += 1

    return TensorData((out_channels, out_h, out_w), out)


def _batchnorm2d(input_tensor, gamma, beta, running_mean, running_var, eps=EPS):
    channels, h, w = input_tensor.shape
    output = [0.0] * len(input_tensor.data)

    idx = 0
    for c in range(channels):
        g = gamma.data[c]
        b = beta.data[c]
        mean = running_mean.data[c]
        var = running_var.data[c]
        inv_std = 1.0 / math.sqrt(var + eps)

        for _ in range(h * w):
            x = input_tensor.data[idx]
            output[idx] = ((x - mean) * inv_std) * g + b
            idx += 1

    return TensorData(input_tensor.shape, output)


def _relu(input_tensor):
    return TensorData(input_tensor.shape, [x if x > 0.0 else 0.0 for x in input_tensor.data])


def _adaptive_avg_pool2d_2x2(input_tensor):
    channels, in_h, in_w = input_tensor.shape
    out_h, out_w = 2, 2
    out = []

    for c in range(channels):
        for oy in range(out_h):
            y0 = math.floor((oy * in_h) / out_h)
            y1 = math.ceil(((oy + 1) * in_h) / out_h)
            for ox in range(out_w):
                x0 = math.floor((ox * in_w) / out_w)
                x1 = math.ceil(((ox + 1) * in_w) / out_w)

                total = 0.0
                count = 0
                for iy in range(y0, y1):
                    for ix in range(x0, x1):
                        total += _tensor_get(input_tensor, (c, iy, ix))
                        count += 1

                out.append(total / max(count, 1))

    return TensorData((channels, out_h, out_w), out)


def _linear(input_vector, weight, bias):
    out_features, in_features = weight.shape
    out = [0.0] * out_features

    for o in range(out_features):
        base = o * in_features
        acc = bias.data[o]
        for i in range(in_features):
            acc += weight.data[base + i] * input_vector[i]
        out[o] = acc

    return out


def _argmax(values):
    best_idx = 0
    best_value = values[0]
    for i in range(1, len(values)):
        if values[i] > best_value:
            best_value = values[i]
            best_idx = i
    return best_idx


def _decode_image(img_byte_values):
    image = Image.open(io.BytesIO(bytes(img_byte_values))).convert("RGB")
    width, height = image.size

    usable_width = min(width, MAX_VALID_WIDTH)
    if usable_width < DIGITS:
        raise ValueError("Captcha image is too narrow.")

    cropped = image.crop((0, 0, usable_width, height))
    width_per_digit = usable_width // DIGITS
    split_points = list(range(0, usable_width, width_per_digit))

    patches = []
    for i in range(DIGITS):
        x0 = split_points[i]
        x1 = split_points[i + 1]
        patch = cropped.crop((x0, 0, x1, height))

        pw, ph = patch.size
        pixels = list(patch.getdata())
        chw = []

        for channel in range(3):
            for y in range(ph):
                row_offset = y * pw
                for x in range(pw):
                    chw.append(pixels[row_offset + x][channel] / 255.0)

        patches.append(TensorData((3, ph, pw), chw))

    return patches


def predict_digits(img_byte_values):
    state = load_weights()

    answer = []
    for patch in _decode_image(img_byte_values):
        x = _conv2d(patch, state["0.weight"], state["0.bias"], stride=2, padding=1)
        x = _batchnorm2d(x, state["1.weight"], state["1.bias"], state["1.running_mean"], state["1.running_var"])
        x = _relu(x)

        x = _conv2d(x, state["3.weight"], state["3.bias"], stride=2, padding=1)
        x = _batchnorm2d(x, state["4.weight"], state["4.bias"], state["4.running_mean"], state["4.running_var"])
        x = _relu(x)

        x = _adaptive_avg_pool2d_2x2(x)
        logits = _linear(x.data, state["8.weight"], state["8.bias"])
        answer.append(str(_argmax(logits)))

    return "".join(answer)


def _cors_response(payload, status=200):
    response = make_response(payload, status)
    origin = request.headers.get("Origin")
    if origin in ALLOWED_CORS_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return response


@app.route("/api/decaptcha", methods=["POST", "OPTIONS"])
def decaptcha():
    origin = request.headers.get("Origin")
    if origin and origin not in ALLOWED_CORS_ORIGINS:
        return _cors_response(jsonify({"error": "Origin is not allowed."}), status=403)

    if request.method == "OPTIONS":
        return _cors_response("")

    payload = request.get_json(silent=True) or {}
    img = payload.get("img")

    if not isinstance(img, list) or not img:
        return _cors_response(jsonify({"error": "`img` must be a non-empty byte array."}), status=400)

    try:
        answer = predict_digits(img)
    except Exception as exc:
        return _cors_response(jsonify({"error": str(exc)}), status=500)

    return _cors_response(jsonify({"answer": answer}))


if __name__ == "__main__":
    app.run(debug=True)
