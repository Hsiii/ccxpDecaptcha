from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class Component:
    pixels: List[Tuple[int, int]]
    area: int
    min_row: int
    max_row: int
    min_col: int
    max_col: int
    touches_border: bool

    @property
    def height(self) -> int:
        return self.max_row - self.min_row + 1

    @property
    def width(self) -> int:
        return self.max_col - self.min_col + 1

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(1, self.height)

    @property
    def fill_ratio(self) -> float:
        return self.area / max(1, self.width * self.height)


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return np.asarray(Image.fromarray(image).convert('L'), dtype=np.uint8)


def otsu_threshold(gray: np.ndarray) -> int:
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    cumulative_weight = np.cumsum(histogram)
    cumulative_sum = np.cumsum(histogram * np.arange(256))
    global_mean = cumulative_sum[-1] / max(1, total)

    between_class = np.zeros(256, dtype=np.float64)
    valid = (cumulative_weight > 0) & (cumulative_weight < total)
    mean_background = np.zeros(256, dtype=np.float64)
    mean_foreground = np.zeros(256, dtype=np.float64)
    mean_background[valid] = cumulative_sum[valid] / cumulative_weight[valid]
    mean_foreground[valid] = (
        (cumulative_sum[-1] - cumulative_sum[valid]) / (total - cumulative_weight[valid])
    )
    between_class[valid] = (
        cumulative_weight[valid]
        * (total - cumulative_weight[valid])
        * (mean_background[valid] - mean_foreground[valid]) ** 2
    )
    return int(np.argmax(between_class))


def label_components(mask: np.ndarray) -> List[Component]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Component] = []

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue

            queue = deque([(row, col)])
            visited[row, col] = True
            pixels: List[Tuple[int, int]] = []
            min_row = max_row = row
            min_col = max_col = col
            touches_border = row in (0, height - 1) or col in (0, width - 1)

            while queue:
                current_row, current_col = queue.popleft()
                pixels.append((current_row, current_col))
                min_row = min(min_row, current_row)
                max_row = max(max_row, current_row)
                min_col = min(min_col, current_col)
                max_col = max(max_col, current_col)

                for row_offset in (-1, 0, 1):
                    for col_offset in (-1, 0, 1):
                        if row_offset == 0 and col_offset == 0:
                            continue
                        next_row = current_row + row_offset
                        next_col = current_col + col_offset
                        if not (0 <= next_row < height and 0 <= next_col < width):
                            continue
                        if visited[next_row, next_col] or not mask[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        if next_row in (0, height - 1) or next_col in (0, width - 1):
                            touches_border = True
                        queue.append((next_row, next_col))

            components.append(
                Component(
                    pixels=pixels,
                    area=len(pixels),
                    min_row=min_row,
                    max_row=max_row,
                    min_col=min_col,
                    max_col=max_col,
                    touches_border=touches_border,
                )
            )

    return components


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    cleaned = mask.copy()
    for component in label_components(mask):
        if component.area >= min_area:
            continue
        for row, col in component.pixels:
            cleaned[row, col] = False
    return cleaned


def remove_border_strips(mask: np.ndarray, min_area: int) -> np.ndarray:
    cleaned = mask.copy()
    for component in label_components(mask):
        is_long_strip = (
            component.area >= min_area
            and component.touches_border
            and (
                component.width >= mask.shape[1] * 0.2
                or component.height >= mask.shape[0] * 0.2
                or component.aspect_ratio >= 4.0
                or component.aspect_ratio <= 0.25
            )
            and component.fill_ratio <= 0.55
        )
        if not is_long_strip:
            continue
        for row, col in component.pixels:
            cleaned[row, col] = False
    return cleaned


def morphological_close(mask: np.ndarray, size: int = 3) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
    dilated = image.filter(ImageFilter.MaxFilter(size=size))
    closed = dilated.filter(ImageFilter.MinFilter(size=size))
    return np.asarray(closed, dtype=np.uint8) > 0


def foreground_bbox(mask: np.ndarray, padding: int = 2) -> Tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return 0, mask.shape[0], 0, mask.shape[1]
    top = max(0, int(rows.min()) - padding)
    bottom = min(mask.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(cols.min()) - padding)
    right = min(mask.shape[1], int(cols.max()) + padding + 1)
    return top, bottom, left, right


def preprocess_oauth_captcha(image: np.ndarray) -> np.ndarray:
    gray = rgb_to_grayscale(image)
    original_height, original_width = gray.shape
    threshold = min(224, otsu_threshold(gray) + 8)
    foreground = gray <= threshold
    foreground = remove_small_components(foreground, min_area=10)
    foreground = remove_border_strips(foreground, min_area=24)
    foreground = morphological_close(foreground, size=3)

    top, bottom, left, right = foreground_bbox(foreground)
    cropped_gray = gray[top:bottom, left:right]
    cropped_mask = foreground[top:bottom, left:right]

    cleaned = np.full_like(cropped_gray, 255, dtype=np.uint8)
    cleaned[cropped_mask] = 96
    resized = Image.fromarray(cleaned, mode='L').resize((original_width, original_height), Image.Resampling.NEAREST)
    resized_array = np.asarray(resized, dtype=np.uint8)
    cleaned_rgb = np.stack([resized_array, resized_array, resized_array], axis=-1)
    return cleaned_rgb
