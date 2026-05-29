from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode image")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def calculate_metrics(image: np.ndarray) -> dict[str, float]:
    gray = to_gray(image)
    gray_f = gray.astype(np.float32)
    b, g, r = cv2.split(image.astype(np.float32))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)

    brightness = float(np.mean(gray_f))
    contrast = float(np.std(gray_f))
    p1, p99 = np.percentile(gray_f, [1, 99])
    overexposed = float(np.mean(gray >= 250))
    underexposed = float(np.mean(gray <= 15))

    noise_estimate = estimate_noise(gray_f)
    color_cast = float(max(abs(np.mean(r) - np.mean(g)), abs(np.mean(b) - np.mean(g)), abs(np.mean(r) - np.mean(b))))
    dark_block_ratio = estimate_dark_block_ratio(gray_f)
    stripe_score = estimate_stripe_score(gray_f)

    return {
        "sharpness_laplacian_var": float(np.var(laplacian)),
        "tenengrad": float(np.mean(gradient)),
        "brightness_mean": brightness,
        "contrast_std": contrast,
        "dynamic_range": float(p99 - p1),
        "noise_estimate": noise_estimate,
        "color_cast_score": color_cast,
        "overexposed_ratio": overexposed,
        "underexposed_ratio": underexposed,
        "occlusion_dark_block_ratio": dark_block_ratio,
        "stripe_score": stripe_score,
    }


def estimate_noise(gray_f: np.ndarray) -> float:
    blurred = cv2.GaussianBlur(gray_f, (5, 5), 0)
    residual = gray_f - blurred
    local_std = cv2.GaussianBlur(residual * residual, (7, 7), 0) ** 0.5
    gradients_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    low_texture_mask = (np.abs(gradients_x) + np.abs(gradients_y)) < 18
    if np.any(low_texture_mask):
        return float(np.median(local_std[low_texture_mask]))
    return float(np.median(local_std))


def estimate_dark_block_ratio(gray_f: np.ndarray) -> float:
    h, w = gray_f.shape
    block = max(16, min(h, w) // 12)
    dark_pixels = 0
    total_pixels = h * w
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = gray_f[y : y + block, x : x + block]
            if tile.size and float(np.mean(tile)) < 28 and float(np.std(tile)) < 18:
                dark_pixels += tile.size
    return float(dark_pixels / total_pixels)


def estimate_stripe_score(gray_f: np.ndarray) -> float:
    row_means = np.mean(gray_f, axis=1)
    col_means = np.mean(gray_f, axis=0)
    row_diff = np.mean(np.abs(np.diff(row_means))) / (np.std(gray_f) + 1.0)
    col_diff = np.mean(np.abs(np.diff(col_means))) / (np.std(gray_f) + 1.0)
    return float(max(row_diff, col_diff))


def make_overlay(image: np.ndarray, metrics: dict[str, float]) -> np.ndarray:
    overlay = image.copy()
    gray = to_gray(image)
    h, w = gray.shape

    dark_mask = gray < 18
    bright_mask = gray > 248
    overlay[dark_mask] = (70, 70, 220)
    overlay[bright_mask] = (40, 190, 255)

    if metrics.get("occlusion_dark_block_ratio", 0.0) > 0.35:
        block = max(16, min(h, w) // 12)
        for y in range(0, h, block):
            for x in range(0, w, block):
                tile = gray[y : y + block, x : x + block]
                if tile.size and float(np.mean(tile)) < 28 and float(np.std(tile)) < 18:
                    cv2.rectangle(overlay, (x, y), (min(x + block, w - 1), min(y + block, h - 1)), (0, 0, 255), 2)

    return cv2.addWeighted(image, 0.62, overlay, 0.38, 0)


def save_image(path: str | Path, image: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = output.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Cannot encode image as {ext}")
    encoded.tofile(str(output))
