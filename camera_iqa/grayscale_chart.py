from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_iqa.metrics import read_image, save_image, to_gray


@dataclass(slots=True)
class GrayBlock:
    index: int
    x: int
    y: int
    width: int
    height: int
    mean_gray: float
    std_gray: float


@dataclass(slots=True)
class GrayStrip:
    x: int
    y: int
    width: int
    height: int
    blocks: list[GrayBlock]
    adjacent_deltas: list[float]
    distinguishable_levels: int


@dataclass(slots=True)
class GrayscaleChartResult:
    distinguishable_levels: int
    expected_levels: int
    min_delta: float
    strips: list[GrayStrip]
    best_strip: GrayStrip
    annotation_path: Path | None = None


def analyze_grayscale_chart_path(
    path: str | Path,
    *,
    expected_levels: int = 11,
    min_delta: float = 5.0,
    annotation_path: str | Path | None = None,
) -> GrayscaleChartResult:
    return analyze_grayscale_chart(
        read_image(path),
        expected_levels=expected_levels,
        min_delta=min_delta,
        annotation_path=annotation_path,
    )


def analyze_grayscale_chart(
    image: np.ndarray,
    *,
    expected_levels: int = 11,
    min_delta: float = 5.0,
    annotation_path: str | Path | None = None,
) -> GrayscaleChartResult:
    if expected_levels < 2:
        raise ValueError("expected_levels must be at least 2")

    gray = to_gray(image) if image.ndim == 3 else image
    strip_boxes = locate_gray_strips(gray)
    if not strip_boxes:
        raise ValueError("Cannot locate grayscale chart strips")

    strips = [
        measure_strip(gray, box, expected_levels=expected_levels, min_delta=min_delta)
        for box in strip_boxes
    ]
    best_strip = max(strips, key=lambda strip: (strip.distinguishable_levels, strip.width * strip.height))
    output_path = Path(annotation_path) if annotation_path is not None else None
    if output_path is not None:
        save_image(output_path, make_annotation(image, strips, best_strip, min_delta))

    return GrayscaleChartResult(
        distinguishable_levels=best_strip.distinguishable_levels,
        expected_levels=expected_levels,
        min_delta=min_delta,
        strips=strips,
        best_strip=best_strip,
        annotation_path=output_path,
    )


def locate_gray_strips(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = gray.shape
    y_margin = max(4, h // 16)
    x_margin = max(4, w // 24)
    central = gray[y_margin : h - y_margin, x_margin : w - x_margin]
    median = float(np.median(central))
    diff = np.abs(gray.astype(np.float32) - median)
    mask = (diff > 10).astype(np.uint8) * 255
    mask[:y_margin, :] = 0
    mask[h - y_margin :, :] = 0
    mask[:, :x_margin] = 0
    mask[:, w - x_margin :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 80), max(3, h // 80)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    min_area = h * w * 0.01
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw * bh >= min_area and bh >= h * 0.12 and bw >= w * 0.10:
            boxes.append((x, y, bw, bh))

    groups: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda item: item[1] + item[3] / 2):
        center_y = box[1] + box[3] / 2
        for group in groups:
            group_center = np.mean([item[1] + item[3] / 2 for item in group])
            if abs(center_y - group_center) < h * 0.12:
                group.append(box)
                break
        else:
            groups.append([box])

    strip_boxes: list[tuple[int, int, int, int]] = []
    for group in groups:
        x1 = min(item[0] for item in group)
        y1 = min(item[1] for item in group)
        x2 = max(item[0] + item[2] for item in group)
        y2 = max(item[1] + item[3] for item in group)
        if x2 - x1 >= w * 0.45:
            strip_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return sorted(strip_boxes, key=lambda item: item[1])


def measure_strip(
    gray: np.ndarray,
    box: tuple[int, int, int, int],
    *,
    expected_levels: int,
    min_delta: float,
) -> GrayStrip:
    x, y, width, height = box
    blocks: list[GrayBlock] = []
    for index in range(expected_levels):
        block_x1 = round(x + width * index / expected_levels)
        block_x2 = round(x + width * (index + 1) / expected_levels)
        sample_margin_x = max(1, int((block_x2 - block_x1) * 0.16))
        sample_margin_y = max(1, int(height * 0.12))
        sample = gray[
            y + sample_margin_y : y + height - sample_margin_y,
            block_x1 + sample_margin_x : block_x2 - sample_margin_x,
        ]
        blocks.append(
            GrayBlock(
                index=index + 1,
                x=block_x1,
                y=y,
                width=block_x2 - block_x1,
                height=height,
                mean_gray=float(np.mean(sample)),
                std_gray=float(np.std(sample)),
            )
        )

    means = [block.mean_gray for block in blocks]
    adjacent_deltas = [abs(means[index + 1] - means[index]) for index in range(len(means) - 1)]
    distinguishable_levels = 1 + sum(delta >= min_delta for delta in adjacent_deltas)
    return GrayStrip(x, y, width, height, blocks, adjacent_deltas, distinguishable_levels)


def make_annotation(
    image: np.ndarray,
    strips: list[GrayStrip],
    best_strip: GrayStrip,
    min_delta: float,
) -> np.ndarray:
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)

    for strip in strips:
        is_best = strip is best_strip
        color = (0, 220, 0) if is_best else (0, 180, 255)
        for block in strip.blocks:
            cv2.rectangle(
                annotated,
                (block.x, block.y),
                (block.x + block.width, block.y + block.height),
                color,
                1,
            )
            label = f"{block.index}:{block.mean_gray:.0f}"
            cv2.putText(
                annotated,
                label,
                (block.x + 2, block.y + max(14, block.height // 2)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )

    text = f"distinguishable: {best_strip.distinguishable_levels}/{len(best_strip.blocks)}  delta>={min_delta:g}"
    cv2.putText(annotated, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return annotated
