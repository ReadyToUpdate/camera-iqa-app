from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


DEFAULT_MARKER_VALUES = (12, 16, 20, 24, 28, 32, 36, 40)


@dataclass(frozen=True, slots=True)
class WedgeROI:
    x: int
    y: int
    width: int
    height: int
    border: int = 4

    @property
    def inner_x(self) -> int:
        return self.x + self.border

    @property
    def inner_y(self) -> int:
        return self.y + self.border

    @property
    def inner_width(self) -> int:
        return max(0, self.width - 2 * self.border)

    @property
    def inner_height(self) -> int:
        return max(0, self.height - 2 * self.border)


@dataclass(frozen=True, slots=True)
class HyResWedgeResult:
    resolved_marker: int
    lines_per_picture_height: int
    resolved_count: int
    marker_scores: dict[int, float]
    roi: WedgeROI | None = None


RedWedgeROI = WedgeROI


def detect_red_wedge_roi(image: np.ndarray, *, min_area_ratio: float = 0.0005) -> WedgeROI:
    """Find the red annotation box around a HyRes wedge target."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR color image")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(hsv, (0, 70, 80), (12, 255, 255))
    upper_red = cv2.inRange(hsv, (168, 70, 80), (180, 255, 255))
    mask = cv2.bitwise_or(lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No red wedge ROI marker was detected")

    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * min_area_ratio
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        if area < min_area or width < 8 or height < 8:
            continue
        aspect = width / max(height, 1)
        if 0.18 <= aspect <= 1.4:
            red_fill = float(np.count_nonzero(mask[y : y + height, x : x + width])) / area
            score = area * (1.0 + red_fill)
            candidates.append((score, (x, y, width, height)))

    if not candidates:
        raise ValueError("Red pixels were found, but none formed a plausible wedge ROI")

    _score, (x, y, width, height) = max(candidates, key=lambda item: item[0])
    border = _estimate_border(mask[y : y + height, x : x + width])
    return WedgeROI(x=x, y=y, width=width, height=height, border=border)


def crop_red_wedge(image: np.ndarray, roi: WedgeROI) -> np.ndarray:
    """Return the inside of a red wedge ROI, excluding the red border."""
    x1 = max(0, roi.inner_x)
    y1 = max(0, roi.inner_y)
    x2 = min(image.shape[1], roi.inner_x + roi.inner_width)
    y2 = min(image.shape[0], roi.inner_y + roi.inner_height)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("ROI inner crop is empty")
    return image[y1:y2, x1:x2].copy()


def detect_hyres_wedge_roi(image: np.ndarray) -> WedgeROI:
    """Find the unannotated HyRes wedge line block, excluding labels and neighbors."""
    scale_roi = _detect_hyres_scale_roi(image)
    scale_crop = crop_wedge(image, scale_roi)
    block_roi = _wedge_block_roi_from_scale_crop(_as_gray(scale_crop))
    return WedgeROI(
        x=scale_roi.x + block_roi.x,
        y=scale_roi.y + block_roi.y,
        width=block_roi.width,
        height=block_roi.height,
        border=0,
    )


def _detect_hyres_scale_roi(image: np.ndarray) -> WedgeROI:
    """Find an unannotated HyRes scale target by its own scale/tick structure."""
    gray = _as_gray(image)
    height, width = gray.shape
    if height < 40 or width < 30:
        raise ValueError("Image is too small to contain a HyRes wedge target")

    best_score = 0.0
    best_box: tuple[int, int, int, int] | None = None
    min_h = max(48, round(height * 0.18))
    max_h = max(min_h, round(height * 0.38))
    for window_h in range(min_h, max_h + 1, max(8, min_h // 5)):
        min_w = max(24, round(window_h * 0.24))
        max_w = min(max(70, round(window_h * 0.48)), width)
        for window_w in range(min_w, max_w + 1, max(6, min_w // 4)):
            step_y = max(6, window_h // 8)
            step_x = max(5, window_w // 6)
            for y in range(0, height - window_h + 1, step_y):
                for x in range(0, width - window_w + 1, step_x):
                    crop = gray[y : y + window_h, x : x + window_w]
                    score = _wedge_structure_score(crop)
                    if score > best_score:
                        best_score = score
                        best_box = (x, y, window_w, window_h)

    if best_box is None or best_score < 2.35:
        raise ValueError("No unannotated HyRes wedge target was detected")

    x, y, box_w, box_h = _refine_wedge_box(gray, best_box)
    return WedgeROI(x=x, y=y, width=box_w, height=box_h, border=0)


def crop_wedge(image: np.ndarray, roi: WedgeROI) -> np.ndarray:
    """Return the full detected wedge ROI. Unlike red-box crops, no border is removed."""
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(image.shape[1], roi.x + roi.width)
    y2 = min(image.shape[0], roi.y + roi.height)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("ROI crop is empty")
    return image[y1:y2, x1:x2].copy()


def analyze_hyres_wedge(
    image: np.ndarray,
    *,
    marker_values: Sequence[int] = DEFAULT_MARKER_VALUES,
) -> tuple[np.ndarray, HyResWedgeResult]:
    roi = detect_hyres_wedge_roi(image)
    crop = crop_wedge(image, roi)
    result = estimate_hyres_wedge(crop, marker_values=marker_values, roi=roi)
    return crop, result


def analyze_red_wedge(
    image: np.ndarray,
    *,
    marker_values: Sequence[int] = DEFAULT_MARKER_VALUES,
) -> tuple[np.ndarray, HyResWedgeResult]:
    roi = detect_red_wedge_roi(image)
    crop = crop_red_wedge(image, roi)
    result = estimate_hyres_wedge(crop, marker_values=marker_values, roi=roi)
    return crop, result


def estimate_hyres_wedge(
    wedge_image: np.ndarray,
    *,
    marker_values: Sequence[int] = DEFAULT_MARKER_VALUES,
    min_contrast: float = 0.055,
    roi: WedgeROI | None = None,
) -> HyResWedgeResult:
    """
    Estimate HyRes wedge resolution using the 100x line markers.

    The target scale is labeled as values in 100x lines per picture height, so
    marker 32 reports 3200 lines/picture-height when it is the last resolved
    marker. A marker is considered resolved when the local band has enough
    black/white modulation and at least two dark line groups.
    """
    gray = _as_gray(wedge_image)
    height, width = gray.shape
    if height < 8 or width < 8:
        raise ValueError("Wedge image is too small to analyze")
    if not marker_values:
        raise ValueError("At least one marker value is required")

    marker_scores: dict[int, float] = {}
    resolved_marker = int(marker_values[0])
    resolved_count = 0
    band_height = max(12, height // len(marker_values))

    for index, marker in enumerate(marker_values):
        if len(marker_values) == 1:
            center_y = height // 2
        else:
            center_y = round(index * (height - 1) / (len(marker_values) - 1))
        top = max(0, center_y - band_height // 2)
        bottom = min(height, center_y + band_height // 2)
        band = gray[top:bottom, :]
        score = _hyres31_band_score(band)
        marker_scores[int(marker)] = score
        if score >= min_contrast:
            resolved_marker = int(marker)
            resolved_count += 1
        elif resolved_count:
            break

    return HyResWedgeResult(
        resolved_marker=resolved_marker,
        lines_per_picture_height=resolved_marker * 100,
        resolved_count=resolved_count,
        marker_scores=marker_scores,
        roi=roi,
    )


def black_line_count(profile: np.ndarray, *, min_gradient: float = 15.0, min_depth: float = 25.0) -> int:
    """Count dark lines whose left and right edges have enough contrast."""
    values = np.asarray(profile, dtype=np.float32).reshape(-1)
    if values.size < 3:
        return 0

    smooth = cv2.GaussianBlur(values.reshape(1, -1), (1, 3), 0).reshape(-1)
    low_threshold = min(float(np.percentile(smooth, 40)), float(np.mean(smooth) - min_depth * 0.25))
    dark = smooth <= low_threshold
    padded = np.concatenate(([False], dark, [False]))
    starts = np.flatnonzero((~padded[:-1]) & padded[1:])
    ends = np.flatnonzero(padded[:-1] & (~padded[1:]))

    count = 0
    for start, end in zip(starts, ends):
        left = smooth[max(0, start - 2) : start]
        center = smooth[start:end]
        right = smooth[end : min(smooth.size, end + 2)]
        if center.size == 0 or left.size == 0 or right.size == 0:
            continue
        center_min = float(np.min(center))
        left_rise = float(np.max(left) - center_min)
        right_rise = float(np.max(right) - center_min)
        if min(left_rise, right_rise) >= min_gradient and max(left_rise, right_rise) >= min_depth:
            count += 1
    return count


def save_wedge_crop(path: str | Path, crop: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = output.suffix or ".png"
    ok, encoded = cv2.imencode(ext, crop)
    if not ok:
        raise ValueError(f"Cannot encode wedge crop as {ext}")
    encoded.tofile(str(output))


def _estimate_border(mask_crop: np.ndarray) -> int:
    rows = np.count_nonzero(mask_crop, axis=1)
    cols = np.count_nonzero(mask_crop, axis=0)
    row_threshold = max(1, mask_crop.shape[1] // 4)
    col_threshold = max(1, mask_crop.shape[0] // 4)
    row_hits = np.flatnonzero(rows > row_threshold)
    col_hits = np.flatnonzero(cols > col_threshold)
    thicknesses: list[int] = []
    if row_hits.size:
        thicknesses.append(_edge_run(row_hits, from_start=True))
        thicknesses.append(_edge_run(mask_crop.shape[0] - 1 - row_hits[::-1], from_start=True))
    if col_hits.size:
        thicknesses.append(_edge_run(col_hits, from_start=True))
        thicknesses.append(_edge_run(mask_crop.shape[1] - 1 - col_hits[::-1], from_start=True))
    positive = [value for value in thicknesses if value > 0]
    if not positive:
        return max(2, round(min(mask_crop.shape[:2]) * 0.04))
    return int(max(2, min(12, round(float(np.median(positive))))))


def _edge_run(indices: np.ndarray, *, from_start: bool) -> int:
    if not from_start or indices.size == 0 or indices[0] > 1:
        return 0
    run = 1
    for previous, current in zip(indices, indices[1:]):
        if current - previous > 1:
            break
        run += 1
    return run


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a grayscale or BGR image")


def _line_modulation_score(band: np.ndarray) -> float:
    if band.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(band, (3, 3), 0)
    col_profile = np.mean(255.0 - blurred.astype(np.float32), axis=0)
    dynamic = float(np.percentile(col_profile, 92) - np.percentile(col_profile, 8))
    if dynamic <= 1.0:
        return 0.0
    normalized = (col_profile - float(np.min(col_profile))) / dynamic
    dark_runs = _count_runs(normalized > 0.45)
    michelson = dynamic / (float(np.percentile(blurred, 92)) + float(np.percentile(255 - blurred, 92)) + 1.0)
    if dark_runs < 2:
        return 0.0
    return float(min(1.0, michelson * min(dark_runs, 6) / 2.0))


def _hyres31_band_score(band: np.ndarray) -> float:
    if band.size == 0:
        return 0.0
    blurred = cv2.GaussianBlur(band, (3, 3), 0)
    profile = np.mean(blurred.astype(np.float32), axis=1)
    line_count = black_line_count(profile)
    modulation = _profile_modulation_score(profile)
    if line_count < 1:
        return 0.0
    return float(min(1.0, modulation * min(line_count, 3)))


def _profile_modulation_score(profile: np.ndarray) -> float:
    values = np.asarray(profile, dtype=np.float32).reshape(-1)
    if values.size < 3:
        return 0.0
    high = float(np.percentile(values, 90))
    low = float(np.percentile(values, 10))
    if high <= low:
        return 0.0
    return float((high - low) / (high + low + 1.0))


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    starts = np.flatnonzero((~padded[:-1]) & padded[1:])
    ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
    return int(np.count_nonzero((ends - starts) >= 1))


def _wedge_structure_score(crop: np.ndarray) -> float:
    height, width = crop.shape
    dark = crop < 150
    density = float(np.mean(dark))
    if density < 0.025 or density > 0.42:
        return 0.0

    col_counts = np.count_nonzero(dark, axis=0)
    row_counts = np.count_nonzero(dark, axis=1)
    vertical_density = float(np.max(col_counts)) / height
    if vertical_density < 0.18:
        return 0.0

    tick_min = max(2, round(width * 0.08))
    tick_max = max(tick_min + 1, round(width * 0.72))
    tick_runs = _count_runs((row_counts >= tick_min) & (row_counts <= tick_max))
    if tick_runs < 4:
        return 0.0

    marker_score = _hyres_marker_sequence_score(crop)
    density_bonus = 1.0 - abs(density - 0.13) / 0.13
    return (
        min(vertical_density, 0.65) * 2.2
        + min(tick_runs / 8.0, 1.0) * 1.5
        + marker_score * 2.2
        + max(0.0, density_bonus) * 0.4
    )


def _wedge_block_roi_from_scale_crop(gray: np.ndarray) -> WedgeROI:
    height, width = gray.shape
    if height < 20 or width < 8:
        raise ValueError("HyRes scale crop is too small")

    scale_mask = gray < 215
    col_counts = np.count_nonzero(scale_mask, axis=0)
    min_vertical = max(12, round(height * 0.35))
    vertical_cols = np.flatnonzero(col_counts >= min_vertical)
    if vertical_cols.size == 0:
        dark_cols = np.flatnonzero(np.count_nonzero(gray < 170, axis=0) >= max(3, round(height * 0.08)))
        if dark_cols.size == 0:
            raise ValueError("Cannot isolate HyRes wedge block from scale crop")
        left = int(dark_cols[0])
    else:
        left = int(vertical_cols[0])

    search = gray[:, left:]
    dark = search < 190
    row_hits = np.flatnonzero(np.count_nonzero(dark, axis=1) >= 1)
    if row_hits.size == 0:
        raise ValueError("Cannot isolate HyRes wedge block rows")

    y1 = max(0, int(row_hits[0]) - 2)
    y2 = min(height, int(row_hits[-1]) + 3)
    row_band = dark[y1:y2]
    col_activity = np.count_nonzero(row_band, axis=0)
    col_hits = np.flatnonzero(col_activity >= 1)
    if col_hits.size == 0:
        raise ValueError("Cannot isolate HyRes wedge block columns")

    # Keep the vertical scale line and its short line groups, but stop before
    # label digits. Digits form disconnected dark columns well to the right.
    raw_right = int(col_hits[-1]) + left + 1
    gap_limit = max(3, round(width * 0.10))
    active_threshold = max(3, round((y2 - y1) * 0.025))
    active_cols = np.flatnonzero(col_activity >= active_threshold)
    if active_cols.size:
        split = active_cols[0]
        for previous, current in zip(active_cols, active_cols[1:]):
            if current - previous >= gap_limit:
                break
            split = current
        raw_right = min(raw_right, left + int(split) + 2)

    x1 = max(0, left - 2)
    x2 = min(width, max(x1 + 4, raw_right + 1))
    return WedgeROI(x=x1, y=y1, width=x2 - x1, height=y2 - y1, border=0)


def _hyres_marker_sequence_score(crop: np.ndarray) -> float:
    height, _width = crop.shape
    band_height = max(6, height // (len(DEFAULT_MARKER_VALUES) * 2))
    scores = []
    for index in range(len(DEFAULT_MARKER_VALUES)):
        center_y = round((index + 0.5) * height / len(DEFAULT_MARKER_VALUES))
        top = max(0, center_y - band_height // 2)
        bottom = min(height, center_y + band_height // 2)
        scores.append(_line_modulation_score(crop[top:bottom, :]))
    positive = [score for score in scores if score > 0.08]
    if not positive:
        return 0.0
    return float(min(1.0, np.mean(positive) * len(positive) / 3.0))


def _refine_wedge_box(gray: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, width, height = box
    crop = gray[y : y + height, x : x + width]
    dark = crop < 165
    rows = np.count_nonzero(dark, axis=1)
    cols = np.count_nonzero(dark, axis=0)
    row_hits = np.flatnonzero(rows >= max(2, round(width * 0.05)))
    col_hits = np.flatnonzero(cols >= max(3, round(height * 0.08)))
    if row_hits.size == 0 or col_hits.size == 0:
        return box

    pad_y = max(2, round(height * 0.04))
    pad_x = max(2, round(width * 0.08))
    y1 = max(0, y + int(row_hits[0]) - pad_y)
    y2 = min(gray.shape[0], y + int(row_hits[-1]) + pad_y + 1)
    x1 = max(0, x + int(col_hits[0]) - pad_x)
    x2 = min(gray.shape[1], x + int(col_hits[-1]) + pad_x + 1)
    tightened = _tighten_to_scale_line(gray[y1:y2, x1:x2])
    if tightened is None:
        return x1, y1, x2 - x1, y2 - y1

    sx, sy, sw, sh = tightened
    return x1 + sx, y1 + sy, sw, sh


def _tighten_to_scale_line(crop: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = crop.shape
    if height < 40 or width < 24:
        return None

    scale_mask = crop < 210
    best: tuple[float, int, int, int] | None = None
    for col in range(2, max(3, width - 12)):
        line_band = scale_mask[:, max(0, col - 1) : min(width, col + 2)]
        line_rows = np.mean(line_band, axis=1) > 0.35
        run_start, run_end = _longest_true_run(line_rows)
        run_len = run_end - run_start
        if run_len < max(35, round(height * 0.35)):
            continue

        vertical_pixels = crop[run_start:run_end, max(0, col - 1) : min(width, col + 2)]
        if vertical_pixels.size and float(np.mean(vertical_pixels)) < 45:
            continue

        right = crop[run_start:run_end, col : min(width, col + 52)]
        right_dark = right < 185
        row_counts = np.count_nonzero(right_dark, axis=1)
        tick_rows = (row_counts >= 2) & (row_counts <= max(3, round(right.shape[1] * 0.75)))
        tick_runs = _count_runs(tick_rows)
        if tick_runs < 5:
            continue

        left = crop[run_start:run_end, : max(0, col - 4)]
        left_penalty = float(np.mean(left < 80)) if left.size else 0.0
        score = run_len / height + min(tick_runs / 8.0, 1.0) - left_penalty
        if best is None or score > best[0]:
            best = (score, col, run_start, run_end)

    if best is None:
        return None

    _score, col, run_start, run_end = best
    pad_y = max(3, round(height * 0.025))
    y1 = max(0, run_start - pad_y)
    y2 = min(height, run_end + pad_y)
    x1 = max(0, col - max(4, round(width * 0.06)))
    x2 = min(width, col + max(36, round(width * 0.58)))

    content = crop[y1:y2, x1:x2] < 190
    col_hits = np.flatnonzero(np.count_nonzero(content, axis=0) >= 1)
    if col_hits.size:
        x1 = max(0, x1 + int(col_hits[0]) - 2)
        x2 = min(width, x1 + int(col_hits[-1] - col_hits[0]) + 5)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    if mask.size == 0:
        return 0, 0
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    starts = np.flatnonzero((~padded[:-1]) & padded[1:])
    ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
    if starts.size == 0:
        return 0, 0
    lengths = ends - starts
    index = int(np.argmax(lengths))
    return int(starts[index]), int(ends[index])
