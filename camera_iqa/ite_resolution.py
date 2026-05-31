from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from camera_iqa.metrics import read_image, save_image
from camera_iqa.models import SUPPORTED_EXTENSIONS


DEFAULT_ITE_MARKERS = (200, 300, 400, 500, 600, 700, 800, 1000)
DEFAULT_MIN_SCORE = 0.18


@dataclass(frozen=True, slots=True)
class ResolutionROI:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ITEResolutionCalibration:
    source_width: int
    source_height: int
    roi: ResolutionROI
    marker_values: tuple[int, ...] = DEFAULT_ITE_MARKERS
    min_score: float = DEFAULT_MIN_SCORE


@dataclass(frozen=True, slots=True)
class ITEResolutionResult:
    resolved_lines: int
    confidence: float
    marker_scores: dict[int, float]
    roi: ResolutionROI


@dataclass(frozen=True, slots=True)
class ITEResolutionRow:
    image: str
    path: str
    lux: float | None
    resolved_lines: int
    confidence: float
    roi: ResolutionROI
    marker_scores: dict[int, float]
    status: str = "done"
    error: str = ""


def calibrate_ite_roi(
    image: np.ndarray,
    roi: ResolutionROI,
    *,
    marker_values: Iterable[int] = DEFAULT_ITE_MARKERS,
    min_score: float = DEFAULT_MIN_SCORE,
    annotation_path: str | Path | None = None,
) -> ITEResolutionCalibration:
    if image.ndim not in (2, 3):
        raise ValueError("Expected a grayscale or BGR image")
    _crop_roi(image, roi)
    calibration = ITEResolutionCalibration(
        source_width=int(image.shape[1]),
        source_height=int(image.shape[0]),
        roi=roi,
        marker_values=tuple(int(value) for value in marker_values),
        min_score=float(min_score),
    )
    if annotation_path is not None:
        save_image(annotation_path, make_roi_annotation(image, roi))
    return calibration


def save_calibration(calibration: ITEResolutionCalibration, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_width": calibration.source_width,
        "source_height": calibration.source_height,
        "roi": asdict(calibration.roi),
        "marker_values": list(calibration.marker_values),
        "min_score": calibration.min_score,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> ITEResolutionCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        roi_payload = payload["roi"]
        roi = ResolutionROI(
            x=int(roi_payload["x"]),
            y=int(roi_payload["y"]),
            width=int(roi_payload["width"]),
            height=int(roi_payload["height"]),
        )
        return ITEResolutionCalibration(
            source_width=int(payload["source_width"]),
            source_height=int(payload["source_height"]),
            roi=roi,
            marker_values=tuple(int(value) for value in payload.get("marker_values", DEFAULT_ITE_MARKERS)),
            min_score=float(payload.get("min_score", DEFAULT_MIN_SCORE)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ITE ROI calibration: {path}") from exc


def scale_roi(
    roi: ResolutionROI,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> ResolutionROI:
    source_width, source_height = source_size
    target_width, target_height = target_size
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_size must be positive")
    return ResolutionROI(
        x=round(roi.x * target_width / source_width),
        y=round(roi.y * target_height / source_height),
        width=max(1, round(roi.width * target_width / source_width)),
        height=max(1, round(roi.height * target_height / source_height)),
    )


def analyze_ite_resolution(
    image: np.ndarray,
    calibration: ITEResolutionCalibration,
) -> ITEResolutionResult:
    roi = scale_roi(
        calibration.roi,
        source_size=(calibration.source_width, calibration.source_height),
        target_size=(int(image.shape[1]), int(image.shape[0])),
    )
    crop = _crop_roi(image, roi)
    gray = _as_gray(crop)
    marker_scores: dict[int, float] = {}
    resolved_lines = 0
    confidence = 0.0
    band_count = len(calibration.marker_values)
    if band_count == 0:
        raise ValueError("At least one marker value is required")

    for index, marker in enumerate(calibration.marker_values):
        y1 = round(index * gray.shape[0] / band_count)
        y2 = round((index + 1) * gray.shape[0] / band_count)
        band = gray[y1:y2, :]
        score = _ite_band_score(band)
        marker_scores[int(marker)] = score
        if score >= calibration.min_score:
            resolved_lines = int(marker)
            confidence = score
        else:
            break

    return ITEResolutionResult(
        resolved_lines=resolved_lines,
        confidence=confidence if resolved_lines else 0.0,
        marker_scores=marker_scores,
        roi=roi,
    )


def analyze_ite_resolution_path(
    path: str | Path,
    calibration: ITEResolutionCalibration,
) -> ITEResolutionRow:
    image_path = Path(path)
    try:
        result = analyze_ite_resolution(read_image(image_path), calibration)
        return ITEResolutionRow(
            image=image_path.name,
            path=str(image_path),
            lux=parse_lux_from_filename(image_path.name),
            resolved_lines=result.resolved_lines,
            confidence=result.confidence,
            roi=result.roi,
            marker_scores=result.marker_scores,
        )
    except Exception as exc:
        roi = calibration.roi
        return ITEResolutionRow(
            image=image_path.name,
            path=str(image_path),
            lux=parse_lux_from_filename(image_path.name),
            resolved_lines=0,
            confidence=0.0,
            roi=roi,
            marker_scores={int(marker): 0.0 for marker in calibration.marker_values},
            status="error",
            error=str(exc),
        )


def analyze_ite_resolution_folder(
    folder: str | Path,
    calibration: ITEResolutionCalibration,
) -> list[ITEResolutionRow]:
    root = Path(folder)
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [analyze_ite_resolution_path(path, calibration) for path in paths]


def write_ite_resolution_csv(rows: list[ITEResolutionRow], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    marker_values = _csv_marker_values(rows)
    fieldnames = [
        "image",
        "path",
        "lux",
        "resolved_lines",
        "confidence",
        "roi_x",
        "roi_y",
        "roi_width",
        "roi_height",
        "status",
        "error",
        *[f"score_{marker}" for marker in marker_values],
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = {
                "image": row.image,
                "path": row.path,
                "lux": "" if row.lux is None else f"{row.lux:g}",
                "resolved_lines": str(row.resolved_lines),
                "confidence": f"{row.confidence:.6g}",
                "roi_x": str(row.roi.x),
                "roi_y": str(row.roi.y),
                "roi_width": str(row.roi.width),
                "roi_height": str(row.roi.height),
                "status": row.status,
                "error": row.error,
            }
            for marker in marker_values:
                data[f"score_{marker}"] = f"{row.marker_scores.get(marker, 0.0):.6g}"
            writer.writerow(data)


def parse_lux_from_filename(filename: str) -> float | None:
    stem = Path(filename).stem.lower()
    patterns = (
        r"(?:^|[_-])lux[_-]?(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*lux(?:$|[_-])",
        r"(?:^|[_-])illum[_-]?(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            return float(match.group(1))
    return None


def make_roi_annotation(image: np.ndarray, roi: ResolutionROI) -> np.ndarray:
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        annotated,
        (roi.x, roi.y),
        (roi.x + roi.width, roi.y + roi.height),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated,
        "ITE ROI",
        (roi.x, max(16, roi.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _crop_roi(image: np.ndarray, roi: ResolutionROI) -> np.ndarray:
    if roi.width <= 0 or roi.height <= 0:
        raise ValueError("ROI width and height must be positive")
    x1 = max(0, roi.x)
    y1 = max(0, roi.y)
    x2 = min(image.shape[1], roi.x + roi.width)
    y2 = min(image.shape[0], roi.y + roi.height)
    if x1 >= x2 or y1 >= y2:
        raise ValueError("ITE ROI crop is empty")
    return image[y1:y2, x1:x2].copy()


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError("Expected a grayscale or BGR image")


def _ite_band_score(band: np.ndarray) -> float:
    if band.size == 0 or band.shape[0] < 4 or band.shape[1] < 8:
        return 0.0
    y_margin = max(1, round(band.shape[0] * 0.16))
    x_margin = max(1, round(band.shape[1] * 0.06))
    sample = band[y_margin : band.shape[0] - y_margin, x_margin : band.shape[1] - x_margin]
    if sample.size == 0:
        sample = band
    blurred = cv2.GaussianBlur(sample, (3, 3), 0)
    profile = np.mean(blurred.astype(np.float32), axis=0)
    high = float(np.percentile(profile, 88))
    low = float(np.percentile(profile, 12))
    dynamic = high - low
    if dynamic <= 4.0:
        return 0.0

    threshold = low + dynamic * 0.38
    dark = profile <= threshold
    dark_runs = _count_runs(dark)
    transitions = int(np.count_nonzero(dark[1:] != dark[:-1])) if dark.size > 1 else 0
    if dark_runs < 2 or transitions < 4:
        return 0.0

    modulation = dynamic / (high + low + 1.0)
    run_factor = min(1.0, dark_runs / 5.0)
    transition_factor = min(1.0, transitions / 10.0)
    return float(min(1.0, modulation * (0.65 + 0.25 * run_factor + 0.10 * transition_factor)))


def _count_runs(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    starts = np.flatnonzero((~padded[:-1]) & padded[1:])
    ends = np.flatnonzero(padded[:-1] & (~padded[1:]))
    return int(np.count_nonzero((ends - starts) >= 1))


def _csv_marker_values(rows: list[ITEResolutionRow]) -> tuple[int, ...]:
    markers: set[int] = set(DEFAULT_ITE_MARKERS)
    for row in rows:
        markers.update(int(marker) for marker in row.marker_scores)
    return tuple(sorted(markers))
