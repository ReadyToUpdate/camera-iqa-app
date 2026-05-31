import csv
import json

import cv2
import numpy as np

from camera_iqa.ite_resolution import (
    DEFAULT_ITE_MARKERS,
    ResolutionROI,
    analyze_ite_resolution,
    analyze_ite_resolution_folder,
    calibrate_ite_roi,
    load_calibration,
    parse_lux_from_filename,
    save_calibration,
    scale_roi,
    write_ite_resolution_csv,
)
from camera_iqa.cli import parse_roi
from camera_iqa.metrics import save_image
from camera_iqa.models import SUPPORTED_EXTENSIONS


def test_analyze_ite_resolution_reports_highest_continuous_resolved_marker():
    image = _synthetic_ite_image(resolved_markers=5)
    calibration = calibrate_ite_roi(image, ResolutionROI(30, 20, 96, 240))

    result = analyze_ite_resolution(image, calibration)

    assert result.resolved_lines == 600
    assert result.confidence > 0.2
    assert result.marker_scores[600] > result.marker_scores[700]


def test_analyze_ite_resolution_returns_zero_when_first_marker_is_unresolved():
    image = _synthetic_ite_image(resolved_markers=0)
    calibration = calibrate_ite_roi(image, ResolutionROI(30, 20, 96, 240))

    result = analyze_ite_resolution(image, calibration)

    assert result.resolved_lines == 0
    assert result.confidence == 0.0


def test_calibration_round_trips_json(tmp_path):
    image = _synthetic_ite_image(resolved_markers=len(DEFAULT_ITE_MARKERS))
    calibration = calibrate_ite_roi(image, ResolutionROI(30, 20, 96, 240))
    path = tmp_path / "roi.json"

    save_calibration(calibration, path)
    loaded = load_calibration(path)

    assert loaded.source_width == image.shape[1]
    assert loaded.source_height == image.shape[0]
    assert loaded.roi == ResolutionROI(30, 20, 96, 240)
    assert loaded.marker_values == DEFAULT_ITE_MARKERS
    assert json.loads(path.read_text(encoding="utf-8"))["roi"]["width"] == 96


def test_scale_roi_preserves_relative_geometry():
    roi = ResolutionROI(10, 20, 40, 80)

    scaled = scale_roi(roi, source_size=(200, 100), target_size=(400, 300))

    assert scaled == ResolutionROI(20, 60, 80, 240)


def test_parse_lux_from_filename_handles_common_patterns():
    assert parse_lux_from_filename("0.1lux.jpg") == 0.1
    assert parse_lux_from_filename("lux_0.05.png") == 0.05
    assert parse_lux_from_filename("illum-2.5.jpeg") == 2.5
    assert parse_lux_from_filename("baseline.jpg") is None


def test_parse_roi_accepts_comma_separated_rectangle():
    assert parse_roi("10,20,30,40") == ResolutionROI(10, 20, 30, 40)


def test_webp_images_are_supported_for_batch_input():
    assert ".webp" in SUPPORTED_EXTENSIONS


def test_folder_analysis_and_csv_output(tmp_path):
    image1 = _synthetic_ite_image(resolved_markers=4)
    image2 = _synthetic_ite_image(resolved_markers=2)
    save_image(tmp_path / "lux_1.0.png", image1)
    save_image(tmp_path / "0.1lux.png", image2)
    calibration = calibrate_ite_roi(image1, ResolutionROI(30, 20, 96, 240))

    rows = analyze_ite_resolution_folder(tmp_path, calibration)
    output = tmp_path / "lines.csv"
    write_ite_resolution_csv(rows, output)

    assert [row.image for row in rows] == ["0.1lux.png", "lux_1.0.png"]
    assert rows[0].lux == 0.1
    assert rows[0].resolved_lines == 300
    assert rows[1].lux == 1.0
    assert rows[1].resolved_lines == 500

    with output.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["image"] == "0.1lux.png"
    assert csv_rows[0]["resolved_lines"] == "300"
    assert "score_1000" in csv_rows[0]


def _synthetic_ite_image(*, resolved_markers: int) -> np.ndarray:
    image = np.full((300, 170, 3), 220, dtype=np.uint8)
    roi_x, roi_y, roi_w, roi_h = 30, 20, 96, 240
    band_h = roi_h // len(DEFAULT_ITE_MARKERS)
    for index, _marker in enumerate(DEFAULT_ITE_MARKERS):
        y1 = roi_y + index * band_h
        y2 = roi_y + (index + 1) * band_h
        band = image[y1:y2, roi_x : roi_x + roi_w]
        if index < resolved_markers:
            period = max(4, 12 - index)
            for x in range(8, roi_w - 8, period):
                cv2.rectangle(band, (x, 5), (min(x + 2, roi_w - 1), band_h - 6), (25, 25, 25), -1)
        else:
            band[:, :] = (178, 178, 178)
            if index == resolved_markers:
                cv2.GaussianBlur(band, (9, 9), 0, dst=band)
    return image
