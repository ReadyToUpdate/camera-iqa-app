import cv2
import numpy as np

from camera_iqa.hyres import (
    analyze_hyres_wedge,
    black_line_count,
    crop_red_wedge,
    crop_wedge,
    detect_hyres_wedge_roi,
    detect_red_wedge_roi,
    estimate_hyres_wedge,
)


def test_detect_red_wedge_roi_returns_inner_crop_without_red_border():
    image = np.full((160, 220, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (70, 30), (145, 132), (0, 0, 255), 4)
    image[42:120, 84:130] = 40

    roi = detect_red_wedge_roi(image)
    crop = crop_red_wedge(image, roi)

    assert roi.x <= 70
    assert roi.y <= 30
    assert roi.width >= 75
    assert roi.height >= 102
    assert crop.shape[:2] == (roi.inner_height, roi.inner_width)
    assert int(np.max(crop[:, :, 2])) < 255


def test_estimate_hyres_wedge_reports_highest_resolved_marker():
    gray = np.full((220, 30), 235, dtype=np.uint8)
    marker_values = (12, 16, 20, 24, 28, 32, 36, 40)
    for index, _marker in enumerate(marker_values[:6]):
        center = round((index + 0.5) * gray.shape[0] / len(marker_values))
        gray[center - 5 : center - 2, 4:24] = 25
        gray[center + 3 : center + 6, 4:24] = 30
    # Make the final two markers unresolved: low contrast, so the result should
    # stop at 32 instead of blindly returning the last scale mark.
    gray[round((6 + 0.5) * gray.shape[0] / len(marker_values)) - 6 :, :] = 180

    result = estimate_hyres_wedge(gray, marker_values=marker_values)

    assert result.resolved_marker == 32
    assert result.lines_per_picture_height == 3200
    assert result.resolved_count >= 5


def test_detect_hyres_wedge_roi_returns_only_unannotated_wedge_block():
    image = np.full((480, 360, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (18, 120), (58, 260), (25, 25, 25), -1)
    _draw_synthetic_wedge(image, 145, 120)

    roi = detect_hyres_wedge_roi(image)
    crop, result = analyze_hyres_wedge(image)

    assert abs(roi.x - 142) <= 6
    assert abs(roi.y - 116) <= 10
    assert 8 <= roi.width <= 22
    assert 132 <= roi.height <= 158
    assert crop.shape[:2] == (roi.height, roi.width)
    assert result.roi == roi
    assert result.lines_per_picture_height >= 1600
    assert not _contains_label_digits(crop)


def test_detect_hyres_wedge_roi_tightens_candidate_around_scale_line():
    image = np.full((240, 180, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (16, 38), (26, 165), (20, 20, 20), -1)
    cv2.line(image, (36, 38), (36, 160), (180, 180, 180), 2)
    cv2.line(image, (34, 78), (66, 78), (180, 180, 180), 2)
    _draw_synthetic_wedge(image, 78, 42)

    roi = detect_hyres_wedge_roi(image)
    crop = image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    assert roi.x >= 74
    assert roi.width <= 22
    assert roi.height <= 158
    assert np.mean(crop[:, :6] < 60) < 0.15


def test_black_line_count_uses_gradient_confirmed_dark_lines():
    profile = np.array([230, 225, 35, 30, 225, 230, 40, 35, 230, 230], dtype=np.uint8)

    assert black_line_count(profile) == 2


def test_estimate_hyres_wedge_uses_first_unresolved_black_line_band():
    wedge = np.full((160, 24), 235, dtype=np.uint8)
    for index in range(5):
        y = 8 + index * 18
        wedge[y : y + 4, 4:20] = 30
        wedge[y + 8 : y + 12, 4:20] = 35
    wedge[8 + 5 * 18 : 8 + 5 * 18 + 12, 4:20] = 150

    result = estimate_hyres_wedge(wedge, marker_values=(12, 16, 20, 24, 28, 32, 36, 40))

    assert result.resolved_marker == 28
    assert result.lines_per_picture_height == 2800
    assert result.resolved_count == 4


def _draw_synthetic_wedge(image: np.ndarray, x: int, y: int) -> None:
    cv2.line(image, (x, y), (x, y + 136), (95, 95, 95), 2)
    for index, value in enumerate((12, 16, 20, 24, 28, 32, 36, 40)):
        yy = y + 7 + index * 16
        cv2.line(image, (x, yy), (x + 15, yy), (55, 55, 55), 1)
        cv2.putText(image, str(value), (x + 20, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (45, 45, 45), 1)


def _contains_label_digits(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    right_half = gray[:, gray.shape[1] // 2 :]
    return bool(np.mean(right_half < 80) > 0.16)
