import cv2
import numpy as np

from camera_iqa.detectors import detect_defects
from camera_iqa.metrics import calculate_metrics


def objective_config():
    return {
        "thresholds": {
            "dark_corner_score": {"warn_above": 0.22, "fail_above": 0.38},
            "bright_corner_score": {"warn_above": 0.22, "fail_above": 0.38},
            "black_border_ratio": {"warn_above": 0.06, "fail_above": 0.12},
            "low_light_stripe_score": {"warn_above": 0.18, "fail_above": 0.30},
            "hot_pixel_ratio": {"warn_above": 0.00005, "fail_above": 0.00012},
            "dead_pixel_ratio": {"warn_above": 0.00005, "fail_above": 0.00012},
        }
    }


def uniform_image(value=120, size=160):
    return np.full((size, size, 3), value, dtype=np.uint8)


def codes_for(image):
    metrics = calculate_metrics(image)
    _, _, defects = detect_defects(metrics, objective_config())
    return metrics, {defect.code for defect in defects}


def test_detects_dark_corners_from_corner_falloff():
    image = uniform_image(130)
    image[:36, :36] = 45
    image[:36, -36:] = 45
    image[-36:, :36] = 45
    image[-36:, -36:] = 45

    metrics, codes = codes_for(image)

    assert metrics["dark_corner_score"] > 0.38
    assert "dark_corner" in codes


def test_detects_bright_corners_from_corner_glare():
    image = uniform_image(95)
    image[:36, :36] = 210
    image[:36, -36:] = 210

    metrics, codes = codes_for(image)

    assert metrics["bright_corner_score"] > 0.38
    assert "bright_corner" in codes


def test_detects_black_border_bars():
    image = uniform_image(118)
    image[:14, :] = 0
    image[-14:, :] = 0
    image[:, :10] = 0
    image[:, -10:] = 0

    metrics, codes = codes_for(image)

    assert metrics["black_border_ratio"] > 0.12
    assert "black_border" in codes


def test_detects_low_light_periodic_stripes():
    image = uniform_image(34)
    for row in range(0, image.shape[0], 8):
        image[row : row + 3] = 64

    metrics, codes = codes_for(image)

    assert metrics["low_light_stripe_score"] > 0.30
    assert "low_light_stripe" in codes


def test_detects_isolated_hot_and_dead_pixels():
    image = uniform_image(115)
    for y, x in [(20, 20), (35, 80), (90, 45), (120, 130)]:
        image[y, x] = 255
    for y, x in [(25, 125), (50, 55), (100, 100), (130, 30)]:
        image[y, x] = 0

    metrics, codes = codes_for(image)

    assert metrics["hot_pixel_ratio"] > 0.00012
    assert metrics["dead_pixel_ratio"] > 0.00012
    assert "hot_pixel" in codes
    assert "dead_pixel" in codes


def test_objective_metrics_do_not_flag_clean_gradient():
    x = np.linspace(60, 180, 160, dtype=np.uint8)
    gray = np.tile(x, (160, 1))
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    metrics, codes = codes_for(image)

    assert metrics["dark_corner_score"] < 0.22
    assert metrics["bright_corner_score"] < 0.22
    assert metrics["black_border_ratio"] < 0.06
    assert metrics["low_light_stripe_score"] < 0.18
    assert metrics["hot_pixel_ratio"] == 0.0
    assert metrics["dead_pixel_ratio"] == 0.0
    assert codes == set()
