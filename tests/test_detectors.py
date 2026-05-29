from camera_iqa.detectors import detect_defects


def base_config():
    return {
        "thresholds": {
            "sharpness_laplacian_var": {"warn_below": 120, "fail_below": 60},
            "brightness_mean": {"warn_below": 55, "fail_below": 35, "warn_above": 205, "fail_above": 230},
            "contrast_std": {"warn_below": 35, "fail_below": 22},
            "dynamic_range": {"warn_below": 90, "fail_below": 55},
            "noise_estimate": {"warn_above": 16, "fail_above": 28},
            "color_cast_score": {"warn_above": 22, "fail_above": 38},
            "overexposed_ratio": {"warn_above": 0.03, "fail_above": 0.10},
            "underexposed_ratio": {"warn_above": 0.12, "fail_above": 0.30},
            "black_screen": {"brightness_below": 12, "contrast_below": 8},
            "white_screen": {"brightness_above": 245, "contrast_below": 8},
            "occlusion_dark_block_ratio": {"warn_above": 0.35, "fail_above": 0.55},
            "stripe_score": {"warn_above": 0.22, "fail_above": 0.34},
        }
    }


def healthy_metrics():
    return {
        "sharpness_laplacian_var": 300,
        "tenengrad": 50,
        "brightness_mean": 120,
        "contrast_std": 50,
        "dynamic_range": 160,
        "noise_estimate": 8,
        "color_cast_score": 6,
        "overexposed_ratio": 0.0,
        "underexposed_ratio": 0.02,
        "occlusion_dark_block_ratio": 0.0,
        "stripe_score": 0.05,
    }


def test_black_screen_detection():
    metrics = healthy_metrics()
    metrics.update({"brightness_mean": 4, "contrast_std": 2, "underexposed_ratio": 1.0})
    verdict, severity, defects = detect_defects(metrics, base_config())
    assert verdict == "fail"
    assert severity == "fail"
    assert "black_screen" in {defect.code for defect in defects}


def test_blur_detection_changes_with_threshold():
    metrics = healthy_metrics()
    metrics["sharpness_laplacian_var"] = 80
    verdict, severity, defects = detect_defects(metrics, base_config())
    assert verdict == "warn"
    assert "blur" in {defect.code for defect in defects}

    config = base_config()
    config["thresholds"]["sharpness_laplacian_var"] = {"warn_below": 50, "fail_below": 25}
    verdict, severity, defects = detect_defects(metrics, config)
    assert verdict == "pass"
    assert defects == []


def test_color_cast_detection():
    metrics = healthy_metrics()
    metrics["color_cast_score"] = 45
    verdict, severity, defects = detect_defects(metrics, base_config())
    assert verdict == "fail"
    assert severity == "fail"
    assert "color_cast" in {defect.code for defect in defects}
