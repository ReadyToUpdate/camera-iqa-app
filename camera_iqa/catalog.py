from __future__ import annotations


METRIC_GROUPS: dict[str, tuple[str, ...]] = {
    "客观指标": (
        "sharpness_laplacian_var",
        "tenengrad",
        "brightness_mean",
        "contrast_std",
        "dynamic_range",
        "noise_estimate",
        "color_cast_score",
        "overexposed_ratio",
        "underexposed_ratio",
        "occlusion_dark_block_ratio",
        "stripe_score",
    ),
    "客观缺陷指标": (
        "dark_corner_score",
        "bright_corner_score",
        "black_border_ratio",
        "low_light_stripe_score",
        "hot_pixel_ratio",
        "dead_pixel_ratio",
    ),
}


DEFECT_GROUPS: dict[str, tuple[str, ...]] = {
    "客观缺陷": (
        "black_screen",
        "white_screen",
        "blur",
        "exposure_abnormal",
        "low_contrast",
        "high_noise",
        "color_cast",
        "over_exposure",
        "under_exposure",
        "occlusion_suspected",
        "stripe_suspected",
        "dark_corner",
        "bright_corner",
        "black_border",
        "low_light_stripe",
        "hot_pixel",
        "dead_pixel",
    ),
}


def all_metric_keys() -> list[str]:
    return [metric for metrics in METRIC_GROUPS.values() for metric in metrics]


def metric_group_for(metric_key: str) -> str:
    for group, metrics in METRIC_GROUPS.items():
        if metric_key in metrics:
            return group
    return "未分类指标"


def defect_group_for(defect_code: str) -> str:
    for group, defects in DEFECT_GROUPS.items():
        if defect_code in defects:
            return group
    return "未分类缺陷"
