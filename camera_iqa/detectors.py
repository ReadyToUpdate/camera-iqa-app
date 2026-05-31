from __future__ import annotations

from typing import Any

from camera_iqa.catalog import defect_group_for
from camera_iqa.config import threshold
from camera_iqa.models import Defect


SEVERITY_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def classify_value(value: float, rule: dict[str, Any]) -> str:
    severity = "pass"
    if "warn_below" in rule and value < float(rule["warn_below"]):
        severity = "warn"
    if "fail_below" in rule and value < float(rule["fail_below"]):
        severity = "fail"
    if "warn_above" in rule and value > float(rule["warn_above"]):
        severity = max_severity(severity, "warn")
    if "fail_above" in rule and value > float(rule["fail_above"]):
        severity = max_severity(severity, "fail")
    return severity


def max_severity(*severities: str) -> str:
    return max(severities, key=lambda item: SEVERITY_ORDER.get(item, 0))


def detect_defects(metrics: dict[str, float], config: dict[str, Any]) -> tuple[str, str, list[Defect]]:
    defects: list[Defect] = []

    black_rule = threshold(config, "black_screen")
    if metrics["brightness_mean"] < float(black_rule.get("brightness_below", 12)) and metrics["contrast_std"] < float(black_rule.get("contrast_below", 8)):
        defects.append(make_defect("black_screen", "黑屏", "fail", "整体亮度和对比度极低", pick(metrics, "brightness_mean", "contrast_std")))

    white_rule = threshold(config, "white_screen")
    if metrics["brightness_mean"] > float(white_rule.get("brightness_above", 245)) and metrics["contrast_std"] < float(white_rule.get("contrast_below", 8)):
        defects.append(make_defect("white_screen", "白屏", "fail", "整体亮度极高且缺少纹理", pick(metrics, "brightness_mean", "contrast_std")))

    add_metric_defect(defects, metrics, config, "sharpness_laplacian_var", "blur", "模糊", "清晰度指标低于阈值")
    add_metric_defect(defects, metrics, config, "brightness_mean", "exposure_abnormal", "曝光异常", "平均亮度超出阈值范围")
    add_metric_defect(defects, metrics, config, "contrast_std", "low_contrast", "低对比度", "对比度低于阈值")
    add_metric_defect(defects, metrics, config, "noise_estimate", "high_noise", "高噪声", "平坦区域噪声估计高于阈值")
    add_metric_defect(defects, metrics, config, "color_cast_score", "color_cast", "偏色", "RGB 通道均衡偏差高于阈值")
    add_metric_defect(defects, metrics, config, "overexposed_ratio", "over_exposure", "过曝", "过曝像素比例高于阈值")
    add_metric_defect(defects, metrics, config, "underexposed_ratio", "under_exposure", "欠曝", "暗部像素比例高于阈值")
    add_metric_defect(defects, metrics, config, "occlusion_dark_block_ratio", "occlusion_suspected", "遮挡疑似", "大面积低亮低纹理区域高于阈值")
    add_metric_defect(defects, metrics, config, "stripe_score", "stripe_suspected", "条纹疑似", "行/列均值突变比例高于阈值")
    add_metric_defect(defects, metrics, config, "dark_corner_score", "dark_corner", "暗角", "四角亮度明显低于中心区域")
    add_metric_defect(defects, metrics, config, "bright_corner_score", "bright_corner", "亮角", "四角亮度明显高于中心区域")
    add_metric_defect(defects, metrics, config, "black_border_ratio", "black_border", "黑边", "图像边缘存在近黑条带")
    add_metric_defect(defects, metrics, config, "low_light_stripe_score", "low_light_stripe", "低照度条纹", "低照度画面存在行/列方向周期性条纹")
    add_metric_defect(defects, metrics, config, "hot_pixel_ratio", "hot_pixel", "亮点", "存在孤立异常亮点")
    add_metric_defect(defects, metrics, config, "dead_pixel_ratio", "dead_pixel", "坏点", "存在孤立异常暗点")

    severity = "pass"
    for defect in defects:
        severity = max_severity(severity, defect.severity)
    verdict = "pass" if severity == "pass" else severity
    return verdict, severity, deduplicate_defects(defects)


def add_metric_defect(
    defects: list[Defect],
    metrics: dict[str, float],
    config: dict[str, Any],
    metric_key: str,
    code: str,
    label: str,
    reason: str,
) -> None:
    if metric_key not in metrics:
        return
    rule = threshold(config, metric_key)
    severity = classify_value(metrics[metric_key], rule)
    if severity != "pass":
        defects.append(make_defect(code, label, severity, reason, {metric_key: metrics[metric_key]}))


def make_defect(code: str, label: str, severity: str, reason: str, evidence: dict[str, float]) -> Defect:
    return Defect(code, label, severity, reason, evidence, defect_group_for(code))


def deduplicate_defects(defects: list[Defect]) -> list[Defect]:
    by_code: dict[str, Defect] = {}
    for defect in defects:
        existing = by_code.get(defect.code)
        if existing is None or SEVERITY_ORDER[defect.severity] > SEVERITY_ORDER[existing.severity]:
            by_code[defect.code] = defect
    return list(by_code.values())


def pick(metrics: dict[str, float], *keys: str) -> dict[str, float]:
    return {key: metrics[key] for key in keys}
