from camera_iqa.catalog import DEFECT_GROUPS, METRIC_GROUPS, all_metric_keys, metric_group_for


def test_metric_catalog_splits_objective_metrics_and_defect_metrics() -> None:
    assert "sharpness_laplacian_var" in METRIC_GROUPS["客观指标"]
    assert "brightness_mean" in METRIC_GROUPS["客观指标"]
    assert "dark_corner_score" in METRIC_GROUPS["客观缺陷指标"]
    assert "black_border_ratio" in METRIC_GROUPS["客观缺陷指标"]


def test_metric_catalog_preserves_report_column_order() -> None:
    keys = all_metric_keys()

    assert keys.index("stripe_score") < keys.index("dark_corner_score")
    assert keys[-1] == "dead_pixel_ratio"


def test_metric_group_lookup_defaults_to_unknown_for_external_metrics() -> None:
    assert metric_group_for("tenengrad") == "客观指标"
    assert metric_group_for("external_model_score") == "未分类指标"


def test_defect_catalog_contains_objective_defects() -> None:
    assert "color_cast" in DEFECT_GROUPS["客观缺陷"]
    assert "dark_corner" in DEFECT_GROUPS["客观缺陷"]
    assert "dead_pixel" in DEFECT_GROUPS["客观缺陷"]
