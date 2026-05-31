from __future__ import annotations

import cv2
import numpy as np

from camera_iqa.grayscale_chart import analyze_grayscale_chart


def make_chart(levels: list[int]) -> np.ndarray:
    image = np.full((260, 420, 3), 95, dtype=np.uint8)
    x1, y1 = 35, 45
    block_w, block_h = 32, 70
    for row, row_levels in enumerate((levels, list(reversed(levels)))):
        y = y1 + row * 110
        for index, level in enumerate(row_levels):
            x = x1 + index * block_w
            image[y : y + block_h, x : x + block_w] = level
    return image


def test_analyze_grayscale_chart_counts_only_distinguishable_adjacent_levels() -> None:
    image = make_chart([18, 30, 45, 70, 96, 128, 160, 190, 218, 250, 252])

    result = analyze_grayscale_chart(image, expected_levels=11, min_delta=5.0)

    assert result.distinguishable_levels == 10
    assert len(result.best_strip.blocks) == 11
    assert result.best_strip.adjacent_deltas[-1] < 5.0


def test_analyze_grayscale_chart_can_save_annotation(tmp_path) -> None:
    image = make_chart([18, 30, 45, 70, 96, 128, 160, 190, 218, 245, 255])
    output = tmp_path / "annotated.jpg"

    result = analyze_grayscale_chart(image, annotation_path=output)

    assert result.distinguishable_levels == 11
    assert output.exists()
    assert cv2.imread(str(output)) is not None
